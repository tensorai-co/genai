//! Neutral [`Request`] → Interactions API request body.

use genai_core::{Effort, Error, Media, MediaKind, MediaSource, Message, OutputFormat, Part, ProviderId, ReasoningDisplay, Request, Result, Role, ToolChoice, ToolResult, Warning};
use serde_json::{Map, Value, json};

use crate::models::Spec;
use crate::options::GoogleOptions;

/// The body, and what was approximated.
pub(crate) struct Converted {
    pub(crate) body: Value,
    pub(crate) warnings: Vec<Warning>,
}

/// Collects warnings, and fails on approximations under [`Request::strict`].
struct Notes {
    strict: bool,
    warnings: Vec<Warning>,
}

impl Notes {
    fn approximate(&mut self, code: &str, message: String) -> Result<()> {
        if self.strict {
            return Err(Error::Unsupported { provider: ProviderId::Google, feature: message });
        }
        self.warnings.push(Warning::new(code, message));
        Ok(())
    }

    fn note(&mut self, code: &str, message: String) {
        self.warnings.push(Warning::new(code, message));
    }
}

/// Builds the request body. `spec` is the model's catalog entry, when known; without it, settings go through unchanged.
pub(crate) fn request_body(request: &Request, spec: Option<&Spec>, stream: bool) -> Result<Converted> {
    let mut notes = Notes { strict: request.strict, warnings: Vec::new() };
    let options: GoogleOptions = match request.options_for(&ProviderId::Google) {
        Some(value) => serde_json::from_value(value.clone()).map_err(|e| Error::InvalidInput(format!("bad Google options: {e}")))?,
        None => GoogleOptions::default(),
    };

    let mut body = Map::new();
    body.insert("model".into(), Value::String(request.model.name().trim_start_matches("models/").to_owned()));
    body.insert("input".into(), Value::Array(steps(&request.messages, &mut notes)?));
    if let Some(instructions) = request.instructions.as_deref().filter(|s| !s.is_empty()) {
        body.insert("system_instruction".into(), Value::String(instructions.to_owned()));
    }

    let mut config = Map::new();
    if let Some(mut max) = request.max_output_tokens {
        if let Some(spec) = spec
            && max > spec.max_output_tokens
        {
            notes.approximate("max_output_tokens_clamped", format!("{} allows at most {} output tokens per response; asked for {max}", spec.id, spec.max_output_tokens))?;
            max = spec.max_output_tokens;
        }
        config.insert("max_output_tokens".into(), json!(max));
    }
    if let Some(temperature) = request.temperature {
        config.insert("temperature".into(), json!(temperature));
    }
    if !request.stop_sequences.is_empty() {
        config.insert("stop_sequences".into(), json!(request.stop_sequences));
    }
    if let Some(level) = thinking_level(request.effort, spec, &mut notes)? {
        config.insert("thinking_level".into(), Value::String(level.as_str().into()));
    }
    if let Some(display) = request.reasoning_display {
        config.insert("thinking_summaries".into(), Value::String(if display == ReasoningDisplay::Hidden { "none" } else { "auto" }.into()));
    }

    if !request.tools.is_empty() {
        let tools = request
            .tools
            .iter()
            .map(|tool| {
                let mut t = Map::new();
                t.insert("type".into(), Value::String("function".into()));
                t.insert("name".into(), Value::String(tool.name.clone()));
                if let Some(description) = &tool.description {
                    t.insert("description".into(), Value::String(description.clone()));
                }
                t.insert("parameters".into(), clean_schema(&tool.parameters));
                Value::Object(t)
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
        if let Some(choice) = tool_choice(request, &mut notes)? {
            config.insert("tool_choice".into(), choice);
        }
    }
    if !config.is_empty() {
        body.insert("generation_config".into(), Value::Object(config));
    }

    if let Some(OutputFormat::JsonSchema { schema, .. }) = &request.output {
        body.insert("response_format".into(), json!({ "type": "text", "mime_type": "application/json", "schema": clean_schema(schema) }));
    }
    if stream {
        body.insert("stream".into(), Value::Bool(true));
    }
    // Stateless by default: the whole conversation goes with each request, and Google keeps nothing.
    body.insert("store".into(), Value::Bool(options.store.unwrap_or(false)));
    if let Some(tier) = options.service_tier {
        body.insert("service_tier".into(), serde_json::to_value(tier).unwrap_or_default());
    }
    if !options.safety_settings.is_empty() {
        let settings = options.safety_settings.iter().map(|s| json!({ "type": s.category, "threshold": s.threshold })).collect();
        body.insert("safety_settings".into(), Value::Array(settings));
    }
    if !options.labels.is_empty() {
        body.insert("labels".into(), json!(options.labels));
    }

    let mut body = Value::Object(body);
    for extra in request.extra_body.iter().filter(|e| e.provider == ProviderId::Google) {
        merge(&mut body, &extra.value);
    }
    Ok(Converted { body, warnings: notes.warnings })
}

/// The `thinking_level` for the asked-for effort.
fn thinking_level(effort: Option<Effort>, spec: Option<&Spec>, notes: &mut Notes) -> Result<Option<Effort>> {
    let Some(asked) = effort else { return Ok(None) };
    let Some(spec) = spec.filter(|s| !s.levels.is_empty()) else {
        // An unknown model: pass through what Google has a name for.
        return Ok(Some(match asked {
            Effort::None => {
                notes.approximate("reasoning_always_on", "Gemini can't turn reasoning off; used minimal".into())?;
                Effort::Minimal
            }
            Effort::Max => {
                notes.approximate("effort_clamped", "Gemini has no max thinking level; used high".into())?;
                Effort::High
            }
            other => other,
        }));
    };
    if spec.levels.contains(&asked) {
        return Ok(Some(asked));
    }
    let lowest = spec.levels.first().copied().unwrap_or(Effort::Low);
    let used = spec.levels.iter().copied().filter(|l| *l <= asked).max().unwrap_or(lowest);
    if asked == Effort::None {
        notes.approximate("reasoning_always_on", format!("{} can't turn reasoning off; used {}", spec.id, used.as_str()))?;
    } else {
        notes.approximate("effort_clamped", format!("{} doesn't support {} thinking; used {}", spec.id, asked.as_str(), used.as_str()))?;
    }
    Ok(Some(used))
}

fn tool_choice(request: &Request, notes: &mut Notes) -> Result<Option<Value>> {
    // `validated` keeps calls valid against their schemas, Google's form of strict tools.
    let strict = request.tools.iter().any(|t| t.strict);
    let choice = match &request.tool_choice {
        None if strict => Some(json!("validated")),
        None => None,
        Some(ToolChoice::Auto) => Some(json!(if strict { "validated" } else { "auto" })),
        Some(ToolChoice::None) => Some(json!("none")),
        Some(ToolChoice::Required) => Some(json!("any")),
        Some(ToolChoice::Named(name)) => Some(json!({ "allowed_tools": { "mode": "any", "tools": [name] } })),
        Some(other) => return Err(Error::Unsupported { provider: ProviderId::Google, feature: format!("tool choice {other:?}") }),
    };
    if request.parallel_tool_calls == Some(false) {
        notes.approximate("parallel_tool_calls_ignored", "Gemini can't be limited to one tool call at a time".into())?;
    }
    Ok(choice)
}

/// The conversation as Interactions API steps: `user_input`, `function_result`, and the model's own steps sent back.
fn steps(messages: &[Message], notes: &mut Notes) -> Result<Vec<Value>> {
    let mut steps = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        match message.role {
            Role::User => {
                let mut results = Vec::new();
                let mut content = Vec::new();
                for part in &message.parts {
                    match part {
                        Part::Text(t) if t.text.is_empty() => {}
                        Part::Text(t) => content.push(json!({ "type": "text", "text": t.text })),
                        Part::Media(m) => content.push(media(m)?),
                        Part::ToolResult(r) => results.push(tool_result(r)?),
                        Part::ToolCall(_) => return Err(Error::InvalidInput(format!("message {i}: tool calls belong in assistant messages"))),
                        Part::Opaque(o) if o.provider == ProviderId::Google => content.push(o.data.clone()),
                        other => notes.note("part_dropped", format!("message {i}: left out a {} part Gemini can't take from the user", part_name(other))),
                    }
                }
                steps.extend(results);
                if !content.is_empty() {
                    steps.push(json!({ "type": "user_input", "content": content }));
                }
            }
            Role::Assistant => {
                let mut output: Vec<Value> = Vec::new();
                let flush = |output: &mut Vec<Value>, steps: &mut Vec<Value>| {
                    if !output.is_empty() {
                        steps.push(json!({ "type": "model_output", "content": std::mem::take(output) }));
                    }
                };
                for part in &message.parts {
                    match part {
                        Part::Text(t) => match &t.opaque {
                            Some(o) if o.provider == ProviderId::Google => {
                                flush(&mut output, &mut steps);
                                steps.push(o.data.clone());
                            }
                            _ if t.text.is_empty() => {}
                            _ => output.push(json!({ "type": "text", "text": t.text })),
                        },
                        Part::Refusal { text } if !text.is_empty() => output.push(json!({ "type": "text", "text": text })),
                        Part::Refusal { .. } => {}
                        Part::Media(m) => output.push(media(m)?),
                        Part::Reasoning(r) => match &r.opaque {
                            Some(o) if o.provider == ProviderId::Google => {
                                flush(&mut output, &mut steps);
                                steps.push(o.data.clone());
                            }
                            Some(o) => notes.note("reasoning_dropped", format!("message {i}: left out reasoning from {}, which Gemini can't read", o.provider)),
                            None => notes.note("reasoning_dropped", format!("message {i}: left out reasoning with no Gemini signature")),
                        },
                        Part::ToolCall(call) => {
                            flush(&mut output, &mut steps);
                            match &call.opaque {
                                Some(o) if o.provider == ProviderId::Google => steps.push(o.data.clone()),
                                _ => steps.push(
                                    json!({ "type": "function_call", "id": call.id, "name": call.name, "arguments": if call.arguments.is_object() { call.arguments.clone() } else { json!({}) } }),
                                ),
                            }
                        }
                        Part::Opaque(o) if o.provider == ProviderId::Google => {
                            flush(&mut output, &mut steps);
                            steps.push(o.data.clone());
                        }
                        Part::Opaque(o) => notes.note("opaque_dropped", format!("message {i}: left out {} data that Gemini can't read", o.provider)),
                        Part::ToolResult(_) => return Err(Error::InvalidInput(format!("message {i}: tool results belong in user messages"))),
                        other => notes.note("part_dropped", format!("message {i}: left out a {} part", part_name(other))),
                    }
                }
                flush(&mut output, &mut steps);
            }
            other => return Err(Error::Unsupported { provider: ProviderId::Google, feature: format!("{other:?} messages") }),
        }
    }
    Ok(steps)
}

fn part_name(part: &Part) -> &'static str {
    match part {
        Part::Text(_) => "text",
        Part::Media(_) => "media",
        Part::ToolCall(_) => "tool call",
        Part::ToolResult(_) => "tool result",
        Part::Reasoning(_) => "reasoning",
        Part::Refusal { .. } => "refusal",
        Part::Opaque(_) => "provider data",
        _ => "unknown",
    }
}

fn tool_result(result: &ToolResult) -> Result<Value> {
    let mut content = Vec::new();
    for part in &result.content {
        match part {
            Part::Text(t) if t.text.is_empty() => {}
            Part::Text(t) => content.push(json!({ "type": "text", "text": t.text })),
            Part::Media(m) => content.push(media(m)?),
            other => return Err(Error::Unsupported { provider: ProviderId::Google, feature: format!("{} inside a tool result", part_name(other)) }),
        }
    }
    let mut step = json!({ "type": "function_result", "call_id": result.call_id, "name": result.name, "result": if content.is_empty() { json!("") } else { Value::Array(content) } });
    if result.is_error {
        step["is_error"] = Value::Bool(true);
    }
    Ok(step)
}

fn media(media: &Media) -> Result<Value> {
    let kind = match media.kind {
        MediaKind::Image => "image",
        MediaKind::Document => "document",
        MediaKind::Audio => "audio",
        MediaKind::Video => "video",
        other => return Err(Error::Unsupported { provider: ProviderId::Google, feature: format!("{other:?} input") }),
    };
    let mut item = match &media.source {
        MediaSource::Base64 { data } => {
            let mime = media.mime_type.as_deref().ok_or_else(|| Error::InvalidInput(format!("inline {kind} needs a media type")))?;
            json!({ "type": kind, "data": data, "mime_type": mime })
        }
        MediaSource::Url { url } => json!({ "type": kind, "uri": url }),
        MediaSource::File { id } => json!({ "type": kind, "uri": id }),
        // Gemini takes no plain-text documents; text is text.
        MediaSource::Text { text } => {
            let text = match &media.title {
                Some(title) => format!("{title}\n\n{text}"),
                None => text.clone(),
            };
            return Ok(json!({ "type": "text", "text": text }));
        }
        other => return Err(Error::Unsupported { provider: ProviderId::Google, feature: format!("media from {other:?}") }),
    };
    if !matches!(media.source, MediaSource::Base64 { .. })
        && let Some(mime) = &media.mime_type
    {
        item["mime_type"] = Value::String(mime.clone());
    }
    Ok(item)
}

/// Drops the `$schema` keyword, which Gemini's schema subset doesn't take.
fn clean_schema(schema: &Value) -> Value {
    let mut schema = schema.clone();
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
    }
    schema
}

/// Merges `extra` into `body`, objects key by key, everything else replaced.
fn merge(body: &mut Value, extra: &Value) {
    match (body, extra) {
        (Value::Object(body), Value::Object(extra)) => {
            for (key, value) in extra {
                match body.get_mut(key) {
                    Some(existing) if existing.is_object() && value.is_object() => merge(existing, value),
                    _ => {
                        body.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (body, extra) => *body = extra.clone(),
    }
}

#[cfg(test)]
mod tests {
    use genai_core::{Opaque, Reasoning, Tool, ToolCall};

    use super::*;
    use crate::models::{catalog, find};
    use crate::options::ServiceTier;

    fn body(request: &Request) -> Result<Converted> {
        let specs = catalog();
        request_body(request, find(&specs, request.model.name()), true)
    }

    #[test]
    fn builds_a_plain_stateless_request() {
        let request = Request::new("google/gemini-3.5-flash-lite")
            .instructions("Be brief.")
            .message(Message::user("Hi"))
            .max_output_tokens(256)
            .effort(Effort::Low)
            .reasoning_display(ReasoningDisplay::Summary)
            .temperature(0.5);
        assert_eq!(
            body(&request).ok().map(|c| c.body),
            Some(json!({
                "model": "gemini-3.5-flash-lite",
                "input": [{ "type": "user_input", "content": [{ "type": "text", "text": "Hi" }] }],
                "system_instruction": "Be brief.",
                "generation_config": { "max_output_tokens": 256, "temperature": 0.5, "thinking_level": "low", "thinking_summaries": "auto" },
                "stream": true,
                "store": false
            }))
        );
    }

    fn level(model: &str, effort: Effort) -> (Value, Vec<String>) {
        let request = Request::new(format!("google/{model}")).message(Message::user("Hi")).effort(effort);
        match body(&request) {
            Ok(c) => (c.body["generation_config"]["thinking_level"].clone(), c.warnings.into_iter().map(|w| w.code).collect()),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn maps_effort_to_thinking_levels() {
        assert_eq!(level("gemini-4-hydrogen", Effort::XHigh), (json!("xhigh"), vec![]));
        assert_eq!(level("gemini-4-carbon", Effort::XHigh), (json!("high"), vec!["effort_clamped".into()]));
        assert_eq!(level("gemini-4-hydrogen", Effort::Max), (json!("xhigh"), vec!["effort_clamped".into()]));
        assert_eq!(level("gemini-4-hydrogen", Effort::Minimal), (json!("low"), vec!["effort_clamped".into()]));
        assert_eq!(level("gemini-3.5-flash-lite", Effort::Minimal), (json!("minimal"), vec![]));
        assert_eq!(level("gemini-3.5-flash-lite", Effort::None), (json!("minimal"), vec!["reasoning_always_on".into()]));
        assert_eq!(level("gemini-9-unknown", Effort::Medium), (json!("medium"), vec![]));
        assert_eq!(level("gemini-9-unknown", Effort::None), (json!("minimal"), vec!["reasoning_always_on".into()]));
    }

    #[test]
    fn strict_fails_instead_of_clamping() {
        let request = Request::new("google/gemini-4-carbon").message(Message::user("Hi")).effort(Effort::Max).strict(true);
        assert!(matches!(body(&request), Err(Error::Unsupported { .. })));
    }

    #[test]
    fn maps_tools_and_choices() {
        let tool = Tool::function("get_time").description("Now").parameters(json!({ "$schema": "x", "type": "object", "properties": {} }));
        let request = Request::new("google/gemini-4-hydrogen").message(Message::user("time?")).tool(tool.clone()).tool_choice(ToolChoice::Named("get_time".into()));
        let c = body(&request).ok().map(|c| c.body);
        assert_eq!(c.as_ref().map(|b| b["tools"].clone()), Some(json!([{ "type": "function", "name": "get_time", "description": "Now", "parameters": { "type": "object", "properties": {} } }])));
        assert_eq!(c.map(|b| b["generation_config"]["tool_choice"].clone()), Some(json!({ "allowed_tools": { "mode": "any", "tools": ["get_time"] } })));

        let strict = Request::new("google/gemini-4-hydrogen").message(Message::user("time?")).tool(tool.strict(true));
        assert_eq!(body(&strict).ok().map(|c| c.body["generation_config"]["tool_choice"].clone()), Some(json!("validated")));
    }

    #[test]
    fn sends_back_model_steps_and_tool_results() {
        let call = ToolCall { id: "c1".into(), name: "add".into(), arguments: json!({ "a": 1 }), opaque: None };
        let thought = json!({ "type": "thought", "signature": "sig" });
        let request = Request::new("google/gemini-4-hydrogen").messages([
            Message::user("1+1?"),
            Message::new(
                Role::Assistant,
                vec![
                    Part::Reasoning(Reasoning { text: None, opaque: Some(Opaque::new(ProviderId::Google, thought.clone())) }),
                    Part::text("Let me add."),
                    Part::ToolCall(call.clone()),
                    Part::Reasoning(Reasoning { text: Some("x".into()), opaque: Some(Opaque::new(ProviderId::Anthropic, json!({}))) }),
                ],
            ),
            Message::new(Role::User, vec![Part::text("go on"), Part::ToolResult(ToolResult::new(&call, "2"))]),
        ]);
        let c = body(&request).ok();
        assert_eq!(
            c.as_ref().map(|c| c.body["input"].clone()),
            Some(json!([
                { "type": "user_input", "content": [{ "type": "text", "text": "1+1?" }] },
                thought,
                { "type": "model_output", "content": [{ "type": "text", "text": "Let me add." }] },
                { "type": "function_call", "id": "c1", "name": "add", "arguments": { "a": 1 } },
                { "type": "function_result", "call_id": "c1", "name": "add", "result": [{ "type": "text", "text": "2" }] },
                { "type": "user_input", "content": [{ "type": "text", "text": "go on" }] }
            ]))
        );
        assert_eq!(c.map(|c| c.warnings.into_iter().map(|w| w.code).collect::<Vec<_>>()), Some(vec!["reasoning_dropped".to_owned()]));
    }

    #[test]
    fn maps_structured_output_media_and_options() {
        let request = Request::new("google/gemini-4-hydrogen")
            .message(Message::user("see").with(Media::image("image/png", [1u8, 2])).with(Media::text_document("notes").title("Notes")))
            .output(OutputFormat::json_schema("x", json!({ "type": "object" })))
            .options(GoogleOptions::new().service_tier(ServiceTier::Flex).store(true));
        let c = body(&request).ok().map(|c| c.body);
        assert_eq!(c.as_ref().map(|b| b["response_format"].clone()), Some(json!({ "type": "text", "mime_type": "application/json", "schema": { "type": "object" } })));
        assert_eq!(c.as_ref().map(|b| b["input"][0]["content"][1].clone()), Some(json!({ "type": "image", "data": "AQI=", "mime_type": "image/png" })));
        assert_eq!(c.as_ref().map(|b| b["input"][0]["content"][2].clone()), Some(json!({ "type": "text", "text": "Notes\n\nnotes" })));
        assert_eq!(c.map(|b| (b["service_tier"].clone(), b["store"].clone())), Some((json!("flex"), json!(true))));
    }
}
