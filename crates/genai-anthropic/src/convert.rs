//! Neutral [`Request`] → Messages API request body.

use genai_core::{Cache, Effort, Error, Media, MediaKind, MediaSource, Message, OutputFormat, Part, ProviderId, ReasoningDisplay, Request, Result, Role, ToolChoice, ToolResult, Warning};
use serde_json::{Map, Value, json};

use crate::models::Spec;
use crate::options::AnthropicOptions;

/// Used when the request sets no output limit (the API requires one). Thinking counts toward it.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 16_384;

/// The body, the beta headers it needs, and what was approximated.
pub(crate) struct Converted {
    pub(crate) body: Value,
    pub(crate) betas: Vec<String>,
    pub(crate) warnings: Vec<Warning>,
}

/// Collects warnings, and fails on approximations under [`Request::strict`].
struct Notes {
    strict: bool,
    warnings: Vec<Warning>,
}

impl Notes {
    /// A request setting that couldn't be honored exactly. An error under `strict`.
    fn approximate(&mut self, code: &str, message: String) -> Result<()> {
        if self.strict {
            return Err(Error::Unsupported { provider: ProviderId::Anthropic, feature: message });
        }
        self.warnings.push(Warning::new(code, message));
        Ok(())
    }

    /// Something left out that isn't a request setting, such as another provider's reasoning. Never an error.
    fn note(&mut self, code: &str, message: String) {
        self.warnings.push(Warning::new(code, message));
    }
}

/// Builds the request body. `spec` is the model's catalog entry, when known; without it, settings go through unchanged.
pub(crate) fn request_body(request: &Request, spec: Option<&Spec>, stream: bool) -> Result<Converted> {
    let mut notes = Notes { strict: request.strict, warnings: Vec::new() };
    let options: AnthropicOptions = match request.options_for(&ProviderId::Anthropic) {
        Some(value) => serde_json::from_value(value.clone()).map_err(|e| Error::InvalidInput(format!("bad Anthropic options: {e}")))?,
        None => AnthropicOptions::default(),
    };

    let mut body = Map::new();
    body.insert("model".into(), Value::String(request.model.name().to_owned()));

    let mut max_tokens = request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    if let Some(spec) = spec
        && max_tokens > spec.max_output_tokens
    {
        if request.max_output_tokens.is_some() {
            notes.approximate("max_output_tokens_clamped", format!("{} allows at most {} output tokens; asked for {max_tokens}", spec.id, spec.max_output_tokens))?;
        }
        max_tokens = spec.max_output_tokens;
    }
    body.insert("max_tokens".into(), json!(max_tokens));

    if let Some(instructions) = request.instructions.as_deref().filter(|s| !s.is_empty()) {
        body.insert("system".into(), Value::String(instructions.to_owned()));
    }
    body.insert("messages".into(), Value::Array(messages(&request.messages, &mut notes)?));

    let (thinking, effort) = reasoning(request.effort, request.reasoning_display, spec, max_tokens, &mut notes)?;
    if let Some(thinking) = thinking {
        body.insert("thinking".into(), thinking);
    }
    let mut output_config = Map::new();
    if let Some(effort) = effort {
        output_config.insert("effort".into(), Value::String(effort.as_str().into()));
    }
    if let Some(OutputFormat::JsonSchema { schema, .. }) = &request.output {
        if spec.is_some_and(|s| !s.structured_output) {
            return Err(Error::Unsupported { provider: ProviderId::Anthropic, feature: format!("structured output on {}", request.model.name()) });
        }
        output_config.insert("format".into(), json!({ "type": "json_schema", "schema": strict_schema(schema) }));
    }
    if !output_config.is_empty() {
        body.insert("output_config".into(), Value::Object(output_config));
    }

    if !request.tools.is_empty() {
        let tools = request
            .tools
            .iter()
            .map(|tool| {
                let mut t = Map::new();
                t.insert("name".into(), Value::String(tool.name.clone()));
                if let Some(description) = &tool.description {
                    t.insert("description".into(), Value::String(description.clone()));
                }
                t.insert("input_schema".into(), if tool.strict { strict_schema(&tool.parameters) } else { tool.parameters.clone() });
                if tool.strict {
                    t.insert("strict".into(), Value::Bool(true));
                }
                Value::Object(t)
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
        if let Some(choice) = tool_choice(request, spec, &mut notes)? {
            body.insert("tool_choice".into(), choice);
        }
    }

    if request.temperature.is_some() {
        notes.approximate("temperature_dropped", "Anthropic's API (version 2026-10-21) no longer accepts temperature; it was left out".into())?;
    }
    if !request.stop_sequences.is_empty() {
        body.insert("stop_sequences".into(), json!(request.stop_sequences));
    }
    if request.cache == Cache::Auto {
        let mut cache = json!({ "type": "ephemeral" });
        if let Some(ttl) = options.cache_ttl {
            cache["ttl"] = Value::String(ttl.as_str().into());
        }
        body.insert("cache_control".into(), cache);
    }
    if let Some(user_id) = &request.user_id {
        body.insert("metadata".into(), json!({ "user_id": user_id }));
    }
    if let Some(speed) = options.speed {
        body.insert("speed".into(), serde_json::to_value(speed).unwrap_or_default());
    }
    if let Some(tier) = options.service_tier {
        body.insert("service_tier".into(), serde_json::to_value(tier).unwrap_or_default());
    }
    if let Some(geo) = &options.inference_geo {
        body.insert("inference_geo".into(), Value::String(geo.clone()));
    }
    if stream {
        body.insert("stream".into(), Value::Bool(true));
    }

    let mut body = Value::Object(body);
    for extra in request.extra_body.iter().filter(|e| e.provider == ProviderId::Anthropic) {
        merge(&mut body, &extra.value);
    }
    Ok(Converted { body, betas: options.betas, warnings: notes.warnings })
}

/// The `thinking` and `output_config.effort` values for the asked-for effort and display.
fn reasoning(effort: Option<Effort>, display: Option<ReasoningDisplay>, spec: Option<&Spec>, max_tokens: u32, notes: &mut Notes) -> Result<(Option<Value>, Option<Effort>)> {
    let display = display.map(|d| if d == ReasoningDisplay::Hidden { "omitted" } else { "summarized" });
    let adaptive = |display: Option<&str>| display.map(|d| json!({ "type": "adaptive", "display": d }));

    let Some(spec) = spec else {
        // An unknown model: pass the settings through and let the API judge them.
        return Ok(match effort {
            Some(Effort::None) => (Some(json!({ "type": "disabled" })), None),
            Some(Effort::Minimal) => {
                notes.approximate("effort_clamped", "Anthropic has no minimal effort; used low".into())?;
                (adaptive(display), Some(Effort::Low))
            }
            other => (adaptive(display), other),
        });
    };

    if spec.efforts.is_empty() {
        // Older models: no effort setting, only a thinking budget.
        return match effort {
            None => Ok((None, None)),
            Some(Effort::None) => Ok((spec.thinking.disabled.then(|| json!({ "type": "disabled" })), None)),
            Some(effort) if spec.thinking.enabled => {
                let budget = thinking_budget(effort).min(max_tokens.saturating_sub(1));
                if budget < 1024 {
                    notes.approximate("reasoning_dropped", format!("{} needs max_output_tokens above 1024 to think; ran without thinking", spec.id))?;
                    return Ok((None, None));
                }
                notes.approximate("effort_as_budget", format!("{} has no effort setting; {} effort became a {budget}-token thinking budget", spec.id, effort.as_str()))?;
                let mut thinking = json!({ "type": "enabled", "budget_tokens": budget });
                if let Some(d) = display {
                    thinking["display"] = Value::String(d.into());
                }
                Ok((Some(thinking), None))
            }
            Some(effort) => {
                notes.approximate("effort_dropped", format!("{} has no effort setting; {} was left out", spec.id, effort.as_str()))?;
                Ok((None, None))
            }
        };
    }

    let lowest = spec.efforts.first().copied().unwrap_or(Effort::Low);
    let thinking_on = |display: Option<&str>| -> Option<Value> {
        let d = display?;
        if spec.thinking.adaptive { Some(json!({ "type": "adaptive", "display": d })) } else { None }
    };
    match effort {
        None => Ok((thinking_on(display), None)),
        Some(Effort::None) if spec.thinking.disabled => Ok((Some(json!({ "type": "disabled" })), None)),
        Some(Effort::None) if spec.thinking.between_tools => Ok((Some(json!({ "type": "between_tools" })), None)),
        Some(Effort::None) => {
            notes.approximate("reasoning_always_on", format!("{} can't turn reasoning off; used {} effort", spec.id, lowest.as_str()))?;
            Ok((thinking_on(display), Some(lowest)))
        }
        Some(asked) => {
            let used = if spec.efforts.contains(&asked) { asked } else { spec.efforts.iter().copied().filter(|e| *e <= asked).max().unwrap_or(lowest) };
            if used != asked {
                notes.approximate("effort_clamped", format!("{} doesn't support {} effort; used {}", spec.id, asked.as_str(), used.as_str()))?;
            }
            Ok((thinking_on(display), Some(used)))
        }
    }
}

/// A thinking budget for models that take one instead of an effort level.
fn thinking_budget(effort: Effort) -> u32 {
    match effort {
        Effort::None | Effort::Minimal | Effort::Low => 1024,
        Effort::Medium => 4096,
        Effort::High => 16_384,
        Effort::XHigh => 32_768,
        _ => 65_536,
    }
}

fn tool_choice(request: &Request, spec: Option<&Spec>, notes: &mut Notes) -> Result<Option<Value>> {
    let mut choice = match &request.tool_choice {
        None => None,
        Some(ToolChoice::Auto) => Some(json!({ "type": "auto" })),
        Some(ToolChoice::None) => return Ok(Some(json!({ "type": "none" }))),
        Some(forced @ (ToolChoice::Required | ToolChoice::Named(_))) => {
            if spec.is_some_and(|s| !s.forced_tool_choice) {
                notes.approximate("tool_choice_relaxed", format!("{} can't be forced to call a tool; used auto", request.model.name()))?;
                Some(json!({ "type": "auto" }))
            } else if let ToolChoice::Named(name) = forced {
                Some(json!({ "type": "tool", "name": name }))
            } else {
                Some(json!({ "type": "any" }))
            }
        }
        Some(other) => return Err(Error::Unsupported { provider: ProviderId::Anthropic, feature: format!("tool choice {other:?}") }),
    };
    if request.parallel_tool_calls == Some(false) {
        choice.get_or_insert_with(|| json!({ "type": "auto" }))["disable_parallel_tool_use"] = Value::Bool(true);
    }
    Ok(choice)
}

fn messages(messages: &[Message], notes: &mut Notes) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(messages.len());
    for (i, message) in messages.iter().enumerate() {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            other => return Err(Error::Unsupported { provider: ProviderId::Anthropic, feature: format!("{other:?} messages") }),
        };
        // Tool results have to come first in a user message.
        let mut results = Vec::new();
        let mut blocks = Vec::new();
        for part in &message.parts {
            match part {
                Part::Text(text) => match &text.opaque {
                    Some(o) if o.provider == ProviderId::Anthropic && message.role == Role::Assistant => blocks.push(o.data.clone()),
                    _ if text.text.is_empty() => {}
                    _ => blocks.push(json!({ "type": "text", "text": text.text })),
                },
                Part::Media(media) => blocks.push(media_block(media)?),
                Part::ToolCall(call) => {
                    if message.role != Role::Assistant {
                        return Err(Error::InvalidInput(format!("message {i}: tool calls belong in assistant messages")));
                    }
                    match &call.opaque {
                        Some(o) if o.provider == ProviderId::Anthropic => blocks.push(o.data.clone()),
                        _ => blocks.push(json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": if call.arguments.is_object() { call.arguments.clone() } else { json!({}) } })),
                    }
                }
                Part::ToolResult(result) => {
                    if message.role != Role::User {
                        return Err(Error::InvalidInput(format!("message {i}: tool results belong in user messages")));
                    }
                    results.push(tool_result(result)?);
                }
                Part::Reasoning(reasoning) => match &reasoning.opaque {
                    Some(o) if o.provider == ProviderId::Anthropic => blocks.push(o.data.clone()),
                    Some(o) => notes.note("reasoning_dropped", format!("message {i}: left out reasoning from {}, which Anthropic can't read", o.provider)),
                    None => notes.note("reasoning_dropped", format!("message {i}: left out reasoning with no Anthropic signature")),
                },
                Part::Refusal { text } if !text.is_empty() => blocks.push(json!({ "type": "text", "text": text })),
                Part::Refusal { .. } => {}
                Part::Opaque(o) if o.provider == ProviderId::Anthropic => blocks.push(o.data.clone()),
                Part::Opaque(o) => notes.note("opaque_dropped", format!("message {i}: left out {} data that Anthropic can't read", o.provider)),
                other => return Err(Error::Unsupported { provider: ProviderId::Anthropic, feature: format!("message part {other:?}") }),
            }
        }
        results.extend(blocks);
        if results.is_empty() {
            notes.note("empty_message_dropped", format!("message {i} had nothing Anthropic can read and was left out"));
            continue;
        }
        out.push(json!({ "role": role, "content": results }));
    }
    Ok(out)
}

fn tool_result(result: &ToolResult) -> Result<Value> {
    let mut content = Vec::new();
    for part in &result.content {
        match part {
            Part::Text(t) if t.text.is_empty() => {}
            Part::Text(t) => content.push(json!({ "type": "text", "text": t.text })),
            Part::Media(m) => content.push(media_block(m)?),
            other => return Err(Error::Unsupported { provider: ProviderId::Anthropic, feature: format!("{other:?} inside a tool result") }),
        }
    }
    let mut block = json!({ "type": "tool_result", "tool_use_id": result.call_id });
    if !content.is_empty() {
        block["content"] = Value::Array(content);
    }
    if result.is_error {
        block["is_error"] = Value::Bool(true);
    }
    Ok(block)
}

const IMAGE_TYPES: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];

fn media_block(media: &Media) -> Result<Value> {
    let mime = media.mime_type.as_deref();
    let mut block = match (media.kind, &media.source) {
        (MediaKind::Image, MediaSource::Base64 { data }) => {
            let mime =
                mime.filter(|m| IMAGE_TYPES.contains(m)).ok_or_else(|| Error::InvalidInput(format!("Anthropic takes JPEG, PNG, GIF and WebP images; got {}", mime.unwrap_or("no media type"))))?;
            json!({ "type": "image", "source": { "type": "base64", "media_type": mime, "data": data } })
        }
        (MediaKind::Image, MediaSource::Url { url }) => json!({ "type": "image", "source": { "type": "url", "url": url } }),
        (MediaKind::Image, MediaSource::File { id }) => json!({ "type": "image", "source": { "type": "file", "file_id": id } }),
        (MediaKind::Document, MediaSource::Base64 { data }) if mime == Some("application/pdf") => {
            json!({ "type": "document", "source": { "type": "base64", "media_type": "application/pdf", "data": data } })
        }
        (MediaKind::Document, MediaSource::Base64 { .. }) => {
            return Err(Error::InvalidInput(format!("Anthropic takes inline documents as PDF or plain text; got {}", mime.unwrap_or("no media type"))));
        }
        (MediaKind::Document, MediaSource::Url { url }) => json!({ "type": "document", "source": { "type": "url", "url": url } }),
        (MediaKind::Document, MediaSource::File { id }) => json!({ "type": "document", "source": { "type": "file", "file_id": id } }),
        (MediaKind::Document, MediaSource::Text { text }) => json!({ "type": "document", "source": { "type": "text", "media_type": "text/plain", "data": text } }),
        (kind, _) => return Err(Error::Unsupported { provider: ProviderId::Anthropic, feature: format!("{kind:?} input from this source") }),
    };
    if let Some(title) = &media.title
        && block["type"] == "document"
    {
        block["title"] = Value::String(title.clone());
    }
    Ok(block)
}

/// Adjusts a JSON Schema to what Anthropic's strict mode accepts, the way its own SDKs do: every object gets
/// `additionalProperties: false`, `oneOf` becomes `anyOf`, and constraints it doesn't enforce move into the description.
pub(crate) fn strict_schema(schema: &Value) -> Value {
    let mut schema = schema.clone();
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
    }
    tighten(&mut schema);
    schema
}

const UNENFORCED: &[&str] = &["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "maxItems", "uniqueItems", "minProperties", "maxProperties"];

fn tighten(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            if let Some(one_of) = map.remove("oneOf") {
                map.insert("anyOf".into(), one_of);
            }
            let mut moved = Vec::new();
            for key in UNENFORCED {
                if let Some(value) = map.remove(*key) {
                    moved.push(format!("{key}: {value}"));
                }
            }
            if map.get("minItems").and_then(Value::as_u64).is_some_and(|n| n > 1)
                && let Some(value) = map.remove("minItems")
            {
                moved.push(format!("minItems: {value}"));
            }
            if !moved.is_empty() {
                let note = format!("{{{}}}", moved.join(", "));
                let description = match map.get("description").and_then(Value::as_str) {
                    Some(d) => format!("{d} {note}"),
                    None => note,
                };
                map.insert("description".into(), Value::String(description));
            }
            let is_object = map.get("type").is_some_and(|t| t == "object" || t.as_array().is_some_and(|a| a.iter().any(|t| t == "object"))) || map.contains_key("properties");
            if is_object && !map.get("additionalProperties").is_some_and(Value::is_object) {
                map.insert("additionalProperties".into(), Value::Bool(false));
            }
            for key in ["properties", "$defs", "definitions"] {
                if let Some(Value::Object(children)) = map.get_mut(key) {
                    children.values_mut().for_each(tighten);
                }
            }
            for key in ["items", "anyOf", "allOf", "prefixItems", "additionalProperties"] {
                if let Some(child) = map.get_mut(key) {
                    tighten(child);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(tighten),
        _ => {}
    }
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

/// The token-counting endpoint takes a subset of the message fields.
pub(crate) fn count_tokens_body(body: Value) -> Value {
    const KEPT: &[&str] = &["model", "messages", "system", "thinking", "tools", "tool_choice", "output_config", "cache_control", "speed"];
    match body {
        Value::Object(map) => Value::Object(map.into_iter().filter(|(k, _)| KEPT.contains(&k.as_str())).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use genai_core::{Opaque, Reasoning, Tool, ToolCall};

    use super::*;
    use crate::models::{catalog, find};

    fn body(request: &Request) -> Result<Converted> {
        let specs = catalog();
        request_body(request, find(&specs, request.model.name()), true)
    }

    #[test]
    fn builds_a_plain_request() {
        let request = Request::new("anthropic/claude-haiku-5-5").instructions("Be brief.").message(Message::user("Hi")).max_output_tokens(256).stop_sequence("END");
        let converted = body(&request).ok();
        assert_eq!(
            converted.map(|c| c.body),
            Some(json!({
                "model": "claude-haiku-5-5", "max_tokens": 256, "system": "Be brief.",
                "messages": [{ "role": "user", "content": [{ "type": "text", "text": "Hi" }] }],
                "stop_sequences": ["END"], "cache_control": { "type": "ephemeral" }, "stream": true
            }))
        );
    }

    fn effort_on(model: &str, effort: Effort) -> (Value, Value, Vec<String>) {
        let request = Request::new(format!("anthropic/{model}")).message(Message::user("Hi")).effort(effort).max_output_tokens(2048);
        match body(&request) {
            Ok(c) => (c.body["thinking"].clone(), c.body["output_config"]["effort"].clone(), c.warnings.into_iter().map(|w| w.code).collect()),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn maps_effort_per_model() {
        assert_eq!(effort_on("claude-opus-5-5", Effort::High), (Value::Null, json!("high"), vec![]));
        assert_eq!(effort_on("claude-haiku-5-5", Effort::None), (json!({ "type": "disabled" }), Value::Null, vec![]));
        assert_eq!(effort_on("claude-sonnet-5-5", Effort::None), (json!({ "type": "between_tools" }), Value::Null, vec![]));
        assert_eq!(effort_on("claude-opus-5-5", Effort::None), (Value::Null, json!("low"), vec!["reasoning_always_on".into()]));
        assert_eq!(effort_on("claude-opus-5-5", Effort::Minimal), (Value::Null, json!("low"), vec!["effort_clamped".into()]));
        assert_eq!(effort_on("claude-sonnet-4-6", Effort::XHigh), (Value::Null, json!("high"), vec!["effort_clamped".into()]));
        assert_eq!(effort_on("claude-haiku-4-5", Effort::Medium), (json!({ "type": "enabled", "budget_tokens": 2047 }), Value::Null, vec!["effort_as_budget".into()]));
        assert_eq!(effort_on("claude-future-9", Effort::XHigh), (Value::Null, json!("xhigh"), vec![]));
    }

    #[test]
    fn strict_turns_approximations_into_errors() {
        let request = Request::new("anthropic/claude-opus-5-5").message(Message::user("Hi")).temperature(0.2).strict(true);
        assert!(matches!(body(&request), Err(Error::Unsupported { .. })));
        let relaxed = body(&request.strict(false)).map(|c| c.warnings.into_iter().map(|w| w.code).collect::<Vec<_>>()).ok();
        assert_eq!(relaxed, Some(vec!["temperature_dropped".to_owned()]));
    }

    #[test]
    fn shows_reasoning_when_asked() {
        let request = Request::new("anthropic/claude-opus-5-5").message(Message::user("Hi")).reasoning_display(ReasoningDisplay::Summary);
        assert_eq!(body(&request).map(|c| c.body["thinking"].clone()).ok(), Some(json!({ "type": "adaptive", "display": "summarized" })));
    }

    #[test]
    fn relaxes_forced_tool_choice_where_unsupported() {
        let tool = Tool::function("get_time");
        let forced = Request::new("anthropic/claude-opus-5-5").message(Message::user("time?")).tool(tool.clone()).tool_choice(ToolChoice::Required).parallel_tool_calls(false);
        let c = body(&forced).ok();
        assert_eq!(c.as_ref().map(|c| c.body["tool_choice"].clone()), Some(json!({ "type": "auto", "disable_parallel_tool_use": true })));
        assert_eq!(c.map(|c| c.warnings[0].code.clone()), Some("tool_choice_relaxed".to_owned()));

        let named = Request::new("anthropic/claude-haiku-5-5").message(Message::user("time?")).tool(tool).tool_choice(ToolChoice::Named("get_time".into()));
        assert_eq!(body(&named).map(|c| c.body["tool_choice"].clone()).ok(), Some(json!({ "type": "tool", "name": "get_time" })));
    }

    #[test]
    fn sends_back_tool_turns_and_reasoning() {
        let call = ToolCall { id: "toolu_1".into(), name: "add".into(), arguments: json!({ "a": 1, "b": 2 }), opaque: None };
        let thinking = json!({ "type": "thinking", "thinking": "add them", "signature": "sig" });
        let request = Request::new("anthropic/claude-haiku-5-5").messages([
            Message::user("1+2?"),
            Message::new(
                Role::Assistant,
                vec![
                    Part::Reasoning(Reasoning { text: Some("add them".into()), opaque: Some(Opaque::new(ProviderId::Anthropic, thinking.clone())) }),
                    Part::Reasoning(Reasoning { text: Some("other".into()), opaque: Some(Opaque::new(ProviderId::OpenAi, json!({}))) }),
                    Part::ToolCall(call.clone()),
                ],
            ),
            Message::new(Role::User, vec![Part::text("here you go"), Part::ToolResult(ToolResult::new(&call, "3"))]),
        ]);
        let c = body(&request).ok();
        let messages = c.as_ref().map(|c| c.body["messages"].clone());
        assert_eq!(
            messages,
            Some(json!([
                { "role": "user", "content": [{ "type": "text", "text": "1+2?" }] },
                { "role": "assistant", "content": [thinking, { "type": "tool_use", "id": "toolu_1", "name": "add", "input": { "a": 1, "b": 2 } }] },
                { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_1", "content": [{ "type": "text", "text": "3" }] }, { "type": "text", "text": "here you go" }] }
            ]))
        );
        assert_eq!(c.map(|c| c.warnings.len()), Some(1));
    }

    #[test]
    fn strict_schema_matches_anthropic_rules() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "age": { "type": "integer", "minimum": 0, "description": "Years" },
                "tags": { "type": "array", "items": { "type": "object", "properties": { "k": { "type": "string" } } }, "minItems": 2 },
                "pick": { "oneOf": [{ "type": "string" }, { "type": "null" }] }
            },
            "required": ["age"]
        });
        assert_eq!(
            strict_schema(&schema),
            json!({
                "type": "object",
                "properties": {
                    "age": { "type": "integer", "description": "Years {minimum: 0}" },
                    "tags": { "type": "array", "items": { "type": "object", "properties": { "k": { "type": "string" } }, "additionalProperties": false }, "description": "{minItems: 2}" },
                    "pick": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
                },
                "required": ["age"],
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn rejects_media_anthropic_cannot_read() {
        let audio = Media::new(MediaKind::Audio, Some("audio/wav".into()), MediaSource::Base64 { data: "AAAA".into() });
        let request = Request::new("anthropic/claude-haiku-5-5").message(Message::user("listen").with(audio));
        assert!(matches!(body(&request), Err(Error::Unsupported { .. })));
        let bmp = Media::image("image/bmp", [0u8]);
        assert!(matches!(body(&Request::new("anthropic/claude-haiku-5-5").message(Message::user("see").with(bmp))), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn merges_extra_body_and_trims_for_counting() {
        let request = Request::new("anthropic/claude-haiku-5-5").message(Message::user("Hi")).extra_body(ProviderId::Anthropic, json!({ "output_config": { "effort": "low" }, "new_field": 1 }));
        let c = body(&request).ok().map(|c| c.body);
        assert_eq!(c.as_ref().map(|b| (b["output_config"]["effort"].clone(), b["new_field"].clone())), Some((json!("low"), json!(1))));
        let counted = c.map(count_tokens_body);
        assert_eq!(counted.as_ref().map(|b| b.get("max_tokens").is_none() && b.get("stream").is_none() && b.get("model").is_some()), Some(true));
    }
}
