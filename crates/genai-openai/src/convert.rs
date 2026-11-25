//! Neutral [`Request`] → Responses API request body.

use genai_core::{Effort, Error, Media, MediaKind, MediaSource, Message, OutputFormat, Part, ProviderId, ReasoningDisplay, Request, Result, Role, ToolChoice, ToolResult, Warning};
use serde_json::{Map, Value, json};

use crate::models::Spec;
use crate::options::OpenAiOptions;

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
            return Err(Error::Unsupported { provider: ProviderId::OpenAi, feature: message });
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
    let options: OpenAiOptions = match request.options_for(&ProviderId::OpenAi) {
        Some(value) => serde_json::from_value(value.clone()).map_err(|e| Error::InvalidInput(format!("bad OpenAI options: {e}")))?,
        None => OpenAiOptions::default(),
    };

    let mut body = Map::new();
    body.insert("model".into(), Value::String(request.model.name().to_owned()));

    // Instructions go first in `input` as a developer message: from GPT-6.2 on, top-level `instructions` are placed after
    // the input, which keeps them out of the prompt cache.
    let mut input = Vec::new();
    if let Some(instructions) = request.instructions.as_deref().filter(|s| !s.is_empty()) {
        input.push(json!({ "role": "developer", "content": instructions }));
    }
    input.extend(items(&request.messages, &mut notes)?);
    body.insert("input".into(), Value::Array(input));

    if let Some(mut max) = request.max_output_tokens {
        if let Some(spec) = spec
            && max > spec.max_output_tokens
        {
            notes.approximate("max_output_tokens_clamped", format!("{} allows at most {} output tokens; asked for {max}", spec.id, spec.max_output_tokens))?;
            max = spec.max_output_tokens;
        }
        body.insert("max_output_tokens".into(), json!(max));
    }

    let mut reasoning = Map::new();
    if let Some(effort) = effort(request.effort, spec, &mut notes)? {
        reasoning.insert("effort".into(), Value::String(effort.as_str().into()));
    }
    match request.reasoning_display {
        Some(ReasoningDisplay::Summary) => {
            reasoning.insert("summary".into(), Value::String("auto".into()));
        }
        Some(ReasoningDisplay::Detailed) => {
            reasoning.insert("summary".into(), Value::String("detailed".into()));
        }
        _ => {}
    }
    if !reasoning.is_empty() {
        body.insert("reasoning".into(), Value::Object(reasoning));
    }

    if let Some(temperature) = request.temperature {
        if spec.is_some_and(|s| !s.sampling) {
            notes.approximate("temperature_dropped", format!("{} doesn't accept temperature; it was left out", request.model.name()))?;
        } else {
            body.insert("temperature".into(), json!(temperature));
        }
    }
    if !request.stop_sequences.is_empty() {
        notes.approximate("stop_sequences_dropped", "the Responses API has no stop sequences; they were left out".into())?;
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
                t.insert("parameters".into(), if tool.strict { strict_schema(&tool.parameters) } else { clean(&tool.parameters) });
                t.insert("strict".into(), Value::Bool(tool.strict));
                Value::Object(t)
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
        if let Some(choice) = &request.tool_choice {
            let choice = match choice {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::None => json!("none"),
                ToolChoice::Required => json!("required"),
                ToolChoice::Named(name) => json!({ "type": "function", "name": name }),
                other => return Err(Error::Unsupported { provider: ProviderId::OpenAi, feature: format!("tool choice {other:?}") }),
            };
            body.insert("tool_choice".into(), choice);
        }
        if let Some(parallel) = request.parallel_tool_calls {
            body.insert("parallel_tool_calls".into(), Value::Bool(parallel));
        }
    }

    let mut text = Map::new();
    if let Some(OutputFormat::JsonSchema { name, schema }) = &request.output {
        text.insert("format".into(), json!({ "type": "json_schema", "name": schema_name(name), "schema": strict_schema(schema), "strict": true }));
    }
    if let Some(verbosity) = options.verbosity {
        text.insert("verbosity".into(), serde_json::to_value(verbosity).unwrap_or_default());
    }
    if !text.is_empty() {
        body.insert("text".into(), Value::Object(text));
    }

    // Stateless by default: OpenAI keeps nothing, and reasoning comes back encrypted so it can be sent on the next turn.
    let store = options.store.unwrap_or(false);
    body.insert("store".into(), Value::Bool(store));
    if !store {
        body.insert("include".into(), json!(["reasoning.encrypted_content"]));
    }
    if let Some(user) = &request.user_id {
        body.insert("safety_identifier".into(), Value::String(user.clone()));
    }
    if let Some(key) = &options.prompt_cache_key {
        body.insert("prompt_cache_key".into(), Value::String(key.clone()));
    }
    if let Some(tier) = options.service_tier {
        body.insert("service_tier".into(), serde_json::to_value(tier).unwrap_or_default());
    }
    if stream {
        body.insert("stream".into(), Value::Bool(true));
    }

    let mut body = Value::Object(body);
    for extra in request.extra_body.iter().filter(|e| e.provider == ProviderId::OpenAi) {
        merge(&mut body, &extra.value);
    }
    Ok(Converted { body, warnings: notes.warnings })
}

/// The `reasoning.effort` for the asked-for effort.
fn effort(effort: Option<Effort>, spec: Option<&Spec>, notes: &mut Notes) -> Result<Option<Effort>> {
    let Some(asked) = effort else { return Ok(None) };
    let Some(spec) = spec.filter(|s| !s.efforts.is_empty()) else { return Ok(Some(asked)) };
    if spec.efforts.contains(&asked) {
        return Ok(Some(asked));
    }
    let lowest = spec.efforts.first().copied().unwrap_or(Effort::Low);
    let used = spec.efforts.iter().copied().filter(|e| *e <= asked).max().unwrap_or(lowest);
    if asked == Effort::None {
        notes.approximate("reasoning_always_on", format!("{} can't turn reasoning off; used {}", spec.id, used.as_str()))?;
    } else {
        notes.approximate("effort_clamped", format!("{} doesn't support {} effort; used {}", spec.id, asked.as_str(), used.as_str()))?;
    }
    Ok(Some(used))
}

/// The conversation as Responses API input items.
fn items(messages: &[Message], notes: &mut Notes) -> Result<Vec<Value>> {
    let mut items = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        match message.role {
            Role::User => {
                let mut results = Vec::new();
                let mut content = Vec::new();
                for part in &message.parts {
                    match part {
                        Part::Text(t) if t.text.is_empty() => {}
                        Part::Text(t) => content.push(json!({ "type": "input_text", "text": t.text })),
                        Part::Media(m) => content.push(media(m)?),
                        Part::ToolResult(r) => results.push(tool_result(r)?),
                        Part::ToolCall(_) => return Err(Error::InvalidInput(format!("message {i}: tool calls belong in assistant messages"))),
                        Part::Opaque(o) if o.provider == ProviderId::OpenAi => items.push(o.data.clone()),
                        _ => notes.note("part_dropped", format!("message {i}: left out a part OpenAI can't take from the user")),
                    }
                }
                items.extend(results);
                if !content.is_empty() {
                    items.push(json!({ "role": "user", "content": content }));
                }
            }
            Role::Assistant => {
                for part in &message.parts {
                    match part {
                        // OpenAI's own items go back exactly as they came, ids and `phase` included.
                        Part::Text(t) => match &t.opaque {
                            Some(o) if o.provider == ProviderId::OpenAi => items.push(o.data.clone()),
                            _ if t.text.is_empty() => {}
                            _ => items.push(json!({ "role": "assistant", "content": t.text })),
                        },
                        Part::Refusal { text } if !text.is_empty() => items.push(json!({ "role": "assistant", "content": text })),
                        Part::Refusal { .. } => {}
                        Part::Reasoning(r) => match &r.opaque {
                            Some(o) if o.provider == ProviderId::OpenAi => items.push(o.data.clone()),
                            Some(o) => notes.note("reasoning_dropped", format!("message {i}: left out reasoning from {}, which OpenAI can't read", o.provider)),
                            None => notes.note("reasoning_dropped", format!("message {i}: left out reasoning with no OpenAI data")),
                        },
                        Part::ToolCall(call) => match &call.opaque {
                            Some(o) if o.provider == ProviderId::OpenAi => items.push(o.data.clone()),
                            _ => items.push(json!({ "type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments.to_string() })),
                        },
                        Part::Opaque(o) if o.provider == ProviderId::OpenAi => items.push(o.data.clone()),
                        Part::Opaque(o) => notes.note("opaque_dropped", format!("message {i}: left out {} data that OpenAI can't read", o.provider)),
                        Part::ToolResult(_) => return Err(Error::InvalidInput(format!("message {i}: tool results belong in user messages"))),
                        _ => notes.note("part_dropped", format!("message {i}: left out a part OpenAI can't take from the assistant")),
                    }
                }
            }
            other => return Err(Error::Unsupported { provider: ProviderId::OpenAi, feature: format!("{other:?} messages") }),
        }
    }
    Ok(items)
}

fn tool_result(result: &ToolResult) -> Result<Value> {
    let only_text = result.content.iter().all(|p| matches!(p, Part::Text(_)));
    let output = if only_text {
        Value::String(result.content.iter().filter_map(Part::as_text).collect())
    } else {
        let mut content = Vec::new();
        for part in &result.content {
            match part {
                Part::Text(t) if t.text.is_empty() => {}
                Part::Text(t) => content.push(json!({ "type": "input_text", "text": t.text })),
                Part::Media(m) => content.push(media(m)?),
                _ => return Err(Error::Unsupported { provider: ProviderId::OpenAi, feature: "that part inside a tool result".into() }),
            }
        }
        Value::Array(content)
    };
    // OpenAI has no error flag on tool output; say it in the output instead.
    let output = match (result.is_error, output) {
        (true, Value::String(s)) => Value::String(format!("Error: {s}")),
        (_, output) => output,
    };
    Ok(json!({ "type": "function_call_output", "call_id": result.call_id, "output": output }))
}

fn media(media: &Media) -> Result<Value> {
    let data_url = |data: &str| -> Result<String> {
        let mime = media.mime_type.as_deref().ok_or_else(|| Error::InvalidInput("inline media needs a media type".into()))?;
        Ok(format!("data:{mime};base64,{data}"))
    };
    Ok(match (media.kind, &media.source) {
        (MediaKind::Image, MediaSource::Base64 { data }) => json!({ "type": "input_image", "image_url": data_url(data)? }),
        (MediaKind::Image, MediaSource::Url { url }) => json!({ "type": "input_image", "image_url": url }),
        (MediaKind::Image, MediaSource::File { id }) => json!({ "type": "input_image", "file_id": id }),
        (MediaKind::Document, MediaSource::Base64 { data }) => {
            let filename = media.title.clone().unwrap_or_else(|| if media.mime_type.as_deref() == Some("application/pdf") { "document.pdf".into() } else { "document".into() });
            json!({ "type": "input_file", "file_data": data_url(data)?, "filename": filename })
        }
        (MediaKind::Document, MediaSource::Url { url }) => json!({ "type": "input_file", "file_url": url }),
        (MediaKind::Document, MediaSource::File { id }) => json!({ "type": "input_file", "file_id": id }),
        (MediaKind::Document, MediaSource::Text { text }) => {
            let text = match &media.title {
                Some(title) => format!("{title}\n\n{text}"),
                None => text.clone(),
            };
            json!({ "type": "input_text", "text": text })
        }
        (kind, _) => return Err(Error::Unsupported { provider: ProviderId::OpenAi, feature: format!("{kind:?} input from this source") }),
    })
}

/// A schema name OpenAI accepts: letters, digits, `_` and `-`, at most 64 characters.
fn schema_name(name: &str) -> String {
    let name: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).take(64).collect();
    if name.is_empty() { "response".into() } else { name }
}

fn clean(schema: &Value) -> Value {
    let mut schema = schema.clone();
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
    }
    schema
}

/// Adjusts a JSON Schema to what OpenAI's strict mode requires: every object lists all its properties as required and
/// has `additionalProperties: false`; properties that were optional become nullable instead.
pub(crate) fn strict_schema(schema: &Value) -> Value {
    let mut schema = clean(schema);
    tighten(&mut schema);
    schema
}

fn tighten(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            if map.get("properties").is_some_and(Value::is_object) {
                let required: Vec<String> = map.get("required").and_then(Value::as_array).map(|r| r.iter().filter_map(Value::as_str).map(str::to_owned).collect()).unwrap_or_default();
                let Some(Value::Object(properties)) = map.get_mut("properties") else { return };
                for (name, property) in properties.iter_mut() {
                    tighten(property);
                    if !required.contains(name) {
                        nullable(property);
                    }
                }
                let all: Vec<Value> = properties.keys().cloned().map(Value::String).collect();
                map.insert("required".into(), Value::Array(all));
                map.insert("additionalProperties".into(), Value::Bool(false));
            } else if map.get("type").is_some_and(|t| t == "object") {
                map.insert("additionalProperties".into(), Value::Bool(false));
                map.entry("properties").or_insert_with(|| json!({}));
                map.entry("required").or_insert_with(|| json!([]));
            }
            for key in ["$defs", "definitions"] {
                if let Some(Value::Object(children)) = map.get_mut(key) {
                    children.values_mut().for_each(tighten);
                }
            }
            for key in ["items", "anyOf", "allOf", "oneOf"] {
                if let Some(child) = map.get_mut(key) {
                    tighten(child);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(tighten),
        _ => {}
    }
}

/// Lets `schema` also be `null`.
fn nullable(schema: &mut Value) {
    let Value::Object(map) = schema else { return };
    match map.get_mut("type") {
        Some(Value::String(t)) if t != "null" => {
            let t = t.clone();
            map.insert("type".into(), json!([t, "null"]));
        }
        Some(Value::Array(types)) if !types.iter().any(|t| t == "null") => types.push(json!("null")),
        Some(_) => {}
        None => {
            let inner = Value::Object(std::mem::take(map));
            map.insert("anyOf".into(), json!([inner, { "type": "null" }]));
        }
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

#[cfg(test)]
mod tests {
    use genai_core::{Opaque, Reasoning, Text, Tool, ToolCall};

    use super::*;
    use crate::models::{catalog, find};
    use crate::options::{ServiceTier, Verbosity};

    fn body(request: &Request) -> Result<Converted> {
        let specs = catalog();
        request_body(request, find(&specs, request.model.name()), true)
    }

    #[test]
    fn builds_a_plain_stateless_request() {
        let request = Request::new("openai/gpt-6-luna").instructions("Be brief.").message(Message::user("Hi")).max_output_tokens(256).effort(Effort::Low).reasoning_display(ReasoningDisplay::Summary);
        assert_eq!(
            body(&request).ok().map(|c| c.body),
            Some(json!({
                "model": "gpt-6-luna",
                "input": [{ "role": "developer", "content": "Be brief." }, { "role": "user", "content": [{ "type": "input_text", "text": "Hi" }] }],
                "max_output_tokens": 256,
                "reasoning": { "effort": "low", "summary": "auto" },
                "store": false,
                "include": ["reasoning.encrypted_content"],
                "stream": true
            }))
        );
    }

    fn effort_on(model: &str, effort: Effort) -> (Value, Vec<String>) {
        let request = Request::new(format!("openai/{model}")).message(Message::user("Hi")).effort(effort);
        match body(&request) {
            Ok(c) => (c.body["reasoning"]["effort"].clone(), c.warnings.into_iter().map(|w| w.code).collect()),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn maps_effort_per_model() {
        assert_eq!(effort_on("gpt-6.2-astra", Effort::Max), (json!("max"), vec![]));
        assert_eq!(effort_on("gpt-6.2-astra", Effort::None), (json!("low"), vec!["reasoning_always_on".into()]));
        assert_eq!(effort_on("gpt-6-luna", Effort::None), (json!("none"), vec![]));
        assert_eq!(effort_on("gpt-6-luna", Effort::Minimal), (json!("none"), vec!["effort_clamped".into()]));
        assert_eq!(effort_on("gpt-5.5", Effort::Max), (json!("xhigh"), vec!["effort_clamped".into()]));
        assert_eq!(effort_on("gpt-4.1-unknown", Effort::Minimal), (json!("minimal"), vec![]));
    }

    #[test]
    fn drops_what_the_model_rejects() {
        let request = Request::new("openai/gpt-6.2-sol").message(Message::user("Hi")).temperature(0.3).stop_sequence("END");
        let c = body(&request).ok();
        assert_eq!(c.as_ref().map(|c| c.body.get("temperature").is_none()), Some(true));
        assert_eq!(c.map(|c| c.warnings.into_iter().map(|w| w.code).collect::<Vec<_>>()), Some(vec!["temperature_dropped".to_owned(), "stop_sequences_dropped".to_owned()]));
        let older = Request::new("openai/gpt-4.1").message(Message::user("Hi")).temperature(0.3);
        assert_eq!(body(&older).ok().map(|c| c.body["temperature"].clone()), Some(json!(0.3_f32)));
        assert!(matches!(body(&request.strict(true)), Err(Error::Unsupported { .. })));
    }

    #[test]
    fn maps_tools_choices_and_formats() {
        let tool = Tool::function("lookup").parameters(json!({ "type": "object", "properties": { "q": { "type": "string" }, "limit": { "type": "integer" } }, "required": ["q"] })).strict(true);
        let request = Request::new("openai/gpt-6.2-sol")
            .message(Message::user("find"))
            .tool(tool)
            .tool_choice(ToolChoice::Named("lookup".into()))
            .parallel_tool_calls(false)
            .output(OutputFormat::json_schema("my answer!", json!({ "type": "object", "properties": { "a": { "type": "integer" } } })))
            .options(OpenAiOptions::new().service_tier(ServiceTier::Flex).verbosity(Verbosity::Low).prompt_cache_key("conv-1"))
            .user_id("user-hash");
        let b = body(&request).ok().map(|c| c.body);
        assert_eq!(
            b.as_ref().map(|b| b["tools"][0].clone()),
            Some(
                json!({ "type": "function", "name": "lookup", "parameters": { "type": "object", "properties": { "q": { "type": "string" }, "limit": { "type": ["integer", "null"] } }, "required": ["q", "limit"], "additionalProperties": false }, "strict": true })
            )
        );
        assert_eq!(b.as_ref().map(|b| (b["tool_choice"].clone(), b["parallel_tool_calls"].clone())), Some((json!({ "type": "function", "name": "lookup" }), json!(false))));
        assert_eq!(
            b.as_ref().map(|b| b["text"].clone()),
            Some(
                json!({ "format": { "type": "json_schema", "name": "my_answer_", "schema": { "type": "object", "properties": { "a": { "type": ["integer", "null"] } }, "required": ["a"], "additionalProperties": false }, "strict": true }, "verbosity": "low" })
            )
        );
        assert_eq!(b.map(|b| (b["service_tier"].clone(), b["prompt_cache_key"].clone(), b["safety_identifier"].clone())), Some((json!("flex"), json!("conv-1"), json!("user-hash"))));
    }

    #[test]
    fn sends_back_items_exactly() {
        let item = json!({ "type": "message", "id": "msg_1", "role": "assistant", "content": [{ "type": "output_text", "text": "Let me look." }], "phase": "commentary" });
        let reasoning = json!({ "type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc" });
        let call_item = json!({ "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "add", "arguments": "{\"a\":1}" });
        let call = ToolCall { id: "call_1".into(), name: "add".into(), arguments: json!({ "a": 1 }), opaque: Some(Opaque::new(ProviderId::OpenAi, call_item.clone())) };
        let bare = ToolCall { id: "call_2".into(), name: "add".into(), arguments: json!({ "a": 2 }), opaque: None };
        let request = Request::new("openai/gpt-6-luna").messages([
            Message::user("add"),
            Message::new(
                Role::Assistant,
                vec![
                    Part::Reasoning(Reasoning { text: None, opaque: Some(Opaque::new(ProviderId::OpenAi, reasoning.clone())) }),
                    Part::Text(Text { text: "Let me look.".into(), opaque: Some(Opaque::new(ProviderId::OpenAi, item.clone())) }),
                    Part::ToolCall(call.clone()),
                    Part::ToolCall(bare.clone()),
                    Part::text("plain"),
                ],
            ),
            Message::tool_results([ToolResult::new(&call, "1"), ToolResult::error(&bare, "boom")]),
        ]);
        assert_eq!(
            body(&request).ok().map(|c| c.body["input"].clone()),
            Some(json!([
                { "role": "user", "content": [{ "type": "input_text", "text": "add" }] },
                reasoning,
                item,
                call_item,
                { "type": "function_call", "call_id": "call_2", "name": "add", "arguments": "{\"a\":2}" },
                { "role": "assistant", "content": "plain" },
                { "type": "function_call_output", "call_id": "call_1", "output": "1" },
                { "type": "function_call_output", "call_id": "call_2", "output": "Error: boom" }
            ]))
        );
    }

    #[test]
    fn maps_media() {
        let request = Request::new("openai/gpt-6-luna").message(Message::user("see").with(Media::image("image/png", [1u8, 2])).with(Media::pdf([0u8]).title("a.pdf")));
        let b = body(&request).ok().map(|c| c.body["input"][0]["content"].clone());
        assert_eq!(
            b,
            Some(json!([
                { "type": "input_text", "text": "see" },
                { "type": "input_image", "image_url": "data:image/png;base64,AQI=" },
                { "type": "input_file", "file_data": "data:application/pdf;base64,AA==", "filename": "a.pdf" }
            ]))
        );
    }
}
