//! Interactions API server-sent events → neutral [`Event`]s.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use futures_util::StreamExt;
use genai_core::http::{SseEvent, sse_events};
use genai_core::{ApiError, ApiErrorKind, Error, Event, EventStream, ModelId, Opaque, Part, ProviderId, Reasoning, Result, StopReason, Text, ToolCall, Usage, Warning};
use serde_json::{Map, Value, json};

/// Codes Google uses when policy or safety blocked the output. Mid-stream they end the response as a refusal.
const BLOCKED: &[&str] =
    &["safety", "recitation", "language", "prohibited_content", "spii", "blocklist", "image_safety", "image_prohibited_content", "image_recitation", "image_other", "content_blocked"];

/// An Interactions API error (`{"code": "not_found", "message": …}`), or the older Google RPC form the models list uses
/// (`{"code": 404, "status": "NOT_FOUND", …}`), as an [`ApiError`].
pub(crate) fn api_error(error: &Value, status: Option<u16>) -> ApiError {
    let message = error.get("message").and_then(Value::as_str).unwrap_or_default().to_owned();
    let code = match error.get("code") {
        Some(Value::String(code)) => code.to_ascii_lowercase(),
        _ => error.get("status").and_then(Value::as_str).unwrap_or_default().to_ascii_lowercase(),
    };
    let details = error.get("details").and_then(Value::as_array).cloned().unwrap_or_default();
    let reason = details.iter().find_map(|d| d.get("reason").and_then(Value::as_str)).map(str::to_owned);
    let mut kind = match code.as_str() {
        "invalid_request" | "invalid_argument" | "parameter_unknown" | "out_of_range" | "failed_precondition" | "unimplemented" => ApiErrorKind::InvalidRequest,
        "authentication" | "unauthenticated" => ApiErrorKind::Auth,
        "payment_required" => ApiErrorKind::Billing,
        "permission_denied" => ApiErrorKind::Permission,
        "not_found" | "model_not_found" => ApiErrorKind::NotFound,
        "rate_limit_exceeded" | "too_many_requests" | "quota_exceeded" | "resource_exhausted" => ApiErrorKind::RateLimited,
        "api_error" | "internal" => ApiErrorKind::Server,
        "service_unavailable" | "unavailable" => ApiErrorKind::Overloaded,
        "deadline_exceeded" => ApiErrorKind::Timeout,
        c if BLOCKED.contains(&c) => ApiErrorKind::InvalidRequest,
        _ => status.map_or(ApiErrorKind::Other, ApiErrorKind::from_status),
    };
    if reason.as_deref() == Some("API_KEY_INVALID") {
        kind = ApiErrorKind::Auth;
    }
    let lower = message.to_ascii_lowercase();
    if kind == ApiErrorKind::InvalidRequest && (lower.contains("input token count") || lower.contains("exceeds the maximum number of tokens")) {
        kind = ApiErrorKind::ContextTooLong;
    }
    let mut e = ApiError::new(ProviderId::Google, kind, message);
    e.status = status;
    e.code = Some(reason.unwrap_or(code.clone())).filter(|c| !c.is_empty());
    // Google sends its retry delay in a RetryInfo detail ("30s"). Per-minute limits without one are still worth a short
    // wait; daily quotas (`quota_exceeded`) are not.
    e.retry_after = details.iter().find_map(|d| d.get("retryDelay").and_then(Value::as_str)).and_then(|s| s.trim_end_matches('s').parse::<f64>().ok()).map(Duration::from_secs_f64);
    if e.retry_after.is_none() && matches!(code.as_str(), "rate_limit_exceeded" | "too_many_requests") {
        e.retry_after = Some(Duration::from_secs(2));
    }
    e
}

/// A step being streamed.
#[derive(Debug)]
enum Block {
    Thought {
        signature: String,
        summary: Vec<Value>,
    },
    Output {
        content: Vec<Value>,
    },
    Call {
        step: Value,
        json: String,
    },
    /// Hosted tool calls and results, and any step type genai doesn't know: kept whole, with deltas merged in.
    Other {
        step: Value,
    },
}

/// Turns one interaction's SSE events into neutral events.
pub(crate) struct Decoder {
    blocks: HashMap<usize, Block>,
    usage: Usage,
    warnings: Vec<Warning>,
    finished: bool,
}

impl Decoder {
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Decoder { blocks: HashMap::new(), usage: Usage::default(), warnings, finished: false }
    }

    pub(crate) fn finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn decode(&mut self, sse: &SseEvent) -> Result<Vec<Event>> {
        if sse.data.trim() == "[DONE]" || sse.event.as_deref() == Some("done") {
            return Ok(vec![]);
        }
        let data: Value = serde_json::from_str(&sse.data).map_err(|e| Error::Decode { provider: ProviderId::Google, message: format!("{} event: {e}", sse.event.as_deref().unwrap_or("unnamed")) })?;
        let index = || data.get("index").and_then(Value::as_u64).map(|i| i as usize).ok_or_else(|| stream_error("step event without an index".into()));
        Ok(match data.get("event_type").and_then(Value::as_str).unwrap_or_default() {
            "interaction.created" => {
                let interaction = data.get("interaction").unwrap_or(&Value::Null);
                // Interactions that aren't stored have no id.
                let id = interaction.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
                let model = interaction.get("model").and_then(Value::as_str).unwrap_or_default().to_owned();
                vec![Event::Start { id, model: ModelId::new(ProviderId::Google, model) }]
            }
            "interaction.status_update" => match data.get("usage") {
                Some(usage) => {
                    self.usage = usage_from(usage);
                    vec![Event::Usage(self.usage.clone())]
                }
                None => vec![],
            },
            "step.start" => {
                let index = index()?;
                let (block, part) = start(data.get("step").cloned().unwrap_or_else(|| json!({})));
                self.blocks.insert(index, block);
                vec![Event::PartStart { index, part }]
            }
            "step.delta" => {
                let index = index()?;
                let delta = data.get("delta").unwrap_or(&Value::Null);
                let events = self.delta(index, delta)?;
                if let Some(usage) = data.get("metadata").and_then(|m| m.get("total_usage")) {
                    self.usage = usage_from(usage);
                }
                events
            }
            "step.stop" => {
                let index = index()?;
                let block = self.blocks.remove(&index).ok_or_else(|| stream_error(format!("step.stop for unknown step {index}")))?;
                if let Some(usage) = data.get("usage") {
                    self.usage = usage_from(usage);
                }
                vec![Event::PartStop { index, part: finish(block)? }]
            }
            "interaction.completed" => {
                let interaction = data.get("interaction").unwrap_or(&Value::Null);
                if let Some(usage) = interaction.get("usage") {
                    self.usage = usage_from(usage);
                }
                let status = interaction.get("status").and_then(Value::as_str).unwrap_or_default();
                let stop_reason = match status {
                    "completed" => StopReason::Complete,
                    "requires_action" => StopReason::ToolCalls,
                    "incomplete" => StopReason::Length,
                    "requires_continuation" => StopReason::Paused,
                    "failed" => {
                        let error = interaction.get("errors").and_then(Value::as_array).and_then(|e| e.first()).cloned().unwrap_or_else(|| json!({ "message": "the interaction failed" }));
                        return Err(api_error(&error, None).into());
                    }
                    other => StopReason::Other(other.to_owned()),
                };
                self.finished = true;
                vec![Event::Finish { stop_reason, usage: self.usage.clone(), warnings: std::mem::take(&mut self.warnings) }]
            }
            "error" => {
                let error = data.get("error").cloned().unwrap_or_default();
                let code = error.get("code").and_then(Value::as_str).unwrap_or_default();
                if BLOCKED.contains(&code) {
                    self.finished = true;
                    let message = error.get("message").and_then(Value::as_str).filter(|m| !m.is_empty()).unwrap_or(code);
                    return Ok(vec![Event::Finish { stop_reason: StopReason::Refused(Some(message.to_owned())), usage: self.usage.clone(), warnings: std::mem::take(&mut self.warnings) }]);
                }
                return Err(api_error(&error, None).into());
            }
            other => {
                tracing::trace!(event = other, "skipping unknown Gemini stream event");
                vec![]
            }
        })
    }

    fn delta(&mut self, index: usize, delta: &Value) -> Result<Vec<Event>> {
        let block = self.blocks.get_mut(&index).ok_or_else(|| stream_error(format!("delta for unknown step {index}")))?;
        let str_field = |key: &str| delta.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
        Ok(match (delta.get("type").and_then(Value::as_str).unwrap_or_default(), block) {
            ("text", Block::Output { content }) => {
                let more = str_field("text");
                match content.last_mut() {
                    Some(last) if last.get("type").and_then(Value::as_str) == Some("text") => {
                        let text = format!("{}{more}", last.get("text").and_then(Value::as_str).unwrap_or_default());
                        last["text"] = Value::String(text);
                    }
                    _ => content.push(json!({ "type": "text", "text": more })),
                }
                if more.is_empty() { vec![] } else { vec![Event::TextDelta { index, text: more }] }
            }
            ("text_annotation_delta", Block::Output { content }) => {
                if let Some(last) = content.last_mut().and_then(Value::as_object_mut) {
                    let annotations = last.entry("annotations").or_insert_with(|| json!([]));
                    if let (Some(list), Some(more)) = (annotations.as_array_mut(), delta.get("annotations").and_then(Value::as_array)) {
                        list.extend(more.iter().cloned());
                    }
                }
                vec![]
            }
            (_, Block::Output { content }) => {
                // Image, audio or other output content arrives whole.
                content.push(delta.clone());
                vec![]
            }
            ("thought_signature", Block::Thought { signature, .. }) => {
                signature.push_str(&str_field("signature"));
                vec![]
            }
            ("thought_summary", Block::Thought { summary, .. }) => {
                let item = delta.get("content").cloned().unwrap_or_default();
                let text = item.get("text").and_then(Value::as_str).unwrap_or_default().to_owned();
                summary.push(item);
                if text.is_empty() { vec![] } else { vec![Event::ReasoningDelta { index, text }] }
            }
            ("arguments_delta", Block::Call { json, .. }) => {
                let more = str_field("arguments");
                json.push_str(&more);
                if more.is_empty() { vec![] } else { vec![Event::ToolCallDelta { index, arguments: more }] }
            }
            (_, Block::Other { step }) => {
                if let (Some(step), Some(delta)) = (step.as_object_mut(), delta.as_object()) {
                    for (key, value) in delta.iter().filter(|(k, _)| k.as_str() != "type") {
                        step.insert(key.clone(), value.clone());
                    }
                }
                vec![]
            }
            (kind, _) => {
                tracing::trace!(kind, index, "skipping unexpected Gemini delta");
                vec![]
            }
        })
    }
}

/// Gemini usage → neutral usage. Gemini counts thoughts apart from output, and tool-use prompts (search results, for
/// example) apart from input; neutral usage folds both in.
fn usage_from(usage: &Value) -> Usage {
    let n = |key: &str| usage.get(key).and_then(Value::as_u64);
    let thoughts = n("total_thought_tokens");
    let mut u = Usage::default();
    u.input_tokens = n("total_input_tokens").unwrap_or(0) + n("total_tool_use_tokens").unwrap_or(0);
    u.output_tokens = n("total_output_tokens").unwrap_or(0) + thoughts.unwrap_or(0);
    u.reasoning_tokens = thoughts;
    u.cache_read_tokens = n("total_cached_tokens");
    u
}

fn stream_error(message: String) -> Error {
    Error::Stream { provider: ProviderId::Google, message }
}

fn google(data: Value) -> Opaque {
    Opaque::new(ProviderId::Google, data)
}

fn start(step: Value) -> (Block, Part) {
    let str_field = |key: &str| step.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
    match step.get("type").and_then(Value::as_str).unwrap_or_default() {
        "thought" => {
            let summary = step.get("summary").and_then(Value::as_array).cloned().unwrap_or_default();
            (Block::Thought { signature: str_field("signature"), summary }, Part::Reasoning(Reasoning { text: None, opaque: None }))
        }
        "model_output" => {
            let content = step.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
            (Block::Output { content }, Part::text(""))
        }
        "function_call" => {
            let part = Part::ToolCall(ToolCall { id: str_field("id"), name: str_field("name"), arguments: json!({}), opaque: None });
            (Block::Call { step, json: String::new() }, part)
        }
        _ => (Block::Other { step: step.clone() }, Part::Opaque(google(step))),
    }
}

fn finish(block: Block) -> Result<Part> {
    Ok(match block {
        Block::Thought { signature, summary } => {
            let text: Vec<&str> = summary.iter().filter_map(|s| s.get("text").and_then(Value::as_str)).collect();
            let text = (!text.is_empty()).then(|| text.concat());
            let mut step = Map::new();
            step.insert("type".into(), Value::String("thought".into()));
            step.insert("signature".into(), Value::String(signature));
            if !summary.is_empty() {
                step.insert("summary".into(), Value::Array(summary));
            }
            Part::Reasoning(Reasoning { text, opaque: Some(google(Value::Object(step))) })
        }
        Block::Output { content } => {
            let plain = content.iter().all(|c| c.get("type").and_then(Value::as_str) == Some("text") && c.get("annotations").is_none());
            let text: String = content.iter().filter_map(|c| c.get("text").and_then(Value::as_str)).collect();
            let step = json!({ "type": "model_output", "content": content });
            if plain {
                Part::text(text)
            } else if !text.is_empty() {
                Part::Text(Text { text, opaque: Some(google(step)) })
            } else {
                Part::Opaque(google(step))
            }
        }
        Block::Call { mut step, json } => {
            if !json.is_empty() {
                step["arguments"] = serde_json::from_str(&json).map_err(|e| Error::Decode { provider: ProviderId::Google, message: format!("tool call arguments aren't JSON: {e}") })?;
            }
            let field = |key: &str| step.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
            let arguments = step.get("arguments").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
            // Keep the original step when it carries more than genai models (a signature, for example).
            let extra = step.as_object().is_some_and(|s| s.keys().any(|k| !matches!(k.as_str(), "type" | "id" | "name" | "arguments")));
            Part::ToolCall(ToolCall { id: field("id"), name: field("name"), arguments, opaque: extra.then(|| google(step)) })
        }
        Block::Other { step } => Part::Opaque(google(step)),
    })
}

/// The neutral event stream for a streaming Interactions API response.
pub(crate) fn events(response: genai_core::http::reqwest::Response, warnings: Vec<Warning>) -> EventStream {
    let sse = Box::pin(sse_events(ProviderId::Google, response));
    let state = (sse, Decoder::new(warnings), VecDeque::<Event>::new(), false);
    Box::pin(futures_util::stream::unfold(state, |(mut sse, mut decoder, mut pending, mut done)| async move {
        loop {
            if let Some(event) = pending.pop_front() {
                return Some((Ok(event), (sse, decoder, pending, done)));
            }
            if done {
                return None;
            }
            match sse.next().await {
                Some(Ok(event)) => match decoder.decode(&event) {
                    Ok(events) => pending.extend(events),
                    Err(e) => {
                        done = true;
                        return Some((Err(e), (sse, decoder, pending, done)));
                    }
                },
                Some(Err(e)) => {
                    done = true;
                    return Some((Err(e), (sse, decoder, pending, done)));
                }
                None => {
                    done = true;
                    if !decoder.finished() {
                        return Some((Err(stream_error("the stream ended before interaction.completed".into())), (sse, decoder, pending, done)));
                    }
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use genai_core::Accumulator;
    use genai_core::http::SseParser;

    use super::*;

    fn decode(fixture: &str) -> Result<Vec<Event>> {
        let mut parser = SseParser::new();
        let mut decoder = Decoder::new(vec![]);
        let mut events = Vec::new();
        for sse in parser.push(fixture.as_bytes()).into_iter().chain(parser.finish()) {
            events.extend(decoder.decode(&sse)?);
        }
        Ok(events)
    }

    fn response(fixture: &str) -> Result<genai_core::Response> {
        let mut acc = Accumulator::new();
        for event in decode(fixture)? {
            acc.push(&event);
        }
        acc.finish()
    }

    #[test]
    fn decodes_a_recorded_stream() {
        let r = response(include_str!("../tests/fixtures/text.sse")).ok();
        assert_eq!(r.as_ref().map(|r| r.text()).as_deref(), Some("Hello there, friend!"));
        assert_eq!(r.as_ref().map(|r| r.stop_reason.clone()), Some(StopReason::Complete));
        assert_eq!(r.as_ref().map(|r| r.model.to_string()).as_deref(), Some("google/gemini-3.5-flash-lite"));
        assert_eq!(r.as_ref().map(|r| (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens)), Some((7, 5, Some(0))));
        // The thought step comes back with its signature, ready to send on the next turn.
        let thought = r.and_then(|r| r.message.parts.into_iter().find_map(|p| if let Part::Reasoning(x) = p { x.opaque } else { None }));
        assert!(thought.is_some_and(|o| o.data["type"] == "thought" && o.data["signature"].as_str().is_some_and(|s| s.len() > 20)));
    }

    #[test]
    fn decodes_tool_calls_and_summaries() {
        let fixture = r#"event: interaction.created
data: {"interaction":{"model":"gemini-4-hydrogen"},"event_type":"interaction.created"}

event: step.start
data: {"index":0,"step":{"type":"thought"},"event_type":"step.start"}

event: step.delta
data: {"index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"Need weather."}},"event_type":"step.delta"}

event: step.delta
data: {"index":0,"delta":{"signature":"sig","type":"thought_signature"},"event_type":"step.delta"}

event: step.stop
data: {"index":0,"event_type":"step.stop"}

event: step.start
data: {"index":1,"step":{"id":"ktr5","type":"function_call","name":"get_weather","arguments":{}},"event_type":"step.start"}

event: step.delta
data: {"index":1,"delta":{"arguments":"{\"location\":\"Paris\"}","type":"arguments_delta"},"event_type":"step.delta"}

event: step.stop
data: {"index":1,"event_type":"step.stop"}

event: interaction.completed
data: {"interaction":{"status":"requires_action","usage":{"total_input_tokens":100,"total_output_tokens":20,"total_thought_tokens":30,"total_cached_tokens":40}},"event_type":"interaction.completed"}

event: done
data: [DONE]
"#;
        let r = response(fixture).ok();
        assert_eq!(r.as_ref().map(|r| r.stop_reason.clone()), Some(StopReason::ToolCalls));
        assert_eq!(r.as_ref().and_then(|r| r.reasoning()).as_deref(), Some("Need weather."));
        let call = r.as_ref().and_then(|r| r.tool_calls().next().cloned());
        assert_eq!(call.map(|c| (c.id, c.name, c.arguments)), Some(("ktr5".into(), "get_weather".into(), json!({ "location": "Paris" }))));
        assert_eq!(r.map(|r| (r.usage.input_tokens, r.usage.output_tokens, r.usage.cache_read_tokens)), Some((100, 50, Some(40))));
    }

    #[test]
    fn blocked_output_is_a_refusal_and_other_errors_fail() {
        let blocked = "event: interaction.created\ndata: {\"interaction\":{\"model\":\"m\"},\"event_type\":\"interaction.created\"}\n\nevent: error\ndata: {\"event_type\":\"error\",\"error\":{\"code\":\"safety\",\"message\":\"Blocked for safety.\"}}\n\n";
        assert_eq!(response(blocked).ok().map(|r| r.stop_reason), Some(StopReason::Refused(Some("Blocked for safety.".into()))));

        let failed = "event: error\ndata: {\"event_type\":\"error\",\"error\":{\"code\":\"service_unavailable\",\"message\":\"Try later.\"}}\n\n";
        match decode(failed) {
            Err(Error::Api(e)) => assert!(e.kind == ApiErrorKind::Overloaded && e.is_retryable()),
            other => panic!("expected an overloaded error, got {other:?}"),
        }
    }

    #[test]
    fn reads_both_error_shapes() {
        let interactions = api_error(&json!({ "code": "model_not_found", "message": "Model 'x' not found." }), Some(404));
        assert_eq!((interactions.kind, interactions.code.as_deref()), (ApiErrorKind::NotFound, Some("model_not_found")));
        let rpc = api_error(
            &json!({ "code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT", "details": [{ "@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID" }] }),
            Some(400),
        );
        assert_eq!((rpc.kind, rpc.code.as_deref()), (ApiErrorKind::Auth, Some("API_KEY_INVALID")));
        let limited =
            api_error(&json!({ "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "slow", "details": [{ "@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s" }] }), Some(429));
        assert_eq!(limited.retry_after, Some(Duration::from_secs(7)));
        assert!(!api_error(&json!({ "code": "quota_exceeded", "message": "daily" }), Some(429)).is_retryable());
        assert!(api_error(&json!({ "code": "rate_limit_exceeded", "message": "minute" }), Some(429)).is_retryable());
    }
}
