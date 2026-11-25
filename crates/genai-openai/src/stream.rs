//! Responses API server-sent events → neutral [`Event`]s.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use futures_util::StreamExt;
use genai_core::http::{SseEvent, sse_events};
use genai_core::{ApiError, ApiErrorKind, Error, Event, EventStream, ModelId, Opaque, Part, ProviderId, Reasoning, Result, StopReason, Text, ToolCall, Usage, Warning};
use serde_json::{Value, json};

/// An OpenAI error object (`{"message", "type", "param", "code"}`) as an [`ApiError`].
pub(crate) fn api_error(error: &Value, status: Option<u16>) -> ApiError {
    let field = |key: &str| error.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
    let (kind_name, code, message) = (field("type"), field("code"), field("message"));
    let mut kind = match code.as_str() {
        "invalid_api_key" => ApiErrorKind::Auth,
        "model_not_found" => ApiErrorKind::NotFound,
        "context_length_exceeded" => ApiErrorKind::ContextTooLong,
        "insufficient_quota" | "credit_balance_exhausted" | "organization_spend_limit_exceeded" | "project_spend_limit_exceeded" | "organization_usage_limit_exceeded" => ApiErrorKind::Billing,
        "rate_limit_exceeded" | "slow_down" => ApiErrorKind::RateLimited,
        "server_is_overloaded" => ApiErrorKind::Overloaded,
        "server_error" => ApiErrorKind::Server,
        _ => match kind_name.as_str() {
            "invalid_request_error" => ApiErrorKind::InvalidRequest,
            "authentication_error" => ApiErrorKind::Auth,
            "permission_error" | "permission_denied_error" => ApiErrorKind::Permission,
            "not_found_error" => ApiErrorKind::NotFound,
            "rate_limit_error" => ApiErrorKind::RateLimited,
            "service_unavailable_error" => ApiErrorKind::Overloaded,
            "server_error" | "api_error" => ApiErrorKind::Server,
            _ => status.map_or(ApiErrorKind::Other, ApiErrorKind::from_status),
        },
    };
    // A 429 is also how OpenAI reports spent credit; that one isn't retryable.
    if status == Some(429) && kind == ApiErrorKind::InvalidRequest {
        kind = ApiErrorKind::RateLimited;
    }
    let mut e = ApiError::new(ProviderId::OpenAi, kind, message);
    e.status = status;
    e.code = Some(if code.is_empty() { kind_name } else { code }).filter(|c| !c.is_empty());
    if kind == ApiErrorKind::RateLimited {
        // Per-minute limits are worth a short wait even when OpenAI doesn't say how long.
        e.retry_after = Some(Duration::from_secs(2));
    }
    e
}

/// An output item being streamed.
#[derive(Debug)]
enum Block {
    Message,
    Call,
    Reasoning,
    Other,
}

/// Turns one response's SSE events into neutral events.
pub(crate) struct Decoder {
    blocks: HashMap<usize, Block>,
    usage: Usage,
    warnings: Vec<Warning>,
    tool_calls: bool,
    finished: bool,
}

impl Decoder {
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Decoder { blocks: HashMap::new(), usage: Usage::default(), warnings, tool_calls: false, finished: false }
    }

    pub(crate) fn finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn decode(&mut self, sse: &SseEvent) -> Result<Vec<Event>> {
        if sse.data.trim() == "[DONE]" {
            return Ok(vec![]);
        }
        let data: Value = serde_json::from_str(&sse.data).map_err(|e| Error::Decode { provider: ProviderId::OpenAi, message: format!("{} event: {e}", sse.event.as_deref().unwrap_or("unnamed")) })?;
        let index = || data.get("output_index").and_then(Value::as_u64).map(|i| i as usize).ok_or_else(|| stream_error("item event without an output_index".into()));
        let delta = || data.get("delta").and_then(Value::as_str).unwrap_or_default().to_owned();
        Ok(match data.get("type").and_then(Value::as_str).unwrap_or_default() {
            "response.created" => {
                let response = data.get("response").unwrap_or(&Value::Null);
                let id = response.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
                let model = response.get("model").and_then(Value::as_str).unwrap_or_default().to_owned();
                vec![Event::Start { id, model: ModelId::new(ProviderId::OpenAi, model) }]
            }
            "response.output_item.added" => {
                let index = index()?;
                let item = data.get("item").cloned().unwrap_or_default();
                let field = |key: &str| item.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
                let (block, part) = match field("type").as_str() {
                    "message" => (Block::Message, Part::text("")),
                    "function_call" => (Block::Call, Part::ToolCall(ToolCall { id: field("call_id"), name: field("name"), arguments: json!({}), opaque: None })),
                    "reasoning" => (Block::Reasoning, Part::Reasoning(Reasoning { text: None, opaque: None })),
                    _ => (Block::Other, Part::Opaque(Opaque::new(ProviderId::OpenAi, item.clone()))),
                };
                self.blocks.insert(index, block);
                vec![Event::PartStart { index, part }]
            }
            "response.output_text.delta" => {
                let text = delta();
                if text.is_empty() { vec![] } else { vec![Event::TextDelta { index: index()?, text }] }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let text = delta();
                if text.is_empty() { vec![] } else { vec![Event::ReasoningDelta { index: index()?, text }] }
            }
            "response.reasoning_summary_part.added" if data.get("summary_index").and_then(Value::as_u64).is_some_and(|i| i > 0) => vec![Event::ReasoningDelta { index: index()?, text: "\n\n".into() }],
            "response.function_call_arguments.delta" => {
                let arguments = delta();
                if arguments.is_empty() { vec![] } else { vec![Event::ToolCallDelta { index: index()?, arguments }] }
            }
            "response.output_item.done" => {
                let index = index()?;
                self.blocks.remove(&index);
                // The finished item is whole: it is the source of truth for the part.
                let part = finish(data.get("item").cloned().unwrap_or_default())?;
                if matches!(part, Part::ToolCall(_)) {
                    self.tool_calls = true;
                }
                vec![Event::PartStop { index, part }]
            }
            kind @ ("response.completed" | "response.incomplete") => {
                let response = data.get("response").unwrap_or(&Value::Null);
                if let Some(usage) = response.get("usage").filter(|u| !u.is_null()) {
                    self.usage = usage_from(usage);
                }
                let stop_reason = if kind == "response.completed" {
                    if self.tool_calls { StopReason::ToolCalls } else { StopReason::Complete }
                } else {
                    match response.get("incomplete_details").and_then(|d| d.get("reason")).and_then(Value::as_str).unwrap_or_default() {
                        "max_output_tokens" => StopReason::Length,
                        "content_filter" => StopReason::Refused(Some("content_filter".into())),
                        other => StopReason::Other(other.to_owned()),
                    }
                };
                self.finished = true;
                vec![Event::Finish { stop_reason, usage: self.usage.clone(), warnings: std::mem::take(&mut self.warnings) }]
            }
            "response.failed" => {
                let error = data.get("response").and_then(|r| r.get("error")).cloned().unwrap_or_else(|| json!({ "message": "the response failed" }));
                return Err(api_error(&error, None).into());
            }
            "error" => return Err(api_error(&data, None).into()),
            other => {
                tracing::trace!(event = other, "skipping OpenAI stream event");
                vec![]
            }
        })
    }
}

/// OpenAI usage → neutral usage. OpenAI already counts cached input in `input_tokens` and reasoning in `output_tokens`.
fn usage_from(usage: &Value) -> Usage {
    let n = |path: &[&str]| path.iter().try_fold(usage, |v, key| v.get(key)).and_then(Value::as_u64);
    let mut u = Usage::default();
    u.input_tokens = n(&["input_tokens"]).unwrap_or(0);
    u.output_tokens = n(&["output_tokens"]).unwrap_or(0);
    u.reasoning_tokens = n(&["output_tokens_details", "reasoning_tokens"]);
    u.cache_read_tokens = n(&["input_tokens_details", "cached_tokens"]);
    u.cache_write_tokens = n(&["input_tokens_details", "cache_write_tokens"]);
    u
}

fn stream_error(message: String) -> Error {
    Error::Stream { provider: ProviderId::OpenAi, message }
}

/// A finished output item as a part, keeping the item itself so it goes back exactly (ids, `phase`, encrypted reasoning).
fn finish(item: Value) -> Result<Part> {
    let field = |key: &str| item.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
    let openai = |data: Value| Some(Opaque::new(ProviderId::OpenAi, data));
    Ok(match field("type").as_str() {
        "message" => {
            let content = item.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
            let text: String = content.iter().filter(|c| c.get("type").and_then(Value::as_str) == Some("output_text")).filter_map(|c| c.get("text").and_then(Value::as_str)).collect();
            let refusal: String = content.iter().filter_map(|c| c.get("refusal").and_then(Value::as_str)).collect();
            if text.is_empty() && !refusal.is_empty() { Part::Refusal { text: refusal } } else { Part::Text(Text { text, opaque: openai(item) }) }
        }
        "function_call" => {
            let raw = field("arguments");
            let arguments = if raw.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&raw).map_err(|e| Error::Decode { provider: ProviderId::OpenAi, message: format!("tool call arguments aren't JSON: {e}") })?
            };
            Part::ToolCall(ToolCall { id: field("call_id"), name: field("name"), arguments, opaque: openai(item) })
        }
        "reasoning" => {
            let pieces: Vec<&str> = ["summary", "content"]
                .iter()
                .filter_map(|key| item.get(*key).and_then(Value::as_array))
                .flatten()
                .filter_map(|s| s.get("text").and_then(Value::as_str))
                .filter(|s| !s.is_empty())
                .collect();
            let text = (!pieces.is_empty()).then(|| pieces.join("\n\n"));
            Part::Reasoning(Reasoning { text, opaque: openai(item) })
        }
        _ => Part::Opaque(Opaque::new(ProviderId::OpenAi, item)),
    })
}

/// The neutral event stream for a streaming Responses API response.
pub(crate) fn events(response: genai_core::http::reqwest::Response, warnings: Vec<Warning>) -> EventStream {
    let sse = Box::pin(sse_events(ProviderId::OpenAi, response));
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
                        return Some((Err(stream_error("the stream ended before response.completed".into())), (sse, decoder, pending, done)));
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

    fn sse(events: &[Value]) -> String {
        events.iter().map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap_or_default())).collect()
    }

    #[test]
    fn decodes_a_recorded_stream() {
        let r = response(include_str!("../tests/fixtures/text.sse")).ok();
        assert_eq!(r.as_ref().map(|r| r.text()).as_deref(), Some("Hi there, friend."));
        assert_eq!(r.as_ref().map(|r| r.stop_reason.clone()), Some(StopReason::Complete));
        assert_eq!(r.as_ref().map(|r| r.model.to_string()).as_deref(), Some("openai/gpt-6-luna"));
        assert!(r.as_ref().is_some_and(|r| r.id.starts_with("resp_") && r.usage.input_tokens == 19));
        // The message item comes back whole, `phase` included, to send on the next turn.
        let item = r.and_then(|r| r.message.parts.into_iter().find_map(|p| if let Part::Text(t) = p { t.opaque } else { None }));
        assert!(item.is_some_and(|o| o.data["phase"] == "final_answer" && o.data["id"].as_str().is_some_and(|id| id.starts_with("msg_"))));
    }

    #[test]
    fn decodes_reasoning_and_tool_calls() {
        let reasoning = json!({ "type": "reasoning", "id": "rs_1", "summary": [{ "type": "summary_text", "text": "Need to add." }], "encrypted_content": "enc" });
        let call = json!({ "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "add", "arguments": "{\"a\":2,\"b\":3}", "status": "completed" });
        let fixture = sse(&[
            json!({ "type": "response.created", "response": { "id": "resp_1", "model": "gpt-6-luna" } }),
            json!({ "type": "response.output_item.added", "output_index": 0, "item": { "type": "reasoning", "id": "rs_1", "summary": [] } }),
            json!({ "type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "Need to add." }),
            json!({ "type": "response.output_item.done", "output_index": 0, "item": reasoning }),
            json!({ "type": "response.output_item.added", "output_index": 1, "item": { "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "add", "arguments": "" } }),
            json!({ "type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"a\":2," }),
            json!({ "type": "response.function_call_arguments.delta", "output_index": 1, "delta": "\"b\":3}" }),
            json!({ "type": "response.output_item.done", "output_index": 1, "item": call }),
            json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "input_tokens": 50, "input_tokens_details": { "cached_tokens": 10 }, "output_tokens": 30, "output_tokens_details": { "reasoning_tokens": 20 } } } }),
        ]);
        let events = decode(&fixture).unwrap_or_default();
        assert!(events.iter().any(|e| matches!(e, Event::ReasoningDelta { index: 0, .. })));
        assert_eq!(events.iter().filter(|e| matches!(e, Event::ToolCallDelta { index: 1, .. })).count(), 2);
        let r = response(&fixture).ok();
        assert_eq!(r.as_ref().map(|r| r.stop_reason.clone()), Some(StopReason::ToolCalls));
        assert_eq!(r.as_ref().and_then(|r| r.reasoning()).as_deref(), Some("Need to add."));
        let tool = r.as_ref().and_then(|r| r.tool_calls().next().cloned());
        assert_eq!(tool.as_ref().map(|c| (c.id.clone(), c.arguments.clone())), Some(("call_1".into(), json!({ "a": 2, "b": 3 }))));
        assert_eq!(tool.and_then(|c| c.opaque).map(|o| o.data), Some(call));
        assert_eq!(r.map(|r| (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens, r.usage.cache_read_tokens)), Some((50, 30, Some(20), Some(10))));
    }

    #[test]
    fn reports_incomplete_and_failed_responses() {
        let incomplete = sse(&[json!({ "type": "response.incomplete", "response": { "status": "incomplete", "incomplete_details": { "reason": "max_output_tokens" } } })]);
        assert_eq!(response(&incomplete).ok().map(|r| r.stop_reason), Some(StopReason::Length));
        let failed = sse(&[json!({ "type": "response.failed", "response": { "status": "failed", "error": { "code": "server_error", "message": "The model failed." } } })]);
        assert!(matches!(decode(&failed), Err(Error::Api(e)) if e.kind == ApiErrorKind::Server && e.is_retryable()));
        let error = sse(&[json!({ "type": "error", "code": "rate_limit_exceeded", "message": "slow", "param": null })]);
        assert!(matches!(decode(&error), Err(Error::Api(e)) if e.kind == ApiErrorKind::RateLimited));
    }

    #[test]
    fn maps_error_codes() {
        let quota = api_error(&json!({ "message": "out of credit", "type": "insufficient_quota", "code": "insufficient_quota" }), Some(429));
        assert_eq!(quota.kind, ApiErrorKind::Billing);
        assert!(!quota.is_retryable());
        let missing = api_error(&json!({ "message": "no", "type": "invalid_request_error", "code": "model_not_found" }), Some(404));
        assert_eq!((missing.kind, missing.code.as_deref()), (ApiErrorKind::NotFound, Some("model_not_found")));
        let long = api_error(&json!({ "message": "too long", "type": "invalid_request_error", "code": "context_length_exceeded" }), Some(400));
        assert_eq!(long.kind, ApiErrorKind::ContextTooLong);
        assert!(api_error(&json!({ "message": "busy", "type": "service_unavailable_error", "code": "server_is_overloaded" }), Some(503)).is_retryable());
    }
}
