//! Messages API server-sent events → neutral [`Event`]s.

use std::collections::{HashMap, VecDeque};

use futures_util::StreamExt;
use genai_core::http::{SseEvent, sse_events};
use genai_core::{ApiError, ApiErrorKind, Error, Event, EventStream, ModelId, Opaque, Part, ProviderId, Reasoning, Result, StopReason, Text, ToolCall, Usage, Warning};
use serde::Deserialize;
use serde_json::{Value, json};

/// One SSE event's data.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Wire {
    MessageStart {
        message: WireMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: Value,
    },
    ContentBlockDelta {
        index: usize,
        delta: Value,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: WireDelta,
        #[serde(default)]
        usage: WireUsage,
    },
    MessageStop,
    Ping,
    Error {
        error: WireError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    id: String,
    model: String,
    #[serde(default)]
    usage: WireUsage,
}

#[derive(Debug, Default, Deserialize)]
struct WireDelta {
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    stop_details: Option<Value>,
}

/// Usage as the API reports it. In `message_delta` the counts are cumulative, and any of them may be missing.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct WireUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    output_tokens_details: Option<WireOutputDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct WireOutputDetails {
    thinking_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct WireError {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) message: String,
    #[serde(default)]
    pub(crate) details: Option<Value>,
}

impl WireError {
    pub(crate) fn into_api_error(self, status: Option<u16>) -> ApiError {
        let mut kind = match self.kind.as_str() {
            "invalid_request_error" => ApiErrorKind::InvalidRequest,
            "authentication_error" => ApiErrorKind::Auth,
            "permission_error" => ApiErrorKind::Permission,
            "billing_error" => ApiErrorKind::Billing,
            "not_found_error" => ApiErrorKind::NotFound,
            "request_too_large" => ApiErrorKind::RequestTooLarge,
            "rate_limit_error" => ApiErrorKind::RateLimited,
            "api_error" => ApiErrorKind::Server,
            "timeout_error" => ApiErrorKind::Timeout,
            "overloaded_error" => ApiErrorKind::Overloaded,
            _ => status.map_or(ApiErrorKind::Other, ApiErrorKind::from_status),
        };
        if kind == ApiErrorKind::InvalidRequest && (self.message.contains("prompt is too long") || self.message.contains("context window")) {
            kind = ApiErrorKind::ContextTooLong;
        }
        let code = self.details.as_ref().and_then(|d| d.get("error_code")).and_then(Value::as_str).map_or(self.kind, str::to_owned);
        let mut error = ApiError::new(ProviderId::Anthropic, kind, self.message);
        error.status = status;
        error.code = Some(code);
        error
    }
}

/// A content block being streamed.
#[derive(Debug)]
enum Block {
    Text {
        text: String,
        citations: Vec<Value>,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    ToolUse {
        block: Value,
        json: String,
    },
    /// Anything else (server tool calls and results, redacted thinking...), kept whole.
    Other {
        block: Value,
        json: String,
    },
}

/// Turns one response's SSE events into neutral events.
pub(crate) struct Decoder {
    blocks: HashMap<usize, Block>,
    usage: Usage,
    stop: Option<StopReason>,
    warnings: Vec<Warning>,
    finished: bool,
}

impl Decoder {
    /// A decoder whose [`Event::Finish`] carries `warnings` from building the request.
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Decoder { blocks: HashMap::new(), usage: Usage::default(), stop: None, warnings, finished: false }
    }

    /// The neutral events for one SSE event.
    pub(crate) fn decode(&mut self, sse: &SseEvent) -> Result<Vec<Event>> {
        let wire: Wire =
            serde_json::from_str(&sse.data).map_err(|e| Error::Decode { provider: ProviderId::Anthropic, message: format!("{} event: {e}", sse.event.as_deref().unwrap_or("unnamed")) })?;
        Ok(match wire {
            Wire::MessageStart { message } => {
                self.add_usage(&message.usage);
                vec![Event::Start { id: message.id, model: ModelId::new(ProviderId::Anthropic, message.model) }]
            }
            Wire::ContentBlockStart { index, content_block } => {
                let (block, part) = start(content_block);
                self.blocks.insert(index, block);
                vec![Event::PartStart { index, part }]
            }
            Wire::ContentBlockDelta { index, delta } => self.delta(index, &delta)?,
            Wire::ContentBlockStop { index } => {
                let block = self.blocks.remove(&index).ok_or_else(|| stream_error(format!("content_block_stop for unknown block {index}")))?;
                vec![Event::PartStop { index, part: finish(block)? }]
            }
            Wire::MessageDelta { delta, usage } => {
                self.add_usage(&usage);
                if let Some(reason) = delta.stop_reason {
                    self.stop = Some(stop_reason(&reason, delta.stop_sequence, delta.stop_details.as_ref()));
                }
                vec![Event::Usage(self.usage.clone())]
            }
            Wire::MessageStop => {
                self.finished = true;
                let stop_reason = self.stop.take().unwrap_or_else(|| StopReason::Other("missing".into()));
                vec![Event::Finish { stop_reason, usage: self.usage.clone(), warnings: std::mem::take(&mut self.warnings) }]
            }
            Wire::Ping => vec![],
            Wire::Error { error } => return Err(error.into_api_error(None).into()),
            Wire::Unknown => {
                tracing::trace!(event = ?sse.event, "skipping unknown Anthropic stream event");
                vec![]
            }
        })
    }

    /// Whether `message_stop` has arrived.
    pub(crate) fn finished(&self) -> bool {
        self.finished
    }

    fn delta(&mut self, index: usize, delta: &Value) -> Result<Vec<Event>> {
        let block = self.blocks.get_mut(&index).ok_or_else(|| stream_error(format!("delta for unknown block {index}")))?;
        let str_field = |key: &str| delta.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
        Ok(match (delta.get("type").and_then(Value::as_str).unwrap_or_default(), block) {
            ("text_delta", Block::Text { text, .. }) => {
                let more = str_field("text");
                text.push_str(&more);
                vec![Event::TextDelta { index, text: more }]
            }
            ("citations_delta", Block::Text { citations, .. }) => {
                citations.extend(delta.get("citation").cloned());
                vec![]
            }
            ("thinking_delta", Block::Thinking { thinking, .. }) => {
                let more = str_field("thinking");
                thinking.push_str(&more);
                if more.is_empty() { vec![] } else { vec![Event::ReasoningDelta { index, text: more }] }
            }
            ("signature_delta", Block::Thinking { signature, .. }) => {
                signature.push_str(&str_field("signature"));
                vec![]
            }
            ("input_json_delta", Block::ToolUse { json, .. }) => {
                let more = str_field("partial_json");
                json.push_str(&more);
                if more.is_empty() { vec![] } else { vec![Event::ToolCallDelta { index, arguments: more }] }
            }
            ("input_json_delta", Block::Other { json, .. }) => {
                json.push_str(&str_field("partial_json"));
                vec![]
            }
            (kind, _) => {
                tracing::trace!(kind, index, "skipping unexpected Anthropic delta");
                vec![]
            }
        })
    }

    fn add_usage(&mut self, wire: &WireUsage) {
        let cache_write = wire.cache_creation_input_tokens.or(self.usage.cache_write_tokens);
        let cache_read = wire.cache_read_input_tokens.or(self.usage.cache_read_tokens);
        let uncached =
            wire.input_tokens.unwrap_or_else(|| self.usage.input_tokens.saturating_sub(self.usage.cache_write_tokens.unwrap_or(0)).saturating_sub(self.usage.cache_read_tokens.unwrap_or(0)));
        self.usage.input_tokens = uncached + cache_write.unwrap_or(0) + cache_read.unwrap_or(0);
        self.usage.cache_write_tokens = cache_write;
        self.usage.cache_read_tokens = cache_read;
        if let Some(output) = wire.output_tokens {
            self.usage.output_tokens = output;
        }
        if let Some(thinking) = wire.output_tokens_details.as_ref().and_then(|d| d.thinking_tokens) {
            self.usage.reasoning_tokens = Some(thinking);
        }
    }
}

fn stream_error(message: String) -> Error {
    Error::Stream { provider: ProviderId::Anthropic, message }
}

fn opaque(data: Value) -> Option<genai_core::Opaque> {
    Some(Opaque::new(ProviderId::Anthropic, data))
}

/// The block's state, and the part announced by `PartStart`.
fn start(block: Value) -> (Block, Part) {
    let str_field = |key: &str| block.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
    match block.get("type").and_then(Value::as_str).unwrap_or_default() {
        "text" => {
            let citations = block.get("citations").and_then(Value::as_array).cloned().unwrap_or_default();
            (Block::Text { text: str_field("text"), citations }, Part::text(""))
        }
        "thinking" => (Block::Thinking { thinking: str_field("thinking"), signature: str_field("signature") }, Part::Reasoning(Reasoning { text: None, opaque: None })),
        "tool_use" => {
            let part = Part::ToolCall(ToolCall { id: str_field("id"), name: str_field("name"), arguments: json!({}), opaque: None });
            (Block::ToolUse { block, json: String::new() }, part)
        }
        "redacted_thinking" => (Block::Other { block, json: String::new() }, Part::Reasoning(Reasoning { text: None, opaque: None })),
        _ => (Block::Other { block: block.clone(), json: String::new() }, Part::Opaque(Opaque::new(ProviderId::Anthropic, block))),
    }
}

/// The finished part.
fn finish(block: Block) -> Result<Part> {
    Ok(match block {
        Block::Text { text, citations } => {
            let original = (!citations.is_empty()).then(|| json!({ "type": "text", "text": text, "citations": citations }));
            Part::Text(Text { text, opaque: original.and_then(opaque) })
        }
        Block::Thinking { thinking, signature } => {
            let data = json!({ "type": "thinking", "thinking": thinking, "signature": signature });
            Part::Reasoning(Reasoning { text: (!thinking.is_empty()).then_some(thinking), opaque: opaque(data) })
        }
        Block::ToolUse { mut block, json } => {
            if !json.is_empty() {
                block["input"] = serde_json::from_str(&json).map_err(|e| Error::Decode { provider: ProviderId::Anthropic, message: format!("tool call arguments aren't JSON: {e}") })?;
            }
            let field = |key: &str| block.get(key).and_then(Value::as_str).unwrap_or_default().to_owned();
            let arguments = block.get("input").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
            // Calls made from code execution carry a `caller`; keep the block so it goes back exactly.
            let keep = block.get("caller").is_some_and(|c| c.get("type").and_then(Value::as_str) != Some("direct"));
            Part::ToolCall(ToolCall { id: field("id"), name: field("name"), arguments, opaque: if keep { opaque(block) } else { None } })
        }
        Block::Other { mut block, json } => {
            if !json.is_empty()
                && let Ok(input) = serde_json::from_str::<Value>(&json)
            {
                block["input"] = input;
            }
            if block.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
                Part::Reasoning(Reasoning { text: None, opaque: opaque(block) })
            } else {
                Part::Opaque(Opaque::new(ProviderId::Anthropic, block))
            }
        }
    })
}

fn stop_reason(reason: &str, sequence: Option<String>, details: Option<&Value>) -> StopReason {
    match reason {
        "end_turn" => StopReason::Complete,
        "tool_use" => StopReason::ToolCalls,
        "max_tokens" | "model_context_window_exceeded" => StopReason::Length,
        "stop_sequence" => StopReason::StopSequence(sequence.unwrap_or_default()),
        "pause_turn" => StopReason::Paused,
        "refusal" => {
            let explanation = details.and_then(|d| d.get("explanation")).and_then(Value::as_str).map(str::to_owned);
            let category = details.and_then(|d| d.get("category")).and_then(Value::as_str);
            StopReason::Refused(explanation.or_else(|| category.map(str::to_owned)))
        }
        other => StopReason::Other(other.to_owned()),
    }
}

/// The neutral event stream for a streaming Messages API response.
pub(crate) fn events(response: genai_core::http::reqwest::Response, warnings: Vec<Warning>) -> EventStream {
    let sse = Box::pin(sse_events(ProviderId::Anthropic, response));
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
                        return Some((Err(stream_error("the stream ended before message_stop".into())), (sse, decoder, pending, done)));
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

    /// Runs a recorded stream through the decoder.
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
    fn decodes_text() {
        let r = response(include_str!("../tests/fixtures/text.sse"));
        let r = r.ok();
        assert_eq!(r.as_ref().map(|r| r.text()).as_deref(), Some("Hello!"));
        assert_eq!(r.as_ref().map(|r| r.stop_reason.clone()), Some(StopReason::Complete));
        assert_eq!(r.as_ref().map(|r| r.model.to_string()).as_deref(), Some("anthropic/claude-haiku-5-5"));
        assert_eq!(r.map(|r| (r.usage.input_tokens, r.usage.output_tokens)), Some((25, 15)));
    }

    #[test]
    fn decodes_tool_calls_from_partial_json() {
        let events = decode(include_str!("../tests/fixtures/tool_use.sse")).unwrap_or_default();
        assert!(events.iter().any(|e| matches!(e, Event::ToolCallDelta { index: 1, .. })));
        let r = response(include_str!("../tests/fixtures/tool_use.sse")).ok();
        let call = r.as_ref().and_then(|r| r.tool_calls().next().cloned());
        assert_eq!(call.as_ref().map(|c| c.name.as_str()), Some("get_weather"));
        assert_eq!(call.map(|c| c.arguments), Some(json!({ "location": "San Francisco, CA" })));
        assert_eq!(r.map(|r| r.stop_reason), Some(StopReason::ToolCalls));
    }

    #[test]
    fn keeps_thinking_with_its_signature() {
        let r = response(include_str!("../tests/fixtures/thinking.sse")).ok();
        let reasoning = r.as_ref().and_then(|r| r.message.parts.iter().find_map(|p| if let Part::Reasoning(x) = p { Some(x.clone()) } else { None }));
        assert_eq!(reasoning.as_ref().and_then(|x| x.text.clone()).as_deref(), Some("GCD is 21."));
        assert_eq!(reasoning.and_then(|x| x.opaque).map(|o| o.data), Some(json!({ "type": "thinking", "thinking": "GCD is 21.", "signature": "EqQBsig" })));
        assert_eq!(r.as_ref().map(|r| r.text()).as_deref(), Some("21"));
        assert_eq!(r.map(|r| r.usage.reasoning_tokens), Some(Some(9)));
    }

    #[test]
    fn surfaces_errors_mid_stream() {
        let fixture = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude-haiku-5-5\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
        match decode(fixture) {
            Err(Error::Api(e)) => {
                assert_eq!(e.kind, ApiErrorKind::Overloaded);
                assert!(e.is_retryable());
            }
            other => panic!("expected an overloaded error, got {other:?}"),
        }
    }

    #[test]
    fn skips_unknown_events_and_deltas() {
        let fixture = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude-haiku-5-5\",\"usage\":{\"input_tokens\":1}}}\n\nevent: brand_new\ndata: {\"type\":\"brand_new\",\"x\":1}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"sparkle_delta\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"brand_new_reason\"},\"usage\":{\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let r = response(fixture).ok();
        assert_eq!(r.map(|r| r.stop_reason), Some(StopReason::Other("brand_new_reason".into())));
    }
}
