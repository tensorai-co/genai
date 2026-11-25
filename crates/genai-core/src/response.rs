//! What comes back.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::message::{Message, Part, ToolCall};
use crate::model::ModelId;

/// Why the model stopped.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", content = "detail", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    /// It finished.
    Complete,
    /// It called tools and waits for their results.
    ToolCalls,
    /// It hit the output limit or the context window; the response is cut short.
    Length,
    /// It wrote one of the stop sequences.
    StopSequence(String),
    /// It declined, with the provider's explanation when there is one.
    Refused(Option<String>),
    /// It paused a long turn; send the response back as-is to let it continue.
    Paused,
    /// A reason genai doesn't know yet, as the provider spelled it.
    Other(String),
}

/// Tokens used. `input_tokens` counts all input, cached or not, so it means the same across providers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    /// All input tokens, including those read from or written to the cache.
    pub input_tokens: u64,
    /// All output tokens, including reasoning.
    pub output_tokens: u64,
    /// The part of `output_tokens` spent reasoning, when reported.
    pub reasoning_tokens: Option<u64>,
    /// The part of `input_tokens` read from the cache, when reported.
    pub cache_read_tokens: Option<u64>,
    /// The part of `input_tokens` written to the cache, when reported.
    pub cache_write_tokens: Option<u64>,
}

/// A setting that couldn't be honored exactly, and what was done instead.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Warning {
    /// A short stable code, such as `effort_clamped` or `temperature_dropped`.
    pub code: String,
    /// What happened, for logs.
    pub message: String,
}

impl Warning {
    /// A warning.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Warning { code: code.into(), message: message.into() }
    }
}

/// A finished response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Response {
    /// The provider's id for the response.
    pub id: String,
    /// The model that answered, as the provider reported it.
    pub model: ModelId,
    /// The reply, as an assistant message. Add it to the conversation as-is: it carries what the next turn needs.
    pub message: Message,
    /// Why the model stopped.
    pub stop_reason: StopReason,
    /// Tokens used.
    pub usage: Usage,
    /// Settings that were approximated.
    pub warnings: Vec<Warning>,
}

impl Response {
    /// All the text, joined.
    pub fn text(&self) -> String {
        self.message.text()
    }

    /// The reasoning that came back to show, joined, if any.
    pub fn reasoning(&self) -> Option<String> {
        let parts: Vec<&str> = self.message.parts.iter().filter_map(|p| if let Part::Reasoning(r) = p { r.text.as_deref() } else { None }).collect();
        if parts.is_empty() { None } else { Some(parts.join("\n\n")) }
    }

    /// The tool calls, in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.message.parts.iter().filter_map(|p| if let Part::ToolCall(c) = p { Some(c) } else { None })
    }

    /// The text decoded as JSON, for responses with an [`OutputFormat`](crate::OutputFormat).
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(self.text().trim()).map_err(|e| Error::Decode { provider: self.model.provider().clone(), message: format!("response text isn't the expected JSON: {e}") })
    }
}
