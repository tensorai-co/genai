//! Conversations: messages, and the parts they are made of.

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::provider::ProviderId;

/// Who wrote a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    /// The person (or app) talking to the model. Tool results are sent in user messages.
    User,
    /// The model.
    Assistant,
}

/// One turn of a conversation. Serializable, so a conversation can be saved and resumed, reasoning included.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who wrote it.
    pub role: Role,
    /// What it says, in order.
    pub parts: Vec<Part>,
}

impl Message {
    /// A message from `role` made of `parts`.
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Message { role, parts }
    }

    /// A user message with just text.
    pub fn user(text: impl Into<String>) -> Self {
        Message::new(Role::User, vec![Part::text(text)])
    }

    /// An assistant message with just text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::new(Role::Assistant, vec![Part::text(text)])
    }

    /// A user message carrying the results of the tools the model called.
    pub fn tool_results(results: impl IntoIterator<Item = ToolResult>) -> Self {
        Message::new(Role::User, results.into_iter().map(Part::ToolResult).collect())
    }

    /// Adds a part.
    pub fn with(mut self, part: impl Into<Part>) -> Self {
        self.parts.push(part.into());
        self
    }

    /// All the text parts, joined.
    pub fn text(&self) -> String {
        self.parts.iter().filter_map(Part::as_text).collect()
    }
}

/// A piece of a message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Part {
    /// Text.
    Text(Text),
    /// An image, document, audio or video.
    Media(Media),
    /// The model calling one of the request's tools.
    ToolCall(ToolCall),
    /// The result of a tool call, sent back by the app.
    ToolResult(ToolResult),
    /// The model's reasoning: a summary to show, and the provider's data to send back on the next turn.
    Reasoning(Reasoning),
    /// The model declining to answer, in its own words.
    Refusal {
        /// What it said.
        text: String,
    },
    /// Provider data genai doesn't interpret (hosted tool calls and results, for example). Kept so it can be sent back.
    Opaque(Opaque),
}

impl Part {
    /// A text part.
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text(Text::new(text))
    }

    /// The text, if this is a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Part::Text(t) => Some(&t.text),
            _ => None,
        }
    }
}

impl From<Text> for Part {
    fn from(t: Text) -> Self {
        Part::Text(t)
    }
}
impl From<Media> for Part {
    fn from(m: Media) -> Self {
        Part::Media(m)
    }
}
impl From<ToolCall> for Part {
    fn from(c: ToolCall) -> Self {
        Part::ToolCall(c)
    }
}
impl From<ToolResult> for Part {
    fn from(r: ToolResult) -> Self {
        Part::ToolResult(r)
    }
}
impl From<&str> for Part {
    fn from(s: &str) -> Self {
        Part::text(s)
    }
}
impl From<String> for Part {
    fn from(s: String) -> Self {
        Part::text(s)
    }
}

/// Text, and the provider's own form of it when it carries more (citations, for example).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Text {
    /// The text.
    pub text: String,
    /// The provider's original block, sent back instead of `text` to the same provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque: Option<Opaque>,
}

impl Text {
    /// Plain text.
    pub fn new(text: impl Into<String>) -> Self {
        Text { text: text.into(), opaque: None }
    }
}

/// The kind of a [`Media`] part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MediaKind {
    /// A picture.
    Image,
    /// A document: PDF, or plain text.
    Document,
    /// Audio.
    Audio,
    /// Video.
    Video,
}

/// Where a [`Media`] part's bytes are.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MediaSource {
    /// Inline, base64-encoded.
    Base64 {
        /// The base64 data.
        data: String,
    },
    /// At a URL the provider fetches.
    Url {
        /// The URL.
        url: String,
    },
    /// A file already uploaded to the provider.
    File {
        /// The provider's file id.
        id: String,
    },
    /// Plain text, for text documents.
    Text {
        /// The text.
        text: String,
    },
}

/// An image, document, audio or video input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Media {
    /// What kind of media.
    pub kind: MediaKind,
    /// The MIME type, such as `image/png` or `application/pdf`. Required for inline data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Where the bytes are.
    #[serde(flatten)]
    pub source: MediaSource,
    /// A title, for documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl Media {
    /// Media of `kind` from `source`.
    pub fn new(kind: MediaKind, mime_type: Option<String>, source: MediaSource) -> Self {
        Media { kind, mime_type, source, title: None }
    }

    /// An image from raw bytes, such as a PNG file's contents.
    pub fn image(mime_type: impl Into<String>, bytes: impl AsRef<[u8]>) -> Self {
        Media::new(MediaKind::Image, Some(mime_type.into()), MediaSource::Base64 { data: base64::engine::general_purpose::STANDARD.encode(bytes) })
    }

    /// An image at a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Media::new(MediaKind::Image, None, MediaSource::Url { url: url.into() })
    }

    /// A PDF from raw bytes.
    pub fn pdf(bytes: impl AsRef<[u8]>) -> Self {
        Media::new(MediaKind::Document, Some("application/pdf".into()), MediaSource::Base64 { data: base64::engine::general_purpose::STANDARD.encode(bytes) })
    }

    /// A PDF at a URL.
    pub fn pdf_url(url: impl Into<String>) -> Self {
        Media::new(MediaKind::Document, Some("application/pdf".into()), MediaSource::Url { url: url.into() })
    }

    /// A plain-text document.
    pub fn text_document(text: impl Into<String>) -> Self {
        Media::new(MediaKind::Document, Some("text/plain".into()), MediaSource::Text { text: text.into() })
    }

    /// Sets the title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// The model calling a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The call's id; the result must carry it back.
    pub id: String,
    /// Which tool.
    pub name: String,
    /// The arguments, as JSON matching the tool's parameter schema.
    pub arguments: serde_json::Value,
    /// Provider data about the call, sent back with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque: Option<Opaque>,
}

impl ToolCall {
    /// The arguments, decoded into `T`.
    pub fn arguments<T: serde::de::DeserializeOwned>(&self) -> serde_json::Result<T> {
        T::deserialize(&self.arguments)
    }
}

/// The result of a [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The id of the call this answers.
    pub call_id: String,
    /// The name of the tool that was called. Some providers require it.
    pub name: String,
    /// What the tool returned: text, and images where the provider accepts them.
    pub content: Vec<Part>,
    /// Whether the tool failed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ToolResult {
    /// A text result for `call`.
    pub fn new(call: &ToolCall, output: impl Into<String>) -> Self {
        ToolResult { call_id: call.id.clone(), name: call.name.clone(), content: vec![Part::text(output)], is_error: false }
    }

    /// A failed result for `call`, with a message for the model.
    pub fn error(call: &ToolCall, message: impl Into<String>) -> Self {
        ToolResult { is_error: true, ..ToolResult::new(call, message) }
    }

    /// A result for `call` made of any parts.
    pub fn with_parts(call: &ToolCall, content: Vec<Part>) -> Self {
        ToolResult { call_id: call.id.clone(), name: call.name.clone(), content, is_error: false }
    }
}

/// The model's reasoning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reasoning {
    /// The reasoning, or a summary of it, to show. `None` when the provider returned none or it was hidden.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// The provider's data (signature or encrypted reasoning), sent back unchanged on the next turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque: Option<Opaque>,
}

/// Provider data genai keeps and sends back unchanged, but never reads. It is sent only to the provider that made it;
/// sending a conversation to another provider drops it, with a [`Warning`](crate::Warning).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Opaque {
    /// The provider that made it.
    pub provider: ProviderId,
    /// The provider's data, exactly as it sent it.
    pub data: serde_json::Value,
}

impl Opaque {
    /// Data from `provider`.
    pub fn new(provider: ProviderId, data: serde_json::Value) -> Self {
        Opaque { provider, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip_through_json() {
        let call = ToolCall { id: "call_1".into(), name: "add".into(), arguments: serde_json::json!({"a": 1}), opaque: None };
        let conversation = vec![
            Message::user("hi").with(Media::image("image/png", [1u8, 2, 3])),
            Message::new(
                Role::Assistant,
                vec![
                    Part::Reasoning(Reasoning { text: Some("thinking".into()), opaque: Some(Opaque::new(ProviderId::Anthropic, serde_json::json!({"signature": "x"}))) }),
                    Part::ToolCall(call.clone()),
                ],
            ),
            Message::tool_results([ToolResult::new(&call, "2")]),
        ];
        let json = serde_json::to_string(&conversation).ok();
        let back: Option<Vec<Message>> = json.as_deref().and_then(|j| serde_json::from_str(j).ok());
        assert_eq!(back.as_ref(), Some(&conversation));
    }
}
