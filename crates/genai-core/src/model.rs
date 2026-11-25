//! Model names and what each model can do.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::provider::ProviderId;
use crate::request::Effort;

/// A model, as `provider/name`, for example `anthropic/claude-opus-5-5`.
///
/// Parsing never fails: a string with no `/` gets an empty provider and is rejected when sent, with
/// [`Error::UnknownProvider`](crate::Error::UnknownProvider). Only the first `/` splits, so model names may contain more.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId {
    provider: ProviderId,
    name: String,
}

impl ModelId {
    /// The model `name` at `provider`.
    pub fn new(provider: ProviderId, name: impl Into<String>) -> Self {
        ModelId { provider, name: name.into() }
    }

    /// The provider.
    pub fn provider(&self) -> &ProviderId {
        &self.provider
    }

    /// The model's name at its provider, such as `claude-opus-5-5`.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.provider, self.name)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        match s.split_once('/') {
            Some((provider, name)) => ModelId::new(ProviderId::from(provider), name),
            None => ModelId::new(ProviderId::Other(String::new()), s),
        }
    }
}

impl From<String> for ModelId {
    fn from(s: String) -> Self {
        ModelId::from(s.as_str())
    }
}

impl From<&String> for ModelId {
    fn from(s: &String) -> Self {
        ModelId::from(s.as_str())
    }
}

impl Serialize for ModelId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(ModelId::from(String::deserialize(deserializer)?))
    }
}

/// A kind of input or output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Modality {
    /// Text.
    Text,
    /// Images.
    Image,
    /// Documents such as PDFs.
    Document,
    /// Audio.
    Audio,
    /// Video.
    Video,
}

/// Where a model is in its life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelStatus {
    /// Generally available.
    Active,
    /// Available for testing; may change or go away with short notice.
    Preview,
    /// Still served, but scheduled to go away.
    Deprecated {
        /// When it stops being served, as an RFC 3339 date-time, if announced.
        retires_at: Option<String>,
    },
}

/// What a model can do. Comes from the provider's models list when it reports this, otherwise from genai's catalog.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Capabilities {
    /// The kinds of input the model accepts.
    pub input: Vec<Modality>,
    /// The effort levels the model accepts, lowest first. Empty when the model has no effort setting.
    pub efforts: Vec<Effort>,
    /// The effort the model uses when none is given, when known.
    pub default_effort: Option<Effort>,
    /// Whether the model reasons before it answers.
    pub reasoning: bool,
    /// Whether reasoning can be turned off ([`Effort::None`]).
    pub reasoning_can_be_off: bool,
    /// Whether the model accepts tools.
    pub tools: bool,
    /// Whether the model can be forced to call a tool ([`ToolChoice::Required`](crate::ToolChoice::Required) or
    /// [`ToolChoice::Named`](crate::ToolChoice::Named)).
    pub forced_tool_choice: bool,
    /// Whether the model supports schema-constrained JSON output.
    pub structured_output: bool,
}

/// One model, for a model picker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelInfo {
    /// The model.
    pub id: ModelId,
    /// A name for people, such as `Claude Opus 5.5`.
    pub display_name: String,
    /// The line it belongs to, such as `opus`, when the provider groups its models.
    pub family: Option<String>,
    /// The most input tokens the model accepts.
    pub context_window: Option<u32>,
    /// The most output tokens the model can produce in one response.
    pub max_output_tokens: Option<u32>,
    /// What it can do.
    pub capabilities: Capabilities,
    /// Whether it is active, in preview or deprecated.
    pub status: ModelStatus,
}

impl ModelInfo {
    /// A model known only by name, with nothing else filled in.
    pub fn new(id: ModelId, display_name: impl Into<String>) -> Self {
        ModelInfo { id, display_name: display_name.into(), family: None, context_window: None, max_output_tokens: None, capabilities: Capabilities::default(), status: ModelStatus::Active }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_provider_and_name() {
        let id = ModelId::from("anthropic/claude-opus-5-5");
        assert_eq!(id.provider(), &ProviderId::Anthropic);
        assert_eq!(id.name(), "claude-opus-5-5");
        assert_eq!(id.to_string(), "anthropic/claude-opus-5-5");

        let nested = ModelId::from("openrouter/meta/llama-5");
        assert_eq!(nested.provider(), &ProviderId::Other("openrouter".into()));
        assert_eq!(nested.name(), "meta/llama-5");

        let bare = ModelId::from("claude-opus-5-5");
        assert_eq!(bare.provider(), &ProviderId::Other(String::new()));
    }
}
