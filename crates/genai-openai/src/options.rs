//! Settings only OpenAI understands.

use genai_core::{ProviderId, ProviderOptions};
use serde::{Deserialize, Serialize};

/// OpenAI-only request settings. Attach with [`Request::options`](genai_core::Request::options); other providers ignore
/// them.
///
/// ```
/// use genai_core::{Message, Request};
/// use genai_openai::{OpenAiOptions, ServiceTier, Verbosity};
///
/// let request = Request::new("openai/gpt-6.2-sol").message(Message::user("Hi")).options(OpenAiOptions::new().service_tier(ServiceTier::Flex).verbosity(Verbosity::Low));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpenAiOptions {
    /// Which processing tier the request uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// How long the answer should be, separately from how hard the model thinks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
    /// Whether OpenAI keeps the response. genai sends `false` unless this says otherwise; conversations are sent in full
    /// each turn either way, with reasoning encrypted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Groups requests that share a long prefix, which raises prompt-cache hits. A conversation id works well.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

impl OpenAiOptions {
    /// No settings.
    pub fn new() -> Self {
        OpenAiOptions::default()
    }

    /// Sets the service tier.
    pub fn service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Sets the verbosity.
    pub fn verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = Some(verbosity);
        self
    }

    /// Lets OpenAI store the response.
    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Sets the prompt cache key.
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }
}

impl From<OpenAiOptions> for ProviderOptions {
    fn from(options: OpenAiOptions) -> Self {
        ProviderOptions { provider: ProviderId::OpenAi, value: serde_json::to_value(options).unwrap_or_default() }
    }
}

/// Which processing tier a request uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ServiceTier {
    /// The project's default.
    Auto,
    /// Standard processing.
    Default,
    /// Cheaper and slower.
    Flex,
    /// Faster and dearer.
    Priority,
    /// Fast mode.
    Fast,
    /// Ultrafast mode, where offered.
    Ultrafast,
}

/// How long answers are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Verbosity {
    /// Short.
    Low,
    /// The default.
    Medium,
    /// Long.
    High,
}
