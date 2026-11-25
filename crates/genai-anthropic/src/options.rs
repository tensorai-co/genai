//! Settings only Anthropic understands.

use genai_core::{ProviderId, ProviderOptions};
use serde::{Deserialize, Serialize};

/// Anthropic-only request settings. Attach with [`Request::options`](genai_core::Request::options); other providers
/// ignore them.
///
/// ```
/// use genai_anthropic::{AnthropicOptions, Speed};
/// use genai_core::{Message, Request};
///
/// let request = Request::new("anthropic/claude-opus-5-5")
///     .message(Message::user("Hi"))
///     .options(AnthropicOptions::new().speed(Speed::Fast).inference_geo("us"));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AnthropicOptions {
    /// Inference speed. `Fast` trades price for output speed on the models that offer it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<Speed>,
    /// Whether the request may use priority capacity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Where inference runs, such as `us`. Unset uses the workspace default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// How long the prompt cache lives. Unset is five minutes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_ttl: Option<CacheTtl>,
    /// Beta features to turn on for this request, sent as `anthropic-beta`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub betas: Vec<String>,
}

impl AnthropicOptions {
    /// No settings.
    pub fn new() -> Self {
        AnthropicOptions::default()
    }

    /// Sets the speed.
    pub fn speed(mut self, speed: Speed) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Sets the service tier.
    pub fn service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Sets where inference runs.
    pub fn inference_geo(mut self, geo: impl Into<String>) -> Self {
        self.inference_geo = Some(geo.into());
        self
    }

    /// Sets the prompt cache's lifetime.
    pub fn cache_ttl(mut self, ttl: CacheTtl) -> Self {
        self.cache_ttl = Some(ttl);
        self
    }

    /// Turns on a beta feature.
    pub fn beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }
}

impl From<AnthropicOptions> for ProviderOptions {
    fn from(options: AnthropicOptions) -> Self {
        ProviderOptions { provider: ProviderId::Anthropic, value: serde_json::to_value(options).unwrap_or_default() }
    }
}

/// Inference speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Speed {
    /// Normal speed.
    Standard,
    /// Faster output, at a higher price, on the models that offer it.
    Fast,
}

/// Which capacity a request may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ServiceTier {
    /// Priority capacity when the organization has it, else standard.
    Auto,
    /// Standard capacity only.
    StandardOnly,
}

/// How long a prompt cache entry lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CacheTtl {
    /// Five minutes, the default.
    #[serde(rename = "5m")]
    FiveMinutes,
    /// One hour, at a higher write price.
    #[serde(rename = "1h")]
    OneHour,
}

impl CacheTtl {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            CacheTtl::FiveMinutes => "5m",
            CacheTtl::OneHour => "1h",
        }
    }
}
