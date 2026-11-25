//! Settings only Google understands.

use std::collections::BTreeMap;

use genai_core::{ProviderId, ProviderOptions};
use serde::{Deserialize, Serialize};

/// Google-only request settings. Attach with [`Request::options`](genai_core::Request::options); other providers ignore
/// them.
///
/// ```
/// use genai_core::{Message, Request};
/// use genai_google::{GoogleOptions, ServiceTier};
///
/// let request = Request::new("google/gemini-4-hydrogen").message(Message::user("Hi")).options(GoogleOptions::new().service_tier(ServiceTier::Flex));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GoogleOptions {
    /// Which capacity the request uses: `Flex` is cheaper and slower, `Priority` faster and dearer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Whether Google keeps the request and response (55 days on the paid tier, 1 day on the free tier). genai sends
    /// `false` unless this says otherwise; conversations are sent in full each turn either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Content-safety thresholds, per harm category.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<SafetySetting>,
    /// Labels for billing reports: lowercase letters, digits, `_` and `-`, at most 63 characters.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
}

impl GoogleOptions {
    /// No settings.
    pub fn new() -> Self {
        GoogleOptions::default()
    }

    /// Sets the service tier.
    pub fn service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Lets Google store the interaction.
    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Adds a safety threshold.
    pub fn safety(mut self, category: impl Into<String>, threshold: SafetyThreshold) -> Self {
        self.safety_settings.push(SafetySetting { category: category.into(), threshold });
        self
    }

    /// Adds a billing label.
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

impl From<GoogleOptions> for ProviderOptions {
    fn from(options: GoogleOptions) -> Self {
        ProviderOptions { provider: ProviderId::Google, value: serde_json::to_value(options).unwrap_or_default() }
    }
}

/// Which capacity a request uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ServiceTier {
    /// Cheaper, with higher latency.
    Flex,
    /// The default.
    Standard,
    /// Faster, at a higher price.
    Priority,
}

/// One content-safety threshold.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetySetting {
    /// The harm category, as Google names it, such as `harassment` or `dangerous_content`.
    pub category: String,
    /// When to block.
    pub threshold: SafetyThreshold,
}

/// When content in a category is blocked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SafetyThreshold {
    /// Block anything with a low or higher chance of harm.
    BlockLowAndAbove,
    /// Block medium or higher.
    BlockMediumAndAbove,
    /// Block only high.
    BlockOnlyHigh,
    /// Never block.
    Off,
}
