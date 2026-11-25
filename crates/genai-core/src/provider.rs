//! Who answers a request, and the trait each of them implements.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, Result};
use crate::model::ModelInfo;
use crate::request::Request;
use crate::response::Response;
use crate::stream::{Accumulator, EventStream};

/// A boxed, sendable future. The provider trait returns these so it can be used as `dyn Provider`.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A model provider, by name. Written as the lowercase strings `anthropic`, `openai` and `google`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ProviderId {
    /// Anthropic (Claude).
    Anthropic,
    /// OpenAI (GPT).
    OpenAi,
    /// Google (Gemini).
    Google,
    /// Any other provider, by its lowercase name.
    Other(String),
}

impl ProviderId {
    /// The name used in model strings such as `anthropic/claude-opus-5-5`.
    pub fn as_str(&self) -> &str {
        match self {
            ProviderId::Anthropic => "anthropic",
            ProviderId::OpenAi => "openai",
            ProviderId::Google => "google",
            ProviderId::Other(name) => name,
        }
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProviderId {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "anthropic" => ProviderId::Anthropic,
            "openai" => ProviderId::OpenAi,
            "google" => ProviderId::Google,
            other => ProviderId::Other(other.to_owned()),
        })
    }
}

impl From<&str> for ProviderId {
    fn from(s: &str) -> Self {
        match s.parse() {
            Ok(id) => id,
            Err(never) => match never {},
        }
    }
}

impl Serialize for ProviderId {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ProviderId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(ProviderId::from(s.as_str()))
    }
}

/// One model provider. Implement this to add a provider; apps use it through `genai::Client`.
///
/// The trait is object-safe, so a client can hold `Arc<dyn Provider>`s and new providers can be plugged in at runtime.
pub trait Provider: Send + Sync + 'static {
    /// Which provider this is.
    fn id(&self) -> ProviderId;

    /// Sends the request and streams the response back as [`Event`](crate::Event)s.
    ///
    /// The future resolves once the provider has accepted the request (HTTP 200); errors after that arrive in the stream.
    /// Dropping the stream cancels the request.
    fn stream(&self, request: Request) -> BoxFuture<'_, Result<EventStream>>;

    /// Sends the request and waits for the whole response. Defaults to [`stream`](Provider::stream) folded through an
    /// [`Accumulator`], which also keeps long outputs from hitting idle-connection timeouts.
    fn generate(&self, request: Request) -> BoxFuture<'_, Result<Response>> {
        Box::pin(async move { Accumulator::collect(self.stream(request).await?).await })
    }

    /// The models this provider offers, merged from its live models list and a built-in catalog.
    fn models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>>;

    /// Counts the input tokens the request would use, without running it. Defaults to [`Error::Unsupported`].
    fn count_tokens<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<u64>> {
        let _ = request;
        let provider = self.id();
        Box::pin(async move { Err(Error::Unsupported { provider, feature: "token counting".into() }) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prints_names() {
        assert_eq!(ProviderId::from("Anthropic"), ProviderId::Anthropic);
        assert_eq!(ProviderId::from("openai").as_str(), "openai");
        assert_eq!(ProviderId::from("xai"), ProviderId::Other("xai".into()));
        assert_eq!(serde_json::to_string(&ProviderId::Google).ok().as_deref(), Some("\"google\""));
    }
}
