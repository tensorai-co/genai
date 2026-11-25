use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use genai_core::http::{Http, HttpConfig, RetryPolicy};
use genai_core::{Error, EventStream, ModelInfo, Provider, ProviderId, Request, Response, Result};
use secrecy::SecretString;

/// Sends requests to whichever provider each request's model names. Cheap to clone; share one per app.
#[derive(Clone)]
pub struct Client {
    providers: Arc<BTreeMap<ProviderId, Arc<dyn Provider>>>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client").field("providers", &self.providers.keys().collect::<Vec<_>>()).finish()
    }
}

impl Client {
    /// A builder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder { http: HttpConfig::default(), keys: Vec::new(), providers: Vec::new() }
    }

    /// A client for every provider whose key is in the environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and
    /// `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Fails if there are none.
    pub fn from_env() -> Result<Self> {
        let mut builder = Client::builder();
        #[cfg(feature = "anthropic")]
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            builder = builder.anthropic(key);
        }
        #[cfg(feature = "openai")]
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            builder = builder.openai(key);
        }
        #[cfg(feature = "google")]
        if let Ok(key) = std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY")) {
            builder = builder.google(key);
        }
        let client = builder.build()?;
        if client.providers.is_empty() {
            return Err(Error::InvalidInput("no provider keys in the environment (looked for ANTHROPIC_API_KEY, OPENAI_API_KEY and GEMINI_API_KEY)".into()));
        }
        Ok(client)
    }

    /// The provider for `id`.
    pub fn provider(&self, id: &ProviderId) -> Result<&Arc<dyn Provider>> {
        self.providers.get(id).ok_or_else(|| match id {
            ProviderId::Other(name) => Error::UnknownProvider(name.clone()),
            known => Error::NotConfigured(known.clone()),
        })
    }

    /// The configured providers.
    pub fn providers(&self) -> impl Iterator<Item = &ProviderId> {
        self.providers.keys()
    }

    fn route(&self, request: &Request) -> Result<&Arc<dyn Provider>> {
        match request.model.provider() {
            ProviderId::Other(name) if name.is_empty() => Err(Error::UnknownProvider(request.model.name().to_owned())),
            id => self.provider(id),
        }
    }

    /// Sends the request and streams the response.
    pub async fn stream(&self, request: Request) -> Result<EventStream> {
        self.route(&request)?.stream(request).await
    }

    /// Sends the request and waits for the whole response.
    pub async fn generate(&self, request: Request) -> Result<Response> {
        self.route(&request)?.generate(request).await
    }

    /// Counts the request's input tokens without running it.
    pub async fn count_tokens(&self, request: &Request) -> Result<u64> {
        self.route(request)?.count_tokens(request).await
    }

    /// Every configured provider's models, from their live lists.
    pub async fn models(&self) -> Result<Vec<ModelInfo>> {
        let mut all = Vec::new();
        for provider in self.providers.values() {
            all.extend(provider.models().await?);
        }
        Ok(all)
    }
}

/// Builds a [`Client`].
#[must_use]
pub struct ClientBuilder {
    http: HttpConfig,
    keys: Vec<(ProviderId, SecretString)>,
    providers: Vec<Arc<dyn Provider>>,
}

impl ClientBuilder {
    /// Uses Anthropic with `key`.
    #[cfg(feature = "anthropic")]
    pub fn anthropic(mut self, key: impl Into<SecretString>) -> Self {
        self.keys.push((ProviderId::Anthropic, key.into()));
        self
    }

    /// Uses OpenAI with `key`.
    #[cfg(feature = "openai")]
    pub fn openai(mut self, key: impl Into<SecretString>) -> Self {
        self.keys.push((ProviderId::OpenAi, key.into()));
        self
    }

    /// Uses Google (Gemini) with `key`.
    #[cfg(feature = "google")]
    pub fn google(mut self, key: impl Into<SecretString>) -> Self {
        self.keys.push((ProviderId::Google, key.into()));
        self
    }

    /// Adds a provider built elsewhere, such as one with custom settings or one genai doesn't ship. It replaces any
    /// provider with the same id.
    pub fn provider(mut self, provider: impl Provider) -> Self {
        self.providers.push(Arc::new(provider));
        self
    }

    /// How many times to retry failures that may pass (2 by default).
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.http.retry.max_retries = retries;
        self
    }

    /// The whole retry policy.
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.http.retry = policy;
        self
    }

    /// How long to wait for a connection (10 s by default).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.http.connect_timeout = timeout;
        self
    }

    /// How long a response may go silent before it's dropped (5 min by default).
    pub fn read_timeout(mut self, timeout: Duration) -> Self {
        self.http.read_timeout = timeout;
        self
    }

    /// The `User-Agent` header.
    pub fn user_agent(mut self, agent: impl Into<String>) -> Self {
        self.http.user_agent = agent.into();
        self
    }

    /// The client.
    pub fn build(self) -> Result<Client> {
        let http = Http::new(&self.http)?;
        let mut providers: BTreeMap<ProviderId, Arc<dyn Provider>> = BTreeMap::new();
        for (id, key) in self.keys {
            let provider: Arc<dyn Provider> = match id {
                #[cfg(feature = "anthropic")]
                ProviderId::Anthropic => Arc::new(genai_anthropic::Anthropic::builder(key).http(http.clone()).build()?),
                #[cfg(feature = "openai")]
                ProviderId::OpenAi => Arc::new(genai_openai::OpenAi::builder(key).http(http.clone()).build()?),
                #[cfg(feature = "google")]
                ProviderId::Google => Arc::new(genai_google::Google::builder(key).http(http.clone()).build()?),
                other => {
                    let _ = key;
                    return Err(Error::NotConfigured(other));
                }
            };
            providers.insert(id_of(&provider), provider);
        }
        for provider in self.providers {
            providers.insert(id_of(&provider), provider);
        }
        Ok(Client { providers: Arc::new(providers) })
    }
}

fn id_of(provider: &Arc<dyn Provider>) -> ProviderId {
    provider.id()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn routes_by_model_string() {
        let client = Client::builder().build().ok();
        let client = client.as_ref();
        let unknown = client.map(|c| c.route(&Request::new("claude-opus-5-5")).err());
        assert!(matches!(unknown, Some(Some(Error::UnknownProvider(name))) if name == "claude-opus-5-5"));
        let missing = client.map(|c| c.route(&Request::new("anthropic/claude-opus-5-5")).err());
        assert!(matches!(missing, Some(Some(Error::NotConfigured(ProviderId::Anthropic)))));
        let other = client.map(|c| c.route(&Request::new("mistral/large")).err());
        assert!(matches!(other, Some(Some(Error::UnknownProvider(name))) if name == "mistral"));
    }
}
