//! The Anthropic (Claude) provider for [genai](https://github.com/tensorai-co/genai), over the
//! [Messages API](https://platform.claude.com/docs/en/api/messages/create) at API version `2026-10-21`. Plain HTTP; no
//! Anthropic SDK.
//!
//! ```no_run
//! use genai_anthropic::Anthropic;
//! use genai_core::{Effort, Message, Provider, Request};
//!
//! # async fn run() -> genai_core::Result<()> {
//! let claude = Anthropic::new(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default())?;
//! let request = Request::new("anthropic/claude-haiku-5-5").message(Message::user("Say hi.")).effort(Effort::Low).max_output_tokens(256);
//! let response = claude.generate(request).await?;
//! println!("{}", response.text());
//! # Ok(()) }
//! ```

mod convert;
mod models;
mod options;
mod stream;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use genai_core::http::reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use genai_core::http::{Http, HttpConfig, reqwest, retry_after};
use genai_core::{ApiError, ApiErrorKind, BoxFuture, Error, EventStream, ModelInfo, Provider, ProviderId, Request, Result};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;

pub use options::{AnthropicOptions, CacheTtl, ServiceTier, Speed};

use crate::models::Spec;

/// The API version this crate speaks.
pub const API_VERSION: &str = "2026-10-21";
/// Where the API is.
pub const BASE_URL: &str = "https://api.anthropic.com";

/// The Anthropic provider. Cheap to clone.
#[derive(Clone)]
pub struct Anthropic {
    inner: Arc<Inner>,
}

struct Inner {
    http: Http,
    key: SecretString,
    base_url: String,
    version: String,
    workspace_id: Option<String>,
    betas: Vec<String>,
    /// The live models list, once [`Provider::models`] has fetched it.
    live: RwLock<HashMap<String, Spec>>,
    catalog: Vec<Spec>,
}

impl std::fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Anthropic").field("base_url", &self.inner.base_url).field("version", &self.inner.version).field("workspace_id", &self.inner.workspace_id).finish_non_exhaustive()
    }
}

impl Anthropic {
    /// A provider using `key`, with default settings.
    pub fn new(key: impl Into<SecretString>) -> Result<Self> {
        Anthropic::builder(key).build()
    }

    /// A provider using the key in `ANTHROPIC_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| Error::NotConfigured(ProviderId::Anthropic))?;
        Anthropic::new(key)
    }

    /// A builder for a provider using `key`.
    pub fn builder(key: impl Into<SecretString>) -> AnthropicBuilder {
        AnthropicBuilder { key: key.into(), base_url: BASE_URL.into(), version: API_VERSION.into(), workspace_id: None, betas: Vec::new(), http: None }
    }

    /// The built-in model catalog, without a network call.
    pub fn catalog(&self) -> Vec<ModelInfo> {
        self.inner.catalog.iter().map(Spec::info).collect()
    }

    /// What genai knows about `model`: from the live list if it has been fetched, else the catalog.
    fn spec(&self, model: &str) -> Option<Spec> {
        let live = self.inner.live.read().ok().and_then(|live| models::find(live.values(), model).cloned());
        live.or_else(|| models::find(&self.inner.catalog, model).cloned())
    }

    fn headers(&self, betas: &[String]) -> Result<HeaderMap> {
        let bad = |what: &str| Error::InvalidInput(format!("the Anthropic {what} isn't a valid header value"));
        let mut headers = HeaderMap::new();
        let mut auth = HeaderValue::from_str(&format!("Bearer {}", self.inner.key.expose_secret())).map_err(|_| bad("key"))?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
        headers.insert(HeaderName::from_static("anthropic-version"), HeaderValue::from_str(&self.inner.version).map_err(|_| bad("version"))?);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(workspace) = &self.inner.workspace_id {
            headers.insert(HeaderName::from_static("anthropic-workspace-id"), HeaderValue::from_str(workspace).map_err(|_| bad("workspace id"))?);
        }
        let betas: Vec<&str> = self.inner.betas.iter().chain(betas).map(String::as_str).collect();
        if !betas.is_empty() {
            headers.insert(HeaderName::from_static("anthropic-beta"), HeaderValue::from_str(&betas.join(",")).map_err(|_| bad("beta list"))?);
        }
        Ok(headers)
    }

    async fn send(&self, method: reqwest::Method, path: &str, headers: HeaderMap, body: Option<&Value>) -> Result<reqwest::Response> {
        let url = format!("{}{path}", self.inner.base_url.trim_end_matches('/'));
        let mut builder = self.inner.http.client().request(method, url).headers(headers);
        if let Some(body) = body {
            builder = builder.body(serde_json::to_vec(body).map_err(|e| Error::InvalidInput(e.to_string()))?);
        }
        let request = builder.build().map_err(|e| Error::InvalidInput(format!("couldn't build the request: {e}")))?;
        self.inner.http.send(&ProviderId::Anthropic, request, api_error).await
    }

    async fn json(&self, response: reqwest::Response) -> Result<Value> {
        response.json().await.map_err(|e| Error::Decode { provider: ProviderId::Anthropic, message: e.to_string() })
    }
}

/// Turns a failed HTTP response into an [`Error`].
fn api_error(status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
    #[derive(serde::Deserialize)]
    struct Body {
        error: stream::WireError,
        request_id: Option<String>,
    }
    let mut error = match serde_json::from_slice::<Body>(body) {
        Ok(parsed) => {
            let mut e = parsed.error.into_api_error(Some(status));
            e.request_id = parsed.request_id;
            e
        }
        Err(_) => {
            let mut e = ApiError::new(ProviderId::Anthropic, ApiErrorKind::from_status(status), String::from_utf8_lossy(body).chars().take(500).collect::<String>());
            e.status = Some(status);
            e
        }
    };
    if error.request_id.is_none() {
        error.request_id = headers.get("request-id").and_then(|v| v.to_str().ok()).map(str::to_owned);
    }
    error.retry_after = retry_after(headers);
    error.into()
}

/// Builds an [`Anthropic`] provider.
#[must_use]
pub struct AnthropicBuilder {
    key: SecretString,
    base_url: String,
    version: String,
    workspace_id: Option<String>,
    betas: Vec<String>,
    http: Option<Http>,
}

impl AnthropicBuilder {
    /// Sends requests somewhere other than `https://api.anthropic.com`, such as a proxy.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Overrides the `anthropic-version` header. The converter is written for [`API_VERSION`].
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// The workspace to run in, for keys that aren't scoped to one (sent as `anthropic-workspace-id`).
    pub fn workspace_id(mut self, id: impl Into<String>) -> Self {
        self.workspace_id = Some(id.into());
        self
    }

    /// Turns on a beta feature for every request (sent in `anthropic-beta`).
    pub fn beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    /// Shares an HTTP client, such as the one `genai::Client` gives every provider.
    pub fn http(mut self, http: Http) -> Self {
        self.http = Some(http);
        self
    }

    /// The provider.
    pub fn build(self) -> Result<Anthropic> {
        let http = match self.http {
            Some(http) => http,
            None => Http::new(&HttpConfig::default())?,
        };
        Ok(Anthropic {
            inner: Arc::new(Inner {
                http,
                key: self.key,
                base_url: self.base_url,
                version: self.version,
                workspace_id: self.workspace_id,
                betas: self.betas,
                live: RwLock::new(HashMap::new()),
                catalog: models::catalog(),
            }),
        })
    }
}

impl Provider for Anthropic {
    fn id(&self) -> ProviderId {
        ProviderId::Anthropic
    }

    fn stream(&self, request: Request) -> BoxFuture<'_, Result<EventStream>> {
        Box::pin(async move {
            let spec = self.spec(request.model.name());
            let converted = convert::request_body(&request, spec.as_ref(), true)?;
            let mut headers = self.headers(&converted.betas)?;
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
            tracing::debug!(model = %request.model, "anthropic messages request");
            let response = self.send(reqwest::Method::POST, "/v1/messages", headers, Some(&converted.body)).await?;
            Ok(stream::events(response, converted.warnings))
        })
    }

    fn models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>> {
        Box::pin(async move {
            let mut specs = Vec::new();
            let mut after: Option<String> = None;
            loop {
                let path = match &after {
                    Some(id) => format!("/v1/models?limit=1000&after_id={id}"),
                    None => "/v1/models?limit=1000".to_owned(),
                };
                let response = self.send(reqwest::Method::GET, &path, self.headers(&[])?, None).await?;
                let page = self.json(response).await?;
                specs.extend(page.get("data").and_then(Value::as_array).into_iter().flatten().filter_map(Spec::from_live));
                after = page.get("last_id").and_then(Value::as_str).map(str::to_owned);
                if !page.get("has_more").and_then(Value::as_bool).unwrap_or(false) || after.is_none() {
                    break;
                }
            }
            if let Ok(mut live) = self.inner.live.write() {
                *live = specs.iter().map(|s| (s.id.clone(), s.clone())).collect();
            }
            Ok(specs.iter().map(Spec::info).collect())
        })
    }

    fn count_tokens<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<u64>> {
        Box::pin(async move {
            let spec = self.spec(request.model.name());
            let converted = convert::request_body(request, spec.as_ref(), false)?;
            let body = convert::count_tokens_body(converted.body);
            let response = self.send(reqwest::Method::POST, "/v1/messages/count_tokens", self.headers(&converted.betas)?, Some(&body)).await?;
            let counted = self.json(response).await?;
            counted.get("input_tokens").and_then(Value::as_u64).ok_or_else(|| Error::Decode { provider: ProviderId::Anthropic, message: "no input_tokens in the count".into() })
        })
    }
}
