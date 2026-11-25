//! The OpenAI (GPT) provider for [genai](https://github.com/tensorai-co/genai), over the
//! [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create). Plain HTTP; no OpenAI
//! SDK.
//!
//! Requests are stateless: `store` is `false`, the whole conversation goes with each request, and reasoning comes back
//! encrypted (`reasoning.encrypted_content`) so it can be sent on the next turn without OpenAI keeping anything.
//!
//! ```no_run
//! use genai_core::{Effort, Message, Provider, Request};
//! use genai_openai::OpenAi;
//!
//! # async fn run() -> genai_core::Result<()> {
//! let openai = OpenAi::new(std::env::var("OPENAI_API_KEY").unwrap_or_default())?;
//! let request = Request::new("openai/gpt-6-luna").message(Message::user("Say hi.")).effort(Effort::Low).max_output_tokens(256);
//! println!("{}", openai.generate(request).await?.text());
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

pub use options::{OpenAiOptions, ServiceTier, Verbosity};

use crate::models::Spec;

/// Where the API is.
pub const BASE_URL: &str = "https://api.openai.com/v1";

/// The OpenAI provider. Cheap to clone.
#[derive(Clone)]
pub struct OpenAi {
    inner: Arc<Inner>,
}

struct Inner {
    http: Http,
    key: SecretString,
    base_url: String,
    organization: Option<String>,
    project: Option<String>,
    live: RwLock<HashMap<String, Spec>>,
    catalog: Vec<Spec>,
}

impl std::fmt::Debug for OpenAi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAi").field("base_url", &self.inner.base_url).field("organization", &self.inner.organization).field("project", &self.inner.project).finish_non_exhaustive()
    }
}

impl OpenAi {
    /// A provider using `key`, with default settings.
    pub fn new(key: impl Into<SecretString>) -> Result<Self> {
        OpenAi::builder(key).build()
    }

    /// A provider using the key in `OPENAI_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY").map_err(|_| Error::NotConfigured(ProviderId::OpenAi))?;
        OpenAi::new(key)
    }

    /// A builder for a provider using `key`.
    pub fn builder(key: impl Into<SecretString>) -> OpenAiBuilder {
        OpenAiBuilder { key: key.into(), base_url: BASE_URL.into(), organization: None, project: None, http: None }
    }

    /// The built-in model catalog, without a network call.
    pub fn catalog(&self) -> Vec<ModelInfo> {
        self.inner.catalog.iter().map(Spec::info).collect()
    }

    fn spec(&self, model: &str) -> Option<Spec> {
        let live = self.inner.live.read().ok().and_then(|live| live.get(model).cloned());
        live.or_else(|| models::find(&self.inner.catalog, model).cloned())
    }

    fn headers(&self) -> Result<HeaderMap> {
        let bad = |what: &str| Error::InvalidInput(format!("the OpenAI {what} isn't a valid header value"));
        let mut headers = HeaderMap::new();
        let mut auth = HeaderValue::from_str(&format!("Bearer {}", self.inner.key.expose_secret())).map_err(|_| bad("key"))?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(org) = &self.inner.organization {
            headers.insert(HeaderName::from_static("openai-organization"), HeaderValue::from_str(org).map_err(|_| bad("organization"))?);
        }
        if let Some(project) = &self.inner.project {
            headers.insert(HeaderName::from_static("openai-project"), HeaderValue::from_str(project).map_err(|_| bad("project"))?);
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
        self.inner.http.send(&ProviderId::OpenAi, request, http_error).await
    }
}

/// Turns a failed HTTP response into an [`Error`].
fn http_error(status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
    let parsed: Option<Value> = serde_json::from_slice(body).ok();
    let mut error = match parsed.as_ref().and_then(|b| b.get("error")).filter(|e| e.is_object()) {
        Some(error) => stream::api_error(error, Some(status)),
        None => {
            let mut e = ApiError::new(ProviderId::OpenAi, ApiErrorKind::from_status(status), String::from_utf8_lossy(body).chars().take(500).collect::<String>());
            e.status = Some(status);
            e
        }
    };
    error.request_id = headers.get("x-request-id").and_then(|v| v.to_str().ok()).map(str::to_owned);
    if let Some(after) = retry_after(headers) {
        error.retry_after = Some(after);
    }
    error.into()
}

/// Builds an [`OpenAi`] provider.
#[must_use]
pub struct OpenAiBuilder {
    key: SecretString,
    base_url: String,
    organization: Option<String>,
    project: Option<String>,
    http: Option<Http>,
}

impl OpenAiBuilder {
    /// Sends requests somewhere other than `https://api.openai.com/v1`, such as a proxy or another server that speaks
    /// the Responses API.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// The organization to bill (sent as `OpenAI-Organization`), for keys that belong to several.
    pub fn organization(mut self, id: impl Into<String>) -> Self {
        self.organization = Some(id.into());
        self
    }

    /// The project to bill (sent as `OpenAI-Project`).
    pub fn project(mut self, id: impl Into<String>) -> Self {
        self.project = Some(id.into());
        self
    }

    /// Shares an HTTP client, such as the one `genai::Client` gives every provider.
    pub fn http(mut self, http: Http) -> Self {
        self.http = Some(http);
        self
    }

    /// The provider.
    pub fn build(self) -> Result<OpenAi> {
        let http = match self.http {
            Some(http) => http,
            None => Http::new(&HttpConfig::default())?,
        };
        Ok(OpenAi {
            inner: Arc::new(Inner {
                http,
                key: self.key,
                base_url: self.base_url,
                organization: self.organization,
                project: self.project,
                live: RwLock::new(HashMap::new()),
                catalog: models::catalog(),
            }),
        })
    }
}

impl Provider for OpenAi {
    fn id(&self) -> ProviderId {
        ProviderId::OpenAi
    }

    fn stream(&self, request: Request) -> BoxFuture<'_, Result<EventStream>> {
        Box::pin(async move {
            let spec = self.spec(request.model.name());
            let converted = convert::request_body(&request, spec.as_ref(), true)?;
            let mut headers = self.headers()?;
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
            tracing::debug!(model = %request.model, "openai responses request");
            let response = self.send(reqwest::Method::POST, "/responses", headers, Some(&converted.body)).await?;
            Ok(stream::events(response, converted.warnings))
        })
    }

    fn models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>> {
        Box::pin(async move {
            let response = self.send(reqwest::Method::GET, "/models", self.headers()?, None).await?;
            let page: Value = response.json().await.map_err(|e| Error::Decode { provider: ProviderId::OpenAi, message: e.to_string() })?;
            let mut specs: Vec<Spec> = page.get("data").and_then(Value::as_array).into_iter().flatten().filter_map(|m| Spec::from_live(m, &self.inner.catalog)).collect();
            specs.sort_by(|a, b| a.id.cmp(&b.id));
            if let Ok(mut live) = self.inner.live.write() {
                *live = specs.iter().map(|s| (s.id.clone(), s.clone())).collect();
            }
            Ok(specs.iter().map(Spec::info).collect())
        })
    }
}
