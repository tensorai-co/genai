//! The Google (Gemini) provider for [genai](https://github.com/tensorai-co/genai), over the
//! [Interactions API](https://aistudio.google.com/docs/interactions-overview) (`/v1beta/interactions`). Plain HTTP; no
//! Google SDK.
//!
//! Requests are stateless: the whole conversation goes with each request and `store` is `false`, so Google keeps nothing
//! (it would otherwise keep interactions for up to 55 days). Reasoning comes back as thought steps with signatures, which
//! genai sends back on the next turn.
//!
//! ```no_run
//! use genai_core::{Effort, Message, Provider, Request};
//! use genai_google::Google;
//!
//! # async fn run() -> genai_core::Result<()> {
//! let gemini = Google::new(std::env::var("GEMINI_API_KEY").unwrap_or_default())?;
//! let request = Request::new("google/gemini-3.5-flash-lite").message(Message::user("Say hi.")).effort(Effort::Low).max_output_tokens(256);
//! println!("{}", gemini.generate(request).await?.text());
//! # Ok(()) }
//! ```

mod convert;
mod models;
mod options;
mod stream;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use genai_core::http::reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use genai_core::http::{Http, HttpConfig, reqwest, retry_after};
use genai_core::{ApiError, ApiErrorKind, BoxFuture, Error, EventStream, ModelInfo, Provider, ProviderId, Request, Result};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;

pub use options::{GoogleOptions, SafetySetting, SafetyThreshold, ServiceTier};

use crate::models::Spec;

/// The Interactions API schema revision this crate speaks, sent as `Api-Revision`.
pub const API_REVISION: &str = "2026-05-20";
/// Where the API is.
pub const BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// The Google provider. Cheap to clone.
#[derive(Clone)]
pub struct Google {
    inner: Arc<Inner>,
}

struct Inner {
    http: Http,
    key: SecretString,
    base_url: String,
    api_version: String,
    revision: String,
    live: RwLock<HashMap<String, Spec>>,
    catalog: Vec<Spec>,
}

impl std::fmt::Debug for Google {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Google").field("base_url", &self.inner.base_url).field("api_version", &self.inner.api_version).finish_non_exhaustive()
    }
}

impl Google {
    /// A provider using `key`, with default settings.
    pub fn new(key: impl Into<SecretString>) -> Result<Self> {
        Google::builder(key).build()
    }

    /// A provider using the key in `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY")).map_err(|_| Error::NotConfigured(ProviderId::Google))?;
        Google::new(key)
    }

    /// A builder for a provider using `key`.
    pub fn builder(key: impl Into<SecretString>) -> GoogleBuilder {
        GoogleBuilder { key: key.into(), base_url: BASE_URL.into(), api_version: "v1beta".into(), revision: API_REVISION.into(), http: None }
    }

    /// The built-in model catalog, without a network call.
    pub fn catalog(&self) -> Vec<ModelInfo> {
        self.inner.catalog.iter().map(Spec::info).collect()
    }

    fn spec(&self, model: &str) -> Option<Spec> {
        let live = self.inner.live.read().ok().and_then(|live| models::find(live.values(), model).cloned());
        live.or_else(|| models::find(&self.inner.catalog, model).cloned())
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        let mut key = HeaderValue::from_str(self.inner.key.expose_secret()).map_err(|_| Error::InvalidInput("the Google key isn't a valid header value".into()))?;
        key.set_sensitive(true);
        headers.insert(HeaderName::from_static("x-goog-api-key"), key);
        headers.insert(HeaderName::from_static("api-revision"), HeaderValue::from_str(&self.inner.revision).map_err(|_| Error::InvalidInput("bad Api-Revision".into()))?);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    async fn send(&self, method: reqwest::Method, path: &str, headers: HeaderMap, body: Option<&Value>) -> Result<reqwest::Response> {
        let url = format!("{}/{}{path}", self.inner.base_url.trim_end_matches('/'), self.inner.api_version);
        let mut builder = self.inner.http.client().request(method, url).headers(headers);
        if let Some(body) = body {
            builder = builder.body(serde_json::to_vec(body).map_err(|e| Error::InvalidInput(e.to_string()))?);
        }
        let request = builder.build().map_err(|e| Error::InvalidInput(format!("couldn't build the request: {e}")))?;
        self.inner.http.send(&ProviderId::Google, request, http_error).await
    }
}

/// Turns a failed HTTP response into an [`Error`].
fn http_error(status: u16, headers: &HeaderMap, body: &[u8]) -> Error {
    let parsed: Option<Value> = serde_json::from_slice(body).ok();
    let mut error = match parsed.as_ref().and_then(|b| b.get("error")) {
        Some(error) => stream::api_error(error, Some(status)),
        None => {
            let mut e = ApiError::new(ProviderId::Google, ApiErrorKind::from_status(status), String::from_utf8_lossy(body).chars().take(500).collect::<String>());
            e.status = Some(status);
            e
        }
    };
    if let Some(after) = retry_after(headers) {
        error.retry_after = Some(after);
    }
    error.into()
}

/// Builds a [`Google`] provider.
#[must_use]
pub struct GoogleBuilder {
    key: SecretString,
    base_url: String,
    api_version: String,
    revision: String,
    http: Option<Http>,
}

impl GoogleBuilder {
    /// Sends requests somewhere other than `https://generativelanguage.googleapis.com`, such as a proxy.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// The API version path, `v1beta` by default. Google also offers a stable `v1`.
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Overrides the `Api-Revision` header. The converter is written for [`API_REVISION`].
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// Shares an HTTP client, such as the one `genai::Client` gives every provider.
    pub fn http(mut self, http: Http) -> Self {
        self.http = Some(http);
        self
    }

    /// The provider.
    pub fn build(self) -> Result<Google> {
        let http = match self.http {
            Some(http) => http,
            None => Http::new(&HttpConfig::default())?,
        };
        Ok(Google {
            inner: Arc::new(Inner {
                http,
                key: self.key,
                base_url: self.base_url,
                api_version: self.api_version,
                revision: self.revision,
                live: RwLock::new(HashMap::new()),
                catalog: models::catalog(),
            }),
        })
    }
}

impl Provider for Google {
    fn id(&self) -> ProviderId {
        ProviderId::Google
    }

    fn stream(&self, request: Request) -> BoxFuture<'_, Result<EventStream>> {
        Box::pin(async move {
            let spec = self.spec(request.model.name());
            let converted = convert::request_body(&request, spec.as_ref(), true)?;
            let mut headers = self.headers()?;
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
            tracing::debug!(model = %request.model, "gemini interactions request");
            let response = self.send(reqwest::Method::POST, "/interactions", headers, Some(&converted.body)).await?;
            Ok(stream::events(response, converted.warnings))
        })
    }

    fn models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>> {
        Box::pin(async move {
            let mut specs = Vec::new();
            let mut token: Option<String> = None;
            loop {
                let path = match &token {
                    Some(t) => format!("/models?pageSize=1000&pageToken={t}"),
                    None => "/models?pageSize=1000".to_owned(),
                };
                let response = self.send(reqwest::Method::GET, &path, self.headers()?, None).await?;
                let page: Value = response.json().await.map_err(|e| Error::Decode { provider: ProviderId::Google, message: e.to_string() })?;
                specs.extend(page.get("models").and_then(Value::as_array).into_iter().flatten().filter_map(|m| Spec::from_live(m, &self.inner.catalog)));
                token = page.get("nextPageToken").and_then(Value::as_str).filter(|t| !t.is_empty()).map(str::to_owned);
                if token.is_none() {
                    break;
                }
            }
            if let Ok(mut live) = self.inner.live.write() {
                *live = specs.iter().map(|s| (s.id.clone(), s.clone())).collect();
            }
            Ok(specs.iter().map(Spec::info).collect())
        })
    }
}
