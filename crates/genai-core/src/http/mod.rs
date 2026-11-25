//! The HTTP plumbing providers share: one client, retries, and a server-sent events parser.
//!
//! This module is for provider crates. It exposes [`reqwest`] types; the app-facing API doesn't.

mod retry;
mod sse;

use std::time::Duration;

pub use reqwest;
pub use retry::RetryPolicy;
pub use sse::{SseEvent, SseParser, sse_events};

use crate::error::{Error, Result};
use crate::provider::ProviderId;

/// How the shared HTTP client behaves.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HttpConfig {
    /// How long to wait for a connection.
    pub connect_timeout: Duration,
    /// How long a response may go without sending a byte. Streams send keep-alive pings, so this only trips on a stalled
    /// connection. There is no limit on a response's total time.
    pub read_timeout: Duration,
    /// Retries for failures that may pass.
    pub retry: RetryPolicy,
    /// The `User-Agent` header.
    pub user_agent: String,
}

impl Default for HttpConfig {
    fn default() -> Self {
        HttpConfig { connect_timeout: Duration::from_secs(10), read_timeout: Duration::from_secs(300), retry: RetryPolicy::default(), user_agent: concat!("genai/", env!("CARGO_PKG_VERSION")).into() }
    }
}

/// A shared HTTP client with retries. Cheap to clone.
#[derive(Clone, Debug)]
pub struct Http {
    client: reqwest::Client,
    retry: RetryPolicy,
}

impl Http {
    /// A client configured by `config`.
    pub fn new(config: &HttpConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .read_timeout(config.read_timeout)
            .user_agent(config.user_agent.clone())
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .map_err(|e| Error::InvalidInput(format!("couldn't build the HTTP client: {e}")))?;
        Ok(Http { client, retry: config.retry.clone() })
    }

    /// The underlying client, to build requests with.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Sends `request`, retrying what may pass, and returns the first successful response.
    ///
    /// A failed response's status, headers and body go to `to_error`, which turns them into the provider's [`Error`]; the
    /// error then decides whether to retry ([`Error::is_retryable`]), and waits as long as it asks
    /// ([`ApiError::retry_after`](crate::ApiError::retry_after)) or the backoff says. The request body must be bytes, so it can be resent.
    pub async fn send(&self, provider: &ProviderId, request: reqwest::Request, to_error: impl Fn(u16, &reqwest::header::HeaderMap, &[u8]) -> Error) -> Result<reqwest::Response> {
        let mut attempt = 0u32;
        loop {
            let this = request.try_clone().ok_or_else(|| Error::InvalidInput("the request body can't be resent".into()))?;
            let error = match self.client.execute(this).await {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status().as_u16();
                    let headers = response.headers().clone();
                    let body = response.bytes().await.unwrap_or_default();
                    to_error(status, &headers, &body)
                }
                Err(e) => Error::Network { provider: provider.clone(), source: Box::new(e) },
            };
            if attempt >= self.retry.max_retries || !error.is_retryable() {
                return Err(error);
            }
            let wait = error.api().and_then(|e| e.retry_after).map_or_else(|| self.retry.delay(attempt), |after| after.min(self.retry.max_delay));
            tracing::debug!(%provider, attempt, ?wait, %error, "retrying");
            tokio::time::sleep(wait).await;
            attempt += 1;
        }
    }
}

/// Reads a `retry-after` header given in seconds.
pub fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    value.trim().parse::<f64>().ok().filter(|s| s.is_finite() && *s >= 0.0).map(Duration::from_secs_f64)
}
