//! What can go wrong.

use std::fmt;
use std::time::Duration;

use crate::provider::ProviderId;

/// A boxed error from a lower layer, such as the HTTP client.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// `Result` with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything genai can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The model string names a provider this client doesn't know, such as `foo/model` or a name with no `provider/` prefix.
    #[error("no provider named `{0}`; model strings look like `anthropic/claude-opus-5-5`")]
    UnknownProvider(String),
    /// The provider is known but wasn't given a key.
    #[error("{0} is not configured; give the client a key for it")]
    NotConfigured(ProviderId),
    /// The request asks for something the provider or model can't do, and either can't be approximated or
    /// [`Request::strict`](crate::Request::strict) is on.
    #[error("{provider} doesn't support {feature}")]
    Unsupported {
        /// The provider.
        provider: ProviderId,
        /// What it doesn't support.
        feature: String,
    },
    /// The request is malformed before it is sent (for example an image with no media type).
    #[error("invalid request: {0}")]
    InvalidInput(String),
    /// The provider answered with an error.
    #[error(transparent)]
    Api(Box<ApiError>),
    /// The response stream broke the provider's own protocol, or ended early.
    #[error("{provider} stream: {message}")]
    Stream {
        /// The provider.
        provider: ProviderId,
        /// What went wrong.
        message: String,
    },
    /// A response couldn't be decoded, including [`Response::json`](crate::Response::json) on text that isn't the asked-for JSON.
    #[error("couldn't decode {provider} response: {message}")]
    Decode {
        /// The provider.
        provider: ProviderId,
        /// What went wrong.
        message: String,
    },
    /// The request didn't reach the provider, or the connection dropped.
    #[error("couldn't reach {provider}: {source}")]
    Network {
        /// The provider.
        provider: ProviderId,
        /// The underlying error.
        #[source]
        source: BoxError,
    },
}

impl Error {
    /// Whether sending the same request again could succeed: rate limits that say when to retry, overload, 5xx server
    /// errors, timeouts and dropped connections.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Api(e) => e.is_retryable(),
            Error::Network { .. } => true,
            _ => false,
        }
    }

    /// The provider the error came from, when there is one.
    pub fn provider(&self) -> Option<&ProviderId> {
        match self {
            Error::NotConfigured(p) => Some(p),
            Error::Unsupported { provider, .. } | Error::Stream { provider, .. } | Error::Decode { provider, .. } | Error::Network { provider, .. } => Some(provider),
            Error::Api(e) => Some(&e.provider),
            Error::UnknownProvider(_) | Error::InvalidInput(_) => None,
        }
    }

    /// The API error, when the provider answered with one.
    pub fn api(&self) -> Option<&ApiError> {
        match self {
            Error::Api(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ApiError> for Error {
    fn from(e: ApiError) -> Self {
        Error::Api(Box::new(e))
    }
}

/// An error the provider answered with.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ApiError {
    /// The provider.
    pub provider: ProviderId,
    /// What kind of error, across providers.
    pub kind: ApiErrorKind,
    /// The HTTP status, or `None` for an error that arrived inside a stream.
    pub status: Option<u16>,
    /// The provider's own error type or code, such as `overloaded_error` or `ramp_limit_exceeded`.
    pub code: Option<String>,
    /// The provider's message.
    pub message: String,
    /// The provider's request id, for support.
    pub request_id: Option<String>,
    /// How long the provider asked us to wait before retrying.
    pub retry_after: Option<Duration>,
}

impl ApiError {
    /// An error of `kind` from `provider`, with no status, code or request id yet.
    pub fn new(provider: ProviderId, kind: ApiErrorKind, message: impl Into<String>) -> Self {
        ApiError { provider, kind, status: None, code: None, message: message.into(), request_id: None, retry_after: None }
    }

    /// Whether sending the same request again could succeed. A rate limit with no `retry-after` is not retryable: providers
    /// send those for spend caps, which keep failing until access resumes.
    pub fn is_retryable(&self) -> bool {
        match self.kind {
            ApiErrorKind::RateLimited => self.retry_after.is_some(),
            ApiErrorKind::Overloaded | ApiErrorKind::Timeout | ApiErrorKind::Server => true,
            _ => false,
        }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.provider, self.kind)?;
        if let Some(status) = self.status {
            write!(f, " ({status})")?;
        }
        write!(f, ": {}", self.message)?;
        if let Some(id) = &self.request_id {
            write!(f, " [request {id}]")?;
        }
        Ok(())
    }
}

impl std::error::Error for ApiError {}

/// The kind of an [`ApiError`], the same across providers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ApiErrorKind {
    /// The key is missing, malformed, revoked or expired (401).
    Auth,
    /// The key can't use this resource or model (403).
    Permission,
    /// Billing or credit problem (402), or a spend limit.
    Billing,
    /// No such model or resource (404).
    NotFound,
    /// The request was rejected as invalid (400, 422).
    InvalidRequest,
    /// The input doesn't fit the model's context window.
    ContextTooLong,
    /// The request body is too large (413).
    RequestTooLarge,
    /// Too many requests or tokens (429).
    RateLimited,
    /// The provider is overloaded (Anthropic's 529, or 503).
    Overloaded,
    /// The provider timed out (504).
    Timeout,
    /// Any other server error (5xx).
    Server,
    /// Anything else.
    Other,
}

impl ApiErrorKind {
    /// The usual kind for an HTTP status, before the provider's own error type refines it.
    pub fn from_status(status: u16) -> Self {
        match status {
            400 | 422 => ApiErrorKind::InvalidRequest,
            401 => ApiErrorKind::Auth,
            402 => ApiErrorKind::Billing,
            403 => ApiErrorKind::Permission,
            404 => ApiErrorKind::NotFound,
            413 => ApiErrorKind::RequestTooLarge,
            429 => ApiErrorKind::RateLimited,
            503 | 529 => ApiErrorKind::Overloaded,
            504 => ApiErrorKind::Timeout,
            500..=599 => ApiErrorKind::Server,
            _ => ApiErrorKind::Other,
        }
    }
}

impl fmt::Display for ApiErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ApiErrorKind::Auth => "authentication error",
            ApiErrorKind::Permission => "permission error",
            ApiErrorKind::Billing => "billing error",
            ApiErrorKind::NotFound => "not found",
            ApiErrorKind::InvalidRequest => "invalid request",
            ApiErrorKind::ContextTooLong => "context too long",
            ApiErrorKind::RequestTooLarge => "request too large",
            ApiErrorKind::RateLimited => "rate limited",
            ApiErrorKind::Overloaded => "overloaded",
            ApiErrorKind::Timeout => "timeout",
            ApiErrorKind::Server => "server error",
            ApiErrorKind::Other => "error",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spend_cap_rate_limit_is_not_retried() {
        let mut e = ApiError::new(ProviderId::Anthropic, ApiErrorKind::RateLimited, "slow down");
        assert!(!e.is_retryable());
        e.retry_after = Some(Duration::from_secs(2));
        assert!(e.is_retryable());
        assert!(ApiError::new(ProviderId::Anthropic, ApiErrorKind::Overloaded, "busy").is_retryable());
        assert!(!ApiError::new(ProviderId::Anthropic, ApiErrorKind::InvalidRequest, "bad").is_retryable());
    }
}
