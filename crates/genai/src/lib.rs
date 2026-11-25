//! One async Rust API for model providers. Talks to each provider's HTTP API directly; no vendor SDKs.
//!
//! ```no_run
//! use genai::{Client, Effort, Event, Message, Request};
//! use futures_util::StreamExt;
//!
//! # async fn run() -> genai::Result<()> {
//! let client = Client::builder().anthropic(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default()).build()?;
//!
//! let request = Request::new("anthropic/claude-haiku-5-5")
//!     .instructions("Answer in one sentence.")
//!     .message(Message::user("Why is the sky blue?"))
//!     .effort(Effort::Low)
//!     .max_output_tokens(512);
//!
//! // The whole response at once...
//! let response = client.generate(request.clone()).await?;
//! println!("{}", response.text());
//!
//! // ...or as it streams in.
//! let mut events = client.stream(request).await?;
//! while let Some(event) = events.next().await {
//!     if let Event::TextDelta { text, .. } = event? {
//!         print!("{text}");
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! Switching provider means changing the model string. Settings a model can't honor exactly are approximated and
//! reported in [`Response::warnings`], or refused under [`Request::strict`].

mod client;

pub use client::{Client, ClientBuilder};
pub use genai_core::http::{HttpConfig, RetryPolicy};
pub use genai_core::{
    Accumulator, ApiError, ApiErrorKind, Cache, Capabilities, Effort, Error, Event, EventStream, Media, MediaKind, MediaSource, Message, Modality, ModelId, ModelInfo, ModelStatus, Opaque,
    OutputFormat, Part, Provider, ProviderId, ProviderOptions, Reasoning, ReasoningDisplay, Request, Response, Result, Role, StopReason, Text, Tool, ToolCall, ToolChoice, ToolResult, Usage, Warning,
};

/// The Anthropic provider and its options.
#[cfg(feature = "anthropic")]
pub mod anthropic {
    pub use genai_anthropic::{API_VERSION, Anthropic, AnthropicBuilder, AnthropicOptions, BASE_URL, CacheTtl, ServiceTier, Speed};
}

/// The OpenAI provider and its options.
#[cfg(feature = "openai")]
pub mod openai {
    pub use genai_openai::{BASE_URL, OpenAi, OpenAiBuilder, OpenAiOptions, ServiceTier, Verbosity};
}

/// The Google provider and its options.
#[cfg(feature = "google")]
pub mod google {
    pub use genai_google::{API_REVISION, BASE_URL, Google, GoogleBuilder, GoogleOptions, SafetySetting, SafetyThreshold, ServiceTier};
}
