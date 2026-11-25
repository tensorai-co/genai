//! The neutral half of [genai](https://github.com/tensorai-co/genai): the types every provider speaks, the [`Provider`]
//! trait a provider implements, and the HTTP plumbing they share (retries, server-sent events).
//!
//! Apps depend on the `genai` crate, which re-exports all of this. Depend on `genai-core` directly only to write a provider.

pub mod error;
pub mod http;
pub mod message;
pub mod model;
pub mod provider;
pub mod request;
pub mod response;
pub mod stream;
pub mod tool;

pub use error::{ApiError, ApiErrorKind, Error, Result};
pub use message::{Media, MediaKind, MediaSource, Message, Opaque, Part, Reasoning, Role, Text, ToolCall, ToolResult};
pub use model::{Capabilities, Modality, ModelId, ModelInfo, ModelStatus};
pub use provider::{BoxFuture, Provider, ProviderId};
pub use request::{Cache, Effort, OutputFormat, ProviderOptions, ReasoningDisplay, Request, ToolChoice};
pub use response::{Response, StopReason, Usage, Warning};
pub use stream::{Accumulator, Event, EventStream};
pub use tool::Tool;
