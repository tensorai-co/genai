//! What to ask a model.

use serde::{Deserialize, Serialize};

use crate::message::Message;
use crate::model::ModelId;
use crate::provider::ProviderId;
use crate::tool::Tool;

/// How much work the model puts into a response: how long it reasons, how thorough it is, and so how fast and costly it
/// is. Each provider maps this to its own setting, and to the nearest level a model supports (with a
/// [`Warning`](crate::Warning), or an error under [`Request::strict`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Effort {
    /// No reasoning, where the model allows turning it off.
    None,
    /// Near-off reasoning.
    Minimal,
    /// Fast and cheap.
    Low,
    /// Balanced.
    Medium,
    /// Thorough. The default on most models.
    High,
    /// For long, hard agentic and coding work.
    #[serde(rename = "xhigh")]
    XHigh,
    /// Everything the model has, regardless of cost.
    Max,
}

impl Effort {
    /// Every level, lowest first.
    pub const ALL: [Effort; 7] = [Effort::None, Effort::Minimal, Effort::Low, Effort::Medium, Effort::High, Effort::XHigh, Effort::Max];

    /// The lowercase name, as most providers spell it.
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::None => "none",
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }
}

/// How much of the model's reasoning comes back to show.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningDisplay {
    /// None. The model still reasons, and the data needed for the next turn still comes back.
    Hidden,
    /// A summary.
    Summary,
    /// As much as the provider shows.
    Detailed,
}

/// Whether and how the model may call tools.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides. The default.
    Auto,
    /// No tool calls.
    None,
    /// At least one tool call.
    Required,
    /// A call to this tool.
    Named(String),
}

/// A shape for the response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum OutputFormat {
    /// JSON matching a schema. Read it with [`Response::json`](crate::Response::json).
    JsonSchema {
        /// A name for the schema; some providers require one.
        name: String,
        /// The JSON Schema. Providers accept different subsets; genai adjusts it where it can.
        schema: serde_json::Value,
    },
}

impl OutputFormat {
    /// JSON matching `schema`.
    pub fn json_schema(name: impl Into<String>, schema: serde_json::Value) -> Self {
        OutputFormat::JsonSchema { name: name.into(), schema }
    }

    /// JSON matching the schema of a Rust type.
    #[cfg(feature = "schemars")]
    pub fn json_schema_for<T: schemars::JsonSchema>(name: impl Into<String>) -> Self {
        OutputFormat::json_schema(name, serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default())
    }
}

/// Prompt caching.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Cache {
    /// Cache the conversation so far, the way each provider does best. The default.
    #[default]
    Auto,
    /// Ask for no caching where the provider allows it.
    Off,
}

/// Settings only one provider understands, as that provider's typed options turned into JSON. Each provider crate
/// exports its own options type with a `From` impl; a request can carry options for several providers, and each provider
/// reads only its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderOptions {
    /// Whose options these are.
    pub provider: ProviderId,
    /// The options.
    pub value: serde_json::Value,
}

/// A request to a model. Build it with [`Request::new`] and the setters, which all take and return `self`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Request {
    /// Which model, such as `anthropic/claude-opus-5-5`.
    pub model: ModelId,
    /// The system prompt.
    pub instructions: Option<String>,
    /// The conversation so far, oldest first. The last message is usually from the user.
    pub messages: Vec<Message>,
    /// The most tokens to produce, reasoning included. Providers that require a limit get a default when this is unset.
    pub max_output_tokens: Option<u32>,
    /// How much work the model puts in. `None` leaves the model's default.
    pub effort: Option<Effort>,
    /// How much reasoning comes back to show. `None` leaves the provider's default.
    pub reasoning_display: Option<ReasoningDisplay>,
    /// Tools the model may call.
    pub tools: Vec<Tool>,
    /// Whether and how it may call them. `None` leaves the provider's default (auto).
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may call several tools at once. `None` leaves the provider's default (yes).
    pub parallel_tool_calls: Option<bool>,
    /// A shape for the response.
    pub output: Option<OutputFormat>,
    /// Sampling temperature. Not every provider still accepts it; those that don't drop it with a warning.
    pub temperature: Option<f32>,
    /// Text that ends the response when the model writes it.
    pub stop_sequences: Vec<String>,
    /// Prompt caching.
    pub cache: Cache,
    /// An opaque id for the end user, which providers use to spot abuse. Never a name or email.
    pub user_id: Option<String>,
    /// Turn every [`Warning`](crate::Warning) about an approximated setting into an [`Error::Unsupported`](crate::Error::Unsupported).
    pub strict: bool,
    /// Provider-specific options.
    pub options: Vec<ProviderOptions>,
    /// Raw JSON merged into the request body, per provider, for features genai doesn't type yet.
    pub extra_body: Vec<ProviderOptions>,
}

impl Request {
    /// A request to `model`, such as `"anthropic/claude-opus-5-5"`.
    pub fn new(model: impl Into<ModelId>) -> Self {
        Request {
            model: model.into(),
            instructions: None,
            messages: Vec::new(),
            max_output_tokens: None,
            effort: None,
            reasoning_display: None,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            output: None,
            temperature: None,
            stop_sequences: Vec::new(),
            cache: Cache::Auto,
            user_id: None,
            strict: false,
            options: Vec::new(),
            extra_body: Vec::new(),
        }
    }

    /// Sets the model.
    pub fn model(mut self, model: impl Into<ModelId>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the system prompt.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Adds a message.
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Adds messages.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Sets the output limit.
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Sets the effort.
    pub fn effort(mut self, effort: Effort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Sets how much reasoning comes back.
    pub fn reasoning_display(mut self, display: ReasoningDisplay) -> Self {
        self.reasoning_display = Some(display);
        self
    }

    /// Adds a tool.
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Sets the tool choice.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Allows or forbids several tool calls at once.
    pub fn parallel_tool_calls(mut self, allowed: bool) -> Self {
        self.parallel_tool_calls = Some(allowed);
        self
    }

    /// Sets the output format.
    pub fn output(mut self, format: OutputFormat) -> Self {
        self.output = Some(format);
        self
    }

    /// Sets the temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Adds a stop sequence.
    pub fn stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    /// Sets prompt caching.
    pub fn cache(mut self, cache: Cache) -> Self {
        self.cache = cache;
        self
    }

    /// Sets the end-user id.
    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    /// Fails instead of approximating unsupported settings.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Sets one provider's options, replacing any earlier ones for that provider.
    pub fn options(mut self, options: impl Into<ProviderOptions>) -> Self {
        let options = options.into();
        self.options.retain(|o| o.provider != options.provider);
        self.options.push(options);
        self
    }

    /// Merges raw JSON into the body sent to `provider`. Use only for features genai doesn't type yet.
    pub fn extra_body(mut self, provider: ProviderId, value: serde_json::Value) -> Self {
        self.extra_body.push(ProviderOptions { provider, value });
        self
    }

    /// The options for `provider`, if any.
    pub fn options_for(&self, provider: &ProviderId) -> Option<&serde_json::Value> {
        self.options.iter().find(|o| &o.provider == provider).map(|o| &o.value)
    }
}
