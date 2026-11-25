# genai

**One async Rust API for Claude, GPT and Gemini: streaming, tools, structured output and reasoning, over each provider's own HTTP API.**

`genai` sends the same `Request` to Anthropic, OpenAI or Google and gives back the same `Response`. Switching provider means changing the model string. Each provider crate talks to its API directly (Anthropic's Messages API, OpenAI's Responses API, Google's Interactions API) with `reqwest` and its own server-sent-events parser. No vendor SDKs are involved. When a model can't honor a setting exactly, the provider maps it to the nearest one and says so in `response.warnings`, or refuses it if you ask for strict mode.

---

## Installation

```toml
[dependencies]
genai = { git = "https://github.com/tensorai-co/genai" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

All three providers are on by default. To build only the ones you use:

```toml
genai = { git = "https://github.com/tensorai-co/genai", default-features = false, features = ["anthropic"] }
```

| Feature | Default | What it adds |
| :--- | :---: | :--- |
| `anthropic` | ✓ | `genai::anthropic` and `ClientBuilder::anthropic` |
| `openai` | ✓ | `genai::openai` and `ClientBuilder::openai` |
| `google` | ✓ | `genai::google` and `ClientBuilder::google` |
| `schemars` | | `Tool::parameters_from::<T>()` and `OutputFormat::json_schema_for::<T>()`, which derive JSON Schemas from Rust types |

Requires Rust 1.88 or newer (edition 2024). TLS is rustls; nothing links to OpenSSL.

---

## Quickstart

```rust
use futures_util::StreamExt;
use genai::{Client, Effort, Event, Message, Request};

#[tokio::main]
async fn main() -> genai::Result<()> {
    let client = Client::builder()
        .anthropic(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default())
        .openai(std::env::var("OPENAI_API_KEY").unwrap_or_default())
        .google(std::env::var("GEMINI_API_KEY").unwrap_or_default())
        .build()?;

    let request = Request::new("anthropic/claude-opus-5-5")
        .instructions("You are a concise coding assistant.")
        .message(Message::user("Why does `Rc<RefCell<T>>` panic at runtime instead of failing to compile?"))
        .effort(Effort::Medium)
        .max_output_tokens(4_096);

    // The whole response at once...
    let response = client.generate(request.clone()).await?;
    println!("{}", response.text());

    // ...or as it streams in, from a different provider.
    let mut events = client.stream(request.model("openai/gpt-6.2-sol")).await?;
    while let Some(event) = events.next().await {
        if let Event::TextDelta { text, .. } = event? {
            print!("{text}");
        }
    }
    Ok(())
}
```

Models are named `provider/model`: `anthropic/claude-haiku-5-5`, `openai/gpt-6-luna`, `google/gemini-4-hydrogen`. A name without a known provider, or for a provider with no key, fails with `Error::UnknownProvider` or `Error::NotConfigured` when sent.

---

## Using it

### Conversations

A conversation is a `Vec<Message>`. Add `response.message` to it as-is: besides the text, it carries the provider data the next turn needs, such as reasoning signatures, encrypted reasoning and item ids. Messages, parts and that data all serialize with serde, so a conversation can be saved to disk and resumed later.

```rust
let mut history = vec![Message::user("Name a prime between 90 and 100.")];
let first = client.generate(Request::new("google/gemini-3.5-flash-lite").messages(history.clone())).await?;
history.push(first.message.clone());
history.push(Message::user("And one between 100 and 110?"));
let second = client.generate(Request::new("google/gemini-3.5-flash-lite").messages(history)).await?;
```

A conversation can move between providers mid-way. Text, images, tool calls and tool results carry over. One provider's reasoning data can't be read by another, so it is left out, with a warning.

### Tools

```rust
use genai::{Message, Request, StopReason, Tool, ToolResult};
use serde_json::json;

let add = Tool::function("add").description("Adds two integers.").parameters(json!({
    "type": "object",
    "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
    "required": ["a", "b"]
}));

let mut history = vec![Message::user("What is 2 + 3?")];
loop {
    let response = client.generate(Request::new("openai/gpt-6-luna").tool(add.clone()).messages(history.clone())).await?;
    history.push(response.message.clone());
    if response.stop_reason != StopReason::ToolCalls {
        println!("{}", response.text());
        break;
    }
    let results: Vec<ToolResult> = response
        .tool_calls()
        .map(|call| {
            let sum = call.arguments["a"].as_i64().unwrap_or(0) + call.arguments["b"].as_i64().unwrap_or(0);
            ToolResult::new(call, sum.to_string())
        })
        .collect();
    history.push(Message::tool_results(results));
}
```

- **Controlling calls:** `ToolChoice::{Auto, None, Required, Named(..)}` decides whether and which tools get called. `parallel_tool_calls(false)` asks for one call at a time.
- **Strict schemas:** `Tool::strict(true)` asks the provider to keep the arguments valid against the schema. genai rewrites the schema into the form each provider's strict mode requires.
- **Typed arguments:** `call.arguments::<T>()` decodes the arguments into a Rust type.
- **Failures:** `ToolResult::error` reports that the tool failed.

### Structured output

```rust
use genai::OutputFormat;

#[derive(serde::Deserialize)]
struct Answer {
    answer: i64,
}

let schema = json!({ "type": "object", "properties": { "answer": { "type": "integer" } }, "required": ["answer"] });
let response = client.generate(Request::new("anthropic/claude-haiku-5-5").message(Message::user("What is 6 × 7?")).output(OutputFormat::json_schema("answer", schema))).await?;
let answer: Answer = response.json()?;
```

### Effort and reasoning

`Effort` is the one setting for how much work the model puts in: how long it reasons, how thorough it is, and so its speed and cost. The levels are `None`, `Minimal`, `Low`, `Medium`, `High`, `XHigh` and `Max`. Each provider maps the level to its own setting, using what the model supports. Leave it unset to get the model's default.

`ReasoningDisplay` (`Hidden`, `Summary`, `Detailed`) sets how much of the reasoning comes back to show. `response.reasoning()` returns it, and streams deliver it as `Event::ReasoningDelta`. The data needed for the next turn comes back regardless of this setting.

### Media

```rust
use genai::Media;

let message = Message::user("What's in these?")
    .with(Media::image("image/png", std::fs::read("diagram.png")?))
    .with(Media::pdf(std::fs::read("spec.pdf")?).title("Spec"))
    .with(Media::image_url("https://example.com/photo.jpg"));
```

The other constructors are `Media::pdf_url` and `Media::text_document`. `Media::new` takes any kind (image, document, audio or video) from base64 data, a URL or a provider file id.

### Streaming events

`client.stream(..)` yields `Event`s in the same order for every provider:

| Event | Meaning |
| :--- | :--- |
| `Start { id, model }` | The response started |
| `PartStart { index, part }` | A part started (text, reasoning, tool call, …). `index` is its position in the final message |
| `TextDelta`, `ReasoningDelta`, `ToolCallDelta` | More text, more reasoning to show, or more of a tool call's JSON arguments |
| `PartStop { index, part }` | The part, complete: tool arguments parsed, provider data attached |
| `Usage(usage)` | Token usage so far, when the provider reports it mid-stream |
| `Finish { stop_reason, usage, warnings }` | The end |

`Accumulator` folds the events back into a `Response`, which is what `generate` does. Dropping the stream cancels the request.

### Provider options

Settings only one provider understands go in that provider's options type. A request can carry options for several providers, and each provider reads only its own.

```rust
use genai::anthropic::{AnthropicOptions, Speed};
use genai::google::GoogleOptions;
use genai::openai::{OpenAiOptions, ServiceTier, Verbosity};

let request = Request::new("openai/gpt-6.2-sol")
    .message(Message::user("Summarize this diff."))
    .options(AnthropicOptions::new().speed(Speed::Fast))
    .options(OpenAiOptions::new().service_tier(ServiceTier::Flex).verbosity(Verbosity::Low).prompt_cache_key("conversation-42"))
    .options(GoogleOptions::new().label("team", "search"));
```

| Options | Settings |
| :--- | :--- |
| `AnthropicOptions` | `speed`, `service_tier`, `inference_geo`, `cache_ttl`, `beta` |
| `OpenAiOptions` | `service_tier`, `verbosity`, `store`, `prompt_cache_key` |
| `GoogleOptions` | `service_tier`, `store`, `safety`, `label` |

For anything not typed yet, `Request::extra_body(ProviderId::OpenAi, json!({ … }))` merges raw JSON into that provider's request body.

### Configuring the client

```rust
use std::time::Duration;

let client = Client::builder()
    .anthropic(anthropic_key)
    .max_retries(3)
    .connect_timeout(Duration::from_secs(5))
    .read_timeout(Duration::from_secs(120))
    .user_agent("my-app/1.0")
    .build()?;
```

- **Shared HTTP client:** all providers share one HTTP client, and `Client` is cheap to clone.
- **Retries:** on by default, twice, with exponential backoff and jitter, following the provider's `retry-after`.
  - Retried: overload, 5xx errors, timeouts, dropped connections, and per-minute rate limits.
  - Not retried: spend caps, daily quotas, used-up credit, and anything else that would fail again.
  - Never retried once a stream has started, since that would repeat output.
- **Timeouts:** none on a response's total time. The read timeout only catches a connection that has gone silent.
- **Keys:** held as `secrecy::SecretString`, never printed by `Debug`, and marked sensitive in headers.
- **Key sources:** genai never reads the environment unless you call `Client::from_env()`. That reads `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
- **Proxies and other hosts:** each provider can also be built on its own with a custom base URL and added with `ClientBuilder::provider`. For example: `genai::openai::OpenAi::builder(key).base_url(..).project(..)`, `Anthropic::builder(key).workspace_id(..)`, `Google::builder(key).api_version("v1")`.

### Models

`client.models()` lists every configured provider's text models. It combines the live models list with a built-in catalog for what the lists don't report: effort levels, defaults, and whether reasoning can be turned off. `ModelInfo` has the display name, family, context window, output limit, capabilities and lifecycle status, enough for a model picker. `client.count_tokens(&request)` counts input tokens without running the request, on Anthropic.

---

## What each provider does

Every request is stateless. The whole conversation goes with each request, and nothing is left stored with the provider.

| | Anthropic | OpenAI | Google |
| :--- | :--- | :--- | :--- |
| API | Messages, version `2026-10-21` | Responses (`/v1/responses`) | Interactions, `v1beta`, revision `2026-05-20` |
| Storage | Keeps no conversation state | `store: false`, reasoning returned encrypted | `store: false` (Google's default would keep requests up to 55 days) |
| `instructions` | `system` | A leading `developer` message (keeps them in the prompt cache) | `system_instruction` |
| `max_output_tokens` | Required by the API; defaults to 16,384 | Optional | Optional |
| `Effort::None` | `thinking: disabled`, or `between_tools` on Sonnet 5.5; Fable and Opus 5.5 can't, so low + warning | `none` where the model takes it; GPT-6.2 Astra and Sol can't, so low + warning | No off switch: the lowest level + warning |
| Other effort levels | `output_config.effort`; nearest supported level + warning; a thinking budget on models without effort | `reasoning.effort`; nearest supported level + warning | `thinking_level`; `xhigh` only on Gemini 4 Neon and Hydrogen; no `max` |
| `ReasoningDisplay` | `thinking.display` | `reasoning.summary` | `thinking_summaries` |
| `temperature` | Removed in this API version: dropped + warning | Dropped + warning on GPT-5 and GPT-6 models | Sent |
| `stop_sequences` | Sent | Not in the Responses API: dropped + warning | Sent |
| Forced tool choice | Relaxed to auto + warning on Fable 5.2, Opus 5.5, Sonnet 5.5 and Fable 5.1 | Sent | Sent (`any`, or `allowed_tools` for one tool) |
| Strict tools | `strict` with a rewritten schema | `strict` with a rewritten schema | `validated` tool choice |
| `parallel_tool_calls(false)` | `disable_parallel_tool_use` | Sent | Not available: warning |
| `Cache::Auto` | Top-level `cache_control` | Automatic; `prompt_cache_key` in options | Automatic |
| `user_id` | `metadata.user_id` | `safety_identifier` | Not available |
| Input | Text, images, PDFs, text documents | Text, images, PDFs and files | Text, images, documents, audio, video |
| Token counting | `count_tokens` | Not yet | Not yet |

`Request::strict(true)` turns every "+ warning" above into `Error::Unsupported`.

---

## Errors

`genai::Error` covers:
- `UnknownProvider` and `NotConfigured`: routing problems.
- `Unsupported` and `InvalidInput`: problems caught before anything is sent.
- `Api`: the provider answered with an error.
- `Stream`, `Decode` and `Network`: the stream broke, a response couldn't be read, or the connection failed.

An `ApiError` carries:
- `kind`: the same across providers (`Auth`, `Permission`, `Billing`, `NotFound`, `InvalidRequest`, `ContextTooLong`, `RequestTooLarge`, `RateLimited`, `Overloaded`, `Timeout`, `Server`, `Other`).
- the HTTP `status`, and the provider's own `code` (such as `overloaded_error`, `insufficient_quota` or `model_not_found`).
- the `message`, the provider's `request_id` where it sends one, and `retry_after`.

```rust
match client.generate(request).await {
    Ok(response) => println!("{}", response.text()),
    Err(genai::Error::Api(e)) if e.kind == genai::ApiErrorKind::ContextTooLong => trim_history(),
    Err(e) if e.is_retryable() => retry_later(e),
    Err(e) => return Err(e),
}
```

Errors that arrive in the middle of a stream come out of the stream as `Err`. A Gemini safety block mid-stream ends the response with `StopReason::Refused` instead.

---

## Adding a provider

A provider is a crate that depends on `genai-core` and implements `Provider`:

```rust
pub trait Provider: Send + Sync + 'static {
    fn id(&self) -> ProviderId;
    fn stream(&self, request: Request) -> BoxFuture<'_, Result<EventStream>>;
    fn generate(&self, request: Request) -> BoxFuture<'_, Result<Response>>;           // default: stream + Accumulator
    fn models(&self) -> BoxFuture<'_, Result<Vec<ModelInfo>>>;
    fn count_tokens<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<u64>>; // default: Unsupported
}
```

The trait is object-safe, so `ClientBuilder::provider` can register a provider at runtime. `genai_core::http` supplies what the built-in providers share:
- `Http`, with retries
- `SseParser` and `sse_events`
- `retry_after`

The three provider crates follow the same layout, and any of them works as a template:

| File | Holds |
| :--- | :--- |
| `convert.rs` | Neutral `Request` → request body |
| `stream.rs` | Server-sent events → `Event` |
| `models.rs` | Catalog and live models list |
| `options.rs` | The provider's typed options |

---

## Status

| Provider | Ready | Not done yet |
| :--- | :--- | :--- |
| **Anthropic** | Text, images, PDFs, streaming, tools, structured output, effort and thinking, prompt caching, token counting, model catalog | Hosted tools (web search, code execution), Files API, batches, compaction, server-side threads |
| **OpenAI** | Text, images, files, streaming, tools, structured output, effort and reasoning summaries, encrypted reasoning, model catalog | Hosted tools, token counting, background mode, server-side state (`previous_response_id`), the Chat Completions format used by OpenAI-compatible servers |
| **Google** | Text, images, documents, audio, video, streaming, tools, structured output, thinking levels and summaries, model catalog | Hosted tools (Google Search, URL context, code execution), token counting, long decode and continuation, server-side state (`previous_interaction_id`) |

Embeddings, image generation, realtime voice and managed agents are out of scope for now.

---

## Run and test

```sh
cargo test --workspace --all-features     # unit tests, recorded streams and a mock HTTP server; no keys needed
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Each provider has three layers of offline tests:
- **Unit tests** check request conversion (efforts, tools, schemas, media, conversation replay).
- **Recorded streams** (`tests/fixtures/*.sse`) check stream decoding.
- **`tests/http.rs`** runs against a mock server for headers, retries, error mapping, cut-off streams and model listing.

The live tests call the real APIs with one-line prompts, low effort and at most 1024 output tokens, and cost a fraction of a cent:

```sh
ANTHROPIC_API_KEY=… cargo test -p genai --test anthropic_live -- --ignored --test-threads=1 --nocapture
OPENAI_API_KEY=…    cargo test -p genai --test openai_live    -- --ignored --test-threads=1 --nocapture
GEMINI_API_KEY=…    cargo test -p genai --test google_live    -- --ignored --test-threads=1 --nocapture
```

The examples need a key in the environment too:

```sh
ANTHROPIC_API_KEY=… cargo run --example chat    # a streaming chat in the terminal
ANTHROPIC_API_KEY=… cargo run --example tools   # a tool loop
```

---

## Repository layout

```text
genai/
├── Cargo.toml                 # workspace (edition 2024)
├── crates/
│   ├── genai/                 # what apps use: Client, ClientBuilder, re-exports, features
│   │   ├── examples/          # chat, tools
│   │   └── tests/             # live tests, one file per provider
│   ├── genai-core/            # neutral types and the Provider trait
│   │   └── src/
│   │       ├── message.rs     # Message, Role, Part, Media, ToolCall, ToolResult, Reasoning, Opaque
│   │       ├── request.rs     # Request, Effort, ReasoningDisplay, ToolChoice, OutputFormat, Cache, ProviderOptions
│   │       ├── response.rs    # Response, StopReason, Usage, Warning
│   │       ├── stream.rs      # Event, EventStream, Accumulator
│   │       ├── model.rs       # ModelId, ModelInfo, Capabilities
│   │       ├── tool.rs        # Tool
│   │       ├── error.rs       # Error, ApiError, ApiErrorKind
│   │       ├── provider.rs    # Provider, ProviderId
│   │       └── http/          # shared client, retries, server-sent events parser
│   ├── genai-anthropic/       # Messages API
│   ├── genai-openai/          # Responses API
│   └── genai-google/          # Interactions API
└── rustfmt.toml
```

---

## License

Licensed under the [MIT License](LICENSE).
