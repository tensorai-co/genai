//! Live checks against the real Anthropic API. Ignored by default; they need a key and spend a few cents at most.
//!
//! ```sh
//! ANTHROPIC_API_KEY=… cargo test -p genai --test anthropic_live -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! Every request is tiny on purpose: a one-line prompt, low effort, and at most 1024 output tokens.
#![cfg(feature = "anthropic")]

use futures_util::StreamExt;
use genai::{Client, Effort, Event, Message, OutputFormat, Part, ReasoningDisplay, Request, Response, StopReason, Tool, ToolResult};
use serde_json::json;

const MODEL: &str = "anthropic/claude-haiku-5-5";

fn client() -> Client {
    Client::from_env().unwrap_or_else(|e| panic!("{e}"))
}

fn small(prompt: &str) -> Request {
    Request::new(MODEL).message(Message::user(prompt)).effort(Effort::Low).max_output_tokens(1024)
}

fn report(name: &str, r: &Response) {
    println!("[{name}] {:?} in={} out={} reasoning={:?} warnings={:?} text={:?}", r.stop_reason, r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens, r.warnings, r.text());
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn generates_text() {
    let r = client().generate(small("Reply with exactly one word: pong").instructions("You answer tersely.")).await.unwrap_or_else(|e| panic!("{e}"));
    report("text", &r);
    assert_eq!(r.stop_reason, StopReason::Complete);
    assert!(r.text().to_lowercase().contains("pong"));
    assert!(r.usage.input_tokens > 0 && r.usage.output_tokens > 0);
    assert_eq!(r.model.name(), "claude-haiku-5-5");
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn streams_text() {
    let mut events = client().stream(small("Count from 1 to 5, separated by spaces.")).await.unwrap_or_else(|e| panic!("{e}"));
    let (mut deltas, mut text, mut finished) = (0, String::new(), None);
    while let Some(event) = events.next().await {
        match event.unwrap_or_else(|e| panic!("{e}")) {
            Event::TextDelta { text: t, .. } => {
                deltas += 1;
                text.push_str(&t);
            }
            Event::Finish { stop_reason, usage, .. } => finished = Some((stop_reason, usage)),
            _ => {}
        }
    }
    println!("[stream] {deltas} deltas, {text:?}, {finished:?}");
    assert!(deltas > 0);
    assert!(text.contains('5'));
    assert!(matches!(finished, Some((StopReason::Complete, _))));
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn runs_a_tool_loop() {
    let client = client();
    let add = Tool::function("add").description("Adds two integers.").parameters(json!({
        "type": "object",
        "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
        "required": ["a", "b"]
    }));
    let mut history = vec![Message::user("What is 2 + 3? Use the add tool, then answer with just the number.")];
    let first = client.generate(Request::new(MODEL).tool(add.clone()).messages(history.clone()).effort(Effort::Low).max_output_tokens(1024)).await.unwrap_or_else(|e| panic!("{e}"));
    report("tool call", &first);
    assert_eq!(first.stop_reason, StopReason::ToolCalls);
    let call = first.tool_calls().next().cloned().unwrap_or_else(|| panic!("no tool call"));
    assert_eq!(call.name, "add");
    #[derive(serde::Deserialize)]
    struct Args {
        a: i64,
        b: i64,
    }
    let args: Args = call.arguments().unwrap_or_else(|e| panic!("{e}"));
    history.push(first.message.clone());
    history.push(Message::tool_results([ToolResult::new(&call, (args.a + args.b).to_string())]));

    let second = client.generate(Request::new(MODEL).tool(add).messages(history).effort(Effort::Low).max_output_tokens(1024)).await.unwrap_or_else(|e| panic!("{e}"));
    report("tool answer", &second);
    assert_eq!(second.stop_reason, StopReason::Complete);
    assert!(second.text().contains('5'));
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn returns_structured_output() {
    let schema = json!({ "type": "object", "properties": { "answer": { "type": "integer" } }, "required": ["answer"] });
    let r = client().generate(small("What is 6 times 7?").output(OutputFormat::json_schema("answer", schema))).await.unwrap_or_else(|e| panic!("{e}"));
    report("structured", &r);
    #[derive(serde::Deserialize)]
    struct Answer {
        answer: i64,
    }
    assert_eq!(r.json::<Answer>().map(|a| a.answer).unwrap_or_else(|e| panic!("{e}")), 42);
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn sends_reasoning_back_on_the_next_turn() {
    let client = client();
    let ask = |history: Vec<Message>| Request::new(MODEL).messages(history).effort(Effort::Medium).reasoning_display(ReasoningDisplay::Summary).max_output_tokens(1024);
    let mut history = vec![Message::user("Is 91 a prime number? Think it through briefly, then answer yes or no.")];
    let first = client.generate(ask(history.clone())).await.unwrap_or_else(|e| panic!("{e}"));
    report("reasoning 1", &first);
    let reasoning = first.message.parts.iter().filter(|p| matches!(p, Part::Reasoning(_))).count();
    println!("[reasoning 1] {reasoning} reasoning part(s), summary: {:?}", first.reasoning());
    history.push(first.message.clone());
    history.push(Message::user("And 97? One word."));
    let second = client.generate(ask(history)).await.unwrap_or_else(|e| panic!("{e}"));
    report("reasoning 2", &second);
    assert_eq!(second.stop_reason, StopReason::Complete);
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn turns_reasoning_off() {
    let client = client();
    let haiku = client.generate(small("Say hi.").effort(Effort::None).max_output_tokens(256)).await.unwrap_or_else(|e| panic!("{e}"));
    report("haiku none", &haiku);
    assert!(haiku.warnings.is_empty());
    assert!(!haiku.message.parts.iter().any(|p| matches!(p, Part::Reasoning(_))));

    let sonnet = client.generate(small("Say hi.").model("anthropic/claude-sonnet-5-5").effort(Effort::None).max_output_tokens(256)).await.unwrap_or_else(|e| panic!("{e}"));
    report("sonnet none", &sonnet);
    assert_eq!(sonnet.stop_reason, StopReason::Complete);
    assert!(sonnet.warnings.is_empty());
}

#[tokio::test]
#[ignore = "calls the Anthropic API"]
async fn counts_tokens_and_lists_models() {
    let client = client();
    let tokens = client.count_tokens(&small("How many tokens is this?")).await.unwrap_or_else(|e| panic!("{e}"));
    println!("[count] {tokens}");
    assert!(tokens > 0 && tokens < 100);

    let models = client.models().await.unwrap_or_else(|e| panic!("{e}"));
    println!("[models] {}", models.iter().map(|m| m.id.to_string()).collect::<Vec<_>>().join(", "));
    let haiku = models.iter().find(|m| m.id.name() == "claude-haiku-5-5").unwrap_or_else(|| panic!("no haiku 5.5"));
    assert!(haiku.capabilities.efforts.contains(&Effort::None));
    assert_eq!(haiku.context_window, Some(1_000_000));
}
