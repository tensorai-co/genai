//! Live checks against the real OpenAI API. Ignored by default; they need a key and cost a fraction of a cent.
//!
//! ```sh
//! OPENAI_API_KEY=… cargo test -p genai --test openai_live -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! Every request is tiny on purpose: a one-line prompt, low effort, and at most 1024 output tokens.
#![cfg(feature = "openai")]

use futures_util::StreamExt;
use genai::{Client, Effort, Event, Message, OutputFormat, Part, ReasoningDisplay, Request, Response, StopReason, Tool, ToolResult};
use serde_json::json;

const MODEL: &str = "openai/gpt-6-luna";

fn client() -> Client {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| panic!("set OPENAI_API_KEY"));
    Client::builder().openai(key).build().unwrap_or_else(|e| panic!("{e}"))
}

fn small(prompt: &str) -> Request {
    Request::new(MODEL).message(Message::user(prompt)).effort(Effort::Low).max_output_tokens(1024)
}

fn report(name: &str, r: &Response) {
    println!("[{name}] {:?} in={} out={} reasoning={:?} warnings={:?} text={:?}", r.stop_reason, r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens, r.warnings, r.text());
}

#[tokio::test]
#[ignore = "calls the OpenAI API"]
async fn generates_text() {
    let r = client().generate(small("Reply with exactly one word: pong").instructions("You answer tersely.")).await.unwrap_or_else(|e| panic!("{e}"));
    report("text", &r);
    assert_eq!(r.stop_reason, StopReason::Complete);
    assert!(r.text().to_lowercase().contains("pong"));
    assert!(r.usage.input_tokens > 0 && r.usage.output_tokens > 0);
}

#[tokio::test]
#[ignore = "calls the OpenAI API"]
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
    assert!(deltas > 0 && text.contains('5'));
    assert!(matches!(finished, Some((StopReason::Complete, _))));
}

#[tokio::test]
#[ignore = "calls the OpenAI API"]
async fn runs_a_tool_loop() {
    let client = client();
    let add = Tool::function("add").description("Adds two integers.").parameters(json!({
        "type": "object",
        "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
        "required": ["a", "b"]
    }));
    let ask = |history: Vec<Message>| Request::new(MODEL).tool(add.clone()).messages(history).effort(Effort::Low).max_output_tokens(1024);
    let mut history = vec![Message::user("What is 2 + 3? Use the add tool, then answer with just the number.")];
    let first = client.generate(ask(history.clone())).await.unwrap_or_else(|e| panic!("{e}"));
    report("tool call", &first);
    assert_eq!(first.stop_reason, StopReason::ToolCalls);
    let call = first.tool_calls().next().cloned().unwrap_or_else(|| panic!("no tool call"));
    assert_eq!(call.name, "add");
    let sum = call.arguments["a"].as_i64().unwrap_or(0) + call.arguments["b"].as_i64().unwrap_or(0);
    history.push(first.message.clone());
    history.push(Message::tool_results([ToolResult::new(&call, sum.to_string())]));

    let second = client.generate(ask(history)).await.unwrap_or_else(|e| panic!("{e}"));
    report("tool answer", &second);
    assert_eq!(second.stop_reason, StopReason::Complete);
    assert!(second.text().contains('5'));
}

#[tokio::test]
#[ignore = "calls the OpenAI API"]
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
#[ignore = "calls the OpenAI API"]
async fn sends_reasoning_back_on_the_next_turn() {
    let client = client();
    let ask = |history: Vec<Message>| Request::new(MODEL).messages(history).effort(Effort::Medium).reasoning_display(ReasoningDisplay::Summary).max_output_tokens(1024);
    let mut history = vec![Message::user("Is 91 a prime number? Answer yes or no with one short reason.")];
    let first = client.generate(ask(history.clone())).await.unwrap_or_else(|e| panic!("{e}"));
    report("reasoning 1", &first);
    let encrypted = first.message.parts.iter().filter(|p| matches!(p, Part::Reasoning(r) if r.opaque.as_ref().is_some_and(|o| o.data.get("encrypted_content").is_some()))).count();
    println!("[reasoning 1] {encrypted} encrypted reasoning item(s), summary: {:?}", first.reasoning());
    history.push(first.message.clone());
    history.push(Message::user("And 97? One word."));
    let second = client.generate(ask(history)).await.unwrap_or_else(|e| panic!("{e}"));
    report("reasoning 2", &second);
    assert_eq!(second.stop_reason, StopReason::Complete);
}

#[tokio::test]
#[ignore = "calls the OpenAI API"]
async fn handles_model_differences() {
    let client = client();
    let off = client.generate(small("Say hi.").effort(Effort::None).max_output_tokens(256)).await.unwrap_or_else(|e| panic!("{e}"));
    report("luna none", &off);
    assert!(off.warnings.is_empty());
    assert_eq!(off.usage.reasoning_tokens, Some(0));

    // GPT-6.2 Sol rejects temperature and `none` effort; both are approximated instead of failing.
    let sol = client.generate(small("Say hi.").model("openai/gpt-6.2-sol").temperature(0.2).effort(Effort::None).max_output_tokens(256)).await.unwrap_or_else(|e| panic!("{e}"));
    report("sol", &sol);
    assert_eq!(sol.stop_reason, StopReason::Complete);
    let codes: Vec<&str> = sol.warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, vec!["reasoning_always_on", "temperature_dropped"]);
}

#[tokio::test]
#[ignore = "calls the OpenAI API"]
async fn lists_models() {
    let models = client().models().await.unwrap_or_else(|e| panic!("{e}"));
    println!("[models] {}", models.iter().map(|m| m.id.to_string()).collect::<Vec<_>>().join(", "));
    let sol = models.iter().find(|m| m.id.name() == "gpt-6.2-sol").unwrap_or_else(|| panic!("no gpt-6.2-sol"));
    assert!(sol.capabilities.efforts.contains(&Effort::Max) && !sol.capabilities.efforts.contains(&Effort::None));
    assert!(!models.iter().any(|m| m.id.name().contains("image") || m.id.name().contains("embedding")));
}
