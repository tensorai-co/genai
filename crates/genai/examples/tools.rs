//! A tool loop: the model calls `add` until it can answer.
//!
//! ```sh
//! ANTHROPIC_API_KEY=… cargo run --example tools
//! ```

use genai::{Client, Effort, Message, Request, StopReason, Tool, ToolResult};
use serde_json::json;

#[tokio::main]
async fn main() -> genai::Result<()> {
    let client = Client::from_env()?;
    let add = Tool::function("add").description("Adds two integers.").parameters(json!({
        "type": "object",
        "properties": { "a": { "type": "integer" }, "b": { "type": "integer" } },
        "required": ["a", "b"]
    }));

    let mut history = vec![Message::user("What is (2 + 3) + 4? Use the add tool for each step.")];
    loop {
        let request = Request::new("anthropic/claude-haiku-5-5").tool(add.clone()).messages(history.clone()).effort(Effort::Low).max_output_tokens(1024);
        let response = client.generate(request).await?;
        history.push(response.message.clone());
        if response.stop_reason != StopReason::ToolCalls {
            println!("{}", response.text());
            return Ok(());
        }
        let results = response
            .tool_calls()
            .map(|call| {
                let sum = call.arguments["a"].as_i64().unwrap_or(0) + call.arguments["b"].as_i64().unwrap_or(0);
                println!("add({}, {}) = {sum}", call.arguments["a"], call.arguments["b"]);
                ToolResult::new(call, sum.to_string())
            })
            .collect::<Vec<_>>();
        history.push(Message::tool_results(results));
    }
}
