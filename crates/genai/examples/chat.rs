//! A tiny chat that streams replies.
//!
//! ```sh
//! ANTHROPIC_API_KEY=… cargo run --example chat
//! ```

use std::io::{BufRead, Write};

use futures_util::StreamExt;
use genai::{Client, Effort, Event, Message, Request};

#[tokio::main]
async fn main() -> genai::Result<()> {
    let client = Client::from_env()?;
    let mut history = Vec::new();
    let stdin = std::io::stdin();
    print!("> ");
    std::io::stdout().flush().ok();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            break;
        }
        history.push(Message::user(line));
        let request = Request::new("anthropic/claude-haiku-5-5").instructions("Be brief.").messages(history.clone()).effort(Effort::Low).max_output_tokens(1024);

        let mut events = client.stream(request).await?;
        let mut reply = genai::Accumulator::new();
        while let Some(event) = events.next().await {
            let event = event?;
            if let Event::TextDelta { text, .. } = &event {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
            reply.push(&event);
        }
        let reply = reply.finish()?;
        println!("\n({} in, {} out)\n", reply.usage.input_tokens, reply.usage.output_tokens);
        history.push(reply.message);
        print!("> ");
        std::io::stdout().flush().ok();
    }
    Ok(())
}
