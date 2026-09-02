//! Streaming chat example.
//!
//! Requires a VeloxQuant (or other OpenAI-compatible) runtime at
//! `http://localhost:8765`. Run with:
//!
//! ```sh
//! cargo run --example streaming --features openai
//! ```

use futures_util::StreamExt;
use veloxquant::{Client, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().auto_detect().build()?;

    let mut stream = client
        .chat()?
        .stream(
            "mlx-community/Qwen3-8B-4bit",
            vec![Message::user("Explain KV cache in simple terms.")],
        )
        .await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.text);
        if chunk.done {
            break;
        }
    }
    println!();

    Ok(())
}
