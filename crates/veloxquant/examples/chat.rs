//! Minimal chat example.
//!
//! Requires a VeloxQuant (or other OpenAI-compatible) runtime at
//! `http://localhost:8765`. Run with:
//!
//! ```sh
//! cargo run --example chat
//! ```

use veloxquant::{Client, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().auto_detect().build()?;

    let response = client
        .chat()?
        .create(
            "mlx-community/Qwen3-8B-4bit",
            vec![Message::user("Explain KV cache in simple terms.")],
        )
        .await?;

    println!("{}", response.text);

    Ok(())
}
