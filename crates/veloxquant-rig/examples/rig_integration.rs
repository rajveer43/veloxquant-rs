//! Runs a completion against a local VeloxQuant runtime through `rig-core`'s
//! [`CompletionModel`] trait.
//!
//! **Manual verification step** (not CI-verifiable — same honesty standard
//! as the `benchmark` phase's hardware-dependent tests): this example needs
//! a real, running OpenAI-compatible VeloxQuant runtime.
//!
//! ```sh
//! # In one terminal: start a runtime (or point at any OpenAI-compatible
//! # server) at http://localhost:8765.
//! vq serve
//!
//! # In another terminal:
//! cargo run -p veloxquant-rig --example rig_integration -- mlx-community/Qwen3-8B-4bit
//! ```

use rig_core::completion::CompletionModel;
use veloxquant::Client;
use veloxquant_rig::VeloxQuantCompletionModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "mlx-community/Qwen3-8B-4bit".to_string());

    let client = Client::builder().build()?;
    let model = VeloxQuantCompletionModel::new(client, model_name);

    let request = model
        .completion_request("Write a haiku about local inference.")
        .build();

    let response = model.completion(request).await?;
    for item in response.choice {
        if let rig_core::completion::AssistantContent::Text(text) = item {
            println!("{}", text.text);
        }
    }

    Ok(())
}
