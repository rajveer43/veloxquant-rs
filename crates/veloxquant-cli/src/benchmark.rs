use std::time::Instant;

use anyhow::{anyhow, Result};
use veloxquant::{format_bytes, Client, MemoryRequest, Message, ModelArchitecture, Precision};

pub async fn run(model: &str, context: usize, prompt: &str) -> Result<()> {
    let client = Client::builder().build()?;

    let status = client.runtime().health().await;
    if status.is_err() || !status.map(|s| s.healthy).unwrap_or(false) {
        return Err(anyhow!(
            "veloxquant runtime unavailable; start the runtime to benchmark inference (see `vq serve`)"
        ));
    }

    let architecture = resolve_architecture(&client, model);
    let estimate =
        client
            .memory()
            .estimate(&MemoryRequest::new(architecture, context, Precision::Int4))?;

    let start = Instant::now();
    let response = client
        .chat()?
        .create(model, vec![Message::user(prompt)])
        .await?;
    let elapsed = start.elapsed();

    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 && response.usage.completion_tokens > 0 {
        response.usage.completion_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("VeloxQuant Benchmark\n");
    println!("Model:\n{model}\n");
    println!("Tokens/sec:\n{tokens_per_sec:.1}\n");
    println!("Total Duration:\n{:.0?}\n", elapsed);
    println!(
        "Peak Memory (estimated):\n{}\n",
        format_bytes(estimate.total_memory_bytes)
    );
    println!(
        "KV Cache (estimated):\n{}\n",
        format_bytes(estimate.kv_cache_memory_bytes)
    );
    println!("Compression Ratio:\n{:.1}% saved", estimate.saved_percent);

    Ok(())
}

fn resolve_architecture(client: &Client, model: &str) -> ModelArchitecture {
    client
        .models()
        .list()
        .into_iter()
        .find(|m| m.name == model)
        .map(|m| m.architecture)
        .unwrap_or(ModelArchitecture {
            name: model.to_string(),
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            parameter_count: 7_000_000_000,
        })
}
