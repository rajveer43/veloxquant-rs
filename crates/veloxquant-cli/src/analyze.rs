use anyhow::Result;
use veloxquant::{format_bytes, Client, MemoryRequest, ModelArchitecture, Precision};

pub async fn run(model: &str, context: usize) -> Result<()> {
    let client = Client::builder().build()?;

    let architecture = resolve_architecture(&client, model);

    let estimate =
        client
            .memory()
            .estimate(&MemoryRequest::new(architecture, context, Precision::Fp16))?;

    println!("Model Analysis\n");
    println!("Model:\n{model}\n");
    println!(
        "Model Memory:\n{}\n",
        format_bytes(estimate.model_memory_bytes)
    );
    println!(
        "KV Cache at {context} tokens:\n{}\n",
        format_bytes(estimate.kv_cache_memory_bytes)
    );
    println!(
        "Without Optimization:\n{}\n",
        format_bytes(estimate.total_memory_bytes)
    );
    println!(
        "With VeloxQuant:\n{}\n",
        format_bytes(estimate.optimized_total_bytes)
    );
    println!(
        "Memory Saved:\n{} ({:.1}%)",
        format_bytes(estimate.saved_bytes),
        estimate.saved_percent
    );

    Ok(())
}

/// Looks up `model` in the curated registry; falls back to a representative
/// 7-8B-class architecture so `vq analyze` remains useful for models
/// outside the registry.
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
