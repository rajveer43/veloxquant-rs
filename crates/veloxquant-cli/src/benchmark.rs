use anyhow::{anyhow, Result};
use veloxquant::{benchmark_pass, format_bytes, BenchmarkInput, Client, MemoryRequest, ModelArchitecture, Precision};

/// `vq benchmark <model>` — benchmarks tokens/sec and time-to-first-token
/// for `model` against a running VeloxQuant runtime, via the reusable
/// `veloxquant::benchmark_pass` library function (see
/// `crates/veloxquant/src/benchmark.rs`). Also reports estimated peak
/// memory/KV-cache/compression ratio for context, matching this command's
/// previous output shape.
///
/// This is a single-pass report (one model, whatever method the runtime
/// happens to be currently serving) — `veloxquant::benchmark`'s full
/// two-pass default-vs-optimized comparison isn't wired into the CLI,
/// since this SDK has no way to make the runtime switch methods between
/// passes (see `benchmark.rs`'s module doc comment for why); a caller
/// wanting that comparison should call `veloxquant::benchmark` directly,
/// restarting the runtime with a different method between calls.
pub async fn run(model: &str, context: usize, _prompt: &str) -> Result<()> {
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

    let pass = benchmark_pass(&client, BenchmarkInput::new(model))
        .await
        .map_err(|e| anyhow!("benchmark failed: {e}"))?;

    println!("VeloxQuant Benchmark\n");
    println!("Model:\n{model}\n");
    println!("Tokens/sec:\n{:.1}\n", pass.timing.tokens_per_second);
    println!("TTFT:\n{:.0}ms\n", pass.timing.time_to_first_token_ms);
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
