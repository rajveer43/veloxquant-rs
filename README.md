# VeloxQuant for Rust

Memory intelligence and optimization for local AI.

`veloxquant` helps Rust developers detect Apple Silicon hardware, estimate
LLM and KV-cache memory requirements, get VeloxQuant compression
recommendations, and talk to a local VeloxQuant (or other OpenAI-compatible)
inference runtime — all with an async-first, idiomatic Rust API.

This crate is part of the [VeloxQuant](https://github.com/veloxquant)
ecosystem, alongside VeloxQuant-MLX (Python), VeloxQuant Studio (macOS),
VeloxQuant VS Code, and SDKs for [Go](https://github.com/rajveer43/veloxquant-go)
and [TypeScript](https://github.com/rajveer43/veloxquant-sdk).

> **Status:** v0.1.0. Hardware detection, memory/KV-cache estimation,
> optimization recommendations, the model registry, and non-streamed chat
> are implemented and tested. Streaming, AutoPilot, live monitoring, and
> native compression are tracked as [open issues](https://github.com/rajveer43/veloxquant-rs/issues) —
> see [Roadmap](#roadmap).

## Installation

```toml
[dependencies]
veloxquant = "0.1"
```

MSRV: Rust 1.75 (edition 2021).

## Quick Start

```rust,no_run
use veloxquant::{Client, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().auto_detect().build()?;

    let response = client
        .chat()?
        .create("Qwen3-8B", vec![Message::user("Hello!")])
        .await?;

    println!("{}", response.text);

    Ok(())
}
```

Requires a running OpenAI-compatible runtime (e.g. a local VeloxQuant
runtime) at `http://localhost:8765` by default — see
[`ClientBuilder::runtime_url`](#openai-compatibility). Hardware detection
and memory estimation (below) work without any runtime running.

## Hardware Detection

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = veloxquant::Client::builder().build()?;
let info = client.system().info().await;

println!("Platform: {}", info.platform);
println!("Apple Silicon: {}", info.apple_silicon);
println!("Total Memory: {}", veloxquant::format_bytes(info.total_memory_bytes));
println!("Recommended Profile: {}", info.recommended_profile);
# Ok(())
# }
```

Detection never panics on unsupported hardware: fields degrade to empty
strings or `0` rather than erroring.

## Memory Estimation

```rust
use veloxquant::{MemoryRequest, ModelArchitecture, Precision};

let model = ModelArchitecture {
    name: "Qwen3-8B".to_string(),
    num_layers: 36,
    num_kv_heads: 8,
    head_dim: 128,
    hidden_size: 4096,
    parameter_count: 8_000_000_000,
};

let client = veloxquant::Client::builder().build().unwrap();
let estimate = client
    .memory()
    .estimate(&MemoryRequest::new(model, 32_768, Precision::Fp16))
    .unwrap();

println!("Total (unoptimized): {}", veloxquant::format_bytes(estimate.total_memory_bytes));
println!("Total (VeloxQuant):  {}", veloxquant::format_bytes(estimate.optimized_total_bytes));
println!("Saved: {:.1}%", estimate.saved_percent);
```

## KV-Cache Analysis

KV-cache memory is computed with the standard formula:

```text
KV Cache Memory = Layers × Tokens × KV Heads × Head Dim × 2 × Bytes Per Element
```

The `2` accounts for storing both keys and values. This is exposed directly
via `veloxquant_memory::estimate_kv_cache_bytes` for callers who only need
the cache component. `ModelArchitecture` is deliberately generic — it isn't
hardcoded to one model family, so Llama, Qwen, Gemma, Mistral, DeepSeek, or
a custom architecture can all populate it.

## Optimization Profiles

```rust
use veloxquant::OptimizationProfile;
```

| Profile          | Prioritizes                                   |
|------------------|------------------------------------------------|
| `Speed`          | Throughput, lower compression overhead          |
| `Balanced`        | Memory savings + good inference speed (default) |
| `Memory`         | Maximum memory reduction                        |
| `MaximumContext` | Long context windows, aggressive KV compression |

```rust
use veloxquant::{OptimizationRequest, ModelArchitecture};
# let model = ModelArchitecture { name: "m".into(), num_layers: 32, num_kv_heads: 8, head_dim: 128, hidden_size: 4096, parameter_count: 7_000_000_000 };

let client = veloxquant::Client::builder().build().unwrap();
let recommendation = client.optimize().recommend(&OptimizationRequest {
    model,
    context_length: 32_768,
    available_memory_bytes: 16 * 1024 * 1024 * 1024,
    profile: None, // let VeloxQuant choose based on available memory
}).unwrap();

println!("{}: {}", recommendation.profile, recommendation.reason);
```

## AutoPilot

Not yet implemented — see [Roadmap](#roadmap) (targeted for v0.3.0). Today,
combine `client.system()`, `client.memory()`, and `client.optimize()`
manually to get the same information AutoPilot will automate.

## Streaming

Not yet implemented — see [Roadmap](#roadmap) (targeted for v0.2.0).
`client.chat()?.create(...)` returns a complete (non-streamed) response
today.

## OpenAI Compatibility

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = veloxquant::Client::builder()
    .openai_compatible("http://localhost:8765/v1")
    .build()?;
# Ok(())
# }
```

`Client::chat` posts to `{base_url}/v1/chat/completions` in the standard
OpenAI request/response shape. `GET /v1/models` and streaming are planned
for v0.2.0.

## Monitoring

`Monitor`/`Metrics` (behind the `monitor` feature) provide working
publish/subscribe plumbing today via `tokio::sync::broadcast`; automatic
periodic sampling of live memory/inference metrics is planned for v0.3.0.

```rust
use veloxquant_monitor::{Monitor, Metrics};

# async fn run() {
let monitor = Monitor::new();
let mut receiver = monitor.subscribe();

monitor.publish(Metrics { memory_used_bytes: 1024, ..Default::default() });

if let Ok(metrics) = receiver.recv().await {
    println!("Memory: {} bytes", metrics.memory_used_bytes);
}
# }
```

## CLI

```sh
cargo install --path crates/veloxquant-cli
```

```text
vq doctor              Check system readiness for local AI
vq analyze <model>     Analyze memory requirements for a model
vq recommend           Recommend models and a profile for this hardware
vq benchmark <model>   Benchmark inference performance (requires a running runtime)
vq serve               Connect to (or report on) the VeloxQuant runtime
```

```text
$ vq doctor
VeloxQuant Doctor

✓ Rust runtime available
✓ Apple Silicon detected
✓ 24.0 GB Unified Memory
✓ VeloxQuant runtime reachable

System ready.
```

## Architecture

A Cargo workspace of small, focused crates, re-exported through the
top-level `veloxquant` facade crate (the one you actually depend on):

```text
veloxquant                  facade crate: Client, ClientBuilder, re-exports
├── veloxquant-core          error types, config, shared HTTP client construction
├── veloxquant-system        hardware/platform/memory detection
├── veloxquant-memory        model + KV-cache memory estimation, optimization recommendations
├── veloxquant-runtime       async client for a VeloxQuant runtime (health checks today)
├── veloxquant-openai        OpenAI-compatible wire types (chat request/response, models)
├── veloxquant-monitor       metrics types + broadcast-based pub/sub
└── veloxquant-cli (vq)      command-line interface
```

Feature flags on the `veloxquant` crate let you opt out of what you don't need:

```toml
[dependencies]
veloxquant = { version = "0.1", default-features = false, features = ["runtime"] }
```

| Feature             | Default | Enables                                   |
|---------------------|---------|--------------------------------------------|
| `runtime`           | ✓       | `Client::runtime()` (health checks)        |
| `openai`            | ✓       | `Client::chat()` (implies `runtime`)       |
| `monitor`           |         | `Monitor`/`Metrics` re-exports             |
| `native-optimizers` |         | Reserved for native Rust compression (v0.5.0+); no implementation ships yet |

## Roadmap

- **v0.1.0** (this release) — workspace, hardware detection, memory/KV-cache
  estimation, optimization profiles, runtime health check, non-streamed
  chat, curated model registry, `vq doctor`/`analyze`/`recommend`/`benchmark`/`serve`, tests.
- **v0.2.0** — SSE streaming chat, `GET /v1/models`, richer model registry.
- **v0.3.0** — Live monitoring/sampling, benchmarking polish, AutoPilot, advanced model recommendation.
- **v0.5.0** — Investigate native Rust KV-cache compression (TurboQuant, RVQ, VecInfer, RateQuant, PolarQuant, QJL).
- **v1.0.0** — Stable API, SemVer guarantees, full CI/release automation.

Tracked as a GitHub Epic and per-feature issues:
<https://github.com/rajveer43/veloxquant-rs/issues>.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
