# VeloxQuant for Rust

Memory intelligence and optimization for local AI.

| Crate | crates.io | docs.rs |
|-------|-----------|---------|
| [`veloxquant`](crates/veloxquant) | [![crates.io](https://img.shields.io/crates/v/veloxquant.svg)](https://crates.io/crates/veloxquant) | [![docs.rs](https://img.shields.io/docsrs/veloxquant)](https://docs.rs/veloxquant) |
| [`veloxquant-core`](crates/veloxquant-core) | [![crates.io](https://img.shields.io/crates/v/veloxquant-core.svg)](https://crates.io/crates/veloxquant-core) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-core)](https://docs.rs/veloxquant-core) |
| [`veloxquant-system`](crates/veloxquant-system) | [![crates.io](https://img.shields.io/crates/v/veloxquant-system.svg)](https://crates.io/crates/veloxquant-system) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-system)](https://docs.rs/veloxquant-system) |
| [`veloxquant-memory`](crates/veloxquant-memory) | [![crates.io](https://img.shields.io/crates/v/veloxquant-memory.svg)](https://crates.io/crates/veloxquant-memory) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-memory)](https://docs.rs/veloxquant-memory) |
| [`veloxquant-runtime`](crates/veloxquant-runtime) | [![crates.io](https://img.shields.io/crates/v/veloxquant-runtime.svg)](https://crates.io/crates/veloxquant-runtime) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-runtime)](https://docs.rs/veloxquant-runtime) |
| [`veloxquant-openai`](crates/veloxquant-openai) | [![crates.io](https://img.shields.io/crates/v/veloxquant-openai.svg)](https://crates.io/crates/veloxquant-openai) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-openai)](https://docs.rs/veloxquant-openai) |
| [`veloxquant-monitor`](crates/veloxquant-monitor) | [![crates.io](https://img.shields.io/crates/v/veloxquant-monitor.svg)](https://crates.io/crates/veloxquant-monitor) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-monitor)](https://docs.rs/veloxquant-monitor) |
| [`veloxquant-models`](crates/veloxquant-models) | [![crates.io](https://img.shields.io/crates/v/veloxquant-models.svg)](https://crates.io/crates/veloxquant-models) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-models)](https://docs.rs/veloxquant-models) |
| [`veloxquant-rig`](crates/veloxquant-rig) | [![crates.io](https://img.shields.io/crates/v/veloxquant-rig.svg)](https://crates.io/crates/veloxquant-rig) | [![docs.rs](https://img.shields.io/docsrs/veloxquant-rig)](https://docs.rs/veloxquant-rig) |
| [`veloxquant-cli`](crates/veloxquant-cli) (`vq`) | not published — see [prebuilt binaries](https://github.com/rajveer43/veloxquant-rs/releases) | — |

`veloxquant` helps Rust developers detect Apple Silicon hardware, estimate
LLM and KV-cache memory requirements, get VeloxQuant compression
recommendations, and talk to a local VeloxQuant (or other OpenAI-compatible)
inference runtime — all with an async-first, idiomatic Rust API.

This crate is part of the [VeloxQuant](https://github.com/rajveer43/VeloxQuant-MLX)
ecosystem, alongside VeloxQuant-MLX (Python), VeloxQuant Studio (macOS),
VeloxQuant VS Code, and SDKs for [Go](https://github.com/rajveer43/veloxquant-go)
and [TypeScript](https://github.com/rajveer43/veloxquant-sdk).

> **Status:** 0.3.0, prepared on `master` but not yet released (crates.io
> still serves 0.2.x; see [Releasing](#releasing)). Hardware
> detection, memory/KV-cache estimation, optimization recommendations,
> the model registry, chat (non-streamed and SSE-streamed), model
> listing, Agent tool calling, MCP tool sources, benchmarking, the `rig`
> adapter, AutoPilot, and live periodic metrics sampling are implemented
> and tested. Native Rust KV-cache compression is still an
> [open issue](https://github.com/rajveer43/veloxquant-rs/issues). See
> [Roadmap](#roadmap).

## Installation

```toml
[dependencies]
veloxquant = "0.3"
```

Until 0.3.0 is on crates.io, depend on the git repository instead
(`veloxquant = { git = "https://github.com/rajveer43/veloxquant-rs" }`).
0.2.1 is only partially published and should not be used (see the
[CHANGELOG](CHANGELOG.md)).

MSRV: Rust 1.90 (edition 2021). This floor comes from the dependency tree
(`ordered-float` 5.5 via `rig-core`, `rmcp` 3.2, the ICU crates behind
`url`), not from the SDK's own code; CI builds and tests it against the
committed `Cargo.lock`.

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

On macOS, `available_memory_bytes` is free + inactive memory read from Mach
`host_statistics64`, the same figure the Swift and Go SDKs report. It is
not `sysinfo`'s, which subtracts compressed pages and can report close to
zero on a Mac with gigabytes free. Other platforms use `sysinfo`.

## Error handling

Fallible calls return `veloxquant::VeloxQuantError`. It is
`#[non_exhaustive]` from 0.3.0, so a `match` on it needs a wildcard arm.
New variants can then be added in minor releases without breaking your
build:

```rust
use veloxquant::VeloxQuantError;

fn hint(err: &VeloxQuantError) -> &'static str {
    match err {
        VeloxQuantError::RuntimeUnavailable => "start the runtime first",
        VeloxQuantError::InsufficientMemory => "try a smaller model",
        _ => "see the error message",
    }
}
```

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

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
use veloxquant::{AutoPilotConfig, Client, Task};

let client = Client::builder().build()?;
let session = client
    .autopilot(AutoPilotConfig { task: Some(Task::Chat), ..Default::default() })
    .await?;

for decision in &session.plan().decisions {
    println!("{decision}");
}
let reply = session.chat("Hello!").await?;
# Ok(())
# }
```

This needs the `autopilot` feature (which implies `openai`), the
`veloxquant` CLI from VeloxQuant-MLX on `PATH`, and an Apple Silicon Mac.
AutoPilot follows Go's `Client.AutoPilot` for hardware inspection, model
selection (the curated registry ranked by `ModelRegistry::recommend_scored`,
or a pinned model), the default 8192-token context, and the 15% safety
margin. **The compression strategy is never computed in Rust.** It comes
from the real CLI: `recommend --json` picks the method, a won't-fit warning
stops the run unless `force: true` is set, `methods --servable-only`
confirms `serve` can run the method, and `auto-config --json` is the
fallback when it can't. `session.plan()` records every decision.

AutoPilot does not launch the runtime. Serve the planned model with
`plan.method` and `plan.bits` yourself. Use `AutoPilot::new(client)` with
`.with_cli(VeloxQuantCli::python_module("/path/to/python"))` to point at a
specific install, or `.with_hardware(...)` to plan for a different Mac.
Use `try_start` to receive "won't fit" as data instead of an error. See
`examples/autopilot.rs`.

## Streaming

```rust,no_run
use futures_util::StreamExt;
use veloxquant::{Client, Message};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::builder().build()?;
let mut stream = client
    .chat()?
    .stream("Qwen3-8B", vec![Message::user("Hello!")])
    .await?;

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    print!("{}", chunk.text);
}
# Ok(())
# }
```

Parses SSE incrementally (no full-response buffering) and cancels the
underlying connection if the stream is dropped before completion. See
`examples/streaming.rs`.

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
OpenAI request/response shape, both non-streamed (`create`) and
SSE-streamed (`stream`). `Client::list_models` calls `GET /v1/models` and
merges the result with the curated registry.

## Agent (tool calling)

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
use serde_json::{json, Value};
use veloxquant::{Agent, AgentRunOptions, Client, Result, Tool};

struct EchoTool;

#[async_trait::async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn parameters(&self) -> Value { json!({"type": "object"}) }
    async fn execute(&self, args: Value) -> Result<Value> { Ok(args) }
}

let client = Client::builder().build()?;
let mut agent = Agent::new(client, "mlx-community/Qwen3-8B-4bit");
agent.tool(Box::new(EchoTool))?;

let result = agent.run("say hi via the echo tool", AgentRunOptions::default()).await?;
println!("{}", result.text);
# Ok(())
# }
```

Behind the `agent` feature (implies `openai`). `Agent::run` drives the same
call-tools -> feed-results-back -> repeat loop as `@veloxquant/sdk`'s
`agent.ts`: a malformed tool-call-arguments JSON, an unregistered tool name,
or a failing tool execution all feed a structured error back as the tool
result rather than aborting the run; exceeding `max_steps` (default 8)
returns `VeloxQuantError::AgentMaxStepsExceeded`. See `examples/agent.rs`.

## MCP tool sources

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
use veloxquant::{Agent, Client, McpServerConfig, McpTransport};

let client = Client::builder().build()?;
let mut agent = Agent::new(client, "mlx-community/Qwen3-8B-4bit");

agent.use_mcp_server(McpServerConfig {
    name: "my-server".to_string(),
    transport: McpTransport::Stdio {
        command: "my-mcp-server".to_string(),
        args: vec![],
        env: vec![],
    },
}).await?;
# Ok(())
# }
```

Behind the `mcp` feature (implies `agent`), built on the official `rmcp`
crate. `Agent::use_mcp_server` registers an MCP server's tools alongside any
manually-registered ones, sharing one name/dispatch namespace; a tool-name
collision closes the newly-opened connection before returning an error, so
a failed call never leaks a connection. Unsupported MCP content types
(image/audio/resource/resource_link) surface as
`VeloxQuantError::UnsupportedMcpContent` rather than being silently
dropped — see `examples/mcp_agent.rs`.

## Benchmarking

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
use veloxquant::{benchmark_pass, BenchmarkInput, Client};

let client = Client::builder().build()?;
let pass = benchmark_pass(&client, BenchmarkInput::new("mlx-community/Qwen3-8B-4bit")).await?;
println!("{:.1} tok/s, {:.0}ms TTFT", pass.timing.tokens_per_second, pass.timing.time_to_first_token_ms);
# Ok(())
# }
```

Behind the `openai` feature. `benchmark_pass` times a single generation
against an already-reachable runtime at the SDK boundary (chunk arrival
timestamps from the streaming chat API); `benchmark` runs two such passes
sequentially (never concurrently) and `BenchmarkResult::to_markdown()`
renders a report. Resident-memory (RSS) sampling is optional and PID-driven
(`BenchmarkInput::pid`) — this SDK talks to an already-running runtime over
HTTP and has no process-ownership concept to discover a PID from, unlike
`@veloxquant/sdk`'s `benchmark()`, which spawns and owns the runtime
process itself; see the doc comment on `veloxquant::benchmark` for the full
rationale. `vq benchmark <model>` uses this under the hood.

## `rig` integration

`crates/veloxquant-rig` (published independently, with its own crate
version, currently 0.1.0, not tied to the workspace's 0.3.0) adapts a `veloxquant::Client`
to [`rig-core`](https://crates.io/crates/rig-core)'s `CompletionModel`
trait, so a local VeloxQuant runtime can be used as the completion backend
in a `rig` pipeline/agent — mirroring the *shape* of Go's `langchain`
adapter (a separate module so `rig-core` stays an opt-in dependency, never
pulled into the core `veloxquant` facade crate), not a literal port, since
`rig` has no equivalent of `langchaingo`'s single `llms.Model` interface.

```rust,no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
use rig_core::completion::CompletionModel;
use veloxquant::Client;
use veloxquant_rig::VeloxQuantCompletionModel;

let client = Client::builder().build()?;
let model = VeloxQuantCompletionModel::new(client, "mlx-community/Qwen3-8B-4bit");

let request = model.completion_request("Hello!").build();
let response = model.completion(request).await?;
# Ok(())
# }
```

Text-only: any non-text `rig` message content (images, audio, documents,
tool calls/results, reasoning blocks) is rejected with an explicit
`RigAdapterError::UnsupportedContent` rather than silently dropped, matching
the VeloxQuant runtime's own text-only chat API and the Go adapter's stated
behavior. `stream()` reuses the existing SSE transport
(`veloxquant_openai::stream_chat_completions`) rather than a second SSE
parser. See `crates/veloxquant-rig/examples/rig_integration.rs` (requires a
running VeloxQuant runtime — not CI-verifiable, same honesty standard as the
benchmark phase's hardware-dependent tests).

## Monitoring

`Monitor`/`Metrics` (behind the `monitor` feature) provide
publish/subscribe over `tokio::sync::broadcast`, and
`Monitor::spawn_sampler` drives them periodically. It polls a `Sampler`
on an interval and publishes each sample on that same channel.
`SystemSampler` samples host memory used/available. Inference-side fields
(tokens/sec, TTFT, KV-cache bytes) are left at their defaults for you to
publish yourself, or for a custom `Sampler` to fill in.

```rust,no_run
use std::time::Duration;
use veloxquant_monitor::{Metrics, Monitor, SystemSampler};

# async fn run() {
let monitor = Monitor::new();
let mut receiver = monitor.subscribe();
let sampling = monitor.spawn_sampler(SystemSampler::new(), Duration::from_secs(1));

// Out-of-band samples can still be published by hand between ticks.
monitor.publish(Metrics { tokens_per_second: 42.0, ..Default::default() });

if let Ok(metrics) = receiver.recv().await {
    println!("Memory used: {} bytes", metrics.memory_used_bytes);
}
sampling.stop().await; // dropping the handle also stops sampling
# }
```

Behaves like Go's monitor. The first sample is taken immediately, then one
per interval. A zero interval means 5 s. A sampler error skips that tick
only. See `examples/monitor.rs`.

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
vq models list         List models in the local Hugging Face cache
vq models pull <id>    Download a model's weights into the local Hugging Face cache
vq models delete <id>  Delete a model's weights from the local Hugging Face cache
```

### Local model cache management

`vq models list/pull/delete` (and the underlying
`veloxquant::list_local_models`/`pull_local_model`/`delete_local_model`,
behind the `local-models` feature) manage weights in the local Hugging Face
cache by shelling out to a short Python snippet that uses
`huggingface_hub`'s own `scan_cache_dir()`/`snapshot_download()`/
`delete_revisions()` — the cache's content-addressed blob layout is owned
by `huggingface_hub`, so this SDK doesn't reimplement it. Requires a Python
interpreter with `huggingface_hub` importable (`PythonInterpreter::default()`
uses `python3` on `$PATH`; construct `PythonInterpreter::new(path)` to point
at a specific interpreter). This is distinct from `client.models()`, which
lists the curated compression-method registry, not downloaded weights.

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
├── veloxquant-monitor       metrics types, broadcast-based pub/sub, periodic sampling
├── veloxquant-models        local Hugging Face model cache management (list/pull/delete)
└── veloxquant-cli (vq)      command-line interface

veloxquant-rig               rig-core CompletionModel adapter (independent crate/version, opt-in)
```

Feature flags on the `veloxquant` crate let you opt out of what you don't need:

```toml
[dependencies]
veloxquant = { version = "0.3", default-features = false, features = ["runtime"] }
```

| Feature             | Default | Enables                                   |
|---------------------|---------|--------------------------------------------|
| `runtime`           | ✓       | `Client::runtime()` (health checks)        |
| `openai`            | ✓       | `Client::chat()` (implies `runtime`)       |
| `monitor`           |         | `Monitor`/`Metrics`/`SystemSampler` re-exports (live sampling) |
| `local-models`      |         | `list_local_models`/`pull_local_model`/`delete_local_model` (local Hugging Face cache management) |
| `agent`             |         | `Agent`/`Tool` tool-calling loop (implies `openai`) |
| `mcp`               |         | `Agent::use_mcp_server` and MCP tool sources (implies `agent`) |
| `autopilot`         |         | `Client::autopilot`/`AutoPilot` (implies `openai`; needs the `veloxquant` CLI) |
| `native-optimizers` |         | Reserved for native Rust compression (v0.5.0+); no implementation ships yet |

## Roadmap

- **v0.1.0** — workspace, hardware detection, memory/KV-cache
  estimation, optimization profiles, runtime health check, non-streamed
  chat, curated model registry, `vq doctor`/`analyze`/`recommend`/`benchmark`/`serve`, tests.
- **v0.2.0** — SSE streaming chat completions.
- **v0.2.1** — `GET /v1/models` model listing.
- **v0.3.0** (prepared, not yet released) — SDK parity with Go/TS: local
  model cache, Agent tool calling, MCP tool sources, `benchmark()`, the
  `rig` adapter, AutoPilot (CLI-backed compression choice), live periodic
  metrics sampling, correct macOS available memory, a `#[non_exhaustive]`
  `VeloxQuantError`, and release automation. Releases are gated on the
  full CI workflow and publish to crates.io in dependency order,
  idempotently. A minor bump because it is semver-breaking. See
  [CHANGELOG](CHANGELOG.md).
- **v0.5.0** — Investigate native Rust KV-cache compression (TurboQuant, RVQ, VecInfer, RateQuant, PolarQuant, QJL).
- **v1.0.0** — Stable API, SemVer guarantees.

### Releasing

Run the **Bump version** workflow (or `scripts/bump-version.sh <x.y.z>` and
push a `v<x.y.z>` tag). The **Release** workflow runs the full CI workflow,
checks that the tag matches the workspace version, builds the `vq`
binaries, and creates the GitHub release. If the `PUBLISH_TO_CRATES_IO`
repo variable is `true` and `CARGO_REGISTRY_TOKEN` is set, it also
publishes to crates.io via `scripts/publish-crates.sh`, which derives the
order from `cargo metadata` and skips versions already published. Run
`scripts/publish-crates.sh --plan` to preview a release.

The workspace manifests are already at 0.3.0. To release it, run **Bump
version** with `0.3.0`: the script sees the manifests already match, only
stamps the changelog, then tags `v0.3.0` and starts the release. The plan
publishes all nine publishable crates, including `veloxquant-models` and
`veloxquant-rig`, which have never been on crates.io.

Tracked as a GitHub Epic and per-feature issues:
<https://github.com/rajveer43/veloxquant-rs/issues>.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
