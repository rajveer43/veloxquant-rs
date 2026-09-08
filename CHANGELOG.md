# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Local Hugging Face model cache management (Phase 1 of SDK parity with
  `@veloxquant/sdk`): new `veloxquant-models` crate with
  `list_local_models`/`pull_local_model`/`delete_local_model`, shelling out
  to `huggingface_hub`'s `scan_cache_dir()`/`snapshot_download()`/
  `delete_revisions()` via a Python subprocess (model ids are always passed
  as a separate argv element, never interpolated into the Python source, as
  a command-injection mitigation). Re-exported from the `veloxquant` facade
  crate behind a new `local-models` feature. New `vq models list/pull/delete`
  CLI subcommands.
- `Agent` tool-calling loop (Phase 2 of SDK parity): `Agent::new`/`tool`/`run`
  in the `veloxquant` facade crate, gated behind a new `agent` feature. Adds
  `tools`/`tool_calls` wire types (`ToolDefinition`, `ToolCall`,
  `FunctionDefinition`, `FunctionCall`) to `veloxquant-openai`'s chat types
  and `ChatApi::send` for full-request chat calls. Matches
  `agent.ts:122-181`'s run loop exactly: malformed tool-call-arguments JSON,
  an unregistered tool name, or a failing tool execution all feed a
  structured error back as the tool result rather than aborting the run;
  exceeding `max_steps` (default 8) returns
  `VeloxQuantError::AgentMaxStepsExceeded`. New `examples/agent.rs`.
- MCP tool sources (Phase 3): `veloxquant::mcp` module (`McpTransport`,
  `McpServerConfig`, `McpToolSource`, `connect_mcp_server`,
  `unwrap_mcp_tool_result`) and `Agent::use_mcp_server`, gated behind a new
  `mcp` feature (depends on `agent`). Built on the official `rmcp` crate.
  Unsupported MCP content types (image/audio/resource/resource_link) return
  `VeloxQuantError::UnsupportedMcpContent` rather than being silently
  dropped. A tool-name collision when registering an MCP server's tools
  closes the newly-opened connection before returning an error. New
  `examples/mcp_agent.rs`.
- `benchmark()`/`benchmark_pass()` (Phase 4): reusable library functions in
  the facade crate (gated behind `openai`) measuring tokens/sec,
  time-to-first-token, and (optionally, given a PID) resident memory (RSS,
  via `ps -o rss= -p <pid>`) for a chat completion against an already
  reachable runtime. `BenchmarkResult::to_markdown()` preserves
  `benchmark.ts:103-106`'s "compression is accounting-only" caveat line
  verbatim when optimized resident memory measures *higher* than the
  default method's. Unlike `benchmark.ts`, this SDK has no
  process-ownership concept for the runtime (no `Client::load()` spawning a
  subprocess it can read a PID from), so `benchmark()` runs its two passes
  sequentially against whatever the runtime is currently serving rather
  than loading/reloading it itself — see the module doc comment in
  `crates/veloxquant/src/benchmark.rs` for the full rationale. `vq
  benchmark` now calls `benchmark_pass` under the hood instead of its
  previous inline timing logic. The real, hardware-dependent two-pass
  comparison is covered by `#[ignore]`d manual tests with instructions for
  running them by hand — not claimed as CI coverage.

## [0.2.1] - 2026-09-03

### Added

- OpenAI-compatible model listing: `RuntimeClient::list_models` calls
  `GET {base_url}/v1/models` and deserializes the standard `{"data": [...]}`
  envelope into `Vec<RemoteModel>`. `ModelRegistry::merge_remote` merges a
  live listing with the curated static registry — matching curated entries
  are kept as-is (richer `ModelArchitecture`/tasks win), unmatched remote
  models are appended as minimal, unsupported entries. `Client::list_models`
  wires the two together as a single async call.

## [0.2.0] - 2026-09-02

### Added

- True async SSE streaming for chat completions: `Client::chat().stream(...)`
  returns a `Stream<Item = Result<ChatChunk, VeloxQuantError>>`
  (`veloxquant_openai::stream_chat_completions`, `streaming::ChatStream`).
  Parses SSE incrementally (no full-response buffering), handles the
  `[DONE]` terminator, and cancels the underlying connection when the
  stream is dropped.
- `examples/streaming.rs`: a working streaming chat example.
- CI now runs against the `master` branch (previously configured for a
  `main` branch that doesn't exist in this repo) and a `scripts/bump-version.sh`
  helper plus a `Bump version` GitHub Actions workflow automate workspace
  version bumps.

## [0.1.0] - 2026-09-02

Initial release. See the [roadmap](https://github.com/rajveer43/veloxquant-rs/issues) for what's next.

### Added

- Workspace with `veloxquant-core`, `veloxquant-system`, `veloxquant-memory`,
  `veloxquant-runtime`, `veloxquant-openai`, `veloxquant-monitor`,
  `veloxquant-cli`, and the `veloxquant` facade crate.
- Hardware detection: platform, architecture, Apple Silicon detection, CPU
  model, total/available memory (`veloxquant_system::detect`).
- Model and KV-cache memory estimation using the standard
  `layers × tokens × kv_heads × head_dim × 2 × bytes_per_element` formula.
- Optimization profile recommendations (`Speed`, `Balanced`, `Memory`,
  `MaximumContext`) driven by available memory and model footprint.
- `RuntimeClient::health` for checking a local VeloxQuant runtime.
- A working (non-streaming) chat client (`Client::chat`) against an
  OpenAI-compatible `/v1/chat/completions` endpoint.
- A curated, offline static model registry with task- and memory-based
  recommendations.
- `Monitor`/`Metrics` pub-sub scaffolding backed by `tokio::sync::broadcast`.
- The `vq` CLI: `doctor`, `analyze`, `recommend`, `benchmark`, `serve`.
- Unit, async, and doc tests across all crates; clippy- and rustfmt-clean.

### Not yet implemented (tracked as follow-up issues)

- Streaming chat responses (SSE) and incremental JSON parsing.
- AutoPilot end-to-end session orchestration.
- Live periodic metrics sampling.
- Native Rust KV-cache compression algorithms.
- CI/release automation.
