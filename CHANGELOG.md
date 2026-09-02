# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
