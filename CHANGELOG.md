# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Prepared as **0.3.0**: `workspace.package.version`, every
`[workspace.dependencies]` pin, and the `veloxquant` pins in
`veloxquant-cli` and `veloxquant-rig` are already bumped from 0.2.1 to
0.3.0, so this section is stamped `[0.3.0]` when the release is cut (see
the README's *Releasing* section). It's a minor bump, not 0.2.2, because
it breaks semver: `VeloxQuantError` gained variants and is now
`#[non_exhaustive]` (see *Changed*). It's also a fresh version rather than
a backfill of the half-published 0.2.1 (see *Fixed*).

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
- `rig-core` `CompletionModel` adapter (Phase 5, final phase of SDK parity):
  new, independently-versioned `veloxquant-rig` crate
  (`VeloxQuantCompletionModel`) implementing `rig-core` 0.42's
  `CompletionModel` trait (`completion`/`stream`), backed by a
  `veloxquant::Client`. Mirrors the *shape* of Go's `langchain` adapter
  (a separate crate/module keeping the third-party framework dependency
  opt-in) rather than a literal port, since `rig` has no equivalent of
  `langchaingo`'s single `llms.Model` interface. Text-only: any non-text
  `rig` message content (images, audio, documents, tool calls/results,
  reasoning blocks) is rejected with an explicit
  `RigAdapterError::UnsupportedContent` rather than silently dropped.
  `stream()` reuses the existing SSE transport
  (`veloxquant_openai::stream_chat_completions`) rather than a second SSE
  parser, verified end-to-end against a real fixture SSE server. New
  `crates/veloxquant-rig/examples/rig_integration.rs` (manual verification
  against a running VeloxQuant runtime, documented as such — not
  CI-verifiable).
- AutoPilot end-to-end session orchestration (gap-closing pass, carried
  since v0.1.0's "Not yet implemented" list): `veloxquant::autopilot`
  (`AutoPilot`, `AutoPilotConfig`, `ModelSelection`, `AutoPilotPlan`,
  `AutoPilotSession`, `AutoPilotOutcome`, `AutoPilotFitError`) and
  `Client::autopilot`, gated behind a new `autopilot` feature (implies
  `openai`). Follows Go's `Client.AutoPilot` (`autopilot.go`) for
  everything *except* the compression decision: hardware inspection,
  registry-ranked model selection (or a pinned model by name), the
  8192-token default context, and the 15% safety margin are Go's. The
  compression strategy is **never computed in Rust** — it comes from the
  real VeloxQuant-MLX CLI (`veloxquant::autopilot::cli::VeloxQuantCli`):
  `recommend --json` picks the method, its warnings are checked against the
  TS SDK's won't-fit pattern (`/will not fit|short of any headroom/i`),
  `methods --json --servable-only` confirms `serve` can run it, and
  `auto-config --json`'s serve-safe pool is the fallback when it can't —
  the same TS/Kotlin/Studio shape the Swift SDK adopted. Every flag was
  checked against `veloxquant_mlx/cli/*.py` and real `--json` output, not
  copied from a sibling SDK. `AutoPilotPlan` records every decision
  (hardware, selection reason, context length, an offline accounting-only
  memory estimate, safety margin, the exact `recommend` inputs/outputs, any
  forced won't-fit warnings, method/bits, the `auto-config` fallback and
  its reason, `methods`' `accounting_only` flag) plus an ordered
  `decisions` trail. `AutoPilot::try_start` returns
  `AutoPilotOutcome::WontFit` as data; `AutoPilot::start`/
  `Client::autopilot` return `VeloxQuantError::AutoPilotWontFit` instead,
  and `AutoPilotFitError::into_error` guarantees the two carry the same
  warnings/method. Subprocesses run via an injectable `CommandRunner`
  (default `TokioCommandRunner`: both pipes drained concurrently, child
  killed if the future is dropped), so the orchestration is unit-tested
  without Python; a stand-in executable exercises the real
  `tokio::process` path, and two `#[ignore]`d manual tests run against the
  real `veloxquant` CLI (both passed by hand on an Apple Silicon Mac for
  this change — not claimed as CI coverage). `examples/autopilot.rs` is now
  a working example instead of a "not yet implemented" stub.
- `ModelRegistry::get` and `ModelRegistry::recommend_scored`
  (`ScoredModel`, `DEFAULT_RANKING_CONTEXT_LENGTH`): ports of Go's
  `Registry.Get`/`models.RecommendScored` (int4 weights + int4 KV fit check
  including runtime overhead, a headroom score, a `recommended` bonus,
  stable ties, and a human-readable reason per candidate), used by
  AutoPilot's selection. The existing `ModelRegistry::recommend` is
  unchanged.
- Live periodic metrics sampling (carried since v0.1.0; the monitor crate's
  doc comment had deferred it to "v0.3.0"): `Monitor::spawn_sampler`
  starts a background task that polls a `Sampler` on an interval and
  publishes each sample through the *existing* broadcast channel (it calls
  `Monitor::publish`; the pub/sub plumbing is unchanged). Returns a
  `SamplingHandle` (`stop().await` waits for the task like Go's
  `Monitor.Stop`; dropping the handle also stops it). Go's semantics
  throughout: first sample immediately, then one per interval; a zero
  interval falls back to Go's 5 s default (`DEFAULT_SAMPLE_INTERVAL`); a
  sampler error skips that tick without ending the loop; manual `publish`
  calls interleave with sampled ones (Go's `Monitor.Report`).
  `SystemSampler` is the default sampler — host memory used/available via
  `veloxquant-system`, the same fields Go's `Client.Monitor` sampler fills
  — and leaves every inference-side field at its default rather than
  fabricating one. `veloxquant-monitor` now depends on `veloxquant-system`.
  Re-exported from the facade behind the existing `monitor` feature. New
  `examples/monitor.rs`.
- `scripts/publish-crates.sh`: publishes every publishable workspace crate
  in an order derived from `cargo metadata` (topological over path
  dependencies), skipping any crate whose current version is already on
  crates.io (`--plan` prints the order and skip decisions without
  publishing).

### Changed

- **Breaking: `VeloxQuantError` is now `#[non_exhaustive]`, and it has
  twelve variants that 0.2.1 didn't.** AutoPilot added five: `NoModelFits`
  (Go's `ErrInsufficientMemory` from selection, with the task and budget),
  `CliUnavailable`, `CliCommandFailed`, `MalformedCliOutput`, and
  `AutoPilotWontFit`. The earlier parity phases added seven:
  `ModelScanFailed`, `ModelPullFailed`, `ModelDeleteFailed`,
  `AgentMaxStepsExceeded`, `ToolAlreadyRegistered`,
  `UnsupportedMcpContent`, and `Mcp`. Each one represents a failure the new
  features can actually hit, so leaving them out wasn't an option. Any of
  them breaks an exhaustive `match` on the error downstream.
  `#[non_exhaustive]` is the standard Rust fix, but adding it is itself
  breaking: code outside `veloxquant-core` must now have a `_ =>` arm, even
  when it only matches 0.2.1's seven variants. This one-time break is
  deliberate. Once the attribute is in place, a new variant is a minor,
  non-breaking addition, so later releases won't have this problem again.
  The version is 0.3.0 rather than 0.2.2 because of this break.
  **Migration:** add a wildcard arm to any `match` on `VeloxQuantError`.
  No match in this workspace needed one (all of them are in
  `veloxquant-core` or already use `_`).
- Workspace version bumped to 0.3.0 (see the note at the top of this
  section). `veloxquant-rig` stays at its own 0.1.0, which has never been
  published, but now pins `veloxquant = "0.3.0"`.
- CI and release jobs moved from `macos-14` to `macos-15`. GitHub is
  retiring the macOS 14 images: brownouts start 2026-10-05, and they
  become fully unsupported on 2026-11-02 (actions/runner-images#13518).
  `macos-15` is GitHub's recommended replacement. The CI and release
  workflows, including `bump-version.yml`, now use only runner labels
  listed as generally available in actions/runner-images.
- `scripts/bump-version.sh` no longer fails when the manifests already
  match the requested version. It then only stamps the changelog, so the
  **Bump version** workflow can cut 0.3.0 even though the manifests were
  bumped by hand.
- `Monitor::start` is documented as the no-op it has always been. It still
  does not start sampling; `Monitor::spawn_sampler` does.
- `veloxquant-cli` is now `publish = false` in its manifest, matching the
  README's existing "not published — see prebuilt binaries" row, so the
  publish script (and an accidental `cargo publish`) skips it.
- **MSRV raised from 1.75 to 1.90.** The 1.75 claim was already false: the
  committed `Cargo.lock` doesn't build on 1.75 (`idna_adapter` 1.2.2, pulled
  in by `reqwest` → `url` → `idna`, is edition 2024 and needs 1.86), so the
  CI MSRV job could never pass. `idna_adapter` isn't the ceiling, though.
  Checking every locked package's `rust-version` shows `ordered-float` 5.5
  (via `rig-core` 0.42) needs 1.90, and `rmcp` 3.2 and the ICU crates behind
  `url` need 1.88. 1.90 is the real floor: `cargo +1.89 build` is refused
  with "ordered-float@5.5.0 requires rustc 1.90", and on 1.90
  `cargo build --workspace --all-features --locked` and the full test suite
  both pass. We didn't pin old versions of these transitive crates to keep
  1.75. That would mean holding back a chain of widely shared dependencies,
  and the next `cargo update` would break it again. Nothing in the project
  needs an old toolchain anyway: `rust-toolchain.toml` tracks stable. The
  CI MSRV job now uses 1.90 and runs `build` and `test` with
  `--all-features --locked`, so it checks the committed lockfile instead
  of re-resolving it. `workspace.package.rust-version` and the README's
  MSRV line now say 1.90.

### Fixed

- **macOS available memory was close to zero on a Mac with plenty
  free.** `veloxquant-system` used sysinfo 0.32's `available_memory()`,
  which on macOS is `free + inactive + purgeable - compressor` pages. After
  a Mac has been up a while, the compressor holds about as many pages as
  are inactive, so the figure collapses toward zero. A live AutoPilot run
  therefore concluded that no model fits (`NoModelFits`). On the 24 GiB M4
  used for this fix, sysinfo reported **0.00 GiB** available and the fixed
  code reports **6.83 GiB**. On macOS (`#[cfg(target_os = "macos")]`),
  `memory_stats()` now reads Mach `host_statistics64(HOST_VM_INFO64)` and
  reports `(free_count + inactive_count) × page size`, capped at total and
  falling back to total if the call fails. That is exactly the Swift
  SDK's `HostMemory.current()`, and the same number as the Go SDK's
  `availableMemoryFromVMStat` (vm_stat's free + inactive + speculative;
  Mach's `free_count` already includes speculative pages). Total memory,
  and available memory on Linux and Windows, still come from `sysinfo`.
  The fix flows to `detect()`, `SystemService`, AutoPilot, and the
  monitor's `SystemSampler`. It adds a macOS-only `libc` dependency (already
  in `Cargo.lock` through sysinfo). New tests:
  - A deterministic regression test on the real page counts from the
    failing machine: sysinfo's formula gives < 1 GiB, and the fixed one
    gives > 7 GiB.
  - Fallback edge cases.
  - A macOS live sanity bound: available is at most total and at least 5%
    of it. The buggy figure was about 3%.
  - A macOS cross-check of the Mach FFI reading against `vm_stat` (within
    10% of total), which catches a struct-layout or page-size mistake.
- **crates.io publishing was broken, and v0.2.1 was only half-published.**
  `release.yml`'s hardcoded crate list published `veloxquant-runtime`
  before `veloxquant-openai`, which runtime started depending on in 0.2.1.
  The v0.2.1 run (Actions run 33751941369) failed with `failed to select a
  version for the requirement veloxquant-openai = "^0.2.1"` after uploading
  only core/system/memory, so crates.io today has `veloxquant-core`/
  `-system`/`-memory` at 0.2.1 but `veloxquant`/`-runtime`/`-openai`/
  `-monitor` stuck at 0.2.0. The list also omitted `veloxquant-models`,
  which the facade crate depends on, so the facade could never have been
  published, and `veloxquant-rig`. The publish job now runs
  `scripts/publish-crates.sh`: order derived from `cargo metadata`, already
  published versions skipped (so a re-run resumes a partial release), and
  `--no-verify` and the fixed `sleep 15` removed (cargo waits for the index
  itself).
  **The recovery is 0.3.0, not a backfill of 0.2.1.** `master` has moved
  well past the `v0.2.1` tag (AutoPilot, the sampler, two new crates, the
  error-enum break above), so republishing 0.2.1 from `master` would ship
  different code under an already-released version number. Rebuilding
  the stuck crates from the tag would still publish nothing people should
  use. 0.2.1 stays as it is on crates.io, and 0.3.0 is the first version
  published consistently across all crates. `scripts/publish-crates.sh
  --plan` (run for this change) reports all nine publishable crates as
  "publish" at their new versions, in this order: `veloxquant-core`,
  `-memory`, `-models`, `-system`, `-monitor`, `-openai`, `-runtime`,
  `veloxquant`, then `veloxquant-rig` 0.1.0. Each crate follows every
  workspace crate it depends on. Only `veloxquant-core` has no workspace
  dependencies (`-system` depends on it). `-openai` precedes `-runtime`,
  and `-system` precedes `-monitor`. `veloxquant-models` and
  `veloxquant-rig` appear for the first time. `cargo package --workspace
  --no-verify` also succeeds for all nine. Consider yanking the stuck
  0.2.x versions once 0.3.0 is live. That's left to the maintainer
  (see *Judgment calls pending*).
- Releases are now gated on the **full CI workflow** (`ci.yml` gained
  `workflow_call`; `release.yml` `uses:` it, which is Go's and Swift's
  pattern) instead of a single `cargo test` job. A new `verify-tag` job
  fails the release if the tag doesn't match `workspace.package.version`.
- CI's "macOS Intel" test leg and the release's `x86_64-apple-darwin`
  build targeted the retired `macos-13` runner. Every CI run since v0.1.0
  left that leg queued until GitHub cancelled it after 24h, and the v0.2.1
  release's GitHub-release job never ran for the same reason. The CI leg
  now uses `macos-15-intel`. That label was checked against
  actions/runner-images: it is listed as a generally available x64 image,
  announced as `macos-13`'s replacement (actions/runner-images#13045), and
  supported until August 2027. It is GitHub's last Intel macOS image, and
  GitHub drops x86_64 macOS entirely after that date. `macos-14-large` and
  `macos-13-large` weren't used: they are paid larger runners, and both
  are deprecated or retired along with their OS images. The release build
  cross-compiles `x86_64-apple-darwin` on `macos-15` (Apple Silicon), so
  Intel `vq` binaries don't depend on an Intel runner. The label is
  verified against GitHub's documentation, but no CI run could be started
  from here to test it.
- The `Bump version` workflow's tag push could never trigger
  `release.yml`, because pushes made with `GITHUB_TOKEN` don't start other
  workflows. It now dispatches `release.yml` on the new tag
  (`workflow_dispatch`, GitHub's documented exception). `release.yml`
  accepts `workflow_dispatch` and refuses to run on a non-tag ref.
- CI clippy now runs with `--all-features`, so feature-gated modules
  (`agent`, `mcp`, `autopilot`, `local-models`, `monitor`) are linted.
  CI also gained a `package + publish plan` job (`cargo package
  --workspace --no-verify` plus `scripts/publish-crates.sh --plan`), which
  catches manifest problems and publish-order problems before a tag is
  pushed.

### Deviations from Go/Kotlin (AutoPilot and monitoring)

- **Compression choice shells out instead of Go's local optimizer.** Go's
  `Client.AutoPilot` calls its own pure-Go `Optimize.Recommend`; this port
  calls the real `veloxquant` CLI as described above, so the plan carries
  `method`/`bits`/`recommendation` in place of Go's `Profile`/
  `CompressionBits`/`EstimatedMemoryBytes`. `AutoPilotPlan::reason()` keeps
  Go's `Reason` shape (selection reason + `; ` + the CLI's rationale). The
  Rust crate's own `OptimizationService` is not consulted by AutoPilot.
- **Kotlin's runtime behaviors that don't match the real CLI were not
  copied** (the Swift port documents the same four). The `recommend` call
  passes the required legacy flags `--chip/--ram-gb/--model-class/--goal`
  and no `--batch-size`. `methods --json` is decoded as its envelope
  object, not a bare array. `resident_savings_likely: false` is **not**
  treated as "won't fit", because the real CLI reports `false` for the
  default `everyday` goal on every Mac.
- **Hardware is mapped conservatively onto `recommend`'s fixed buckets**
  (Swift's rule). RAM rounds **down** to an allowed `--ram-gb`, and
  parameter count rounds **up** to a `--model-class`, so Qwen3-8B is
  described as `14B`. Chips newer than M4 map to `M4`. A non-M-series CPU,
  or under 8 GB of RAM, is `VeloxQuantError::UnsupportedPlatform`.
- **Additions beyond Go's config:** `goal`, `force`, and `model_class`
  (from TS/Kotlin/Swift), plus `ModelSelection::Custom` for a model outside
  the curated registry.
- **No session-level conversation or runtime launch.** `AutoPilotSession`
  offers `chat`/`stream` with the planned model; this SDK has no
  `Conversation` type. Like Go's and Kotlin's AutoPilot, it doesn't launch
  `serve` itself. The plan carries the method/bits to serve with.
- **Monitor shape:** a broadcast `Receiver` instead of Go's callback
  `Subscribe`, and a `SamplingHandle` instead of `Start`/`Stop` methods on
  `Monitor`. There's no `Monitor.Metrics()`-style "latest sample"
  accessor. Go's automatic merging of inference metrics from chat calls
  into the monitor (`setMetricsSink`) is not ported, so callers publish
  those themselves.

### Judgment calls pending

These were decided without a precedent to follow; defaults are
provisional:

- **AutoPilot as a facade module, not a `veloxquant-autopilot` crate.**
  It needs `Client`, `ModelRegistry` (which lives in the facade), and chat,
  and it adds `Client::autopilot`. A separate crate would have to depend on
  the facade (like `veloxquant-rig`) and couldn't add an inherent method to
  `Client`. That matches how `agent`/`mcp`/`benchmark` were placed.
- **CLI invocation defaults to the `veloxquant` console script on `PATH`**
  (Go's `runtime.DefaultCommand`, Kotlin). `VeloxQuantCli::python_module`
  offers Swift/Studio's `<python> -m veloxquant_mlx`. There's no
  environment-variable override and no interpreter auto-detection (Swift
  has both).
- **No timeout on CLI shell-outs** (Swift has none either). A hung
  `veloxquant` subprocess hangs `AutoPilot::start`. Wrapping the future in
  `tokio::time::timeout` kills the child, because it's spawned with
  `kill_on_drop`.
- **`UnsupportedPlatform` stays a unit variant**, so an AutoPilot
  chip/RAM rejection carries no detail. Adding a payload would be a second
  breaking change to an existing variant.
- **Keep or drop the macOS Intel test leg.** It is kept, on the verified
  `macos-15-intel` label. The case for dropping it:
  - None of the sibling SDKs tests on Intel macOS. Swift and Kotlin CI run
    only on `macos-14` (arm64), and Go CI runs only on `ubuntu-latest`.
  - Apple Silicon is the ecosystem's target: AutoPilot rejects non-M-series
    CPUs.
  - GitHub removes Intel macOS runners in August 2027 anyway.

  The case for keeping it: the release still ships an
  `x86_64-apple-darwin` `vq`, and this leg is the only place that code is
  *run* rather than just cross-compiled. The new Mach memory code also
  depends on page size, which is 4 KiB on Intel and 16 KiB on Apple
  Silicon. Recommendation: keep it until August 2027, or until
  `x86_64-apple-darwin` release binaries are dropped, whichever comes
  first, then remove both together.
- **`veloxquant-rig` keeps its independent 0.1.0** rather than joining the
  workspace's 0.3.0, because the earlier *Added* entry made it
  independently versioned on purpose. It has never been published, so
  0.1.0 publishes cleanly. The alternative is `version.workspace = true`.
- **Should `RigAdapterError` also be `#[non_exhaustive]`?** This is
  time-sensitive. No 0.1.0 has been published yet, so adding the attribute
  is free now and breaking after the first publish. It was left alone
  because this change was scoped to `VeloxQuantError`.
- **Other public enums** (`OptimizationProfile`, `Precision`, `Task`,
  `Role`, `ModelSelection`, `AutoPilotOutcome`, `RecommendGoal`,
  `McpTransport`, `ToolDefinitionKind`) are still exhaustive. The
  workspace had no `#[non_exhaustive]` precedent before this change, so
  none was applied beyond the error type. If any of them should be, 0.3.0
  is the cheapest release to do it in, since it's already breaking.
- **The macOS available-memory formula is free + inactive**, matching the
  Swift and Go SDKs so that every SDK makes the same AutoPilot fit
  decision on the same Mac. Purgeable pages aren't added and compressed
  pages aren't counted as available. Reclaiming compressed memory means
  decompressing or swapping, so counting it would overstate headroom.
  Activity Monitor's "memory pressure" view is more generous. Changing
  this would need to happen across all the SDKs together.
- **Yank the stuck 0.2.x crates?** On crates.io, `veloxquant`/`-runtime`/
  `-openai`/`-monitor` 0.2.0 and `-core`/`-system`/`-memory` 0.2.1 are
  mutually consistent (0.2.0's crates accept `^0.2.0` core), so nothing is
  unresolvable. Yanking them is optional, and it is the maintainer's call
  once 0.3.0 is live.
- **`veloxquant-rig` publishes from the same tag-driven job** (skipped
  once its independent version is already on crates.io). It still has no
  release trigger of its own.

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
