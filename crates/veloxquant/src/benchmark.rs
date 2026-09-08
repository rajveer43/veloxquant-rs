//! `benchmark()`: tokens/sec, time-to-first-token, and (optionally)
//! measured resident memory for a chat completion against an already
//! reachable runtime.
//!
//! Mirrors `@veloxquant/sdk`'s `src/benchmark.ts`, with one deliberate,
//! documented architectural deviation: `benchmark.ts` (and Go's
//! equivalent) *owns* the runtime's process lifecycle — `vq.load(...)`
//! spawns a `veloxquant serve` subprocess it can later read the PID of and
//! `stop()`. `veloxquant-rs` has no such process-ownership concept
//! anywhere in the workspace today (`veloxquant-cli`'s `serve` subcommand
//! only connects to and reports on an *already-running* runtime — see
//! `crates/veloxquant-cli/src/serve.rs` — there is no `Client::load()`
//! that spawns a process). Reimplementing subprocess lifecycle management
//! here, scoped to just this one function, would be a much larger and
//! architecturally inconsistent addition than "port benchmark.ts" implies.
//!
//! So this module benchmarks a single already-loaded/reachable model in
//! one call ([`benchmark_pass`]), and [`benchmark`] runs two such passes
//! **sequentially** (never concurrently, matching `benchmark.ts:122-148`'s
//! explicit reasoning) against whatever the runtime is currently serving
//! for each pass — the caller is responsible for pointing `client` at a
//! runtime already loaded with the default method before the first call
//! and the optimized method before the second (e.g. restarting
//! `veloxquant serve --method ...` between passes), exactly mirroring the
//! fact that this SDK talks to the runtime over HTTP and does not start it.
//! Resident-memory sampling is correspondingly `pid`-driven: pass the
//! runtime process's PID (however the caller obtained it) to sample RSS,
//! or omit it to skip resident-memory measurement entirely — `None` is a
//! completely normal outcome, matching `benchmark.ts`'s
//! `getResidentBytes`'s `catch { return null }` never being a hard error.

use std::time::Instant;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{Client, Result};

/// Input to a single [`benchmark_pass`] call.
#[derive(Debug, Clone)]
pub struct BenchmarkInput {
    /// Model identifier to benchmark against (must already be loaded/
    /// reachable on the configured runtime).
    pub model: String,
    /// Compression method label recorded in the result purely for
    /// reporting (this SDK does not select or apply a method itself —
    /// see the module-level doc comment). Optional; `None` reports as
    /// `"default"` in [`BenchmarkResult::to_markdown`].
    pub optimized_method: Option<String>,
    /// Maximum tokens to generate for the timing prompt. Defaults to 128,
    /// matching `benchmark.ts`'s `DEFAULT_MAX_TOKENS`.
    pub max_tokens: Option<u32>,
    /// PID of the runtime process to sample resident memory (RSS) from,
    /// if known. `None` skips resident-memory measurement for this pass
    /// (never a hard error either way — see [`resident_bytes`]).
    pub pid: Option<u32>,
}

impl BenchmarkInput {
    /// Creates a benchmark input for `model` with every optional field at
    /// its default (no method label, default max_tokens, no RSS sampling).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            optimized_method: None,
            max_tokens: None,
            pid: None,
        }
    }
}

const DEFAULT_MAX_TOKENS: u32 = 128;
const BENCHMARK_PROMPT: &str = "Write a three-sentence summary of how photosynthesis works.";

/// Timing measured for a single [`benchmark_pass`] call.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BenchmarkTiming {
    /// Generation throughput, in tokens/sec, measured from first token to
    /// stream completion (matching `benchmark.ts`'s `timeGeneration`: the
    /// clock for this rate starts at the first token, not at request
    /// send, so time-to-first-token is not double-counted into the rate).
    pub tokens_per_second: f64,
    /// Time to first token, in milliseconds, measured from request send.
    pub time_to_first_token_ms: f64,
}

/// The result of a single [`benchmark_pass`] call: one model, one method
/// label, one timing + (optional) resident-memory sample.
#[derive(Debug, Clone)]
pub struct BenchmarkPass {
    /// The model that was benchmarked.
    pub model: String,
    /// The method label this pass was run under (see
    /// [`BenchmarkInput::optimized_method`]).
    pub method: String,
    /// Measured generation timing.
    pub timing: BenchmarkTiming,
    /// Measured resident memory (RSS) of the runtime process, sampled
    /// once immediately before generation, if [`BenchmarkInput::pid`] was
    /// provided and sampling succeeded.
    pub resident_bytes: Option<u64>,
}

/// The combined result of a full two-pass [`benchmark`] call, comparing a
/// default-method pass against an optimized-method pass.
///
/// Structural equivalent of `benchmark.ts:19-41`'s `BenchmarkResult`.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The model that was benchmarked.
    pub model: String,
    /// This machine's chip/CPU model, if detected.
    pub chip: Option<String>,
    /// This machine's total unified/physical memory, in bytes, if
    /// detected.
    pub unified_memory_bytes: Option<u64>,
    /// Generation throughput from the *default*-method pass, in
    /// tokens/sec.
    pub tokens_per_second: f64,
    /// Time to first token from the default-method pass, in milliseconds.
    pub time_to_first_token_ms: f64,
    /// Measured resident memory (RSS) of the default-method pass, if a
    /// PID was supplied for it.
    ///
    /// Like `benchmark.ts`'s `defaultMethodResidentBytes`, this is real,
    /// measured memory sampled once right after the model finishes
    /// loading/becoming reachable — **not** the accounting-only byte
    /// counts `Client::memory().estimate()` reports (see the "Compression
    /// byte counts are accounting-only" note in the README). It reflects
    /// idle model-load RSS, not KV-cache growth under load, and
    /// compression is not guaranteed to lower it — see
    /// [`BenchmarkResult::to_markdown`]'s caveat line.
    pub default_method_resident_bytes: Option<u64>,
    /// Measured resident memory (RSS) of the optimized-method pass, if a
    /// PID was supplied for it.
    pub optimized_resident_bytes: Option<u64>,
    /// The default pass's method label (usually `"default"`).
    pub method: String,
    /// The optimized pass's method label.
    pub optimized_method_used: String,
}

/// Reads RSS (bytes) for `pid` via `ps -o rss= -p <pid>` (macOS/Linux;
/// Linux could alternatively read `/proc/<pid>/status`, but `ps` covers
/// both without a platform-specific code path, matching `benchmark.ts:44-52`'s
/// "same approach, cross-platform" choice over a `/proc`-only fast path).
///
/// Returns `None` — never a hard error — if the process has already
/// exited, `ps` isn't available, or its output can't be parsed. Matches
/// TS's `catch { return null }`.
pub async fn resident_bytes(pid: u32) -> Option<u64> {
    let output = tokio::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let kb: u64 = stdout.trim().parse().ok()?;
    Some(kb * 1024)
}

/// Times a single generation against the model in `input`, measuring
/// tokens/sec and time-to-first-token at the SDK boundary (chunk arrival
/// timestamps from the streaming chat API) — no new instrumentation is
/// added to the runtime itself, matching `benchmark.ts`'s "measure at the
/// SDK boundary" design (`timeGeneration`).
async fn time_generation(client: &Client, input: &BenchmarkInput) -> Result<BenchmarkTiming> {
    let chat = client.chat()?;
    let mut stream = chat
        .stream(
            input.model.clone(),
            vec![veloxquant_openai::Message::user(BENCHMARK_PROMPT)],
        )
        .await?;

    let start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut token_count: u32 = 0;
    let max_tokens = input.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.done {
            break;
        }
        if chunk.text.is_empty() {
            continue;
        }
        if first_token_at.is_none() {
            first_token_at = Some(Instant::now());
        }
        token_count += 1;
        if token_count >= max_tokens {
            break;
        }
    }

    let end = Instant::now();
    let time_to_first_token_ms = match first_token_at {
        Some(t) => t.duration_since(start).as_secs_f64() * 1000.0,
        None => end.duration_since(start).as_secs_f64() * 1000.0,
    };
    let generation_seconds = end
        .duration_since(first_token_at.unwrap_or(start))
        .as_secs_f64();
    let tokens_per_second = if generation_seconds > 0.0 {
        token_count as f64 / generation_seconds
    } else {
        0.0
    };

    Ok(BenchmarkTiming {
        tokens_per_second,
        time_to_first_token_ms,
    })
}

/// Runs a single benchmark pass against `input.model` on the runtime
/// `client` is configured to talk to.
///
/// Samples resident memory (if `input.pid` is set) immediately before
/// issuing the generation request, then times the generation itself.
pub async fn benchmark_pass(client: &Client, input: BenchmarkInput) -> Result<BenchmarkPass> {
    let resident_bytes = match input.pid {
        Some(pid) => resident_bytes(pid).await,
        None => None,
    };

    let timing = time_generation(client, &input).await?;

    Ok(BenchmarkPass {
        model: input.model,
        method: input
            .optimized_method
            .unwrap_or_else(|| "default".to_string()),
        timing,
        resident_bytes,
    })
}

/// Benchmarks tokens/sec, TTFT, and (optionally) measured resident memory
/// for `model` on this machine, comparing a default-method pass against
/// an optimized-method pass.
///
/// Runs the two passes **sequentially, never concurrently** (matching
/// `benchmark.ts:122-148`'s explicit "avoid resource contention skewing
/// the timing" reasoning) — `default_input` is benchmarked first, then
/// `optimized_input`. Both are run against whatever `client` is currently
/// configured to reach; unlike `benchmark.ts`, this function does not load
/// or reload the runtime between passes (see the module-level doc comment
/// for why) — the caller must ensure `client` is pointed at a runtime
/// already serving the appropriate method before each pass. A typical
/// pattern is to call [`benchmark_pass`] and [`resident_bytes`] directly,
/// restarting the runtime process between calls, rather than calling this
/// convenience wrapper.
pub async fn benchmark(
    client: &Client,
    default_input: BenchmarkInput,
    optimized_input: BenchmarkInput,
) -> Result<BenchmarkResult> {
    let system_info = client.system().info().await;

    let default_pass = benchmark_pass(client, default_input).await?;
    let optimized_pass = benchmark_pass(client, optimized_input).await?;

    Ok(BenchmarkResult {
        model: default_pass.model,
        chip: if system_info.cpu_model.is_empty() {
            None
        } else {
            Some(system_info.cpu_model)
        },
        unified_memory_bytes: if system_info.total_memory_bytes > 0 {
            Some(system_info.total_memory_bytes)
        } else {
            None
        },
        tokens_per_second: default_pass.timing.tokens_per_second,
        time_to_first_token_ms: default_pass.timing.time_to_first_token_ms,
        default_method_resident_bytes: default_pass.resident_bytes,
        optimized_resident_bytes: optimized_pass.resident_bytes,
        method: default_pass.method,
        optimized_method_used: optimized_pass.method,
    })
}

impl BenchmarkResult {
    /// Renders this result as the same Markdown report shape as
    /// `benchmark.ts:83-110`'s `buildMarkdown`, including the
    /// "compression is accounting-only" caveat line verbatim when
    /// optimized resident memory comes out *higher* than the default
    /// method's — a deliberately-preserved honesty note from the TS
    /// implementation's own measured finding (kivi measured higher idle
    /// RSS than the default method against a 1B model in practice), not
    /// boilerplate.
    pub fn to_markdown(&self) -> String {
        let mut lines = vec!["VeloxQuant Benchmark".to_string(), String::new()];
        lines.push(format!("Model: {}", self.model));
        lines.push(format!(
            "Machine: {}",
            self.chip.as_deref().unwrap_or("unknown")
        ));
        if let Some(mem) = self.unified_memory_bytes {
            lines.push(format!("RAM: {:.0}GB", mem as f64 / 1024f64.powi(3)));
        }
        lines.push(String::new());
        lines.push(format!("Tokens/sec: {:.1}", self.tokens_per_second));
        lines.push(format!("TTFT: {:.0}ms", self.time_to_first_token_ms));
        lines.push(String::new());

        if let (Some(before_bytes), Some(after_bytes)) = (
            self.default_method_resident_bytes,
            self.optimized_resident_bytes,
        ) {
            let before_mb = before_bytes as f64 / 1024f64.powi(2);
            let after_mb = after_bytes as f64 / 1024f64.powi(2);
            let delta_percent = ((before_mb - after_mb) / before_mb) * 100.0;

            lines.push(format!(
                "{} resident memory: {:.0}MB",
                self.method, before_mb
            ));
            lines.push(format!(
                "{} resident memory: {:.0}MB",
                self.optimized_method_used, after_mb
            ));
            if delta_percent >= 0.0 {
                lines.push(format!("Resident memory reduced: {:.0}%", delta_percent));
            } else {
                lines.push(format!(
                    "Resident memory increased: {:.0}% (compression is accounting-only — see README)",
                    delta_percent.abs()
                ));
            }
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(default_bytes: Option<u64>, optimized_bytes: Option<u64>) -> BenchmarkResult {
        BenchmarkResult {
            model: "test-model".to_string(),
            chip: Some("Apple M2".to_string()),
            unified_memory_bytes: Some(16 * 1024 * 1024 * 1024),
            tokens_per_second: 42.5,
            time_to_first_token_ms: 123.0,
            default_method_resident_bytes: default_bytes,
            optimized_resident_bytes: optimized_bytes,
            method: "default".to_string(),
            optimized_method_used: "kivi".to_string(),
        }
    }

    #[test]
    fn to_markdown_includes_model_and_timing() {
        let result = sample_result(None, None);
        let md = result.to_markdown();
        assert!(md.contains("Model: test-model"));
        assert!(md.contains("Tokens/sec: 42.5"));
        assert!(md.contains("TTFT: 123ms"));
        assert!(md.contains("Machine: Apple M2"));
        assert!(md.contains("RAM: 16GB"));
    }

    #[test]
    fn to_markdown_omits_resident_memory_section_when_unmeasured() {
        let result = sample_result(None, None);
        let md = result.to_markdown();
        assert!(!md.contains("resident memory"));
    }

    #[test]
    fn to_markdown_reports_reduction_when_optimized_is_smaller() {
        let before = 2000 * 1024 * 1024u64;
        let after = 1000 * 1024 * 1024u64;
        let result = sample_result(Some(before), Some(after));
        let md = result.to_markdown();
        assert!(md.contains("Resident memory reduced: 50%"));
        assert!(!md.contains("accounting-only"));
    }

    #[test]
    fn to_markdown_preserves_accounting_only_caveat_when_optimized_is_larger() {
        // Mirrors benchmark.ts's own measured finding: kivi measured
        // higher idle RSS than the default method against a 1B model in
        // practice. This case must not be silently dropped in the port.
        let before = 1000 * 1024 * 1024u64;
        let after = 1200 * 1024 * 1024u64;
        let result = sample_result(Some(before), Some(after));
        let md = result.to_markdown();
        assert!(md.contains("Resident memory increased: 20%"));
        assert!(md.contains("compression is accounting-only — see README"));
    }

    #[test]
    fn to_markdown_reports_unknown_machine_when_chip_not_detected() {
        let mut result = sample_result(None, None);
        result.chip = None;
        result.unified_memory_bytes = None;
        let md = result.to_markdown();
        assert!(md.contains("Machine: unknown"));
        assert!(!md.contains("RAM:"));
    }

    #[tokio::test]
    async fn resident_bytes_returns_none_for_a_pid_ps_cannot_find() {
        // PID 1 exists on virtually every Unix system's process table but
        // is never accessible without escalated privileges in a
        // sandboxed/containerized CI runner, and an obviously-invalid PID
        // (u32::MAX) is guaranteed not to exist on any system — either
        // way `ps` reports nothing, exercising the "unmeasurable, return
        // None, never a hard error" path without depending on real
        // process visibility.
        let result = resident_bytes(u32::MAX).await;
        assert!(result.is_none());
    }

    #[test]
    fn benchmark_input_new_has_no_method_label_by_default() {
        let input = BenchmarkInput::new("m");
        assert_eq!(input.model, "m");
        assert!(input.optimized_method.is_none());
        assert!(input.max_tokens.is_none());
        assert!(input.pid.is_none());
    }
}

/// Manual/hardware-dependent integration tests.
///
/// These are **not unit-testable in CI**: they require a real, already
/// running VeloxQuant (or other OpenAI-compatible) runtime serving an
/// actual model — exactly matching `benchmark.ts`'s own acknowledgment at
/// `test/integration/benchmark.manual.ts` and its doc comment at
/// `benchmark.ts:118-121`. Both tests are `#[ignore]`d so `cargo test`
/// never attempts them by default; this module makes no claim of CI
/// coverage it cannot actually have.
///
/// # Running these tests by hand
///
/// 1. Start a VeloxQuant runtime (or any OpenAI-compatible server) at
///    `http://localhost:8765` serving a real model, e.g.:
///    ```sh
///    veloxquant serve --model mlx-community/Qwen3-8B-4bit
///    ```
/// 2. Set `VQ_BENCHMARK_MODEL` to the model id the runtime is serving if
///    it differs from the default used below.
/// 3. Run:
///    ```sh
///    cargo test -p veloxquant --features openai --test-threads=1 \
///        -- --ignored benchmark_manual
///    ```
#[cfg(test)]
mod manual_tests {
    use super::*;

    fn model_under_test() -> String {
        std::env::var("VQ_BENCHMARK_MODEL")
            .unwrap_or_else(|_| "mlx-community/Qwen3-8B-4bit".to_string())
    }

    /// Runs a single [`benchmark_pass`] against a real, already-running
    /// runtime and prints its timing. Requires manual setup — see the
    /// module doc comment.
    #[tokio::test]
    #[ignore = "requires a real VeloxQuant runtime serving a real model at http://localhost:8765 — see module docs"]
    async fn benchmark_manual_single_pass_against_live_runtime() {
        let client = Client::builder().build().unwrap();
        let input = BenchmarkInput::new(model_under_test());

        let pass = benchmark_pass(&client, input)
            .await
            .expect("benchmark_pass should succeed against a live runtime");

        assert!(pass.timing.tokens_per_second >= 0.0);
        assert!(pass.timing.time_to_first_token_ms >= 0.0);
        println!(
            "manual benchmark: {:.1} tok/s, {:.0}ms TTFT",
            pass.timing.tokens_per_second, pass.timing.time_to_first_token_ms
        );
    }

    /// Runs a full two-pass [`benchmark`] comparison against a real,
    /// already-running runtime. Since this SDK does not own the runtime's
    /// process lifecycle (see the module-level doc comment), both passes
    /// hit whatever the runtime happens to be serving — this test mainly
    /// exercises that the two-pass orchestration and `to_markdown()`
    /// rendering work end to end against real timing data, not that a
    /// method switch actually occurred between passes.
    #[tokio::test]
    #[ignore = "requires a real VeloxQuant runtime serving a real model at http://localhost:8765 — see module docs"]
    async fn benchmark_manual_two_pass_against_live_runtime() {
        let client = Client::builder().build().unwrap();
        let model = model_under_test();

        let result = benchmark(
            &client,
            BenchmarkInput::new(model.clone()),
            BenchmarkInput {
                optimized_method: Some("manual".to_string()),
                ..BenchmarkInput::new(model)
            },
        )
        .await
        .expect("benchmark should succeed against a live runtime");

        println!("{}", result.to_markdown());
    }
}
