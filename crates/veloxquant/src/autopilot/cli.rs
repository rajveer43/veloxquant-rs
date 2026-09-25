//! One-shot shell-outs to the real `veloxquant` CLI (VeloxQuant-MLX) —
//! the only place AutoPilot's compression decision comes from.
//!
//! Three subcommands, each run with `--json`:
//!
//! - `recommend` (legacy `--chip/--ram-gb/--model-class/--goal` mode) — the
//!   compression recommendation itself.
//! - `methods --servable-only` — whether `veloxquant serve` can actually run
//!   the recommended method.
//! - `auto-config` — a pick from the CLI's serve-safe method pool, used only
//!   when `recommend`'s method isn't servable.
//!
//! Every flag here was checked against the Python source
//! (`veloxquant_mlx/cli/recommend.py`, `cli/auto_config.py`,
//! `cli/methods.py`, `tools/mac_recommender.py`) and against real `--json`
//! output, not copied from a sibling SDK. Argument lists come from pure
//! `*_args` functions so they are unit-testable without launching anything.

use std::fmt;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value};

use veloxquant_core::{Result, VeloxQuantError};

/// Go's `runtime.DefaultCommand`: the `veloxquant` console script
/// VeloxQuant-MLX installs (`[project.scripts]` in its `pyproject.toml`).
pub const DEFAULT_CLI_PROGRAM: &str = "veloxquant";

/// `recommend --ram-gb` choices (`ALLOWED_RAM_GB`,
/// `tools/mac_recommender.py`).
pub const ALLOWED_RAM_GB: [u32; 11] = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 512];

/// `recommend --model-class` choices with their size in billions of
/// parameters (`MODEL_WEIGHT_GB_4BIT`'s keys, `tools/mac_recommender.py`).
pub const MODEL_CLASSES: [(&str, f64); 9] = [
    ("1B", 1.0),
    ("3B", 3.0),
    ("7B", 7.0),
    ("14B", 14.0),
    ("32B", 32.0),
    ("70B", 70.0),
    ("120B", 120.0),
    ("235B", 235.0),
    ("671B", 671.0),
];

/// Knobs that carry a method's bit width, in the TS SDK's
/// `extractBitWidth` order.
pub const BIT_WIDTH_KNOBS: [&str; 4] =
    ["bit_width_inlier", "kvquant_bits", "gear_bits", "kivi_bits"];

/// Phrases in a `recommend` warning that mean "this will not fit" — the TS
/// SDK's `WONT_FIT_PATTERN` (`/will not fit|short of any headroom/i`). The
/// CLI has no structured severity field, so this depends on
/// `mac_recommender.py`'s wording; inspect `warnings` directly if in doubt.
pub const WONT_FIT_PHRASES: [&str; 2] = ["will not fit", "short of any headroom"];

/// Whether a `recommend` warning matches [`WONT_FIT_PHRASES`]
/// (case-insensitive).
pub fn is_wont_fit_warning(warning: &str) -> bool {
    let lowered = warning.to_lowercase();
    WONT_FIT_PHRASES.iter().any(|p| lowered.contains(p))
}

/// The largest allowed `--ram-gb` bucket not exceeding `bytes` (in GiB), or
/// `None` below 8 GiB. Rounds **down** (an 18 GB Mac is described as 16) —
/// the conservative direction for a fit check.
pub fn ram_bucket(bytes: u64) -> Option<u32> {
    let gib = bytes as f64 / (1u64 << 30) as f64;
    ALLOWED_RAM_GB
        .iter()
        .rev()
        .copied()
        .find(|&gb| f64::from(gb) <= gib + 0.01)
}

/// The smallest `--model-class` at least as large as `parameter_count`, or
/// `None` above 671B. Rounds **up** (an 8B model is described as `14B`) —
/// conservative for a fit check.
pub fn model_class_for(parameter_count: u64) -> Option<&'static str> {
    let billions = parameter_count as f64 / 1e9;
    MODEL_CLASSES
        .iter()
        .find(|(_, size)| *size >= billions - 0.05)
        .map(|(name, _)| *name)
}

/// Maps a CPU brand string (e.g. `"Apple M3 Pro"`) to `recommend --chip`'s
/// `M1`–`M4`. Chips newer than M4 map to `M4` (the CLI's newest choice, as
/// VeloxQuant Studio and the Swift SDK do). Anything that isn't a
/// recognizable Apple M-series chip returns `None`.
pub fn chip_argument(cpu_model: &str) -> Option<String> {
    let rest = &cpu_model[cpu_model.find("Apple M")? + "Apple M".len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let generation: u32 = digits.parse().ok()?;
    match generation {
        0 => None,
        1..=4 => Some(format!("M{generation}")),
        _ => Some("M4".to_string()),
    }
}

/// The first integer among [`BIT_WIDTH_KNOBS`] in `knobs` — the TS SDK's
/// `extractBitWidth`.
pub fn extract_bit_width(knobs: &Map<String, Value>) -> Option<u32> {
    BIT_WIDTH_KNOBS.iter().find_map(|key| {
        knobs
            .get(*key)
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
    })
}

/// `recommend --goal` values (`cli/recommend.py`'s `choices`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RecommendGoal {
    /// Balanced default.
    #[default]
    Everyday,
    /// Maximize key-cache compression accounting.
    MaxKeyAccounting,
    /// Fit the longest context.
    MaxContext,
    /// Favor output quality.
    BestQuality,
    /// Never grow past a fixed memory budget.
    ConstantMemory,
}

impl RecommendGoal {
    /// The CLI's wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Everyday => "everyday",
            Self::MaxKeyAccounting => "max_key_accounting",
            Self::MaxContext => "max_context",
            Self::BestQuality => "best_quality",
            Self::ConstantMemory => "constant_memory",
        }
    }
}

/// Inputs for `recommend --json` in legacy mode. All four of
/// `chip`/`ram_gb`/`model_class`/`goal` are required by the CLI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendCliRequest {
    /// `--chip`: `M1`–`M4` (see [`chip_argument`]).
    pub chip: String,
    /// `--ram-gb`: one of [`ALLOWED_RAM_GB`] (see [`ram_bucket`]).
    pub ram_gb: u32,
    /// `--model-class`: one of [`MODEL_CLASSES`] (see [`model_class_for`]).
    pub model_class: String,
    /// `--goal`.
    pub goal: RecommendGoal,
    /// `--seq-len`.
    pub seq_len: usize,
    /// `--n-layers`.
    pub n_layers: usize,
    /// `--n-kv-heads`.
    pub n_kv_heads: usize,
    /// `--head-dim`.
    pub head_dim: usize,
}

/// `recommend --json`'s `recommendation` object
/// (`RecommendResult.to_dict()`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CliRecommendation {
    /// Recommended compression method (a registry name, e.g.
    /// `turboquant_rvq`).
    pub method: String,
    /// Method-specific knobs (e.g. `bit_width_inlier`, `seed`).
    #[serde(default)]
    pub knobs: Map<String, Value>,
    /// Key-cache compression ratio (accounting).
    pub key_accounting_ratio: f64,
    /// Whether resident memory is likely to actually drop. Informational
    /// only — AutoPilot does **not** treat `false` as "won't fit" (the
    /// default `everyday` goal reports `false` on every Mac).
    pub resident_savings_likely: bool,
    /// Uncompressed (fp16) KV-cache size, MB.
    pub kv_fp16_mb: f64,
    /// Compressed KV-cache size estimate, MB (accounting-only).
    pub kv_compressed_mb_estimate: f64,
    /// Human-readable warnings.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Why this method was picked.
    pub rationale: String,
}

#[derive(Deserialize)]
struct RecommendEnvelope {
    recommendation: CliRecommendation,
}

/// `methods --json`'s envelope (`cli/methods.py`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MethodsResponse {
    /// Schema version (1 today).
    pub schema_version: u32,
    /// The method `serve` uses when `--method` is omitted.
    pub default_serve_method: String,
    /// Compression byte counts are accounting-only. Defaults to `true` when
    /// absent, so a missing field never reads as a real memory win (VeloxQuant
    /// Studio's convention).
    #[serde(default = "default_true")]
    pub accounting_only: bool,
    /// The accounting caveat, in words.
    #[serde(default)]
    pub accounting_note: Option<String>,
    /// The listed methods.
    pub methods: Vec<CliMethod>,
}

fn default_true() -> bool {
    true
}

/// One `methods --json` entry (`MethodInfo.to_dict()`), reduced to the
/// fields AutoPilot uses. Unknown fields are ignored and enum-like fields
/// stay strings, so a new upstream tier or family never fails the decode.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CliMethod {
    /// Registry name.
    pub name: String,
    /// Method family (`quantization`/`eviction`/`hybrid`/...).
    #[serde(default)]
    pub family: String,
    /// Serving tier (`accounting_only`/`not_trimmable`/`crashes`/...).
    #[serde(default)]
    pub serve_tier: String,
    /// Whether `veloxquant serve` can run it.
    pub is_servable: bool,
    /// Why it can't be served, when `is_servable` is false.
    #[serde(default)]
    pub unsupported_reason: Option<String>,
}

/// Inputs for `auto-config --json` (`cli/auto_config.py`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoConfigCliRequest {
    /// `--head-dim`.
    pub head_dim: usize,
    /// `--seq-len`.
    pub seq_len: usize,
    /// `--n-layers`.
    pub n_layers: usize,
    /// `--batch-size`.
    pub batch_size: usize,
    /// `--total-memory-bytes`; `None` lets the CLI detect it itself.
    pub total_memory_bytes: Option<u64>,
}

/// `auto-config --json`'s output.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AutoConfigResponse {
    /// The selected configuration.
    pub config: AutoConfigSelection,
    /// The CLI's explanation.
    pub reason: String,
}

/// `auto-config`'s `config` object: `method`/`head_dim` plus only the
/// selected method's own knobs.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AutoConfigSelection {
    /// Selected method.
    pub method: String,
    /// Head dimension the config was computed for.
    pub head_dim: usize,
    /// Every other key (e.g. `bit_width_inlier`, `kivi_group_size`).
    #[serde(flatten)]
    pub knobs: Map<String, Value>,
}

/// `recommend ... --json` arguments (kebab-case flags).
pub fn recommend_args(req: &RecommendCliRequest) -> Vec<String> {
    vec![
        "recommend".into(),
        "--chip".into(),
        req.chip.clone(),
        "--ram-gb".into(),
        req.ram_gb.to_string(),
        "--model-class".into(),
        req.model_class.clone(),
        "--goal".into(),
        req.goal.as_str().into(),
        "--seq-len".into(),
        req.seq_len.to_string(),
        "--n-layers".into(),
        req.n_layers.to_string(),
        "--n-kv-heads".into(),
        req.n_kv_heads.to_string(),
        "--head-dim".into(),
        req.head_dim.to_string(),
        "--json".into(),
    ]
}

/// `methods --json [--servable-only]` arguments.
pub fn methods_args(servable_only: bool) -> Vec<String> {
    let mut args = vec!["methods".to_string(), "--json".to_string()];
    if servable_only {
        args.push("--servable-only".into());
    }
    args
}

/// `auto-config ... --json` arguments (kebab-case flags).
pub fn auto_config_args(req: &AutoConfigCliRequest) -> Vec<String> {
    let mut args = vec![
        "auto-config".to_string(),
        "--head-dim".into(),
        req.head_dim.to_string(),
        "--seq-len".into(),
        req.seq_len.to_string(),
        "--n-layers".into(),
        req.n_layers.to_string(),
        "--batch-size".into(),
        req.batch_size.to_string(),
    ];
    if let Some(total) = req.total_memory_bytes {
        args.push("--total-memory-bytes".into());
        args.push(total.to_string());
    }
    args.push("--json".into());
    args
}

/// A finished subprocess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    /// Exit code (`None` if terminated by a signal).
    pub exit_code: Option<i32>,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
}

/// Runs a program to completion and captures its output. Injectable so
/// AutoPilot can be tested without a Python install.
#[async_trait::async_trait]
pub trait CommandRunner: Send + Sync {
    /// Runs `program args...`. An `Err` means the process could not be
    /// launched at all; a non-zero exit is an `Ok` with that exit code.
    async fn run(&self, program: &str, args: &[String]) -> std::io::Result<CommandOutput>;
}

/// The real [`CommandRunner`], via [`tokio::process::Command`]. Both pipes
/// are drained concurrently while the child runs (so a large `methods
/// --json` payload can't deadlock on a full pipe buffer), and the child is
/// killed if the calling future is dropped.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioCommandRunner;

#[async_trait::async_trait]
impl CommandRunner for TokioCommandRunner {
    async fn run(&self, program: &str, args: &[String]) -> std::io::Result<CommandOutput> {
        let output = tokio::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output()
            .await?;
        Ok(CommandOutput {
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

/// How to invoke the `veloxquant` CLI, and the runner that invokes it.
///
/// Defaults to the `veloxquant` console script on `PATH` (Go's and
/// Kotlin's convention). [`VeloxQuantCli::python_module`] instead runs
/// `<python> -m veloxquant_mlx` (the Swift SDK's and VeloxQuant Studio's
/// convention), for environments where the console script isn't on `PATH`.
#[derive(Clone)]
pub struct VeloxQuantCli {
    program: String,
    prefix_args: Vec<String>,
    runner: Arc<dyn CommandRunner>,
}

impl fmt::Debug for VeloxQuantCli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VeloxQuantCli")
            .field("program", &self.program)
            .field("prefix_args", &self.prefix_args)
            .finish_non_exhaustive()
    }
}

impl Default for VeloxQuantCli {
    fn default() -> Self {
        Self::program(DEFAULT_CLI_PROGRAM)
    }
}

impl VeloxQuantCli {
    /// Runs the CLI as `program <subcommand> ...` (e.g. an absolute path to
    /// a venv's `veloxquant` script).
    pub fn program(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            prefix_args: Vec::new(),
            runner: Arc::new(TokioCommandRunner),
        }
    }

    /// Runs the CLI as `<interpreter> -m veloxquant_mlx <subcommand> ...`.
    pub fn python_module(interpreter: impl Into<String>) -> Self {
        Self {
            prefix_args: vec!["-m".into(), "veloxquant_mlx".into()],
            ..Self::program(interpreter)
        }
    }

    /// Replaces the subprocess runner (mainly for tests).
    pub fn with_runner(mut self, runner: Arc<dyn CommandRunner>) -> Self {
        self.runner = runner;
        self
    }

    /// The full argv (program first) for a subcommand's arguments.
    pub fn command_line(&self, subcommand_args: &[String]) -> Vec<String> {
        std::iter::once(self.program.clone())
            .chain(self.prefix_args.iter().cloned())
            .chain(subcommand_args.iter().cloned())
            .collect()
    }

    /// Runs `recommend --json` and returns its recommendation.
    pub async fn recommend(&self, req: &RecommendCliRequest) -> Result<CliRecommendation> {
        let envelope: RecommendEnvelope = self.run_json(&recommend_args(req)).await?;
        Ok(envelope.recommendation)
    }

    /// Runs `methods --json`, optionally `--servable-only`.
    pub async fn methods(&self, servable_only: bool) -> Result<MethodsResponse> {
        self.run_json(&methods_args(servable_only)).await
    }

    /// Runs `auto-config --json`.
    pub async fn auto_config(&self, req: &AutoConfigCliRequest) -> Result<AutoConfigResponse> {
        self.run_json(&auto_config_args(req)).await
    }

    async fn run_json<T: serde::de::DeserializeOwned>(
        &self,
        subcommand_args: &[String],
    ) -> Result<T> {
        let argv = self.command_line(subcommand_args);
        let command = argv.join(" ");
        let output = self.runner.run(&argv[0], &argv[1..]).await.map_err(|e| {
            VeloxQuantError::CliUnavailable {
                command: command.clone(),
                detail: e.to_string(),
            }
        })?;
        if output.exit_code != Some(0) {
            return Err(VeloxQuantError::CliCommandFailed {
                command,
                exit_code: output.exit_code,
                stderr: output.stderr.trim().to_string(),
            });
        }
        serde_json::from_str(&output.stdout).map_err(|e| VeloxQuantError::MalformedCliOutput {
            command,
            detail: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GIB: u64 = 1 << 30;

    #[test]
    fn ram_bucket_rounds_down_to_allowed_values() {
        assert_eq!(ram_bucket(4 * GIB), None);
        assert_eq!(ram_bucket(8 * GIB), Some(8));
        assert_eq!(ram_bucket(18 * GIB), Some(16));
        assert_eq!(ram_bucket(24 * GIB), Some(24));
        assert_eq!(ram_bucket(36 * GIB), Some(36));
        assert_eq!(ram_bucket(1024 * GIB), Some(512));
    }

    #[test]
    fn model_class_rounds_up() {
        assert_eq!(model_class_for(1_000_000_000), Some("1B"));
        assert_eq!(model_class_for(8_000_000_000), Some("14B"));
        assert_eq!(model_class_for(7_000_000_000), Some("7B"));
        assert_eq!(model_class_for(30_000_000_000), Some("32B"));
        assert_eq!(model_class_for(700_000_000_000), None);
    }

    #[test]
    fn chip_argument_parses_brand_strings() {
        assert_eq!(chip_argument("Apple M1").as_deref(), Some("M1"));
        assert_eq!(chip_argument("Apple M3 Pro").as_deref(), Some("M3"));
        assert_eq!(chip_argument("Apple M4 Max").as_deref(), Some("M4"));
        assert_eq!(chip_argument("Apple M5").as_deref(), Some("M4"));
        assert_eq!(chip_argument("Apple M12 Ultra").as_deref(), Some("M4"));
        assert_eq!(chip_argument("Intel(R) Core(TM) i9"), None);
        assert_eq!(chip_argument(""), None);
        assert_eq!(chip_argument("Apple M"), None);
    }

    #[test]
    fn wont_fit_pattern_is_case_insensitive() {
        assert!(is_wont_fit_warning(
            "A 70B model will not fit in 16 GB. Its weights..."
        ));
        assert!(is_wont_fit_warning("1.2 GB SHORT OF ANY HEADROOM"));
        assert!(!is_wont_fit_warning("RAM is tight for a model this size."));
    }

    #[test]
    fn extract_bit_width_follows_knob_order() {
        let knobs = json!({"seed": 42, "kivi_bits": 4, "bit_width_inlier": 2});
        assert_eq!(extract_bit_width(knobs.as_object().unwrap()), Some(2));
        let knobs = json!({"gear_bits": 3});
        assert_eq!(extract_bit_width(knobs.as_object().unwrap()), Some(3));
        let knobs = json!({"seed": 42, "bit_width_inlier": "2"});
        assert_eq!(extract_bit_width(knobs.as_object().unwrap()), None);
    }

    #[test]
    fn recommend_args_match_cli_flags() {
        let req = RecommendCliRequest {
            chip: "M3".into(),
            ram_gb: 16,
            model_class: "14B".into(),
            goal: RecommendGoal::MaxContext,
            seq_len: 8192,
            n_layers: 36,
            n_kv_heads: 8,
            head_dim: 128,
        };
        assert_eq!(
            recommend_args(&req).join(" "),
            "recommend --chip M3 --ram-gb 16 --model-class 14B --goal max_context \
             --seq-len 8192 --n-layers 36 --n-kv-heads 8 --head-dim 128 --json"
        );
    }

    #[test]
    fn methods_and_auto_config_args_match_cli_flags() {
        assert_eq!(methods_args(false).join(" "), "methods --json");
        assert_eq!(
            methods_args(true).join(" "),
            "methods --json --servable-only"
        );
        let mut req = AutoConfigCliRequest {
            head_dim: 128,
            seq_len: 8192,
            n_layers: 36,
            batch_size: 1,
            total_memory_bytes: None,
        };
        assert_eq!(
            auto_config_args(&req).join(" "),
            "auto-config --head-dim 128 --seq-len 8192 --n-layers 36 --batch-size 1 --json"
        );
        req.total_memory_bytes = Some(17_179_869_184);
        assert!(auto_config_args(&req)
            .join(" ")
            .ends_with("--total-memory-bytes 17179869184 --json"));
    }

    #[test]
    fn goal_wire_values_match_cli_choices() {
        let all = [
            RecommendGoal::Everyday,
            RecommendGoal::MaxKeyAccounting,
            RecommendGoal::MaxContext,
            RecommendGoal::BestQuality,
            RecommendGoal::ConstantMemory,
        ];
        let wire: Vec<_> = all.iter().map(|g| g.as_str()).collect();
        assert_eq!(
            wire,
            [
                "everyday",
                "max_key_accounting",
                "max_context",
                "best_quality",
                "constant_memory"
            ]
        );
    }

    #[test]
    fn command_line_for_console_script_and_python_module() {
        let args = methods_args(true);
        assert_eq!(
            VeloxQuantCli::default().command_line(&args),
            ["veloxquant", "methods", "--json", "--servable-only"]
        );
        assert_eq!(
            VeloxQuantCli::python_module("/usr/bin/python3").command_line(&args),
            [
                "/usr/bin/python3",
                "-m",
                "veloxquant_mlx",
                "methods",
                "--json",
                "--servable-only"
            ]
        );
    }

    /// Real `recommend --json` output captured from VeloxQuant-MLX
    /// (M3/16 GB/14B/everyday, 8192 tokens, Qwen3-8B's shape).
    const REAL_RECOMMEND_OUTPUT: &str = r#"{
  "request": {"chip": "M3", "ram_gb": 16, "model_class": "14B", "goal": "everyday",
              "seq_len": 8192, "n_layers": 36, "n_kv_heads": 8, "head_dim": 128},
  "recommendation": {
    "method": "turboquant_rvq",
    "knobs": {"bit_width_inlier": 1, "seed": 42},
    "key_accounting_ratio": 7.5,
    "resident_savings_likely": false,
    "kv_fp16_mb": 1152.0,
    "kv_compressed_mb_estimate": 153.6,
    "warnings": ["RAM is tight for a model this size."],
    "rationale": "The safe everyday pick."
  }
}"#;

    #[test]
    fn decodes_real_recommend_output() {
        let env: RecommendEnvelope = serde_json::from_str(REAL_RECOMMEND_OUTPUT).unwrap();
        let rec = env.recommendation;
        assert_eq!(rec.method, "turboquant_rvq");
        assert_eq!(extract_bit_width(&rec.knobs), Some(1));
        assert!(!rec.resident_savings_likely);
        assert_eq!(rec.kv_fp16_mb, 1152.0);
        assert_eq!(rec.warnings.len(), 1);
    }

    #[test]
    fn decodes_methods_envelope_leniently() {
        let raw = json!({
            "schema_version": 1,
            "default_serve_method": "turboquant_rvq",
            "methods": [
                {"name": "turboquant_rvq", "family": "quantization",
                 "serve_tier": "accounting_only", "is_servable": true,
                 "blurb": "x", "coverage": "keys_only", "field_schema": []},
                {"name": "future", "family": "brand_new_family",
                 "serve_tier": "brand_new_tier", "is_servable": false,
                 "unsupported_reason": "not yet"}
            ]
        });
        let parsed: MethodsResponse = serde_json::from_value(raw).unwrap();
        // accounting_only absent -> true (never reads as a real memory win).
        assert!(parsed.accounting_only);
        assert_eq!(parsed.methods.len(), 2);
        assert_eq!(parsed.methods[1].family, "brand_new_family");
        assert_eq!(
            parsed.methods[1].unsupported_reason.as_deref(),
            Some("not yet")
        );
    }

    #[test]
    fn decodes_real_auto_config_output_with_flattened_knobs() {
        let raw = r#"{
          "workload": {"head_dim": 128, "seq_len": 8192, "n_layers": 36, "batch_size": 1},
          "hardware": {"total_memory_bytes": 17179869184, "active_memory_bytes": 0},
          "config": {"method": "kivi", "head_dim": 128, "bit_width_inlier": 2, "kivi_group_size": 32},
          "reason": "2048 <= seq_len=8192 < 16384 (mid-length context): selected kivi"
        }"#;
        let parsed: AutoConfigResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.config.method, "kivi");
        assert_eq!(parsed.config.head_dim, 128);
        assert_eq!(parsed.config.knobs.len(), 2);
        assert_eq!(extract_bit_width(&parsed.config.knobs), Some(2));
    }

    struct CannedRunner(std::io::Result<CommandOutput>);

    #[async_trait::async_trait]
    impl CommandRunner for CannedRunner {
        async fn run(&self, _: &str, _: &[String]) -> std::io::Result<CommandOutput> {
            match &self.0 {
                Ok(out) => Ok(out.clone()),
                Err(e) => Err(std::io::Error::new(e.kind(), e.to_string())),
            }
        }
    }

    fn cli_with(result: std::io::Result<CommandOutput>) -> VeloxQuantCli {
        VeloxQuantCli::default().with_runner(Arc::new(CannedRunner(result)))
    }

    #[tokio::test]
    async fn launch_failure_maps_to_cli_unavailable() {
        let cli = cli_with(Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No such file or directory",
        )));
        let err = cli.methods(true).await.unwrap_err();
        match err {
            VeloxQuantError::CliUnavailable { command, detail } => {
                assert_eq!(command, "veloxquant methods --json --servable-only");
                assert!(detail.contains("No such file"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn nonzero_exit_maps_to_cli_command_failed_with_stderr() {
        let cli = cli_with(Ok(CommandOutput {
            exit_code: Some(2),
            stdout: String::new(),
            stderr: "error: argument --chip: invalid choice\n".into(),
        }));
        let err = cli.methods(false).await.unwrap_err();
        match err {
            VeloxQuantError::CliCommandFailed {
                exit_code, stderr, ..
            } => {
                assert_eq!(exit_code, Some(2));
                assert_eq!(stderr, "error: argument --chip: invalid choice");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_json_stdout_maps_to_malformed_cli_output() {
        let cli = cli_with(Ok(CommandOutput {
            exit_code: Some(0),
            stdout: "VeloxQuant-MLX method recommender\n  method=kivi\n".into(),
            stderr: String::new(),
        }));
        let err = cli.methods(false).await.unwrap_err();
        assert!(matches!(err, VeloxQuantError::MalformedCliOutput { .. }));
    }

    /// Exercises the real `tokio::process` path end to end with a stand-in
    /// executable (a shell script printing canned JSON) — no Python needed.
    #[cfg(unix)]
    #[tokio::test]
    async fn tokio_runner_runs_a_real_process() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("vq-cli-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake-veloxquant");
        std::fs::write(
            &script,
            "#!/bin/sh\n\
             if [ \"$1\" = methods ] && [ \"$3\" = --servable-only ]; then\n\
               echo '{\"schema_version\":1,\"default_serve_method\":\"kivi\",\"accounting_only\":true,\"methods\":[{\"name\":\"kivi\",\"is_servable\":true}]}'\n\
               exit 0\n\
             fi\n\
             echo \"unexpected args: $*\" >&2\n\
             exit 3\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let cli = VeloxQuantCli::program(script.to_string_lossy());
        let methods = cli.methods(true).await.unwrap();
        assert_eq!(methods.default_serve_method, "kivi");
        assert_eq!(methods.methods[0].name, "kivi");

        let err = cli.methods(false).await.unwrap_err();
        assert!(matches!(
            err,
            VeloxQuantError::CliCommandFailed {
                exit_code: Some(3),
                ..
            }
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
