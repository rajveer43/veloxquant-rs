//! AutoPilot: inspect the host, pick a compatible model, and choose a
//! compression strategy — with every decision recorded in an
//! [`AutoPilotPlan`] — then hand back a ready-to-use [`AutoPilotSession`].
//!
//! This is a port of Go's `Client.AutoPilot` (`veloxquant-go/autopilot.go`)
//! with one deliberate change, the same one the Swift SDK made:
//!
//! - **Hardware inspection, model selection, context length, safety margin,
//!   and the plan** follow Go: the curated registry ranked by
//!   [`ModelRegistry::recommend_scored`](crate::ModelRegistry::recommend_scored)
//!   (a port of Go's `RecommendScored`) against available memory, or a
//!   pinned model looked up by name; an 8192-token default context; a 15%
//!   safety margin.
//! - **The compression strategy is never computed in Rust.** Go's AutoPilot
//!   calls its local, pure-Go optimizer; this one shells out to the real
//!   VeloxQuant-MLX CLI instead (see [`cli`]): `recommend --json` picks the
//!   method, `recommend`'s warnings are checked against the TS SDK's
//!   won't-fit pattern, `methods --servable-only` confirms `serve` can run
//!   the method, and `auto-config --json`'s serve-safe pool is the fallback
//!   when it can't. The Rust crate's own `OptimizationService` is not
//!   consulted.
//!
//! AutoPilot does not launch or reload the runtime (neither Go's nor
//! Kotlin's does, and this SDK has no process-ownership concept — see
//! `benchmark`'s module docs). [`AutoPilotSession`] chats with whatever the
//! configured runtime is serving, using the planned model id; start the
//! runtime with `plan.method`/`plan.bits` yourself (e.g. `veloxquant serve
//! --model <plan.selected_model.name> --method <plan.method> --bits
//! <plan.bits>`).
//!
//! Requires the `veloxquant` CLI (VeloxQuant-MLX) and, for `recommend
//! --chip`, an Apple Silicon Mac (or a [`SystemInfo`] override via
//! [`AutoPilot::with_hardware`]).

pub mod cli;

use veloxquant_core::{format_bytes, Result, VeloxQuantError};
use veloxquant_memory::{MemoryEstimate, MemoryRequest, Precision};
use veloxquant_openai::{ChatResponse, Message};
use veloxquant_system::SystemInfo;

use crate::models::{ModelInfo, ModelRecommendationRequest, Task};
use crate::Client;

pub use cli::{
    AutoConfigCliRequest, AutoConfigResponse, AutoConfigSelection, CliMethod, CliRecommendation,
    CommandOutput, CommandRunner, MethodsResponse, RecommendCliRequest, RecommendGoal,
    TokioCommandRunner, VeloxQuantCli,
};

/// Go's `defaultContextLength`.
pub const DEFAULT_CONTEXT_LENGTH: usize = 8192;

/// Fraction of available memory reserved as headroom — Go's
/// `safetyMarginRatio`.
pub const SAFETY_MARGIN_RATIO: f64 = 0.15;

/// Which model AutoPilot should plan for — Go's `AutoPilotConfig.Model`,
/// plus a caller-described model outside the curated registry.
#[derive(Debug, Clone, Default)]
pub enum ModelSelection {
    /// Rank the registry for [`AutoPilotConfig::task`] against available
    /// memory (Go's `Model: ""`/`"auto"`).
    #[default]
    Auto,
    /// A curated registry entry by exact name; an unknown name is
    /// [`VeloxQuantError::ModelNotFound`] (Go's `ErrModelNotFound`).
    Named(String),
    /// A model you describe yourself. Its architecture must have non-zero
    /// `num_layers`/`num_kv_heads`/`head_dim`, and a non-zero
    /// `parameter_count` unless [`AutoPilotConfig::model_class`] is set.
    Custom(ModelInfo),
}

/// An AutoPilot request — Go's `AutoPilotConfig` plus the TS/Kotlin/Swift
/// `goal`/`force` fields and a `--model-class` override.
#[derive(Debug, Clone, Default)]
pub struct AutoPilotConfig {
    /// Task used to rank models under [`ModelSelection::Auto`]; `None`
    /// considers every task.
    pub task: Option<Task>,
    /// Model to plan for.
    pub model: ModelSelection,
    /// Context length; `None` (or `Some(0)`) uses
    /// [`DEFAULT_CONTEXT_LENGTH`].
    pub context_length: Option<usize>,
    /// `recommend --goal`.
    pub goal: RecommendGoal,
    /// Overrides the `--model-class` derived from the model's parameter
    /// count.
    pub model_class: Option<String>,
    /// Proceed even when `recommend` warns the workload won't fit.
    pub force: bool,
}

/// Every decision AutoPilot made — Go's `AutoPilotPlan`, extended with the
/// CLI inputs/outputs this shell-out design adds.
#[derive(Debug, Clone)]
pub struct AutoPilotPlan {
    /// The host, as detected (or as overridden).
    pub hardware: SystemInfo,
    /// The chosen model.
    pub selected_model: ModelInfo,
    /// Why it was chosen (the ranking reason, or that it was pinned).
    pub selection_reason: String,
    /// Context length planned for.
    pub context_length: usize,
    /// Offline estimate for the chosen model at `context_length` (fp16
    /// weights/KV vs. int4 KV). Accounting-only, like every compression
    /// byte count here; it does not drive the compression choice.
    pub memory_estimate: MemoryEstimate,
    /// 15% of available memory reserved as headroom.
    pub safety_margin_bytes: u64,
    /// The exact `recommend` inputs used.
    pub recommend_request: RecommendCliRequest,
    /// `recommend`'s answer.
    pub recommendation: CliRecommendation,
    /// `recommend` warnings matching the won't-fit pattern (non-empty only
    /// when `force` overrode them).
    pub wont_fit_warnings: Vec<String>,
    /// The compression method to serve with.
    pub method: String,
    /// The bit width to serve with, when the chosen knobs name one.
    pub bits: Option<u32>,
    /// Whether `recommend`'s method wasn't servable and `auto-config`'s
    /// pick was used instead.
    pub used_serve_safe_fallback: bool,
    /// `auto-config`'s reason, when the fallback was used.
    pub fallback_reason: Option<String>,
    /// `methods --json`'s `accounting_only` flag: compression byte counts
    /// don't correspond to resident memory reduction.
    pub accounting_only: bool,
    /// Ordered, human-readable trail of every decision above.
    pub decisions: Vec<String>,
}

impl AutoPilotPlan {
    /// Go's `AutoPilotPlan.Reason`: the selection reason joined with the
    /// compression rationale.
    pub fn reason(&self) -> String {
        format!(
            "{}; {}",
            self.selection_reason, self.recommendation.rationale
        )
    }
}

/// A ready-to-use session bound to AutoPilot's chosen model — Go's
/// `Session`.
#[derive(Clone)]
pub struct AutoPilotSession {
    client: Client,
    plan: AutoPilotPlan,
}

impl std::fmt::Debug for AutoPilotSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoPilotSession")
            .field("runtime_url", &self.client.runtime_url())
            .field("plan", &self.plan)
            .finish()
    }
}

impl AutoPilotSession {
    /// Every decision behind this session — Go's `Session.Plan()`.
    pub fn plan(&self) -> &AutoPilotPlan {
        &self.plan
    }

    /// The client requests go through.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Sends one user message with the planned model — Go's
    /// `Session.Chat`.
    pub async fn chat(&self, prompt: impl Into<String>) -> Result<ChatResponse> {
        self.client
            .chat()?
            .create(
                self.plan.selected_model.name.clone(),
                vec![Message::user(prompt.into())],
            )
            .await
    }

    /// Streams a reply to one user message with the planned model.
    pub async fn stream(
        &self,
        prompt: impl Into<String>,
    ) -> Result<veloxquant_openai::streaming::ChatStream> {
        self.client
            .chat()?
            .stream(
                self.plan.selected_model.name.clone(),
                vec![Message::user(prompt.into())],
            )
            .await
    }
}

/// "This workload likely won't fit" as data: the matching warnings plus the
/// full recommendation. [`AutoPilotFitError::into_error`] gives the same
/// information as the [`VeloxQuantError::AutoPilotWontFit`] that
/// [`AutoPilot::start`] returns, so the two entry points can't drift.
#[derive(Debug, Clone)]
pub struct AutoPilotFitError {
    /// `recommend` warnings that matched the won't-fit pattern.
    pub warnings: Vec<String>,
    /// The full `recommend` result.
    pub recommendation: CliRecommendation,
}

impl AutoPilotFitError {
    /// Converts to the error [`AutoPilot::start`] returns.
    pub fn into_error(self) -> VeloxQuantError {
        VeloxQuantError::AutoPilotWontFit {
            method: self.recommendation.method,
            warnings: self.warnings,
        }
    }
}

/// [`AutoPilot::try_start`]'s result.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // returned once per AutoPilot run; boxing buys nothing
pub enum AutoPilotOutcome {
    /// Ready to use.
    Started(AutoPilotSession),
    /// `recommend` warned the workload won't fit and `force` was false.
    WontFit(AutoPilotFitError),
}

/// Plans AutoPilot sessions for a [`Client`]. Use [`Client::autopilot`] for
/// the defaults; construct this directly to point at a specific CLI
/// install or to override the detected hardware.
#[derive(Clone)]
pub struct AutoPilot {
    client: Client,
    cli: VeloxQuantCli,
    hardware: Option<SystemInfo>,
}

impl std::fmt::Debug for AutoPilot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoPilot")
            .field("runtime_url", &self.client.runtime_url())
            .field("cli", &self.cli)
            .field("hardware", &self.hardware)
            .finish()
    }
}

impl AutoPilot {
    /// An AutoPilot using the `veloxquant` CLI on `PATH` and live hardware
    /// detection.
    pub fn new(client: Client) -> Self {
        Self {
            client,
            cli: VeloxQuantCli::default(),
            hardware: None,
        }
    }

    /// Uses `cli` for the compression shell-outs.
    pub fn with_cli(mut self, cli: VeloxQuantCli) -> Self {
        self.cli = cli;
        self
    }

    /// Plans against `hardware` instead of detecting the host (e.g. to plan
    /// for a different target Mac, or in tests).
    pub fn with_hardware(mut self, hardware: SystemInfo) -> Self {
        self.hardware = Some(hardware);
        self
    }

    /// Plans a session, returning [`AutoPilotOutcome::WontFit`] (rather than
    /// an error) when `recommend` warns the workload won't fit and
    /// `config.force` is false.
    ///
    /// Errors: [`VeloxQuantError::ModelNotFound`] /
    /// [`VeloxQuantError::NoModelFits`] (selection),
    /// [`VeloxQuantError::InvalidRequest`] (an unusable custom model),
    /// [`VeloxQuantError::UnsupportedPlatform`] (no recognizable Apple
    /// M-series chip, or under 8 GB of RAM, for `recommend`), and
    /// [`VeloxQuantError::CliUnavailable`] /
    /// [`VeloxQuantError::CliCommandFailed`] /
    /// [`VeloxQuantError::MalformedCliOutput`] (shell-outs).
    pub async fn try_start(&self, config: AutoPilotConfig) -> Result<AutoPilotOutcome> {
        let mut decisions = Vec::new();

        let hardware = match &self.hardware {
            Some(h) => h.clone(),
            None => self.client.system().info().await,
        };
        decisions.push(format!(
            "hardware: {} ({}), {} total, {} available",
            if hardware.cpu_model.is_empty() {
                "unknown CPU"
            } else {
                &hardware.cpu_model
            },
            hardware.architecture,
            format_bytes(hardware.total_memory_bytes),
            format_bytes(hardware.available_memory_bytes)
        ));

        let requested = config.context_length.unwrap_or(0);
        let context_length = if requested > 0 {
            requested
        } else {
            DEFAULT_CONTEXT_LENGTH
        };
        decisions.push(format!(
            "context length: {context_length} tokens ({})",
            if requested > 0 {
                "requested"
            } else {
                "default"
            }
        ));

        let (model, selection_reason) = self.select_model(&config, &hardware)?;
        decisions.push(format!("model: {} — {selection_reason}", model.name));

        let memory_estimate = self.client.memory().estimate(&MemoryRequest::new(
            model.architecture.clone(),
            context_length,
            Precision::Fp16,
        ))?;
        let safety_margin_bytes =
            (hardware.available_memory_bytes as f64 * SAFETY_MARGIN_RATIO) as u64;
        decisions.push(format!(
            "offline estimate (fp16 weights + fp16 KV): {}; {}: {} (accounting-only); safety margin {}",
            format_bytes(memory_estimate.total_memory_bytes),
            memory_estimate.recommended_strategy,
            format_bytes(memory_estimate.optimized_total_bytes),
            format_bytes(safety_margin_bytes)
        ));

        let recommend_request = make_recommend_request(&config, &model, &hardware, context_length)?;
        let recommendation = self.cli.recommend(&recommend_request).await?;
        decisions.push(format!(
            "recommend (--chip {} --ram-gb {} --model-class {} --goal {}): {} — {}",
            recommend_request.chip,
            recommend_request.ram_gb,
            recommend_request.model_class,
            recommend_request.goal.as_str(),
            recommendation.method,
            recommendation.rationale
        ));

        let wont_fit_warnings: Vec<String> = recommendation
            .warnings
            .iter()
            .filter(|w| cli::is_wont_fit_warning(w))
            .cloned()
            .collect();
        if !wont_fit_warnings.is_empty() {
            if !config.force {
                return Ok(AutoPilotOutcome::WontFit(AutoPilotFitError {
                    warnings: wont_fit_warnings,
                    recommendation,
                }));
            }
            decisions.push(format!(
                "won't-fit warnings overridden by force: {}",
                wont_fit_warnings.join(" | ")
            ));
        }

        let servable = self.cli.methods(true).await?;
        let recommended_bits = cli::extract_bit_width(&recommendation.knobs);
        let (method, bits, fallback_reason) = if servable
            .methods
            .iter()
            .any(|m| m.name == recommendation.method && m.is_servable)
        {
            decisions.push(format!(
                "serve: {} is servable{}",
                recommendation.method,
                bits_suffix(recommended_bits)
            ));
            (recommendation.method.clone(), recommended_bits, None)
        } else {
            let auto = self
                .cli
                .auto_config(&AutoConfigCliRequest {
                    head_dim: model.architecture.head_dim,
                    seq_len: context_length,
                    n_layers: model.architecture.num_layers,
                    batch_size: 1,
                    total_memory_bytes: (hardware.total_memory_bytes > 0)
                        .then_some(hardware.total_memory_bytes),
                })
                .await?;
            let bits = cli::extract_bit_width(&auto.config.knobs);
            decisions.push(format!(
                "serve: {} is not servable; auto-config picked {}{} — {}",
                recommendation.method,
                auto.config.method,
                bits_suffix(bits),
                auto.reason
            ));
            (auto.config.method, bits, Some(auto.reason))
        };

        let plan = AutoPilotPlan {
            hardware,
            selected_model: model,
            selection_reason,
            context_length,
            memory_estimate,
            safety_margin_bytes,
            recommend_request,
            recommendation,
            wont_fit_warnings,
            method,
            bits,
            used_serve_safe_fallback: fallback_reason.is_some(),
            fallback_reason,
            accounting_only: servable.accounting_only,
            decisions,
        };
        Ok(AutoPilotOutcome::Started(AutoPilotSession {
            client: self.client.clone(),
            plan,
        }))
    }

    /// Like [`AutoPilot::try_start`], but a won't-fit outcome is returned
    /// as [`VeloxQuantError::AutoPilotWontFit`].
    pub async fn start(&self, config: AutoPilotConfig) -> Result<AutoPilotSession> {
        match self.try_start(config).await? {
            AutoPilotOutcome::Started(session) => Ok(session),
            AutoPilotOutcome::WontFit(fit) => Err(fit.into_error()),
        }
    }

    /// Go's `selectModel`.
    fn select_model(
        &self,
        config: &AutoPilotConfig,
        hardware: &SystemInfo,
    ) -> Result<(ModelInfo, String)> {
        match &config.model {
            ModelSelection::Custom(info) => {
                let arch = &info.architecture;
                if arch.num_layers == 0 || arch.num_kv_heads == 0 || arch.head_dim == 0 {
                    return Err(VeloxQuantError::InvalidRequest(format!(
                        "autopilot: custom model {} needs non-zero num_layers, num_kv_heads, and head_dim",
                        info.name
                    )));
                }
                Ok((info.clone(), "caller-supplied model".to_string()))
            }
            ModelSelection::Named(name) => self
                .client
                .models()
                .get(name)
                .map(|info| (info, "explicitly requested".to_string()))
                .ok_or_else(|| VeloxQuantError::ModelNotFound(name.clone())),
            ModelSelection::Auto => {
                let registry = self.client.models();
                let candidates = registry.recommend_scored(
                    &ModelRecommendationRequest {
                        task: config.task,
                        available_memory_bytes: hardware.available_memory_bytes,
                    },
                    config.context_length,
                )?;
                match candidates.into_iter().next() {
                    Some(best) => Ok((best.info, best.reason)),
                    None if registry.list().is_empty() => Err(VeloxQuantError::ModelNotFound(
                        "no models in registry".to_string(),
                    )),
                    None => Err(VeloxQuantError::NoModelFits {
                        task: config.task.map(|t| format!("{t:?}").to_lowercase()),
                        available_memory_bytes: hardware.available_memory_bytes,
                    }),
                }
            }
        }
    }
}

fn bits_suffix(bits: Option<u32>) -> String {
    bits.map(|b| format!(", {b}-bit")).unwrap_or_default()
}

/// Maps the host and model onto `recommend`'s fixed buckets,
/// conservatively (RAM rounds down, model class rounds up).
fn make_recommend_request(
    config: &AutoPilotConfig,
    model: &ModelInfo,
    hardware: &SystemInfo,
    context_length: usize,
) -> Result<RecommendCliRequest> {
    let chip =
        cli::chip_argument(&hardware.cpu_model).ok_or(VeloxQuantError::UnsupportedPlatform)?;
    let ram_gb =
        cli::ram_bucket(hardware.total_memory_bytes).ok_or(VeloxQuantError::UnsupportedPlatform)?;
    let arch = &model.architecture;
    let model_class = match &config.model_class {
        Some(class) => class.clone(),
        None => cli::model_class_for(arch.parameter_count)
            .filter(|_| arch.parameter_count > 0)
            .ok_or_else(|| {
                VeloxQuantError::InvalidRequest(format!(
                    "autopilot: no `recommend --model-class` covers {} ({} parameters); set AutoPilotConfig::model_class",
                    model.name, arch.parameter_count
                ))
            })?
            .to_string(),
    };
    Ok(RecommendCliRequest {
        chip,
        ram_gb,
        model_class,
        goal: config.goal,
        seq_len: context_length,
        n_layers: arch.num_layers,
        n_kv_heads: arch.num_kv_heads,
        head_dim: arch.head_dim,
    })
}

impl Client {
    /// Runs AutoPilot with the `veloxquant` CLI on `PATH` and live hardware
    /// detection — Go's `Client.AutoPilot`. See [`AutoPilot`] to configure
    /// either, or [`AutoPilot::try_start`] to handle "won't fit" as data.
    pub async fn autopilot(&self, config: AutoPilotConfig) -> Result<AutoPilotSession> {
        AutoPilot::new(self.clone()).start(config).await
    }
}

#[cfg(test)]
mod tests;
