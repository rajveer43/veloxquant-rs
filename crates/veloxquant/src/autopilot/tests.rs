use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use veloxquant_core::OptimizationProfile;
use veloxquant_memory::ModelArchitecture;

use super::cli::CommandOutput;
use super::*;

const GIB: u64 = 1 << 30;

fn m3_16gb() -> SystemInfo {
    SystemInfo {
        platform: "macos".into(),
        architecture: "aarch64".into(),
        cpu_model: "Apple M3".into(),
        apple_silicon: true,
        total_memory_bytes: 16 * GIB,
        available_memory_bytes: 12 * GIB,
        recommended_profile: OptimizationProfile::Balanced,
    }
}

fn recommend_json(method: &str, knobs: &str, warnings: &[&str]) -> String {
    format!(
        r#"{{"request": {{}}, "recommendation": {{
            "method": "{method}", "knobs": {knobs},
            "key_accounting_ratio": 7.5, "resident_savings_likely": false,
            "kv_fp16_mb": 1152.0, "kv_compressed_mb_estimate": 153.6,
            "warnings": {}, "rationale": "the {method} rationale"}}}}"#,
        serde_json::to_string(warnings).unwrap()
    )
}

const METHODS_JSON: &str = r#"{"schema_version": 1, "default_serve_method": "turboquant_rvq",
    "accounting_only": true,
    "methods": [{"name": "turboquant_rvq", "is_servable": true},
                {"name": "kivi", "is_servable": true}]}"#;

const AUTO_CONFIG_JSON: &str = r#"{"config": {"method": "kivi", "head_dim": 128,
    "bit_width_inlier": 2, "kivi_group_size": 32}, "reason": "mid-length context: kivi"}"#;

/// Answers each subcommand with canned stdout and records every argv.
#[derive(Default)]
struct FakeCli {
    recommend: String,
    calls: Mutex<Vec<Vec<String>>>,
    fail: Option<&'static str>,
}

#[async_trait::async_trait]
impl CommandRunner for FakeCli {
    async fn run(&self, program: &str, args: &[String]) -> std::io::Result<CommandOutput> {
        let mut argv = vec![program.to_string()];
        argv.extend(args.iter().cloned());
        self.calls.lock().unwrap().push(argv);

        let sub = args[0].as_str();
        if self.fail == Some(sub) {
            return Ok(CommandOutput {
                exit_code: Some(1),
                stdout: String::new(),
                stderr: "boom".into(),
            });
        }
        let stdout = match sub {
            "recommend" => self.recommend.clone(),
            "methods" => METHODS_JSON.to_string(),
            "auto-config" => AUTO_CONFIG_JSON.to_string(),
            other => panic!("unexpected subcommand {other}"),
        };
        Ok(CommandOutput {
            exit_code: Some(0),
            stdout,
            stderr: String::new(),
        })
    }
}

impl FakeCli {
    fn new(recommend: String) -> Arc<Self> {
        Arc::new(Self {
            recommend,
            ..Default::default()
        })
    }

    fn subcommands(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|argv| argv[1].clone())
            .collect()
    }

    fn argv_for(&self, sub: &str) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|argv| argv[1] == sub)
            .cloned()
            .unwrap_or_else(|| panic!("{sub} was never run"))
    }
}

fn autopilot_with(fake: &Arc<FakeCli>, hardware: SystemInfo) -> AutoPilot {
    let client = Client::builder().build().unwrap();
    AutoPilot::new(client)
        .with_cli(VeloxQuantCli::default().with_runner(fake.clone()))
        .with_hardware(hardware)
}

fn started(outcome: AutoPilotOutcome) -> AutoPilotSession {
    match outcome {
        AutoPilotOutcome::Started(session) => session,
        AutoPilotOutcome::WontFit(fit) => panic!("unexpected won't-fit: {fit:?}"),
    }
}

#[tokio::test]
async fn servable_recommendation_is_used_directly() {
    let fake = FakeCli::new(recommend_json(
        "turboquant_rvq",
        r#"{"bit_width_inlier": 1, "seed": 42}"#,
        &["RAM is tight for a model this size."],
    ));
    let session = started(
        autopilot_with(&fake, m3_16gb())
            .try_start(AutoPilotConfig::default())
            .await
            .unwrap(),
    );
    let plan = session.plan();

    // Go's selection: the recommended Qwen3-8B wins the ranking.
    assert_eq!(plan.selected_model.name, "mlx-community/Qwen3-8B-4bit");
    assert!(plan.selection_reason.contains("memory headroom"));
    assert_eq!(plan.context_length, DEFAULT_CONTEXT_LENGTH);
    assert_eq!(plan.safety_margin_bytes, (12 * GIB) * 15 / 100);

    // The compression decision came from the CLI, verbatim.
    assert_eq!(plan.method, "turboquant_rvq");
    assert_eq!(plan.bits, Some(1));
    assert!(!plan.used_serve_safe_fallback);
    assert!(plan.fallback_reason.is_none());
    assert!(plan.accounting_only);
    // A soft warning is not a won't-fit.
    assert!(plan.wont_fit_warnings.is_empty());
    assert_eq!(plan.recommendation.warnings.len(), 1);
    assert_eq!(
        plan.reason(),
        format!("{}; the turboquant_rvq rationale", plan.selection_reason)
    );

    // recommend -> methods, no auto-config.
    assert_eq!(fake.subcommands(), ["recommend", "methods"]);
    assert_eq!(
        fake.argv_for("recommend").join(" "),
        "veloxquant recommend --chip M3 --ram-gb 16 --model-class 14B --goal everyday \
         --seq-len 8192 --n-layers 36 --n-kv-heads 8 --head-dim 128 --json"
    );
    assert_eq!(
        fake.argv_for("methods").join(" "),
        "veloxquant methods --json --servable-only"
    );

    // The decision trail covers every step, in order.
    let trail = plan.decisions.join("\n");
    for (i, needle) in [
        "hardware: Apple M3",
        "context length: 8192 tokens (default)",
        "model: mlx-community/Qwen3-8B-4bit",
        "offline estimate (fp16 weights + fp16 KV):",
        "recommend (--chip M3 --ram-gb 16 --model-class 14B --goal everyday): turboquant_rvq",
        "serve: turboquant_rvq is servable, 1-bit",
    ]
    .iter()
    .enumerate()
    {
        assert!(
            plan.decisions[i].starts_with(needle),
            "decision {i} = {:?}, trail:\n{trail}",
            plan.decisions[i]
        );
    }
}

#[tokio::test]
async fn unservable_recommendation_falls_back_to_auto_config() {
    let fake = FakeCli::new(recommend_json("rabitq", r#"{"seed": 42}"#, &[]));
    let session = started(
        autopilot_with(&fake, m3_16gb())
            .try_start(AutoPilotConfig {
                context_length: Some(4096),
                ..Default::default()
            })
            .await
            .unwrap(),
    );
    let plan = session.plan();

    assert_eq!(plan.recommendation.method, "rabitq");
    assert_eq!(plan.method, "kivi");
    assert_eq!(plan.bits, Some(2));
    assert!(plan.used_serve_safe_fallback);
    assert_eq!(
        plan.fallback_reason.as_deref(),
        Some("mid-length context: kivi")
    );
    assert_eq!(fake.subcommands(), ["recommend", "methods", "auto-config"]);
    assert_eq!(
        fake.argv_for("auto-config").join(" "),
        format!(
            "veloxquant auto-config --head-dim 128 --seq-len 4096 --n-layers 36 --batch-size 1 \
             --total-memory-bytes {} --json",
            16 * GIB
        )
    );
    assert!(plan
        .decisions
        .iter()
        .any(|d| d == "context length: 4096 tokens (requested)"));
    assert!(plan
        .decisions
        .last()
        .unwrap()
        .starts_with("serve: rabitq is not servable; auto-config picked kivi, 2-bit"));
}

const WONT_FIT: &str = "A 14B model will not fit in 8 GB. Its weights alone need about 8.0 GB, \
                        leaving you 0.5 GB short of any headroom.";

#[tokio::test]
async fn wont_fit_without_force_is_returned_as_data_and_as_error() {
    let fake = FakeCli::new(recommend_json(
        "turboquant_rvq",
        r#"{"bit_width_inlier": 1}"#,
        &[WONT_FIT, "RAM is tight."],
    ));
    let autopilot = autopilot_with(&fake, m3_16gb());

    match autopilot
        .try_start(AutoPilotConfig::default())
        .await
        .unwrap()
    {
        AutoPilotOutcome::WontFit(fit) => {
            assert_eq!(fit.warnings, [WONT_FIT]);
            assert_eq!(fit.recommendation.method, "turboquant_rvq");
            match fit.into_error() {
                VeloxQuantError::AutoPilotWontFit { method, warnings } => {
                    assert_eq!(method, "turboquant_rvq");
                    assert_eq!(warnings, [WONT_FIT]);
                }
                other => panic!("unexpected error {other:?}"),
            }
        }
        AutoPilotOutcome::Started(_) => panic!("should not have started"),
    }
    // Stopped before checking servability.
    assert_eq!(fake.subcommands(), ["recommend"]);

    let err = autopilot
        .start(AutoPilotConfig::default())
        .await
        .unwrap_err();
    assert!(matches!(err, VeloxQuantError::AutoPilotWontFit { .. }));
    assert!(err.to_string().contains("force: true"));
}

#[tokio::test]
async fn force_overrides_wont_fit_and_records_it() {
    let fake = FakeCli::new(recommend_json(
        "turboquant_rvq",
        r#"{"bit_width_inlier": 1}"#,
        &[WONT_FIT],
    ));
    let session = autopilot_with(&fake, m3_16gb())
        .start(AutoPilotConfig {
            force: true,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(session.plan().wont_fit_warnings, [WONT_FIT]);
    assert!(session
        .plan()
        .decisions
        .iter()
        .any(|d| d.starts_with("won't-fit warnings overridden by force")));
}

#[tokio::test]
async fn named_model_is_used_and_unknown_name_fails_before_any_shell_out() {
    let fake = FakeCli::new(recommend_json(
        "turboquant_rvq",
        r#"{"bit_width_inlier": 1}"#,
        &[],
    ));
    let autopilot = autopilot_with(&fake, m3_16gb());

    let session = autopilot
        .start(AutoPilotConfig {
            model: ModelSelection::Named("mlx-community/gemma-2-9b-4bit".into()),
            goal: RecommendGoal::BestQuality,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        session.plan().selected_model.name,
        "mlx-community/gemma-2-9b-4bit"
    );
    assert_eq!(session.plan().selection_reason, "explicitly requested");
    let recommend = fake.argv_for("recommend").join(" ");
    assert!(recommend.contains("--goal best_quality"));
    assert!(recommend.contains("--head-dim 256 "));

    let before = fake.subcommands().len();
    let err = autopilot
        .start(AutoPilotConfig {
            model: ModelSelection::Named("nope".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, VeloxQuantError::ModelNotFound(ref n) if n == "nope"));
    assert_eq!(fake.subcommands().len(), before);
}

#[tokio::test]
async fn auto_selection_with_no_fitting_model_is_no_model_fits() {
    let fake = FakeCli::new(String::new());
    let mut hardware = m3_16gb();
    hardware.available_memory_bytes = GIB; // nothing in the registry fits in 1 GiB
    let err = autopilot_with(&fake, hardware)
        .start(AutoPilotConfig {
            task: Some(Task::Coding),
            ..Default::default()
        })
        .await
        .unwrap_err();
    match err {
        VeloxQuantError::NoModelFits {
            task,
            available_memory_bytes,
        } => {
            assert_eq!(task.as_deref(), Some("coding"));
            assert_eq!(available_memory_bytes, GIB);
        }
        other => panic!("unexpected error {other:?}"),
    }
    assert!(fake.subcommands().is_empty());
}

#[tokio::test]
async fn non_apple_silicon_or_tiny_ram_is_unsupported_platform() {
    let fake = FakeCli::new(String::new());

    let mut intel = m3_16gb();
    intel.cpu_model = "Intel(R) Core(TM) i9-9980HK".into();
    let err = autopilot_with(&fake, intel)
        .start(AutoPilotConfig::default())
        .await
        .unwrap_err();
    assert!(matches!(err, VeloxQuantError::UnsupportedPlatform));

    let mut tiny = m3_16gb();
    tiny.total_memory_bytes = 4 * GIB;
    tiny.available_memory_bytes = 3 * GIB;
    let err = autopilot_with(&fake, tiny)
        .start(AutoPilotConfig {
            model: ModelSelection::Named("mlx-community/Qwen3-8B-4bit".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, VeloxQuantError::UnsupportedPlatform));

    assert!(fake.subcommands().is_empty());
}

fn custom(parameter_count: u64, num_layers: usize) -> ModelInfo {
    ModelInfo {
        name: "my-org/custom".into(),
        architecture: ModelArchitecture {
            name: "custom".into(),
            num_layers,
            num_kv_heads: 4,
            head_dim: 64,
            hidden_size: 2048,
            parameter_count,
        },
        supported: true,
        recommended: false,
        tasks: vec![],
    }
}

#[tokio::test]
async fn custom_models_are_validated_and_model_class_can_be_overridden() {
    let fake = FakeCli::new(recommend_json(
        "turboquant_rvq",
        r#"{"bit_width_inlier": 1}"#,
        &[],
    ));
    let autopilot = autopilot_with(&fake, m3_16gb());

    let err = autopilot
        .start(AutoPilotConfig {
            model: ModelSelection::Custom(custom(3_000_000_000, 0)),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, VeloxQuantError::InvalidRequest(_)));

    // Unknown parameter count and no override: can't pick --model-class.
    let err = autopilot
        .start(AutoPilotConfig {
            model: ModelSelection::Custom(custom(0, 24)),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(matches!(err, VeloxQuantError::InvalidRequest(ref m) if m.contains("model_class")));

    let session = autopilot
        .start(AutoPilotConfig {
            model: ModelSelection::Custom(custom(0, 24)),
            model_class: Some("3B".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(session.plan().selection_reason, "caller-supplied model");
    let recommend = fake.argv_for("recommend").join(" ");
    assert!(recommend.contains("--model-class 3B --goal everyday"));
    assert!(recommend.contains("--n-layers 24 --n-kv-heads 4 --head-dim 64"));
}

#[tokio::test]
async fn cli_failures_propagate() {
    let fake = Arc::new(FakeCli {
        recommend: recommend_json("turboquant_rvq", "{}", &[]),
        fail: Some("methods"),
        ..Default::default()
    });
    let err = autopilot_with(&fake, m3_16gb())
        .start(AutoPilotConfig::default())
        .await
        .unwrap_err();
    match err {
        VeloxQuantError::CliCommandFailed {
            command, stderr, ..
        } => {
            assert_eq!(command, "veloxquant methods --json --servable-only");
            assert_eq!(stderr, "boom");
        }
        other => panic!("unexpected error {other:?}"),
    }
}

/// A one-shot HTTP server that records the first request body and replies
/// with a canned chat completion.
async fn spawn_capturing_chat_server() -> (String, Arc<Mutex<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(String::new()));
    let sink = captured.clone();
    tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let mut buf = vec![0u8; 16384];
        let n = socket.read(&mut buf).await.unwrap_or(0);
        *sink.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
        let body = r#"{"id":"1","model":"mlx-community/Qwen3-8B-4bit","text":"hi from runtime"}"#;
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let _ = socket.write_all(header.as_bytes()).await;
        let _ = socket.write_all(body.as_bytes()).await;
        let _ = socket.shutdown().await;
    });
    (format!("http://{addr}"), captured)
}

#[tokio::test]
async fn session_chat_uses_the_planned_model() {
    let (base_url, captured) = spawn_capturing_chat_server().await;
    let fake = FakeCli::new(recommend_json(
        "turboquant_rvq",
        r#"{"bit_width_inlier": 1}"#,
        &[],
    ));
    let client = Client::builder().runtime_url(&base_url).build().unwrap();
    let session = AutoPilot::new(client)
        .with_cli(VeloxQuantCli::default().with_runner(fake.clone()))
        .with_hardware(m3_16gb())
        .start(AutoPilotConfig::default())
        .await
        .unwrap();

    let reply = session.chat("hello").await.unwrap();
    assert_eq!(reply.text, "hi from runtime");

    let request = captured.lock().unwrap().clone();
    assert!(request.starts_with("POST /v1/chat/completions"));
    assert!(request.contains(r#""model":"mlx-community/Qwen3-8B-4bit""#));
    assert!(request.contains(r#""content":"hello""#));
    assert_eq!(session.client().runtime_url(), base_url);
}

/// Hand-run checks against the **real** VeloxQuant-MLX CLI. They need the
/// `veloxquant` console script on `PATH` (`pip install veloxquant-mlx`, or
/// set `VQ_AUTOPILOT_PYTHON` to an interpreter that can import
/// `veloxquant_mlx`, to use `python -m veloxquant_mlx` instead), so they are
/// `#[ignore]`d and never claimed as CI coverage.
///
/// ```sh
/// cargo test -p veloxquant --features autopilot -- --ignored autopilot_manual --nocapture
/// ```
mod manual_tests {
    use super::*;

    fn real_cli() -> VeloxQuantCli {
        match std::env::var("VQ_AUTOPILOT_PYTHON") {
            Ok(python) => VeloxQuantCli::python_module(python),
            Err(_) => VeloxQuantCli::default(),
        }
    }

    /// Fixed M3/16 GB hardware, so the result is comparable across machines;
    /// only the CLI is real.
    #[tokio::test]
    #[ignore = "requires the real `veloxquant` CLI (VeloxQuant-MLX) — see module docs"]
    async fn autopilot_manual_real_cli_fixed_hardware() {
        let session = AutoPilot::new(Client::builder().build().unwrap())
            .with_cli(real_cli())
            .with_hardware(m3_16gb())
            .start(AutoPilotConfig {
                force: true,
                ..Default::default()
            })
            .await
            .expect("AutoPilot against the real CLI");
        let plan = session.plan();
        for decision in &plan.decisions {
            println!("{decision}");
        }
        assert!(!plan.method.is_empty());
        assert!(plan.accounting_only);
    }

    /// Live hardware detection plus the real CLI — must run on an Apple
    /// Silicon Mac. The model is pinned so the result doesn't depend on how
    /// much memory happens to be free (under `ModelSelection::Auto`, a
    /// loaded machine correctly yields `NoModelFits` before the CLI runs).
    #[tokio::test]
    #[ignore = "requires the real `veloxquant` CLI on an Apple Silicon Mac — see module docs"]
    async fn autopilot_manual_real_cli_live_hardware() {
        let outcome = AutoPilot::new(Client::builder().build().unwrap())
            .with_cli(real_cli())
            .try_start(AutoPilotConfig {
                model: ModelSelection::Named("mlx-community/Qwen3-8B-4bit".into()),
                ..Default::default()
            })
            .await
            .expect("AutoPilot against the real CLI and live hardware");
        match outcome {
            AutoPilotOutcome::Started(session) => {
                for decision in &session.plan().decisions {
                    println!("{decision}");
                }
            }
            AutoPilotOutcome::WontFit(fit) => println!("won't fit: {:?}", fit.warnings),
        }
    }
}
