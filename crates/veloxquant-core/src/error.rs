//! Error types shared across the VeloxQuant SDK.

/// The single error type returned by fallible VeloxQuant SDK operations.
///
/// This enum is `#[non_exhaustive]`: new variants may be added in minor
/// releases as the SDK gains features, so a `match` on it outside this
/// crate must include a wildcard (`_ =>`) arm.
///
/// ```
/// use veloxquant_core::VeloxQuantError;
///
/// fn describe(err: &VeloxQuantError) -> &'static str {
///     match err {
///         VeloxQuantError::RuntimeUnavailable => "start the runtime first",
///         VeloxQuantError::InsufficientMemory => "try a smaller model",
///         _ => "see the error message",
///     }
/// }
///
/// assert_eq!(describe(&VeloxQuantError::RuntimeUnavailable), "start the runtime first");
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VeloxQuantError {
    /// The VeloxQuant runtime could not be reached or did not respond.
    #[error("runtime unavailable")]
    RuntimeUnavailable,

    /// The current platform is not supported for this operation.
    #[error("unsupported platform")]
    UnsupportedPlatform,

    /// The host does not have enough memory for the requested operation.
    #[error("insufficient memory")]
    InsufficientMemory,

    /// A requested model could not be found in the registry.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// A request parameter was invalid.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// An HTTP/network-level error occurred.
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    /// A JSON serialization or deserialization error occurred.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Scanning the local Hugging Face model cache failed (e.g. the Python
    /// interpreter could not be launched, or `huggingface_hub` is not
    /// importable).
    #[error("failed to scan the local Hugging Face cache: {0}")]
    ModelScanFailed(String),

    /// Pulling (downloading) a model into the local Hugging Face cache
    /// failed.
    #[error("failed to pull model \"{id}\": {detail}")]
    ModelPullFailed {
        /// The model id that failed to download.
        id: String,
        /// Human-readable failure detail (subprocess stderr or the
        /// underlying `huggingface_hub` error message).
        detail: String,
    },

    /// Deleting a model from the local Hugging Face cache failed.
    #[error("failed to delete model \"{id}\": {detail}")]
    ModelDeleteFailed {
        /// The model id that failed to delete.
        id: String,
        /// Human-readable failure detail.
        detail: String,
    },

    /// An agent's tool-calling loop exceeded its configured maximum number
    /// of steps without the model returning a final, non-tool-calling
    /// response.
    #[error(
        "Agent::run() exceeded max_steps ({max_steps}) without a final response — \
         the model kept calling tools. Pass a higher max_steps if this is expected."
    )]
    AgentMaxStepsExceeded {
        /// The `max_steps` limit that was exceeded.
        max_steps: usize,
    },

    /// Registering a tool (manually, or via an MCP server) under a name
    /// that's already registered on the agent.
    #[error("a tool named \"{0}\" is already registered on this agent")]
    ToolAlreadyRegistered(String),

    /// An MCP tool result contained a content block type (image, audio,
    /// resource, or resource_link) that isn't yet unwrapped into a tool
    /// result — surfacing an image/audio/resource to a text-only chat model
    /// needs a deliberate design decision that hasn't been made yet, so this
    /// fails loudly instead of silently dropping the content.
    #[error(
        "MCP tool result contained unsupported content type \"{0}\" — only \"text\" content \
         (or structured_content) is currently unwrapped into a tool result"
    )]
    UnsupportedMcpContent(String),

    /// An error connecting to, or communicating with, an MCP server.
    #[error("MCP error: {0}")]
    Mcp(String),

    /// No model in the registry fits the host's available memory for the
    /// requested task — Go's `ErrInsufficientMemory` from AutoPilot model
    /// selection, with the task and budget it was checked against.
    #[error("no model fits available memory ({available_memory_bytes} bytes) for task {task:?}")]
    NoModelFits {
        /// The task selection was filtered to (`None` = any task).
        task: Option<String>,
        /// The available-memory budget models were checked against.
        available_memory_bytes: u64,
    },

    /// The `veloxquant` CLI could not be launched at all (not on `PATH`,
    /// not executable, or the configured interpreter doesn't exist).
    #[error("could not launch `{command}`: {detail}")]
    CliUnavailable {
        /// The command line that was attempted.
        command: String,
        /// The OS-level launch error.
        detail: String,
    },

    /// A `veloxquant` CLI subcommand ran but exited unsuccessfully.
    #[error("`{command}` failed (exit code {exit_code:?}): {stderr}")]
    CliCommandFailed {
        /// The command line that was run.
        command: String,
        /// The process exit code (`None` if it was killed by a signal).
        exit_code: Option<i32>,
        /// The subcommand's stderr, trimmed.
        stderr: String,
    },

    /// A `veloxquant` CLI subcommand succeeded but its `--json` output
    /// didn't match the expected shape.
    #[error("`{command}` printed output that could not be parsed: {detail}")]
    MalformedCliOutput {
        /// The command line that was run.
        command: String,
        /// The parse error.
        detail: String,
    },

    /// `veloxquant recommend` warned that the planned workload likely will
    /// not fit, and AutoPilot was not told to proceed anyway (`force`).
    /// `AutoPilot::try_start` returns the same information (plus the full
    /// recommendation) as data instead of an error.
    #[error(
        "veloxquant recommend reports this configuration likely will not fit (recommended \
         method: {method}): {}. Set `force: true` to start anyway.",
        warnings.join(" | ")
    )]
    AutoPilotWontFit {
        /// The method `recommend` picked.
        method: String,
        /// The `recommend` warnings that matched the won't-fit pattern.
        warnings: Vec<String>,
    },
}

/// Convenience alias for `Result<T, VeloxQuantError>`.
pub type Result<T> = std::result::Result<T, VeloxQuantError>;
