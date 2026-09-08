//! Error types shared across the VeloxQuant SDK.

/// The single error type returned by fallible VeloxQuant SDK operations.
#[derive(Debug, thiserror::Error)]
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
}

/// Convenience alias for `Result<T, VeloxQuantError>`.
pub type Result<T> = std::result::Result<T, VeloxQuantError>;
