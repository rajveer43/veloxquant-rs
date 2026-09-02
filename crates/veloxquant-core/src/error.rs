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
}

/// Convenience alias for `Result<T, VeloxQuantError>`.
pub type Result<T> = std::result::Result<T, VeloxQuantError>;
