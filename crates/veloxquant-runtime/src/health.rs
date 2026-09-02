//! Runtime health checks.

use serde::Deserialize;

/// The health/status of a VeloxQuant runtime.
#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeStatus {
    /// Whether the runtime reports itself as healthy.
    #[serde(default)]
    pub healthy: bool,
    /// Runtime version string.
    #[serde(default)]
    pub version: String,
    /// Name of the inference engine backing the runtime (e.g. `"mlx"`).
    #[serde(default)]
    pub engine: String,
}
