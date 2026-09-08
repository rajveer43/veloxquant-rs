//! Client configuration shared by SDK crates.

use std::time::Duration;

use crate::types::OptimizationProfile;

/// Default base URL for a local VeloxQuant runtime.
pub const DEFAULT_RUNTIME_URL: &str = "http://localhost:8765";

/// Default per-request timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Resolved configuration for a `Client` (see the `veloxquant` facade
/// crate's `Client`, which this type backs).
#[derive(Debug, Clone)]
pub struct Config {
    /// Base URL of the VeloxQuant runtime (or an OpenAI-compatible endpoint).
    pub runtime_url: String,
    /// Whether hardware/system properties should be auto-detected on build.
    pub auto_detect: bool,
    /// The optimization profile to use when one isn't explicitly requested.
    pub profile: OptimizationProfile,
    /// Per-request timeout applied to runtime HTTP calls.
    pub timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            runtime_url: DEFAULT_RUNTIME_URL.to_string(),
            auto_detect: false,
            profile: OptimizationProfile::default(),
            timeout: DEFAULT_TIMEOUT,
        }
    }
}
