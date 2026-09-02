//! Base HTTP client construction shared by SDK crates that talk to a
//! VeloxQuant runtime.

use std::time::Duration;

use crate::error::{Result, VeloxQuantError};

/// Builds a [`reqwest::Client`] configured with the given timeout.
///
/// Centralized so every crate that speaks HTTP to the runtime (chat,
/// health checks, streaming) shares one construction path.
pub fn build_http_client(timeout: Duration) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(VeloxQuantError::from)
}
