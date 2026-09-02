//! Async HTTP client for a VeloxQuant runtime.

use std::time::Duration;

use veloxquant_core::{Result, VeloxQuantError};

use crate::health::RuntimeStatus;

/// Async client for a VeloxQuant runtime's control endpoints (health, and
/// in future releases, model management).
///
/// Chat/completion endpoints live in `veloxquant-openai`, which is built on
/// top of the same [`reqwest::Client`] construction path.
#[derive(Debug, Clone)]
pub struct RuntimeClient {
    http: reqwest::Client,
    base_url: String,
}

impl RuntimeClient {
    /// Creates a new runtime client targeting `base_url` with the given
    /// per-request `timeout`.
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self> {
        let http = veloxquant_core::build_http_client(timeout)?;
        Ok(Self {
            http,
            base_url: base_url.into(),
        })
    }

    /// The runtime's configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Queries the runtime's health endpoint (`GET {base_url}/health`).
    ///
    /// Returns [`VeloxQuantError::RuntimeUnavailable`] if the runtime
    /// cannot be reached at all (connection refused, DNS failure, timeout).
    /// A reachable runtime that responds with a non-success status or
    /// unparseable body surfaces the underlying network/serialization
    /// error instead.
    pub async fn health(&self) -> Result<RuntimeStatus> {
        let url = format!("{}/health", self.base_url.trim_end_matches('/'));

        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|_| VeloxQuantError::RuntimeUnavailable)?;

        if !response.status().is_success() {
            return Err(VeloxQuantError::RuntimeUnavailable);
        }

        let status = response.json::<RuntimeStatus>().await?;
        Ok(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_reports_runtime_unavailable_when_unreachable() {
        let client = RuntimeClient::new("http://127.0.0.1:1", Duration::from_millis(200)).unwrap();
        let result = client.health().await;
        assert!(matches!(result, Err(VeloxQuantError::RuntimeUnavailable)));
    }

    #[test]
    fn base_url_is_stored_verbatim() {
        let client = RuntimeClient::new("http://localhost:8765", Duration::from_secs(1)).unwrap();
        assert_eq!(client.base_url(), "http://localhost:8765");
    }
}
