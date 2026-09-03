//! Async HTTP client for a VeloxQuant runtime.

use std::time::Duration;

use serde::Deserialize;

use veloxquant_core::{Result, VeloxQuantError};
use veloxquant_openai::RemoteModel;

use crate::health::RuntimeStatus;

/// The `GET /v1/models` response envelope, per the OpenAI-compatible list
/// format: `{"object": "list", "data": [...]}`.
#[derive(Debug, Deserialize)]
struct ModelListResponse {
    #[serde(default)]
    data: Vec<RemoteModel>,
}

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

    /// Lists models currently available on the runtime
    /// (`GET {base_url}/v1/models`).
    ///
    /// Returns [`VeloxQuantError::RuntimeUnavailable`] if the runtime
    /// cannot be reached or responds with a non-success status. An empty
    /// `data` array is a valid response and yields an empty `Vec`.
    pub async fn list_models(&self) -> Result<Vec<RemoteModel>> {
        let url = format!("{}/v1/models", self.base_url.trim_end_matches('/'));

        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|_| VeloxQuantError::RuntimeUnavailable)?;

        if !response.status().is_success() {
            return Err(VeloxQuantError::RuntimeUnavailable);
        }

        let list = response.json::<ModelListResponse>().await?;
        Ok(list.data)
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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

    /// Starts a bare-bones HTTP server on an ephemeral port that ignores
    /// the request and writes `body` verbatim as a JSON response.
    async fn spawn_json_server(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = socket.read(&mut buf).await;
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(header.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });

        format!("http://{addr}")
    }

    #[tokio::test]
    async fn list_models_parses_populated_listing() {
        let base_url = spawn_json_server(
            r#"{"object":"list","data":[{"id":"Qwen3-8B","object":"model"},{"id":"gemma-2-9b","object":"model"}]}"#,
        )
        .await;
        let client = RuntimeClient::new(base_url, Duration::from_secs(1)).unwrap();

        let models = client.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "Qwen3-8B");
        assert_eq!(models[1].id, "gemma-2-9b");
    }

    #[tokio::test]
    async fn list_models_handles_empty_listing() {
        let base_url = spawn_json_server(r#"{"object":"list","data":[]}"#).await;
        let client = RuntimeClient::new(base_url, Duration::from_secs(1)).unwrap();

        let models = client.list_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn list_models_reports_runtime_unavailable_when_unreachable() {
        let client = RuntimeClient::new("http://127.0.0.1:1", Duration::from_millis(200)).unwrap();
        let result = client.list_models().await;
        assert!(matches!(result, Err(VeloxQuantError::RuntimeUnavailable)));
    }
}
