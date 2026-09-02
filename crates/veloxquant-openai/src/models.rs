//! Model listing types for the OpenAI-compatible `GET /v1/models` endpoint.

use serde::Deserialize;

/// A model entry as reported by a runtime's `/v1/models` endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct RemoteModel {
    /// The model identifier used in chat requests.
    pub id: String,
    /// The object type, typically `"model"`.
    #[serde(default)]
    pub object: String,
}
