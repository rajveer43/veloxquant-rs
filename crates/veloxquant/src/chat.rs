//! Chat completion API: `client.chat().create(...)`.

use std::time::Duration;

use veloxquant_core::{Result, VeloxQuantError};
use veloxquant_openai::{ChatRequest, ChatResponse, Message};

/// Handle for issuing chat completions against the configured runtime.
///
/// Obtained via [`crate::Client::chat`]. Cheap to construct; holds only a
/// [`reqwest::Client`] and the runtime's base URL.
#[derive(Debug, Clone)]
pub struct ChatApi {
    http: reqwest::Client,
    base_url: String,
}

impl ChatApi {
    pub(crate) fn new(base_url: String, timeout: Duration) -> Result<Self> {
        Ok(Self {
            http: veloxquant_core::build_http_client(timeout)?,
            base_url,
        })
    }

    /// Sends a chat completion request and waits for the full response.
    ///
    /// Posts to `{base_url}/v1/chat/completions` in the OpenAI-compatible
    /// shape. Returns [`VeloxQuantError::RuntimeUnavailable`] if the
    /// runtime cannot be reached.
    pub async fn create(
        &self,
        model: impl Into<String>,
        messages: Vec<Message>,
    ) -> Result<ChatResponse> {
        let request = ChatRequest {
            model: model.into(),
            messages,
            temperature: None,
            max_tokens: None,
            stream: false,
        };

        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );

        let response = self
            .http
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|_| VeloxQuantError::RuntimeUnavailable)?;

        if !response.status().is_success() {
            return Err(VeloxQuantError::RuntimeUnavailable);
        }

        let chat_response = response.json::<ChatResponse>().await?;
        Ok(chat_response)
    }
}
