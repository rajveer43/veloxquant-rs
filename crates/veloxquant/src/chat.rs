//! Chat completion API: `client.chat().create(...)`.

use std::time::Duration;

use veloxquant_core::{Result, VeloxQuantError};
use veloxquant_openai::{ChatRequest, ChatResponse, ChatStream, Message};

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
            tools: None,
            stream: false,
        };

        self.send(request).await
    }

    /// Sends a fully-specified [`ChatRequest`] (e.g. one carrying `tools`
    /// or sampling overrides) and waits for the full response.
    ///
    /// This is the lower-level entry point [`create`](Self::create) is
    /// built on; callers that need `tools` (see the `agent` feature's
    /// `Agent`) or other request fields not exposed by `create`'s
    /// convenience signature should use this directly. `request.stream` is
    /// forced to `false` regardless of the value passed in — use
    /// [`stream`](Self::stream) for streamed output.
    pub async fn send(&self, mut request: ChatRequest) -> Result<ChatResponse> {
        request.stream = false;

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

    /// Sends a chat completion request and streams the response
    /// incrementally over Server-Sent Events.
    ///
    /// Posts to `{base_url}/v1/chat/completions` with `stream: true`. The
    /// returned [`ChatStream`] yields chunks as they arrive; dropping it
    /// before it's exhausted cancels the underlying connection.
    ///
    /// ```no_run
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// use futures_util::StreamExt;
    /// use veloxquant::{Client, Message};
    ///
    /// let client = Client::builder().build()?;
    /// let mut stream = client
    ///     .chat()?
    ///     .stream("Qwen3-8B", vec![Message::user("hi")])
    ///     .await?;
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     print!("{}", chunk.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream(
        &self,
        model: impl Into<String>,
        messages: Vec<Message>,
    ) -> Result<ChatStream> {
        let request = ChatRequest {
            model: model.into(),
            messages,
            temperature: None,
            max_tokens: None,
            tools: None,
            stream: true,
        };

        veloxquant_openai::stream_chat_completions(self.http.clone(), &self.base_url, request).await
    }
}
