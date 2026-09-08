//! `rig-core` [`CompletionModel`] adapter for the VeloxQuant Rust SDK.
//!
//! Mirrors the *shape* of Go's `langchain` adapter
//! (`veloxquant-go/langchain/langchain.go`): a separate crate, so `rig-core`
//! is an opt-in dependency only paid for by callers who actually want a
//! `rig` integration, never a transitive dependency of the core
//! [`veloxquant`] facade crate. Not a literal port — `rig` has no
//! equivalent of `langchaingo`'s single `llms.Model` interface; instead this
//! crate implements `rig-core` 0.42's [`CompletionModel`] trait, backed by
//! a [`veloxquant::Client`].
//!
//! **Text-only.** Like the Go adapter (which rejects any non-text
//! `llms.MessageContent` part outright), this adapter supports only plain
//! text message content. Any `rig` message content that isn't plain text
//! (images, audio, documents, tool calls/results, reasoning blocks) is
//! rejected with a clear [`RigAdapterError`] rather than silently dropped —
//! the underlying VeloxQuant runtime's chat completion API is text-only, so
//! there is nothing correct to do with those content types today.
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_core::completion::CompletionModel;
//! use veloxquant::Client;
//! use veloxquant_rig::VeloxQuantCompletionModel;
//!
//! let client = Client::builder().build()?;
//! let model = VeloxQuantCompletionModel::new(client, "mlx-community/Qwen3-8B-4bit");
//!
//! let request = model.completion_request("Hello!").build();
//! let response = model.completion(request).await?;
//! # Ok(())
//! # }
//! ```

use futures_util::StreamExt;
use rig_core::completion::{
    self, AssistantContent, CompletionError, CompletionModel, CompletionRequest,
    CompletionResponse, Message as RigMessage, Usage as RigUsage,
};
use rig_core::message::{Text, UserContent};
use rig_core::streaming::{
    normalize_stream, RawStreamingChoice, StreamFinal, StreamingCompletionResponse,
};

use veloxquant::{Client, Message as VqMessage, Role as VqRole};
use veloxquant_core::VeloxQuantError;

/// Errors specific to adapting `rig-core` requests/responses to and from
/// the VeloxQuant runtime's text-only chat API.
#[derive(Debug, thiserror::Error)]
pub enum RigAdapterError {
    /// A message (in the request's `chat_history`, or a content block
    /// within one) carried non-text content this adapter cannot represent.
    /// The VeloxQuant runtime's chat completion API is text-only, so images,
    /// audio, documents, tool calls, tool results, and reasoning blocks are
    /// rejected explicitly here rather than silently dropped.
    #[error(
        "veloxquant-rig: unsupported non-text content in a {0} message — \
         the VeloxQuant runtime's chat API is text-only"
    )]
    UnsupportedContent(&'static str),

    /// The underlying VeloxQuant SDK call failed.
    #[error("veloxquant-rig: {0}")]
    VeloxQuant(#[from] VeloxQuantError),
}

/// `rig-core`'s [`CompletionError`] has no direct "provider adapter failed"
/// variant that preserves a source error cleanly other than
/// [`CompletionError::RequestError`], which is exactly what it's for
/// (wrapping an arbitrary boxed error raised while building/issuing the
/// request) — so this is not a loss of information, just routing through
/// the variant `rig` itself designates for this.
impl From<RigAdapterError> for CompletionError {
    fn from(err: RigAdapterError) -> Self {
        CompletionError::RequestError(Box::new(err))
    }
}

/// Adapts a [`veloxquant::Client`] to `rig-core`'s
/// [`CompletionModel`] trait, so a local VeloxQuant runtime can be used as
/// the completion backend in a `rig` pipeline/agent.
///
/// Construct with [`VeloxQuantCompletionModel::new`], naming the model the
/// underlying runtime should run inference against (VeloxQuant does not
/// have rig's separate "provider client + named model" split — one
/// `Client` talks to one runtime, which serves whatever model it was
/// started with, so `model` here is threaded through as
/// [`veloxquant_openai::ChatRequest::model`] on every call).
#[derive(Clone)]
pub struct VeloxQuantCompletionModel {
    client: Client,
    model: String,
}

impl VeloxQuantCompletionModel {
    /// Creates a [`VeloxQuantCompletionModel`] backed by `client`, issuing
    /// every completion against `model`.
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// The model id this adapter sends on every request.
    pub fn model_name(&self) -> &str {
        &self.model
    }
}

impl std::fmt::Debug for VeloxQuantCompletionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VeloxQuantCompletionModel")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// Converts a `rig` chat-history entry into a [`veloxquant_openai::Message`].
///
/// Only plain-text content is supported: a [`RigMessage::System`] or
/// [`RigMessage::User`]/[`RigMessage::Assistant`] whose content is entirely
/// [`Text`] blocks (concatenated) round-trips; anything else fails with
/// [`RigAdapterError::UnsupportedContent`].
fn to_vq_message(message: &RigMessage) -> Result<VqMessage, RigAdapterError> {
    match message {
        RigMessage::System { content } => Ok(VqMessage {
            role: VqRole::System,
            content: content.clone(),
            tool_calls: None,
            tool_call_id: None,
        }),
        RigMessage::User { content } => {
            let mut text = String::new();
            for block in content {
                match block {
                    UserContent::Text(Text { text: t, .. }) => text.push_str(t),
                    _ => return Err(RigAdapterError::UnsupportedContent("user")),
                }
            }
            Ok(VqMessage::user(text))
        }
        RigMessage::Assistant { content, .. } => {
            let mut text = String::new();
            for block in content {
                match block {
                    AssistantContent::Text(Text { text: t, .. }) => text.push_str(t),
                    _ => return Err(RigAdapterError::UnsupportedContent("assistant")),
                }
            }
            Ok(VqMessage::assistant(text))
        }
    }
}

/// Builds a [`veloxquant_openai::ChatRequest`] from a `rig`
/// [`CompletionRequest`], rejecting any request that isn't text-only.
fn to_chat_request(
    model: &str,
    request: &CompletionRequest,
) -> Result<veloxquant_openai::ChatRequest, RigAdapterError> {
    if !request.tools.is_empty() {
        return Err(RigAdapterError::UnsupportedContent(
            "request (tools are not supported by this text-only adapter)",
        ));
    }

    let mut messages = Vec::with_capacity(request.chat_history.len() + 1);
    if let Some(preamble) = &request.preamble {
        messages.push(VqMessage::system(preamble.clone()));
    }
    for message in &request.chat_history {
        messages.push(to_vq_message(message)?);
    }

    Ok(veloxquant_openai::ChatRequest {
        model: model.to_string(),
        messages,
        temperature: request.temperature.map(|t| t as f32),
        max_tokens: request.max_tokens.map(|t| t as u32),
        tools: None,
        stream: false,
    })
}

impl CompletionModel for VeloxQuantCompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let chat_request = to_chat_request(&self.model, &request)?;

        let response = self
            .client
            .chat()
            .map_err(RigAdapterError::from)?
            .send(chat_request)
            .await
            .map_err(RigAdapterError::from)?;

        let usage = RigUsage {
            input_tokens: response.usage.prompt_tokens as u64,
            output_tokens: response.usage.completion_tokens as u64,
            total_tokens: response.usage.total_tokens as u64,
            ..RigUsage::new()
        };

        Ok(CompletionResponse::new(
            vec![AssistantContent::Text(Text::new(response.text))],
            usage,
            "veloxquant",
        ))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let mut chat_request = to_chat_request(&self.model, &request)?;
        chat_request.stream = true;

        // Reuses the existing SSE streaming transport
        // (`veloxquant_openai::stream_chat_completions`, built on
        // `veloxquant_runtime`'s `SseParser`/`SseEvent`) rather than
        // reparsing SSE a second time for this adapter.
        let http = self.client.chat().map_err(RigAdapterError::from)?;
        let mut chat_stream = http
            .stream(self.model.clone(), chat_request.messages)
            .await
            .map_err(RigAdapterError::from)?;

        let raw_stream = async_stream::stream! {
            let usage = RigUsage::new();
            while let Some(chunk) = chat_stream.next().await {
                match chunk {
                    Ok(chunk) => {
                        if !chunk.text.is_empty() {
                            yield Ok(RawStreamingChoice::Message(chunk.text));
                        }
                        if chunk.done {
                            let final_record = StreamFinal::new("veloxquant", usage)
                                .with_finish_reason(completion::FinishReason::Stop);
                            yield Ok(RawStreamingChoice::FinalResponse(final_record));
                            return;
                        }
                    }
                    Err(err) => {
                        yield Err(CompletionError::from(RigAdapterError::from(err)));
                        return;
                    }
                }
            }
            // Stream ended without an explicit `done: true` chunk: still
            // emit a terminal record so `StreamingCompletionResponse`
            // finishes cleanly rather than the consumer waiting forever.
            let final_record = StreamFinal::new("veloxquant", usage);
            yield Ok(RawStreamingChoice::FinalResponse(final_record));
        };

        let normalized = normalize_stream(Box::pin(raw_stream), |final_record: StreamFinal| {
            Ok(final_record)
        });

        Ok(StreamingCompletionResponse::stream(
            "veloxquant",
            normalized,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::Message as RigMsg;

    #[test]
    fn to_vq_message_maps_system_text() {
        let msg = RigMsg::system("be helpful");
        let vq = to_vq_message(&msg).unwrap();
        assert_eq!(vq.role, VqRole::System);
        assert_eq!(vq.content, "be helpful");
    }

    #[test]
    fn to_vq_message_maps_user_text() {
        let msg = RigMsg::user("hello");
        let vq = to_vq_message(&msg).unwrap();
        assert_eq!(vq.role, VqRole::User);
        assert_eq!(vq.content, "hello");
    }

    #[test]
    fn to_vq_message_maps_assistant_text() {
        let msg = RigMsg::assistant("hi there");
        let vq = to_vq_message(&msg).unwrap();
        assert_eq!(vq.role, VqRole::Assistant);
        assert_eq!(vq.content, "hi there");
    }

    #[test]
    fn to_vq_message_rejects_non_text_user_content() {
        let msg = RigMsg::User {
            content: vec![UserContent::Image(rig_core::message::Image {
                data: rig_core::message::DocumentSourceKind::Url(
                    "https://example.com/x.png".into(),
                ),
                media_type: None,
                detail: None,
                additional_params: None,
            })],
        };
        let err = to_vq_message(&msg).unwrap_err();
        assert!(matches!(err, RigAdapterError::UnsupportedContent("user")));
    }

    #[test]
    fn to_chat_request_rejects_tools() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![RigMsg::user("hi")],
            documents: Vec::new(),
            tools: vec![completion::ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: serde_json::json!({}),
            }],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };
        let err = to_chat_request("model", &request).unwrap_err();
        assert!(matches!(err, RigAdapterError::UnsupportedContent(_)));
    }

    #[test]
    fn model_name_returns_configured_model() {
        let client = Client::builder().build().unwrap();
        let model = VeloxQuantCompletionModel::new(client, "mlx-community/Qwen3-8B-4bit");
        assert_eq!(model.model_name(), "mlx-community/Qwen3-8B-4bit");
    }

    /// Starts a bare-bones HTTP server on an ephemeral port that ignores the
    /// request and writes `body` verbatim as an SSE response, matching the
    /// fixture pattern `veloxquant-openai/src/chat.rs`'s own streaming tests
    /// use — reused here rather than reinvented, so this test exercises the
    /// real SSE transport (`ChatApi::stream` ->
    /// `stream_chat_completions`/`SseParser`) end to end instead of a mock.
    async fn spawn_sse_server(body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = socket.read(&mut buf).await;
                let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
                if socket.write_all(header.as_bytes()).await.is_err() {
                    return;
                }
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });

        format!("http://{addr}")
    }

    #[tokio::test]
    async fn stream_reuses_sse_transport_and_aggregates_text() {
        use futures_util::StreamExt as _;
        use rig_core::streaming::StreamedAssistantContent;

        let base_url = spawn_sse_server(
            "data: {\"text\":\"hel\"}\n\ndata: {\"text\":\"lo\",\"done\":true}\n\ndata: [DONE]\n\n",
        )
        .await;

        let client = Client::builder().runtime_url(base_url).build().unwrap();
        let model = VeloxQuantCompletionModel::new(client, "m");

        let request = model.completion_request("hi").build();
        let mut stream = model.stream(request).await.unwrap();

        let mut collected = String::new();
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Text(text) = item.unwrap() {
                collected.push_str(&text.text);
            }
        }

        assert_eq!(collected, "hello");
        // The aggregated final `choice` is populated once the stream drains.
        assert!(!stream.choice.is_empty());
    }
}
