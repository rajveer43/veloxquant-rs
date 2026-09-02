//! Incremental Server-Sent Events (SSE) parsing for streamed chat
//! completions, and the [`ChatStream`] type returned by
//! [`crate::chat::stream_chat_completions`].

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;
use serde::Deserialize;

pub use veloxquant_core::VeloxQuantError;

/// A single incremental piece of a streamed chat response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatChunk {
    /// The response id this chunk belongs to.
    #[serde(default)]
    pub id: String,
    /// Model that produced this chunk.
    #[serde(default)]
    pub model: String,
    /// Incremental text content for this chunk.
    #[serde(default)]
    pub text: String,
    /// Whether this is the final chunk in the stream.
    #[serde(default)]
    pub done: bool,
}

/// An async stream of [`ChatChunk`]s produced by a streaming chat
/// completion request.
///
/// Dropping the stream before it is exhausted cancels the underlying HTTP
/// request and cleans up the connection (this falls out of dropping the
/// inner [`reqwest`] byte stream, no extra bookkeeping is needed).
pub struct ChatStream {
    inner: Pin<Box<dyn Stream<Item = Result<ChatChunk, VeloxQuantError>> + Send>>,
}

impl ChatStream {
    /// Wraps an inner chunk stream. Used internally by
    /// [`crate::chat::stream_chat_completions`]; exposed so alternative
    /// transports can construct a [`ChatStream`] too.
    pub fn new<S>(inner: S) -> Self
    where
        S: Stream<Item = Result<ChatChunk, VeloxQuantError>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
        }
    }
}

impl Stream for ChatStream {
    type Item = Result<ChatChunk, VeloxQuantError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// The SSE terminator sentinel used by OpenAI-compatible streaming APIs.
const DONE_SENTINEL: &str = "[DONE]";

/// Incrementally parses raw SSE bytes into [`ChatChunk`]s.
///
/// SSE frames are separated by a blank line and carry their payload on
/// `data: ` lines. This parser is fed arbitrary byte fragments (as they
/// arrive off the wire, which may split a frame anywhere, including
/// mid-line) via [`SseParser::push`], and yields complete, parsed events as
/// they become available.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buf: String,
}

/// The result of parsing one complete SSE `data:` line.
pub(crate) enum SseEvent {
    /// A parsed chat chunk.
    Chunk(ChatChunk),
    /// The `[DONE]` terminator was received.
    Done,
}

impl SseParser {
    /// Feeds newly received bytes into the parser, returning any complete
    /// events extracted from the buffer so far, in order.
    ///
    /// A `data:` line whose payload fails to parse as JSON yields a
    /// [`VeloxQuantError::Serialization`] in place of that event; the
    /// parser remains usable for subsequent lines.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<Result<SseEvent, VeloxQuantError>> {
        self.buf.push_str(&String::from_utf8_lossy(bytes));

        let mut events = Vec::new();
        while let Some(pos) = self.buf.find('\n') {
            let line = self.buf[..pos].trim_end_matches('\r').to_string();
            self.buf.drain(..=pos);

            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = payload.trim();
            if payload.is_empty() {
                continue;
            }

            if payload == DONE_SENTINEL {
                events.push(Ok(SseEvent::Done));
                continue;
            }

            match serde_json::from_str::<ChatChunk>(payload) {
                Ok(chunk) => events.push(Ok(SseEvent::Chunk(chunk))),
                Err(err) => events.push(Err(VeloxQuantError::from(err))),
            }
        }

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_complete_frame() {
        let mut parser = SseParser::default();
        let events =
            parser.push(b"data: {\"id\":\"1\",\"model\":\"m\",\"text\":\"hi\",\"done\":false}\n\n");
        assert_eq!(events.len(), 1);
        match events.into_iter().next().unwrap() {
            Ok(SseEvent::Chunk(chunk)) => assert_eq!(chunk.text, "hi"),
            _ => panic!("expected chunk"),
        }
    }

    #[test]
    fn parses_done_sentinel() {
        let mut parser = SseParser::default();
        let events = parser.push(b"data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events.into_iter().next().unwrap(),
            Ok(SseEvent::Done)
        ));
    }

    #[test]
    fn handles_frame_split_across_pushes() {
        let mut parser = SseParser::default();
        let events = parser.push(b"data: {\"id\":\"1\",\"mod");
        assert!(events.is_empty());
        let events = parser.push(b"el\":\"m\",\"text\":\"hi\",\"done\":true}\n\n");
        assert_eq!(events.len(), 1);
        match events.into_iter().next().unwrap() {
            Ok(SseEvent::Chunk(chunk)) => {
                assert_eq!(chunk.model, "m");
                assert!(chunk.done);
            }
            _ => panic!("expected chunk"),
        }
    }

    #[test]
    fn invalid_json_payload_yields_serialization_error() {
        let mut parser = SseParser::default();
        let events = parser.push(b"data: {not json}\n\n");
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events.into_iter().next().unwrap(),
            Err(VeloxQuantError::Serialization(_))
        ));
    }

    #[test]
    fn parser_remains_usable_after_a_bad_frame() {
        let mut parser = SseParser::default();
        let mut events = parser.push(b"data: {bad}\n\ndata: {\"text\":\"ok\"}\n\n");
        assert_eq!(events.len(), 2);
        assert!(events.remove(0).is_err());
        assert!(matches!(events.remove(0), Ok(SseEvent::Chunk(_))));
    }

    #[test]
    fn ignores_non_data_lines_like_event_and_comments() {
        let mut parser = SseParser::default();
        let events = parser.push(b": comment\nevent: chunk\ndata: {\"text\":\"x\"}\n\n");
        assert_eq!(events.len(), 1);
    }
}
