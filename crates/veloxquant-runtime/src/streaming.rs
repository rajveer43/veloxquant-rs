//! Streaming primitives shared by runtime-facing crates.
//!
//! This crate defines the shared chunk type only; the SSE-consuming
//! `Stream<Item = Result<ChatChunk, VeloxQuantError>>` implementation lives
//! in `veloxquant-openai` (see its `streaming::ChatStream`).

use serde::Deserialize;

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
