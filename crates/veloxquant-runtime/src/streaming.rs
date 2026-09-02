//! Streaming primitives shared by runtime-facing crates.
//!
//! v0.1.0 defines the shared chunk type only; the SSE-consuming
//! `Stream<Item = Result<ChatChunk, VeloxQuantError>>` implementation lands
//! in `veloxquant-openai` as part of v0.2.0 (see the crate root docs).

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
