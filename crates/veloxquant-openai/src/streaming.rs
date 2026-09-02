//! Re-exports of streaming chunk types shared with `veloxquant-runtime`.
//!
//! Implemented here only as a re-export in v0.1.0; SSE parsing and the
//! `Stream<Item = Result<ChatChunk, VeloxQuantError>>` implementation are
//! planned for v0.2.0.

pub use veloxquant_core::VeloxQuantError;
