//! OpenAI-compatible chat types and transport for the VeloxQuant Rust SDK.
//!
//! Defines the wire types (`Message`, `ChatRequest`, `ChatResponse`,
//! `RemoteModel`) and both the non-streaming (`chat::create`, via
//! `veloxquant::ChatApi`) and streaming
//! (`chat::stream_chat_completions`/`streaming::ChatStream`) chat
//! completion transports.

pub mod chat;
pub mod models;
pub mod streaming;

pub use chat::{
    stream_chat_completions, ChatRequest, ChatResponse, InferenceMetrics, Message, Role, Usage,
};
pub use models::RemoteModel;
pub use streaming::{ChatChunk, ChatStream};
