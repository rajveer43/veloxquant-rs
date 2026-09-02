//! OpenAI-compatible chat types for the VeloxQuant Rust SDK.
//!
//! **v0.1.0 status:** this crate defines the wire types (`Message`,
//! `ChatRequest`, `ChatResponse`, `RemoteModel`) that a chat client will
//! use, but does not yet perform any HTTP calls or SSE streaming — those
//! land in v0.2.0. See the crate's `README.md` roadmap section.

pub mod chat;
pub mod models;
pub mod streaming;

pub use chat::{ChatRequest, ChatResponse, InferenceMetrics, Message, Role, Usage};
pub use models::RemoteModel;
