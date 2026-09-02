//! Async client for communicating with a local VeloxQuant runtime.
//!
//! v0.1.0 implements the health-check endpoint used by `vq doctor` and
//! [`RuntimeClient::health`]. Chat completions and SSE streaming are
//! planned for v0.2.0 (see `veloxquant-openai`); [`streaming::ChatChunk`]
//! is defined here now so the type is stable across that transition.

pub mod client;
pub mod health;
pub mod streaming;

pub use client::RuntimeClient;
pub use health::RuntimeStatus;
pub use streaming::ChatChunk;
