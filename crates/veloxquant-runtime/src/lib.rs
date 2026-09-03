//! Async client for communicating with a local VeloxQuant runtime.
//!
//! Implements the health-check endpoint used by `vq doctor`
//! ([`RuntimeClient::health`]) and model listing
//! ([`RuntimeClient::list_models`]). Chat completions (streaming and
//! non-streaming) live in `veloxquant-openai`; [`streaming::ChatChunk`] is
//! kept here as a stable shared chunk shape.

pub mod client;
pub mod health;
pub mod streaming;

pub use client::RuntimeClient;
pub use health::RuntimeStatus;
pub use streaming::ChatChunk;
