//! Memory and inference metrics for the VeloxQuant Rust SDK.
//!
//! **v0.1.0 status:** the [`Monitor`] pub/sub plumbing is fully functional;
//! automatic periodic sampling of live metrics lands in v0.3.0.

pub mod metrics;
pub mod sampler;

pub use metrics::Metrics;
pub use sampler::Monitor;
