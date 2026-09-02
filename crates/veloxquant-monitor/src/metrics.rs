//! Metrics types shared between the sampler and its subscribers.

use std::time::Duration;

/// A single sampled snapshot of memory and inference metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    /// Memory currently in use, in bytes.
    pub memory_used_bytes: u64,
    /// Memory currently available, in bytes.
    pub memory_available_bytes: u64,
    /// Generation throughput, in tokens/sec (`0.0` when not inferring).
    pub tokens_per_second: f64,
    /// Time to first token for the most recent request.
    pub time_to_first_token: Duration,
    /// Context length in use, in tokens.
    pub context_length: usize,
    /// KV-cache memory in use, in bytes.
    pub kv_cache_bytes: u64,
    /// Compression ratio currently in effect (compressed / uncompressed).
    pub compression_ratio: f64,
}
