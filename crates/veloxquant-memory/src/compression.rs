//! Native optimization abstraction for future VeloxQuant KV-cache
//! compression algorithms.
//!
//! No native Rust compression algorithm is implemented in v0.1.0 — this
//! module defines the trait future implementations (TurboQuant, RVQ,
//! VecInfer, RateQuant, PolarQuant, QJL, ...) will conform to, gated behind
//! the `native-optimizers` feature at the facade crate level.

use veloxquant_core::Result;

/// A pluggable KV-cache compressor.
///
/// Implementations must be safe to share across threads, since inference
/// serving typically compresses/decompresses cache entries concurrently
/// across requests.
pub trait KVCacheOptimizer: Send + Sync {
    /// Compresses a raw KV-cache buffer.
    fn compress(&self, input: &[u8]) -> Result<Vec<u8>>;

    /// Decompresses a buffer previously produced by [`compress`](Self::compress).
    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>>;

    /// The expected compression ratio (compressed size / original size)
    /// this optimizer achieves, in `(0.0, 1.0]`.
    fn compression_ratio(&self) -> f64;
}

/// Applies a known compression ratio to an uncompressed byte count, for
/// estimation purposes (no actual compression is performed).
///
/// A `ratio` outside `(0.0, 1.0]` is treated as "no compression" and the
/// input is returned unchanged.
pub fn apply_compression_ratio(uncompressed_bytes: u64, ratio: f64) -> u64 {
    if ratio <= 0.0 || ratio > 1.0 {
        return uncompressed_bytes;
    }
    (uncompressed_bytes as f64 * ratio) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_valid_ratio() {
        assert_eq!(apply_compression_ratio(1000, 0.5), 500);
    }

    #[test]
    fn out_of_range_ratio_is_a_no_op() {
        assert_eq!(apply_compression_ratio(1000, 0.0), 1000);
        assert_eq!(apply_compression_ratio(1000, 1.5), 1000);
        assert_eq!(apply_compression_ratio(1000, -1.0), 1000);
    }
}
