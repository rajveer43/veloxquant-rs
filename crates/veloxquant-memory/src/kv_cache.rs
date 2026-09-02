//! KV-cache memory estimation.

use crate::estimator::{ModelArchitecture, Precision};

/// Parameters needed to estimate KV-cache memory usage for a single
/// inference sequence.
#[derive(Debug, Clone)]
pub struct KVCacheRequest {
    /// The model's architecture (layers, heads, head dimension).
    pub architecture: ModelArchitecture,
    /// Sequence length, in tokens, the cache must hold.
    pub context_length: usize,
    /// Numeric precision used to store cache entries.
    pub precision: Precision,
}

/// Computes uncompressed KV-cache memory in bytes using the standard
/// formula:
///
/// ```text
/// KV Cache Memory = Layers × Tokens × KV Heads × Head Dim × 2 × BytesPerElement
/// ```
///
/// The factor of 2 accounts for storing both keys and values. Returns `0`
/// if any architecture dimension or the context length is zero, rather than
/// panicking.
pub fn estimate_kv_cache_bytes(req: &KVCacheRequest) -> u64 {
    let arch = &req.architecture;
    if arch.num_layers == 0
        || arch.num_kv_heads == 0
        || arch.head_dim == 0
        || req.context_length == 0
    {
        return 0;
    }

    let elements = arch.num_layers as f64
        * req.context_length as f64
        * arch.num_kv_heads as f64
        * arch.head_dim as f64
        * 2.0;

    (elements * req.precision.bytes_per_element()) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_architecture() -> ModelArchitecture {
        ModelArchitecture {
            name: "test-model".to_string(),
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            parameter_count: 7_000_000_000,
        }
    }

    #[test]
    fn computes_expected_kv_cache_bytes() {
        let req = KVCacheRequest {
            architecture: test_architecture(),
            context_length: 4096,
            precision: Precision::Fp16,
        };
        // 32 * 4096 * 8 * 128 * 2 * 2 bytes
        let expected = 32u64 * 4096 * 8 * 128 * 2 * 2;
        assert_eq!(estimate_kv_cache_bytes(&req), expected);
    }

    #[test]
    fn zero_context_length_yields_zero() {
        let req = KVCacheRequest {
            architecture: test_architecture(),
            context_length: 0,
            precision: Precision::Fp16,
        };
        assert_eq!(estimate_kv_cache_bytes(&req), 0);
    }

    #[test]
    fn lower_precision_yields_smaller_cache() {
        let arch = test_architecture();
        let fp16 = estimate_kv_cache_bytes(&KVCacheRequest {
            architecture: arch.clone(),
            context_length: 4096,
            precision: Precision::Fp16,
        });
        let int4 = estimate_kv_cache_bytes(&KVCacheRequest {
            architecture: arch,
            context_length: 4096,
            precision: Precision::Int4,
        });
        assert!(int4 < fp16);
        assert_eq!(fp16 / 4, int4);
    }
}
