//! Model and total memory estimation.

use serde::{Deserialize, Serialize};
use veloxquant_core::{Result, VeloxQuantError};

use crate::kv_cache::{estimate_kv_cache_bytes, KVCacheRequest};

/// A conservative fixed estimate of the memory overhead of the inference
/// runtime itself (buffers, framework, activation scratch space) beyond
/// weights and KV cache.
pub const RUNTIME_OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// Numeric precision used to store model weights or KV-cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// 16-bit floating point (2 bytes/element).
    Fp16,
    /// 8-bit floating point (1 byte/element).
    Fp8,
    /// 8-bit integer quantization (1 byte/element).
    Int8,
    /// 4-bit integer quantization (0.5 bytes/element).
    Int4,
}

impl Precision {
    /// Storage size, in bytes, of a single scalar at this precision.
    pub fn bytes_per_element(self) -> f64 {
        match self {
            Self::Fp16 => 2.0,
            Self::Fp8 => 1.0,
            Self::Int8 => 1.0,
            Self::Int4 => 0.5,
        }
    }

    /// Bit width used for reporting recommended compression strategies.
    pub fn bits(self) -> u8 {
        match self {
            Self::Fp16 => 16,
            Self::Fp8 => 8,
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }
}

/// Describes the shape of a transformer model, sufficient to compute
/// weight and KV-cache memory. Deliberately architecture-agnostic so any
/// model family (Llama, Qwen, Gemma, Mistral, DeepSeek, or a custom
/// architecture) can populate it without changing estimator logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// A human-readable model name, used only for error messages/context.
    pub name: String,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of key/value attention heads.
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Model hidden size, used to approximate parameter count when
    /// [`parameter_count`](Self::parameter_count) is `0`.
    pub hidden_size: usize,
    /// Total parameter count, if known. When `0`, it is approximated from
    /// `num_layers` and `hidden_size`.
    pub parameter_count: u64,
}

/// A memory estimation request for a specific model, context length, and
/// precision.
#[derive(Debug, Clone)]
pub struct MemoryRequest {
    /// The model architecture to estimate memory for.
    pub model: ModelArchitecture,
    /// Sequence length, in tokens, the KV cache must hold.
    pub context_length: usize,
    /// Precision used for the naive (unoptimized) estimate.
    pub precision: Precision,
    /// Precision VeloxQuant would use for the KV cache when compression is
    /// applied. Defaults to [`Precision::Int4`] via
    /// [`MemoryRequest::new`].
    pub optimized_precision: Precision,
}

impl MemoryRequest {
    /// Creates a request with `optimized_precision` defaulted to
    /// [`Precision::Int4`].
    pub fn new(model: ModelArchitecture, context_length: usize, precision: Precision) -> Self {
        Self {
            model,
            context_length,
            precision,
            optimized_precision: Precision::Int4,
        }
    }
}

/// The result of a memory estimation, covering both the unoptimized
/// ("naive") footprint and the VeloxQuant-optimized footprint.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryEstimate {
    /// Estimated model weight memory, in bytes.
    pub model_memory_bytes: u64,
    /// Estimated uncompressed KV-cache memory, in bytes.
    pub kv_cache_memory_bytes: u64,
    /// Fixed runtime overhead, in bytes.
    pub runtime_overhead_bytes: u64,
    /// Total memory without VeloxQuant optimization, in bytes.
    pub total_memory_bytes: u64,

    /// Estimated KV-cache memory after VeloxQuant compression, in bytes.
    pub optimized_kv_bytes: u64,
    /// Total memory with VeloxQuant optimization, in bytes.
    pub optimized_total_bytes: u64,
    /// Bytes saved by optimization (`total - optimized_total`, floored at 0).
    pub saved_bytes: u64,
    /// Percentage of memory saved by optimization.
    pub saved_percent: f64,

    /// A human-readable description of the recommended compression strategy.
    pub recommended_strategy: String,
}

/// Estimates model, KV-cache, and total memory for a request.
///
/// Returns [`VeloxQuantError::InvalidRequest`] if `context_length` is zero.
pub fn estimate(req: &MemoryRequest) -> Result<MemoryEstimate> {
    if req.context_length == 0 {
        return Err(VeloxQuantError::InvalidRequest(format!(
            "estimate memory for {}: context length must be positive",
            req.model.name
        )));
    }

    let model_memory = model_memory_bytes(&req.model, req.precision);

    let kv_bytes = estimate_kv_cache_bytes(&KVCacheRequest {
        architecture: req.model.clone(),
        context_length: req.context_length,
        precision: req.precision,
    });

    let optimized_kv = estimate_kv_cache_bytes(&KVCacheRequest {
        architecture: req.model.clone(),
        context_length: req.context_length,
        precision: req.optimized_precision,
    });

    let total = model_memory + kv_bytes + RUNTIME_OVERHEAD_BYTES;
    let optimized_total = model_memory + optimized_kv + RUNTIME_OVERHEAD_BYTES;

    let (saved_bytes, saved_percent) = if total > optimized_total {
        let saved = total - optimized_total;
        let percent = if total > 0 {
            saved as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        (saved, percent)
    } else {
        (0, 0.0)
    };

    Ok(MemoryEstimate {
        model_memory_bytes: model_memory,
        kv_cache_memory_bytes: kv_bytes,
        runtime_overhead_bytes: RUNTIME_OVERHEAD_BYTES,
        total_memory_bytes: total,
        optimized_kv_bytes: optimized_kv,
        optimized_total_bytes: optimized_total,
        saved_bytes,
        saved_percent,
        recommended_strategy: format!(
            "{}-bit KV cache compression",
            req.optimized_precision.bits()
        ),
    })
}

/// Estimates model weight memory from parameter count and precision. If
/// `parameter_count` is unset (`0`), falls back to a rough estimate derived
/// from architecture dimensions.
fn model_memory_bytes(arch: &ModelArchitecture, precision: Precision) -> u64 {
    let params = if arch.parameter_count > 0 {
        arch.parameter_count
    } else {
        estimate_parameter_count(arch)
    };
    (params as f64 * precision.bytes_per_element()) as u64
}

/// Provides a rough transformer parameter count from architecture
/// dimensions when an explicit count isn't known. This is a coarse
/// approximation (roughly `12 * hidden_size^2` per layer, the standard
/// order-of-magnitude for attention + MLP blocks) intended only as a
/// fallback.
fn estimate_parameter_count(arch: &ModelArchitecture) -> u64 {
    if arch.num_layers == 0 || arch.hidden_size == 0 {
        return 0;
    }
    let per_layer = 12u64 * arch.hidden_size as u64 * arch.hidden_size as u64;
    per_layer * arch.num_layers as u64
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
    fn estimate_reports_expected_model_memory() {
        let req = MemoryRequest::new(test_architecture(), 4096, Precision::Fp16);
        let result = estimate(&req).unwrap();
        assert_eq!(result.model_memory_bytes, 7_000_000_000 * 2);
        assert!(result.kv_cache_memory_bytes > 0);
        assert_eq!(
            result.total_memory_bytes,
            result.model_memory_bytes + result.kv_cache_memory_bytes + RUNTIME_OVERHEAD_BYTES
        );
        assert!(result.optimized_total_bytes < result.total_memory_bytes);
        assert!(result.saved_bytes > 0);
        assert!(result.saved_percent > 0.0 && result.saved_percent < 100.0);
    }

    #[test]
    fn zero_context_length_is_an_error() {
        let req = MemoryRequest::new(test_architecture(), 0, Precision::Fp16);
        assert!(estimate(&req).is_err());
    }

    #[test]
    fn falls_back_to_architecture_derived_params() {
        let arch = ModelArchitecture {
            name: "no-param-count".to_string(),
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            parameter_count: 0,
        };
        let req = MemoryRequest::new(arch, 2048, Precision::Fp16);
        let result = estimate(&req).unwrap();
        assert!(result.model_memory_bytes > 0);
    }

    #[test]
    fn default_optimized_precision_is_int4() {
        let with_default = estimate(&MemoryRequest::new(
            test_architecture(),
            4096,
            Precision::Fp16,
        ))
        .unwrap();

        let mut explicit_req = MemoryRequest::new(test_architecture(), 4096, Precision::Fp16);
        explicit_req.optimized_precision = Precision::Int4;
        let explicit = estimate(&explicit_req).unwrap();

        assert_eq!(
            with_default.optimized_total_bytes,
            explicit.optimized_total_bytes
        );
    }

    #[test]
    fn precision_bytes_per_element() {
        assert_eq!(Precision::Fp16.bytes_per_element(), 2.0);
        assert_eq!(Precision::Fp8.bytes_per_element(), 1.0);
        assert_eq!(Precision::Int8.bytes_per_element(), 1.0);
        assert_eq!(Precision::Int4.bytes_per_element(), 0.5);
    }
}
