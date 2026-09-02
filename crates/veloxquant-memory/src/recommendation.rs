//! Optimization profile and precision recommendations.

use veloxquant_core::OptimizationProfile;

use crate::estimator::{MemoryEstimate, Precision};

/// Chooses a KV-cache precision given how much memory is available versus
/// how much the naive (uncompressed) estimate would need. Returns the
/// recommended precision and a human-readable reason.
pub fn recommend_strategy(
    estimate: &MemoryEstimate,
    available_memory_bytes: u64,
) -> (Precision, String) {
    if available_memory_bytes == 0 {
        return (
            Precision::Int4,
            "available memory unknown; defaulting to the most memory-efficient option".to_string(),
        );
    }
    if estimate.total_memory_bytes <= available_memory_bytes {
        return (
            Precision::Fp16,
            "sufficient memory available; no compression required".to_string(),
        );
    }
    if estimate.optimized_total_bytes <= available_memory_bytes {
        return (
            Precision::Int4,
            "uncompressed footprint exceeds available memory; 4-bit KV compression fits within budget".to_string(),
        );
    }
    (
        Precision::Int4,
        "even with maximum compression, memory is tight; consider a smaller model or shorter context".to_string(),
    )
}

/// Full VeloxQuant optimization recommendation for a model/context/profile
/// combination.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// The profile this recommendation targets.
    pub profile: OptimizationProfile,
    /// Human-readable name of the compression method.
    pub compression_method: String,
    /// Bit width of the recommended compression.
    pub compression_bits: u8,
    /// Estimated total memory before optimization, in bytes.
    pub estimated_memory_before: u64,
    /// Estimated total memory after optimization, in bytes.
    pub estimated_memory_after: u64,
    /// Context length this recommendation was computed for.
    pub context_length: usize,
    /// Why this recommendation was chosen.
    pub reason: String,
}

/// Maps an [`OptimizationProfile`] to the [`Precision`] it targets.
pub fn precision_for_profile(profile: OptimizationProfile) -> Precision {
    match profile {
        OptimizationProfile::Speed => Precision::Fp16,
        OptimizationProfile::Balanced => Precision::Int8,
        OptimizationProfile::Memory | OptimizationProfile::MaximumContext => Precision::Int4,
    }
}

/// Infers the [`OptimizationProfile`] that best matches a chosen precision
/// and memory pressure, for cases where the precision was derived from
/// [`recommend_strategy`] rather than an explicit profile request.
pub fn profile_for_precision(
    precision: Precision,
    estimate: &MemoryEstimate,
    available_memory_bytes: u64,
) -> OptimizationProfile {
    match precision {
        Precision::Fp16 => OptimizationProfile::Speed,
        Precision::Int8 | Precision::Fp8 => OptimizationProfile::Balanced,
        Precision::Int4 => {
            if available_memory_bytes > 0
                && estimate.optimized_total_bytes > 0
                && available_memory_bytes < estimate.optimized_total_bytes * 2
            {
                OptimizationProfile::Memory
            } else {
                OptimizationProfile::MaximumContext
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::{estimate, MemoryRequest, ModelArchitecture};

    fn sample_estimate(context_length: usize) -> MemoryEstimate {
        let arch = ModelArchitecture {
            name: "test-model".to_string(),
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            parameter_count: 7_000_000_000,
        };
        estimate(&MemoryRequest::new(arch, context_length, Precision::Fp16)).unwrap()
    }

    #[test]
    fn recommends_fp16_when_memory_is_abundant() {
        let est = sample_estimate(4096);
        let (precision, _) = recommend_strategy(&est, u64::MAX);
        assert_eq!(precision, Precision::Fp16);
    }

    #[test]
    fn recommends_int4_when_memory_is_unknown() {
        let est = sample_estimate(4096);
        let (precision, _) = recommend_strategy(&est, 0);
        assert_eq!(precision, Precision::Int4);
    }

    #[test]
    fn recommends_int4_when_only_optimized_fits() {
        let est = sample_estimate(4096);
        let budget = est.optimized_total_bytes + 1;
        let (precision, _) = recommend_strategy(&est, budget);
        assert_eq!(precision, Precision::Int4);
    }

    #[test]
    fn profile_precision_roundtrip_for_speed_and_balanced() {
        assert_eq!(
            precision_for_profile(OptimizationProfile::Speed),
            Precision::Fp16
        );
        assert_eq!(
            precision_for_profile(OptimizationProfile::Balanced),
            Precision::Int8
        );
        assert_eq!(
            profile_for_precision(Precision::Fp16, &sample_estimate(4096), 0),
            OptimizationProfile::Speed
        );
    }
}
