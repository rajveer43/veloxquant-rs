//! VeloxQuant's memory intelligence: estimating how much RAM a model and
//! its KV cache will need, and how much VeloxQuant compression can save.

pub mod compression;
pub mod estimator;
pub mod kv_cache;
pub mod recommendation;

pub use compression::{apply_compression_ratio, KVCacheOptimizer};
pub use estimator::{
    estimate, MemoryEstimate, MemoryRequest, ModelArchitecture, Precision, RUNTIME_OVERHEAD_BYTES,
};
pub use kv_cache::{estimate_kv_cache_bytes, KVCacheRequest};
pub use recommendation::{
    precision_for_profile, profile_for_precision, recommend_strategy, OptimizationRecommendation,
};

use veloxquant_core::{OptimizationProfile, Result, VeloxQuantError};

/// A request for a full optimization recommendation, combining a model,
/// context length, available memory, and (optionally) a forced profile.
#[derive(Debug, Clone)]
pub struct OptimizationRequest {
    /// The model to recommend an optimization strategy for.
    pub model: ModelArchitecture,
    /// The target context length, in tokens.
    pub context_length: usize,
    /// Memory available on the host, in bytes. `0` means unknown.
    pub available_memory_bytes: u64,
    /// If set, forces this profile instead of deriving one from available
    /// memory.
    pub profile: Option<OptimizationProfile>,
}

/// Async-friendly facade over memory estimation, mirroring
/// `client.memory()` in the public SDK surface.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryService;

impl MemoryService {
    /// Creates a new [`MemoryService`].
    pub fn new() -> Self {
        Self
    }

    /// Estimates model, KV-cache, and total memory for a request.
    pub fn estimate(&self, req: &MemoryRequest) -> Result<MemoryEstimate> {
        estimate(req)
    }
}

/// Async-friendly facade over optimization recommendations, mirroring
/// `client.optimize()` in the public SDK surface.
#[derive(Debug, Clone, Copy, Default)]
pub struct OptimizationService;

impl OptimizationService {
    /// Creates a new [`OptimizationService`].
    pub fn new() -> Self {
        Self
    }

    /// Produces a full optimization recommendation for a model/context
    /// combination, either using the forced `profile` in `req` or deriving
    /// one from `available_memory_bytes`.
    pub fn recommend(&self, req: &OptimizationRequest) -> Result<OptimizationRecommendation> {
        if req.context_length == 0 {
            return Err(VeloxQuantError::InvalidRequest(format!(
                "optimize recommend for {}: context length must be positive",
                req.model.name
            )));
        }

        let base_estimate = estimate(&MemoryRequest::new(
            req.model.clone(),
            req.context_length,
            Precision::Fp16,
        ))?;

        let (precision, reason, profile) = if let Some(profile) = req.profile {
            let precision = precision_for_profile(profile);
            (
                precision,
                format!("using explicitly requested {profile} profile"),
                profile,
            )
        } else {
            let (precision, reason) =
                recommend_strategy(&base_estimate, req.available_memory_bytes);
            let profile =
                profile_for_precision(precision, &base_estimate, req.available_memory_bytes);
            (precision, reason, profile)
        };

        let mut optimized_req =
            MemoryRequest::new(req.model.clone(), req.context_length, Precision::Fp16);
        optimized_req.optimized_precision = precision;
        let optimized_estimate = estimate(&optimized_req)?;

        Ok(OptimizationRecommendation {
            profile,
            compression_method: "VeloxQuant KV-cache compression".to_string(),
            compression_bits: precision.bits(),
            estimated_memory_before: optimized_estimate.total_memory_bytes,
            estimated_memory_after: optimized_estimate.optimized_total_bytes,
            context_length: req.context_length,
            reason,
        })
    }
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
    fn optimization_service_recommends_without_forced_profile() {
        let svc = OptimizationService::new();
        let rec = svc
            .recommend(&OptimizationRequest {
                model: test_architecture(),
                context_length: 32768,
                available_memory_bytes: 8 * 1024 * 1024 * 1024,
                profile: None,
            })
            .unwrap();
        assert!(rec.estimated_memory_after <= rec.estimated_memory_before);
        assert_eq!(rec.context_length, 32768);
    }

    #[test]
    fn optimization_service_honors_forced_profile() {
        let svc = OptimizationService::new();
        let rec = svc
            .recommend(&OptimizationRequest {
                model: test_architecture(),
                context_length: 4096,
                available_memory_bytes: 0,
                profile: Some(OptimizationProfile::Speed),
            })
            .unwrap();
        assert_eq!(rec.profile, OptimizationProfile::Speed);
        assert_eq!(rec.compression_bits, 16);
    }

    #[test]
    fn optimization_service_rejects_zero_context_length() {
        let svc = OptimizationService::new();
        let result = svc.recommend(&OptimizationRequest {
            model: test_architecture(),
            context_length: 0,
            available_memory_bytes: 0,
            profile: None,
        });
        assert!(result.is_err());
    }
}
