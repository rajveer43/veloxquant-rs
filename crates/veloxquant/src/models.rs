//! Curated static model registry.
//!
//! **v0.1.0 status:** a small, hand-curated set of well-known local models.
//! The architecture is intentionally layered so a local cache and, later, a
//! remote VeloxQuant registry can be added without changing this API:
//! `StaticRegistry -> LocalRegistryCache -> RemoteRegistry`. No network
//! access is required for basic functionality.

use veloxquant_core::Result;
use veloxquant_memory::{MemoryRequest, ModelArchitecture, Precision};
#[cfg(feature = "openai")]
use veloxquant_openai::RemoteModel;

/// A task an LLM might be used for, used to filter model recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// General-purpose coding assistance.
    Coding,
    /// Conversational chat.
    Chat,
    /// Multi-step reasoning.
    Reasoning,
    /// Vision-language tasks.
    Vision,
    /// Autonomous agent workloads.
    Agent,
    /// Translation between languages.
    Translation,
}

/// A curated model registry entry.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The model's identifier, as passed to [`crate::chat::ChatApi::create`].
    pub name: String,
    /// The model's architecture, used for memory estimation.
    pub architecture: ModelArchitecture,
    /// Whether VeloxQuant Rust currently supports running this model.
    pub supported: bool,
    /// Whether this model is a recommended default for general use.
    pub recommended: bool,
    /// Tasks this model is well-suited for.
    pub tasks: Vec<Task>,
}

fn registry() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            name: "mlx-community/Qwen3-8B-4bit".to_string(),
            architecture: ModelArchitecture {
                name: "Qwen3-8B".to_string(),
                num_layers: 36,
                num_kv_heads: 8,
                head_dim: 128,
                hidden_size: 4096,
                parameter_count: 8_000_000_000,
            },
            supported: true,
            recommended: true,
            tasks: vec![Task::Chat, Task::Reasoning, Task::Agent],
        },
        ModelInfo {
            name: "mlx-community/Qwen3-Coder-30B-4bit".to_string(),
            architecture: ModelArchitecture {
                name: "Qwen3-Coder-30B".to_string(),
                num_layers: 48,
                num_kv_heads: 8,
                head_dim: 128,
                hidden_size: 6144,
                parameter_count: 30_000_000_000,
            },
            supported: true,
            recommended: true,
            tasks: vec![Task::Coding, Task::Agent],
        },
        ModelInfo {
            name: "mlx-community/gemma-2-9b-4bit".to_string(),
            architecture: ModelArchitecture {
                name: "Gemma-2-9B".to_string(),
                num_layers: 42,
                num_kv_heads: 8,
                head_dim: 256,
                hidden_size: 3584,
                parameter_count: 9_000_000_000,
            },
            supported: true,
            recommended: false,
            tasks: vec![Task::Chat, Task::Translation],
        },
    ]
}

/// A request for model recommendations, filtered by task and available
/// memory.
#[derive(Debug, Clone)]
pub struct ModelRecommendationRequest {
    /// Task the recommended models should suit. `None` matches any task.
    pub task: Option<Task>,
    /// Memory available on the host, in bytes. `0` disables the memory
    /// filter (recommends by task only).
    pub available_memory_bytes: u64,
}

/// Context length [`ModelRegistry::recommend_scored`] ranks against when
/// none is given — Go's `RecommendScored` default.
pub const DEFAULT_RANKING_CONTEXT_LENGTH: usize = 8192;

/// How much memory headroom contributes to a candidate's score, relative to
/// [`RECOMMENDED_BONUS`] — Go's `headroomWeight`.
const HEADROOM_WEIGHT: f64 = 1.0;

/// Added to the score of registry entries marked `recommended`, so a
/// curated pick outranks an equally-fitting alternative — Go's
/// `recommendedBonus`.
const RECOMMENDED_BONUS: f64 = 0.5;

/// A candidate model with the score and reason
/// [`ModelRegistry::recommend_scored`] ranked it by — Go's `models.Scored`.
#[derive(Debug, Clone)]
pub struct ScoredModel {
    /// The candidate.
    pub info: ModelInfo,
    /// Ranking value, higher is better. Only meaningful relative to other
    /// candidates from the same call.
    pub score: f64,
    /// Human-readable explanation of the score.
    pub reason: String,
}

fn task_label(task: Option<Task>) -> String {
    match task {
        Some(task) => format!("{task:?}").to_lowercase(),
        None => String::new(),
    }
}

/// Facade over the curated model registry, mirroring `client.models()` in
/// the public SDK surface.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelRegistry;

impl ModelRegistry {
    /// Creates a new [`ModelRegistry`].
    pub fn new() -> Self {
        Self
    }

    /// Lists every model in the curated registry.
    pub fn list(&self) -> Vec<ModelInfo> {
        registry()
    }

    /// Looks up a curated registry entry by its exact `name` — Go's
    /// `Registry.Get`.
    pub fn get(&self, name: &str) -> Option<ModelInfo> {
        registry().into_iter().find(|m| m.name == name)
    }

    /// Ranks supported models for `req.task` (any task if `None`), best
    /// first, with a human-readable reason for each — a port of Go's
    /// `models.RecommendScored`.
    ///
    /// When `req.available_memory_bytes` is non-zero, each candidate's
    /// total footprint is estimated at int4 weights + int4 KV cache for
    /// `context_length` tokens (default
    /// [`DEFAULT_RANKING_CONTEXT_LENGTH`]); models that don't fit are
    /// dropped, and the rest score higher the more headroom they leave.
    /// Registry entries marked `recommended` get a fixed bonus. Ties keep
    /// registry order.
    ///
    /// Unlike [`ModelRegistry::recommend`] (unchanged, weights-only, and
    /// unranked), this includes the KV cache and the runtime overhead in
    /// the fit check, matching Go.
    pub fn recommend_scored(
        &self,
        req: &ModelRecommendationRequest,
        context_length: Option<usize>,
    ) -> Result<Vec<ScoredModel>> {
        let context_length = context_length
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_RANKING_CONTEXT_LENGTH);
        let task = task_label(req.task);

        let mut candidates = Vec::new();
        for model in registry() {
            if !model.supported {
                continue;
            }
            if let Some(t) = req.task {
                if !model.tasks.contains(&t) {
                    continue;
                }
            }

            let mut score = if model.recommended {
                RECOMMENDED_BONUS
            } else {
                0.0
            };

            if req.available_memory_bytes == 0 {
                candidates.push(ScoredModel {
                    reason: format!(
                        "matches task {task:?}; no memory budget given to rank by headroom"
                    ),
                    info: model,
                    score,
                });
                continue;
            }

            let estimate = veloxquant_memory::estimate(&MemoryRequest {
                model: model.architecture.clone(),
                context_length,
                precision: Precision::Int4,
                optimized_precision: Precision::Int4,
            })?;
            if estimate.total_memory_bytes > req.available_memory_bytes {
                continue;
            }
            let headroom =
                1.0 - estimate.total_memory_bytes as f64 / req.available_memory_bytes as f64;
            score += headroom * HEADROOM_WEIGHT;
            candidates.push(ScoredModel {
                reason: format!(
                    "fits task {task:?} with {:.0}% memory headroom at {context_length}-token context",
                    headroom * 100.0
                ),
                info: model,
                score,
            });
        }

        // Stable sort: equal scores keep registry order, as in Go.
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(candidates)
    }

    /// Merges live models reported by a runtime's `GET /v1/models` with the
    /// curated registry.
    ///
    /// Curated entries take precedence: when a remote model's `id` matches
    /// a curated entry's `name`, the curated entry (with its richer
    /// [`ModelArchitecture`], task list, etc.) is kept as-is. Remote models
    /// with no curated match are appended as minimal, unsupported entries —
    /// their `architecture` fields are zeroed since nothing is known about
    /// them beyond the id, so they should not be used for memory
    /// estimation.
    #[cfg(feature = "openai")]
    pub fn merge_remote(&self, remote: Vec<RemoteModel>) -> Vec<ModelInfo> {
        let mut models = registry();

        for remote_model in remote {
            if models.iter().any(|m| m.name == remote_model.id) {
                continue;
            }
            models.push(ModelInfo {
                name: remote_model.id,
                architecture: ModelArchitecture {
                    name: String::new(),
                    num_layers: 0,
                    num_kv_heads: 0,
                    head_dim: 0,
                    hidden_size: 0,
                    parameter_count: 0,
                },
                supported: true,
                recommended: false,
                tasks: Vec::new(),
            });
        }

        models
    }

    /// Recommends models matching a task and/or available memory, using a
    /// conservative FP16-weights-only check against `available_memory_bytes`
    /// (KV cache is excluded since context length isn't known here).
    pub fn recommend(&self, req: &ModelRecommendationRequest) -> Vec<ModelInfo> {
        registry()
            .into_iter()
            .filter(|m| match req.task {
                Some(t) => m.tasks.contains(&t),
                None => true,
            })
            .filter(|m| {
                if req.available_memory_bytes == 0 {
                    return true;
                }
                // int4 weights ≈ 0.5 bytes/param; a rough fit check.
                (m.architecture.parameter_count / 2) < req.available_memory_bytes
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_returns_curated_entries() {
        let registry = ModelRegistry::new();
        assert!(!registry.list().is_empty());
    }

    #[test]
    fn recommend_filters_by_task() {
        let registry = ModelRegistry::new();
        let coding = registry.recommend(&ModelRecommendationRequest {
            task: Some(Task::Coding),
            available_memory_bytes: 0,
        });
        assert!(coding.iter().all(|m| m.tasks.contains(&Task::Coding)));
        assert!(!coding.is_empty());
    }

    #[test]
    fn recommend_filters_by_available_memory() {
        let registry = ModelRegistry::new();
        let tiny_budget = registry.recommend(&ModelRecommendationRequest {
            task: None,
            available_memory_bytes: 1024, // 1 KB — nothing should fit
        });
        assert!(tiny_budget.is_empty());
    }

    #[test]
    fn get_finds_exact_name_only() {
        let registry = ModelRegistry::new();
        let first = registry.list().remove(0);
        assert_eq!(registry.get(&first.name).unwrap().name, first.name);
        assert!(registry.get("does-not-exist").is_none());
    }

    #[test]
    fn recommend_scored_ranks_recommended_model_first_without_budget() {
        let registry = ModelRegistry::new();
        // Chat: Qwen3-8B (recommended) and Gemma-2-9B (not recommended).
        let scored = registry
            .recommend_scored(
                &ModelRecommendationRequest {
                    task: Some(Task::Chat),
                    available_memory_bytes: 0,
                },
                None,
            )
            .unwrap();
        assert!(scored.len() >= 2);
        assert!(scored[0].info.recommended);
        assert!(scored.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(scored[0].reason.contains("no memory budget"));
        assert!(scored[0].reason.contains("\"chat\""));
    }

    #[test]
    fn recommend_scored_drops_models_that_do_not_fit_and_explains_headroom() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let registry = ModelRegistry::new();
        // 8 GiB: Qwen3-8B at int4 (~4 GB weights + small KV + 0.5 GB
        // overhead) fits; the 30B coder (~15 GB of int4 weights) does not.
        let scored = registry
            .recommend_scored(
                &ModelRecommendationRequest {
                    task: None,
                    available_memory_bytes: 8 * GIB,
                },
                Some(8192),
            )
            .unwrap();
        assert!(scored
            .iter()
            .all(|s| s.info.name != "mlx-community/Qwen3-Coder-30B-4bit"));
        assert_eq!(scored[0].info.name, "mlx-community/Qwen3-8B-4bit");
        assert!(scored[0]
            .reason
            .contains("memory headroom at 8192-token context"));
        assert!(scored.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn recommend_scored_returns_empty_when_nothing_fits() {
        let registry = ModelRegistry::new();
        let scored = registry
            .recommend_scored(
                &ModelRecommendationRequest {
                    task: None,
                    available_memory_bytes: 1024,
                },
                None,
            )
            .unwrap();
        assert!(scored.is_empty());
    }

    #[cfg(feature = "openai")]
    #[test]
    fn merge_remote_keeps_curated_entry_on_name_match() {
        let registry = ModelRegistry::new();
        let merged = registry.merge_remote(vec![RemoteModel {
            id: "mlx-community/Qwen3-8B-4bit".to_string(),
            object: "model".to_string(),
        }]);

        // No duplicate: the curated entry is reused rather than appended.
        assert_eq!(
            merged
                .iter()
                .filter(|m| m.name == "mlx-community/Qwen3-8B-4bit")
                .count(),
            1
        );
        let matched = merged
            .iter()
            .find(|m| m.name == "mlx-community/Qwen3-8B-4bit")
            .unwrap();
        assert_eq!(matched.architecture.num_layers, 36); // curated data preserved
    }

    #[cfg(feature = "openai")]
    #[test]
    fn merge_remote_appends_unmatched_models_as_minimal_entries() {
        let registry = ModelRegistry::new();
        let curated_count = registry.list().len();
        let merged = registry.merge_remote(vec![RemoteModel {
            id: "some-custom-local-model".to_string(),
            object: "model".to_string(),
        }]);

        assert_eq!(merged.len(), curated_count + 1);
        let extra = merged
            .iter()
            .find(|m| m.name == "some-custom-local-model")
            .unwrap();
        assert!(extra.supported);
        assert!(!extra.recommended);
        assert_eq!(extra.architecture.num_layers, 0);
    }

    #[cfg(feature = "openai")]
    #[test]
    fn merge_remote_with_empty_list_returns_curated_only() {
        let registry = ModelRegistry::new();
        let merged = registry.merge_remote(vec![]);
        assert_eq!(merged.len(), registry.list().len());
    }
}
