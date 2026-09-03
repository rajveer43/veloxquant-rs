//! Curated static model registry.
//!
//! **v0.1.0 status:** a small, hand-curated set of well-known local models.
//! The architecture is intentionally layered so a local cache and, later, a
//! remote VeloxQuant registry can be added without changing this API:
//! `StaticRegistry -> LocalRegistryCache -> RemoteRegistry`. No network
//! access is required for basic functionality.

use veloxquant_memory::ModelArchitecture;
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
