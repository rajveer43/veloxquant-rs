//! Curated static model registry.
//!
//! **v0.1.0 status:** a small, hand-curated set of well-known local models.
//! The architecture is intentionally layered so a local cache and, later, a
//! remote VeloxQuant registry can be added without changing this API:
//! `StaticRegistry -> LocalRegistryCache -> RemoteRegistry`. No network
//! access is required for basic functionality.

use veloxquant_memory::ModelArchitecture;

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
}
