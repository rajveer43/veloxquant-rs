//! # VeloxQuant for Rust
//!
//! Memory intelligence and optimization for local AI.
//!
//! ```no_run
//! use veloxquant::{Client, Message};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::builder().auto_detect().build()?;
//!
//!     let response = client
//!         .chat()?
//!         .create("mlx-community/Qwen3-8B-4bit", vec![Message::user("Hello!")])
//!         .await?;
//!
//!     println!("{}", response.text);
//!     Ok(())
//! }
//! ```
//!
//! See the crate [README](https://github.com/rajveer43/veloxquant-rs) for
//! the full feature roadmap and CLI documentation.

#[cfg(feature = "openai")]
pub mod chat;
pub mod client;
pub mod models;

pub use client::{Client, ClientBuilder};
pub use models::{ModelInfo, ModelRecommendationRequest, ModelRegistry, Task};

pub use veloxquant_core::{format_bytes, OptimizationProfile, Result, VeloxQuantError};
pub use veloxquant_memory::{
    MemoryEstimate, MemoryRequest, ModelArchitecture, OptimizationRecommendation,
    OptimizationRequest, Precision,
};
pub use veloxquant_system::SystemInfo;

#[cfg(feature = "openai")]
pub use chat::ChatApi;
#[cfg(feature = "openai")]
pub use veloxquant_openai::{ChatResponse, Message, RemoteModel, Role};

#[cfg(feature = "runtime")]
pub use veloxquant_runtime::RuntimeStatus;

#[cfg(feature = "monitor")]
pub use veloxquant_monitor::{Metrics, Monitor};
