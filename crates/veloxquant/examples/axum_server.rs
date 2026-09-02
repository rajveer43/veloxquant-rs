//! Exposes VeloxQuant memory estimation over a small Axum HTTP API.
//!
//! Demonstrates sharing a [`Client`] across request handlers via `Arc`
//! (`Client` is already cheap to clone, so wrapping it isn't strictly
//! required — this shows the idiom for services that also hold other
//! shared state).
//!
//! This example only depends on `veloxquant`'s memory/system services (no
//! running runtime needed). Run with:
//!
//! ```sh
//! cargo run --example axum_server --features examples-axum
//! curl 'http://localhost:3000/estimate?context=32768'
//! ```
//!
//! Requires adding `axum` as a dev-dependency; see the `Cargo.toml`
//! `[dev-dependencies]` section for this workspace.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use veloxquant::{Client, MemoryRequest, ModelArchitecture, Precision};

#[derive(Clone)]
struct AppState {
    client: Arc<Client>,
}

#[derive(Deserialize)]
struct EstimateParams {
    #[serde(default = "default_context")]
    context: usize,
}

fn default_context() -> usize {
    32768
}

async fn estimate(
    State(state): State<AppState>,
    Query(params): Query<EstimateParams>,
) -> Json<serde_json::Value> {
    let architecture = ModelArchitecture {
        name: "Qwen3-8B".to_string(),
        num_layers: 36,
        num_kv_heads: 8,
        head_dim: 128,
        hidden_size: 4096,
        parameter_count: 8_000_000_000,
    };

    let request = MemoryRequest::new(architecture, params.context, Precision::Fp16);

    match state.client.memory().estimate(&request) {
        Ok(estimate) => Json(serde_json::json!({
            "total_memory_bytes": estimate.total_memory_bytes,
            "optimized_total_bytes": estimate.optimized_total_bytes,
            "saved_percent": estimate.saved_percent,
        })),
        Err(err) => Json(serde_json::json!({ "error": err.to_string() })),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(Client::builder().auto_detect().build()?);
    let state = AppState { client };

    let app = Router::new()
        .route("/estimate", get(estimate))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("listening on http://localhost:3000");
    axum::serve(listener, app).await?;

    Ok(())
}
