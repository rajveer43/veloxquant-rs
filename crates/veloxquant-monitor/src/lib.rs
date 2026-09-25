//! Memory and inference metrics for the VeloxQuant Rust SDK.
//!
//! [`Monitor`] is broadcast-based pub/sub for [`Metrics`] samples.
//! [`Monitor::spawn_sampler`] drives it periodically from a [`Sampler`]
//! ([`SystemSampler`] samples host memory); callers can also publish
//! out-of-band samples by hand with [`Monitor::publish`].
//!
//! ```no_run
//! use std::time::Duration;
//! use veloxquant_monitor::{Monitor, SystemSampler};
//!
//! # async fn run() {
//! let monitor = Monitor::new();
//! let mut samples = monitor.subscribe();
//! let sampling = monitor.spawn_sampler(SystemSampler::new(), Duration::from_secs(1));
//!
//! if let Ok(metrics) = samples.recv().await {
//!     println!("used: {} bytes", metrics.memory_used_bytes);
//! }
//! sampling.stop().await;
//! # }
//! ```

pub mod metrics;
pub mod sampler;

pub use metrics::Metrics;
pub use sampler::{
    metrics_from_memory_stats, Monitor, Sampler, SamplingHandle, SystemSampler,
    DEFAULT_SAMPLE_INTERVAL,
};
