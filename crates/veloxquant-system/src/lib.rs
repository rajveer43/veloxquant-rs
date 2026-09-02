//! Hardware and platform detection for VeloxQuant: Apple Silicon detection,
//! CPU identification, and system memory inspection.
//!
//! ```
//! let info = veloxquant_system::detect();
//! println!("platform: {}, apple_silicon: {}", info.platform, info.apple_silicon);
//! ```

pub mod hardware;
pub mod memory;
pub mod platform;

pub use hardware::{detect, SystemInfo};
pub use memory::{memory_stats, MemoryStats};
pub use platform::{architecture, is_apple_silicon, platform};

/// Async-friendly system detector, mirroring the shape of other VeloxQuant
/// SDKs' `Detector` abstraction. Detection is CPU-bound and fast, so this
/// simply wraps [`detect`] — it exists so callers behind an async
/// [`Client`](veloxquant_core::client) can `.await` it uniformly.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemService;

impl SystemService {
    /// Creates a new [`SystemService`].
    pub fn new() -> Self {
        Self
    }

    /// Detects the current system's hardware and memory characteristics.
    pub async fn info(&self) -> SystemInfo {
        detect()
    }
}
