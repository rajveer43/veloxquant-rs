//! Hardware detection: combines platform, CPU, and memory information into
//! a single [`SystemInfo`] snapshot.

use serde::Serialize;
use sysinfo::System;
use veloxquant_core::OptimizationProfile;

use crate::memory::memory_stats;
use crate::platform::{architecture, is_apple_silicon, platform};

/// A snapshot of the host system relevant to running local LLM inference.
#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    /// OS platform, e.g. `"macos"`, `"linux"`, `"windows"`.
    pub platform: String,
    /// CPU architecture, e.g. `"aarch64"`, `"x86_64"`.
    pub architecture: String,
    /// Human-readable CPU model/brand string, when available.
    pub cpu_model: String,
    /// Whether this host is Apple Silicon (macOS on `aarch64`).
    pub apple_silicon: bool,
    /// Total physical/unified memory, in bytes.
    pub total_memory_bytes: u64,
    /// Currently available memory, in bytes.
    pub available_memory_bytes: u64,
    /// A default optimization profile recommended for this hardware.
    pub recommended_profile: OptimizationProfile,
}

/// Detects the current system's hardware and memory characteristics.
///
/// This never panics or errors: on unsupported or restricted platforms,
/// fields that can't be determined degrade to empty strings or `0`, and
/// `apple_silicon` is `false`.
pub fn detect() -> SystemInfo {
    let mem = memory_stats();
    let cpu_model = cpu_model();
    let apple_silicon = is_apple_silicon();

    SystemInfo {
        platform: platform().to_string(),
        architecture: architecture().to_string(),
        cpu_model,
        apple_silicon,
        total_memory_bytes: mem.total_bytes,
        available_memory_bytes: mem.available_bytes,
        recommended_profile: recommend_profile(mem.total_bytes),
    }
}

fn cpu_model() -> String {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_default()
}

/// Picks a default [`OptimizationProfile`] from total memory alone. This is
/// a coarse hardware-only default; [`crate::memory`]-aware callers should
/// prefer a model- and context-aware recommendation instead.
fn recommend_profile(total_memory_bytes: u64) -> OptimizationProfile {
    const GIB: u64 = 1024 * 1024 * 1024;

    if total_memory_bytes == 0 {
        return OptimizationProfile::Balanced;
    }
    if total_memory_bytes < 8 * GIB {
        OptimizationProfile::Memory
    } else if total_memory_bytes < 32 * GIB {
        OptimizationProfile::Balanced
    } else {
        OptimizationProfile::MaximumContext
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics_and_fills_platform_fields() {
        let info = detect();
        assert!(!info.platform.is_empty());
        assert!(!info.architecture.is_empty());
    }

    #[test]
    fn recommend_profile_scales_with_memory() {
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(recommend_profile(0), OptimizationProfile::Balanced);
        assert_eq!(recommend_profile(4 * GIB), OptimizationProfile::Memory);
        assert_eq!(recommend_profile(16 * GIB), OptimizationProfile::Balanced);
        assert_eq!(
            recommend_profile(64 * GIB),
            OptimizationProfile::MaximumContext
        );
    }

    #[test]
    fn apple_silicon_flag_matches_platform_module() {
        let info = detect();
        assert_eq!(info.apple_silicon, crate::platform::is_apple_silicon());
    }
}
