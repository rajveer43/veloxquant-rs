//! Shared types used across VeloxQuant SDK crates.

use serde::{Deserialize, Serialize};

/// A VeloxQuant optimization profile, trading off speed, memory usage, and
/// context length.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizationProfile {
    /// Prioritize throughput and lower compression overhead.
    Speed,
    /// Balance memory savings against inference speed. The default profile.
    #[default]
    Balanced,
    /// Prioritize maximum memory reduction.
    Memory,
    /// Prioritize long context windows via aggressive KV-cache optimization.
    MaximumContext,
}

impl std::fmt::Display for OptimizationProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Speed => "speed",
            Self::Balanced => "balanced",
            Self::Memory => "memory",
            Self::MaximumContext => "maximum-context",
        };
        f.write_str(s)
    }
}

/// Formats a byte count as a human-readable string using binary
/// (1024-based) units, e.g. `25_769_803_776` -> `"24.0 GB"`.
pub fn format_bytes(bytes: u64) -> String {
    const UNIT: f64 = 1024.0;
    const UNITS: [&str; 5] = ["KB", "MB", "GB", "TB", "PB"];

    if bytes < UNIT as u64 {
        return format!("{bytes} B");
    }

    let mut value = bytes as f64 / UNIT;
    let mut idx = 0;
    while value >= UNIT && idx < UNITS.len() - 1 {
        value /= UNIT;
        idx += 1;
    }

    format!("{value:.1} {}", UNITS[idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_bytes_across_units() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.0 KB");
        assert_eq!(format_bytes(25_769_803_776), "24.0 GB");
    }

    #[test]
    fn default_profile_is_balanced() {
        assert_eq!(
            OptimizationProfile::default(),
            OptimizationProfile::Balanced
        );
    }

    #[test]
    fn displays_kebab_case() {
        assert_eq!(
            OptimizationProfile::MaximumContext.to_string(),
            "maximum-context"
        );
    }
}
