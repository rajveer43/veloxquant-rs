//! System memory inspection.

use sysinfo::System;

/// Total and available physical memory on the host, in bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryStats {
    /// Total installed physical (or unified) memory, in bytes.
    pub total_bytes: u64,
    /// Memory currently available for new allocations, in bytes.
    pub available_bytes: u64,
}

/// Queries total and available system memory. Never panics: on platforms
/// where `sysinfo` cannot determine memory, both fields are `0`.
pub fn memory_stats() -> MemoryStats {
    let mut sys = System::new();
    sys.refresh_memory();

    MemoryStats {
        total_bytes: sys.total_memory(),
        available_bytes: sys.available_memory(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_stats_available_never_exceeds_total_when_total_known() {
        let stats = memory_stats();
        if stats.total_bytes > 0 {
            assert!(stats.available_bytes <= stats.total_bytes);
        }
    }
}
