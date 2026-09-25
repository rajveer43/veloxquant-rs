//! System memory inspection.
//!
//! Total memory comes from `sysinfo` on every platform. Available memory
//! comes from `sysinfo` everywhere **except macOS**, where it is read from
//! Mach `host_statistics64` as `(free + inactive)` pages — the same number
//! the Swift SDK (`HostMemory.current()`) and Go SDK
//! (`availableMemoryFromVMStat`) report.
//!
//! sysinfo 0.32's macOS figure is `free + inactive + purgeable -
//! compressor`. On a Mac that has been up a while the compressor can hold
//! about as many pages as are inactive, so that figure collapses toward zero
//! (a 24 GiB machine with ~7 GiB free + inactive reported ~0.7 GiB), and
//! AutoPilot concluded that no model fits.

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
///
/// On macOS, available memory is `(free + inactive)` pages from Mach
/// `host_statistics64` (see the [module docs](self)), capped at total and
/// falling back to total if the Mach call fails, matching the Swift and Go
/// SDKs.
pub fn memory_stats() -> MemoryStats {
    let mut sys = System::new();
    sys.refresh_memory();
    let total_bytes = sys.total_memory();

    #[cfg(target_os = "macos")]
    let available_bytes = macos::available_memory(total_bytes);
    #[cfg(not(target_os = "macos"))]
    let available_bytes = sys.available_memory();

    MemoryStats {
        total_bytes,
        available_bytes,
    }
}

/// `(free + inactive) × page_size`, capped at `total_bytes`, with the Swift
/// and Go SDKs' fallback to `total_bytes` when the result is `0` or larger
/// than total (an implausible reading).
///
/// `free_count` is Mach's `free_count`, which already includes speculative
/// pages. (`vm_stat` prints "Pages free" as `free_count - speculative_count`,
/// which is why Go adds "Pages speculative" back in.) Compressor pages are
/// neither added nor subtracted.
#[cfg(any(target_os = "macos", test))]
fn available_from_page_counts(
    free_count: u64,
    inactive_count: u64,
    page_size: u64,
    total_bytes: u64,
) -> u64 {
    let available = free_count
        .saturating_add(inactive_count)
        .saturating_mul(page_size);
    if available == 0 || available > total_bytes {
        total_bytes
    } else {
        available
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::sync::OnceLock;

    /// Page counts from one `host_statistics64(HOST_VM_INFO64)` call.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct VmPageCounts {
        pub free_count: u64,
        pub inactive_count: u64,
        pub page_size: u64,
    }

    /// The host port. `mach_host_self()` adds a user reference to the same
    /// send right on every call, so it's fetched once per process rather than
    /// on every sample (the monitor sampler calls this every few seconds).
    fn host_port() -> libc::mach_port_t {
        static HOST: OnceLock<libc::mach_port_t> = OnceLock::new();
        // SAFETY: `mach_host_self` has no preconditions. libc marks it
        // deprecated in favour of the `mach2` crate, but the binding is
        // correct and avoids a second FFI dependency.
        #[allow(deprecated)]
        *HOST.get_or_init(|| unsafe { libc::mach_host_self() })
    }

    /// Reads free/inactive page counts and the page size, or `None` if the
    /// Mach call fails.
    pub(super) fn vm_page_counts() -> Option<VmPageCounts> {
        // SAFETY: `vm_statistics64` is plain old data, so all-zeroes is a
        // valid value.
        let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `stats` is a writable `vm_statistics64` and `count` is its
        // size in `integer_t` units, as `host_statistics64` requires. The
        // kernel writes at most `count` integers and updates `count`.
        let result = unsafe {
            libc::host_statistics64(
                host_port(),
                libc::HOST_VM_INFO64,
                std::ptr::addr_of_mut!(stats).cast::<libc::integer_t>(),
                &mut count,
            )
        };
        if result != libc::KERN_SUCCESS {
            return None;
        }

        // SAFETY: `sysconf` has no preconditions.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return None;
        }

        // Copy out of the `repr(packed)` struct before widening.
        let free_count = stats.free_count;
        let inactive_count = stats.inactive_count;
        Some(VmPageCounts {
            free_count: u64::from(free_count),
            inactive_count: u64::from(inactive_count),
            page_size: page_size as u64,
        })
    }

    /// Available memory on macOS, falling back to `total_bytes` if the Mach
    /// call fails (the Swift and Go SDKs' fallback).
    pub(super) fn available_memory(total_bytes: u64) -> u64 {
        match vm_page_counts() {
            Some(c) => super::available_from_page_counts(
                c.free_count,
                c.inactive_count,
                c.page_size,
                total_bytes,
            ),
            None => total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn memory_stats_available_never_exceeds_total_when_total_known() {
        let stats = memory_stats();
        if stats.total_bytes > 0 {
            assert!(stats.available_bytes <= stats.total_bytes);
        }
    }

    /// Regression test for the sysinfo macOS bug, using a real
    /// `host_statistics64` snapshot from a 24 GiB Apple Silicon Mac where
    /// AutoPilot wrongly reported that no model fits.
    #[test]
    fn page_count_formula_does_not_subtract_compressor_pages() {
        let page_size = 16_384;
        let total = 24 * GIB;
        // vm_stat: free 17,133 + speculative 13,576 (= Mach free_count),
        // inactive 448,166, purgeable 16,160, occupied by compressor 446,825.
        let (free_count, inactive, purgeable, compressor) =
            (17_133 + 13_576, 448_166_u64, 16_160_u64, 446_825_u64);

        // sysinfo 0.32's macOS formula: about 0.7 GiB.
        let sysinfo_available =
            (free_count + inactive + purgeable).saturating_sub(compressor) * page_size;
        assert!(sysinfo_available < GIB);

        // The fixed formula: about 7.3 GiB of free + inactive memory.
        let available = available_from_page_counts(free_count, inactive, page_size, total);
        assert_eq!(available, (free_count + inactive) * page_size);
        assert!(available > 7 * GIB, "available = {available}");
    }

    #[test]
    fn page_count_formula_falls_back_to_total_on_implausible_readings() {
        let total = 16 * GIB;
        assert_eq!(available_from_page_counts(0, 0, 16_384, total), total);
        assert_eq!(
            available_from_page_counts(u64::MAX, 1, 16_384, total),
            total
        );
        assert_eq!(available_from_page_counts(1, 1, 4_096, total), 8_192);
    }

    /// Sanity bounds on the live reading. The buggy sysinfo figure was about
    /// 3% of total on the machine above, so a 5% floor catches it; a healthy
    /// Mac's free + inactive memory is normally well above that.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_available_memory_is_not_absurdly_small() {
        let stats = memory_stats();
        assert!(stats.total_bytes > 0, "hw.memsize should be known on macOS");
        assert!(stats.available_bytes <= stats.total_bytes);
        assert!(
            stats.available_bytes >= stats.total_bytes / 20,
            "available {} bytes is under 5% of total {} bytes",
            stats.available_bytes,
            stats.total_bytes
        );
    }

    /// Cross-checks the Mach FFI reading (struct layout and page size)
    /// against `vm_stat`'s free + inactive + speculative pages, the Go SDK's
    /// formula. The two snapshots are taken moments apart, so the tolerance
    /// is 10% of total memory.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_page_counts_agree_with_vm_stat() {
        let Ok(output) = std::process::Command::new("vm_stat").output() else {
            return; // vm_stat unavailable (sandboxed); nothing to compare.
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let pages = |label: &str| -> u64 {
            text.lines()
                .find_map(|line| line.strip_prefix(label))
                .and_then(|rest| rest.trim().trim_end_matches('.').parse().ok())
                .unwrap_or_else(|| panic!("vm_stat has no {label:?} line:\n{text}"))
        };
        let vm_stat_pages =
            pages("Pages free:") + pages("Pages inactive:") + pages("Pages speculative:");

        let counts = macos::vm_page_counts().expect("host_statistics64 should succeed");
        let mach_pages = counts.free_count + counts.inactive_count;
        let total = memory_stats().total_bytes;

        let diff = mach_pages.abs_diff(vm_stat_pages) * counts.page_size;
        assert!(
            diff <= total / 10,
            "Mach free+inactive = {mach_pages} pages, vm_stat free+inactive+speculative = \
             {vm_stat_pages} pages (page size {})",
            counts.page_size
        );
    }
}
