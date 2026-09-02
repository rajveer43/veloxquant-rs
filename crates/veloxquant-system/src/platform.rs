//! Platform and architecture identification.

/// Returns the current OS platform name, e.g. `"macos"`, `"linux"`, `"windows"`.
pub fn platform() -> &'static str {
    std::env::consts::OS
}

/// Returns the current CPU architecture, e.g. `"aarch64"`, `"x86_64"`.
pub fn architecture() -> &'static str {
    std::env::consts::ARCH
}

/// Reports whether the current process is running on Apple Silicon
/// (macOS on `aarch64`).
pub fn is_apple_silicon() -> bool {
    cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_and_architecture_are_non_empty() {
        assert!(!platform().is_empty());
        assert!(!architecture().is_empty());
    }

    #[test]
    fn apple_silicon_implies_macos_aarch64() {
        // Whatever the result, it must be consistent with cfg! at compile time.
        let expected = cfg!(target_os = "macos") && cfg!(target_arch = "aarch64");
        assert_eq!(is_apple_silicon(), expected);
    }
}
