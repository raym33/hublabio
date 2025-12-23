//! CPU abstraction

/// CPU information
pub struct CpuInfo {
    /// Number of cores
    pub cores: usize,
    /// Current core ID
    pub current_core: usize,
    /// Architecture name
    pub arch: &'static str,
}

impl CpuInfo {
    /// Get current CPU info
    pub fn current() -> Self {
        #[cfg(feature = "arm64")]
        let arch = "aarch64";
        #[cfg(feature = "riscv")]
        let arch = "riscv64";
        #[cfg(feature = "x86")]
        let arch = "x86_64";
        #[cfg(not(any(feature = "arm64", feature = "riscv", feature = "x86")))]
        let arch = "unknown";

        Self {
            cores: 4, // TODO: Detect actual core count
            current_core: super::cpu_id(),
            arch,
        }
    }
}

/// Halt the CPU
pub fn halt() {
    #[cfg(feature = "arm64")]
    unsafe {
        core::arch::asm!("wfe");
    }

    #[cfg(feature = "x86")]
    unsafe {
        core::arch::asm!("hlt");
    }

    #[cfg(feature = "riscv")]
    unsafe {
        core::arch::asm!("wfi");
    }
}

/// Memory barrier
pub fn memory_barrier() {
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}
