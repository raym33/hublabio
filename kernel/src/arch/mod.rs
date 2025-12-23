//! Architecture-specific code
//!
//! Currently supports ARM64 (AArch64).

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

/// Initialize architecture-specific features
pub fn init() {
    #[cfg(target_arch = "aarch64")]
    aarch64::init();

    crate::kprintln!("  Architecture: aarch64");
}

/// Halt the CPU
pub fn halt() {
    #[cfg(target_arch = "aarch64")]
    aarch64::halt();
}

/// Dump CPU state for debugging
pub fn dump_state() {
    #[cfg(target_arch = "aarch64")]
    aarch64::dump_state();
}

/// Enable interrupts
pub fn enable_interrupts() {
    #[cfg(target_arch = "aarch64")]
    aarch64::enable_interrupts();
}

/// Disable interrupts
pub fn disable_interrupts() {
    #[cfg(target_arch = "aarch64")]
    aarch64::disable_interrupts();
}

/// Check if interrupts are enabled
pub fn interrupts_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    return aarch64::interrupts_enabled();

    #[cfg(not(target_arch = "aarch64"))]
    false
}

/// Reboot the system
pub fn reboot() -> ! {
    #[cfg(target_arch = "aarch64")]
    aarch64::reboot();

    // Fallback: just halt
    loop {
        halt();
    }
}
