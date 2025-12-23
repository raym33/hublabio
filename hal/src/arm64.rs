//! ARM64-specific implementations

/// Initialize ARM64 HAL
pub fn init() {
    // Enable FPU/SIMD
    // Set up exception vectors
    // Initialize GIC (if present)
}

/// Get current CPU ID
pub fn cpu_id() -> usize {
    let mpidr: u64;
    unsafe {
        core::arch::asm!("mrs {}, MPIDR_EL1", out(reg) mpidr);
    }
    (mpidr & 0xFF) as usize
}

/// Exception levels
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExceptionLevel {
    EL0, // User
    EL1, // Kernel
    EL2, // Hypervisor
    EL3, // Secure Monitor
}

/// Get current exception level
pub fn current_el() -> ExceptionLevel {
    let el: u64;
    unsafe {
        core::arch::asm!("mrs {}, CurrentEL", out(reg) el);
    }
    match (el >> 2) & 0x3 {
        0 => ExceptionLevel::EL0,
        1 => ExceptionLevel::EL1,
        2 => ExceptionLevel::EL2,
        3 => ExceptionLevel::EL3,
        _ => unreachable!(),
    }
}

/// BCM2712 (Pi 5) specific addresses
pub mod bcm2712 {
    pub const PERIPHERAL_BASE: usize = 0x1_0000_0000;
    pub const GPIO_BASE: usize = PERIPHERAL_BASE + 0xD0000;
    pub const UART0_BASE: usize = PERIPHERAL_BASE + 0x40000;
    pub const GIC_DIST_BASE: usize = 0x4_0F00_0000;
    pub const GIC_CPU_BASE: usize = 0x4_0F00_1000;
}
