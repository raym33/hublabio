//! Hardware Abstraction Layer for HubLab IO
//!
//! Provides architecture-specific implementations for:
//! - ARM64 (Raspberry Pi 5, Apple Silicon)
//! - RISC-V (upcoming boards)
//! - x86_64 (development/testing)

#![no_std]
#![cfg_attr(feature = "arm64", feature(asm_const))]

extern crate alloc;

pub mod cpu;
pub mod gpio;
pub mod interrupts;
pub mod memory;
pub mod timer;
pub mod uart;

#[cfg(feature = "arm64")]
pub mod arm64;

#[cfg(feature = "riscv")]
pub mod riscv;

#[cfg(feature = "x86")]
pub mod x86;

/// HAL initialization
pub fn init() {
    #[cfg(feature = "arm64")]
    arm64::init();

    #[cfg(feature = "riscv")]
    riscv::init();

    #[cfg(feature = "x86")]
    x86::init();
}

/// Get current CPU ID
pub fn cpu_id() -> usize {
    #[cfg(feature = "arm64")]
    return arm64::cpu_id();

    #[cfg(feature = "riscv")]
    return riscv::cpu_id();

    #[cfg(feature = "x86")]
    return x86::cpu_id();

    #[cfg(not(any(feature = "arm64", feature = "riscv", feature = "x86")))]
    0
}
