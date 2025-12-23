//! Interrupt handling abstraction

/// Disable interrupts and return previous state
pub fn disable() -> bool {
    #[cfg(feature = "arm64")]
    {
        let daif: u64;
        unsafe {
            core::arch::asm!("mrs {}, DAIF", out(reg) daif);
            core::arch::asm!("msr DAIFSet, #0xf");
        }
        (daif & 0x3c0) == 0
    }

    #[cfg(feature = "x86")]
    {
        let flags: u64;
        unsafe {
            core::arch::asm!("pushfq; pop {}", out(reg) flags);
            core::arch::asm!("cli");
        }
        (flags & 0x200) != 0
    }

    #[cfg(feature = "riscv")]
    {
        let status: usize;
        unsafe {
            core::arch::asm!("csrr {}, sstatus", out(reg) status);
            core::arch::asm!("csrc sstatus, {}", in(reg) 0x2);
        }
        (status & 0x2) != 0
    }

    #[cfg(not(any(feature = "arm64", feature = "riscv", feature = "x86")))]
    false
}

/// Enable interrupts
pub fn enable() {
    #[cfg(feature = "arm64")]
    unsafe {
        core::arch::asm!("msr DAIFClr, #0xf");
    }

    #[cfg(feature = "x86")]
    unsafe {
        core::arch::asm!("sti");
    }

    #[cfg(feature = "riscv")]
    unsafe {
        core::arch::asm!("csrs sstatus, {}", in(reg) 0x2);
    }
}

/// Restore interrupt state
pub fn restore(enabled: bool) {
    if enabled {
        enable();
    }
}

/// Execute closure with interrupts disabled
pub fn without_interrupts<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let was_enabled = disable();
    let result = f();
    restore(was_enabled);
    result
}
