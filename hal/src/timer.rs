//! Timer abstraction

/// Get current timestamp in nanoseconds
pub fn now_ns() -> u64 {
    #[cfg(feature = "arm64")]
    {
        let cnt: u64;
        let freq: u64;
        unsafe {
            core::arch::asm!("mrs {}, CNTPCT_EL0", out(reg) cnt);
            core::arch::asm!("mrs {}, CNTFRQ_EL0", out(reg) freq);
        }
        if freq > 0 {
            (cnt * 1_000_000_000) / freq
        } else {
            cnt
        }
    }

    #[cfg(feature = "x86")]
    {
        let lo: u32;
        let hi: u32;
        unsafe {
            core::arch::asm!("rdtsc", out("eax") lo, out("edx") hi);
        }
        let tsc = ((hi as u64) << 32) | (lo as u64);
        // Assume ~2GHz for now, should be calibrated
        tsc / 2
    }

    #[cfg(feature = "riscv")]
    {
        let time: u64;
        unsafe {
            core::arch::asm!("rdtime {}", out(reg) time);
        }
        // Assume 10MHz timer frequency
        time * 100
    }

    #[cfg(not(any(feature = "arm64", feature = "riscv", feature = "x86")))]
    0
}

/// Busy wait for specified nanoseconds
pub fn delay_ns(ns: u64) {
    let start = now_ns();
    while now_ns() - start < ns {
        core::hint::spin_loop();
    }
}

/// Busy wait for specified microseconds
pub fn delay_us(us: u64) {
    delay_ns(us * 1000);
}

/// Busy wait for specified milliseconds
pub fn delay_ms(ms: u64) {
    delay_ns(ms * 1_000_000);
}
