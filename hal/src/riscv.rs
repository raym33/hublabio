//! RISC-V specific implementations

/// Initialize RISC-V HAL
pub fn init() {
    // Set up trap handlers
    // Initialize PLIC (if present)
}

/// Get current CPU ID
pub fn cpu_id() -> usize {
    let hartid: usize;
    unsafe {
        core::arch::asm!("csrr {}, mhartid", out(reg) hartid);
    }
    hartid
}

/// Privilege mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrivilegeMode {
    User,
    Supervisor,
    Machine,
}

/// SBI calls
pub mod sbi {
    /// Console putchar
    pub fn console_putchar(c: u8) {
        unsafe {
            core::arch::asm!(
                "li a7, 0x01",
                "ecall",
                in("a0") c as usize,
            );
        }
    }

    /// Set timer
    pub fn set_timer(time: u64) {
        unsafe {
            core::arch::asm!(
                "li a7, 0x00",
                "ecall",
                in("a0") time,
            );
        }
    }

    /// Shutdown
    pub fn shutdown() -> ! {
        unsafe {
            core::arch::asm!("li a7, 0x08", "ecall",);
        }
        loop {}
    }
}
