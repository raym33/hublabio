//! ARM64 Interrupt Handling
//!
//! Exception vectors, interrupt controller, and IRQ handling.

use spin::Mutex;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Exception types
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ExceptionType {
    SynchronousCurrentEl0 = 0,
    IrqCurrentEl0 = 1,
    FiqCurrentEl0 = 2,
    SerrorCurrentEl0 = 3,
    SynchronousCurrentElx = 4,
    IrqCurrentElx = 5,
    FiqCurrentElx = 6,
    SerrorCurrentElx = 7,
    SynchronousLowerEl64 = 8,
    IrqLowerEl64 = 9,
    FiqLowerEl64 = 10,
    SerrorLowerEl64 = 11,
    SynchronousLowerEl32 = 12,
    IrqLowerEl32 = 13,
    FiqLowerEl32 = 14,
    SerrorLowerEl32 = 15,
}

/// Exception syndrome register (ESR) classes
#[derive(Clone, Copy, Debug)]
pub enum ExceptionClass {
    Unknown = 0x00,
    WfeWfi = 0x01,
    Cp15 = 0x03,
    Cp14 = 0x05,
    Cp14Ldc = 0x06,
    SveSimdFp = 0x07,
    Cp10 = 0x0D,
    Pac = 0x09,
    Cp14Mrrc = 0x0C,
    BranchTarget = 0x0D,
    IllegalExec = 0x0E,
    Svc32 = 0x11,
    Hvc32 = 0x12,
    Smc32 = 0x13,
    Svc64 = 0x15,
    Hvc64 = 0x16,
    Smc64 = 0x17,
    Msr = 0x18,
    Sve = 0x19,
    InstructionAbortLower = 0x20,
    InstructionAbortCurrent = 0x21,
    PcAlignment = 0x22,
    DataAbortLower = 0x24,
    DataAbortCurrent = 0x25,
    SpAlignment = 0x26,
    Fp32 = 0x28,
    Fp64 = 0x2C,
    Serror = 0x2F,
    BreakpointLower = 0x30,
    BreakpointCurrent = 0x31,
    SoftwareStepLower = 0x32,
    SoftwareStepCurrent = 0x33,
    WatchpointLower = 0x34,
    WatchpointCurrent = 0x35,
    Brk32 = 0x38,
    Brk64 = 0x3C,
}

impl From<u8> for ExceptionClass {
    fn from(value: u8) -> Self {
        match value {
            0x15 => Self::Svc64,
            0x20 => Self::InstructionAbortLower,
            0x21 => Self::InstructionAbortCurrent,
            0x24 => Self::DataAbortLower,
            0x25 => Self::DataAbortCurrent,
            0x26 => Self::SpAlignment,
            0x2F => Self::Serror,
            _ => Self::Unknown,
        }
    }
}

/// Saved CPU state
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CpuState {
    pub x: [u64; 31],  // X0-X30
    pub sp: u64,       // Stack pointer
    pub pc: u64,       // Program counter (ELR_EL1)
    pub pstate: u64,   // Saved PSTATE (SPSR_EL1)
}

impl Default for CpuState {
    fn default() -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            pstate: 0,
        }
    }
}

/// IRQ handler type
pub type IrqHandler = fn(irq: u32) -> bool;

/// Registered IRQ handlers
static IRQ_HANDLERS: Mutex<[Option<IrqHandler>; 256]> = Mutex::new([None; 256]);

/// IRQ statistics
static IRQ_COUNT: [AtomicU64; 256] = {
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; 256]
};

/// Register IRQ handler
pub fn register_handler(irq: u32, handler: IrqHandler) {
    if irq < 256 {
        IRQ_HANDLERS.lock()[irq as usize] = Some(handler);
    }
}

/// Unregister IRQ handler
pub fn unregister_handler(irq: u32) {
    if irq < 256 {
        IRQ_HANDLERS.lock()[irq as usize] = None;
    }
}

/// Handle IRQ
pub fn handle_irq(irq: u32) {
    if irq < 256 {
        IRQ_COUNT[irq as usize].fetch_add(1, Ordering::Relaxed);

        let handler = IRQ_HANDLERS.lock()[irq as usize];
        if let Some(h) = handler {
            h(irq);
        }
    }
}

/// GIC (Generic Interrupt Controller) support
pub mod gic {
    use super::*;

    /// GIC Distributor registers
    pub const GICD_CTLR: usize = 0x000;
    pub const GICD_TYPER: usize = 0x004;
    pub const GICD_ISENABLER: usize = 0x100;
    pub const GICD_ICENABLER: usize = 0x180;
    pub const GICD_ISPENDR: usize = 0x200;
    pub const GICD_ICPENDR: usize = 0x280;
    pub const GICD_IPRIORITYR: usize = 0x400;
    pub const GICD_ITARGETSR: usize = 0x800;
    pub const GICD_ICFGR: usize = 0xC00;

    /// GIC CPU Interface registers
    pub const GICC_CTLR: usize = 0x000;
    pub const GICC_PMR: usize = 0x004;
    pub const GICC_BPR: usize = 0x008;
    pub const GICC_IAR: usize = 0x00C;
    pub const GICC_EOIR: usize = 0x010;

    /// GIC addresses for different platforms
    pub const GICD_BASE_RPI4: usize = 0xFF841000;
    pub const GICC_BASE_RPI4: usize = 0xFF842000;
    pub const GICD_BASE_QEMU: usize = 0x08000000;
    pub const GICC_BASE_QEMU: usize = 0x08010000;

    /// GIC state
    static GICD_BASE: AtomicU64 = AtomicU64::new(0);
    static GICC_BASE: AtomicU64 = AtomicU64::new(0);

    /// Initialize GIC
    pub fn init(gicd: usize, gicc: usize) {
        GICD_BASE.store(gicd as u64, Ordering::SeqCst);
        GICC_BASE.store(gicc as u64, Ordering::SeqCst);

        unsafe {
            // Disable distributor
            core::ptr::write_volatile((gicd + GICD_CTLR) as *mut u32, 0);

            // Get number of IRQs
            let typer = core::ptr::read_volatile((gicd + GICD_TYPER) as *const u32);
            let num_irqs = ((typer & 0x1F) + 1) * 32;

            // Disable all interrupts
            for i in (0..num_irqs).step_by(32) {
                core::ptr::write_volatile(
                    (gicd + GICD_ICENABLER + (i / 8) as usize) as *mut u32,
                    0xFFFFFFFF,
                );
            }

            // Set all interrupts to lowest priority
            for i in (0..num_irqs).step_by(4) {
                core::ptr::write_volatile(
                    (gicd + GICD_IPRIORITYR + i as usize) as *mut u32,
                    0xA0A0A0A0,
                );
            }

            // Target all interrupts to CPU 0
            for i in (32..num_irqs).step_by(4) {
                core::ptr::write_volatile(
                    (gicd + GICD_ITARGETSR + i as usize) as *mut u32,
                    0x01010101,
                );
            }

            // Configure all interrupts as level-triggered
            for i in (32..num_irqs).step_by(16) {
                core::ptr::write_volatile(
                    (gicd + GICD_ICFGR + (i / 4) as usize) as *mut u32,
                    0,
                );
            }

            // Enable distributor
            core::ptr::write_volatile((gicd + GICD_CTLR) as *mut u32, 1);

            // CPU interface
            // Set priority mask
            core::ptr::write_volatile((gicc + GICC_PMR) as *mut u32, 0xFF);

            // Enable CPU interface
            core::ptr::write_volatile((gicc + GICC_CTLR) as *mut u32, 1);
        }

        crate::kinfo!("GIC: Initialized with {} interrupts", num_irqs);
    }

    /// Enable interrupt
    pub fn enable_irq(irq: u32) {
        let gicd = GICD_BASE.load(Ordering::SeqCst) as usize;
        if gicd == 0 {
            return;
        }

        let reg = irq / 32;
        let bit = irq % 32;

        unsafe {
            core::ptr::write_volatile(
                (gicd + GICD_ISENABLER + (reg * 4) as usize) as *mut u32,
                1 << bit,
            );
        }
    }

    /// Disable interrupt
    pub fn disable_irq(irq: u32) {
        let gicd = GICD_BASE.load(Ordering::SeqCst) as usize;
        if gicd == 0 {
            return;
        }

        let reg = irq / 32;
        let bit = irq % 32;

        unsafe {
            core::ptr::write_volatile(
                (gicd + GICD_ICENABLER + (reg * 4) as usize) as *mut u32,
                1 << bit,
            );
        }
    }

    /// Acknowledge interrupt
    pub fn acknowledge() -> u32 {
        let gicc = GICC_BASE.load(Ordering::SeqCst) as usize;
        if gicc == 0 {
            return 0x3FF; // Spurious
        }

        unsafe { core::ptr::read_volatile((gicc + GICC_IAR) as *const u32) }
    }

    /// End of interrupt
    pub fn end_of_interrupt(irq: u32) {
        let gicc = GICC_BASE.load(Ordering::SeqCst) as usize;
        if gicc == 0 {
            return;
        }

        unsafe {
            core::ptr::write_volatile((gicc + GICC_EOIR) as *mut u32, irq);
        }
    }
}

/// BCM2711 interrupt controller (Raspberry Pi 4)
pub mod bcm2711 {
    /// Base addresses
    pub const IRQ_BASE: usize = 0xFE00B200;
    pub const LOCAL_IRQ_BASE: usize = 0xFF800000;

    /// Registers
    pub const IRQ_PENDING0: usize = 0x00;
    pub const IRQ_PENDING1: usize = 0x04;
    pub const IRQ_PENDING2: usize = 0x08;
    pub const FIQ_CONTROL: usize = 0x0C;
    pub const ENABLE_IRQ1: usize = 0x10;
    pub const ENABLE_IRQ2: usize = 0x14;
    pub const ENABLE_BASIC: usize = 0x18;
    pub const DISABLE_IRQ1: usize = 0x1C;
    pub const DISABLE_IRQ2: usize = 0x20;
    pub const DISABLE_BASIC: usize = 0x24;

    /// IRQ numbers
    pub const TIMER_IRQ: u32 = 1;
    pub const UART_IRQ: u32 = 57;
    pub const USB_IRQ: u32 = 9;
    pub const SD_IRQ: u32 = 56;
    pub const ETH_IRQ: u32 = 29;

    /// Enable IRQ
    pub fn enable_irq(irq: u32) {
        let base = IRQ_BASE;
        unsafe {
            if irq < 32 {
                core::ptr::write_volatile((base + ENABLE_IRQ1) as *mut u32, 1 << irq);
            } else if irq < 64 {
                core::ptr::write_volatile((base + ENABLE_IRQ2) as *mut u32, 1 << (irq - 32));
            }
        }
    }

    /// Disable IRQ
    pub fn disable_irq(irq: u32) {
        let base = IRQ_BASE;
        unsafe {
            if irq < 32 {
                core::ptr::write_volatile((base + DISABLE_IRQ1) as *mut u32, 1 << irq);
            } else if irq < 64 {
                core::ptr::write_volatile((base + DISABLE_IRQ2) as *mut u32, 1 << (irq - 32));
            }
        }
    }

    /// Get pending IRQs
    pub fn pending() -> (u32, u32) {
        let base = IRQ_BASE;
        unsafe {
            let p1 = core::ptr::read_volatile((base + IRQ_PENDING1) as *const u32);
            let p2 = core::ptr::read_volatile((base + IRQ_PENDING2) as *const u32);
            (p1, p2)
        }
    }
}

/// Exception vector table setup
pub fn setup_vectors() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        extern "C" {
            static __vectors: u8;
        }
        let vbar = &__vectors as *const u8 as u64;
        core::arch::asm!("msr vbar_el1, {}", in(reg) vbar);
    }
}

/// Enable interrupts
pub fn enable() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("msr daifclr, #2");
    }
}

/// Disable interrupts
pub fn disable() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("msr daifset, #2");
    }
}

/// Check if interrupts are enabled
pub fn are_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        let daif: u64;
        unsafe {
            core::arch::asm!("mrs {}, daif", out(reg) daif);
        }
        (daif & (1 << 7)) == 0
    }

    #[cfg(not(target_arch = "aarch64"))]
    false
}

/// Get IRQ count
pub fn irq_count(irq: u32) -> u64 {
    if irq < 256 {
        IRQ_COUNT[irq as usize].load(Ordering::Relaxed)
    } else {
        0
    }
}

/// Initialize interrupt subsystem
pub fn init() {
    setup_vectors();
    crate::kprintln!("  Interrupt controller initialized");
}
