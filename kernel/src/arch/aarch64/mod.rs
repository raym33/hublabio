//! AArch64 (ARM64) specific code

pub mod exception;
pub mod interrupt;
pub mod mmu;

use core::arch::asm;

/// Initialize ARM64 specific features
pub fn init() {
    // Set up exception vectors
    exception::init();

    // Enable FPU/SIMD
    enable_fpu();
    crate::kprintln!("  FPU/SIMD enabled");

    // Initialize GIC interrupt controller
    init_gic();

    // Initialize and start timer
    init_timer();
}

/// Initialize the GIC interrupt controller
fn init_gic() {
    // Detect platform and use appropriate GIC addresses
    // For QEMU virt machine
    const GICD_QEMU: usize = 0x0800_0000;
    const GICC_QEMU: usize = 0x0801_0000;

    // For Raspberry Pi 4
    const GICD_RPI4: usize = 0xFF84_1000;
    const GICC_RPI4: usize = 0xFF84_2000;

    // Try to detect which platform we're on
    // For now, assume QEMU unless we detect RPi4
    let (gicd, gicc) = if is_rpi4() {
        (GICD_RPI4, GICC_RPI4)
    } else {
        (GICD_QEMU, GICC_QEMU)
    };

    interrupt::gic::init(gicd, gicc);

    // Enable timer interrupt in GIC
    // Physical timer IRQ is typically 30 (QEMU) or 27 (non-secure)
    interrupt::gic::enable_irq(30);
    interrupt::gic::enable_irq(27);

    crate::kprintln!("  GIC initialized at GICD=0x{:x} GICC=0x{:x}", gicd, gicc);
}

/// Detect if we're running on Raspberry Pi 4
fn is_rpi4() -> bool {
    // Check MIDR_EL1 for Cortex-A72 (RPi4)
    let midr: u64;
    unsafe {
        asm!("mrs {}, midr_el1", out(reg) midr);
    }
    // Cortex-A72 part number is 0xD08
    let part = (midr >> 4) & 0xFFF;
    part == 0xD08
}

/// Initialize the system timer
fn init_timer() {
    // Get timer frequency
    let freq = read_timer_freq();
    crate::kprintln!("  Timer frequency: {} Hz", freq);

    // Set up timer for 10ms tick (100 Hz)
    let tick_interval = freq / 100;

    unsafe {
        // Set timer compare value
        asm!(
            "msr cntp_tval_el0, {0}",
            in(reg) tick_interval,
        );

        // Enable timer, unmask interrupt
        asm!(
            "mov x0, #1",
            "msr cntp_ctl_el0, x0",
            out("x0") _,
        );
    }

    crate::kprintln!("  Timer started with {} ticks/interval", tick_interval);
}

/// Start the preemptive scheduler timer
pub fn start_scheduler_timer() {
    // Enable IRQ
    enable_interrupts();
    crate::kprintln!("  Scheduler timer enabled");
}

/// Enable FPU and SIMD
fn enable_fpu() {
    unsafe {
        // CPACR_EL1: Enable FP/SIMD
        asm!(
            "mrs x0, cpacr_el1",
            "orr x0, x0, #(3 << 20)",  // FPEN = 11
            "msr cpacr_el1, x0",
            "isb",
            out("x0") _,
        );
    }
}

/// Halt the CPU (wait for interrupt)
pub fn halt() {
    unsafe {
        asm!("wfi");
    }
}

/// Enable interrupts
pub fn enable_interrupts() {
    unsafe {
        asm!("msr daifclr, #0xf");
    }
}

/// Disable interrupts
pub fn disable_interrupts() {
    unsafe {
        asm!("msr daifset, #0xf");
    }
}

/// Check if interrupts are enabled
pub fn interrupts_enabled() -> bool {
    let daif: u64;
    unsafe {
        asm!("mrs {}, daif", out(reg) daif);
    }
    (daif & 0x3C0) == 0
}

/// Get current exception level
pub fn current_el() -> u8 {
    let el: u64;
    unsafe {
        asm!("mrs {}, CurrentEL", out(reg) el);
    }
    ((el >> 2) & 3) as u8
}

/// Dump CPU state for debugging
pub fn dump_state() {
    let pc: u64;
    let sp: u64;
    let elr: u64;
    let spsr: u64;
    let esr: u64;
    let far: u64;

    unsafe {
        asm!(
            "adr {pc}, .",
            "mov {sp}, sp",
            "mrs {elr}, elr_el1",
            "mrs {spsr}, spsr_el1",
            "mrs {esr}, esr_el1",
            "mrs {far}, far_el1",
            pc = out(reg) pc,
            sp = out(reg) sp,
            elr = out(reg) elr,
            spsr = out(reg) spsr,
            esr = out(reg) esr,
            far = out(reg) far,
        );
    }

    crate::kprintln!("CPU State:");
    crate::kprintln!("  PC:   0x{:016x}", pc);
    crate::kprintln!("  SP:   0x{:016x}", sp);
    crate::kprintln!("  ELR:  0x{:016x}", elr);
    crate::kprintln!("  SPSR: 0x{:016x}", spsr);
    crate::kprintln!("  ESR:  0x{:016x}", esr);
    crate::kprintln!("  FAR:  0x{:016x}", far);
    crate::kprintln!("  EL:   {}", current_el());
}

/// Memory barrier - data synchronization
#[inline(always)]
pub fn dsb() {
    unsafe {
        asm!("dsb sy");
    }
}

/// Memory barrier - instruction synchronization
#[inline(always)]
pub fn isb() {
    unsafe {
        asm!("isb");
    }
}

/// Memory barrier - data memory barrier
#[inline(always)]
pub fn dmb() {
    unsafe {
        asm!("dmb sy");
    }
}

/// Invalidate TLB
pub fn tlb_invalidate_all() {
    unsafe {
        asm!("dsb ishst", "tlbi vmalle1is", "dsb ish", "isb",);
    }
}

/// Invalidate instruction cache
pub fn icache_invalidate_all() {
    unsafe {
        asm!("ic iallu", "isb",);
    }
}

/// Invalidate data cache
pub fn dcache_invalidate_all() {
    unsafe {
        asm!(
            "dsb sy", // Would need a loop over cache levels
            "isb",
        );
    }
}

/// Read system timer counter
pub fn read_timer() -> u64 {
    let cnt: u64;
    unsafe {
        asm!("mrs {}, cntpct_el0", out(reg) cnt);
    }
    cnt
}

/// Read system timer frequency
pub fn read_timer_freq() -> u64 {
    let freq: u64;
    unsafe {
        asm!("mrs {}, cntfrq_el0", out(reg) freq);
    }
    freq
}

/// RAII guard for disabling interrupts in critical sections
/// Automatically restores interrupt state when dropped
pub struct InterruptGuard {
    was_enabled: bool,
}

impl InterruptGuard {
    /// Create a new interrupt guard, disabling interrupts if they were enabled
    pub fn new() -> Self {
        let was_enabled = interrupts_enabled();
        if was_enabled {
            disable_interrupts();
        }
        Self { was_enabled }
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        if self.was_enabled {
            enable_interrupts();
        }
    }
}

impl Default for InterruptGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Delay for approximately N cycles
pub fn delay_cycles(cycles: u64) {
    let start = read_timer();
    while read_timer() - start < cycles {}
}

/// Delay for approximately N microseconds
pub fn delay_us(us: u64) {
    let freq = read_timer_freq();
    let cycles = (freq * us) / 1_000_000;
    delay_cycles(cycles);
}

/// Reboot the system (Raspberry Pi specific)
pub fn reboot() -> ! {
    // Raspberry Pi watchdog reset
    // PM_RSTC and PM_WDOG registers
    const PM_BASE: usize = 0xFE100000; // BCM2711
    const PM_RSTC: usize = PM_BASE + 0x1C;
    const PM_WDOG: usize = PM_BASE + 0x24;
    const PM_PASSWORD: u32 = 0x5A000000;
    const PM_RSTC_WRCFG_FULL_RESET: u32 = 0x00000020;

    unsafe {
        // Disable interrupts
        asm!("msr daifset, #0xf");

        let rstc = PM_RSTC as *mut u32;
        let wdog = PM_WDOG as *mut u32;

        // Set watchdog timeout to 10 ticks
        core::ptr::write_volatile(wdog, PM_PASSWORD | 10);

        // Trigger full reset
        let mut val = core::ptr::read_volatile(rstc);
        val &= !0x30; // Clear reset config
        val |= PM_PASSWORD | PM_RSTC_WRCFG_FULL_RESET;
        core::ptr::write_volatile(rstc, val);
    }

    // Wait for reset
    loop {
        halt();
    }
}
