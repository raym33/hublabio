//! SMP (Symmetric Multi-Processing) Support
//!
//! Multi-core CPU management for ARM64 systems.
//! Handles CPU detection, secondary core boot, per-CPU data,
//! and inter-processor interrupts (IPI).

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use spin::{Mutex, RwLock};

/// Maximum number of CPUs supported
pub const MAX_CPUS: usize = 8;

/// IPI vector numbers
pub const IPI_RESCHEDULE: u32 = 0;
pub const IPI_CALL_FUNC: u32 = 1;
pub const IPI_TLB_SHOOTDOWN: u32 = 2;
pub const IPI_HALT: u32 = 3;

/// GIC SGI (Software Generated Interrupt) range: 0-15
const SGI_BASE: u32 = 0;

/// CPU state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CpuState {
    /// CPU is offline/not started
    Offline = 0,
    /// CPU is booting
    Booting = 1,
    /// CPU is online and running
    Online = 2,
    /// CPU is in idle loop
    Idle = 3,
    /// CPU is halted
    Halted = 4,
}

impl From<u8> for CpuState {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Offline,
            1 => Self::Booting,
            2 => Self::Online,
            3 => Self::Idle,
            4 => Self::Halted,
            _ => Self::Offline,
        }
    }
}

/// Per-CPU data structure
#[repr(C, align(64))] // Cache-line aligned to prevent false sharing
pub struct PerCpu {
    /// CPU ID (0-based)
    pub id: u32,
    /// MPIDR register value
    pub mpidr: u64,
    /// CPU state
    pub state: AtomicU32,
    /// Current running task ID
    pub current_task: AtomicU64,
    /// Idle task for this CPU
    pub idle_task: AtomicU64,
    /// Number of timer ticks
    pub ticks: AtomicU64,
    /// Number of context switches
    pub context_switches: AtomicU64,
    /// Number of interrupts handled
    pub interrupts: AtomicU64,
    /// CPU is in interrupt handler
    pub in_interrupt: AtomicBool,
    /// Preemption disabled depth
    pub preempt_count: AtomicU32,
    /// Need reschedule flag
    pub need_resched: AtomicBool,
    /// IPI pending flags
    pub ipi_pending: AtomicU32,
    /// Kernel stack for this CPU
    pub kernel_stack: usize,
    /// Padding for cache alignment
    _padding: [u64; 2],
}

impl PerCpu {
    const fn new() -> Self {
        Self {
            id: 0,
            mpidr: 0,
            state: AtomicU32::new(CpuState::Offline as u32),
            current_task: AtomicU64::new(0),
            idle_task: AtomicU64::new(0),
            ticks: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            interrupts: AtomicU64::new(0),
            in_interrupt: AtomicBool::new(false),
            preempt_count: AtomicU32::new(0),
            need_resched: AtomicBool::new(false),
            ipi_pending: AtomicU32::new(0),
            kernel_stack: 0,
            _padding: [0; 2],
        }
    }

    /// Get CPU state
    pub fn get_state(&self) -> CpuState {
        CpuState::from(self.state.load(Ordering::Acquire) as u8)
    }

    /// Set CPU state
    pub fn set_state(&self, state: CpuState) {
        self.state.store(state as u32, Ordering::Release);
    }

    /// Increment tick count
    pub fn tick(&self) {
        self.ticks.fetch_add(1, Ordering::Relaxed);
    }

    /// Disable preemption
    pub fn preempt_disable(&self) {
        self.preempt_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Enable preemption
    pub fn preempt_enable(&self) {
        let old = self.preempt_count.fetch_sub(1, Ordering::AcqRel);
        if old == 1 && self.need_resched.load(Ordering::Acquire) {
            // Preemption just became enabled and reschedule is needed
            crate::scheduler::schedule();
        }
    }

    /// Check if preemption is enabled
    pub fn is_preemptible(&self) -> bool {
        self.preempt_count.load(Ordering::Acquire) == 0
            && !self.in_interrupt.load(Ordering::Acquire)
    }
}

/// Per-CPU data for all CPUs
static PER_CPU: [PerCpu; MAX_CPUS] = {
    const INIT: PerCpu = PerCpu::new();
    [INIT; MAX_CPUS]
};

/// Number of online CPUs
static ONLINE_CPUS: AtomicUsize = AtomicUsize::new(0);

/// BSP (Bootstrap Processor) has finished initialization
static BSP_READY: AtomicBool = AtomicBool::new(false);

/// Secondary CPUs can start
static SECONDARIES_GO: AtomicBool = AtomicBool::new(false);

/// Spin table addresses for secondary CPUs (RPi4)
static SPIN_TABLE: [AtomicU64; MAX_CPUS] = {
    const ZERO: AtomicU64 = AtomicU64::new(0);
    [ZERO; MAX_CPUS]
};

/// IPI call function type
pub type IpiFunction = fn(arg: usize);

/// Pending IPI function calls
static IPI_FUNC: Mutex<Option<(IpiFunction, usize)>> = Mutex::new(None);

/// TLB shootdown address
static TLB_SHOOTDOWN_ADDR: AtomicU64 = AtomicU64::new(0);

/// TLB shootdown completion count
static TLB_SHOOTDOWN_DONE: AtomicUsize = AtomicUsize::new(0);

/// Get current CPU ID
#[inline]
pub fn cpu_id() -> u32 {
    let mpidr: u64;
    unsafe {
        asm!("mrs {}, mpidr_el1", out(reg) mpidr);
    }
    // Extract Aff0 (CPU ID within cluster)
    // For most ARM64 systems, Aff0 is the CPU ID
    (mpidr & 0xFF) as u32
}

/// Get current CPU's per-CPU data
#[inline]
pub fn current() -> &'static PerCpu {
    let id = cpu_id() as usize;
    if id < MAX_CPUS {
        &PER_CPU[id]
    } else {
        &PER_CPU[0]
    }
}

/// Get per-CPU data for a specific CPU
#[inline]
pub fn get(cpu: u32) -> Option<&'static PerCpu> {
    if (cpu as usize) < MAX_CPUS {
        Some(&PER_CPU[cpu as usize])
    } else {
        None
    }
}

/// Get number of online CPUs
#[inline]
pub fn online_count() -> usize {
    ONLINE_CPUS.load(Ordering::Acquire)
}

/// Check if a CPU is online
#[inline]
pub fn is_online(cpu: u32) -> bool {
    if let Some(pcpu) = get(cpu) {
        pcpu.get_state() == CpuState::Online || pcpu.get_state() == CpuState::Idle
    } else {
        false
    }
}

/// Initialize SMP on the bootstrap processor (BSP)
pub fn init() {
    let cpu = cpu_id();
    let mpidr: u64;
    unsafe {
        asm!("mrs {}, mpidr_el1", out(reg) mpidr);
    }

    // Initialize BSP's per-CPU data
    let pcpu = &PER_CPU[cpu as usize];
    unsafe {
        // Use raw pointer to modify const fields during init
        let pcpu_mut = pcpu as *const PerCpu as *mut PerCpu;
        (*pcpu_mut).id = cpu;
        (*pcpu_mut).mpidr = mpidr;
    }
    pcpu.set_state(CpuState::Online);
    ONLINE_CPUS.fetch_add(1, Ordering::AcqRel);

    // Detect number of CPUs from DTB or hardware
    let num_cpus = detect_cpu_count();

    crate::kprintln!("  SMP initialized on CPU {} (MPIDR: 0x{:x})", cpu, mpidr);
    crate::kprintln!("  Detected {} CPUs", num_cpus);

    // Register IPI handlers
    register_ipi_handlers();

    BSP_READY.store(true, Ordering::Release);
}

/// Detect number of CPUs in the system
fn detect_cpu_count() -> usize {
    // Try to get from DTB first
    // For now, check if we're on RPi4 (Cortex-A72 has 4 cores)
    let midr: u64;
    unsafe {
        asm!("mrs {}, midr_el1", out(reg) midr);
    }

    let part = (midr >> 4) & 0xFFF;

    match part {
        0xD03 => 4, // Cortex-A53 (RPi3)
        0xD08 => 4, // Cortex-A72 (RPi4)
        0xD07 => 4, // Cortex-A57 (QEMU default)
        0xD05 => 4, // Cortex-A55
        _ => {
            // QEMU reports different parts, try to detect from MPIDR
            // For QEMU virt machine, check if CPU 1+ responds
            1 // Default to 1 if unknown
        }
    }
}

/// Register IPI interrupt handlers
fn register_ipi_handlers() {
    use crate::arch::interrupt;

    // Register handlers for SGI 0-3 (our IPI vectors)
    for sgi in 0..4 {
        interrupt::register_handler(sgi, |irq| {
            handle_ipi(irq);
            true
        });
    }
}

/// Start secondary CPUs
pub fn start_secondaries() {
    if !BSP_READY.load(Ordering::Acquire) {
        crate::kwarn!("BSP not ready, cannot start secondary CPUs");
        return;
    }

    let num_cpus = detect_cpu_count();
    let bsp_id = cpu_id();

    crate::kprintln!("[SMP] Starting {} secondary CPUs...", num_cpus - 1);

    for cpu in 0..num_cpus as u32 {
        if cpu == bsp_id {
            continue; // Skip BSP
        }

        start_cpu(cpu);
    }

    // Wait for all CPUs to come online
    let mut timeout = 1_000_000;
    while ONLINE_CPUS.load(Ordering::Acquire) < num_cpus && timeout > 0 {
        core::hint::spin_loop();
        timeout -= 1;
    }

    let online = ONLINE_CPUS.load(Ordering::Acquire);
    if online == num_cpus {
        crate::kprintln!("[SMP] All {} CPUs online", online);
    } else {
        crate::kwarn!("[SMP] Only {} of {} CPUs came online", online, num_cpus);
    }
}

/// Start a specific secondary CPU
fn start_cpu(cpu: u32) {
    if cpu as usize >= MAX_CPUS {
        return;
    }

    // Allocate kernel stack for this CPU
    let stack = allocate_cpu_stack();
    if stack == 0 {
        crate::kerror!("Failed to allocate stack for CPU {}", cpu);
        return;
    }

    // Initialize per-CPU data
    let pcpu = &PER_CPU[cpu as usize];
    unsafe {
        let pcpu_mut = pcpu as *const PerCpu as *mut PerCpu;
        (*pcpu_mut).id = cpu;
        (*pcpu_mut).kernel_stack = stack;
    }
    pcpu.set_state(CpuState::Booting);

    // Method 1: PSCI (Power State Coordination Interface)
    // Used by most modern ARM64 systems
    if psci_cpu_on(cpu, secondary_entry as usize, stack) {
        crate::kinfo!("CPU {} started via PSCI", cpu);
        return;
    }

    // Method 2: Spin table (used by older systems like RPi3)
    // Write entry point to spin table location
    let spin_addr = get_spin_table_addr(cpu);
    if spin_addr != 0 {
        unsafe {
            let entry = secondary_entry as usize;
            core::ptr::write_volatile(spin_addr as *mut u64, entry as u64);
            // Send event to wake up CPU
            asm!("sev");
        }
        crate::kinfo!("CPU {} started via spin table", cpu);
    }
}

/// Get spin table address for a CPU (RPi specific)
fn get_spin_table_addr(cpu: u32) -> usize {
    // RPi4 spin table addresses
    const SPIN_BASE: usize = 0xD8;
    const SPIN_STRIDE: usize = 8;

    SPIN_BASE + (cpu as usize) * SPIN_STRIDE
}

/// PSCI CPU_ON function
fn psci_cpu_on(cpu: u32, entry: usize, context: usize) -> bool {
    // PSCI function IDs
    const PSCI_CPU_ON_64: u32 = 0xC4000003;

    // Build MPIDR for target CPU
    let target_mpidr = cpu as u64;

    let result: i64;
    unsafe {
        asm!(
            "mov x0, {func}",
            "mov x1, {mpidr}",
            "mov x2, {entry}",
            "mov x3, {context}",
            "smc #0",
            "mov {result}, x0",
            func = in(reg) PSCI_CPU_ON_64 as u64,
            mpidr = in(reg) target_mpidr,
            entry = in(reg) entry as u64,
            context = in(reg) context as u64,
            result = out(reg) result,
            out("x0") _,
            out("x1") _,
            out("x2") _,
            out("x3") _,
        );
    }

    result == 0 // PSCI_SUCCESS
}

/// Secondary CPU entry point (called from assembly)
#[no_mangle]
pub extern "C" fn secondary_entry(cpu_id: u64, stack_top: u64) -> ! {
    // We're now running on a secondary CPU
    let cpu = cpu_id as u32;

    // Set up stack pointer
    unsafe {
        asm!(
            "mov sp, {}",
            in(reg) stack_top,
        );
    }

    // Initialize CPU-local features
    secondary_init(cpu);

    // Enter idle loop
    secondary_idle(cpu);
}

/// Initialize secondary CPU
fn secondary_init(cpu: u32) {
    // Enable FPU/SIMD
    unsafe {
        asm!(
            "mrs x0, cpacr_el1",
            "orr x0, x0, #(3 << 20)",
            "msr cpacr_el1, x0",
            "isb",
            out("x0") _,
        );
    }

    // Set up exception vectors
    crate::arch::exception::init();

    // Initialize GIC CPU interface
    init_gic_cpu_interface();

    // Initialize timer
    init_cpu_timer();

    // Get MPIDR
    let mpidr: u64;
    unsafe {
        asm!("mrs {}, mpidr_el1", out(reg) mpidr);
    }

    // Update per-CPU data
    let pcpu = &PER_CPU[cpu as usize];
    unsafe {
        let pcpu_mut = pcpu as *const PerCpu as *mut PerCpu;
        (*pcpu_mut).mpidr = mpidr;
    }
    pcpu.set_state(CpuState::Online);

    // Increment online count
    ONLINE_CPUS.fetch_add(1, Ordering::AcqRel);

    crate::kinfo!("CPU {} online (MPIDR: 0x{:x})", cpu, mpidr);

    // Enable interrupts
    crate::arch::enable_interrupts();
}

/// Initialize GIC CPU interface for secondary CPU
fn init_gic_cpu_interface() {
    use crate::arch::interrupt::gic;

    // The GIC distributor is already initialized by BSP
    // We just need to enable the CPU interface

    let gicc = unsafe {
        core::ptr::read_volatile(&crate::arch::interrupt::gic::GICC_BASE as *const _ as *const u64)
    } as usize;

    if gicc == 0 {
        return;
    }

    unsafe {
        // Set priority mask
        core::ptr::write_volatile((gicc + gic::GICC_PMR) as *mut u32, 0xFF);
        // Enable CPU interface
        core::ptr::write_volatile((gicc + gic::GICC_CTLR) as *mut u32, 1);
    }
}

/// Initialize timer for secondary CPU
fn init_cpu_timer() {
    // Get timer frequency and set up periodic tick
    let freq: u64;
    unsafe {
        asm!("mrs {}, cntfrq_el0", out(reg) freq);
    }

    // 10ms tick interval
    let interval = freq / 100;

    unsafe {
        // Set timer compare value
        asm!(
            "msr cntp_tval_el0, {}",
            in(reg) interval,
        );

        // Enable timer
        asm!(
            "mov x0, #1",
            "msr cntp_ctl_el0, x0",
            out("x0") _,
        );
    }
}

/// Secondary CPU idle loop
fn secondary_idle(cpu: u32) -> ! {
    let pcpu = &PER_CPU[cpu as usize];

    loop {
        // Check if we should run scheduler
        if SECONDARIES_GO.load(Ordering::Acquire) {
            pcpu.set_state(CpuState::Idle);

            // Try to get work from scheduler
            if let Some(_task) = crate::scheduler::pick_next_for_cpu(cpu) {
                pcpu.set_state(CpuState::Online);
                // Run the task
                crate::scheduler::run_task(cpu);
            }
        }

        // Process any pending IPIs
        process_pending_ipis(cpu);

        // Wait for interrupt
        pcpu.set_state(CpuState::Idle);
        unsafe {
            asm!("wfi");
        }
    }
}

/// Allocate a kernel stack for a CPU
fn allocate_cpu_stack() -> usize {
    const CPU_STACK_SIZE: usize = 16 * 1024; // 16KB per CPU

    // Allocate pages for stack
    let pages = (CPU_STACK_SIZE + crate::memory::PAGE_SIZE - 1) / crate::memory::PAGE_SIZE;

    let mut stack_base = 0usize;
    for i in 0..pages {
        if let Some(frame) = crate::memory::allocate_frame() {
            if i == 0 {
                stack_base = frame.start_address();
            }
        } else {
            return 0;
        }
    }

    // Return stack top (stack grows down)
    stack_base + CPU_STACK_SIZE
}

/// Send IPI to a specific CPU
pub fn send_ipi(target_cpu: u32, ipi: u32) {
    if target_cpu as usize >= MAX_CPUS || !is_online(target_cpu) {
        return;
    }

    // Mark IPI as pending
    PER_CPU[target_cpu as usize]
        .ipi_pending
        .fetch_or(1 << ipi, Ordering::Release);

    // Send SGI via GIC
    send_sgi(target_cpu, SGI_BASE + ipi);
}

/// Send IPI to all CPUs except self
pub fn send_ipi_all_others(ipi: u32) {
    let self_cpu = cpu_id();

    for cpu in 0..MAX_CPUS as u32 {
        if cpu != self_cpu && is_online(cpu) {
            send_ipi(cpu, ipi);
        }
    }
}

/// Send IPI to all online CPUs including self
pub fn send_ipi_all(ipi: u32) {
    for cpu in 0..MAX_CPUS as u32 {
        if is_online(cpu) {
            send_ipi(cpu, ipi);
        }
    }
}

/// Send SGI (Software Generated Interrupt) via GIC
fn send_sgi(target_cpu: u32, sgi: u32) {
    // GIC SGI register offset
    const GICD_SGIR: usize = 0xF00;

    let gicd = unsafe {
        core::ptr::read_volatile(&crate::arch::interrupt::gic::GICD_BASE as *const _ as *const u64)
    } as usize;

    if gicd == 0 {
        return;
    }

    // SGI register format:
    // [25:24] = target list filter (0 = use target list)
    // [23:16] = CPU target list (1 << cpu)
    // [3:0] = SGI interrupt ID
    let value = ((1u32 << target_cpu) << 16) | (sgi & 0xF);

    unsafe {
        core::ptr::write_volatile((gicd + GICD_SGIR) as *mut u32, value);
    }
}

/// Handle incoming IPI
fn handle_ipi(sgi: u32) {
    let ipi = sgi - SGI_BASE;

    match ipi {
        IPI_RESCHEDULE => {
            // Request reschedule
            current().need_resched.store(true, Ordering::Release);
        }
        IPI_CALL_FUNC => {
            // Execute function call
            if let Some((func, arg)) = IPI_FUNC.lock().take() {
                func(arg);
            }
        }
        IPI_TLB_SHOOTDOWN => {
            // Invalidate TLB for address
            let addr = TLB_SHOOTDOWN_ADDR.load(Ordering::Acquire);
            if addr != 0 {
                unsafe {
                    asm!(
                        "tlbi vaae1is, {}",
                        "dsb ish",
                        "isb",
                        in(reg) addr >> 12,
                    );
                }
            } else {
                // Full TLB invalidation
                crate::arch::tlb_invalidate_all();
            }
            TLB_SHOOTDOWN_DONE.fetch_add(1, Ordering::Release);
        }
        IPI_HALT => {
            // Halt this CPU
            current().set_state(CpuState::Halted);
            crate::arch::disable_interrupts();
            loop {
                unsafe {
                    asm!("wfi");
                }
            }
        }
        _ => {}
    }

    // Clear pending flag
    current()
        .ipi_pending
        .fetch_and(!(1 << ipi), Ordering::Release);
}

/// Process any pending IPIs for this CPU
fn process_pending_ipis(cpu: u32) {
    let pcpu = &PER_CPU[cpu as usize];
    let pending = pcpu.ipi_pending.load(Ordering::Acquire);

    for ipi in 0..4 {
        if pending & (1 << ipi) != 0 {
            handle_ipi(SGI_BASE + ipi);
        }
    }
}

/// Send a function call to another CPU
pub fn smp_call_function(target_cpu: u32, func: IpiFunction, arg: usize) {
    if target_cpu == cpu_id() {
        // Call locally
        func(arg);
        return;
    }

    // Store function and argument
    *IPI_FUNC.lock() = Some((func, arg));

    // Send IPI
    send_ipi(target_cpu, IPI_CALL_FUNC);
}

/// Send a function call to all other CPUs
pub fn smp_call_function_all_others(func: IpiFunction, arg: usize) {
    *IPI_FUNC.lock() = Some((func, arg));
    send_ipi_all_others(IPI_CALL_FUNC);
}

/// Request TLB shootdown on all CPUs
pub fn tlb_shootdown(addr: Option<usize>) {
    let online = ONLINE_CPUS.load(Ordering::Acquire);
    if online <= 1 {
        // Single CPU, just invalidate locally
        if let Some(a) = addr {
            unsafe {
                asm!(
                    "tlbi vaae1is, {}",
                    "dsb ish",
                    "isb",
                    in(reg) a >> 12,
                );
            }
        } else {
            crate::arch::tlb_invalidate_all();
        }
        return;
    }

    // Store address for other CPUs
    TLB_SHOOTDOWN_ADDR.store(addr.unwrap_or(0) as u64, Ordering::Release);
    TLB_SHOOTDOWN_DONE.store(0, Ordering::Release);

    // Send IPI to all other CPUs
    send_ipi_all_others(IPI_TLB_SHOOTDOWN);

    // Wait for all CPUs to complete
    let target = online - 1;
    while TLB_SHOOTDOWN_DONE.load(Ordering::Acquire) < target {
        core::hint::spin_loop();
    }

    // Do local invalidation
    if let Some(a) = addr {
        unsafe {
            asm!(
                "tlbi vaae1is, {}",
                "dsb ish",
                "isb",
                in(reg) a >> 12,
            );
        }
    } else {
        crate::arch::tlb_invalidate_all();
    }
}

/// Request reschedule on a specific CPU
pub fn reschedule_cpu(cpu: u32) {
    if cpu == cpu_id() {
        current().need_resched.store(true, Ordering::Release);
    } else {
        send_ipi(cpu, IPI_RESCHEDULE);
    }
}

/// Halt all CPUs (for panic)
pub fn halt_all_cpus() {
    // Disable interrupts on this CPU
    crate::arch::disable_interrupts();

    // Send halt IPI to all others
    send_ipi_all_others(IPI_HALT);

    // Halt this CPU
    current().set_state(CpuState::Halted);
    loop {
        unsafe {
            asm!("wfi");
        }
    }
}

/// Get SMP statistics
pub fn stats() -> SmpStats {
    let mut stats = SmpStats::default();

    for cpu in 0..MAX_CPUS {
        let pcpu = &PER_CPU[cpu];
        if pcpu.get_state() != CpuState::Offline {
            stats.online_cpus += 1;
            stats.total_ticks += pcpu.ticks.load(Ordering::Relaxed);
            stats.total_context_switches += pcpu.context_switches.load(Ordering::Relaxed);
            stats.total_interrupts += pcpu.interrupts.load(Ordering::Relaxed);
        }
    }

    stats
}

/// SMP statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct SmpStats {
    pub online_cpus: usize,
    pub total_ticks: u64,
    pub total_context_switches: u64,
    pub total_interrupts: u64,
}

/// Enable secondary CPUs to participate in scheduling
pub fn enable_scheduling() {
    SECONDARIES_GO.store(true, Ordering::Release);

    // Wake up all secondary CPUs
    send_ipi_all_others(IPI_RESCHEDULE);
}

/// Preemption guard - disables preemption while held
pub struct PreemptGuard;

impl PreemptGuard {
    pub fn new() -> Self {
        current().preempt_disable();
        Self
    }
}

impl Drop for PreemptGuard {
    fn drop(&mut self) {
        current().preempt_enable();
    }
}

/// Check if current CPU needs to reschedule
pub fn need_resched() -> bool {
    current().need_resched.swap(false, Ordering::AcqRel)
}

/// Called from timer interrupt on each CPU
pub fn timer_tick() {
    let pcpu = current();
    pcpu.tick();

    // Check if current task's time slice is exhausted
    // The scheduler will handle the actual time slice checking
    pcpu.need_resched.store(true, Ordering::Release);
}
