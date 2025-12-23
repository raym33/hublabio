//! Exception handling for ARM64
//!
//! Sets up the exception vector table and handles interrupts, syscalls, etc.

use core::arch::asm;

/// Exception types
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum ExceptionClass {
    Unknown = 0x00,
    WfeWfi = 0x01,
    Svc64 = 0x15,      // Syscall from AArch64
    Svc32 = 0x11,      // Syscall from AArch32
    InstrAbortLower = 0x20,
    InstrAbortSame = 0x21,
    DataAbortLower = 0x24,
    DataAbortSame = 0x25,
    SpAlignment = 0x26,
    Fpexc = 0x2c,
    Serror = 0x2f,
    Breakpoint = 0x30,
    SoftwareStep = 0x32,
    Watchpoint = 0x34,
    BkptLower = 0x38,
    BrkInst = 0x3c,
}

impl From<u32> for ExceptionClass {
    fn from(ec: u32) -> Self {
        match ec {
            0x00 => Self::Unknown,
            0x01 => Self::WfeWfi,
            0x15 => Self::Svc64,
            0x11 => Self::Svc32,
            0x20 => Self::InstrAbortLower,
            0x21 => Self::InstrAbortSame,
            0x24 => Self::DataAbortLower,
            0x25 => Self::DataAbortSame,
            0x26 => Self::SpAlignment,
            0x2c => Self::Fpexc,
            0x2f => Self::Serror,
            0x30 => Self::Breakpoint,
            0x32 => Self::SoftwareStep,
            0x34 => Self::Watchpoint,
            0x38 => Self::BkptLower,
            0x3c => Self::BrkInst,
            _ => Self::Unknown,
        }
    }
}

/// Saved register state during exception
#[repr(C)]
#[derive(Debug)]
pub struct ExceptionFrame {
    pub x: [u64; 31],   // x0-x30
    pub sp: u64,        // Stack pointer
    pub elr: u64,       // Exception Link Register (return address)
    pub spsr: u64,      // Saved Program Status Register
    pub esr: u64,       // Exception Syndrome Register
    pub far: u64,       // Fault Address Register
}

/// Initialize exception handling
pub fn init() {
    extern "C" {
        static __exception_vectors: u64;
    }

    unsafe {
        // Set VBAR_EL1 to our exception vector table
        let vbar = &__exception_vectors as *const u64 as u64;
        asm!(
            "msr vbar_el1, {}",
            "isb",
            in(reg) vbar,
        );
    }

    crate::kprintln!("  Exception vectors installed");
}

/// Main exception handler called from assembly
#[no_mangle]
pub extern "C" fn exception_handler(frame: &mut ExceptionFrame, exception_type: u32) {
    let esr = frame.esr;
    let ec = ExceptionClass::from((esr >> 26) as u32);

    match (exception_type, ec) {
        // Synchronous exception from EL0 - likely a syscall
        (0, ExceptionClass::Svc64) => {
            handle_syscall(frame);
        }

        // Synchronous exception from EL0 - page faults, etc.
        (0, _) => {
            handle_sync_exception(frame, ec);
        }

        // Synchronous exception from EL1
        (1, _) => {
            handle_sync_exception(frame, ec);
        }

        // IRQ from EL0
        (2, _) => {
            handle_irq(frame);
        }

        // IRQ from EL1
        (3, _) => {
            handle_irq(frame);
        }

        // FIQ
        (4, _) | (5, _) => {
            handle_fiq(frame);
        }

        // SError
        (6, _) | (7, _) => {
            handle_serror(frame);
        }

        _ => {
            crate::kprintln!("Unknown exception: type={}, ec={:?}", exception_type, ec);
            crate::kprintln!("  ELR: 0x{:016x}", frame.elr);
            crate::kprintln!("  ESR: 0x{:016x}", frame.esr);
            crate::kprintln!("  FAR: 0x{:016x}", frame.far);
            loop {
                super::halt();
            }
        }
    }
}

/// Handle a syscall (SVC instruction)
fn handle_syscall(frame: &mut ExceptionFrame) {
    // Syscall number is in x8
    let nr = frame.x[8] as usize;

    // Arguments are in x0-x5
    let result = crate::syscall::syscall_handler(
        nr,
        frame.x[0] as usize,
        frame.x[1] as usize,
        frame.x[2] as usize,
        frame.x[3] as usize,
        frame.x[4] as usize,
        frame.x[5] as usize,
    );

    // Return value goes in x0
    frame.x[0] = result as u64;
}

/// Handle synchronous exceptions (not syscalls)
fn handle_sync_exception(frame: &mut ExceptionFrame, ec: ExceptionClass) {
    match ec {
        ExceptionClass::InstrAbortSame | ExceptionClass::InstrAbortLower => {
            // Instruction fault - handle via page fault handler
            let user_mode = matches!(ec, ExceptionClass::InstrAbortLower);
            handle_page_fault(frame, user_mode, crate::pagefault::FaultType::Execute);
        }

        ExceptionClass::DataAbortSame | ExceptionClass::DataAbortLower => {
            // Data fault - handle via page fault handler
            let user_mode = matches!(ec, ExceptionClass::DataAbortLower);
            let iss = frame.esr & 0x1FFFFFF;
            let is_write = (iss >> 6) & 1 == 1;
            let fault_type = if is_write {
                crate::pagefault::FaultType::Write
            } else {
                crate::pagefault::FaultType::Read
            };
            handle_page_fault(frame, user_mode, fault_type);
        }

        ExceptionClass::SpAlignment => {
            crate::kprintln!("SP Alignment Fault!");
            crate::kprintln!("  SP:  0x{:016x}", frame.sp);
            if is_user_mode(frame) {
                // Send SIGBUS to user process
                send_fault_signal(crate::pagefault::FaultSignal::Bus);
            } else {
                panic!("SP alignment fault in kernel mode");
            }
        }

        ExceptionClass::Breakpoint | ExceptionClass::BrkInst => {
            crate::kprintln!("Breakpoint at 0x{:016x}", frame.elr);
            // Handle via ptrace if being traced
            if let Some(task) = crate::task::current() {
                if crate::ptrace::is_traced(task.process.pid.0 as u32) {
                    // Notify tracer
                    let _ = crate::ptrace::report_event(
                        task.process.pid.0 as u32,
                        crate::ptrace::PtraceEvent::Trap,
                    );
                    // Block until tracer continues
                    task.block(crate::task::WaitReason::Signal);
                    crate::scheduler::schedule();
                    return;
                }
            }
            // Not being traced - send SIGTRAP
            send_fault_signal(crate::pagefault::FaultSignal::Segv);
        }

        ExceptionClass::SoftwareStep => {
            // Single-step trap from ptrace
            if let Some(task) = crate::task::current() {
                if crate::ptrace::is_traced(task.process.pid.0 as u32) {
                    let _ = crate::ptrace::report_event(
                        task.process.pid.0 as u32,
                        crate::ptrace::PtraceEvent::SingleStep,
                    );
                    task.block(crate::task::WaitReason::Signal);
                    crate::scheduler::schedule();
                    return;
                }
            }
        }

        ExceptionClass::Watchpoint => {
            // Hardware watchpoint hit
            if let Some(task) = crate::task::current() {
                if crate::ptrace::is_traced(task.process.pid.0 as u32) {
                    let _ = crate::ptrace::report_event(
                        task.process.pid.0 as u32,
                        crate::ptrace::PtraceEvent::Trap,
                    );
                    task.block(crate::task::WaitReason::Signal);
                    crate::scheduler::schedule();
                    return;
                }
            }
        }

        _ => {
            crate::kprintln!("Unhandled synchronous exception: {:?}", ec);
            crate::kprintln!("  ELR: 0x{:016x}", frame.elr);
            crate::kprintln!("  ESR: 0x{:016x}", frame.esr);
            if is_user_mode(frame) {
                send_fault_signal(crate::pagefault::FaultSignal::Segv);
            } else {
                panic!("Unhandled exception in kernel");
            }
        }
    }
}

/// Handle page fault via the pagefault module
fn handle_page_fault(
    frame: &mut ExceptionFrame,
    user_mode: bool,
    fault_type: crate::pagefault::FaultType,
) {
    // Get current process ID
    let pid = if let Some(task) = crate::task::current() {
        task.process.pid
    } else {
        crate::process::Pid(0)
    };

    // Parse fault context
    let ctx = crate::pagefault::parse_fault_info(
        frame.esr,
        frame.far,
        frame.elr,
        user_mode,
        pid,
    );

    // Let the page fault handler deal with it
    let result = crate::pagefault::handle_page_fault(&ctx);

    match result {
        crate::pagefault::FaultResult::Handled => {
            // Successfully handled - return and retry the instruction
        }
        crate::pagefault::FaultResult::Retry => {
            // Need to retry after some action (OOM kill, etc.)
            // Return and retry
        }
        crate::pagefault::FaultResult::Signal(sig) => {
            if user_mode {
                send_fault_signal(sig);
            } else {
                // Kernel fault - this is bad
                crate::kerror!("Page fault in kernel mode!");
                crate::kerror!("  FAR: 0x{:016x}", frame.far);
                crate::kerror!("  ELR: 0x{:016x}", frame.elr);
                crate::kerror!("  ESR: 0x{:016x}", frame.esr);
                panic!("Unhandled kernel page fault");
            }
        }
        crate::pagefault::FaultResult::KernelBug => {
            crate::kerror!("Kernel bug triggered by page fault!");
            crate::kerror!("  FAR: 0x{:016x}", frame.far);
            crate::kerror!("  ELR: 0x{:016x}", frame.elr);
            panic!("Kernel bug");
        }
    }
}

/// Check if fault was in user mode
fn is_user_mode(frame: &ExceptionFrame) -> bool {
    // Check SPSR.M[3:0] - if 0, it was EL0 (user mode)
    (frame.spsr & 0xF) == 0
}

/// Send signal for fault
fn send_fault_signal(sig: crate::pagefault::FaultSignal) {
    if let Some(task) = crate::task::current() {
        let signal = match sig {
            crate::pagefault::FaultSignal::Segv => crate::signal::Signal::SIGSEGV,
            crate::pagefault::FaultSignal::Bus => crate::signal::Signal::SIGBUS,
            crate::pagefault::FaultSignal::Kill => crate::signal::Signal::SIGKILL,
        };

        // Send signal to process
        let _ = crate::signal::send_signal(
            task.process.pid.0 as u32,
            signal,
            0, // kernel is sender
        );

        // If SIGKILL, process will be killed immediately
        // Otherwise, deliver signal before returning to user
        if matches!(sig, crate::pagefault::FaultSignal::Kill) {
            crate::scheduler::schedule();
        }
    }
}

/// Handle IRQ (timer, device interrupts)
fn handle_irq(_frame: &ExceptionFrame) {
    // Read interrupt controller to determine source
    // TODO: Implement GIC handling

    // For now, just clear any pending timer interrupt
    unsafe {
        // Disable timer
        asm!(
            "msr cntp_ctl_el0, xzr",
        );
    }

    // TODO: Call appropriate interrupt handler
    // TODO: Trigger scheduler if timer interrupt
}

/// Handle FIQ (fast interrupt)
fn handle_fiq(_frame: &ExceptionFrame) {
    // FIQ is typically used for secure interrupts
    crate::kprintln!("FIQ received");
}

/// Handle SError (system error)
fn handle_serror(frame: &ExceptionFrame) {
    crate::kprintln!("SError (System Error)!");
    crate::kprintln!("  ELR: 0x{:016x}", frame.elr);
    crate::kprintln!("  ESR: 0x{:016x}", frame.esr);
    panic!("System error");
}

/// Enable IRQ
pub fn enable_irq() {
    unsafe {
        asm!("msr daifclr, #2");  // Clear I bit
    }
}

/// Disable IRQ
pub fn disable_irq() {
    unsafe {
        asm!("msr daifset, #2");  // Set I bit
    }
}

/// Enable timer interrupt
pub fn enable_timer(interval_us: u64) {
    let freq = super::read_timer_freq();
    let ticks = (freq * interval_us) / 1_000_000;

    unsafe {
        asm!(
            // Set timer compare value
            "msr cntp_tval_el0, {ticks}",
            // Enable timer, unmask interrupt
            "mov x0, #1",
            "msr cntp_ctl_el0, x0",
            ticks = in(reg) ticks,
            out("x0") _,
        );
    }
}

/// Disable timer interrupt
pub fn disable_timer() {
    unsafe {
        asm!("msr cntp_ctl_el0, xzr");
    }
}
