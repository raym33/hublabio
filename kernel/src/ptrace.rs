//! Ptrace - Process Tracing
//!
//! Linux-compatible ptrace implementation for debugging and tracing.
//! Supports register access, memory access, single-stepping, and syscall tracing.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use spin::{Mutex, RwLock};

use crate::process::Pid;
use crate::signal::Signal;

/// Ptrace requests
pub mod request {
    pub const PTRACE_TRACEME: i32 = 0;
    pub const PTRACE_PEEKTEXT: i32 = 1;
    pub const PTRACE_PEEKDATA: i32 = 2;
    pub const PTRACE_PEEKUSER: i32 = 3;
    pub const PTRACE_POKETEXT: i32 = 4;
    pub const PTRACE_POKEDATA: i32 = 5;
    pub const PTRACE_POKEUSER: i32 = 6;
    pub const PTRACE_CONT: i32 = 7;
    pub const PTRACE_KILL: i32 = 8;
    pub const PTRACE_SINGLESTEP: i32 = 9;
    pub const PTRACE_GETREGS: i32 = 12;
    pub const PTRACE_SETREGS: i32 = 13;
    pub const PTRACE_GETFPREGS: i32 = 14;
    pub const PTRACE_SETFPREGS: i32 = 15;
    pub const PTRACE_ATTACH: i32 = 16;
    pub const PTRACE_DETACH: i32 = 17;
    pub const PTRACE_GETFPXREGS: i32 = 18;
    pub const PTRACE_SETFPXREGS: i32 = 19;
    pub const PTRACE_SYSCALL: i32 = 24;
    pub const PTRACE_SETOPTIONS: i32 = 0x4200;
    pub const PTRACE_GETEVENTMSG: i32 = 0x4201;
    pub const PTRACE_GETSIGINFO: i32 = 0x4202;
    pub const PTRACE_SETSIGINFO: i32 = 0x4203;
    pub const PTRACE_GETREGSET: i32 = 0x4204;
    pub const PTRACE_SETREGSET: i32 = 0x4205;
    pub const PTRACE_SEIZE: i32 = 0x4206;
    pub const PTRACE_INTERRUPT: i32 = 0x4207;
    pub const PTRACE_LISTEN: i32 = 0x4208;
    pub const PTRACE_PEEKSIGINFO: i32 = 0x4209;
    pub const PTRACE_GETSIGMASK: i32 = 0x420a;
    pub const PTRACE_SETSIGMASK: i32 = 0x420b;
    pub const PTRACE_SECCOMP_GET_FILTER: i32 = 0x420c;
    pub const PTRACE_SECCOMP_GET_METADATA: i32 = 0x420d;
    pub const PTRACE_GET_SYSCALL_INFO: i32 = 0x420e;
}

/// Ptrace options
pub mod options {
    pub const PTRACE_O_TRACESYSGOOD: u32 = 1 << 0;
    pub const PTRACE_O_TRACEFORK: u32 = 1 << 1;
    pub const PTRACE_O_TRACEVFORK: u32 = 1 << 2;
    pub const PTRACE_O_TRACECLONE: u32 = 1 << 3;
    pub const PTRACE_O_TRACEEXEC: u32 = 1 << 4;
    pub const PTRACE_O_TRACEVFORKDONE: u32 = 1 << 5;
    pub const PTRACE_O_TRACEEXIT: u32 = 1 << 6;
    pub const PTRACE_O_TRACESECCOMP: u32 = 1 << 7;
    pub const PTRACE_O_EXITKILL: u32 = 1 << 20;
    pub const PTRACE_O_SUSPEND_SECCOMP: u32 = 1 << 21;
}

/// Ptrace events
pub mod events {
    pub const PTRACE_EVENT_FORK: u32 = 1;
    pub const PTRACE_EVENT_VFORK: u32 = 2;
    pub const PTRACE_EVENT_CLONE: u32 = 3;
    pub const PTRACE_EVENT_EXEC: u32 = 4;
    pub const PTRACE_EVENT_VFORK_DONE: u32 = 5;
    pub const PTRACE_EVENT_EXIT: u32 = 6;
    pub const PTRACE_EVENT_SECCOMP: u32 = 7;
    pub const PTRACE_EVENT_STOP: u32 = 128;
}

/// Register set types
pub mod regset {
    pub const NT_PRSTATUS: u32 = 1;
    pub const NT_PRFPREG: u32 = 2;
    pub const NT_PRPSINFO: u32 = 3;
    pub const NT_TASKSTRUCT: u32 = 4;
    pub const NT_AUXV: u32 = 6;
    pub const NT_ARM_VFP: u32 = 0x400;
    pub const NT_ARM_TLS: u32 = 0x401;
    pub const NT_ARM_HW_BREAK: u32 = 0x402;
    pub const NT_ARM_HW_WATCH: u32 = 0x403;
    pub const NT_ARM_SYSTEM_CALL: u32 = 0x404;
}

/// General purpose registers (AArch64)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UserRegs {
    pub regs: [u64; 31],  // x0-x30
    pub sp: u64,
    pub pc: u64,
    pub pstate: u64,
}

/// Floating point registers (AArch64)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UserFpRegs {
    pub vregs: [u128; 32],
    pub fpsr: u32,
    pub fpcr: u32,
}

/// Ptrace state for a tracee
pub struct PtraceState {
    /// Tracer process
    pub tracer: Pid,
    /// Options set by tracer
    pub options: AtomicU32,
    /// Currently stopped
    pub stopped: AtomicBool,
    /// Stop signal
    pub stop_signal: Mutex<Option<Signal>>,
    /// Event message (for GETEVENTMSG)
    pub event_msg: AtomicU32,
    /// Last syscall number
    pub syscall_nr: AtomicU32,
    /// Syscall entry (vs exit)
    pub syscall_entry: AtomicBool,
    /// Single step mode
    pub single_step: AtomicBool,
    /// Trace syscalls
    pub trace_syscalls: AtomicBool,
    /// Saved registers
    pub saved_regs: Mutex<UserRegs>,
    /// Saved FP registers
    pub saved_fpregs: Mutex<UserFpRegs>,
}

impl PtraceState {
    pub fn new(tracer: Pid) -> Self {
        Self {
            tracer,
            options: AtomicU32::new(0),
            stopped: AtomicBool::new(false),
            stop_signal: Mutex::new(None),
            event_msg: AtomicU32::new(0),
            syscall_nr: AtomicU32::new(0),
            syscall_entry: AtomicBool::new(true),
            single_step: AtomicBool::new(false),
            trace_syscalls: AtomicBool::new(false),
            saved_regs: Mutex::new(UserRegs::default()),
            saved_fpregs: Mutex::new(UserFpRegs::default()),
        }
    }
}

/// Tracee map: pid -> ptrace state
static TRACEES: RwLock<BTreeMap<Pid, PtraceState>> = RwLock::new(BTreeMap::new());

/// Children being traced by each tracer
static TRACER_CHILDREN: RwLock<BTreeMap<Pid, Vec<Pid>>> = RwLock::new(BTreeMap::new());

/// Ptrace error
#[derive(Clone, Copy, Debug)]
pub enum PtraceError {
    /// No such process
    NoProcess,
    /// Permission denied
    Permission,
    /// Invalid argument
    Invalid,
    /// Not attached
    NotAttached,
    /// Already attached
    AlreadyAttached,
    /// Bad address
    Fault,
    /// Process not stopped
    NotStopped,
    /// Busy (process running)
    Busy,
}

impl PtraceError {
    pub fn to_errno(&self) -> i32 {
        match self {
            PtraceError::NoProcess => -3,      // ESRCH
            PtraceError::Permission => -1,      // EPERM
            PtraceError::Invalid => -22,        // EINVAL
            PtraceError::NotAttached => -3,     // ESRCH
            PtraceError::AlreadyAttached => -16,// EBUSY
            PtraceError::Fault => -14,          // EFAULT
            PtraceError::NotStopped => -3,      // ESRCH
            PtraceError::Busy => -16,           // EBUSY
        }
    }
}

/// Request tracing by parent
pub fn traceme() -> Result<(), PtraceError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let ppid = crate::process::get(pid)
        .and_then(|p| p.ppid)
        .ok_or(PtraceError::NoProcess)?;

    // Check if already traced
    if TRACEES.read().contains_key(&pid) {
        return Err(PtraceError::AlreadyAttached);
    }

    // Set up tracing relationship
    let state = PtraceState::new(ppid);
    TRACEES.write().insert(pid, state);

    // Add to tracer's children list
    TRACER_CHILDREN.write()
        .entry(ppid)
        .or_insert_with(Vec::new)
        .push(pid);

    Ok(())
}

/// Attach to a process
pub fn attach(pid: Pid) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    // Check target exists
    if crate::process::get(pid).is_none() {
        return Err(PtraceError::NoProcess);
    }

    // Check permissions
    if !can_trace(tracer, pid) {
        return Err(PtraceError::Permission);
    }

    // Check if already traced
    if TRACEES.read().contains_key(&pid) {
        return Err(PtraceError::AlreadyAttached);
    }

    // Set up tracing
    let state = PtraceState::new(tracer);
    TRACEES.write().insert(pid, state);

    TRACER_CHILDREN.write()
        .entry(tracer)
        .or_insert_with(Vec::new)
        .push(pid);

    // Send SIGSTOP to target
    crate::signal::send_signal(pid, Signal::SIGSTOP);

    Ok(())
}

/// Seize a process (like attach but doesn't stop immediately)
pub fn seize(pid: Pid, options: u32) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    if crate::process::get(pid).is_none() {
        return Err(PtraceError::NoProcess);
    }

    if !can_trace(tracer, pid) {
        return Err(PtraceError::Permission);
    }

    if TRACEES.read().contains_key(&pid) {
        return Err(PtraceError::AlreadyAttached);
    }

    let state = PtraceState::new(tracer);
    state.options.store(options, Ordering::SeqCst);
    TRACEES.write().insert(pid, state);

    TRACER_CHILDREN.write()
        .entry(tracer)
        .or_insert_with(Vec::new)
        .push(pid);

    Ok(())
}

/// Detach from a process
pub fn detach(pid: Pid, sig: Option<Signal>) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    // Verify we're the tracer
    {
        let tracees = TRACEES.read();
        let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;
        if state.tracer != tracer {
            return Err(PtraceError::Permission);
        }
    }

    // Remove tracing
    TRACEES.write().remove(&pid);

    // Remove from tracer's children list
    if let Some(children) = TRACER_CHILDREN.write().get_mut(&tracer) {
        children.retain(|&p| p != pid);
    }

    // Continue the process
    if let Some(signal) = sig {
        crate::signal::send_signal(pid, signal);
    }

    // Wake process if stopped
    crate::scheduler::wake(pid);

    Ok(())
}

/// Continue execution
pub fn cont(pid: Pid, sig: Option<Signal>) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    if !state.stopped.load(Ordering::SeqCst) {
        return Err(PtraceError::NotStopped);
    }

    // Clear stop state
    state.stopped.store(false, Ordering::SeqCst);
    state.single_step.store(false, Ordering::SeqCst);
    state.trace_syscalls.store(false, Ordering::SeqCst);

    // Deliver signal if specified
    if let Some(signal) = sig {
        crate::signal::send_signal(pid, signal);
    }

    // Wake process
    crate::scheduler::wake(pid);

    Ok(())
}

/// Single step execution
pub fn singlestep(pid: Pid, sig: Option<Signal>) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    // Enable single step mode
    state.stopped.store(false, Ordering::SeqCst);
    state.single_step.store(true, Ordering::SeqCst);

    // Deliver signal if specified
    if let Some(signal) = sig {
        crate::signal::send_signal(pid, signal);
    }

    // Wake process (will stop after one instruction)
    crate::scheduler::wake(pid);

    Ok(())
}

/// Trace syscalls
pub fn syscall(pid: Pid, sig: Option<Signal>) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    // Enable syscall tracing
    state.stopped.store(false, Ordering::SeqCst);
    state.trace_syscalls.store(true, Ordering::SeqCst);

    if let Some(signal) = sig {
        crate::signal::send_signal(pid, signal);
    }

    crate::scheduler::wake(pid);

    Ok(())
}

/// Read memory from tracee
pub fn peek_data(pid: Pid, addr: usize) -> Result<usize, PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    // Read from tracee's address space
    // Would need to map tracee's memory and read
    let data = unsafe {
        (addr as *const usize).read_volatile()
    };

    Ok(data)
}

/// Write memory to tracee
pub fn poke_data(pid: Pid, addr: usize, data: usize) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    // Write to tracee's address space
    unsafe {
        (addr as *mut usize).write_volatile(data);
    }

    Ok(())
}

/// Get tracee registers
pub fn getregs(pid: Pid) -> Result<UserRegs, PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    if !state.stopped.load(Ordering::SeqCst) {
        return Err(PtraceError::NotStopped);
    }

    Ok(*state.saved_regs.lock())
}

/// Set tracee registers
pub fn setregs(pid: Pid, regs: &UserRegs) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    if !state.stopped.load(Ordering::SeqCst) {
        return Err(PtraceError::NotStopped);
    }

    *state.saved_regs.lock() = *regs;

    Ok(())
}

/// Get floating point registers
pub fn getfpregs(pid: Pid) -> Result<UserFpRegs, PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    if !state.stopped.load(Ordering::SeqCst) {
        return Err(PtraceError::NotStopped);
    }

    Ok(*state.saved_fpregs.lock())
}

/// Set floating point registers
pub fn setfpregs(pid: Pid, fpregs: &UserFpRegs) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    if !state.stopped.load(Ordering::SeqCst) {
        return Err(PtraceError::NotStopped);
    }

    *state.saved_fpregs.lock() = *fpregs;

    Ok(())
}

/// Set ptrace options
pub fn setoptions(pid: Pid, options: u32) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    state.options.store(options, Ordering::SeqCst);

    Ok(())
}

/// Get event message
pub fn geteventmsg(pid: Pid) -> Result<u32, PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    Ok(state.event_msg.load(Ordering::SeqCst))
}

/// Kill tracee
pub fn kill_tracee(pid: Pid) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    {
        let tracees = TRACEES.read();
        let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

        if state.tracer != tracer {
            return Err(PtraceError::Permission);
        }
    }

    // Send SIGKILL
    crate::signal::send_signal(pid, Signal::SIGKILL);

    // Detach
    detach(pid, None)?;

    Ok(())
}

/// Interrupt a running tracee (for SEIZE)
pub fn interrupt(pid: Pid) -> Result<(), PtraceError> {
    let tracer = crate::process::current()
        .map(|p| p.pid)
        .ok_or(PtraceError::NoProcess)?;

    let tracees = TRACEES.read();
    let state = tracees.get(&pid).ok_or(PtraceError::NotAttached)?;

    if state.tracer != tracer {
        return Err(PtraceError::Permission);
    }

    // Send SIGSTOP
    crate::signal::send_signal(pid, Signal::SIGSTOP);

    Ok(())
}

// ============================================================================
// Internal Hooks
// ============================================================================

/// Check if tracer can trace target
fn can_trace(tracer: Pid, target: Pid) -> bool {
    // Check CAP_SYS_PTRACE
    if crate::capability::has_capability(tracer, crate::capability::Capability::SysPtrace) {
        return true;
    }

    // Check if tracer is parent
    if let Some(target_proc) = crate::process::get(target) {
        if target_proc.ppid == Some(tracer) {
            return true;
        }
    }

    // Check same UID
    if let (Some(tracer_proc), Some(target_proc)) =
        (crate::process::get(tracer), crate::process::get(target))
    {
        if tracer_proc.uid == target_proc.uid {
            return true;
        }
    }

    false
}

/// Called when a traced process receives a signal
pub fn on_signal(pid: Pid, signal: Signal) -> bool {
    let tracees = TRACEES.read();
    if let Some(state) = tracees.get(&pid) {
        // Stop tracee and notify tracer
        state.stopped.store(true, Ordering::SeqCst);
        *state.stop_signal.lock() = Some(signal);

        // Wake tracer (for wait())
        crate::scheduler::wake(state.tracer);

        return true; // Signal handled by ptrace
    }

    false // Not traced
}

/// Called at syscall entry/exit
pub fn on_syscall(pid: Pid, syscall_nr: u32, entry: bool) {
    let tracees = TRACEES.read();
    if let Some(state) = tracees.get(&pid) {
        if state.trace_syscalls.load(Ordering::SeqCst) {
            state.syscall_nr.store(syscall_nr, Ordering::SeqCst);
            state.syscall_entry.store(entry, Ordering::SeqCst);
            state.stopped.store(true, Ordering::SeqCst);

            // Create trap signal
            let signal = if state.options.load(Ordering::SeqCst) & options::PTRACE_O_TRACESYSGOOD != 0 {
                Signal::SIGTRAP // Would set 0x80 bit
            } else {
                Signal::SIGTRAP
            };

            *state.stop_signal.lock() = Some(signal);
            crate::scheduler::wake(state.tracer);
        }
    }
}

/// Called after single step
pub fn on_singlestep(pid: Pid) {
    let tracees = TRACEES.read();
    if let Some(state) = tracees.get(&pid) {
        if state.single_step.load(Ordering::SeqCst) {
            state.stopped.store(true, Ordering::SeqCst);
            *state.stop_signal.lock() = Some(Signal::SIGTRAP);
            crate::scheduler::wake(state.tracer);
        }
    }
}

/// Called on fork/clone/exec
pub fn on_event(pid: Pid, event: u32, msg: u32) {
    let tracees = TRACEES.read();
    if let Some(state) = tracees.get(&pid) {
        let opts = state.options.load(Ordering::SeqCst);

        let should_stop = match event {
            events::PTRACE_EVENT_FORK => opts & options::PTRACE_O_TRACEFORK != 0,
            events::PTRACE_EVENT_VFORK => opts & options::PTRACE_O_TRACEVFORK != 0,
            events::PTRACE_EVENT_CLONE => opts & options::PTRACE_O_TRACECLONE != 0,
            events::PTRACE_EVENT_EXEC => opts & options::PTRACE_O_TRACEEXEC != 0,
            events::PTRACE_EVENT_VFORK_DONE => opts & options::PTRACE_O_TRACEVFORKDONE != 0,
            events::PTRACE_EVENT_EXIT => opts & options::PTRACE_O_TRACEEXIT != 0,
            events::PTRACE_EVENT_SECCOMP => opts & options::PTRACE_O_TRACESECCOMP != 0,
            _ => false,
        };

        if should_stop {
            state.stopped.store(true, Ordering::SeqCst);
            state.event_msg.store(msg, Ordering::SeqCst);
            *state.stop_signal.lock() = Some(Signal::SIGTRAP);
            crate::scheduler::wake(state.tracer);
        }
    }
}

/// Called when tracee exits
pub fn on_exit(pid: Pid) {
    let tracer = {
        let tracees = TRACEES.read();
        tracees.get(&pid).map(|s| s.tracer)
    };

    if let Some(tracer) = tracer {
        // Remove from tracking
        TRACEES.write().remove(&pid);

        if let Some(children) = TRACER_CHILDREN.write().get_mut(&tracer) {
            children.retain(|&p| p != pid);
        }
    }
}

/// Called when tracer exits
pub fn on_tracer_exit(tracer: Pid) {
    let children: Vec<Pid> = TRACER_CHILDREN.write()
        .remove(&tracer)
        .unwrap_or_default();

    for child in children {
        // Option: kill or detach children
        TRACEES.write().remove(&child);

        // Option: EXITKILL
        // crate::signal::send_signal(child, Signal::SIGKILL);

        // Continue children
        crate::scheduler::wake(child);
    }
}

/// Check if process is traced and stopped
pub fn is_stopped(pid: Pid) -> bool {
    TRACEES.read()
        .get(&pid)
        .map(|s| s.stopped.load(Ordering::SeqCst))
        .unwrap_or(false)
}

/// Check if process is traced
pub fn is_traced(pid: Pid) -> bool {
    TRACEES.read().contains_key(&pid)
}

/// Get tracer of a process
pub fn get_tracer(pid: Pid) -> Option<Pid> {
    TRACEES.read().get(&pid).map(|s| s.tracer)
}

// ============================================================================
// Syscall Interface
// ============================================================================

/// Main ptrace syscall
pub fn sys_ptrace(request: i32, pid: i32, addr: usize, data: usize) -> isize {
    let target_pid = Pid(pid as u32);

    let result = match request {
        request::PTRACE_TRACEME => traceme(),
        request::PTRACE_ATTACH => attach(target_pid),
        request::PTRACE_SEIZE => seize(target_pid, data as u32),
        request::PTRACE_DETACH => {
            let sig = if data != 0 { Signal::from_num(data as i32) } else { None };
            detach(target_pid, sig)
        }
        request::PTRACE_CONT => {
            let sig = if data != 0 { Signal::from_num(data as i32) } else { None };
            cont(target_pid, sig)
        }
        request::PTRACE_SINGLESTEP => {
            let sig = if data != 0 { Signal::from_num(data as i32) } else { None };
            singlestep(target_pid, sig)
        }
        request::PTRACE_SYSCALL => {
            let sig = if data != 0 { Signal::from_num(data as i32) } else { None };
            syscall(target_pid, sig)
        }
        request::PTRACE_KILL => kill_tracee(target_pid),
        request::PTRACE_INTERRUPT => interrupt(target_pid),
        request::PTRACE_PEEKDATA | request::PTRACE_PEEKTEXT => {
            return match peek_data(target_pid, addr) {
                Ok(v) => v as isize,
                Err(e) => e.to_errno() as isize,
            };
        }
        request::PTRACE_POKEDATA | request::PTRACE_POKETEXT => {
            poke_data(target_pid, addr, data)
        }
        request::PTRACE_GETREGS => {
            match getregs(target_pid) {
                Ok(regs) => {
                    unsafe {
                        *(data as *mut UserRegs) = regs;
                    }
                    return 0;
                }
                Err(e) => return e.to_errno() as isize,
            }
        }
        request::PTRACE_SETREGS => {
            let regs = unsafe { &*(data as *const UserRegs) };
            setregs(target_pid, regs)
        }
        request::PTRACE_GETFPREGS => {
            match getfpregs(target_pid) {
                Ok(fpregs) => {
                    unsafe {
                        *(data as *mut UserFpRegs) = fpregs;
                    }
                    return 0;
                }
                Err(e) => return e.to_errno() as isize,
            }
        }
        request::PTRACE_SETFPREGS => {
            let fpregs = unsafe { &*(data as *const UserFpRegs) };
            setfpregs(target_pid, fpregs)
        }
        request::PTRACE_SETOPTIONS => {
            setoptions(target_pid, data as u32)
        }
        request::PTRACE_GETEVENTMSG => {
            match geteventmsg(target_pid) {
                Ok(msg) => {
                    unsafe { *(data as *mut u32) = msg; }
                    return 0;
                }
                Err(e) => return e.to_errno() as isize,
            }
        }
        _ => Err(PtraceError::Invalid),
    };

    match result {
        Ok(()) => 0,
        Err(e) => e.to_errno() as isize,
    }
}

/// Initialize ptrace subsystem
pub fn init() {
    crate::kprintln!("  Ptrace debugging support initialized");
}
