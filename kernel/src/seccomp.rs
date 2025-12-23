//! Seccomp - Secure Computing Mode
//!
//! Syscall filtering using BPF (Berkeley Packet Filter) programs.
//! Allows processes to restrict which syscalls they can make.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use spin::RwLock;

use crate::process::Pid;

/// Seccomp mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeccompMode {
    /// No filtering
    Disabled = 0,
    /// Strict mode: only read, write, exit, sigreturn allowed
    Strict = 1,
    /// Filter mode: BPF program decides
    Filter = 2,
}

/// Seccomp action returned by BPF filter
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SeccompAction {
    /// Kill the process (SECCOMP_RET_KILL_PROCESS)
    KillProcess = 0x80000000,
    /// Kill the thread (SECCOMP_RET_KILL_THREAD)
    KillThread = 0x00000000,
    /// Send SIGSYS and allow to continue (SECCOMP_RET_TRAP)
    Trap = 0x00030000,
    /// Return errno (SECCOMP_RET_ERRNO | errno)
    Errno(u16),
    /// Notify userspace supervisor (SECCOMP_RET_USER_NOTIF)
    UserNotify = 0x7fc00000,
    /// Log the syscall (SECCOMP_RET_LOG)
    Log = 0x7ffc0000,
    /// Allow the syscall (SECCOMP_RET_ALLOW)
    Allow = 0x7fff0000,
    /// Trace/ptrace the syscall (SECCOMP_RET_TRACE)
    Trace(u16),
}

impl SeccompAction {
    /// Parse from BPF return value
    pub fn from_ret(ret: u32) -> Self {
        let action = ret & 0xffff0000;
        let data = (ret & 0xffff) as u16;

        match action {
            0x80000000 => SeccompAction::KillProcess,
            0x00000000 => SeccompAction::KillThread,
            0x00030000 => SeccompAction::Trap,
            0x00050000 => SeccompAction::Errno(data),
            0x7fc00000 => SeccompAction::UserNotify,
            0x7ffc0000 => SeccompAction::Log,
            0x7fff0000 => SeccompAction::Allow,
            0x7ff00000 => SeccompAction::Trace(data),
            _ => SeccompAction::KillThread, // Default to kill
        }
    }

    /// Convert to return value
    pub fn to_ret(&self) -> u32 {
        match self {
            SeccompAction::KillProcess => 0x80000000,
            SeccompAction::KillThread => 0x00000000,
            SeccompAction::Trap => 0x00030000,
            SeccompAction::Errno(e) => 0x00050000 | (*e as u32),
            SeccompAction::UserNotify => 0x7fc00000,
            SeccompAction::Log => 0x7ffc0000,
            SeccompAction::Allow => 0x7fff0000,
            SeccompAction::Trace(d) => 0x7ff00000 | (*d as u32),
        }
    }
}

/// BPF instruction
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BpfInsn {
    /// Opcode
    pub code: u16,
    /// Jump if true
    pub jt: u8,
    /// Jump if false
    pub jf: u8,
    /// Constant/offset
    pub k: u32,
}

impl BpfInsn {
    pub const fn new(code: u16, jt: u8, jf: u8, k: u32) -> Self {
        Self { code, jt, jf, k }
    }
}

/// BPF opcodes
pub mod bpf {
    // Instruction classes
    pub const BPF_LD: u16 = 0x00;
    pub const BPF_LDX: u16 = 0x01;
    pub const BPF_ST: u16 = 0x02;
    pub const BPF_STX: u16 = 0x03;
    pub const BPF_ALU: u16 = 0x04;
    pub const BPF_JMP: u16 = 0x05;
    pub const BPF_RET: u16 = 0x06;
    pub const BPF_MISC: u16 = 0x07;

    // Load sizes
    pub const BPF_W: u16 = 0x00; // 32-bit word
    pub const BPF_H: u16 = 0x08; // 16-bit half-word
    pub const BPF_B: u16 = 0x10; // 8-bit byte

    // Load modes
    pub const BPF_IMM: u16 = 0x00;
    pub const BPF_ABS: u16 = 0x20;
    pub const BPF_IND: u16 = 0x40;
    pub const BPF_MEM: u16 = 0x60;
    pub const BPF_LEN: u16 = 0x80;
    pub const BPF_MSH: u16 = 0xa0;

    // ALU operations
    pub const BPF_ADD: u16 = 0x00;
    pub const BPF_SUB: u16 = 0x10;
    pub const BPF_MUL: u16 = 0x20;
    pub const BPF_DIV: u16 = 0x30;
    pub const BPF_OR: u16 = 0x40;
    pub const BPF_AND: u16 = 0x50;
    pub const BPF_LSH: u16 = 0x60;
    pub const BPF_RSH: u16 = 0x70;
    pub const BPF_NEG: u16 = 0x80;
    pub const BPF_MOD: u16 = 0x90;
    pub const BPF_XOR: u16 = 0xa0;

    // Jump conditions
    pub const BPF_JA: u16 = 0x00;
    pub const BPF_JEQ: u16 = 0x10;
    pub const BPF_JGT: u16 = 0x20;
    pub const BPF_JGE: u16 = 0x30;
    pub const BPF_JSET: u16 = 0x40;

    // Sources
    pub const BPF_K: u16 = 0x00; // Constant
    pub const BPF_X: u16 = 0x08; // Index register

    // Return
    pub const BPF_A: u16 = 0x10; // Accumulator
}

/// Syscall data passed to BPF filter
#[repr(C)]
pub struct SeccompData {
    /// Syscall number
    pub nr: i32,
    /// CPU architecture (AUDIT_ARCH_*)
    pub arch: u32,
    /// Instruction pointer at syscall
    pub instruction_pointer: u64,
    /// Syscall arguments
    pub args: [u64; 6],
}

/// Seccomp BPF program
#[derive(Clone)]
pub struct SeccompFilter {
    /// BPF instructions
    instructions: Vec<BpfInsn>,
    /// Previous filter in chain (filters are AND'd)
    prev: Option<Box<SeccompFilter>>,
}

impl SeccompFilter {
    /// Create new filter from instructions
    pub fn new(instructions: Vec<BpfInsn>) -> Result<Self, SeccompError> {
        // Validate the BPF program
        Self::validate(&instructions)?;

        Ok(Self {
            instructions,
            prev: None,
        })
    }

    /// Validate BPF program
    fn validate(insns: &[BpfInsn]) -> Result<(), SeccompError> {
        if insns.is_empty() {
            return Err(SeccompError::InvalidProgram);
        }

        if insns.len() > 4096 {
            return Err(SeccompError::ProgramTooLarge);
        }

        // Check each instruction
        for (i, insn) in insns.iter().enumerate() {
            let class = insn.code & 0x07;

            match class {
                bpf::BPF_LD | bpf::BPF_LDX => {
                    // Load instruction - check mode
                    let mode = insn.code & 0xe0;
                    match mode {
                        bpf::BPF_ABS | bpf::BPF_MEM | bpf::BPF_IMM => {}
                        _ => return Err(SeccompError::InvalidInstruction),
                    }
                }
                bpf::BPF_ST | bpf::BPF_STX => {
                    // Store - check memory index
                    if insn.k >= 16 {
                        return Err(SeccompError::InvalidInstruction);
                    }
                }
                bpf::BPF_ALU => {
                    // ALU ops are generally OK
                    let op = insn.code & 0xf0;
                    if op == bpf::BPF_DIV || op == bpf::BPF_MOD {
                        // Check for division by zero with constant
                        if (insn.code & bpf::BPF_X) == 0 && insn.k == 0 {
                            return Err(SeccompError::InvalidInstruction);
                        }
                    }
                }
                bpf::BPF_JMP => {
                    // Check jump targets are in bounds
                    let target_t = i + 1 + insn.jt as usize;
                    let target_f = i + 1 + insn.jf as usize;
                    if target_t > insns.len() || target_f > insns.len() {
                        return Err(SeccompError::InvalidJump);
                    }
                }
                bpf::BPF_RET => {
                    // Return is always OK
                }
                bpf::BPF_MISC => {
                    // TAX/TXA
                }
                _ => return Err(SeccompError::InvalidInstruction),
            }
        }

        // Last instruction must be RET
        let last = &insns[insns.len() - 1];
        if (last.code & 0x07) != bpf::BPF_RET {
            return Err(SeccompError::NoReturn);
        }

        Ok(())
    }

    /// Execute filter on syscall data
    pub fn run(&self, data: &SeccompData) -> SeccompAction {
        // Run this filter
        let ret = self.execute(data);
        let action = SeccompAction::from_ret(ret);

        // If we have a previous filter, AND the results
        // (most restrictive wins)
        if let Some(ref prev) = self.prev {
            let prev_action = prev.run(data);
            // Return the more restrictive action
            if action.to_ret() < prev_action.to_ret() {
                return action;
            }
            return prev_action;
        }

        action
    }

    /// Execute BPF program
    fn execute(&self, data: &SeccompData) -> u32 {
        let mut a: u32 = 0; // Accumulator
        let mut x: u32 = 0; // Index register
        let mut mem: [u32; 16] = [0; 16]; // Scratch memory
        let mut pc: usize = 0;

        // Convert data to byte array for BPF_ABS loads
        let data_bytes = unsafe {
            core::slice::from_raw_parts(
                data as *const SeccompData as *const u8,
                core::mem::size_of::<SeccompData>(),
            )
        };

        while pc < self.instructions.len() {
            let insn = &self.instructions[pc];
            let class = insn.code & 0x07;
            let size = insn.code & 0x18;
            let mode = insn.code & 0xe0;

            match class {
                bpf::BPF_LD => match mode {
                    bpf::BPF_IMM => {
                        a = insn.k;
                    }
                    bpf::BPF_ABS => {
                        let offset = insn.k as usize;
                        a = match size {
                            bpf::BPF_W => {
                                if offset + 4 <= data_bytes.len() {
                                    u32::from_ne_bytes([
                                        data_bytes[offset],
                                        data_bytes[offset + 1],
                                        data_bytes[offset + 2],
                                        data_bytes[offset + 3],
                                    ])
                                } else {
                                    0
                                }
                            }
                            bpf::BPF_H => {
                                if offset + 2 <= data_bytes.len() {
                                    u16::from_ne_bytes([data_bytes[offset], data_bytes[offset + 1]])
                                        as u32
                                } else {
                                    0
                                }
                            }
                            bpf::BPF_B => {
                                if offset < data_bytes.len() {
                                    data_bytes[offset] as u32
                                } else {
                                    0
                                }
                            }
                            _ => 0,
                        };
                    }
                    bpf::BPF_MEM => {
                        let idx = insn.k as usize;
                        if idx < 16 {
                            a = mem[idx];
                        }
                    }
                    _ => {}
                },
                bpf::BPF_LDX => match mode {
                    bpf::BPF_IMM => {
                        x = insn.k;
                    }
                    bpf::BPF_MEM => {
                        let idx = insn.k as usize;
                        if idx < 16 {
                            x = mem[idx];
                        }
                    }
                    _ => {}
                },
                bpf::BPF_ST => {
                    let idx = insn.k as usize;
                    if idx < 16 {
                        mem[idx] = a;
                    }
                }
                bpf::BPF_STX => {
                    let idx = insn.k as usize;
                    if idx < 16 {
                        mem[idx] = x;
                    }
                }
                bpf::BPF_ALU => {
                    let op = insn.code & 0xf0;
                    let src = if insn.code & bpf::BPF_X != 0 {
                        x
                    } else {
                        insn.k
                    };

                    a = match op {
                        bpf::BPF_ADD => a.wrapping_add(src),
                        bpf::BPF_SUB => a.wrapping_sub(src),
                        bpf::BPF_MUL => a.wrapping_mul(src),
                        bpf::BPF_DIV => {
                            if src != 0 {
                                a / src
                            } else {
                                0
                            }
                        }
                        bpf::BPF_OR => a | src,
                        bpf::BPF_AND => a & src,
                        bpf::BPF_LSH => a << (src & 31),
                        bpf::BPF_RSH => a >> (src & 31),
                        bpf::BPF_NEG => (!a).wrapping_add(1),
                        bpf::BPF_MOD => {
                            if src != 0 {
                                a % src
                            } else {
                                0
                            }
                        }
                        bpf::BPF_XOR => a ^ src,
                        _ => a,
                    };
                }
                bpf::BPF_JMP => {
                    let op = insn.code & 0xf0;
                    let src = if insn.code & bpf::BPF_X != 0 {
                        x
                    } else {
                        insn.k
                    };

                    let cond = match op {
                        bpf::BPF_JA => {
                            pc = pc.wrapping_add(insn.k as usize);
                            continue;
                        }
                        bpf::BPF_JEQ => a == src,
                        bpf::BPF_JGT => a > src,
                        bpf::BPF_JGE => a >= src,
                        bpf::BPF_JSET => a & src != 0,
                        _ => false,
                    };

                    pc += if cond {
                        insn.jt as usize
                    } else {
                        insn.jf as usize
                    };
                }
                bpf::BPF_RET => {
                    return if insn.code & bpf::BPF_A != 0 {
                        a
                    } else {
                        insn.k
                    };
                }
                bpf::BPF_MISC => {
                    let op = insn.code & 0xf8;
                    if op == 0x00 {
                        // TAX
                        x = a;
                    } else if op == 0x80 {
                        // TXA
                        a = x;
                    }
                }
                _ => {}
            }

            pc += 1;
        }

        // Should not reach here, return KILL
        0
    }

    /// Chain this filter after another
    pub fn chain(mut self, prev: SeccompFilter) -> Self {
        self.prev = Some(Box::new(prev));
        self
    }
}

/// Seccomp error
#[derive(Clone, Copy, Debug)]
pub enum SeccompError {
    /// Invalid BPF program
    InvalidProgram,
    /// Program too large
    ProgramTooLarge,
    /// Invalid instruction
    InvalidInstruction,
    /// Invalid jump target
    InvalidJump,
    /// No return instruction
    NoReturn,
    /// Mode already set
    AlreadyActive,
    /// Permission denied
    PermissionDenied,
    /// Filter would be ignored
    FilterWouldBeIgnored,
}

/// Process seccomp state
#[derive(Clone)]
pub struct SeccompState {
    /// Current mode
    pub mode: SeccompMode,
    /// Active filter (if Filter mode)
    pub filter: Option<SeccompFilter>,
    /// Flags
    pub flags: SeccompFlags,
}

impl Default for SeccompState {
    fn default() -> Self {
        Self {
            mode: SeccompMode::Disabled,
            filter: None,
            flags: SeccompFlags::default(),
        }
    }
}

/// Seccomp flags
#[derive(Clone, Copy, Debug, Default)]
pub struct SeccompFlags {
    /// Log allowed syscalls
    pub log: bool,
    /// Synchronize threads when installing filter
    pub tsync: bool,
    /// Wait for filter to be notified
    pub wait_killable: bool,
}

/// Global seccomp state per process
static SECCOMP_STATE: RwLock<alloc::collections::BTreeMap<Pid, SeccompState>> =
    RwLock::new(alloc::collections::BTreeMap::new());

/// Architecture constants
pub mod arch {
    pub const AUDIT_ARCH_AARCH64: u32 = 0xC00000B7;
    pub const AUDIT_ARCH_X86_64: u32 = 0xC000003E;
    pub const AUDIT_ARCH_RISCV64: u32 = 0xC00000F3;
}

/// seccomp operations
pub mod ops {
    pub const SECCOMP_SET_MODE_STRICT: u32 = 0;
    pub const SECCOMP_SET_MODE_FILTER: u32 = 1;
    pub const SECCOMP_GET_ACTION_AVAIL: u32 = 2;
    pub const SECCOMP_GET_NOTIF_SIZES: u32 = 3;
}

/// seccomp flags
pub mod flags {
    pub const SECCOMP_FILTER_FLAG_TSYNC: u32 = 1 << 0;
    pub const SECCOMP_FILTER_FLAG_LOG: u32 = 1 << 1;
    pub const SECCOMP_FILTER_FLAG_SPEC_ALLOW: u32 = 1 << 2;
    pub const SECCOMP_FILTER_FLAG_NEW_LISTENER: u32 = 1 << 3;
    pub const SECCOMP_FILTER_FLAG_TSYNC_ESRCH: u32 = 1 << 4;
    pub const SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV: u32 = 1 << 5;
}

/// Initialize seccomp state for a process
pub fn init_process(pid: Pid) {
    SECCOMP_STATE.write().insert(pid, SeccompState::default());
}

/// Remove seccomp state for exited process
pub fn cleanup_process(pid: Pid) {
    SECCOMP_STATE.write().remove(&pid);
}

/// Get seccomp state
pub fn get_state(pid: Pid) -> Option<SeccompState> {
    SECCOMP_STATE.read().get(&pid).cloned()
}

/// Set strict mode (only read, write, exit, sigreturn)
pub fn set_strict_mode(pid: Pid) -> Result<(), SeccompError> {
    let mut states = SECCOMP_STATE.write();
    let state = states.entry(pid).or_insert(SeccompState::default());

    if state.mode != SeccompMode::Disabled {
        return Err(SeccompError::AlreadyActive);
    }

    state.mode = SeccompMode::Strict;
    Ok(())
}

/// Set filter mode with BPF program
pub fn set_filter_mode(
    pid: Pid,
    filter: SeccompFilter,
    sync_flags: u32,
) -> Result<(), SeccompError> {
    let mut states = SECCOMP_STATE.write();
    let state = states.entry(pid).or_insert(SeccompState::default());

    // Can add filters to existing filter mode, but not switch from strict
    if state.mode == SeccompMode::Strict {
        return Err(SeccompError::AlreadyActive);
    }

    // Chain with existing filter if any
    let new_filter = if let Some(existing) = state.filter.take() {
        filter.chain(existing)
    } else {
        filter
    };

    state.mode = SeccompMode::Filter;
    state.filter = Some(new_filter);

    // Handle flags
    if sync_flags & flags::SECCOMP_FILTER_FLAG_LOG != 0 {
        state.flags.log = true;
    }
    if sync_flags & flags::SECCOMP_FILTER_FLAG_TSYNC != 0 {
        state.flags.tsync = true;
        // TODO: Sync to all threads in process
    }

    Ok(())
}

/// Check syscall against seccomp filter
pub fn check_syscall(pid: Pid, nr: i32, args: &[u64; 6], ip: u64) -> SeccompAction {
    let states = SECCOMP_STATE.read();
    let state = match states.get(&pid) {
        Some(s) => s,
        None => return SeccompAction::Allow,
    };

    match state.mode {
        SeccompMode::Disabled => SeccompAction::Allow,
        SeccompMode::Strict => {
            // Only allow read, write, exit, sigreturn
            match nr {
                0 | 1 | 60 | 231 | 15 => SeccompAction::Allow, // read, write, exit, exit_group, rt_sigreturn
                _ => SeccompAction::KillThread,
            }
        }
        SeccompMode::Filter => {
            if let Some(ref filter) = state.filter {
                let data = SeccompData {
                    nr,
                    arch: arch::AUDIT_ARCH_AARCH64, // TODO: detect arch
                    instruction_pointer: ip,
                    args: *args,
                };
                let action = filter.run(&data);

                // Log if requested
                if state.flags.log {
                    if let SeccompAction::Allow = action {
                        crate::kdebug!("seccomp: allowed syscall {} for pid {}", nr, pid.0);
                    }
                }

                action
            } else {
                SeccompAction::Allow
            }
        }
    }
}

/// Handle seccomp action result
pub fn handle_action(action: SeccompAction) -> Result<(), i32> {
    match action {
        SeccompAction::Allow => Ok(()),
        SeccompAction::Log => {
            // Log and continue
            Ok(())
        }
        SeccompAction::Errno(e) => Err(-(e as i32)),
        SeccompAction::Trap => {
            // Send SIGSYS
            if let Some(proc) = crate::process::current() {
                crate::signal::send_signal(proc.pid, crate::signal::Signal::Sys);
            }
            Err(-1)
        }
        SeccompAction::Trace(_) => {
            // ptrace notification
            Ok(())
        }
        SeccompAction::UserNotify => {
            // User notification (requires listener)
            Err(-1)
        }
        SeccompAction::KillThread => {
            if let Some(proc) = crate::process::current() {
                crate::signal::send_signal(proc.pid, crate::signal::Signal::Kill);
            }
            Err(-1)
        }
        SeccompAction::KillProcess => {
            if let Some(proc) = crate::process::current() {
                crate::signal::send_signal(proc.pid, crate::signal::Signal::Kill);
            }
            Err(-1)
        }
    }
}

/// seccomp syscall implementation
pub fn sys_seccomp(operation: u32, flags: u32, args: *const u8) -> Result<i32, SeccompError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(SeccompError::PermissionDenied)?;

    match operation {
        ops::SECCOMP_SET_MODE_STRICT => {
            set_strict_mode(pid)?;
            Ok(0)
        }
        ops::SECCOMP_SET_MODE_FILTER => {
            if args.is_null() {
                return Err(SeccompError::InvalidProgram);
            }

            // Parse BPF program from args
            // args points to struct sock_fprog { len: u16, filter: *const sock_filter }
            let prog = unsafe { &*(args as *const SockFprog) };

            if prog.len == 0 || prog.filter.is_null() {
                return Err(SeccompError::InvalidProgram);
            }

            let insns: Vec<BpfInsn> = unsafe {
                core::slice::from_raw_parts(prog.filter, prog.len as usize)
                    .iter()
                    .map(|sf| BpfInsn {
                        code: sf.code,
                        jt: sf.jt,
                        jf: sf.jf,
                        k: sf.k,
                    })
                    .collect()
            };

            let filter = SeccompFilter::new(insns)?;
            set_filter_mode(pid, filter, flags)?;
            Ok(0)
        }
        ops::SECCOMP_GET_ACTION_AVAIL => {
            // Check if action is available
            Ok(0)
        }
        ops::SECCOMP_GET_NOTIF_SIZES => {
            // Return notification sizes
            Ok(0)
        }
        _ => Err(SeccompError::InvalidProgram),
    }
}

/// BPF program descriptor (matches Linux struct sock_fprog)
#[repr(C)]
pub struct SockFprog {
    pub len: u16,
    pub filter: *const SockFilter,
}

/// BPF filter instruction (matches Linux struct sock_filter)
#[repr(C)]
pub struct SockFilter {
    pub code: u16,
    pub jt: u8,
    pub jf: u8,
    pub k: u32,
}

/// Helper: create a filter that allows specific syscalls
pub fn allow_syscalls(allowed: &[i32]) -> SeccompFilter {
    let mut insns = Vec::new();

    // Load syscall number
    insns.push(BpfInsn::new(
        bpf::BPF_LD | bpf::BPF_W | bpf::BPF_ABS,
        0,
        0,
        0, // offsetof(SeccompData, nr)
    ));

    // Check each allowed syscall
    for (i, &nr) in allowed.iter().enumerate() {
        let remaining = allowed.len() - i - 1;
        insns.push(BpfInsn::new(
            bpf::BPF_JMP | bpf::BPF_JEQ | bpf::BPF_K,
            (remaining + 1) as u8, // Jump to ALLOW if match
            0,                     // Continue checking
            nr as u32,
        ));
    }

    // Default: kill
    insns.push(BpfInsn::new(
        bpf::BPF_RET | bpf::BPF_K,
        0,
        0,
        SeccompAction::KillThread.to_ret(),
    ));

    // Allow
    insns.push(BpfInsn::new(
        bpf::BPF_RET | bpf::BPF_K,
        0,
        0,
        SeccompAction::Allow.to_ret(),
    ));

    SeccompFilter::new(insns).expect("Invalid filter generated")
}

/// Helper: create a filter that denies specific syscalls
pub fn deny_syscalls(denied: &[i32], default_action: SeccompAction) -> SeccompFilter {
    let mut insns = Vec::new();

    // Load syscall number
    insns.push(BpfInsn::new(
        bpf::BPF_LD | bpf::BPF_W | bpf::BPF_ABS,
        0,
        0,
        0,
    ));

    // Check each denied syscall
    for (i, &nr) in denied.iter().enumerate() {
        let remaining = denied.len() - i - 1;
        insns.push(BpfInsn::new(
            bpf::BPF_JMP | bpf::BPF_JEQ | bpf::BPF_K,
            (remaining + 1) as u8, // Jump to KILL if match
            0,
            nr as u32,
        ));
    }

    // Default: allow (or specified action)
    insns.push(BpfInsn::new(
        bpf::BPF_RET | bpf::BPF_K,
        0,
        0,
        default_action.to_ret(),
    ));

    // Deny action
    insns.push(BpfInsn::new(
        bpf::BPF_RET | bpf::BPF_K,
        0,
        0,
        SeccompAction::KillThread.to_ret(),
    ));

    SeccompFilter::new(insns).expect("Invalid filter generated")
}

/// Fork: inherit seccomp state
pub fn fork_state(parent: Pid, child: Pid) {
    let states = SECCOMP_STATE.read();
    if let Some(parent_state) = states.get(&parent) {
        drop(states);
        SECCOMP_STATE.write().insert(child, parent_state.clone());
    }
}

/// Initialize seccomp subsystem
pub fn init() {
    crate::kprintln!("  Seccomp (syscall filtering) initialized");
}
