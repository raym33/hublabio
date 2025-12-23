//! Interactive Debugger
//!
//! GDB-like debugger for HubLab IO processes.
//! Supports breakpoints, stepping, memory inspection, and register viewing.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;

/// Breakpoint types
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BreakpointType {
    /// Software breakpoint (instruction replacement)
    Software,
    /// Hardware breakpoint (debug registers)
    Hardware,
    /// Watchpoint (memory access)
    Watchpoint { read: bool, write: bool },
}

/// Breakpoint information
#[derive(Clone, Debug)]
pub struct Breakpoint {
    /// Breakpoint ID
    pub id: u32,
    /// Address
    pub address: u64,
    /// Type
    pub bp_type: BreakpointType,
    /// Enabled
    pub enabled: bool,
    /// Hit count
    pub hit_count: u64,
    /// Condition (expression to evaluate)
    pub condition: Option<String>,
    /// Original instruction (for software breakpoints)
    pub original_instr: u32,
}

/// Register set for ARM64
#[derive(Clone, Debug, Default)]
pub struct RegisterSet {
    /// General purpose registers (x0-x30)
    pub x: [u64; 31],
    /// Stack pointer
    pub sp: u64,
    /// Program counter
    pub pc: u64,
    /// Current program status register
    pub cpsr: u64,
    /// Floating point registers (v0-v31)
    pub v: [u128; 32],
    /// Floating point status register
    pub fpsr: u32,
    /// Floating point control register
    pub fpcr: u32,
}

impl RegisterSet {
    /// Get register by name
    pub fn get(&self, name: &str) -> Option<u64> {
        match name.to_lowercase().as_str() {
            "sp" => Some(self.sp),
            "pc" => Some(self.pc),
            "cpsr" | "pstate" => Some(self.cpsr),
            "lr" => Some(self.x[30]),
            "fp" => Some(self.x[29]),
            _ if name.starts_with('x') || name.starts_with('X') => {
                let idx: usize = name[1..].parse().ok()?;
                if idx < 31 { Some(self.x[idx]) } else { None }
            }
            _ if name.starts_with('w') || name.starts_with('W') => {
                let idx: usize = name[1..].parse().ok()?;
                if idx < 31 { Some(self.x[idx] & 0xFFFFFFFF) } else { None }
            }
            _ => None,
        }
    }

    /// Set register by name
    pub fn set(&mut self, name: &str, value: u64) -> bool {
        match name.to_lowercase().as_str() {
            "sp" => { self.sp = value; true }
            "pc" => { self.pc = value; true }
            "cpsr" | "pstate" => { self.cpsr = value; true }
            _ if name.starts_with('x') || name.starts_with('X') => {
                if let Ok(idx) = name[1..].parse::<usize>() {
                    if idx < 31 {
                        self.x[idx] = value;
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Format registers for display
    pub fn format(&self) -> String {
        let mut output = String::new();

        // General purpose registers
        for i in 0..31 {
            if i % 4 == 0 {
                output.push_str("\n");
            }
            output.push_str(&format!("x{:02}={:016x}  ", i, self.x[i]));
        }

        output.push_str(&format!("\nsp ={:016x}  pc ={:016x}  cpsr={:016x}\n",
            self.sp, self.pc, self.cpsr));

        output
    }
}

/// Debugger state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebuggerState {
    /// Not attached to any process
    Detached,
    /// Attached and running
    Running,
    /// Stopped at breakpoint or signal
    Stopped,
    /// Process has exited
    Exited,
}

/// Stop reason
#[derive(Clone, Debug)]
pub enum StopReason {
    /// Hit breakpoint
    Breakpoint(u32),
    /// Single step completed
    Step,
    /// Received signal
    Signal(u32),
    /// Watchpoint triggered
    Watchpoint { address: u64, access: &'static str },
    /// Exception occurred
    Exception(u64),
    /// Process exited
    Exit(i32),
}

/// Memory region info
#[derive(Clone, Debug)]
pub struct MemoryRegion {
    /// Start address
    pub start: u64,
    /// End address
    pub end: u64,
    /// Readable
    pub read: bool,
    /// Writable
    pub write: bool,
    /// Executable
    pub execute: bool,
    /// Name (e.g., "[stack]", "/lib/libc.so")
    pub name: String,
}

/// Stack frame information
#[derive(Clone, Debug)]
pub struct StackFrame {
    /// Frame index (0 = current)
    pub index: u32,
    /// Program counter
    pub pc: u64,
    /// Stack pointer
    pub sp: u64,
    /// Frame pointer
    pub fp: u64,
    /// Function name (if known)
    pub function: Option<String>,
    /// Source file (if known)
    pub file: Option<String>,
    /// Line number (if known)
    pub line: Option<u32>,
}

/// Symbol information
#[derive(Clone, Debug)]
pub struct Symbol {
    /// Symbol name
    pub name: String,
    /// Symbol address
    pub address: u64,
    /// Symbol size
    pub size: u64,
    /// Symbol type (function, object, etc.)
    pub sym_type: SymbolType,
}

/// Symbol types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolType {
    Function,
    Object,
    Section,
    File,
    Unknown,
}

/// Debugger for HubLab IO processes
pub struct Debugger {
    /// Current state
    state: DebuggerState,
    /// Target process ID
    target_pid: Option<u32>,
    /// Breakpoints
    breakpoints: BTreeMap<u32, Breakpoint>,
    /// Next breakpoint ID
    next_bp_id: u32,
    /// Register set
    registers: RegisterSet,
    /// Memory regions
    memory_regions: Vec<MemoryRegion>,
    /// Symbol table
    symbols: Vec<Symbol>,
    /// Backtrace frames
    backtrace: Vec<StackFrame>,
    /// Last stop reason
    stop_reason: Option<StopReason>,
    /// Command history
    history: Vec<String>,
}

impl Debugger {
    /// Create new debugger instance
    pub fn new() -> Self {
        Self {
            state: DebuggerState::Detached,
            target_pid: None,
            breakpoints: BTreeMap::new(),
            next_bp_id: 1,
            registers: RegisterSet::default(),
            memory_regions: Vec::new(),
            symbols: Vec::new(),
            backtrace: Vec::new(),
            stop_reason: None,
            history: Vec::new(),
        }
    }

    /// Attach to a running process
    pub fn attach(&mut self, pid: u32) -> Result<(), DebugError> {
        if self.state != DebuggerState::Detached {
            return Err(DebugError::AlreadyAttached);
        }

        // TODO: Actually attach using ptrace-like mechanism
        self.target_pid = Some(pid);
        self.state = DebuggerState::Stopped;
        self.stop_reason = Some(StopReason::Signal(19)); // SIGSTOP

        Ok(())
    }

    /// Detach from process
    pub fn detach(&mut self) -> Result<(), DebugError> {
        if self.state == DebuggerState::Detached {
            return Err(DebugError::NotAttached);
        }

        // Remove all breakpoints
        for bp in self.breakpoints.values_mut() {
            bp.enabled = false;
        }

        self.target_pid = None;
        self.state = DebuggerState::Detached;

        Ok(())
    }

    /// Continue execution
    pub fn continue_execution(&mut self) -> Result<(), DebugError> {
        self.check_stopped()?;

        self.state = DebuggerState::Running;
        // TODO: Actually resume process

        Ok(())
    }

    /// Single step one instruction
    pub fn step(&mut self) -> Result<(), DebugError> {
        self.check_stopped()?;

        // TODO: Set single-step flag and resume
        self.stop_reason = Some(StopReason::Step);

        Ok(())
    }

    /// Step over function call
    pub fn step_over(&mut self) -> Result<(), DebugError> {
        self.check_stopped()?;

        // Check if current instruction is a call
        // If so, set temporary breakpoint at return address
        let pc = self.registers.pc;

        // TODO: Decode instruction and check if it's a call
        // For now, just single step
        self.step()
    }

    /// Step out of current function
    pub fn step_out(&mut self) -> Result<(), DebugError> {
        self.check_stopped()?;

        // Set breakpoint at return address
        let lr = self.registers.x[30];
        self.set_breakpoint(lr)?;
        self.continue_execution()
    }

    /// Set a breakpoint at address
    pub fn set_breakpoint(&mut self, address: u64) -> Result<u32, DebugError> {
        self.set_breakpoint_with_type(address, BreakpointType::Software)
    }

    /// Set a breakpoint with specific type
    pub fn set_breakpoint_with_type(
        &mut self,
        address: u64,
        bp_type: BreakpointType,
    ) -> Result<u32, DebugError> {
        if self.state == DebuggerState::Detached {
            return Err(DebugError::NotAttached);
        }

        let id = self.next_bp_id;
        self.next_bp_id += 1;

        let bp = Breakpoint {
            id,
            address,
            bp_type,
            enabled: true,
            hit_count: 0,
            condition: None,
            original_instr: 0, // TODO: Read original instruction
        };

        self.breakpoints.insert(id, bp);

        Ok(id)
    }

    /// Set a watchpoint
    pub fn set_watchpoint(&mut self, address: u64, read: bool, write: bool) -> Result<u32, DebugError> {
        self.set_breakpoint_with_type(address, BreakpointType::Watchpoint { read, write })
    }

    /// Delete a breakpoint
    pub fn delete_breakpoint(&mut self, id: u32) -> Result<(), DebugError> {
        if self.breakpoints.remove(&id).is_none() {
            return Err(DebugError::BreakpointNotFound);
        }
        Ok(())
    }

    /// Enable a breakpoint
    pub fn enable_breakpoint(&mut self, id: u32) -> Result<(), DebugError> {
        let bp = self.breakpoints.get_mut(&id).ok_or(DebugError::BreakpointNotFound)?;
        bp.enabled = true;
        Ok(())
    }

    /// Disable a breakpoint
    pub fn disable_breakpoint(&mut self, id: u32) -> Result<(), DebugError> {
        let bp = self.breakpoints.get_mut(&id).ok_or(DebugError::BreakpointNotFound)?;
        bp.enabled = false;
        Ok(())
    }

    /// List all breakpoints
    pub fn list_breakpoints(&self) -> &BTreeMap<u32, Breakpoint> {
        &self.breakpoints
    }

    /// Read memory
    pub fn read_memory(&self, address: u64, size: usize) -> Result<Vec<u8>, DebugError> {
        self.check_stopped()?;

        // TODO: Actually read from process memory
        Ok(vec![0u8; size])
    }

    /// Write memory
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> Result<(), DebugError> {
        self.check_stopped()?;

        // TODO: Actually write to process memory
        Ok(())
    }

    /// Get register value
    pub fn get_register(&self, name: &str) -> Result<u64, DebugError> {
        self.check_stopped()?;
        self.registers.get(name).ok_or(DebugError::InvalidRegister)
    }

    /// Set register value
    pub fn set_register(&mut self, name: &str, value: u64) -> Result<(), DebugError> {
        self.check_stopped()?;
        if self.registers.set(name, value) {
            Ok(())
        } else {
            Err(DebugError::InvalidRegister)
        }
    }

    /// Get all registers
    pub fn get_registers(&self) -> Result<&RegisterSet, DebugError> {
        self.check_stopped()?;
        Ok(&self.registers)
    }

    /// Get backtrace
    pub fn backtrace(&mut self) -> Result<&[StackFrame], DebugError> {
        self.check_stopped()?;

        // Build backtrace by walking frame pointers
        self.backtrace.clear();

        let mut fp = self.registers.x[29];
        let mut pc = self.registers.pc;
        let mut idx = 0;

        while fp != 0 && idx < 100 {
            let function = self.lookup_symbol(pc);

            self.backtrace.push(StackFrame {
                index: idx,
                pc,
                sp: fp,
                fp,
                function: function.map(|s| s.name.clone()),
                file: None,
                line: None,
            });

            // Read next frame pointer and return address
            // TODO: Actually read from memory
            break;
        }

        Ok(&self.backtrace)
    }

    /// Lookup symbol at address
    pub fn lookup_symbol(&self, address: u64) -> Option<&Symbol> {
        self.symbols.iter().find(|s| {
            address >= s.address && address < s.address + s.size
        })
    }

    /// Lookup address for symbol name
    pub fn lookup_address(&self, name: &str) -> Option<u64> {
        self.symbols.iter()
            .find(|s| s.name == name)
            .map(|s| s.address)
    }

    /// Load symbols from file
    pub fn load_symbols(&mut self, path: &str) -> Result<usize, DebugError> {
        // TODO: Parse ELF symbol table
        Ok(0)
    }

    /// Get memory regions
    pub fn memory_map(&self) -> Result<&[MemoryRegion], DebugError> {
        self.check_stopped()?;
        Ok(&self.memory_regions)
    }

    /// Get current state
    pub fn state(&self) -> DebuggerState {
        self.state
    }

    /// Get stop reason
    pub fn stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }

    /// Get target PID
    pub fn target_pid(&self) -> Option<u32> {
        self.target_pid
    }

    /// Execute debugger command
    pub fn execute_command(&mut self, cmd: &str) -> Result<String, DebugError> {
        self.history.push(String::from(cmd));

        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() {
            return Ok(String::new());
        }

        match parts[0] {
            "attach" | "a" => {
                let pid: u32 = parts.get(1)
                    .and_then(|s| s.parse().ok())
                    .ok_or(DebugError::InvalidCommand)?;
                self.attach(pid)?;
                Ok(format!("Attached to process {}", pid))
            }

            "detach" | "d" => {
                self.detach()?;
                Ok(String::from("Detached"))
            }

            "continue" | "c" => {
                self.continue_execution()?;
                Ok(String::from("Continuing..."))
            }

            "step" | "s" => {
                self.step()?;
                Ok(format!("Stepped to {:#x}", self.registers.pc))
            }

            "next" | "n" => {
                self.step_over()?;
                Ok(format!("Stepped to {:#x}", self.registers.pc))
            }

            "finish" | "fin" => {
                self.step_out()?;
                Ok(String::from("Running until return..."))
            }

            "break" | "b" => {
                let addr = self.parse_address(parts.get(1).unwrap_or(&""))?;
                let id = self.set_breakpoint(addr)?;
                Ok(format!("Breakpoint {} at {:#x}", id, addr))
            }

            "delete" | "del" => {
                let id: u32 = parts.get(1)
                    .and_then(|s| s.parse().ok())
                    .ok_or(DebugError::InvalidCommand)?;
                self.delete_breakpoint(id)?;
                Ok(format!("Deleted breakpoint {}", id))
            }

            "info" => {
                match parts.get(1).unwrap_or(&"") {
                    &"breakpoints" | &"b" => {
                        let mut output = String::from("Breakpoints:\n");
                        for bp in self.breakpoints.values() {
                            output.push_str(&format!(
                                "  {} {} at {:#x} (hits: {})\n",
                                bp.id,
                                if bp.enabled { "enabled" } else { "disabled" },
                                bp.address,
                                bp.hit_count
                            ));
                        }
                        Ok(output)
                    }
                    &"registers" | &"r" => {
                        Ok(self.registers.format())
                    }
                    _ => Ok(String::from("Usage: info [breakpoints|registers]"))
                }
            }

            "x" => {
                // Examine memory: x/16xw 0x1000
                let count = 16;
                let addr = self.parse_address(parts.get(1).unwrap_or(&"0"))?;
                let data = self.read_memory(addr, count * 4)?;

                let mut output = String::new();
                for (i, chunk) in data.chunks(4).enumerate() {
                    if i % 4 == 0 {
                        output.push_str(&format!("{:#x}:  ", addr + (i * 4) as u64));
                    }
                    let word = u32::from_le_bytes(chunk.try_into().unwrap_or([0; 4]));
                    output.push_str(&format!("{:08x}  ", word));
                    if i % 4 == 3 {
                        output.push('\n');
                    }
                }
                Ok(output)
            }

            "print" | "p" => {
                let name = parts.get(1).unwrap_or(&"");
                if name.starts_with('$') {
                    let reg = &name[1..];
                    let value = self.get_register(reg)?;
                    Ok(format!("{} = {:#x} ({})", name, value, value))
                } else {
                    // Try as address/symbol
                    if let Some(sym) = self.lookup_symbol(
                        self.parse_address(name).unwrap_or(0)
                    ) {
                        Ok(format!("{} at {:#x}", sym.name, sym.address))
                    } else {
                        Err(DebugError::SymbolNotFound)
                    }
                }
            }

            "bt" | "backtrace" => {
                let frames = self.backtrace()?;
                let mut output = String::new();
                for frame in frames {
                    let func = frame.function.as_deref().unwrap_or("??");
                    output.push_str(&format!(
                        "#{} {:#x} in {} ()\n",
                        frame.index, frame.pc, func
                    ));
                }
                Ok(output)
            }

            "help" | "h" => {
                Ok(String::from(
                    "Commands:\n\
                     attach <pid>    - Attach to process\n\
                     detach          - Detach from process\n\
                     continue (c)    - Continue execution\n\
                     step (s)        - Single step instruction\n\
                     next (n)        - Step over function call\n\
                     finish          - Run until function return\n\
                     break (b) <addr> - Set breakpoint\n\
                     delete <id>     - Delete breakpoint\n\
                     info breakpoints - List breakpoints\n\
                     info registers  - Show registers\n\
                     x <addr>        - Examine memory\n\
                     print (p) <expr> - Print expression\n\
                     bt              - Show backtrace\n\
                     help (h)        - Show this help\n"
                ))
            }

            _ => Err(DebugError::UnknownCommand),
        }
    }

    /// Parse address from string (hex, decimal, or symbol)
    fn parse_address(&self, s: &str) -> Result<u64, DebugError> {
        if s.starts_with("0x") || s.starts_with("0X") {
            u64::from_str_radix(&s[2..], 16).map_err(|_| DebugError::InvalidAddress)
        } else if let Ok(n) = s.parse::<u64>() {
            Ok(n)
        } else if let Some(addr) = self.lookup_address(s) {
            Ok(addr)
        } else {
            Err(DebugError::InvalidAddress)
        }
    }

    /// Check that we're in stopped state
    fn check_stopped(&self) -> Result<(), DebugError> {
        match self.state {
            DebuggerState::Stopped => Ok(()),
            DebuggerState::Detached => Err(DebugError::NotAttached),
            DebuggerState::Running => Err(DebugError::ProcessRunning),
            DebuggerState::Exited => Err(DebugError::ProcessExited),
        }
    }
}

impl Default for Debugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Debugger errors
#[derive(Clone, Debug)]
pub enum DebugError {
    /// Not attached to any process
    NotAttached,
    /// Already attached to a process
    AlreadyAttached,
    /// Process is running (need to stop first)
    ProcessRunning,
    /// Process has exited
    ProcessExited,
    /// Invalid address
    InvalidAddress,
    /// Invalid register name
    InvalidRegister,
    /// Symbol not found
    SymbolNotFound,
    /// Breakpoint not found
    BreakpointNotFound,
    /// Memory access error
    MemoryError,
    /// Invalid command
    InvalidCommand,
    /// Unknown command
    UnknownCommand,
    /// Permission denied
    PermissionDenied,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_access() {
        let mut regs = RegisterSet::default();
        regs.set("x0", 0x1234);
        assert_eq!(regs.get("x0"), Some(0x1234));
        assert_eq!(regs.get("w0"), Some(0x1234));
    }

    #[test]
    fn test_debugger_commands() {
        let mut dbg = Debugger::new();
        let help = dbg.execute_command("help");
        assert!(help.is_ok());
    }
}
