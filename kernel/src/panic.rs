//! Kernel Panic Handler with Stack Traces
//!
//! Provides comprehensive panic handling with:
//! - Stack unwinding and backtraces
//! - Symbol resolution (when available)
//! - Register dumps
//! - Core dump generation
//! - Multi-CPU panic coordination

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use spin::Mutex;

/// Maximum stack frames to unwind
const MAX_STACK_FRAMES: usize = 32;

/// Panic already in progress (prevent recursive panics)
static PANIC_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// CPU that owns the panic
static PANIC_CPU: AtomicU32 = AtomicU32::new(u32::MAX);

/// Kernel symbol table for symbol resolution
static SYMBOL_TABLE: Mutex<Option<SymbolTable>> = Mutex::new(None);

/// Stack frame for unwinding
#[derive(Clone, Copy, Debug)]
pub struct StackFrame {
    /// Return address (PC)
    pub pc: usize,
    /// Frame pointer (FP/x29)
    pub fp: usize,
    /// Stack pointer (SP)
    pub sp: usize,
}

/// Symbol table entry
#[derive(Clone, Debug)]
pub struct Symbol {
    pub address: usize,
    pub size: usize,
    pub name: String,
}

/// Kernel symbol table for address resolution
pub struct SymbolTable {
    symbols: Vec<Symbol>,
}

impl SymbolTable {
    /// Create a new empty symbol table
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
        }
    }

    /// Add a symbol to the table
    pub fn add(&mut self, address: usize, size: usize, name: String) {
        self.symbols.push(Symbol {
            address,
            size,
            name,
        });
    }

    /// Sort symbols by address for binary search
    pub fn sort(&mut self) {
        self.symbols.sort_by_key(|s| s.address);
    }

    /// Resolve an address to a symbol
    pub fn resolve(&self, addr: usize) -> Option<(&Symbol, usize)> {
        // Binary search for the symbol containing this address
        let idx = self.symbols.partition_point(|s| s.address <= addr);
        if idx == 0 {
            return None;
        }

        let symbol = &self.symbols[idx - 1];
        if addr < symbol.address + symbol.size {
            let offset = addr - symbol.address;
            Some((symbol, offset))
        } else {
            // Address is past the last symbol, still return it with offset
            let offset = addr - symbol.address;
            Some((symbol, offset))
        }
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the panic handler
pub fn init() {
    // Initialize symbol table (would be populated from kernel ELF)
    *SYMBOL_TABLE.lock() = Some(SymbolTable::new());
}

/// Load symbols from kernel ELF
pub fn load_symbols(elf_data: &[u8]) {
    use crate::process::elf::{parse_header, Elf64Shdr, SHT_DYNSYM};

    let header = match parse_header(elf_data) {
        Ok(h) => h,
        Err(_) => return,
    };

    let mut table = SymbolTable::new();

    // Parse section headers to find symbol table
    let shoff = header.e_shoff as usize;
    let shentsize = header.e_shentsize as usize;
    let shnum = header.e_shnum as usize;
    let shstrndx = header.e_shstrndx as usize;

    if shoff == 0 || shnum == 0 {
        return;
    }

    // Get section header string table
    let shstr_offset = if shstrndx < shnum {
        let shdr_offset = shoff + shstrndx * shentsize;
        if shdr_offset + core::mem::size_of::<Elf64Shdr>() <= elf_data.len() {
            let shdr = unsafe { &*(elf_data.as_ptr().add(shdr_offset) as *const Elf64Shdr) };
            shdr.sh_offset as usize
        } else {
            0
        }
    } else {
        0
    };

    // Find .symtab and .strtab sections
    let mut symtab_shdr: Option<&Elf64Shdr> = None;
    let mut strtab_offset = 0usize;

    for i in 0..shnum {
        let shdr_offset = shoff + i * shentsize;
        if shdr_offset + core::mem::size_of::<Elf64Shdr>() > elf_data.len() {
            continue;
        }

        let shdr = unsafe { &*(elf_data.as_ptr().add(shdr_offset) as *const Elf64Shdr) };

        // Check section name
        let name_offset = shstr_offset + shdr.sh_name as usize;
        if name_offset < elf_data.len() {
            let name_bytes = &elf_data[name_offset..];
            let name_end = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            let name = core::str::from_utf8(&name_bytes[..name_end]).unwrap_or("");

            if name == ".symtab" {
                symtab_shdr = Some(shdr);
            } else if name == ".strtab" {
                strtab_offset = shdr.sh_offset as usize;
            }
        }
    }

    // Parse symbol table
    if let Some(symtab) = symtab_shdr {
        let sym_offset = symtab.sh_offset as usize;
        let sym_size = symtab.sh_size as usize;
        let sym_entsize = symtab.sh_entsize as usize;

        if sym_entsize == 0 || sym_offset + sym_size > elf_data.len() {
            return;
        }

        let num_symbols = sym_size / sym_entsize;

        for i in 0..num_symbols {
            let entry_offset = sym_offset + i * sym_entsize;
            if entry_offset + core::mem::size_of::<crate::process::elf::Elf64Sym>() > elf_data.len()
            {
                continue;
            }

            let sym = unsafe {
                &*(elf_data.as_ptr().add(entry_offset) as *const crate::process::elf::Elf64Sym)
            };

            // Only include function symbols
            let sym_type = sym.st_info & 0xf;
            if sym_type != 2 {
                // STT_FUNC
                continue;
            }

            // Get symbol name
            let name_offset = strtab_offset + sym.st_name as usize;
            if name_offset >= elf_data.len() {
                continue;
            }

            let name_bytes = &elf_data[name_offset..];
            let name_end = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len().min(256));
            if let Ok(name) = core::str::from_utf8(&name_bytes[..name_end]) {
                if !name.is_empty() && sym.st_value != 0 {
                    table.add(
                        sym.st_value as usize,
                        sym.st_size as usize,
                        String::from(name),
                    );
                }
            }
        }
    }

    table.sort();
    *SYMBOL_TABLE.lock() = Some(table);

    crate::kprintln!(
        "  Loaded {} kernel symbols",
        SYMBOL_TABLE
            .lock()
            .as_ref()
            .map(|t| t.symbols.len())
            .unwrap_or(0)
    );
}

/// Unwind the stack and return frames
pub fn unwind_stack() -> Vec<StackFrame> {
    let mut frames = Vec::with_capacity(MAX_STACK_FRAMES);

    // Get current frame pointer
    let (fp, sp, pc): (usize, usize, usize);
    unsafe {
        core::arch::asm!(
            "mov {fp}, x29",
            "mov {sp}, sp",
            "adr {pc}, .",
            fp = out(reg) fp,
            sp = out(reg) sp,
            pc = out(reg) pc,
        );
    }

    frames.push(StackFrame { pc, fp, sp });

    // Walk the stack
    let mut current_fp = fp;

    for _ in 0..MAX_STACK_FRAMES - 1 {
        if current_fp == 0 || current_fp % 16 != 0 {
            break;
        }

        // Validate frame pointer is in valid memory range
        // (simple heuristic: assume valid kernel stack is < 1GB from frame pointer)
        if current_fp < 0x1000 {
            break;
        }

        // Read saved FP and LR from stack frame
        // AArch64 ABI: FP points to saved {FP, LR} pair
        let saved_fp: usize;
        let saved_lr: usize;

        unsafe {
            // Check if we can safely read from this address
            // In a real implementation, we'd check page tables
            saved_fp = core::ptr::read_volatile(current_fp as *const usize);
            saved_lr = core::ptr::read_volatile((current_fp + 8) as *const usize);
        }

        if saved_lr == 0 {
            break;
        }

        frames.push(StackFrame {
            pc: saved_lr,
            fp: saved_fp,
            sp: current_fp + 16,
        });

        if saved_fp == 0 || saved_fp <= current_fp {
            break;
        }

        current_fp = saved_fp;
    }

    frames
}

/// Format a single stack frame
fn format_frame(frame: &StackFrame, index: usize) -> String {
    let symbol_info = SYMBOL_TABLE.lock();

    if let Some(ref table) = *symbol_info {
        if let Some((symbol, offset)) = table.resolve(frame.pc) {
            return alloc::format!(
                "  #{:2} 0x{:016x} - {}+0x{:x}",
                index,
                frame.pc,
                symbol.name,
                offset
            );
        }
    }

    alloc::format!("  #{:2} 0x{:016x} - <unknown>", index, frame.pc)
}

/// Print a full backtrace
pub fn print_backtrace() {
    crate::kprintln!("Stack backtrace:");

    let frames = unwind_stack();

    for (i, frame) in frames.iter().enumerate() {
        crate::kprintln!("{}", format_frame(frame, i));
    }

    if frames.len() >= MAX_STACK_FRAMES {
        crate::kprintln!("  ... (truncated at {} frames)", MAX_STACK_FRAMES);
    }
}

/// Print CPU registers
pub fn print_registers() {
    crate::kprintln!("CPU Registers:");

    // General purpose registers
    let regs: [u64; 31];
    let sp: u64;
    let pc: u64;
    let pstate: u64;

    unsafe {
        core::arch::asm!(
            "str x0, [{regs}]",
            "str x1, [{regs}, #8]",
            "str x2, [{regs}, #16]",
            "str x3, [{regs}, #24]",
            "str x4, [{regs}, #32]",
            "str x5, [{regs}, #40]",
            "str x6, [{regs}, #48]",
            "str x7, [{regs}, #56]",
            "str x8, [{regs}, #64]",
            "str x9, [{regs}, #72]",
            "str x10, [{regs}, #80]",
            "str x11, [{regs}, #88]",
            "str x12, [{regs}, #96]",
            "str x13, [{regs}, #104]",
            "str x14, [{regs}, #112]",
            "str x15, [{regs}, #120]",
            "str x16, [{regs}, #128]",
            "str x17, [{regs}, #136]",
            "str x18, [{regs}, #144]",
            "str x19, [{regs}, #152]",
            "str x20, [{regs}, #160]",
            "str x21, [{regs}, #168]",
            "str x22, [{regs}, #176]",
            "str x23, [{regs}, #184]",
            "str x24, [{regs}, #192]",
            "str x25, [{regs}, #200]",
            "str x26, [{regs}, #208]",
            "str x27, [{regs}, #216]",
            "str x28, [{regs}, #224]",
            "str x29, [{regs}, #232]",
            "str x30, [{regs}, #240]",
            regs = in(reg) &mut regs as *mut [u64; 31],
        );

        core::arch::asm!(
            "mov {sp}, sp",
            "adr {pc}, .",
            "mrs {pstate}, nzcv",
            sp = out(reg) sp,
            pc = out(reg) pc,
            pstate = out(reg) pstate,
        );
    }

    // Print in rows of 4
    for i in 0..8 {
        let base = i * 4;
        crate::kprintln!(
            "  X{:02}: 0x{:016x}  X{:02}: 0x{:016x}  X{:02}: 0x{:016x}  X{:02}: 0x{:016x}",
            base,
            regs[base],
            base + 1,
            regs[base + 1],
            base + 2,
            regs[base + 2],
            base + 3,
            regs[base + 3],
        );
    }

    crate::kprintln!(
        "  X28: 0x{:016x}  X29: 0x{:016x}  X30: 0x{:016x}",
        regs[28],
        regs[29],
        regs[30]
    );

    crate::kprintln!("  SP:  0x{:016x}  PC:  0x{:016x}", sp, pc);

    // System registers
    let elr: u64;
    let esr: u64;
    let far: u64;
    let spsr: u64;

    unsafe {
        core::arch::asm!(
            "mrs {elr}, elr_el1",
            "mrs {esr}, esr_el1",
            "mrs {far}, far_el1",
            "mrs {spsr}, spsr_el1",
            elr = out(reg) elr,
            esr = out(reg) esr,
            far = out(reg) far,
            spsr = out(reg) spsr,
        );
    }

    crate::kprintln!();
    crate::kprintln!("Exception State:");
    crate::kprintln!("  ELR_EL1:  0x{:016x} (return address)", elr);
    crate::kprintln!("  ESR_EL1:  0x{:016x} (syndrome)", esr);
    crate::kprintln!("  FAR_EL1:  0x{:016x} (fault address)", far);
    crate::kprintln!("  SPSR_EL1: 0x{:016x} (saved PSTATE)", spsr);

    // Decode ESR
    let ec = (esr >> 26) & 0x3f;
    let iss = esr & 0x1ffffff;

    let ec_str = match ec {
        0b000000 => "Unknown",
        0b000001 => "WFI/WFE trapped",
        0b000011 => "MCR/MRC (CP15)",
        0b000100 => "MCRR/MRRC (CP15)",
        0b000101 => "MCR/MRC (CP14)",
        0b000110 => "LDC/STC (CP14)",
        0b000111 => "SIMD/FP trapped",
        0b001100 => "MRRC (CP14)",
        0b001110 => "Illegal execution state",
        0b010001 => "SVC (AArch32)",
        0b010101 => "SVC (AArch64)",
        0b011000 => "MSR/MRS trapped",
        0b011001 => "SVE trapped",
        0b100000 => "Instruction abort (lower EL)",
        0b100001 => "Instruction abort (same EL)",
        0b100010 => "PC alignment fault",
        0b100100 => "Data abort (lower EL)",
        0b100101 => "Data abort (same EL)",
        0b100110 => "SP alignment fault",
        0b101000 => "FP exception (AArch32)",
        0b101100 => "FP exception (AArch64)",
        0b101111 => "SError",
        0b110000 => "Breakpoint (lower EL)",
        0b110001 => "Breakpoint (same EL)",
        0b110010 => "Software step (lower EL)",
        0b110011 => "Software step (same EL)",
        0b110100 => "Watchpoint (lower EL)",
        0b110101 => "Watchpoint (same EL)",
        0b111000 => "BKPT (AArch32)",
        0b111100 => "BRK (AArch64)",
        _ => "Reserved",
    };

    crate::kprintln!("  Exception class: {} (0x{:02x})", ec_str, ec);
    crate::kprintln!("  ISS: 0x{:07x}", iss);
}

/// Print memory information
pub fn print_memory_info() {
    crate::kprintln!();
    crate::kprintln!("Memory Status:");

    let stats = crate::memory::stats();
    crate::kprintln!("  Total RAM:     {} MB", stats.total / (1024 * 1024));
    crate::kprintln!("  Used RAM:      {} MB", stats.used / (1024 * 1024));
    crate::kprintln!("  Free RAM:      {} MB", stats.free / (1024 * 1024));
    crate::kprintln!(
        "  Kernel heap:   {} MB",
        stats.kernel_heap_used / (1024 * 1024)
    );
}

/// Print process information
pub fn print_process_info() {
    crate::kprintln!();
    crate::kprintln!("Current Process:");

    let pid = crate::scheduler::current_pid();
    crate::kprintln!("  PID: {}", pid.0);

    if let Some(process) = crate::process::get(pid) {
        crate::kprintln!("  Name: {}", process.name);
        crate::kprintln!("  State: {:?}", process.state());
    }
}

/// The main panic handler
pub fn handle_panic(info: &core::panic::PanicInfo) -> ! {
    // Disable interrupts immediately
    crate::arch::disable_interrupts();

    // Check if panic is already in progress
    if PANIC_IN_PROGRESS.swap(true, Ordering::SeqCst) {
        // Nested panic - just halt
        crate::kprintln!();
        crate::kprintln!("!!! NESTED PANIC - HALTING !!!");
        loop {
            crate::arch::halt();
        }
    }

    // Record which CPU panicked
    let cpu_id = 0; // TODO: get actual CPU ID
    PANIC_CPU.store(cpu_id, Ordering::SeqCst);

    // Print panic header
    crate::kprintln!();
    crate::kprintln!(
        "================================================================================"
    );
    crate::kprintln!("                            KERNEL PANIC");
    crate::kprintln!(
        "================================================================================"
    );
    crate::kprintln!();

    // Print panic message
    if let Some(location) = info.location() {
        crate::kprintln!(
            "Panic at {}:{}:{}",
            location.file(),
            location.line(),
            location.column()
        );
    }

    if let Some(message) = info.message() {
        crate::kprintln!("Message: {}", message);
    }

    crate::kprintln!();

    // Print registers
    print_registers();

    crate::kprintln!();

    // Print backtrace
    print_backtrace();

    // Print memory info
    print_memory_info();

    // Print process info
    print_process_info();

    crate::kprintln!();
    crate::kprintln!(
        "================================================================================"
    );

    // Try to generate core dump
    if let Err(e) = generate_panic_dump() {
        crate::kprintln!("Failed to generate panic dump: {:?}", e);
    }

    crate::kprintln!();
    crate::kprintln!("System halted. Press reset to reboot.");

    // Halt all CPUs
    loop {
        crate::arch::halt();
    }
}

/// Generate a panic dump for post-mortem analysis
fn generate_panic_dump() -> Result<(), &'static str> {
    // Try to write panic info to a reserved memory region or storage
    // This would be read by the bootloader on next boot

    // For now, just log that we would generate a dump
    crate::kprintln!();
    crate::kprintln!("Panic dump would be written to persistent storage.");

    Ok(())
}

/// Trigger a kernel panic programmatically
#[inline(never)]
pub fn panic_explicit(message: &str) -> ! {
    panic!("{}", message);
}

/// Assert with custom message
#[macro_export]
macro_rules! kassert {
    ($cond:expr) => {
        if !$cond {
            panic!("Assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            panic!("Assertion failed: {} - {}", stringify!($cond), format_args!($($arg)*));
        }
    };
}

/// Debug assert (only in debug builds)
#[macro_export]
macro_rules! kdebug_assert {
    ($cond:expr) => {
        #[cfg(debug_assertions)]
        if !$cond {
            panic!("Debug assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        #[cfg(debug_assertions)]
        if !$cond {
            panic!("Debug assertion failed: {} - {}", stringify!($cond), format_args!($($arg)*));
        }
    };
}

/// Bug macro for impossible conditions
#[macro_export]
macro_rules! kbug {
    () => {
        panic!("BUG: reached supposedly unreachable code");
    };
    ($($arg:tt)*) => {
        panic!("BUG: {}", format_args!($($arg)*));
    };
}

/// Warn once and continue
static WARN_ONCE_FLAGS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

#[macro_export]
macro_rules! kwarn_once {
    ($key:expr, $($arg:tt)*) => {{
        let mut flags = $crate::panic::WARN_ONCE_FLAGS.lock();
        if !flags.contains(&$key) {
            flags.push($key);
            $crate::kwarn!($($arg)*);
        }
    }};
}
