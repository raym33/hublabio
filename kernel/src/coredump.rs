//! Core Dump Support
//!
//! Generates ELF core dumps for crashed processes.
//! Used for post-mortem debugging with GDB and other tools.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::Mutex;

use crate::process::Pid;
use crate::signal::Signal;

/// Core dump format
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreFormat {
    /// ELF core file
    Elf,
    /// Compressed ELF
    ElfCompressed,
    /// Minimal (registers only)
    Minimal,
}

/// Core dump configuration
pub struct CoreConfig {
    /// Enable core dumps
    pub enabled: bool,
    /// Default format
    pub format: CoreFormat,
    /// Core file pattern (like /proc/sys/kernel/core_pattern)
    pub pattern: String,
    /// Maximum core size (0 = unlimited)
    pub max_size: u64,
    /// Dump filter (bitmask of what to include)
    pub filter: u32,
    /// Pipe to program instead of file
    pub pipe_program: Option<String>,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            format: CoreFormat::Elf,
            pattern: String::from("core.%p"),
            max_size: 0,  // Unlimited
            filter: 0x3F, // Default filter
            pipe_program: None,
        }
    }
}

/// Core dump filter bits
pub mod filter {
    /// Anonymous private mappings
    pub const ANON_PRIVATE: u32 = 1 << 0;
    /// Anonymous shared mappings
    pub const ANON_SHARED: u32 = 1 << 1;
    /// File-backed private mappings
    pub const MAPPED_PRIVATE: u32 = 1 << 2;
    /// File-backed shared mappings
    pub const MAPPED_SHARED: u32 = 1 << 3;
    /// ELF headers
    pub const ELF_HEADERS: u32 = 1 << 4;
    /// Private huge pages
    pub const HUGETLB_PRIVATE: u32 = 1 << 5;
    /// Shared huge pages
    pub const HUGETLB_SHARED: u32 = 1 << 6;
}

/// ELF header constants
mod elf {
    pub const ELFMAG: [u8; 4] = [0x7f, b'E', b'L', b'F'];
    pub const ELFCLASS64: u8 = 2;
    pub const ELFDATA2LSB: u8 = 1;
    pub const EV_CURRENT: u8 = 1;
    pub const ELFOSABI_NONE: u8 = 0;

    pub const ET_CORE: u16 = 4;
    pub const EM_AARCH64: u16 = 183;
    pub const EM_RISCV: u16 = 243;
    pub const EM_X86_64: u16 = 62;

    pub const PT_NULL: u32 = 0;
    pub const PT_LOAD: u32 = 1;
    pub const PT_NOTE: u32 = 4;

    pub const PF_X: u32 = 1;
    pub const PF_W: u32 = 2;
    pub const PF_R: u32 = 4;

    pub const NT_PRSTATUS: u32 = 1;
    pub const NT_PRFPREG: u32 = 2;
    pub const NT_PRPSINFO: u32 = 3;
    pub const NT_AUXV: u32 = 6;
    pub const NT_FILE: u32 = 0x46494c45; // "FILE"
    pub const NT_SIGINFO: u32 = 0x53494749; // "SIGI"
}

/// ELF64 header
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Elf64Ehdr {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

/// ELF64 program header
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Elf64Phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

/// ELF note header
#[repr(C)]
#[derive(Clone, Copy)]
struct Elf64Nhdr {
    n_namesz: u32,
    n_descsz: u32,
    n_type: u32,
}

/// Process status for core dump
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PrStatus {
    /// Signal info
    si_signo: i32,
    si_code: i32,
    si_errno: i32,
    /// Current signal
    pr_cursig: i16,
    _pad: u16,
    /// Pending signal set
    pr_sigpend: u64,
    /// Held signal set
    pr_sighold: u64,
    /// Process ID
    pr_pid: i32,
    pr_ppid: i32,
    pr_pgrp: i32,
    pr_sid: i32,
    /// User time
    pr_utime: [i64; 2],
    /// System time
    pr_stime: [i64; 2],
    /// Cumulative user time
    pr_cutime: [i64; 2],
    /// Cumulative system time
    pr_cstime: [i64; 2],
    /// General purpose registers
    pr_reg: [u64; 34], // For AArch64: x0-x30, sp, pc, pstate
}

/// Process info for core dump
#[repr(C)]
#[derive(Clone, Copy)]
struct PrPsInfo {
    /// Process state
    pr_state: u8,
    /// State character
    pr_sname: u8,
    /// Zombie flag
    pr_zomb: u8,
    /// Nice value
    pr_nice: i8,
    /// Flags
    pr_flag: u64,
    /// User ID
    pr_uid: u32,
    pr_gid: u32,
    /// Process ID
    pr_pid: i32,
    pr_ppid: i32,
    pr_pgrp: i32,
    pr_sid: i32,
    /// Filename of executable
    pr_fname: [u8; 16],
    /// Command line arguments
    pr_psargs: [u8; 80],
}

impl Default for PrPsInfo {
    fn default() -> Self {
        Self {
            pr_state: 0,
            pr_sname: b'R',
            pr_zomb: 0,
            pr_nice: 0,
            pr_flag: 0,
            pr_uid: 0,
            pr_gid: 0,
            pr_pid: 0,
            pr_ppid: 0,
            pr_pgrp: 0,
            pr_sid: 0,
            pr_fname: [0; 16],
            pr_psargs: [0; 80],
        }
    }
}

/// Memory mapping entry for core dump
#[derive(Clone)]
struct CoreMapping {
    /// Virtual address start
    start: u64,
    /// Virtual address end
    end: u64,
    /// File offset
    file_offset: u64,
    /// Flags (readable, writable, executable)
    flags: u32,
    /// Optional file path
    file_path: Option<String>,
}

/// Global configuration
static CONFIG: Mutex<CoreConfig> = Mutex::new(CoreConfig {
    enabled: true,
    format: CoreFormat::Elf,
    pattern: String::new(), // Will be set in init
    max_size: 0,
    filter: 0x3F,
    pipe_program: None,
});

/// Core dump in progress
static DUMP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Core dumps generated
static DUMPS_GENERATED: AtomicU64 = AtomicU64::new(0);

/// Generate core dump for crashed process
pub fn generate_core_dump(pid: Pid, signal: Signal) -> Result<String, CoreDumpError> {
    // Check if core dumps are enabled
    let config = CONFIG.lock();
    if !config.enabled {
        return Err(CoreDumpError::Disabled);
    }

    // Check rlimit
    let max_core = crate::rlimit::get_core_limit(pid);
    if max_core == 0 {
        return Err(CoreDumpError::LimitZero);
    }

    // Prevent concurrent dumps
    if DUMP_IN_PROGRESS.swap(true, Ordering::SeqCst) {
        return Err(CoreDumpError::Busy);
    }

    let result = do_generate_core_dump(pid, signal, &config, max_core);

    DUMP_IN_PROGRESS.store(false, Ordering::SeqCst);

    result
}

fn do_generate_core_dump(
    pid: Pid,
    signal: Signal,
    config: &CoreConfig,
    max_size: u64,
) -> Result<String, CoreDumpError> {
    // Get process info
    let proc = crate::process::get(pid).ok_or(CoreDumpError::NoProcess)?;

    // Generate filename from pattern
    let filename = expand_pattern(&config.pattern, pid, &proc.name, signal);

    // Collect memory mappings
    let mappings = collect_mappings(pid, config.filter)?;

    // Collect registers
    let regs = collect_registers(pid)?;

    // Build ELF core file
    let mut core_data = Vec::new();

    // Write ELF header
    let ehdr = build_elf_header(mappings.len());
    core_data.extend_from_slice(as_bytes(&ehdr));

    // Calculate offsets
    let phdr_offset = core::mem::size_of::<Elf64Ehdr>();
    let num_phdrs = 1 + mappings.len(); // NOTE + LOADs
    let note_offset = phdr_offset + num_phdrs * core::mem::size_of::<Elf64Phdr>();

    // Build notes section
    let notes = build_notes(pid, signal, &regs, &proc)?;
    let notes_size = notes.len();

    // Calculate data offset (aligned)
    let data_offset = (note_offset + notes_size + 0xFFF) & !0xFFF;

    // Write program headers
    // PT_NOTE
    let note_phdr = Elf64Phdr {
        p_type: elf::PT_NOTE,
        p_flags: elf::PF_R,
        p_offset: note_offset as u64,
        p_vaddr: 0,
        p_paddr: 0,
        p_filesz: notes_size as u64,
        p_memsz: notes_size as u64,
        p_align: 1,
    };
    core_data.extend_from_slice(as_bytes(&note_phdr));

    // PT_LOAD for each mapping
    let mut current_offset = data_offset;
    for mapping in &mappings {
        let size = mapping.end - mapping.start;
        let phdr = Elf64Phdr {
            p_type: elf::PT_LOAD,
            p_flags: mapping.flags,
            p_offset: current_offset as u64,
            p_vaddr: mapping.start,
            p_paddr: 0,
            p_filesz: size,
            p_memsz: size,
            p_align: 0x1000,
        };
        core_data.extend_from_slice(as_bytes(&phdr));
        current_offset += size as usize;
    }

    // Pad to note offset
    while core_data.len() < note_offset {
        core_data.push(0);
    }

    // Write notes
    core_data.extend_from_slice(&notes);

    // Pad to data offset
    while core_data.len() < data_offset {
        core_data.push(0);
    }

    // Write memory contents
    for mapping in &mappings {
        let size = (mapping.end - mapping.start) as usize;

        // Check size limit
        if max_size != 0 && core_data.len() + size > max_size as usize {
            // Truncate
            let remaining = max_size as usize - core_data.len();
            let mem = read_process_memory(pid, mapping.start, remaining)?;
            core_data.extend_from_slice(&mem);
            break;
        }

        let mem = read_process_memory(pid, mapping.start, size)?;
        core_data.extend_from_slice(&mem);
    }

    // Write to file or pipe
    if let Some(ref program) = config.pipe_program {
        // Pipe to program
        pipe_core_dump(program, &core_data, pid, signal)?;
    } else {
        // Write to file
        write_core_file(&filename, &core_data)?;
    }

    DUMPS_GENERATED.fetch_add(1, Ordering::Relaxed);

    crate::kinfo!("Core dumped: {} ({} bytes)", filename, core_data.len());

    Ok(filename)
}

/// Build ELF header for core file
fn build_elf_header(num_mappings: usize) -> Elf64Ehdr {
    let mut ehdr = Elf64Ehdr::default();

    // Magic
    ehdr.e_ident[0..4].copy_from_slice(&elf::ELFMAG);
    ehdr.e_ident[4] = elf::ELFCLASS64;
    ehdr.e_ident[5] = elf::ELFDATA2LSB;
    ehdr.e_ident[6] = elf::EV_CURRENT;
    ehdr.e_ident[7] = elf::ELFOSABI_NONE;

    ehdr.e_type = elf::ET_CORE;

    #[cfg(target_arch = "aarch64")]
    {
        ehdr.e_machine = elf::EM_AARCH64;
    }

    #[cfg(target_arch = "riscv64")]
    {
        ehdr.e_machine = elf::EM_RISCV;
    }

    #[cfg(target_arch = "x86_64")]
    {
        ehdr.e_machine = elf::EM_X86_64;
    }

    ehdr.e_version = 1;
    ehdr.e_phoff = core::mem::size_of::<Elf64Ehdr>() as u64;
    ehdr.e_ehsize = core::mem::size_of::<Elf64Ehdr>() as u16;
    ehdr.e_phentsize = core::mem::size_of::<Elf64Phdr>() as u16;
    ehdr.e_phnum = (1 + num_mappings) as u16; // NOTE + LOADs

    ehdr
}

/// Build notes section
fn build_notes(
    pid: Pid,
    signal: Signal,
    regs: &[u64],
    proc: &crate::process::Process,
) -> Result<Vec<u8>, CoreDumpError> {
    let mut notes = Vec::new();

    // NT_PRSTATUS
    let mut prstatus = PrStatus::default();
    prstatus.pr_cursig = signal.as_num() as i16;
    prstatus.si_signo = signal.as_num();
    prstatus.pr_pid = pid.0 as i32;
    prstatus.pr_ppid = proc.ppid.map(|p| p.0 as i32).unwrap_or(0);

    // Copy registers
    let reg_count = regs.len().min(34);
    prstatus.pr_reg[..reg_count].copy_from_slice(&regs[..reg_count]);

    add_note(&mut notes, b"CORE\0", elf::NT_PRSTATUS, as_bytes(&prstatus));

    // NT_PRPSINFO
    let mut prpsinfo = PrPsInfo::default();
    prpsinfo.pr_pid = pid.0 as i32;
    prpsinfo.pr_ppid = proc.ppid.map(|p| p.0 as i32).unwrap_or(0);
    prpsinfo.pr_uid = proc.uid;
    prpsinfo.pr_gid = proc.gid;

    let name_bytes = proc.name.as_bytes();
    let name_len = name_bytes.len().min(15);
    prpsinfo.pr_fname[..name_len].copy_from_slice(&name_bytes[..name_len]);

    add_note(&mut notes, b"CORE\0", elf::NT_PRPSINFO, as_bytes(&prpsinfo));

    // NT_AUXV (auxiliary vector)
    let auxv: [(u64, u64); 1] = [(0, 0)]; // AT_NULL
    add_note(&mut notes, b"CORE\0", elf::NT_AUXV, as_bytes(&auxv));

    Ok(notes)
}

/// Add a note to notes section
fn add_note(notes: &mut Vec<u8>, name: &[u8], note_type: u32, data: &[u8]) {
    let nhdr = Elf64Nhdr {
        n_namesz: name.len() as u32,
        n_descsz: data.len() as u32,
        n_type: note_type,
    };

    notes.extend_from_slice(as_bytes(&nhdr));
    notes.extend_from_slice(name);

    // Pad name to 4-byte boundary
    while notes.len() % 4 != 0 {
        notes.push(0);
    }

    notes.extend_from_slice(data);

    // Pad data to 4-byte boundary
    while notes.len() % 4 != 0 {
        notes.push(0);
    }
}

/// Collect memory mappings for core dump
fn collect_mappings(pid: Pid, filter: u32) -> Result<Vec<CoreMapping>, CoreDumpError> {
    let mut mappings = Vec::new();

    // Would get from process address space
    // For now, create placeholder
    let space = crate::cow::get_address_space(pid).ok_or(CoreDumpError::NoProcess)?;

    // Would iterate VMAs
    // Placeholder: add stack
    if filter & filter::ANON_PRIVATE != 0 {
        mappings.push(CoreMapping {
            start: 0x7FFF_FF00_0000,
            end: 0x7FFF_FFFF_0000,
            file_offset: 0,
            flags: elf::PF_R | elf::PF_W,
            file_path: None,
        });
    }

    Ok(mappings)
}

/// Collect registers from process
fn collect_registers(pid: Pid) -> Result<Vec<u64>, CoreDumpError> {
    // Would get from ptrace or process context
    // For now, return zeros
    Ok(vec![0u64; 34])
}

/// Read process memory
fn read_process_memory(pid: Pid, addr: u64, len: usize) -> Result<Vec<u8>, CoreDumpError> {
    // Would read from process address space
    // For now, return zeros
    Ok(vec![0u8; len])
}

/// Expand core pattern
fn expand_pattern(pattern: &str, pid: Pid, name: &str, signal: Signal) -> String {
    let mut result = pattern.to_string();

    result = result.replace("%p", &pid.0.to_string());
    result = result.replace("%u", "0"); // Would get UID
    result = result.replace("%g", "0"); // Would get GID
    result = result.replace("%s", &(signal.as_num() as u32).to_string());
    result = result.replace(
        "%t",
        &(crate::time::monotonic_ns() / 1_000_000_000).to_string(),
    );
    result = result.replace("%h", "hublab");
    result = result.replace("%e", name);
    result = result.replace("%%", "%");

    if result.is_empty() {
        result = alloc::format!("core.{}", pid.0);
    }

    result
}

/// Write core file
fn write_core_file(path: &str, data: &[u8]) -> Result<(), CoreDumpError> {
    // Would write through VFS
    crate::kdebug!("Would write {} bytes to {}", data.len(), path);
    Ok(())
}

/// Pipe core dump to program
fn pipe_core_dump(
    program: &str,
    data: &[u8],
    pid: Pid,
    signal: Signal,
) -> Result<(), CoreDumpError> {
    // Would execute program and pipe data
    crate::kdebug!("Would pipe {} bytes to {}", data.len(), program);
    Ok(())
}

/// Convert struct to bytes
fn as_bytes<T: Sized>(p: &T) -> &[u8] {
    unsafe { core::slice::from_raw_parts((p as *const T) as *const u8, core::mem::size_of::<T>()) }
}

/// Core dump error
#[derive(Clone, Debug)]
pub enum CoreDumpError {
    /// Core dumps disabled
    Disabled,
    /// Core limit is zero
    LimitZero,
    /// Process not found
    NoProcess,
    /// Already dumping
    Busy,
    /// I/O error
    IoError,
    /// Out of memory
    OutOfMemory,
}

impl CoreDumpError {
    pub fn to_errno(&self) -> i32 {
        match self {
            CoreDumpError::Disabled => -1,     // EPERM
            CoreDumpError::LimitZero => -1,    // EPERM
            CoreDumpError::NoProcess => -3,    // ESRCH
            CoreDumpError::Busy => -16,        // EBUSY
            CoreDumpError::IoError => -5,      // EIO
            CoreDumpError::OutOfMemory => -12, // ENOMEM
        }
    }
}

// ============================================================================
// Configuration Interface
// ============================================================================

/// Set core pattern
pub fn set_pattern(pattern: &str) {
    CONFIG.lock().pattern = String::from(pattern);
}

/// Set max core size
pub fn set_max_size(size: u64) {
    CONFIG.lock().max_size = size;
}

/// Set core dump filter
pub fn set_filter(filter: u32) {
    CONFIG.lock().filter = filter;
}

/// Enable/disable core dumps
pub fn set_enabled(enabled: bool) {
    CONFIG.lock().enabled = enabled;
}

/// Set pipe program
pub fn set_pipe_program(program: Option<&str>) {
    CONFIG.lock().pipe_program = program.map(String::from);
}

/// Get statistics
pub fn get_stats() -> u64 {
    DUMPS_GENERATED.load(Ordering::Relaxed)
}

/// Check if signal should generate core dump
pub fn should_dump(signal: Signal) -> bool {
    matches!(
        signal,
        Signal::SIGQUIT
            | Signal::SIGILL
            | Signal::SIGTRAP
            | Signal::SIGABRT
            | Signal::SIGBUS
            | Signal::SIGFPE
            | Signal::SIGSEGV
            | Signal::SIGXCPU
            | Signal::SIGXFSZ
            | Signal::SIGSYS
    )
}

/// Initialize core dump subsystem
pub fn init() {
    CONFIG.lock().pattern = String::from("core.%p");
    crate::kprintln!("  Core dump support initialized");
}
