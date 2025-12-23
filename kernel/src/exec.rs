//! Program Execution
//!
//! ELF loading and process execution.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

/// ELF magic number
const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];

/// ELF class
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElfClass {
    Elf32 = 1,
    Elf64 = 2,
}

/// ELF machine type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElfMachine {
    X86 = 3,
    Arm = 40,
    X86_64 = 62,
    Aarch64 = 183,
    RiscV = 243,
}

/// ELF segment type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentType {
    Null = 0,
    Load = 1,
    Dynamic = 2,
    Interp = 3,
    Note = 4,
    Shlib = 5,
    Phdr = 6,
    Tls = 7,
}

/// ELF segment flags
pub mod segment_flags {
    pub const EXECUTE: u32 = 1;
    pub const WRITE: u32 = 2;
    pub const READ: u32 = 4;
}

/// ELF64 header
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64Header {
    pub magic: [u8; 4],
    pub class: u8,
    pub endian: u8,
    pub version: u8,
    pub os_abi: u8,
    pub abi_version: u8,
    pub _pad: [u8; 7],
    pub elf_type: u16,
    pub machine: u16,
    pub version2: u32,
    pub entry: u64,
    pub phoff: u64,
    pub shoff: u64,
    pub flags: u32,
    pub ehsize: u16,
    pub phentsize: u16,
    pub phnum: u16,
    pub shentsize: u16,
    pub shnum: u16,
    pub shstrndx: u16,
}

/// ELF64 program header
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64ProgramHeader {
    pub seg_type: u32,
    pub flags: u32,
    pub offset: u64,
    pub vaddr: u64,
    pub paddr: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub align: u64,
}

/// Execution errors
#[derive(Clone, Debug)]
pub enum ExecError {
    NotExecutable,
    InvalidFormat,
    UnsupportedArch,
    LoadError,
    OutOfMemory,
    FileNotFound,
    PermissionDenied,
    TooManyArgs,
}

/// Loaded program information
#[derive(Clone, Debug)]
pub struct LoadedProgram {
    pub entry_point: usize,
    pub base_addr: usize,
    pub brk: usize, // Program break (end of data segment)
    pub stack_top: usize,
    pub interp: Option<String>,
}

/// Parse ELF header
pub fn parse_elf_header(data: &[u8]) -> Result<Elf64Header, ExecError> {
    if data.len() < 64 {
        return Err(ExecError::InvalidFormat);
    }

    if data[0..4] != ELF_MAGIC {
        return Err(ExecError::NotExecutable);
    }

    if data[4] != ElfClass::Elf64 as u8 {
        return Err(ExecError::UnsupportedArch);
    }

    let header: Elf64Header =
        unsafe { core::ptr::read_unaligned(data.as_ptr() as *const Elf64Header) };

    // Verify architecture
    #[cfg(target_arch = "aarch64")]
    if header.machine != ElfMachine::Aarch64 as u16 {
        return Err(ExecError::UnsupportedArch);
    }

    #[cfg(target_arch = "riscv64")]
    if header.machine != ElfMachine::RiscV as u16 {
        return Err(ExecError::UnsupportedArch);
    }

    Ok(header)
}

/// Parse program headers
pub fn parse_program_headers(data: &[u8], header: &Elf64Header) -> Vec<Elf64ProgramHeader> {
    let mut headers = Vec::new();

    let offset = header.phoff as usize;
    let size = header.phentsize as usize;
    let count = header.phnum as usize;

    for i in 0..count {
        let start = offset + i * size;
        if start + size > data.len() {
            break;
        }

        let ph: Elf64ProgramHeader = unsafe {
            core::ptr::read_unaligned(data[start..].as_ptr() as *const Elf64ProgramHeader)
        };

        headers.push(ph);
    }

    headers
}

/// Load ELF program into memory
pub fn load_elf(data: &[u8]) -> Result<LoadedProgram, ExecError> {
    let header = parse_elf_header(data)?;
    let phdrs = parse_program_headers(data, &header);

    let mut base_addr = usize::MAX;
    let mut brk = 0usize;
    let mut interp = None;

    // First pass: calculate memory requirements
    for ph in &phdrs {
        if ph.seg_type == SegmentType::Load as u32 {
            let vaddr = ph.vaddr as usize;
            let memsz = ph.memsz as usize;

            if vaddr < base_addr {
                base_addr = vaddr;
            }

            let end = vaddr + memsz;
            if end > brk {
                brk = end;
            }
        } else if ph.seg_type == SegmentType::Interp as u32 {
            // Dynamic linker path
            let start = ph.offset as usize;
            let size = ph.filesz as usize;
            if start + size <= data.len() {
                if let Ok(path) = core::str::from_utf8(&data[start..start + size - 1]) {
                    interp = Some(String::from(path));
                }
            }
        }
    }

    if base_addr == usize::MAX {
        return Err(ExecError::InvalidFormat);
    }

    // Second pass: load segments
    for ph in &phdrs {
        if ph.seg_type != SegmentType::Load as u32 {
            continue;
        }

        let vaddr = ph.vaddr as usize;
        let offset = ph.offset as usize;
        let filesz = ph.filesz as usize;
        let memsz = ph.memsz as usize;

        // In a real implementation:
        // 1. Allocate pages for this segment
        // 2. Map them into the process address space
        // 3. Copy data from file
        // 4. Zero BSS portion (memsz - filesz)
        // 5. Set proper permissions based on ph.flags

        crate::kdebug!(
            "ELF: Load segment at 0x{:x}, file size {}, mem size {}",
            vaddr,
            filesz,
            memsz
        );
    }

    // Set up stack
    let stack_top = 0x7FFF_FFFF_F000; // User stack top

    Ok(LoadedProgram {
        entry_point: header.entry as usize,
        base_addr,
        brk,
        stack_top,
        interp,
    })
}

/// Execute a program
pub fn execve(path: &str, argv: &[&str], envp: &[&str]) -> Result<LoadedProgram, ExecError> {
    crate::kinfo!("exec: Executing {}", path);

    // Read the file from VFS
    let data = crate::vfs::read_file(path).map_err(|_| ExecError::FileNotFound)?;

    if data.is_empty() {
        return Err(ExecError::InvalidFormat);
    }

    // Check for script (#!)
    if let Some((interp, arg)) = detect_interpreter(&data) {
        crate::kinfo!("exec: Script interpreter: {} {:?}", interp, arg);
        // Would recursively exec the interpreter
        return Err(ExecError::NotExecutable);
    }

    // Load ELF
    let program = load_elf(&data)?;

    crate::kinfo!(
        "exec: Loaded {} at 0x{:x}, entry 0x{:x}",
        path,
        program.base_addr,
        program.entry_point
    );

    // Set up stack
    let (sp, argc) = setup_stack(program.stack_top, argv, envp)?;
    crate::kdebug!("exec: Stack at 0x{:x}, argc={}", sp, argc);

    Ok(program)
}

/// Execute program from syscall
pub fn exec_program(
    path: &str,
    args: &[alloc::string::String],
    env: &[alloc::string::String],
) -> Result<(), ExecError> {
    // Convert to slices
    let argv: alloc::vec::Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let envp: alloc::vec::Vec<&str> = env.iter().map(|s| s.as_str()).collect();

    let program = execve(path, &argv, &envp)?;

    // Update current process
    if let Some(proc) = crate::process::current() {
        // Update process memory layout
        let mut memory = proc.memory.lock();
        memory.heap_start = program.brk;
        memory.heap_end = program.brk;
        memory.stack_top = program.stack_top;

        crate::kinfo!(
            "exec: Process {} executing {}, entry=0x{:x}",
            proc.pid.0,
            path,
            program.entry_point
        );
    }

    Ok(())
}

/// Set up user stack with arguments and environment
pub fn setup_stack(
    stack_top: usize,
    argv: &[&str],
    envp: &[&str],
) -> Result<(usize, usize), ExecError> {
    // Stack layout (growing down):
    // - Environment strings
    // - Argument strings
    // - Padding for alignment
    // - NULL (end of envp)
    // - envp[n-1] pointer
    // - ...
    // - envp[0] pointer
    // - NULL (end of argv)
    // - argv[n-1] pointer
    // - ...
    // - argv[0] pointer
    // - argc

    let mut sp = stack_top;

    // Push environment strings
    let mut env_ptrs = Vec::new();
    for env in envp.iter().rev() {
        sp -= env.len() + 1;
        env_ptrs.push(sp);
        // Would copy string to sp
    }
    env_ptrs.reverse();

    // Push argument strings
    let mut arg_ptrs = Vec::new();
    for arg in argv.iter().rev() {
        sp -= arg.len() + 1;
        arg_ptrs.push(sp);
        // Would copy string to sp
    }
    arg_ptrs.reverse();

    // Align to 16 bytes
    sp &= !0xF;

    // Push NULL terminator for envp
    sp -= 8;
    // Would write 0 to sp

    // Push environment pointers
    for ptr in env_ptrs.iter().rev() {
        sp -= 8;
        // Would write ptr to sp
    }

    // Push NULL terminator for argv
    sp -= 8;
    // Would write 0 to sp

    // Push argument pointers
    for ptr in arg_ptrs.iter().rev() {
        sp -= 8;
        // Would write ptr to sp
    }

    // Push argc
    sp -= 8;
    // Would write argv.len() to sp

    let argc = argv.len();
    let argv_ptr = sp + 8;

    Ok((sp, argc))
}

/// Script interpreter detection
pub fn detect_interpreter(data: &[u8]) -> Option<(String, Option<String>)> {
    if data.len() < 2 || data[0] != b'#' || data[1] != b'!' {
        return None;
    }

    // Find end of first line
    let end = data.iter().position(|&b| b == b'\n').unwrap_or(data.len());
    let line = &data[2..end];

    // Parse interpreter path and optional argument
    let line = core::str::from_utf8(line).ok()?;
    let line = line.trim();

    let mut parts = line.splitn(2, char::is_whitespace);
    let interp = parts.next()?;
    let arg = parts.next().map(|s| String::from(s.trim()));

    Some((String::from(interp), arg))
}

/// Initialize exec subsystem
pub fn init() {
    crate::kprintln!("  Exec subsystem initialized");
}
