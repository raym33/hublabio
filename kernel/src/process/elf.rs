//! ELF Loader
//!
//! Loads ELF64 executables for ARM64.

use alloc::vec::Vec;
use alloc::string::String;

/// ELF magic number
pub const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];

/// ELF class: 64-bit
pub const ELFCLASS64: u8 = 2;

/// ELF data: little endian
pub const ELFDATA2LSB: u8 = 1;

/// ELF machine: AArch64
pub const EM_AARCH64: u16 = 183;

/// ELF type: executable
pub const ET_EXEC: u16 = 2;

/// ELF type: shared object (PIE)
pub const ET_DYN: u16 = 3;

/// Program header type: loadable segment
pub const PT_LOAD: u32 = 1;

/// Program header type: interpreter
pub const PT_INTERP: u32 = 3;

/// Segment flags
pub const PF_X: u32 = 1; // Execute
pub const PF_W: u32 = 2; // Write
pub const PF_R: u32 = 4; // Read

/// ELF64 header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64Header {
    pub e_ident: [u8; 16],
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

/// ELF64 program header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64Phdr {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

/// ELF64 section header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64Shdr {
    pub sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    pub sh_link: u32,
    pub sh_info: u32,
    pub sh_addralign: u64,
    pub sh_entsize: u64,
}

/// Loaded segment information
#[derive(Clone, Debug)]
pub struct LoadedSegment {
    pub vaddr: usize,
    pub size: usize,
    pub flags: u32,
}

/// ELF load result
#[derive(Clone, Debug)]
pub struct ElfInfo {
    /// Entry point address
    pub entry: usize,
    /// Loaded segments
    pub segments: Vec<LoadedSegment>,
    /// Interpreter path (if dynamic)
    pub interpreter: Option<String>,
    /// Base address (for PIE)
    pub base: usize,
    /// Program header info for aux vector
    pub phdr: usize,
    pub phent: usize,
    pub phnum: usize,
}

/// ELF loading errors
#[derive(Debug)]
pub enum ElfError {
    InvalidMagic,
    InvalidClass,
    InvalidEndian,
    InvalidMachine,
    InvalidType,
    NoLoadableSegments,
    OverlappingSegments,
    OutOfMemory,
}

/// Parse and validate ELF header
pub fn parse_header(data: &[u8]) -> Result<&Elf64Header, ElfError> {
    if data.len() < core::mem::size_of::<Elf64Header>() {
        return Err(ElfError::InvalidMagic);
    }

    let header = unsafe { &*(data.as_ptr() as *const Elf64Header) };

    // Validate magic
    if header.e_ident[0..4] != ELF_MAGIC {
        return Err(ElfError::InvalidMagic);
    }

    // Validate class (64-bit)
    if header.e_ident[4] != ELFCLASS64 {
        return Err(ElfError::InvalidClass);
    }

    // Validate endianness (little)
    if header.e_ident[5] != ELFDATA2LSB {
        return Err(ElfError::InvalidEndian);
    }

    // Validate machine (AArch64)
    if header.e_machine != EM_AARCH64 {
        return Err(ElfError::InvalidMachine);
    }

    // Validate type (executable or shared object)
    if header.e_type != ET_EXEC && header.e_type != ET_DYN {
        return Err(ElfError::InvalidType);
    }

    Ok(header)
}

/// Get program headers from ELF
pub fn get_program_headers<'a>(
    data: &'a [u8],
    header: &Elf64Header,
) -> impl Iterator<Item = &'a Elf64Phdr> {
    let phoff = header.e_phoff as usize;
    let phentsize = header.e_phentsize as usize;
    let phnum = header.e_phnum as usize;

    (0..phnum).map(move |i| {
        let offset = phoff + i * phentsize;
        unsafe { &*(data.as_ptr().add(offset) as *const Elf64Phdr) }
    })
}

/// Load an ELF file
///
/// Returns the entry point and load information
pub fn load(data: &[u8], base_addr: usize) -> Result<ElfInfo, ElfError> {
    let header = parse_header(data)?;

    let mut segments = Vec::new();
    let mut interpreter = None;
    let mut min_addr = usize::MAX;
    let mut max_addr = 0usize;

    // Calculate base for PIE executables
    let base = if header.e_type == ET_DYN {
        base_addr
    } else {
        0
    };

    // First pass: find address range and interpreter
    for phdr in get_program_headers(data, header) {
        match phdr.p_type {
            PT_LOAD => {
                let start = phdr.p_vaddr as usize;
                let end = start + phdr.p_memsz as usize;
                min_addr = min_addr.min(start);
                max_addr = max_addr.max(end);
            }
            PT_INTERP => {
                let start = phdr.p_offset as usize;
                let end = start + phdr.p_filesz as usize;
                if end <= data.len() {
                    let path_bytes = &data[start..end];
                    if let Some(nul) = path_bytes.iter().position(|&b| b == 0) {
                        if let Ok(path) = core::str::from_utf8(&path_bytes[..nul]) {
                            interpreter = Some(String::from(path));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if min_addr == usize::MAX {
        return Err(ElfError::NoLoadableSegments);
    }

    // Second pass: load segments
    for phdr in get_program_headers(data, header) {
        if phdr.p_type != PT_LOAD {
            continue;
        }

        let vaddr = base + phdr.p_vaddr as usize;
        let file_start = phdr.p_offset as usize;
        let file_end = file_start + phdr.p_filesz as usize;

        // TODO: Actually map memory and copy data
        // For now, just record the segment info

        segments.push(LoadedSegment {
            vaddr,
            size: phdr.p_memsz as usize,
            flags: phdr.p_flags,
        });

        // Would need to:
        // 1. Allocate pages for the segment
        // 2. Map them with correct permissions
        // 3. Copy data from file
        // 4. Zero BSS region (memsz > filesz)
    }

    let entry = base + header.e_entry as usize;
    let phdr_addr = base + header.e_phoff as usize;

    Ok(ElfInfo {
        entry,
        segments,
        interpreter,
        base,
        phdr: phdr_addr,
        phent: header.e_phentsize as usize,
        phnum: header.e_phnum as usize,
    })
}

/// Check if data is a valid ELF file
pub fn is_elf(data: &[u8]) -> bool {
    data.len() >= 4 && data[0..4] == ELF_MAGIC
}

/// Get ELF type (executable or shared object)
pub fn get_type(data: &[u8]) -> Option<u16> {
    if data.len() < core::mem::size_of::<Elf64Header>() {
        return None;
    }
    let header = unsafe { &*(data.as_ptr() as *const Elf64Header) };
    Some(header.e_type)
}
