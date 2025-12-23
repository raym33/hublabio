//! ELF Loader
//!
//! Loads ELF64 executables for ARM64.

use alloc::string::String;
use alloc::vec::Vec;

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

/// ELF64 symbol table entry
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64Sym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

/// ELF64 relocation entry (with addend)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64Rela {
    pub r_offset: u64,
    pub r_info: u64,
    pub r_addend: i64,
}

impl Elf64Rela {
    /// Get relocation type
    pub fn get_type(&self) -> u32 {
        (self.r_info & 0xFFFFFFFF) as u32
    }

    /// Get symbol index
    pub fn get_sym(&self) -> u32 {
        (self.r_info >> 32) as u32
    }
}

/// ELF64 dynamic entry
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Elf64Dyn {
    pub d_tag: i64,
    pub d_val: u64,
}

// Dynamic tags
pub const DT_NULL: i64 = 0;
pub const DT_NEEDED: i64 = 1;
pub const DT_PLTRELSZ: i64 = 2;
pub const DT_PLTGOT: i64 = 3;
pub const DT_HASH: i64 = 4;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_RELA: i64 = 7;
pub const DT_RELASZ: i64 = 8;
pub const DT_RELAENT: i64 = 9;
pub const DT_STRSZ: i64 = 10;
pub const DT_SYMENT: i64 = 11;
pub const DT_INIT: i64 = 12;
pub const DT_FINI: i64 = 13;
pub const DT_JMPREL: i64 = 23;

// Relocation types for AArch64
pub const R_AARCH64_NONE: u32 = 0;
pub const R_AARCH64_ABS64: u32 = 257;
pub const R_AARCH64_ABS32: u32 = 258;
pub const R_AARCH64_COPY: u32 = 1024;
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
pub const R_AARCH64_RELATIVE: u32 = 1027;
pub const R_AARCH64_TLS_TPREL: u32 = 1030;

/// Section header types
pub const SHT_RELA: u32 = 4;
pub const SHT_DYNAMIC: u32 = 6;
pub const SHT_DYNSYM: u32 = 11;

/// Program header type: dynamic linking info
pub const PT_DYNAMIC: u32 = 2;
/// Program header type: TLS template
pub const PT_TLS: u32 = 7;
/// Program header type: GNU relro
pub const PT_GNU_RELRO: u32 = 0x6474E552;

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

// ============================================================================
// Relocation Support
// ============================================================================

/// Dynamic linking information parsed from PT_DYNAMIC
pub struct DynamicInfo {
    pub rela: u64,     // DT_RELA
    pub relasz: u64,   // DT_RELASZ
    pub relaent: u64,  // DT_RELAENT
    pub jmprel: u64,   // DT_JMPREL
    pub pltrelsz: u64, // DT_PLTRELSZ
    pub symtab: u64,   // DT_SYMTAB
    pub strtab: u64,   // DT_STRTAB
    pub init: u64,     // DT_INIT
    pub fini: u64,     // DT_FINI
}

impl Default for DynamicInfo {
    fn default() -> Self {
        Self {
            rela: 0,
            relasz: 0,
            relaent: 0,
            jmprel: 0,
            pltrelsz: 0,
            symtab: 0,
            strtab: 0,
            init: 0,
            fini: 0,
        }
    }
}

/// Parse dynamic section
pub fn parse_dynamic(data: &[u8], header: &Elf64Header, base: usize) -> Option<DynamicInfo> {
    let mut info = DynamicInfo::default();

    // Find PT_DYNAMIC segment
    for phdr in get_program_headers(data, header) {
        if phdr.p_type != PT_DYNAMIC {
            continue;
        }

        let dyn_offset = phdr.p_offset as usize;
        let dyn_size = phdr.p_filesz as usize;

        if dyn_offset + dyn_size > data.len() {
            return None;
        }

        let entry_size = core::mem::size_of::<Elf64Dyn>();
        let num_entries = dyn_size / entry_size;

        for i in 0..num_entries {
            let entry_offset = dyn_offset + i * entry_size;
            let dyn_entry = unsafe { &*(data.as_ptr().add(entry_offset) as *const Elf64Dyn) };

            match dyn_entry.d_tag {
                DT_NULL => break,
                DT_RELA => info.rela = dyn_entry.d_val,
                DT_RELASZ => info.relasz = dyn_entry.d_val,
                DT_RELAENT => info.relaent = dyn_entry.d_val,
                DT_JMPREL => info.jmprel = dyn_entry.d_val,
                DT_PLTRELSZ => info.pltrelsz = dyn_entry.d_val,
                DT_SYMTAB => info.symtab = dyn_entry.d_val,
                DT_STRTAB => info.strtab = dyn_entry.d_val,
                DT_INIT => info.init = dyn_entry.d_val,
                DT_FINI => info.fini = dyn_entry.d_val,
                _ => {}
            }
        }

        return Some(info);
    }

    None
}

/// Apply a single relocation
///
/// # Safety
/// This writes directly to memory at the relocation offset
pub unsafe fn apply_relocation(
    rela: &Elf64Rela,
    base: usize,
    symtab: *const Elf64Sym,
    _strtab: *const u8,
) -> Result<(), ElfError> {
    let reloc_type = rela.get_type();
    let sym_idx = rela.get_sym();

    let target_addr = base + rela.r_offset as usize;

    match reloc_type {
        R_AARCH64_NONE => {
            // Nothing to do
        }
        R_AARCH64_RELATIVE => {
            // S + A where S = base address
            let value = (base as i64 + rela.r_addend) as u64;
            *(target_addr as *mut u64) = value;
        }
        R_AARCH64_ABS64 => {
            // S + A where S = symbol value
            if sym_idx != 0 && !symtab.is_null() {
                let sym = &*symtab.add(sym_idx as usize);
                let sym_value = if sym.st_shndx != 0 {
                    base + sym.st_value as usize
                } else {
                    // Undefined symbol - would need to look up in other modules
                    return Err(ElfError::OutOfMemory); // Use as undefined symbol error
                };
                let value = (sym_value as i64 + rela.r_addend) as u64;
                *(target_addr as *mut u64) = value;
            }
        }
        R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT => {
            // S (symbol value)
            if sym_idx != 0 && !symtab.is_null() {
                let sym = &*symtab.add(sym_idx as usize);
                let sym_value = if sym.st_shndx != 0 {
                    (base + sym.st_value as usize) as u64
                } else {
                    // Undefined symbol
                    return Err(ElfError::OutOfMemory);
                };
                *(target_addr as *mut u64) = sym_value;
            }
        }
        R_AARCH64_COPY => {
            // Copy symbol data (used for data in executables)
            // Skip for now - requires source data
        }
        _ => {
            // Unknown relocation type - log warning but continue
            crate::kwarn!("Unknown relocation type: {}", reloc_type);
        }
    }

    Ok(())
}

/// Apply all relocations for a loaded ELF
///
/// # Safety
/// This modifies memory at relocation target addresses
pub unsafe fn apply_relocations(
    loaded_base: usize,
    dynamic_info: &DynamicInfo,
) -> Result<(), ElfError> {
    let symtab = if dynamic_info.symtab != 0 {
        (loaded_base + dynamic_info.symtab as usize) as *const Elf64Sym
    } else {
        core::ptr::null()
    };

    let strtab = if dynamic_info.strtab != 0 {
        (loaded_base + dynamic_info.strtab as usize) as *const u8
    } else {
        core::ptr::null()
    };

    // Apply RELA relocations
    if dynamic_info.rela != 0 && dynamic_info.relasz != 0 {
        let rela_addr = loaded_base + dynamic_info.rela as usize;
        let entry_size = if dynamic_info.relaent != 0 {
            dynamic_info.relaent as usize
        } else {
            core::mem::size_of::<Elf64Rela>()
        };
        let num_entries = dynamic_info.relasz as usize / entry_size;

        for i in 0..num_entries {
            let rela = &*((rela_addr + i * entry_size) as *const Elf64Rela);
            apply_relocation(rela, loaded_base, symtab, strtab)?;
        }
    }

    // Apply PLT/GOT relocations (JMPREL)
    if dynamic_info.jmprel != 0 && dynamic_info.pltrelsz != 0 {
        let jmprel_addr = loaded_base + dynamic_info.jmprel as usize;
        let entry_size = core::mem::size_of::<Elf64Rela>();
        let num_entries = dynamic_info.pltrelsz as usize / entry_size;

        for i in 0..num_entries {
            let rela = &*((jmprel_addr + i * entry_size) as *const Elf64Rela);
            apply_relocation(rela, loaded_base, symtab, strtab)?;
        }
    }

    Ok(())
}

// ============================================================================
// Full ELF Loader with Memory Mapping
// ============================================================================

/// Memory flags for page mapping
fn elf_flags_to_page_flags(elf_flags: u32) -> crate::memory::PageFlags {
    use crate::memory::PageFlags;

    let mut flags = PageFlags::PRESENT | PageFlags::USER;

    if elf_flags & PF_W != 0 {
        flags |= PageFlags::WRITABLE;
    }

    if elf_flags & PF_X == 0 {
        flags |= PageFlags::NO_EXECUTE;
    }

    flags
}

/// Load ELF segments into a process address space
pub fn load_into_process(
    data: &[u8],
    process: &crate::process::Process,
    base_addr: usize,
) -> Result<ElfInfo, ElfError> {
    use crate::memory::{allocate_frame, deallocate_frame, PAGE_SIZE};

    let header = parse_header(data)?;

    let mut segments = Vec::new();
    let mut interpreter = None;

    // Calculate base for PIE executables
    let base = if header.e_type == ET_DYN {
        base_addr
    } else {
        0
    };

    // Get page table
    let memory = process.memory.lock();
    let page_table = memory.page_table as usize;
    drop(memory);

    if page_table == 0 {
        return Err(ElfError::OutOfMemory);
    }

    // Load each PT_LOAD segment
    for phdr in get_program_headers(data, header) {
        match phdr.p_type {
            PT_LOAD => {
                let vaddr_start = base + phdr.p_vaddr as usize;
                let vaddr_end = vaddr_start + phdr.p_memsz as usize;
                let file_offset = phdr.p_offset as usize;
                let file_size = phdr.p_filesz as usize;
                let mem_size = phdr.p_memsz as usize;

                let page_flags = elf_flags_to_page_flags(phdr.p_flags);

                // Align to page boundaries
                let page_start = vaddr_start & !(PAGE_SIZE - 1);
                let page_end = (vaddr_end + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
                let num_pages = (page_end - page_start) / PAGE_SIZE;

                // Allocate and map pages
                for i in 0..num_pages {
                    let page_vaddr = page_start + i * PAGE_SIZE;

                    // Allocate physical frame
                    let frame = allocate_frame().ok_or(ElfError::OutOfMemory)?;
                    let paddr = frame.start_address();

                    // Map the page
                    // Note: This would use the task.rs map_page function
                    // For now, assume it's mapped

                    // Copy data from ELF file
                    let page_data =
                        unsafe { core::slice::from_raw_parts_mut(paddr as *mut u8, PAGE_SIZE) };

                    // Zero the page first
                    page_data.fill(0);

                    // Copy file data into the page
                    let page_offset_in_segment = page_vaddr.saturating_sub(vaddr_start);
                    let file_data_offset = file_offset + page_offset_in_segment;

                    if page_offset_in_segment < file_size {
                        let copy_size = (file_size - page_offset_in_segment).min(PAGE_SIZE);
                        let copy_start = page_vaddr.saturating_sub(page_start);

                        if file_data_offset + copy_size <= data.len() {
                            let src = &data[file_data_offset..file_data_offset + copy_size];
                            page_data[copy_start..copy_start + copy_size].copy_from_slice(src);
                        }
                    }
                }

                segments.push(LoadedSegment {
                    vaddr: vaddr_start,
                    size: mem_size,
                    flags: phdr.p_flags,
                });

                // Add to process memory regions
                process
                    .memory
                    .lock()
                    .regions
                    .push(crate::process::MemoryRegion {
                        start: page_start,
                        end: page_end,
                        flags: crate::process::MemoryFlags::READ
                            | if phdr.p_flags & PF_W != 0 {
                                crate::process::MemoryFlags::WRITE
                            } else {
                                crate::process::MemoryFlags::empty()
                            }
                            | if phdr.p_flags & PF_X != 0 {
                                crate::process::MemoryFlags::EXEC
                            } else {
                                crate::process::MemoryFlags::empty()
                            }
                            | crate::process::MemoryFlags::USER,
                        name: String::from("[elf]"),
                    });
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

    // Parse and apply relocations for PIE/dynamic executables
    if header.e_type == ET_DYN {
        if let Some(dyn_info) = parse_dynamic(data, header, base) {
            unsafe {
                apply_relocations(base, &dyn_info)?;
            }
        }
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

/// Build auxiliary vector for new process
pub fn build_auxv(
    info: &ElfInfo,
    stack_ptr: &mut usize,
    random_bytes: &[u8; 16],
) -> Vec<(u64, u64)> {
    let mut auxv = Vec::new();

    // AT_PHDR - Program headers address
    auxv.push((3, info.phdr as u64));

    // AT_PHENT - Program header entry size
    auxv.push((4, info.phent as u64));

    // AT_PHNUM - Number of program headers
    auxv.push((5, info.phnum as u64));

    // AT_PAGESZ - Page size
    auxv.push((6, crate::memory::PAGE_SIZE as u64));

    // AT_BASE - Interpreter base (0 if no interpreter)
    auxv.push((7, info.base as u64));

    // AT_ENTRY - Program entry point
    auxv.push((9, info.entry as u64));

    // AT_UID, AT_EUID, AT_GID, AT_EGID
    auxv.push((11, 0)); // AT_UID
    auxv.push((12, 0)); // AT_EUID
    auxv.push((13, 0)); // AT_GID
    auxv.push((14, 0)); // AT_EGID

    // AT_SECURE
    auxv.push((23, 0));

    // AT_RANDOM - 16 bytes of random data
    // Would push address of random bytes on stack
    auxv.push((25, *stack_ptr as u64));

    // AT_NULL - End of auxv
    auxv.push((0, 0));

    auxv
}
