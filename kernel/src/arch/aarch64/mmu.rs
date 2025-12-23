//! Memory Management Unit (MMU) for ARM64
//!
//! Handles page table management and virtual memory.

use core::arch::asm;

/// Page size (4KB)
pub const PAGE_SIZE: usize = 4096;
pub const PAGE_SHIFT: usize = 12;

/// Page table levels
pub const PAGE_LEVELS: usize = 4;

/// Entries per page table
pub const ENTRIES_PER_TABLE: usize = 512;

/// Page table entry flags
pub mod flags {
    /// Entry is valid
    pub const VALID: u64 = 1 << 0;

    /// Entry is a table descriptor (not block)
    pub const TABLE: u64 = 1 << 1;

    /// Block descriptor (for large pages)
    pub const BLOCK: u64 = 0 << 1;

    /// Page descriptor (at level 3)
    pub const PAGE: u64 = 1 << 1;

    /// Access flag
    pub const AF: u64 = 1 << 10;

    /// Shareability (inner shareable)
    pub const ISH: u64 = 3 << 8;

    /// User accessible
    pub const AP_USER: u64 = 1 << 6;

    /// Read-only
    pub const AP_RO: u64 = 1 << 7;

    /// Execute never (EL0)
    pub const UXN: u64 = 1 << 54;

    /// Execute never (EL1)
    pub const PXN: u64 = 1 << 53;

    /// Normal memory (for MAIR index 0)
    pub const ATTR_NORMAL: u64 = 0 << 2;

    /// Device memory (for MAIR index 1)
    pub const ATTR_DEVICE: u64 = 1 << 2;

    /// Non-cacheable memory (for MAIR index 2)
    pub const ATTR_NC: u64 = 2 << 2;
}

/// Page table entry
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct PageTableEntry(u64);

impl PageTableEntry {
    /// Create an invalid entry
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Create a table entry pointing to next level
    pub fn table(addr: u64) -> Self {
        Self(addr | flags::VALID | flags::TABLE)
    }

    /// Create a block entry (1GB or 2MB page)
    pub fn block(addr: u64, attrs: u64) -> Self {
        Self(addr | flags::VALID | flags::BLOCK | flags::AF | attrs)
    }

    /// Create a page entry (4KB page)
    pub fn page(addr: u64, attrs: u64) -> Self {
        Self(addr | flags::VALID | flags::PAGE | flags::AF | attrs)
    }

    /// Check if entry is valid
    pub fn is_valid(&self) -> bool {
        self.0 & flags::VALID != 0
    }

    /// Check if entry is a table descriptor
    pub fn is_table(&self) -> bool {
        self.is_valid() && (self.0 & flags::TABLE != 0)
    }

    /// Get the physical address
    pub fn addr(&self) -> u64 {
        self.0 & 0x0000_FFFF_FFFF_F000
    }

    /// Get raw value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Page table (512 entries = 4KB)
#[repr(C, align(4096))]
pub struct PageTable {
    pub entries: [PageTableEntry; ENTRIES_PER_TABLE],
}

impl PageTable {
    /// Create an empty page table
    pub const fn new() -> Self {
        Self {
            entries: [PageTableEntry::empty(); ENTRIES_PER_TABLE],
        }
    }
}

/// Virtual address breakdown
pub struct VirtualAddress {
    pub l0_idx: usize, // Level 0 index (bits 47:39)
    pub l1_idx: usize, // Level 1 index (bits 38:30)
    pub l2_idx: usize, // Level 2 index (bits 29:21)
    pub l3_idx: usize, // Level 3 index (bits 20:12)
    pub offset: usize, // Page offset (bits 11:0)
}

impl VirtualAddress {
    /// Parse a virtual address
    pub fn from_addr(addr: usize) -> Self {
        Self {
            l0_idx: (addr >> 39) & 0x1FF,
            l1_idx: (addr >> 30) & 0x1FF,
            l2_idx: (addr >> 21) & 0x1FF,
            l3_idx: (addr >> 12) & 0x1FF,
            offset: addr & 0xFFF,
        }
    }
}

/// Get the current page table base (TTBR0_EL1)
pub fn get_ttbr0() -> u64 {
    let ttbr: u64;
    unsafe {
        asm!("mrs {}, ttbr0_el1", out(reg) ttbr);
    }
    ttbr
}

/// Get the kernel page table base (TTBR1_EL1)
pub fn get_ttbr1() -> u64 {
    let ttbr: u64;
    unsafe {
        asm!("mrs {}, ttbr1_el1", out(reg) ttbr);
    }
    ttbr
}

/// Set the user page table base (TTBR0_EL1)
///
/// # Safety
/// The page table must be valid
pub unsafe fn set_ttbr0(addr: u64) {
    asm!(
        "msr ttbr0_el1, {}",
        "isb",
        in(reg) addr,
    );
}

/// Set the kernel page table base (TTBR1_EL1)
///
/// # Safety
/// The page table must be valid
pub unsafe fn set_ttbr1(addr: u64) {
    asm!(
        "msr ttbr1_el1, {}",
        "isb",
        in(reg) addr,
    );
}

/// Invalidate TLB for address
pub fn tlb_invalidate_addr(addr: u64) {
    unsafe {
        asm!(
            "dsb ishst",
            "tlbi vaae1is, {}",
            "dsb ish",
            "isb",
            in(reg) addr >> 12,
        );
    }
}

/// Invalidate entire TLB
pub fn tlb_invalidate_all() {
    unsafe {
        asm!("dsb ishst", "tlbi vmalle1is", "dsb ish", "isb",);
    }
}

/// Configure MMU attributes (MAIR_EL1)
///
/// Index 0: Normal memory (write-back cacheable)
/// Index 1: Device memory (nGnRnE)
/// Index 2: Normal non-cacheable
pub fn configure_mair() {
    let mair: u64 = (0xFF << 0) |  // Index 0: Normal, Write-Back
        (0x00 << 8) |  // Index 1: Device-nGnRnE
        (0x44 << 16); // Index 2: Normal, Non-Cacheable

    unsafe {
        asm!(
            "msr mair_el1, {}",
            "isb",
            in(reg) mair,
        );
    }
}

/// Configure TCR_EL1 (Translation Control Register)
///
/// Sets up 48-bit virtual addresses, 4KB pages
pub fn configure_tcr() {
    let tcr: u64 = (16 << 0) |   // T0SZ = 16 (48-bit VA for TTBR0)
        (16 << 16) |  // T1SZ = 16 (48-bit VA for TTBR1)
        (0 << 6) |    // EPD0 = 0 (enable TTBR0 walks)
        (0 << 23) |   // EPD1 = 0 (enable TTBR1 walks)
        (0b10 << 8) | // IRGN0 = Write-back
        (0b10 << 10) |// ORGN0 = Write-back
        (0b11 << 12) |// SH0 = Inner shareable
        (0b10 << 24) |// IRGN1 = Write-back
        (0b10 << 26) |// ORGN1 = Write-back
        (0b11 << 28) |// SH1 = Inner shareable
        (0b10 << 30) |// TG0 = 4KB
        (0b10 << 14); // TG1 = 4KB

    unsafe {
        asm!(
            "msr tcr_el1, {}",
            "isb",
            in(reg) tcr,
        );
    }
}

/// Enable the MMU
///
/// # Safety
/// Page tables must be properly set up first
pub unsafe fn enable() {
    asm!(
        // Ensure all page table writes are visible
        "dsb sy",
        "isb",

        // Enable MMU (SCTLR_EL1.M = 1)
        "mrs x0, sctlr_el1",
        "orr x0, x0, #1",      // M bit
        "orr x0, x0, #(1<<2)", // C bit (cache enable)
        "orr x0, x0, #(1<<12)",// I bit (instruction cache enable)
        "msr sctlr_el1, x0",

        // Synchronization barrier
        "isb",

        out("x0") _,
    );
}

/// Disable the MMU
///
/// # Safety
/// This can cause undefined behavior if virtual addresses are in use
pub unsafe fn disable() {
    asm!(
        "mrs x0, sctlr_el1",
        "bic x0, x0, #1",      // Clear M bit
        "msr sctlr_el1, x0",
        "isb",
        out("x0") _,
    );
}

/// Check if MMU is enabled
pub fn is_enabled() -> bool {
    let sctlr: u64;
    unsafe {
        asm!("mrs {}, sctlr_el1", out(reg) sctlr);
    }
    sctlr & 1 != 0
}
