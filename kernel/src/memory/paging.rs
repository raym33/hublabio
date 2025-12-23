//! Virtual Memory Paging
//!
//! Page table management for ARM64.

use core::ptr::{read_volatile, write_volatile};
use spin::Mutex;

use super::{PAGE_SIZE, PAGE_SHIFT, PageFlags, PhysFrame, VirtPage, allocate_frame};

/// Number of entries per page table
const ENTRIES_PER_TABLE: usize = 512;

/// Page table levels for 4-level paging (48-bit VA)
const PAGE_TABLE_LEVELS: usize = 4;

/// Page table entry
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PageTableEntry(u64);

impl PageTableEntry {
    /// Create an empty entry
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Create an entry pointing to a frame
    pub fn new(frame: PhysFrame, flags: PageFlags) -> Self {
        let addr = frame.start_address() as u64;
        Self(addr | flags.bits() | 0x3) // Valid + Table/Page descriptor
    }

    /// Check if entry is present
    pub fn is_present(&self) -> bool {
        self.0 & 0x1 != 0
    }

    /// Check if entry is a table pointer
    pub fn is_table(&self) -> bool {
        self.0 & 0x2 != 0
    }

    /// Get the physical frame
    pub fn frame(&self) -> Option<PhysFrame> {
        if self.is_present() {
            let addr = (self.0 & 0x0000_FFFF_FFFF_F000) as usize;
            Some(PhysFrame::containing_address(addr))
        } else {
            None
        }
    }

    /// Get flags
    pub fn flags(&self) -> PageFlags {
        PageFlags::from_bits_truncate(self.0)
    }

    /// Set flags
    pub fn set_flags(&mut self, flags: PageFlags) {
        let addr = self.0 & 0x0000_FFFF_FFFF_F000;
        self.0 = addr | flags.bits() | 0x3;
    }

    /// Clear the entry
    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

/// Page table structure
#[repr(C, align(4096))]
pub struct PageTable {
    entries: [PageTableEntry; ENTRIES_PER_TABLE],
}

impl PageTable {
    /// Create an empty page table
    pub const fn new() -> Self {
        Self {
            entries: [PageTableEntry::empty(); ENTRIES_PER_TABLE],
        }
    }

    /// Get entry at index
    pub fn entry(&self, index: usize) -> &PageTableEntry {
        &self.entries[index]
    }

    /// Get mutable entry at index
    pub fn entry_mut(&mut self, index: usize) -> &mut PageTableEntry {
        &mut self.entries[index]
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            entry.clear();
        }
    }

    /// Get next level table
    pub fn next_table(&self, index: usize) -> Option<&PageTable> {
        let entry = &self.entries[index];
        if entry.is_present() && entry.is_table() {
            let addr = entry.frame()?.start_address();
            Some(unsafe { &*(addr as *const PageTable) })
        } else {
            None
        }
    }

    /// Get next level table (mutable)
    pub fn next_table_mut(&mut self, index: usize) -> Option<&mut PageTable> {
        let entry = &self.entries[index];
        if entry.is_present() && entry.is_table() {
            let addr = entry.frame()?.start_address();
            Some(unsafe { &mut *(addr as *mut PageTable) })
        } else {
            None
        }
    }

    /// Get or create next level table
    pub fn next_table_create(&mut self, index: usize) -> Option<&mut PageTable> {
        if !self.entries[index].is_present() {
            let frame = allocate_frame()?;
            let addr = frame.start_address();

            // Zero out the new table
            unsafe {
                let table = &mut *(addr as *mut PageTable);
                table.clear();
            }

            self.entries[index] = PageTableEntry::new(
                frame,
                PageFlags::PRESENT | PageFlags::WRITABLE,
            );
        }

        self.next_table_mut(index)
    }
}

/// Address space (collection of page tables)
pub struct AddressSpace {
    /// Root page table (L0/PGD)
    root: *mut PageTable,
    /// ASID (Address Space ID) for TLB tagging
    asid: u16,
}

impl AddressSpace {
    /// Create a new address space
    pub fn new() -> Option<Self> {
        let frame = allocate_frame()?;
        let root = frame.start_address() as *mut PageTable;

        unsafe {
            (*root).clear();
        }

        // Allocate an ASID
        static NEXT_ASID: Mutex<u16> = Mutex::new(1);
        let asid = {
            let mut next = NEXT_ASID.lock();
            let asid = *next;
            *next = next.wrapping_add(1);
            if *next == 0 {
                *next = 1; // Skip 0
            }
            asid
        };

        Some(Self { root, asid })
    }

    /// Get root table physical address
    pub fn root_addr(&self) -> usize {
        self.root as usize
    }

    /// Get ASID
    pub fn asid(&self) -> u16 {
        self.asid
    }

    /// Map a virtual page to a physical frame
    pub fn map(
        &mut self,
        page: VirtPage,
        frame: PhysFrame,
        flags: PageFlags,
    ) -> Result<(), &'static str> {
        let virt_addr = page.start_address();

        // Extract page table indices
        let l0_index = (virt_addr >> 39) & 0x1FF;
        let l1_index = (virt_addr >> 30) & 0x1FF;
        let l2_index = (virt_addr >> 21) & 0x1FF;
        let l3_index = (virt_addr >> 12) & 0x1FF;

        let root = unsafe { &mut *self.root };

        // Walk/create page tables
        let l1 = root.next_table_create(l0_index)
            .ok_or("Failed to create L1 table")?;
        let l2 = l1.next_table_create(l1_index)
            .ok_or("Failed to create L2 table")?;
        let l3 = l2.next_table_create(l2_index)
            .ok_or("Failed to create L3 table")?;

        // Set the final entry
        if l3.entries[l3_index].is_present() {
            return Err("Page already mapped");
        }

        l3.entries[l3_index] = PageTableEntry::new(frame, flags);

        Ok(())
    }

    /// Unmap a virtual page
    pub fn unmap(&mut self, page: VirtPage) -> Result<PhysFrame, &'static str> {
        let virt_addr = page.start_address();

        let l0_index = (virt_addr >> 39) & 0x1FF;
        let l1_index = (virt_addr >> 30) & 0x1FF;
        let l2_index = (virt_addr >> 21) & 0x1FF;
        let l3_index = (virt_addr >> 12) & 0x1FF;

        let root = unsafe { &mut *self.root };

        let l1 = root.next_table_mut(l0_index).ok_or("L1 not present")?;
        let l2 = l1.next_table_mut(l1_index).ok_or("L2 not present")?;
        let l3 = l2.next_table_mut(l2_index).ok_or("L3 not present")?;

        let frame = l3.entries[l3_index].frame().ok_or("Page not mapped")?;
        l3.entries[l3_index].clear();

        // Invalidate TLB for this page
        unsafe {
            core::arch::asm!(
                "dsb ishst",
                "tlbi vaae1is, {0}",
                "dsb ish",
                "isb",
                in(reg) virt_addr >> 12,
            );
        }

        Ok(frame)
    }

    /// Translate virtual address to physical
    pub fn translate(&self, virt_addr: usize) -> Option<usize> {
        let l0_index = (virt_addr >> 39) & 0x1FF;
        let l1_index = (virt_addr >> 30) & 0x1FF;
        let l2_index = (virt_addr >> 21) & 0x1FF;
        let l3_index = (virt_addr >> 12) & 0x1FF;
        let offset = virt_addr & 0xFFF;

        let root = unsafe { &*self.root };

        let l1 = root.next_table(l0_index)?;
        let l2 = l1.next_table(l1_index)?;
        let l3 = l2.next_table(l2_index)?;

        let frame = l3.entries[l3_index].frame()?;
        Some(frame.start_address() + offset)
    }

    /// Activate this address space
    pub fn activate(&self) {
        let ttbr0 = self.root as u64 | ((self.asid as u64) << 48);

        unsafe {
            core::arch::asm!(
                "msr ttbr0_el1, {0}",
                "isb",
                in(reg) ttbr0,
            );
        }
    }
}

impl Drop for AddressSpace {
    fn drop(&mut self) {
        // TODO: Free all page tables
        // This is complex as we need to walk and free all levels
    }
}

/// Kernel address space (shared by all processes)
static KERNEL_SPACE: Mutex<Option<AddressSpace>> = Mutex::new(None);

/// Initialize kernel address space
pub fn init_kernel_space() {
    if let Some(space) = AddressSpace::new() {
        *KERNEL_SPACE.lock() = Some(space);
        crate::kprintln!("  Kernel address space initialized");
    }
}

/// Map kernel memory
pub fn kernel_map(virt: usize, phys: usize, size: usize, flags: PageFlags) -> Result<(), &'static str> {
    let mut space = KERNEL_SPACE.lock();
    let space = space.as_mut().ok_or("Kernel space not initialized")?;

    let start_page = VirtPage::containing_address(virt);
    let end_page = VirtPage::containing_address(virt + size - 1);

    for page_num in start_page.0..=end_page.0 {
        let page = VirtPage(page_num);
        let frame_offset = (page_num - start_page.0) * PAGE_SIZE;
        let frame = PhysFrame::containing_address(phys + frame_offset);
        space.map(page, frame, flags)?;
    }

    Ok(())
}
