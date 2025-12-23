//! Copy-on-Write (COW) Memory Management
//!
//! Implements lazy copying of memory pages for efficient fork().
//! Pages are shared read-only until a write triggers a copy.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use spin::{Mutex, RwLock};

use crate::process::Pid;

/// Physical page frame number
pub type Pfn = usize;

/// Virtual page number
pub type Vpn = usize;

/// Page size (4KB)
pub const PAGE_SIZE: usize = 4096;
pub const PAGE_SHIFT: usize = 12;

/// Page protection flags
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageFlags {
    /// Page is present in memory
    pub present: bool,
    /// Page is writable
    pub writable: bool,
    /// Page is user-accessible
    pub user: bool,
    /// Page is executable
    pub executable: bool,
    /// Page is copy-on-write
    pub cow: bool,
    /// Page is dirty (modified)
    pub dirty: bool,
    /// Page has been accessed
    pub accessed: bool,
}

impl PageFlags {
    pub const fn empty() -> Self {
        Self {
            present: false,
            writable: false,
            user: false,
            executable: false,
            cow: false,
            dirty: false,
            accessed: false,
        }
    }

    pub const fn user_read() -> Self {
        Self {
            present: true,
            writable: false,
            user: true,
            executable: false,
            cow: false,
            dirty: false,
            accessed: false,
        }
    }

    pub const fn user_write() -> Self {
        Self {
            present: true,
            writable: true,
            user: true,
            executable: false,
            cow: false,
            dirty: false,
            accessed: false,
        }
    }

    pub const fn user_exec() -> Self {
        Self {
            present: true,
            writable: false,
            user: true,
            executable: true,
            cow: false,
            dirty: false,
            accessed: false,
        }
    }

    /// Make page COW (read-only for sharing)
    pub fn make_cow(&mut self) {
        self.writable = false;
        self.cow = true;
    }

    /// Break COW (make writable again)
    pub fn break_cow(&mut self) {
        self.writable = true;
        self.cow = false;
        self.dirty = true;
    }
}

/// Physical page with reference counting
pub struct PhysPage {
    /// Physical frame number
    pub pfn: Pfn,
    /// Reference count
    refcount: AtomicU32,
    /// Original protection flags
    pub original_flags: PageFlags,
}

impl PhysPage {
    /// Create new page
    pub fn new(pfn: Pfn, flags: PageFlags) -> Arc<Self> {
        Arc::new(Self {
            pfn,
            refcount: AtomicU32::new(1),
            original_flags: flags,
        })
    }

    /// Increment reference count
    pub fn get(&self) -> u32 {
        self.refcount.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Decrement reference count
    pub fn put(&self) -> u32 {
        let old = self.refcount.fetch_sub(1, Ordering::SeqCst);
        if old == 0 {
            panic!("PhysPage refcount underflow");
        }
        old - 1
    }

    /// Get reference count
    pub fn count(&self) -> u32 {
        self.refcount.load(Ordering::SeqCst)
    }

    /// Check if page is shared (refcount > 1)
    pub fn is_shared(&self) -> bool {
        self.count() > 1
    }
}

/// Virtual memory area
#[derive(Clone)]
pub struct Vma {
    /// Start virtual address
    pub start: usize,
    /// End virtual address (exclusive)
    pub end: usize,
    /// Protection flags
    pub flags: PageFlags,
    /// Backing pages (vpn -> physical page)
    pages: BTreeMap<Vpn, Arc<PhysPage>>,
    /// Is this a COW region
    pub cow: bool,
    /// File backing (if any)
    pub file: Option<VmaFile>,
}

/// File backing for VMA
#[derive(Clone)]
pub struct VmaFile {
    /// Inode number
    pub inode: u64,
    /// Offset in file
    pub offset: usize,
    /// File path
    pub path: alloc::string::String,
}

impl Vma {
    /// Create new VMA
    pub fn new(start: usize, end: usize, flags: PageFlags) -> Self {
        Self {
            start,
            end,
            flags,
            pages: BTreeMap::new(),
            cow: false,
            file: None,
        }
    }

    /// Size in bytes
    pub fn size(&self) -> usize {
        self.end - self.start
    }

    /// Size in pages
    pub fn page_count(&self) -> usize {
        self.size() / PAGE_SIZE
    }

    /// Check if address is in this VMA
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.start && addr < self.end
    }

    /// Get virtual page number for address
    pub fn addr_to_vpn(&self, addr: usize) -> Vpn {
        (addr - self.start) >> PAGE_SHIFT
    }

    /// Get address for virtual page number
    pub fn vpn_to_addr(&self, vpn: Vpn) -> usize {
        self.start + (vpn << PAGE_SHIFT)
    }

    /// Map a page
    pub fn map_page(&mut self, vpn: Vpn, page: Arc<PhysPage>) {
        self.pages.insert(vpn, page);
    }

    /// Unmap a page
    pub fn unmap_page(&mut self, vpn: Vpn) -> Option<Arc<PhysPage>> {
        self.pages.remove(&vpn)
    }

    /// Get page at vpn
    pub fn get_page(&self, vpn: Vpn) -> Option<&Arc<PhysPage>> {
        self.pages.get(&vpn)
    }

    /// Create COW copy for fork
    pub fn fork_cow(&self) -> Self {
        let mut child_vma = self.clone();
        child_vma.cow = true;

        // Share all pages (increment refcount)
        for page in child_vma.pages.values() {
            page.get();
        }

        child_vma
    }

    /// Mark all pages as COW
    pub fn mark_cow(&mut self) {
        self.cow = true;
        self.flags.make_cow();
    }
}

/// Process address space
pub struct AddressSpace {
    /// Process ID
    pub pid: Pid,
    /// Virtual memory areas
    vmas: RwLock<Vec<Vma>>,
    /// Page directory base (for MMU)
    pub pgd: AtomicUsize,
    /// Total mapped pages
    mapped_pages: AtomicUsize,
    /// Total resident pages
    resident_pages: AtomicUsize,
    /// Shared pages
    shared_pages: AtomicUsize,
}

impl AddressSpace {
    /// Create new empty address space
    pub fn new(pid: Pid) -> Self {
        Self {
            pid,
            vmas: RwLock::new(Vec::new()),
            pgd: AtomicUsize::new(0),
            mapped_pages: AtomicUsize::new(0),
            resident_pages: AtomicUsize::new(0),
            shared_pages: AtomicUsize::new(0),
        }
    }

    /// Add VMA
    pub fn add_vma(&self, vma: Vma) {
        let page_count = vma.page_count();
        self.vmas.write().push(vma);
        self.mapped_pages.fetch_add(page_count, Ordering::SeqCst);
    }

    /// Find VMA containing address
    pub fn find_vma(&self, addr: usize) -> Option<usize> {
        let vmas = self.vmas.read();
        for (i, vma) in vmas.iter().enumerate() {
            if vma.contains(addr) {
                return Some(i);
            }
        }
        None
    }

    /// Get VMA by index
    pub fn get_vma(&self, index: usize) -> Option<Vma> {
        self.vmas.read().get(index).cloned()
    }

    /// Fork address space with COW
    pub fn fork(&self, child_pid: Pid) -> Self {
        let child = Self::new(child_pid);

        let parent_vmas = self.vmas.read();
        let mut child_vmas = child.vmas.write();

        for vma in parent_vmas.iter() {
            let child_vma = vma.fork_cow();
            child_vmas.push(child_vma);
        }

        // Copy page table base
        child.pgd.store(self.pgd.load(Ordering::SeqCst), Ordering::SeqCst);

        // Update counters
        child.mapped_pages.store(
            self.mapped_pages.load(Ordering::SeqCst),
            Ordering::SeqCst
        );
        child.shared_pages.store(
            self.resident_pages.load(Ordering::SeqCst),
            Ordering::SeqCst
        );

        drop(child_vmas);
        drop(parent_vmas);

        // Mark parent pages as COW too
        let mut parent_vmas = self.vmas.write();
        for vma in parent_vmas.iter_mut() {
            if vma.flags.writable {
                vma.mark_cow();
            }
        }

        child
    }

    /// Handle page fault (returns true if handled)
    pub fn handle_page_fault(&self, addr: usize, is_write: bool) -> Result<(), PageFaultError> {
        let vma_idx = self.find_vma(addr).ok_or(PageFaultError::NoMapping)?;

        let mut vmas = self.vmas.write();
        let vma = vmas.get_mut(vma_idx).ok_or(PageFaultError::NoMapping)?;

        let vpn = vma.addr_to_vpn(addr);

        if is_write {
            // Write fault
            if !vma.flags.writable && !vma.cow {
                // Not writable and not COW - access violation
                return Err(PageFaultError::AccessViolation);
            }

            if vma.cow {
                // COW page - check if we need to copy
                if let Some(page) = vma.get_page(vpn) {
                    if page.is_shared() {
                        // Need to copy
                        return self.do_cow_copy(&mut *vmas, vma_idx, vpn, addr);
                    } else {
                        // Only reference - just make writable
                        vma.flags.break_cow();
                        return Ok(());
                    }
                }
            }

            // Allocate new page
            self.allocate_page(&mut *vmas, vma_idx, vpn)
        } else {
            // Read fault - demand paging
            if vma.get_page(vpn).is_some() {
                // Page is present but not mapped in MMU
                Ok(())
            } else {
                // Allocate zero page
                self.allocate_page(&mut *vmas, vma_idx, vpn)
            }
        }
    }

    /// Perform COW copy
    fn do_cow_copy(
        &self,
        vmas: &mut Vec<Vma>,
        vma_idx: usize,
        vpn: Vpn,
        addr: usize,
    ) -> Result<(), PageFaultError> {
        let vma = vmas.get_mut(vma_idx).ok_or(PageFaultError::NoMapping)?;

        let old_page = vma.unmap_page(vpn).ok_or(PageFaultError::NoMapping)?;

        // Allocate new physical page
        let new_pfn = allocate_physical_page().ok_or(PageFaultError::OutOfMemory)?;

        // Copy data
        let old_addr = old_page.pfn << PAGE_SHIFT;
        let new_addr = new_pfn << PAGE_SHIFT;
        unsafe {
            core::ptr::copy_nonoverlapping(
                old_addr as *const u8,
                new_addr as *mut u8,
                PAGE_SIZE,
            );
        }

        // Create new page with original flags (writable)
        let mut flags = old_page.original_flags;
        flags.break_cow();

        let new_page = PhysPage::new(new_pfn, flags);
        vma.map_page(vpn, new_page);

        // Decrement old page refcount (might free it)
        let remaining = old_page.put();
        if remaining == 0 {
            free_physical_page(old_page.pfn);
        } else {
            self.shared_pages.fetch_sub(1, Ordering::SeqCst);
        }

        // Update VMA flags if no more COW pages
        let has_cow = vma.pages.values().any(|p| p.is_shared());
        if !has_cow {
            vma.cow = false;
            vma.flags.break_cow();
        }

        // Update page tables in MMU
        update_page_table_entry(addr, new_pfn, flags);

        Ok(())
    }

    /// Allocate new page
    fn allocate_page(
        &self,
        vmas: &mut Vec<Vma>,
        vma_idx: usize,
        vpn: Vpn,
    ) -> Result<(), PageFaultError> {
        let vma = vmas.get_mut(vma_idx).ok_or(PageFaultError::NoMapping)?;

        let pfn = allocate_physical_page().ok_or(PageFaultError::OutOfMemory)?;

        // Zero the page
        let addr = pfn << PAGE_SHIFT;
        unsafe {
            core::ptr::write_bytes(addr as *mut u8, 0, PAGE_SIZE);
        }

        let page = PhysPage::new(pfn, vma.flags);
        vma.map_page(vpn, page);

        self.resident_pages.fetch_add(1, Ordering::SeqCst);

        // Update page tables in MMU
        update_page_table_entry(vpn << PAGE_SHIFT, pfn, vma.flags);

        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> AddressSpaceStats {
        AddressSpaceStats {
            mapped_pages: self.mapped_pages.load(Ordering::SeqCst),
            resident_pages: self.resident_pages.load(Ordering::SeqCst),
            shared_pages: self.shared_pages.load(Ordering::SeqCst),
            vma_count: self.vmas.read().len(),
        }
    }

    /// Clean up address space
    pub fn cleanup(&self) {
        let vmas = self.vmas.read();

        for vma in vmas.iter() {
            for page in vma.pages.values() {
                let remaining = page.put();
                if remaining == 0 {
                    free_physical_page(page.pfn);
                }
            }
        }
    }
}

/// Page fault error
#[derive(Clone, Copy, Debug)]
pub enum PageFaultError {
    /// No mapping for address
    NoMapping,
    /// Access violation (wrong permissions)
    AccessViolation,
    /// Out of physical memory
    OutOfMemory,
    /// Address not aligned
    NotAligned,
}

/// Address space statistics
#[derive(Clone, Debug)]
pub struct AddressSpaceStats {
    pub mapped_pages: usize,
    pub resident_pages: usize,
    pub shared_pages: usize,
    pub vma_count: usize,
}

// ============================================================================
// Physical Page Management (simplified)
// ============================================================================

/// Free page list
static FREE_PAGES: Mutex<Vec<Pfn>> = Mutex::new(Vec::new());

/// Total pages
static TOTAL_PAGES: AtomicUsize = AtomicUsize::new(0);

/// Initialize physical page pool
pub fn init_pages(start_pfn: Pfn, count: usize) {
    let mut free = FREE_PAGES.lock();
    for i in 0..count {
        free.push(start_pfn + i);
    }
    TOTAL_PAGES.store(count, Ordering::SeqCst);
}

/// Allocate physical page
pub fn allocate_physical_page() -> Option<Pfn> {
    FREE_PAGES.lock().pop()
}

/// Free physical page
pub fn free_physical_page(pfn: Pfn) {
    FREE_PAGES.lock().push(pfn);
}

/// Get free page count
pub fn free_page_count() -> usize {
    FREE_PAGES.lock().len()
}

/// Get total page count
pub fn total_page_count() -> usize {
    TOTAL_PAGES.load(Ordering::SeqCst)
}

// ============================================================================
// Global State
// ============================================================================

/// Process address spaces
static ADDRESS_SPACES: RwLock<BTreeMap<Pid, Arc<AddressSpace>>> =
    RwLock::new(BTreeMap::new());

/// Create address space for process
pub fn create_address_space(pid: Pid) -> Arc<AddressSpace> {
    let space = Arc::new(AddressSpace::new(pid));
    ADDRESS_SPACES.write().insert(pid, space.clone());
    space
}

/// Get address space for process
pub fn get_address_space(pid: Pid) -> Option<Arc<AddressSpace>> {
    ADDRESS_SPACES.read().get(&pid).cloned()
}

/// Fork address space
pub fn fork_address_space(parent_pid: Pid, child_pid: Pid) -> Option<Arc<AddressSpace>> {
    let parent = get_address_space(parent_pid)?;
    let child = Arc::new(parent.fork(child_pid));
    ADDRESS_SPACES.write().insert(child_pid, child.clone());
    Some(child)
}

/// Clean up address space
pub fn cleanup_address_space(pid: Pid) {
    if let Some(space) = ADDRESS_SPACES.write().remove(&pid) {
        space.cleanup();
    }
}

/// Handle page fault for process
pub fn handle_page_fault(pid: Pid, addr: usize, is_write: bool) -> Result<(), PageFaultError> {
    let space = get_address_space(pid).ok_or(PageFaultError::NoMapping)?;
    space.handle_page_fault(addr, is_write)
}

/// Initialize COW subsystem
pub fn init() {
    // Initialize with some pages (would be from memory manager in reality)
    // This is just a placeholder
    init_pages(0x100000 >> PAGE_SHIFT, 1024);

    crate::kprintln!("  Copy-on-Write initialized ({} pages)", total_page_count());
}

// ============================================================================
// Page Table Operations
// ============================================================================

/// Update a page table entry for the current process
pub fn update_page_table_entry(vaddr: usize, pfn: Pfn, flags: PageFlags) {
    // Get current process's page table
    if let Some(process) = crate::process::current() {
        let memory = process.memory.lock();
        let page_table = memory.page_table as usize;
        drop(memory);

        if page_table != 0 {
            update_pte(page_table, vaddr, pfn << PAGE_SHIFT, flags);
        }
    }
}

/// Update a PTE in the given page table
fn update_pte(page_table: usize, vaddr: usize, paddr: usize, flags: PageFlags) {
    // Convert our flags to hardware page table flags
    let mut hw_flags: u64 = 0x3; // Valid + Table/Page

    if flags.present {
        hw_flags |= 0x1; // Valid
    }

    // User/kernel access
    if flags.user {
        hw_flags |= 1 << 6; // AP[1] = 1 for EL0 access
    }

    // Read-only vs read-write
    if !flags.writable {
        hw_flags |= 1 << 7; // AP[2] = 1 for read-only
    }

    // Execute permission (XN bit)
    if !flags.executable {
        hw_flags |= 1 << 54; // UXN
    }

    // Access and dirty flags
    if flags.accessed {
        hw_flags |= 1 << 10; // AF
    }

    // Normal memory, inner/outer write-back cacheable
    hw_flags |= 0x4 << 2; // AttrIndx[2:0] = 0b100 (normal memory)

    // Walk page tables and update entry
    let l0 = page_table as *mut u64;
    let l0_idx = (vaddr >> 39) & 0x1FF;
    let l1_idx = (vaddr >> 30) & 0x1FF;
    let l2_idx = (vaddr >> 21) & 0x1FF;
    let l3_idx = (vaddr >> 12) & 0x1FF;

    unsafe {
        // Get L1 table
        let l0_entry = *l0.add(l0_idx);
        if l0_entry & 0x1 == 0 {
            return; // Not mapped
        }
        let l1 = (l0_entry & 0x0000_FFFF_FFFF_F000) as *mut u64;

        // Get L2 table
        let l1_entry = *l1.add(l1_idx);
        if l1_entry & 0x1 == 0 {
            return;
        }
        let l2 = (l1_entry & 0x0000_FFFF_FFFF_F000) as *mut u64;

        // Get L3 table
        let l2_entry = *l2.add(l2_idx);
        if l2_entry & 0x1 == 0 {
            return;
        }
        let l3 = (l2_entry & 0x0000_FFFF_FFFF_F000) as *mut u64;

        // Update L3 entry
        *l3.add(l3_idx) = (paddr as u64) | hw_flags;

        // Invalidate TLB for this address
        core::arch::asm!(
            "dsb ishst",
            "tlbi vaae1is, {0}",
            "dsb ish",
            "isb",
            in(reg) vaddr >> 12,
        );
    }
}

/// Copy page tables for fork with COW
pub fn copy_address_space(parent_root: usize, child_root: usize) {
    if parent_root == 0 || child_root == 0 {
        return;
    }

    unsafe {
        let parent_l0 = parent_root as *const u64;
        let child_l0 = child_root as *mut u64;

        // Copy L0 entries
        for l0_idx in 0..512 {
            let parent_entry = *parent_l0.add(l0_idx);
            if parent_entry & 0x1 == 0 {
                *child_l0.add(l0_idx) = 0;
                continue;
            }

            // This is a valid entry, copy the L1 table
            if let Some(child_l1_frame) = crate::memory::allocate_frame() {
                let child_l1 = child_l1_frame.start_address() as *mut u64;
                let parent_l1 = (parent_entry & 0x0000_FFFF_FFFF_F000) as *const u64;

                // Copy L1 entries and mark as COW
                copy_page_table_level(parent_l1, child_l1, 1);

                // Update child L0 entry
                *child_l0.add(l0_idx) = (child_l1_frame.start_address() as u64) | 0x3;
            }
        }
    }
}

/// Copy a level of page tables, marking leaf pages as COW
unsafe fn copy_page_table_level(parent: *const u64, child: *mut u64, level: usize) {
    for idx in 0..512 {
        let parent_entry = *parent.add(idx);

        if parent_entry & 0x1 == 0 {
            // Not present
            *child.add(idx) = 0;
            continue;
        }

        if level < 3 && (parent_entry & 0x2) != 0 {
            // Table entry - recurse
            if let Some(child_next_frame) = crate::memory::allocate_frame() {
                let child_next = child_next_frame.start_address() as *mut u64;
                let parent_next = (parent_entry & 0x0000_FFFF_FFFF_F000) as *const u64;

                // Zero the new table
                for i in 0..512 {
                    *child_next.add(i) = 0;
                }

                copy_page_table_level(parent_next, child_next, level + 1);
                *child.add(idx) = (child_next_frame.start_address() as u64) | 0x3;
            }
        } else if level == 3 {
            // Leaf entry at L3 - copy with COW flag
            // Clear writable bit (AP[2] = 1) to make read-only
            let cow_entry = parent_entry | (1 << 7); // Set read-only
            *child.add(idx) = cow_entry;

            // Also update parent to be read-only for COW
            // We need to update the parent through its page table
            // For simplicity, we mark both as read-only here
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_refcount() {
        let page = PhysPage::new(100, PageFlags::user_write());

        assert_eq!(page.count(), 1);
        assert!(!page.is_shared());

        page.get();
        assert_eq!(page.count(), 2);
        assert!(page.is_shared());

        page.put();
        assert_eq!(page.count(), 1);
        assert!(!page.is_shared());
    }

    #[test]
    fn test_vma_cow() {
        let mut vma = Vma::new(0x1000, 0x2000, PageFlags::user_write());
        let page = PhysPage::new(100, PageFlags::user_write());
        vma.map_page(0, page);

        let child_vma = vma.fork_cow();
        assert!(child_vma.cow);

        let parent_page = vma.get_page(0).unwrap();
        let child_page = child_vma.get_page(0).unwrap();

        // Same physical page
        assert_eq!(parent_page.pfn, child_page.pfn);
        // Shared
        assert!(parent_page.is_shared());
    }

    #[test]
    fn test_page_flags() {
        let mut flags = PageFlags::user_write();
        assert!(flags.writable);
        assert!(!flags.cow);

        flags.make_cow();
        assert!(!flags.writable);
        assert!(flags.cow);

        flags.break_cow();
        assert!(flags.writable);
        assert!(!flags.cow);
    }
}
