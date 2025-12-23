//! Page Fault Handler
//!
//! Handles memory access exceptions including:
//! - Demand paging (lazy allocation)
//! - Copy-on-write
//! - Stack growth
//! - Swap-in from disk
//! - Memory-mapped file access

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::RwLock;

use crate::process::Pid;

/// Page fault statistics
pub struct PageFaultStats {
    /// Total page faults
    pub total: AtomicU64,
    /// Minor faults (no I/O needed)
    pub minor: AtomicU64,
    /// Major faults (I/O needed, e.g., swap)
    pub major: AtomicU64,
    /// COW faults
    pub cow: AtomicU64,
    /// Stack expansion faults
    pub stack_grow: AtomicU64,
    /// Protection faults (segfaults)
    pub protection: AtomicU64,
}

impl PageFaultStats {
    pub const fn new() -> Self {
        Self {
            total: AtomicU64::new(0),
            minor: AtomicU64::new(0),
            major: AtomicU64::new(0),
            cow: AtomicU64::new(0),
            stack_grow: AtomicU64::new(0),
            protection: AtomicU64::new(0),
        }
    }
}

/// Global page fault statistics
static STATS: PageFaultStats = PageFaultStats::new();

/// Page fault type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultType {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Instruction fetch
    Execute,
}

/// Page fault cause
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultCause {
    /// Page not present
    NotPresent,
    /// Protection violation
    Protection,
    /// Reserved bit set
    Reserved,
    /// Instruction fetch from non-executable page
    InstructionFetch,
}

/// Page fault context
#[derive(Clone, Debug)]
pub struct PageFaultContext {
    /// Faulting address
    pub address: usize,
    /// Instruction pointer when fault occurred
    pub ip: usize,
    /// Fault type (read/write/execute)
    pub fault_type: FaultType,
    /// Fault cause
    pub cause: FaultCause,
    /// User mode fault
    pub user_mode: bool,
    /// Process ID
    pub pid: Pid,
}

/// Page fault result
#[derive(Clone, Copy, Debug)]
pub enum FaultResult {
    /// Fault handled, continue execution
    Handled,
    /// Need to retry after handling
    Retry,
    /// Fatal fault - send signal
    Signal(FaultSignal),
    /// Kernel bug - panic
    KernelBug,
}

/// Signal to send for fatal fault
#[derive(Clone, Copy, Debug)]
pub enum FaultSignal {
    /// SIGSEGV - segmentation fault
    Segv,
    /// SIGBUS - bus error
    Bus,
    /// SIGKILL - unrecoverable
    Kill,
}

/// Handle page fault
pub fn handle_page_fault(ctx: &PageFaultContext) -> FaultResult {
    STATS.total.fetch_add(1, Ordering::Relaxed);

    crate::kdebug!(
        "Page fault: addr=0x{:x} ip=0x{:x} type={:?} cause={:?} user={}",
        ctx.address, ctx.ip, ctx.fault_type, ctx.cause, ctx.user_mode
    );

    // Get process address space
    let space = match crate::cow::get_address_space(ctx.pid) {
        Some(s) => s,
        None => {
            if ctx.user_mode {
                return FaultResult::Signal(FaultSignal::Segv);
            } else {
                return FaultResult::KernelBug;
            }
        }
    };

    // Find VMA containing the faulting address
    let vma_idx = match space.find_vma(ctx.address) {
        Some(idx) => idx,
        None => {
            // Check if this might be stack growth
            if try_stack_grow(&space, ctx.address) {
                STATS.stack_grow.fetch_add(1, Ordering::Relaxed);
                return FaultResult::Retry;
            }

            STATS.protection.fetch_add(1, Ordering::Relaxed);
            return FaultResult::Signal(FaultSignal::Segv);
        }
    };

    let vma = match space.get_vma(vma_idx) {
        Some(v) => v,
        None => return FaultResult::Signal(FaultSignal::Segv),
    };

    // Check permissions
    match ctx.fault_type {
        FaultType::Read => {
            if !vma.flags.present {
                // Demand paging
                return handle_demand_paging(&space, ctx, &vma);
            }
        }
        FaultType::Write => {
            if vma.cow {
                // Copy-on-write
                STATS.cow.fetch_add(1, Ordering::Relaxed);
                return handle_cow_fault(&space, ctx);
            }
            if !vma.flags.writable {
                STATS.protection.fetch_add(1, Ordering::Relaxed);
                return FaultResult::Signal(FaultSignal::Segv);
            }
        }
        FaultType::Execute => {
            if !vma.flags.executable {
                STATS.protection.fetch_add(1, Ordering::Relaxed);
                return FaultResult::Signal(FaultSignal::Segv);
            }
        }
    }

    // Try to handle the fault
    match space.handle_page_fault(ctx.address, ctx.fault_type == FaultType::Write) {
        Ok(()) => {
            STATS.minor.fetch_add(1, Ordering::Relaxed);
            FaultResult::Handled
        }
        Err(crate::cow::PageFaultError::OutOfMemory) => {
            // Try OOM killer
            if crate::oom::invoke_oom_killer() {
                FaultResult::Retry
            } else {
                FaultResult::Signal(FaultSignal::Kill)
            }
        }
        Err(crate::cow::PageFaultError::AccessViolation) => {
            STATS.protection.fetch_add(1, Ordering::Relaxed);
            FaultResult::Signal(FaultSignal::Segv)
        }
        Err(_) => FaultResult::Signal(FaultSignal::Segv),
    }
}

/// Handle demand paging (lazy allocation)
fn handle_demand_paging(
    space: &crate::cow::AddressSpace,
    ctx: &PageFaultContext,
    vma: &crate::cow::Vma,
) -> FaultResult {
    // Check if this is a file-backed mapping
    if let Some(ref file) = vma.file {
        // Need to read from file - major fault
        STATS.major.fetch_add(1, Ordering::Relaxed);
        return handle_file_fault(space, ctx, vma, file);
    }

    // Check if page is in swap
    if let Some(swap_entry) = check_swap(ctx.pid, ctx.address) {
        // Need to swap in - major fault
        STATS.major.fetch_add(1, Ordering::Relaxed);
        return handle_swap_in(space, ctx, swap_entry);
    }

    // Anonymous page - allocate zero page
    match space.handle_page_fault(ctx.address, false) {
        Ok(()) => {
            STATS.minor.fetch_add(1, Ordering::Relaxed);
            FaultResult::Handled
        }
        Err(_) => FaultResult::Signal(FaultSignal::Segv),
    }
}

/// Handle copy-on-write fault
fn handle_cow_fault(
    space: &crate::cow::AddressSpace,
    ctx: &PageFaultContext,
) -> FaultResult {
    match space.handle_page_fault(ctx.address, true) {
        Ok(()) => FaultResult::Handled,
        Err(crate::cow::PageFaultError::OutOfMemory) => {
            if crate::oom::invoke_oom_killer() {
                FaultResult::Retry
            } else {
                FaultResult::Signal(FaultSignal::Kill)
            }
        }
        Err(_) => FaultResult::Signal(FaultSignal::Segv),
    }
}

/// Try to grow stack
fn try_stack_grow(space: &crate::cow::AddressSpace, addr: usize) -> bool {
    // Check if address is just below current stack
    // Stack grows down on most architectures

    // Get memory layout
    // For now, just check if it's in a reasonable stack range
    let stack_max = 0x7FFF_FFFF_0000usize;
    let stack_limit = 0x7FFF_0000_0000usize; // Max 256MB stack

    if addr >= stack_limit && addr < stack_max {
        // Could be stack growth
        // In reality, we'd check against the stack VMA and grow it
        crate::kdebug!("Stack growth attempt at 0x{:x}", addr);

        // Allocate the page
        match space.handle_page_fault(addr, true) {
            Ok(()) => true,
            Err(_) => false,
        }
    } else {
        false
    }
}

/// Handle file-backed page fault
fn handle_file_fault(
    _space: &crate::cow::AddressSpace,
    ctx: &PageFaultContext,
    vma: &crate::cow::Vma,
    file: &crate::cow::VmaFile,
) -> FaultResult {
    crate::kdebug!(
        "File fault: addr=0x{:x} file={} offset=0x{:x}",
        ctx.address, file.path, file.offset
    );

    // Calculate offset in file
    let page_offset = (ctx.address - vma.start) & !(crate::cow::PAGE_SIZE - 1);
    let file_offset = file.offset + page_offset;

    // Read page from file
    // This would involve the VFS and possibly block I/O
    // For now, just allocate a zero page

    // TODO: Read from file into page

    FaultResult::Handled
}

// ============================================================================
// Swap Support
// ============================================================================

/// Swap entry (location in swap device)
#[derive(Clone, Copy, Debug)]
pub struct SwapEntry {
    /// Swap device index
    pub dev: u8,
    /// Offset in swap device (in pages)
    pub offset: u64,
}

/// Swap cache
static SWAP_CACHE: RwLock<alloc::collections::BTreeMap<(Pid, usize), SwapEntry>> =
    RwLock::new(alloc::collections::BTreeMap::new());

/// Check if page is in swap
fn check_swap(pid: Pid, addr: usize) -> Option<SwapEntry> {
    let page_addr = addr & !(crate::cow::PAGE_SIZE - 1);
    SWAP_CACHE.read().get(&(pid, page_addr)).copied()
}

/// Handle swap-in
fn handle_swap_in(
    _space: &crate::cow::AddressSpace,
    ctx: &PageFaultContext,
    entry: SwapEntry,
) -> FaultResult {
    crate::kdebug!(
        "Swap in: addr=0x{:x} dev={} offset={}",
        ctx.address, entry.dev, entry.offset
    );

    // 1. Allocate physical page
    let pfn = match crate::cow::allocate_physical_page() {
        Some(p) => p,
        None => {
            // Try reclaim
            if crate::oom::invoke_oom_killer() {
                return FaultResult::Retry;
            }
            return FaultResult::Signal(FaultSignal::Kill);
        }
    };

    // 2. Read from swap device
    // This would involve block I/O
    // For now, just zero the page
    let addr = pfn << crate::cow::PAGE_SHIFT;
    unsafe {
        core::ptr::write_bytes(addr as *mut u8, 0, crate::cow::PAGE_SIZE);
    }

    // 3. Remove from swap cache
    let page_addr = ctx.address & !(crate::cow::PAGE_SIZE - 1);
    SWAP_CACHE.write().remove(&(ctx.pid, page_addr));

    // 4. Update page tables
    // TODO: Map the page

    FaultResult::Handled
}

/// Swap out a page
pub fn swap_out(pid: Pid, addr: usize) -> Result<SwapEntry, &'static str> {
    // 1. Find free swap slot
    let entry = allocate_swap_slot()?;

    // 2. Write page to swap device
    // TODO: Actual I/O

    // 3. Add to swap cache
    let page_addr = addr & !(crate::cow::PAGE_SIZE - 1);
    SWAP_CACHE.write().insert((pid, page_addr), entry);

    // 4. Free physical page
    // TODO: Unmap and free

    Ok(entry)
}

/// Swap slot allocator (simplified)
static NEXT_SWAP_SLOT: AtomicU64 = AtomicU64::new(0);
static SWAP_SIZE: AtomicU64 = AtomicU64::new(0);

/// Initialize swap device
pub fn init_swap(size_pages: u64) {
    SWAP_SIZE.store(size_pages, Ordering::SeqCst);
    NEXT_SWAP_SLOT.store(0, Ordering::SeqCst);
}

/// Allocate swap slot
fn allocate_swap_slot() -> Result<SwapEntry, &'static str> {
    let size = SWAP_SIZE.load(Ordering::SeqCst);
    if size == 0 {
        return Err("No swap configured");
    }

    let offset = NEXT_SWAP_SLOT.fetch_add(1, Ordering::SeqCst);
    if offset >= size {
        return Err("Swap full");
    }

    Ok(SwapEntry { dev: 0, offset })
}

// ============================================================================
// Architecture-specific interface
// ============================================================================

/// Parse architecture-specific fault info (AArch64)
#[cfg(target_arch = "aarch64")]
pub fn parse_fault_info(esr: u64, far: u64, elr: u64, user: bool, pid: Pid) -> PageFaultContext {
    // ESR_EL1 decoding for data/instruction aborts
    let ec = (esr >> 26) & 0x3F;
    let iss = esr & 0x1FFFFFF;

    let (fault_type, cause) = match ec {
        // Instruction abort
        0x20 | 0x21 => {
            (FaultType::Execute, parse_abort_cause(iss))
        }
        // Data abort
        0x24 | 0x25 => {
            let is_write = (iss & (1 << 6)) != 0;
            let fault_type = if is_write { FaultType::Write } else { FaultType::Read };
            (fault_type, parse_abort_cause(iss))
        }
        _ => (FaultType::Read, FaultCause::Protection),
    };

    PageFaultContext {
        address: far as usize,
        ip: elr as usize,
        fault_type,
        cause,
        user_mode: user,
        pid,
    }
}

#[cfg(target_arch = "aarch64")]
fn parse_abort_cause(iss: u64) -> FaultCause {
    let dfsc = iss & 0x3F;

    match dfsc {
        // Translation fault
        0b000100..=0b000111 => FaultCause::NotPresent,
        // Access flag fault
        0b001000..=0b001011 => FaultCause::NotPresent,
        // Permission fault
        0b001100..=0b001111 => FaultCause::Protection,
        _ => FaultCause::Protection,
    }
}

/// Fallback for other architectures
#[cfg(not(target_arch = "aarch64"))]
pub fn parse_fault_info(_esr: u64, far: u64, elr: u64, user: bool, pid: Pid) -> PageFaultContext {
    PageFaultContext {
        address: far as usize,
        ip: elr as usize,
        fault_type: FaultType::Read,
        cause: FaultCause::NotPresent,
        user_mode: user,
        pid,
    }
}

/// Get page fault statistics
pub fn get_stats() -> (u64, u64, u64, u64, u64, u64) {
    (
        STATS.total.load(Ordering::Relaxed),
        STATS.minor.load(Ordering::Relaxed),
        STATS.major.load(Ordering::Relaxed),
        STATS.cow.load(Ordering::Relaxed),
        STATS.stack_grow.load(Ordering::Relaxed),
        STATS.protection.load(Ordering::Relaxed),
    )
}

/// Generate /proc/vmstat content
pub fn generate_vmstat() -> alloc::string::String {
    let (total, minor, major, cow, stack, prot) = get_stats();

    alloc::format!(
        "pgfault {}\n\
         pgmajfault {}\n\
         pgminorfault {}\n\
         cow_faults {}\n\
         stack_faults {}\n\
         protection_faults {}\n",
        total, major, minor, cow, stack, prot
    )
}

/// Initialize page fault handler
pub fn init() {
    crate::kprintln!("  Page fault handler initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_context() {
        let ctx = PageFaultContext {
            address: 0x1000,
            ip: 0x400000,
            fault_type: FaultType::Read,
            cause: FaultCause::NotPresent,
            user_mode: true,
            pid: Pid(100),
        };

        assert_eq!(ctx.address, 0x1000);
        assert_eq!(ctx.fault_type, FaultType::Read);
    }
}
