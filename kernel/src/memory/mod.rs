//! Memory Management Subsystem
//!
//! Provides physical and virtual memory management for the kernel.
//! Supports ARM64 and RISC-V memory models with 4KB/16KB/64KB pages.

use alloc::vec::Vec;
use bitflags::bitflags;
use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;

use crate::{MemoryKind, MemoryMap, MemoryRegion};

pub mod allocator;
pub mod heap;
pub mod paging;

/// Kernel heap start address (16 MB mark)
pub const KERNEL_HEAP_START: usize = 0x1000000;

/// Kernel heap size (64 MB)
pub const KERNEL_HEAP_SIZE: usize = 64 * 1024 * 1024;

/// Page size (4 KB default)
pub const PAGE_SIZE: usize = 4096;

/// Page shift for calculations
pub const PAGE_SHIFT: usize = 12;

/// Physical frame allocator
static FRAME_ALLOCATOR: Mutex<Option<FrameAllocator>> = Mutex::new(None);

/// Total available physical memory
static TOTAL_MEMORY: AtomicUsize = AtomicUsize::new(0);

/// Used physical memory
static USED_MEMORY: AtomicUsize = AtomicUsize::new(0);

bitflags! {
    /// Page table entry flags
    #[derive(Clone, Copy, Debug)]
    pub struct PageFlags: u64 {
        /// Page is present/valid
        const PRESENT = 1 << 0;
        /// Page is writable
        const WRITABLE = 1 << 1;
        /// Page is accessible from user mode
        const USER = 1 << 2;
        /// Page write-through caching
        const WRITE_THROUGH = 1 << 3;
        /// Page cache disabled
        const NO_CACHE = 1 << 4;
        /// Page was accessed
        const ACCESSED = 1 << 5;
        /// Page was modified
        const DIRTY = 1 << 6;
        /// Huge page (2MB or 1GB)
        const HUGE = 1 << 7;
        /// Global page
        const GLOBAL = 1 << 8;
        /// No execute
        const NO_EXECUTE = 1 << 63;
    }
}

/// Physical frame number
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhysFrame(pub usize);

impl PhysFrame {
    /// Create from physical address
    pub fn containing_address(addr: usize) -> Self {
        Self(addr / PAGE_SIZE)
    }

    /// Get physical address of frame start
    pub fn start_address(&self) -> usize {
        self.0 * PAGE_SIZE
    }
}

/// Virtual page number
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VirtPage(pub usize);

impl VirtPage {
    /// Create from virtual address
    pub fn containing_address(addr: usize) -> Self {
        Self(addr / PAGE_SIZE)
    }

    /// Get virtual address of page start
    pub fn start_address(&self) -> usize {
        self.0 * PAGE_SIZE
    }
}

/// Physical frame allocator using buddy system
pub struct FrameAllocator {
    /// Free lists for each order (0 = 4KB, 1 = 8KB, ..., 10 = 4MB)
    free_lists: [Vec<PhysFrame>; 11],
    /// Memory regions from boot info
    regions: Vec<MemoryRegion>,
}

impl FrameAllocator {
    /// Create a new frame allocator from memory map
    pub fn new(memory_map: &MemoryMap) -> Self {
        let mut allocator = Self {
            free_lists: Default::default(),
            regions: Vec::new(),
        };

        let mut total = 0usize;

        // Add available memory regions
        for region in memory_map.entries {
            if region.kind == MemoryKind::Available {
                allocator.regions.push(*region);
                total += region.size;

                // Add frames to free list
                let start_frame = (region.base + PAGE_SIZE - 1) / PAGE_SIZE;
                let end_frame = (region.base + region.size) / PAGE_SIZE;

                for frame in start_frame..end_frame {
                    allocator.free_lists[0].push(PhysFrame(frame));
                }
            }
        }

        TOTAL_MEMORY.store(total, Ordering::Release);

        // Coalesce frames into larger blocks
        allocator.coalesce_all();

        allocator
    }

    /// Allocate a physical frame
    pub fn allocate(&mut self) -> Option<PhysFrame> {
        self.allocate_order(0)
    }

    /// Allocate frames of given order (2^order pages)
    pub fn allocate_order(&mut self, order: usize) -> Option<PhysFrame> {
        if order > 10 {
            return None;
        }

        // Try to get a block of requested size
        if let Some(frame) = self.free_lists[order].pop() {
            USED_MEMORY.fetch_add(PAGE_SIZE << order, Ordering::AcqRel);
            return Some(frame);
        }

        // Try to split a larger block
        for larger_order in (order + 1)..=10 {
            if let Some(frame) = self.free_lists[larger_order].pop() {
                // Split the block
                let mut current_order = larger_order;
                let mut current_frame = frame;

                while current_order > order {
                    current_order -= 1;
                    let buddy_frame = PhysFrame(current_frame.0 + (1 << current_order));
                    self.free_lists[current_order].push(buddy_frame);
                }

                USED_MEMORY.fetch_add(PAGE_SIZE << order, Ordering::AcqRel);
                return Some(current_frame);
            }
        }

        None
    }

    /// Deallocate a physical frame
    pub fn deallocate(&mut self, frame: PhysFrame) {
        self.deallocate_order(frame, 0);
    }

    /// Deallocate frames of given order
    pub fn deallocate_order(&mut self, frame: PhysFrame, order: usize) {
        if order > 10 {
            return;
        }

        USED_MEMORY.fetch_sub(PAGE_SIZE << order, Ordering::AcqRel);

        // Try to merge with buddy
        let buddy = PhysFrame(frame.0 ^ (1 << order));

        if let Some(pos) = self.free_lists[order].iter().position(|f| *f == buddy) {
            self.free_lists[order].remove(pos);
            let merged = PhysFrame(frame.0.min(buddy.0));
            self.deallocate_order(merged, order + 1);
        } else {
            self.free_lists[order].push(frame);
        }
    }

    /// Coalesce all free frames into larger blocks
    fn coalesce_all(&mut self) {
        for order in 0..10 {
            let mut to_merge = Vec::new();

            self.free_lists[order].sort();

            let mut i = 0;
            while i + 1 < self.free_lists[order].len() {
                let frame = self.free_lists[order][i];
                let next = self.free_lists[order][i + 1];

                let expected_buddy = PhysFrame(frame.0 + (1 << order));
                if next == expected_buddy && (frame.0 & (1 << order)) == 0 {
                    to_merge.push((i, frame));
                    i += 2;
                } else {
                    i += 1;
                }
            }

            // Remove merged frames and add to higher order
            for (idx, frame) in to_merge.into_iter().rev() {
                self.free_lists[order].remove(idx + 1);
                self.free_lists[order].remove(idx);
                self.free_lists[order + 1].push(frame);
            }
        }
    }
}

/// Initialize memory subsystem
pub fn init(memory_map: &MemoryMap) {
    let allocator = FrameAllocator::new(memory_map);
    *FRAME_ALLOCATOR.lock() = Some(allocator);

    crate::kprintln!("  Total RAM: {} MB", total_memory() / (1024 * 1024));
}

/// Allocate a physical frame
/// This function disables interrupts while holding the allocator lock
pub fn allocate_frame() -> Option<PhysFrame> {
    // Disable interrupts during allocation to prevent deadlock
    let was_enabled = crate::arch::interrupts_enabled();
    if was_enabled {
        crate::arch::disable_interrupts();
    }

    let result = FRAME_ALLOCATOR.lock().as_mut()?.allocate();

    if was_enabled {
        crate::arch::enable_interrupts();
    }

    result
}

/// Deallocate a physical frame
/// This function disables interrupts while holding the allocator lock
pub fn deallocate_frame(frame: PhysFrame) {
    // Disable interrupts during deallocation to prevent deadlock
    let was_enabled = crate::arch::interrupts_enabled();
    if was_enabled {
        crate::arch::disable_interrupts();
    }

    if let Some(allocator) = FRAME_ALLOCATOR.lock().as_mut() {
        allocator.deallocate(frame);
    }

    if was_enabled {
        crate::arch::enable_interrupts();
    }
}

/// Get total physical memory
pub fn total_memory() -> usize {
    TOTAL_MEMORY.load(Ordering::Acquire)
}

/// Get used physical memory
pub fn used_memory() -> usize {
    USED_MEMORY.load(Ordering::Acquire)
}

/// Get free physical memory
pub fn free_memory() -> usize {
    total_memory() - used_memory()
}

/// Memory statistics
pub struct MemoryStats {
    pub total: usize,
    pub used: usize,
    pub free: usize,
    pub kernel_heap_used: usize,
}

/// Get memory statistics
pub fn stats() -> MemoryStats {
    MemoryStats {
        total: total_memory(),
        used: used_memory(),
        free: free_memory(),
        kernel_heap_used: KERNEL_HEAP_SIZE, // TODO: Track actual usage
    }
}
