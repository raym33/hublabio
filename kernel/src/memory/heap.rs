//! Kernel Heap Management
//!
//! Integration with the linked_list_allocator crate.

use core::alloc::{GlobalAlloc, Layout};
use linked_list_allocator::LockedHeap;

use super::{KERNEL_HEAP_SIZE, KERNEL_HEAP_START};

/// Heap statistics
pub struct HeapStats {
    pub total: usize,
    pub used: usize,
    pub free: usize,
}

/// Get heap statistics
pub fn stats(heap: &LockedHeap) -> HeapStats {
    let locked = heap.lock();
    let free = locked.free();
    let used = locked.used();

    HeapStats {
        total: KERNEL_HEAP_SIZE,
        used,
        free,
    }
}

/// Initialize the kernel heap
///
/// # Safety
/// Must only be called once during kernel initialization.
pub unsafe fn init(heap: &LockedHeap) {
    heap.lock()
        .init(KERNEL_HEAP_START as *mut u8, KERNEL_HEAP_SIZE);
}

/// Grow the heap by allocating more physical frames
pub fn grow(heap: &LockedHeap, additional: usize) -> Result<(), &'static str> {
    // This would require extending the heap region
    // For now, the heap is fixed size
    Err("Heap growth not implemented")
}

/// Debug: dump heap state
pub fn dump(heap: &LockedHeap) {
    let stats = stats(heap);
    crate::kprintln!("Heap Statistics:");
    crate::kprintln!("  Total: {} KB", stats.total / 1024);
    crate::kprintln!("  Used:  {} KB", stats.used / 1024);
    crate::kprintln!("  Free:  {} KB", stats.free / 1024);
}
