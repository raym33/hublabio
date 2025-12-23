//! Kernel Memory Allocator
//!
//! Slab allocator for efficient small object allocation.

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::NonNull;
use spin::Mutex;

use super::{allocate_frame, deallocate_frame, PhysFrame, PAGE_SIZE};

/// Slab sizes (8, 16, 32, 64, 128, 256, 512, 1024, 2048 bytes)
const SLAB_SIZES: [usize; 9] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];

/// Maximum slab size
const MAX_SLAB_SIZE: usize = 2048;

/// Slab header stored at the beginning of each slab page
#[repr(C)]
struct SlabHeader {
    /// Next free object in this slab
    free_list: Option<NonNull<FreeObject>>,
    /// Number of free objects
    free_count: usize,
    /// Total objects in this slab
    total_objects: usize,
    /// Object size for this slab
    object_size: usize,
    /// Next slab in the cache
    next_slab: Option<NonNull<SlabHeader>>,
}

/// Free object entry (intrusive linked list)
#[repr(C)]
struct FreeObject {
    next: Option<NonNull<FreeObject>>,
}

/// Slab cache for a specific object size
struct SlabCache {
    object_size: usize,
    partial_slabs: Option<NonNull<SlabHeader>>,
    full_slabs: Option<NonNull<SlabHeader>>,
    empty_slabs: Option<NonNull<SlabHeader>>,
    slabs_count: usize,
}

impl SlabCache {
    const fn new(size: usize) -> Self {
        Self {
            object_size: size,
            partial_slabs: None,
            full_slabs: None,
            empty_slabs: None,
            slabs_count: 0,
        }
    }

    /// Check if a slab belongs to this cache
    fn owns_slab(&self, slab_ptr: NonNull<SlabHeader>) -> bool {
        // Check partial slabs
        let mut current = self.partial_slabs;
        while let Some(ptr) = current {
            if ptr == slab_ptr {
                return true;
            }
            current = unsafe { ptr.as_ref().next_slab };
        }

        // Check full slabs
        current = self.full_slabs;
        while let Some(ptr) = current {
            if ptr == slab_ptr {
                return true;
            }
            current = unsafe { ptr.as_ref().next_slab };
        }

        // Check empty slabs
        current = self.empty_slabs;
        while let Some(ptr) = current {
            if ptr == slab_ptr {
                return true;
            }
            current = unsafe { ptr.as_ref().next_slab };
        }

        false
    }

    /// Validate that a pointer belongs to a valid object within a slab
    fn validate_object(&self, ptr: NonNull<u8>, slab: &SlabHeader) -> bool {
        let slab_addr = slab as *const SlabHeader as usize;
        let ptr_addr = ptr.as_ptr() as usize;

        // Object must be within the slab page
        if ptr_addr < slab_addr || ptr_addr >= slab_addr + PAGE_SIZE {
            return false;
        }

        // Calculate header size and first object offset
        let header_size = core::mem::size_of::<SlabHeader>();
        let first_obj = slab_addr + header_size;

        // Object must be at or after the header
        if ptr_addr < first_obj {
            return false;
        }

        // Object must be aligned to object size
        let offset = ptr_addr - first_obj;
        if offset % slab.object_size != 0 {
            return false;
        }

        // Object index must be within bounds
        let obj_index = offset / slab.object_size;
        if obj_index >= slab.total_objects {
            return false;
        }

        true
    }

    /// Allocate an object from this cache
    fn allocate(&mut self) -> Option<NonNull<u8>> {
        // Try partial slabs first
        if let Some(slab_ptr) = self.partial_slabs {
            let slab = unsafe { slab_ptr.as_ref() };
            if let Some(obj) = slab.free_list {
                // Update free list
                let slab = unsafe { slab_ptr.as_ptr().as_mut()? };
                slab.free_list = unsafe { obj.as_ref().next };
                slab.free_count -= 1;

                // Move to full list if needed
                if slab.free_count == 0 {
                    self.partial_slabs = slab.next_slab;
                    slab.next_slab = self.full_slabs;
                    self.full_slabs = Some(slab_ptr);
                }

                return Some(obj.cast());
            }
        }

        // Try empty slabs
        if let Some(slab_ptr) = self.empty_slabs {
            let slab = unsafe { slab_ptr.as_ptr().as_mut()? };
            self.empty_slabs = slab.next_slab;
            slab.next_slab = self.partial_slabs;
            self.partial_slabs = Some(slab_ptr);
            return self.allocate();
        }

        // Need to create a new slab
        self.grow()?;
        self.allocate()
    }

    /// Deallocate an object back to this cache
    /// Returns false if the pointer doesn't belong to this cache
    fn deallocate(&mut self, ptr: NonNull<u8>) -> bool {
        // Find which slab this belongs to
        let page_addr = (ptr.as_ptr() as usize) & !(PAGE_SIZE - 1);

        // Validate page_addr is non-null
        if page_addr == 0 {
            return false;
        }

        let slab_ptr = match NonNull::new(page_addr as *mut SlabHeader) {
            Some(p) => p,
            None => return false,
        };

        // Verify this slab belongs to this cache
        if !self.owns_slab(slab_ptr) {
            return false;
        }

        let slab = unsafe { slab_ptr.as_ptr().as_mut().unwrap() };

        // Validate the object size matches
        if slab.object_size != self.object_size {
            return false;
        }

        // Validate the pointer is a valid object within the slab
        if !self.validate_object(ptr, slab) {
            return false;
        }

        // Check for double-free by scanning free list
        let mut current = slab.free_list;
        while let Some(free_obj) = current {
            if free_obj.cast() == ptr {
                // Double free detected!
                return false;
            }
            current = unsafe { free_obj.as_ref().next };
        }

        // Add to free list
        let obj = ptr.cast::<FreeObject>();
        unsafe {
            (*obj.as_ptr()).next = slab.free_list;
        }
        slab.free_list = Some(obj);
        slab.free_count += 1;

        // Move between lists if needed
        if slab.free_count == slab.total_objects {
            // Move to empty list
            self.remove_from_partial(slab_ptr);
            slab.next_slab = self.empty_slabs;
            self.empty_slabs = Some(slab_ptr);
        } else if slab.free_count == 1 {
            // Was full, move to partial
            self.remove_from_full(slab_ptr);
            slab.next_slab = self.partial_slabs;
            self.partial_slabs = Some(slab_ptr);
        }

        true
    }

    /// Grow the cache by adding a new slab
    fn grow(&mut self) -> Option<()> {
        let frame = allocate_frame()?;
        let slab_addr = frame.start_address();

        // Calculate how many objects fit in a page
        let header_size = core::mem::size_of::<SlabHeader>();
        let available = PAGE_SIZE - header_size;
        let objects_per_slab = available / self.object_size;

        if objects_per_slab == 0 {
            deallocate_frame(frame);
            return None;
        }

        // Initialize slab header
        let slab = unsafe { &mut *(slab_addr as *mut SlabHeader) };
        slab.object_size = self.object_size;
        slab.total_objects = objects_per_slab;
        slab.free_count = objects_per_slab;
        slab.next_slab = None;

        // Initialize free list
        let first_obj_addr = slab_addr + header_size;
        let mut prev: Option<NonNull<FreeObject>> = None;

        for i in (0..objects_per_slab).rev() {
            let obj_addr = first_obj_addr + i * self.object_size;
            let obj = unsafe { &mut *(obj_addr as *mut FreeObject) };
            obj.next = prev;
            prev = NonNull::new(obj);
        }

        slab.free_list = prev;

        // Add to empty slabs
        let slab_ptr = unsafe { NonNull::new_unchecked(slab_addr as *mut SlabHeader) };
        slab.next_slab = self.empty_slabs;
        self.empty_slabs = Some(slab_ptr);
        self.slabs_count += 1;

        Some(())
    }

    fn remove_from_partial(&mut self, target: NonNull<SlabHeader>) {
        let mut current = &mut self.partial_slabs;
        while let Some(ptr) = *current {
            if ptr == target {
                *current = unsafe { ptr.as_ref().next_slab };
                return;
            }
            current = unsafe { &mut (*ptr.as_ptr()).next_slab };
        }
    }

    fn remove_from_full(&mut self, target: NonNull<SlabHeader>) {
        let mut current = &mut self.full_slabs;
        while let Some(ptr) = *current {
            if ptr == target {
                *current = unsafe { ptr.as_ref().next_slab };
                return;
            }
            current = unsafe { &mut (*ptr.as_ptr()).next_slab };
        }
    }
}

/// Global slab allocator
pub struct SlabAllocator {
    caches: [SlabCache; 9],
}

impl SlabAllocator {
    pub const fn new() -> Self {
        Self {
            caches: [
                SlabCache::new(8),
                SlabCache::new(16),
                SlabCache::new(32),
                SlabCache::new(64),
                SlabCache::new(128),
                SlabCache::new(256),
                SlabCache::new(512),
                SlabCache::new(1024),
                SlabCache::new(2048),
            ],
        }
    }

    /// Find the appropriate cache for a given size
    fn find_cache(&mut self, size: usize) -> Option<&mut SlabCache> {
        for (i, &slab_size) in SLAB_SIZES.iter().enumerate() {
            if size <= slab_size {
                return Some(&mut self.caches[i]);
            }
        }
        None
    }

    /// Allocate memory
    pub fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size().max(layout.align());

        if size <= MAX_SLAB_SIZE {
            self.find_cache(size)?.allocate()
        } else {
            // Large allocation: use page allocator directly
            let pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
            let order = pages.next_power_of_two().trailing_zeros() as usize;

            let frame = super::FRAME_ALLOCATOR
                .lock()
                .as_mut()?
                .allocate_order(order)?;
            NonNull::new(frame.start_address() as *mut u8)
        }
    }

    /// Deallocate memory
    /// Returns true if deallocation succeeded, false if the pointer was invalid
    pub fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) -> bool {
        let size = layout.size().max(layout.align());

        if size <= MAX_SLAB_SIZE {
            if let Some(cache) = self.find_cache(size) {
                return cache.deallocate(ptr);
            }
            return false;
        } else {
            // Large allocation - validate pointer alignment
            let ptr_addr = ptr.as_ptr() as usize;
            if ptr_addr % PAGE_SIZE != 0 {
                return false;
            }

            let pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
            let order = pages.next_power_of_two().trailing_zeros() as usize;
            let frame = PhysFrame::containing_address(ptr_addr);

            if let Some(allocator) = super::FRAME_ALLOCATOR.lock().as_mut() {
                allocator.deallocate_order(frame, order);
                return true;
            }
            return false;
        }
    }
}

/// Global kernel allocator instance
pub static KERNEL_ALLOCATOR: Mutex<SlabAllocator> = Mutex::new(SlabAllocator::new());

/// Allocate kernel memory
pub fn kmalloc(layout: Layout) -> Option<NonNull<u8>> {
    KERNEL_ALLOCATOR.lock().allocate(layout)
}

/// Free kernel memory
/// Returns true if deallocation succeeded, false if the pointer was invalid
/// (e.g., double-free, unaligned, or not from this allocator)
pub fn kfree(ptr: NonNull<u8>, layout: Layout) -> bool {
    KERNEL_ALLOCATOR.lock().deallocate(ptr, layout)
}
