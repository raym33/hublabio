//! Memory Pools
//!
//! High-performance memory allocation for fixed-size objects.
//! Reduces fragmentation and improves allocation speed for hot paths.

use alloc::vec::Vec;
use alloc::boxed::Box;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::{size_of, align_of, MaybeUninit};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};

/// Memory pool for fixed-size objects
pub struct MemoryPool<T> {
    /// Free list head
    free_list: AtomicPtr<PoolNode>,
    /// Total capacity
    capacity: usize,
    /// Currently allocated
    allocated: AtomicUsize,
    /// High water mark
    high_water: AtomicUsize,
    /// Storage blocks
    blocks: UnsafeCell<Vec<Box<[MaybeUninit<PoolSlot<T>>]>>>,
    /// Block size
    block_size: usize,
    /// Phantom
    _marker: PhantomData<T>,
}

struct PoolNode {
    next: *mut PoolNode,
}

#[repr(C)]
union PoolSlot<T> {
    node: PoolNode,
    value: MaybeUninit<T>,
}

unsafe impl<T: Send> Send for MemoryPool<T> {}
unsafe impl<T: Send> Sync for MemoryPool<T> {}

impl<T> MemoryPool<T> {
    /// Create new pool with initial capacity
    pub fn new(initial_capacity: usize) -> Self {
        let block_size = initial_capacity.max(64);
        let mut pool = Self {
            free_list: AtomicPtr::new(core::ptr::null_mut()),
            capacity: 0,
            allocated: AtomicUsize::new(0),
            high_water: AtomicUsize::new(0),
            blocks: UnsafeCell::new(Vec::new()),
            block_size,
            _marker: PhantomData,
        };

        // Allocate initial block
        pool.grow();

        pool
    }

    /// Allocate object from pool
    pub fn alloc(&self) -> Option<PoolBox<T>> {
        loop {
            let head = self.free_list.load(Ordering::Acquire);

            if head.is_null() {
                // Try to grow
                // Note: This is not safe for concurrent growth
                // In production, would need proper locking
                return None;
            }

            let next = unsafe { (*head).next };

            if self.free_list.compare_exchange_weak(
                head,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                let allocated = self.allocated.fetch_add(1, Ordering::Relaxed) + 1;

                // Update high water mark
                let mut high = self.high_water.load(Ordering::Relaxed);
                while allocated > high {
                    match self.high_water.compare_exchange_weak(
                        high,
                        allocated,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(h) => high = h,
                    }
                }

                let ptr = head as *mut T;
                return Some(PoolBox {
                    ptr: NonNull::new(ptr).unwrap(),
                    pool: self,
                });
            }
        }
    }

    /// Free object back to pool
    fn free(&self, ptr: *mut T) {
        let node = ptr as *mut PoolNode;
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            unsafe { (*node).next = head };

            if self.free_list.compare_exchange_weak(
                head,
                node,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                self.allocated.fetch_sub(1, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Grow pool by adding a new block
    fn grow(&mut self) {
        let block: Box<[MaybeUninit<PoolSlot<T>>]> = (0..self.block_size)
            .map(|_| MaybeUninit::uninit())
            .collect();

        // Link all slots in new block
        let block_ptr = block.as_ptr() as *mut PoolSlot<T>;
        for i in 0..self.block_size {
            let slot = unsafe { block_ptr.add(i) };
            let next = if i + 1 < self.block_size {
                unsafe { block_ptr.add(i + 1) as *mut PoolNode }
            } else {
                self.free_list.load(Ordering::Relaxed)
            };
            unsafe {
                (*slot).node.next = next;
            }
        }

        self.free_list.store(block_ptr as *mut PoolNode, Ordering::Release);
        self.capacity += self.block_size;

        unsafe {
            (*self.blocks.get()).push(block);
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            capacity: self.capacity,
            allocated: self.allocated.load(Ordering::Relaxed),
            high_water: self.high_water.load(Ordering::Relaxed),
            object_size: size_of::<T>(),
            block_size: self.block_size,
        }
    }

    /// Reset high water mark
    pub fn reset_high_water(&self) {
        let current = self.allocated.load(Ordering::Relaxed);
        self.high_water.store(current, Ordering::Relaxed);
    }
}

/// Pool-allocated box
pub struct PoolBox<'a, T> {
    ptr: NonNull<T>,
    pool: &'a MemoryPool<T>,
}

impl<'a, T> PoolBox<'a, T> {
    /// Initialize with value
    pub fn init(mut self, value: T) -> InitPoolBox<'a, T> {
        unsafe {
            self.ptr.as_ptr().write(value);
        }
        let ptr = self.ptr;
        let pool = self.pool;
        core::mem::forget(self);
        InitPoolBox { ptr, pool }
    }
}

impl<'a, T> Drop for PoolBox<'a, T> {
    fn drop(&mut self) {
        self.pool.free(self.ptr.as_ptr());
    }
}

/// Initialized pool-allocated box
pub struct InitPoolBox<'a, T> {
    ptr: NonNull<T>,
    pool: &'a MemoryPool<T>,
}

impl<'a, T> core::ops::Deref for InitPoolBox<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T> core::ops::DerefMut for InitPoolBox<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T> Drop for InitPoolBox<'a, T> {
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(self.ptr.as_ptr());
        }
        self.pool.free(self.ptr.as_ptr());
    }
}

/// Pool statistics
#[derive(Clone, Debug)]
pub struct PoolStats {
    /// Total capacity
    pub capacity: usize,
    /// Currently allocated
    pub allocated: usize,
    /// High water mark
    pub high_water: usize,
    /// Object size
    pub object_size: usize,
    /// Block size
    pub block_size: usize,
}

impl PoolStats {
    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.allocated as f32 / self.capacity as f32) * 100.0
        }
    }

    /// Get memory used
    pub fn memory_used(&self) -> usize {
        self.allocated * self.object_size
    }

    /// Get total memory allocated
    pub fn memory_total(&self) -> usize {
        self.capacity * self.object_size
    }
}

/// Arena allocator for temporary allocations
pub struct Arena {
    /// Memory chunks
    chunks: Vec<ArenaChunk>,
    /// Current chunk index
    current: usize,
    /// Chunk size
    chunk_size: usize,
    /// Total allocated
    total_allocated: usize,
}

struct ArenaChunk {
    data: Box<[u8]>,
    used: usize,
}

impl Arena {
    /// Create new arena with default chunk size
    pub fn new() -> Self {
        Self::with_chunk_size(64 * 1024) // 64KB chunks
    }

    /// Create arena with custom chunk size
    pub fn with_chunk_size(size: usize) -> Self {
        let chunk = ArenaChunk {
            data: vec![0u8; size].into_boxed_slice(),
            used: 0,
        };

        Self {
            chunks: vec![chunk],
            current: 0,
            chunk_size: size,
            total_allocated: 0,
        }
    }

    /// Allocate memory from arena
    pub fn alloc<T>(&mut self) -> Option<&mut MaybeUninit<T>> {
        let size = size_of::<T>();
        let align = align_of::<T>();

        let ptr = self.alloc_raw(size, align)?;
        Some(unsafe { &mut *(ptr as *mut MaybeUninit<T>) })
    }

    /// Allocate raw bytes
    pub fn alloc_raw(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        // Try current chunk
        if let Some(ptr) = self.try_alloc_from_chunk(self.current, size, align) {
            self.total_allocated += size;
            return Some(ptr);
        }

        // Try other chunks
        for i in 0..self.chunks.len() {
            if i != self.current {
                if let Some(ptr) = self.try_alloc_from_chunk(i, size, align) {
                    self.current = i;
                    self.total_allocated += size;
                    return Some(ptr);
                }
            }
        }

        // Allocate new chunk
        let chunk_size = self.chunk_size.max(size);
        let chunk = ArenaChunk {
            data: vec![0u8; chunk_size].into_boxed_slice(),
            used: 0,
        };
        self.chunks.push(chunk);
        self.current = self.chunks.len() - 1;

        self.try_alloc_from_chunk(self.current, size, align).map(|ptr| {
            self.total_allocated += size;
            ptr
        })
    }

    fn try_alloc_from_chunk(&mut self, chunk_idx: usize, size: usize, align: usize) -> Option<*mut u8> {
        let chunk = &mut self.chunks[chunk_idx];
        let start = chunk.data.as_ptr() as usize + chunk.used;
        let aligned = (start + align - 1) & !(align - 1);
        let padding = aligned - start;
        let total = padding + size;

        if chunk.used + total <= chunk.data.len() {
            chunk.used += total;
            Some(aligned as *mut u8)
        } else {
            None
        }
    }

    /// Allocate and initialize
    pub fn alloc_init<T>(&mut self, value: T) -> Option<&mut T> {
        let slot = self.alloc::<T>()?;
        slot.write(value);
        Some(unsafe { slot.assume_init_mut() })
    }

    /// Allocate slice
    pub fn alloc_slice<T: Copy>(&mut self, len: usize) -> Option<&mut [MaybeUninit<T>]> {
        let size = size_of::<T>() * len;
        let align = align_of::<T>();

        let ptr = self.alloc_raw(size, align)?;
        Some(unsafe {
            core::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<T>, len)
        })
    }

    /// Reset arena (reuse memory)
    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.used = 0;
        }
        self.current = 0;
        self.total_allocated = 0;
    }

    /// Get total memory allocated
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.data.len()).sum()
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// Object cache for frequently allocated types
pub struct ObjectCache<T> {
    /// Cached objects
    cache: Vec<Box<T>>,
    /// Maximum cache size
    max_size: usize,
    /// Hits
    hits: AtomicUsize,
    /// Misses
    misses: AtomicUsize,
    /// Constructor
    constructor: fn() -> T,
}

impl<T> ObjectCache<T> {
    /// Create new cache with constructor
    pub fn new(max_size: usize, constructor: fn() -> T) -> Self {
        Self {
            cache: Vec::with_capacity(max_size),
            max_size,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            constructor,
        }
    }

    /// Get object from cache or create new
    pub fn get(&mut self) -> Box<T> {
        if let Some(obj) = self.cache.pop() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            obj
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            Box::new((self.constructor)())
        }
    }

    /// Return object to cache
    pub fn put(&mut self, obj: Box<T>) {
        if self.cache.len() < self.max_size {
            self.cache.push(obj);
        }
        // Otherwise, object is dropped
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            (hits as f32 / total as f32) * 100.0
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Prewarm cache
    pub fn prewarm(&mut self, count: usize) {
        let count = count.min(self.max_size - self.cache.len());
        for _ in 0..count {
            self.cache.push(Box::new((self.constructor)()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool: MemoryPool<u64> = MemoryPool::new(10);

        let obj1 = pool.alloc().unwrap().init(42);
        let obj2 = pool.alloc().unwrap().init(100);

        assert_eq!(*obj1, 42);
        assert_eq!(*obj2, 100);

        let stats = pool.stats();
        assert_eq!(stats.allocated, 2);

        drop(obj1);
        let stats = pool.stats();
        assert_eq!(stats.allocated, 1);
    }

    #[test]
    fn test_arena() {
        let mut arena = Arena::new();

        let val1 = arena.alloc_init(42u32).unwrap();
        let val2 = arena.alloc_init(100u64).unwrap();

        assert_eq!(*val1, 42);
        assert_eq!(*val2, 100);

        assert!(arena.total_allocated() > 0);

        arena.reset();
        assert_eq!(arena.total_allocated(), 0);
    }

    #[test]
    fn test_object_cache() {
        let mut cache: ObjectCache<Vec<u8>> = ObjectCache::new(10, Vec::new);

        // First get is a miss
        let obj1 = cache.get();
        assert_eq!(cache.hit_rate(), 0.0);

        // Return and get again is a hit
        cache.put(obj1);
        let _obj2 = cache.get();
        assert!(cache.hit_rate() > 0.0);
    }
}
