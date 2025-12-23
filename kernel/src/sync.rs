//! Synchronization Primitives for SMP
//!
//! Provides spinlocks, ticketlocks, RW locks, and other synchronization
//! primitives optimized for ARM64 multi-core systems.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence};
use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::arch::asm;

/// Memory barrier types
#[inline]
pub fn dmb_sy() {
    unsafe { asm!("dmb sy"); }
}

#[inline]
pub fn dmb_ld() {
    unsafe { asm!("dmb ld"); }
}

#[inline]
pub fn dmb_st() {
    unsafe { asm!("dmb st"); }
}

#[inline]
pub fn dsb_sy() {
    unsafe { asm!("dsb sy"); }
}

#[inline]
pub fn isb() {
    unsafe { asm!("isb"); }
}

/// Spin hint for busy-wait loops
#[inline]
pub fn spin_hint() {
    unsafe { asm!("yield"); }
}

/// Simple spinlock using atomic exchange
///
/// This is a test-and-set lock, simple but not fair.
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
    #[cfg(debug_assertions)]
    owner_cpu: AtomicU32,
}

unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
            #[cfg(debug_assertions)]
            owner_cpu: AtomicU32::new(u32::MAX),
        }
    }

    /// Try to acquire the lock without blocking
    pub fn try_lock(&self) -> Option<SpinLockGuard<T>> {
        if self.locked.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed
        ).is_ok() {
            #[cfg(debug_assertions)]
            self.owner_cpu.store(crate::smp::cpu_id(), Ordering::Relaxed);

            Some(SpinLockGuard { lock: self })
        } else {
            None
        }
    }

    /// Acquire the lock, spinning until available
    pub fn lock(&self) -> SpinLockGuard<T> {
        loop {
            // Try to acquire
            if let Some(guard) = self.try_lock() {
                return guard;
            }

            // Spin while locked
            while self.locked.load(Ordering::Relaxed) {
                spin_hint();
            }
        }
    }

    /// Acquire lock with interrupts disabled
    pub fn lock_irq(&self) -> SpinLockIrqGuard<T> {
        let irq_enabled = crate::arch::interrupts_enabled();
        if irq_enabled {
            crate::arch::disable_interrupts();
        }

        let guard = self.lock();

        SpinLockIrqGuard {
            guard,
            irq_was_enabled: irq_enabled,
        }
    }

    /// Check if locked (for debugging)
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Relaxed)
    }
}

/// Guard for SpinLock
pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> Deref for SpinLockGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        self.lock.owner_cpu.store(u32::MAX, Ordering::Relaxed);

        self.lock.locked.store(false, Ordering::Release);
    }
}

/// Guard with interrupt restoration
pub struct SpinLockIrqGuard<'a, T> {
    guard: SpinLockGuard<'a, T>,
    irq_was_enabled: bool,
}

impl<T> Deref for SpinLockIrqGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for SpinLockIrqGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.guard
    }
}

impl<T> Drop for SpinLockIrqGuard<'_, T> {
    fn drop(&mut self) {
        // Guard is dropped first, then we restore interrupts
        // Note: guard is dropped automatically, we just restore IRQ state
    }
}

impl<'a, T> SpinLockIrqGuard<'a, T> {
    fn restore_irq(&self) {
        if self.irq_was_enabled {
            crate::arch::enable_interrupts();
        }
    }
}

/// Ticket lock - provides fair FIFO ordering
///
/// More overhead than SpinLock but guarantees fairness.
pub struct TicketLock<T> {
    next_ticket: AtomicU32,
    now_serving: AtomicU32,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for TicketLock<T> {}
unsafe impl<T: Send> Sync for TicketLock<T> {}

impl<T> TicketLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            next_ticket: AtomicU32::new(0),
            now_serving: AtomicU32::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire the lock
    pub fn lock(&self) -> TicketLockGuard<T> {
        // Get our ticket
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);

        // Wait for our turn
        while self.now_serving.load(Ordering::Acquire) != ticket {
            spin_hint();
        }

        TicketLockGuard { lock: self }
    }

    /// Try to acquire without blocking
    pub fn try_lock(&self) -> Option<TicketLockGuard<T>> {
        let ticket = self.next_ticket.load(Ordering::Relaxed);
        let serving = self.now_serving.load(Ordering::Relaxed);

        if ticket == serving {
            // No one waiting, try to take next ticket
            if self.next_ticket.compare_exchange(
                ticket,
                ticket + 1,
                Ordering::Acquire,
                Ordering::Relaxed
            ).is_ok() {
                return Some(TicketLockGuard { lock: self });
            }
        }

        None
    }
}

/// Guard for TicketLock
pub struct TicketLockGuard<'a, T> {
    lock: &'a TicketLock<T>,
}

impl<T> Deref for TicketLockGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for TicketLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for TicketLockGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.now_serving.fetch_add(1, Ordering::Release);
    }
}

/// Reader-Writer lock
///
/// Multiple readers OR single writer, but not both.
/// Writers have priority (no writer starvation).
pub struct RwLock<T> {
    /// Bit 31: writer active, Bits 30-0: reader count
    state: AtomicU32,
    /// Number of waiting writers (for writer priority)
    writer_waiting: AtomicU32,
    data: UnsafeCell<T>,
}

const WRITER_BIT: u32 = 1 << 31;
const READER_MASK: u32 = !(1 << 31);

unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicU32::new(0),
            writer_waiting: AtomicU32::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire read lock
    pub fn read(&self) -> RwLockReadGuard<T> {
        loop {
            // Wait if there's a writer or writers waiting
            while self.state.load(Ordering::Relaxed) & WRITER_BIT != 0
                || self.writer_waiting.load(Ordering::Relaxed) > 0
            {
                spin_hint();
            }

            // Try to increment reader count
            let state = self.state.fetch_add(1, Ordering::Acquire);
            if state & WRITER_BIT == 0 {
                // Success, no writer
                return RwLockReadGuard { lock: self };
            }

            // Writer sneaked in, undo and retry
            self.state.fetch_sub(1, Ordering::Release);
        }
    }

    /// Try to acquire read lock
    pub fn try_read(&self) -> Option<RwLockReadGuard<T>> {
        let state = self.state.load(Ordering::Relaxed);

        // Check for writer or waiting writers
        if state & WRITER_BIT != 0 || self.writer_waiting.load(Ordering::Relaxed) > 0 {
            return None;
        }

        // Try to increment
        let new_state = self.state.fetch_add(1, Ordering::Acquire);
        if new_state & WRITER_BIT == 0 {
            Some(RwLockReadGuard { lock: self })
        } else {
            self.state.fetch_sub(1, Ordering::Release);
            None
        }
    }

    /// Acquire write lock
    pub fn write(&self) -> RwLockWriteGuard<T> {
        // Indicate writer is waiting
        self.writer_waiting.fetch_add(1, Ordering::Relaxed);

        loop {
            // Wait for no readers and no other writer
            while self.state.load(Ordering::Relaxed) != 0 {
                spin_hint();
            }

            // Try to set writer bit
            if self.state.compare_exchange(
                0,
                WRITER_BIT,
                Ordering::Acquire,
                Ordering::Relaxed
            ).is_ok() {
                self.writer_waiting.fetch_sub(1, Ordering::Relaxed);
                return RwLockWriteGuard { lock: self };
            }
        }
    }

    /// Try to acquire write lock
    pub fn try_write(&self) -> Option<RwLockWriteGuard<T>> {
        if self.state.compare_exchange(
            0,
            WRITER_BIT,
            Ordering::Acquire,
            Ordering::Relaxed
        ).is_ok() {
            Some(RwLockWriteGuard { lock: self })
        } else {
            None
        }
    }
}

pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.store(0, Ordering::Release);
    }
}

/// Sequence lock for read-mostly data
///
/// Readers never block, but may need to retry if writer was active.
/// Best for data that is read very frequently but written rarely.
pub struct SeqLock<T: Copy> {
    seq: AtomicU32,
    data: UnsafeCell<T>,
}

unsafe impl<T: Copy + Send> Send for SeqLock<T> {}
unsafe impl<T: Copy + Send + Sync> Sync for SeqLock<T> {}

impl<T: Copy> SeqLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            seq: AtomicU32::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Read data, retrying if a write was in progress
    pub fn read(&self) -> T {
        loop {
            let seq1 = self.seq.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                // Write in progress, retry
                spin_hint();
                continue;
            }

            // Read data
            let data = unsafe { *self.data.get() };

            // Verify no write happened
            fence(Ordering::Acquire);
            let seq2 = self.seq.load(Ordering::Relaxed);

            if seq1 == seq2 {
                return data;
            }

            // Sequence changed, retry
            spin_hint();
        }
    }

    /// Write data
    pub fn write(&self, data: T) {
        // Increment sequence (now odd = write in progress)
        self.seq.fetch_add(1, Ordering::Release);
        fence(Ordering::Release);

        // Write data
        unsafe {
            *self.data.get() = data;
        }

        // Increment sequence (now even = write complete)
        fence(Ordering::Release);
        self.seq.fetch_add(1, Ordering::Release);
    }
}

/// Per-CPU variable wrapper
///
/// Provides lock-free access to per-CPU data.
pub struct PerCpuVar<T> {
    data: [UnsafeCell<T>; crate::smp::MAX_CPUS],
}

unsafe impl<T: Send> Send for PerCpuVar<T> {}
unsafe impl<T: Send> Sync for PerCpuVar<T> {}

impl<T: Copy + Default> PerCpuVar<T> {
    pub const fn new() -> Self {
        Self {
            data: [const { UnsafeCell::new(unsafe { core::mem::zeroed() }) }; crate::smp::MAX_CPUS],
        }
    }

    /// Get reference to current CPU's data (preemption should be disabled)
    pub fn get(&self) -> &T {
        let cpu = crate::smp::cpu_id() as usize;
        unsafe { &*self.data[cpu].get() }
    }

    /// Get mutable reference to current CPU's data
    pub fn get_mut(&self) -> &mut T {
        let cpu = crate::smp::cpu_id() as usize;
        unsafe { &mut *self.data[cpu].get() }
    }

    /// Get reference to specific CPU's data
    pub fn get_for(&self, cpu: u32) -> &T {
        unsafe { &*self.data[cpu as usize].get() }
    }
}

/// Once cell - initialized exactly once
pub struct Once<T> {
    state: AtomicU32,
    data: UnsafeCell<Option<T>>,
}

const ONCE_INIT: u32 = 0;
const ONCE_RUNNING: u32 = 1;
const ONCE_DONE: u32 = 2;

unsafe impl<T: Send + Sync> Send for Once<T> {}
unsafe impl<T: Send + Sync> Sync for Once<T> {}

impl<T> Once<T> {
    pub const fn new() -> Self {
        Self {
            state: AtomicU32::new(ONCE_INIT),
            data: UnsafeCell::new(None),
        }
    }

    /// Initialize with the given closure, or return existing value
    pub fn call_once<F: FnOnce() -> T>(&self, f: F) -> &T {
        loop {
            match self.state.compare_exchange(
                ONCE_INIT,
                ONCE_RUNNING,
                Ordering::Acquire,
                Ordering::Acquire
            ) {
                Ok(_) => {
                    // We won the race, initialize
                    unsafe {
                        *self.data.get() = Some(f());
                    }
                    self.state.store(ONCE_DONE, Ordering::Release);
                    return unsafe { (*self.data.get()).as_ref().unwrap() };
                }
                Err(ONCE_DONE) => {
                    // Already initialized
                    return unsafe { (*self.data.get()).as_ref().unwrap() };
                }
                Err(ONCE_RUNNING) => {
                    // Someone else is initializing, wait
                    while self.state.load(Ordering::Acquire) == ONCE_RUNNING {
                        spin_hint();
                    }
                }
                _ => unreachable!()
            }
        }
    }

    /// Check if initialized
    pub fn is_completed(&self) -> bool {
        self.state.load(Ordering::Acquire) == ONCE_DONE
    }

    /// Get value if initialized
    pub fn get(&self) -> Option<&T> {
        if self.is_completed() {
            unsafe { (*self.data.get()).as_ref() }
        } else {
            None
        }
    }
}

/// Barrier for synchronizing multiple CPUs
pub struct Barrier {
    count: AtomicUsize,
    generation: AtomicU32,
    num_threads: usize,
}

impl Barrier {
    pub const fn new(num_threads: usize) -> Self {
        Self {
            count: AtomicUsize::new(0),
            generation: AtomicU32::new(0),
            num_threads,
        }
    }

    /// Wait at barrier until all threads arrive
    pub fn wait(&self) {
        let gen = self.generation.load(Ordering::Relaxed);

        let arrived = self.count.fetch_add(1, Ordering::AcqRel) + 1;

        if arrived == self.num_threads {
            // Last one to arrive, release everyone
            self.count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            // Wait for generation to change
            while self.generation.load(Ordering::Acquire) == gen {
                spin_hint();
            }
        }
    }
}

/// CPU-local critical section
///
/// Disables preemption for the duration.
pub struct CriticalSection;

impl CriticalSection {
    pub fn enter() -> CriticalSectionGuard {
        crate::smp::current().preempt_disable();
        CriticalSectionGuard { _private: () }
    }
}

pub struct CriticalSectionGuard {
    _private: (),
}

impl Drop for CriticalSectionGuard {
    fn drop(&mut self) {
        crate::smp::current().preempt_enable();
    }
}

/// IRQ-safe critical section
///
/// Disables interrupts and preemption.
pub struct IrqCriticalSection;

impl IrqCriticalSection {
    pub fn enter() -> IrqCriticalSectionGuard {
        let irq_enabled = crate::arch::interrupts_enabled();
        if irq_enabled {
            crate::arch::disable_interrupts();
        }
        crate::smp::current().preempt_disable();

        IrqCriticalSectionGuard { irq_was_enabled: irq_enabled }
    }
}

pub struct IrqCriticalSectionGuard {
    irq_was_enabled: bool,
}

impl Drop for IrqCriticalSectionGuard {
    fn drop(&mut self) {
        crate::smp::current().preempt_enable();
        if self.irq_was_enabled {
            crate::arch::enable_interrupts();
        }
    }
}
