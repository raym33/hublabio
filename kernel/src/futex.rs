//! Futex - Fast Userspace Mutex
//!
//! Linux-compatible futex implementation for efficient userspace synchronization.
//! Provides atomic wait/wake operations on memory addresses.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::{Mutex, RwLock};

use crate::process::Pid;

/// Futex operations
pub mod ops {
    pub const FUTEX_WAIT: i32 = 0;
    pub const FUTEX_WAKE: i32 = 1;
    pub const FUTEX_FD: i32 = 2;
    pub const FUTEX_REQUEUE: i32 = 3;
    pub const FUTEX_CMP_REQUEUE: i32 = 4;
    pub const FUTEX_WAKE_OP: i32 = 5;
    pub const FUTEX_LOCK_PI: i32 = 6;
    pub const FUTEX_UNLOCK_PI: i32 = 7;
    pub const FUTEX_TRYLOCK_PI: i32 = 8;
    pub const FUTEX_WAIT_BITSET: i32 = 9;
    pub const FUTEX_WAKE_BITSET: i32 = 10;
    pub const FUTEX_WAIT_REQUEUE_PI: i32 = 11;
    pub const FUTEX_CMP_REQUEUE_PI: i32 = 12;

    /// Private futex (not shared between processes)
    pub const FUTEX_PRIVATE_FLAG: i32 = 128;
    /// Clock realtime
    pub const FUTEX_CLOCK_REALTIME: i32 = 256;

    /// Wake all waiters
    pub const FUTEX_BITSET_MATCH_ANY: u32 = 0xFFFFFFFF;
}

/// Futex key - identifies a futex location
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FutexKey {
    /// Address in memory
    pub address: usize,
    /// Process ID (for private futexes)
    pub pid: Option<Pid>,
    /// Shared (across processes via shared memory)
    pub shared: bool,
}

impl FutexKey {
    /// Create private futex key (process-local)
    pub fn private(address: usize, pid: Pid) -> Self {
        Self {
            address,
            pid: Some(pid),
            shared: false,
        }
    }

    /// Create shared futex key (for shared memory)
    pub fn shared(physical_address: usize) -> Self {
        Self {
            address: physical_address,
            pid: None,
            shared: true,
        }
    }
}

/// Waiter on a futex
#[derive(Clone)]
struct FutexWaiter {
    /// Waiting process
    pid: Pid,
    /// Thread ID
    tid: u32,
    /// Bitset for selective wake
    bitset: u32,
    /// Wake flag
    woken: Arc<AtomicU32>,
    /// Wait queue reference
    wait_queue: Arc<crate::waitqueue::WaitQueue>,
}

/// Futex queue for a specific key
struct FutexQueue {
    /// Waiters on this futex
    waiters: VecDeque<FutexWaiter>,
}

impl FutexQueue {
    fn new() -> Self {
        Self {
            waiters: VecDeque::new(),
        }
    }
}

/// Global futex table
static FUTEX_TABLE: RwLock<BTreeMap<FutexKey, FutexQueue>> = RwLock::new(BTreeMap::new());

/// Statistics
static STATS: FutexStats = FutexStats::new();

struct FutexStats {
    waits: AtomicU64,
    wakes: AtomicU64,
    requeues: AtomicU64,
    timeouts: AtomicU64,
}

impl FutexStats {
    const fn new() -> Self {
        Self {
            waits: AtomicU64::new(0),
            wakes: AtomicU64::new(0),
            requeues: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
        }
    }
}

/// Futex error
#[derive(Clone, Copy, Debug)]
pub enum FutexError {
    /// Value doesn't match expected
    WouldBlock,
    /// Timeout expired
    TimedOut,
    /// Invalid argument
    Invalid,
    /// Interrupted by signal
    Interrupted,
    /// Bad address
    Fault,
    /// Deadlock detected
    Deadlock,
    /// Owner died
    OwnerDied,
    /// Not owner
    NotOwner,
    /// Resource unavailable
    Again,
}

impl FutexError {
    pub fn to_errno(&self) -> i32 {
        match self {
            FutexError::WouldBlock => -11, // EAGAIN
            FutexError::TimedOut => -110,  // ETIMEDOUT
            FutexError::Invalid => -22,    // EINVAL
            FutexError::Interrupted => -4, // EINTR
            FutexError::Fault => -14,      // EFAULT
            FutexError::Deadlock => -35,   // EDEADLK
            FutexError::OwnerDied => -130, // EOWNERDEAD
            FutexError::NotOwner => -1,    // EPERM
            FutexError::Again => -11,      // EAGAIN
        }
    }
}

/// Validate a user-space address is accessible
fn validate_user_address(address: usize, size: usize) -> Result<(), FutexError> {
    // Check for null pointer
    if address == 0 {
        return Err(FutexError::Fault);
    }

    // Check alignment for atomic operations
    if address % core::mem::size_of::<u32>() != 0 {
        return Err(FutexError::Invalid);
    }

    // Check for overflow
    if address.checked_add(size).is_none() {
        return Err(FutexError::Fault);
    }

    // Check address is in valid user space range (not kernel space)
    const USER_SPACE_END: usize = 0x0000_FFFF_FFFF_FFFF;
    const KERNEL_SPACE_START: usize = 0xFFFF_0000_0000_0000;

    if address >= KERNEL_SPACE_START || address + size > USER_SPACE_END {
        return Err(FutexError::Fault);
    }

    // Verify the address is mapped in the current process's address space
    if let Some(process) = crate::process::current() {
        let memory = process.memory.lock();
        let mut found = false;
        for region in &memory.regions {
            if address >= region.start && address + size <= region.end {
                found = true;
                break;
            }
        }
        if !found {
            return Err(FutexError::Fault);
        }
    }

    Ok(())
}

/// Safely read an atomic u32 from user space
fn safe_read_user_atomic(address: usize) -> Result<u32, FutexError> {
    validate_user_address(address, core::mem::size_of::<u32>())?;

    // Now safe to dereference
    let atomic = unsafe { (address as *const AtomicU32).as_ref() }.ok_or(FutexError::Fault)?;

    Ok(atomic.load(Ordering::SeqCst))
}

/// Wait on futex
pub fn futex_wait(
    address: usize,
    expected: u32,
    timeout_ns: Option<u64>,
    bitset: u32,
    private: bool,
) -> Result<(), FutexError> {
    STATS.waits.fetch_add(1, Ordering::Relaxed);

    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(FutexError::Invalid)?;

    // Validate address before any operations
    validate_user_address(address, core::mem::size_of::<u32>())?;

    // Create key
    let key = if private {
        FutexKey::private(address, pid)
    } else {
        // For shared, would need to translate to physical address
        FutexKey::shared(address)
    };

    // Read current value atomically with bounds checking
    let current = safe_read_user_atomic(address)?;

    // Check if value matches expected
    if current != expected {
        return Err(FutexError::WouldBlock);
    }

    // Create waiter
    let woken = Arc::new(AtomicU32::new(0));
    let wait_queue = Arc::new(crate::waitqueue::WaitQueue::new());

    let waiter = FutexWaiter {
        pid,
        tid: 0, // Would get thread ID
        bitset,
        woken: woken.clone(),
        wait_queue: wait_queue.clone(),
    };

    // Add to wait queue
    {
        let mut table = FUTEX_TABLE.write();
        let queue = table.entry(key).or_insert_with(FutexQueue::new);
        queue.waiters.push_back(waiter);
    }

    // Calculate deadline
    let deadline = timeout_ns.map(|t| crate::time::monotonic_ns() + t);

    // Wait loop
    loop {
        // Check if woken
        if woken.load(Ordering::SeqCst) != 0 {
            return Ok(());
        }

        // Check for signals
        if crate::signal::has_pending_signals(pid) {
            // Remove from queue
            remove_waiter(&key, pid);
            return Err(FutexError::Interrupted);
        }

        // Check timeout
        if let Some(dl) = deadline {
            if crate::time::monotonic_ns() >= dl {
                // Remove from queue
                remove_waiter(&key, pid);
                STATS.timeouts.fetch_add(1, Ordering::Relaxed);
                return Err(FutexError::TimedOut);
            }

            let remaining = dl - crate::time::monotonic_ns();
            wait_queue.wait_timeout(remaining / 1_000_000);
        } else {
            wait_queue.wait();
        }
    }
}

/// Wake waiters on futex
pub fn futex_wake(
    address: usize,
    count: u32,
    bitset: u32,
    private: bool,
) -> Result<u32, FutexError> {
    STATS.wakes.fetch_add(1, Ordering::Relaxed);

    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(FutexError::Invalid)?;

    let key = if private {
        FutexKey::private(address, pid)
    } else {
        FutexKey::shared(address)
    };

    let mut woken = 0u32;

    let mut table = FUTEX_TABLE.write();
    if let Some(queue) = table.get_mut(&key) {
        let mut i = 0;
        while i < queue.waiters.len() && woken < count {
            let waiter = &queue.waiters[i];

            // Check bitset match
            if (waiter.bitset & bitset) != 0 {
                // Wake this waiter
                waiter.woken.store(1, Ordering::SeqCst);
                waiter.wait_queue.wake_all();
                queue.waiters.remove(i);
                woken += 1;
            } else {
                i += 1;
            }
        }

        // Clean up empty queue
        if queue.waiters.is_empty() {
            table.remove(&key);
        }
    }

    Ok(woken)
}

/// Requeue waiters from one futex to another
pub fn futex_requeue(
    src_addr: usize,
    dst_addr: usize,
    wake_count: u32,
    requeue_count: u32,
    expected: Option<u32>,
    private: bool,
) -> Result<u32, FutexError> {
    STATS.requeues.fetch_add(1, Ordering::Relaxed);

    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(FutexError::Invalid)?;

    // Validate both addresses
    validate_user_address(src_addr, core::mem::size_of::<u32>())?;
    validate_user_address(dst_addr, core::mem::size_of::<u32>())?;

    // Check expected value if provided
    if let Some(exp) = expected {
        let current = safe_read_user_atomic(src_addr)?;

        if current != exp {
            return Err(FutexError::WouldBlock);
        }
    }

    let src_key = if private {
        FutexKey::private(src_addr, pid)
    } else {
        FutexKey::shared(src_addr)
    };

    let dst_key = if private {
        FutexKey::private(dst_addr, pid)
    } else {
        FutexKey::shared(dst_addr)
    };

    let mut woken = 0u32;
    let mut requeued = 0u32;

    let mut table = FUTEX_TABLE.write();

    // Get source queue
    if let Some(src_queue) = table.get_mut(&src_key) {
        // First wake some
        while !src_queue.waiters.is_empty() && woken < wake_count {
            if let Some(waiter) = src_queue.waiters.pop_front() {
                waiter.woken.store(1, Ordering::SeqCst);
                waiter.wait_queue.wake_all();
                woken += 1;
            }
        }

        // Then requeue rest
        let mut to_requeue = Vec::new();
        while !src_queue.waiters.is_empty() && requeued < requeue_count {
            if let Some(waiter) = src_queue.waiters.pop_front() {
                to_requeue.push(waiter);
                requeued += 1;
            }
        }

        // Add to destination queue
        if !to_requeue.is_empty() {
            let dst_queue = table.entry(dst_key).or_insert_with(FutexQueue::new);
            for waiter in to_requeue {
                dst_queue.waiters.push_back(waiter);
            }
        }

        // Clean up empty source queue
        if src_queue.waiters.is_empty() {
            table.remove(&src_key);
        }
    }

    Ok(woken + requeued)
}

/// Wake operation on futex
pub fn futex_wake_op(
    addr1: usize,
    addr2: usize,
    wake1_count: u32,
    wake2_count: u32,
    op: u32,
    private: bool,
) -> Result<u32, FutexError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(FutexError::Invalid)?;

    // Validate both addresses before any operations
    validate_user_address(addr1, core::mem::size_of::<u32>())?;
    validate_user_address(addr2, core::mem::size_of::<u32>())?;

    // Decode operation
    let op_type = (op >> 28) & 0xF;
    let cmp_type = (op >> 24) & 0xF;
    let op_arg = ((op >> 12) & 0xFFF) as i32;
    let cmp_arg = (op & 0xFFF) as u32;

    // Perform atomic operation on addr2 (already validated)
    let atomic = unsafe { (addr2 as *const AtomicU32).as_ref() }.ok_or(FutexError::Fault)?;

    let old_val = match op_type {
        0 => atomic.fetch_add(op_arg as u32, Ordering::SeqCst), // FUTEX_OP_SET
        1 => atomic.fetch_add(op_arg as u32, Ordering::SeqCst), // FUTEX_OP_ADD
        2 => atomic.fetch_or(op_arg as u32, Ordering::SeqCst),  // FUTEX_OP_OR
        3 => atomic.fetch_and(!op_arg as u32, Ordering::SeqCst), // FUTEX_OP_ANDN
        4 => atomic.fetch_xor(op_arg as u32, Ordering::SeqCst), // FUTEX_OP_XOR
        _ => return Err(FutexError::Invalid),
    };

    // Wake on addr1
    let woken1 = futex_wake(addr1, wake1_count, ops::FUTEX_BITSET_MATCH_ANY, private)?;

    // Compare and conditionally wake on addr2
    let should_wake2 = match cmp_type {
        0 => old_val == cmp_arg, // FUTEX_OP_CMP_EQ
        1 => old_val != cmp_arg, // FUTEX_OP_CMP_NE
        2 => old_val < cmp_arg,  // FUTEX_OP_CMP_LT
        3 => old_val <= cmp_arg, // FUTEX_OP_CMP_LE
        4 => old_val > cmp_arg,  // FUTEX_OP_CMP_GT
        5 => old_val >= cmp_arg, // FUTEX_OP_CMP_GE
        _ => false,
    };

    let woken2 = if should_wake2 {
        futex_wake(addr2, wake2_count, ops::FUTEX_BITSET_MATCH_ANY, private)?
    } else {
        0
    };

    Ok(woken1 + woken2)
}

/// Remove waiter from queue
fn remove_waiter(key: &FutexKey, pid: Pid) {
    let mut table = FUTEX_TABLE.write();
    if let Some(queue) = table.get_mut(key) {
        queue.waiters.retain(|w| w.pid != pid);
        if queue.waiters.is_empty() {
            table.remove(key);
        }
    }
}

// ============================================================================
// Priority Inheritance Futex (PI)
// ============================================================================

/// PI futex state
struct PiFutex {
    /// Current owner
    owner: Option<Pid>,
    /// Waiters with priorities
    waiters: VecDeque<(Pid, i32)>, // (pid, priority)
}

/// PI futex table
static PI_TABLE: RwLock<BTreeMap<FutexKey, PiFutex>> = RwLock::new(BTreeMap::new());

/// Lock PI futex
pub fn futex_lock_pi(
    address: usize,
    timeout_ns: Option<u64>,
    private: bool,
) -> Result<(), FutexError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(FutexError::Invalid)?;

    // Validate address before any operations
    validate_user_address(address, core::mem::size_of::<u32>())?;

    let key = if private {
        FutexKey::private(address, pid)
    } else {
        FutexKey::shared(address)
    };

    let deadline = timeout_ns.map(|t| crate::time::monotonic_ns() + t);

    loop {
        {
            let mut table = PI_TABLE.write();
            let pi = table.entry(key).or_insert_with(|| PiFutex {
                owner: None,
                waiters: VecDeque::new(),
            });

            match pi.owner {
                None => {
                    // Lock is free, acquire it
                    pi.owner = Some(pid);

                    // Update userspace word (already validated)
                    let atomic = unsafe { (address as *const AtomicU32).as_ref() }
                        .ok_or(FutexError::Fault)?;
                    atomic.store(pid.0 as u32, Ordering::SeqCst);

                    return Ok(());
                }
                Some(owner) if owner == pid => {
                    // Deadlock - we already own it
                    return Err(FutexError::Deadlock);
                }
                Some(owner) => {
                    // Add to waiters with priority inheritance
                    let priority = crate::process::get(pid).map(|p| p.nice).unwrap_or(0);

                    // Insert sorted by priority
                    let pos = pi
                        .waiters
                        .iter()
                        .position(|(_, p)| *p > priority)
                        .unwrap_or(pi.waiters.len());
                    pi.waiters.insert(pos, (pid, priority));

                    // Boost owner priority if needed
                    if let Some(owner_proc) = crate::process::get(owner) {
                        if priority < owner_proc.nice {
                            // Would boost owner priority here
                        }
                    }
                }
            }
        }

        // Check timeout
        if let Some(dl) = deadline {
            if crate::time::monotonic_ns() >= dl {
                // Remove from waiters
                let mut table = PI_TABLE.write();
                if let Some(pi) = table.get_mut(&key) {
                    pi.waiters.retain(|(p, _)| *p != pid);
                }
                return Err(FutexError::TimedOut);
            }
        }

        // Wait
        crate::scheduler::schedule();

        // Check signals
        if crate::signal::has_pending_signals(pid) {
            let mut table = PI_TABLE.write();
            if let Some(pi) = table.get_mut(&key) {
                pi.waiters.retain(|(p, _)| *p != pid);
            }
            return Err(FutexError::Interrupted);
        }
    }
}

/// Unlock PI futex
pub fn futex_unlock_pi(address: usize, private: bool) -> Result<(), FutexError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(FutexError::Invalid)?;

    // Validate address before any operations
    validate_user_address(address, core::mem::size_of::<u32>())?;

    let key = if private {
        FutexKey::private(address, pid)
    } else {
        FutexKey::shared(address)
    };

    let mut table = PI_TABLE.write();
    let pi = table.get_mut(&key).ok_or(FutexError::Invalid)?;

    // Verify ownership
    if pi.owner != Some(pid) {
        return Err(FutexError::NotOwner);
    }

    // Transfer to highest priority waiter
    if let Some((new_owner, _)) = pi.waiters.pop_front() {
        pi.owner = Some(new_owner);

        // Update userspace word
        let atomic = unsafe { (address as *const AtomicU32).as_ref() }.ok_or(FutexError::Fault)?;
        atomic.store(new_owner.0 as u32, Ordering::SeqCst);

        // Wake new owner
        crate::scheduler::wake(new_owner);
    } else {
        pi.owner = None;

        // Update userspace word
        let atomic = unsafe { (address as *const AtomicU32).as_ref() }.ok_or(FutexError::Fault)?;
        atomic.store(0, Ordering::SeqCst);

        table.remove(&key);
    }

    Ok(())
}

// ============================================================================
// Robust Futex List
// ============================================================================

/// Robust list head (per-thread)
#[repr(C)]
pub struct RobustListHead {
    /// Pointer to first entry
    pub list: usize,
    /// Offset of futex word in entry
    pub futex_offset: i64,
    /// Pending operation
    pub list_op_pending: usize,
}

/// Per-process robust list registration
static ROBUST_LISTS: RwLock<BTreeMap<Pid, usize>> = RwLock::new(BTreeMap::new());

/// Set robust list
pub fn set_robust_list(head: usize, len: usize) -> Result<(), FutexError> {
    if len != core::mem::size_of::<RobustListHead>() {
        return Err(FutexError::Invalid);
    }

    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(FutexError::Invalid)?;

    ROBUST_LISTS.write().insert(pid, head);
    Ok(())
}

/// Get robust list
pub fn get_robust_list(pid: Pid) -> Option<usize> {
    ROBUST_LISTS.read().get(&pid).copied()
}

/// Handle thread exit - wake robust futexes
pub fn handle_thread_exit(pid: Pid) {
    if let Some(head_addr) = ROBUST_LISTS.write().remove(&pid) {
        // Walk robust list and wake waiters
        // Each entry contains a futex that needs to be marked as FUTEX_OWNER_DIED
        // and waiters woken

        let mut addr = head_addr;
        let mut count = 0;

        // Validate head address before walking
        if validate_user_address(head_addr, core::mem::size_of::<RobustListHead>()).is_err() {
            return;
        }

        while addr != 0 && count < 4096 {
            // Validate current address before reading
            if validate_user_address(addr, core::mem::size_of::<usize>()).is_err() {
                break;
            }

            // Read robust list entry
            let head = unsafe { (head_addr as *const RobustListHead).as_ref() };
            if head.is_none() {
                break;
            }

            let head = head.unwrap();

            // Validate futex_offset doesn't cause overflow
            let futex_addr = match addr.checked_add(head.futex_offset as usize) {
                Some(a) => a,
                None => break,
            };

            // Validate futex address before accessing
            if validate_user_address(futex_addr, core::mem::size_of::<u32>()).is_ok() {
                // Mark futex with FUTEX_OWNER_DIED
                if let Some(atomic) = unsafe { (futex_addr as *const AtomicU32).as_ref() } {
                    let old = atomic.fetch_or(0x40000000, Ordering::SeqCst); // FUTEX_OWNER_DIED
                    if old != 0 {
                        // Wake one waiter
                        let _ = futex_wake(futex_addr, 1, ops::FUTEX_BITSET_MATCH_ANY, true);
                    }
                }
            }

            // Validate next entry address before reading
            if validate_user_address(addr, core::mem::size_of::<usize>()).is_err() {
                break;
            }

            // Move to next entry
            let next = unsafe { (addr as *const usize).read_volatile() };
            if next == head.list || next == 0 {
                break;
            }
            addr = next;
            count += 1;
        }
    }
}

// ============================================================================
// Syscall Interface
// ============================================================================

/// Main futex syscall
pub fn sys_futex(
    uaddr: usize,
    op: i32,
    val: u32,
    timeout: usize,
    uaddr2: usize,
    val3: u32,
) -> isize {
    let cmd = op & 0x7F;
    let private = (op & ops::FUTEX_PRIVATE_FLAG) != 0;

    let timeout_ns = if timeout != 0 {
        // Would parse timespec
        Some(timeout as u64 * 1_000_000_000)
    } else {
        None
    };

    let result = match cmd {
        ops::FUTEX_WAIT => {
            futex_wait(uaddr, val, timeout_ns, ops::FUTEX_BITSET_MATCH_ANY, private).map(|_| 0)
        }
        ops::FUTEX_WAKE => {
            futex_wake(uaddr, val, ops::FUTEX_BITSET_MATCH_ANY, private).map(|n| n as i32)
        }
        ops::FUTEX_REQUEUE => {
            futex_requeue(uaddr, uaddr2, val, timeout as u32, None, private).map(|n| n as i32)
        }
        ops::FUTEX_CMP_REQUEUE => {
            futex_requeue(uaddr, uaddr2, val, timeout as u32, Some(val3), private).map(|n| n as i32)
        }
        ops::FUTEX_WAKE_OP => {
            futex_wake_op(uaddr, uaddr2, val, timeout as u32, val3, private).map(|n| n as i32)
        }
        ops::FUTEX_WAIT_BITSET => futex_wait(uaddr, val, timeout_ns, val3, private).map(|_| 0),
        ops::FUTEX_WAKE_BITSET => futex_wake(uaddr, val, val3, private).map(|n| n as i32),
        ops::FUTEX_LOCK_PI => futex_lock_pi(uaddr, timeout_ns, private).map(|_| 0),
        ops::FUTEX_UNLOCK_PI => futex_unlock_pi(uaddr, private).map(|_| 0),
        _ => Err(FutexError::Invalid),
    };

    match result {
        Ok(n) => n as isize,
        Err(e) => e.to_errno() as isize,
    }
}

/// Set robust list syscall
pub fn sys_set_robust_list(head: usize, len: usize) -> isize {
    match set_robust_list(head, len) {
        Ok(()) => 0,
        Err(e) => e.to_errno() as isize,
    }
}

/// Get robust list syscall
pub fn sys_get_robust_list(pid: i32, head_ptr: usize, len_ptr: usize) -> isize {
    let target_pid = if pid == 0 {
        crate::process::current().map(|p| p.pid)
    } else {
        Some(Pid(pid as u32))
    };

    let target_pid = match target_pid {
        Some(p) => p,
        None => return -22, // EINVAL
    };

    // Validate output pointers before writing
    if validate_user_address(head_ptr, core::mem::size_of::<usize>()).is_err() {
        return -14; // EFAULT
    }
    if validate_user_address(len_ptr, core::mem::size_of::<usize>()).is_err() {
        return -14; // EFAULT
    }

    match get_robust_list(target_pid) {
        Some(head) => {
            // Write head and len to user pointers (already validated)
            unsafe {
                *(head_ptr as *mut usize) = head;
                *(len_ptr as *mut usize) = core::mem::size_of::<RobustListHead>();
            }
            0
        }
        None => -22, // EINVAL
    }
}

/// Get statistics
pub fn get_stats() -> (u64, u64, u64, u64) {
    (
        STATS.waits.load(Ordering::Relaxed),
        STATS.wakes.load(Ordering::Relaxed),
        STATS.requeues.load(Ordering::Relaxed),
        STATS.timeouts.load(Ordering::Relaxed),
    )
}

/// Initialize futex subsystem
pub fn init() {
    crate::kprintln!("  Futex subsystem initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_futex_key() {
        let key1 = FutexKey::private(0x1000, Pid(1));
        let key2 = FutexKey::private(0x1000, Pid(2));
        let key3 = FutexKey::shared(0x1000);

        assert_ne!(key1, key2);
        assert_ne!(key1, key3);
    }
}
