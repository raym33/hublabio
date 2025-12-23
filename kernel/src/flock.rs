//! File Locking
//!
//! Provides POSIX file locking (flock/fcntl) support.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::RwLock;

use crate::process::Pid;
use crate::waitqueue::WaitQueue;

/// Lock type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockType {
    /// No lock
    Unlock = 0,
    /// Shared (read) lock
    Shared = 1,
    /// Exclusive (write) lock
    Exclusive = 2,
}

/// flock operation flags
pub mod flock_ops {
    pub const LOCK_SH: i32 = 1;  // Shared lock
    pub const LOCK_EX: i32 = 2;  // Exclusive lock
    pub const LOCK_NB: i32 = 4;  // Non-blocking
    pub const LOCK_UN: i32 = 8;  // Unlock
}

/// fcntl lock command
pub mod fcntl_cmd {
    pub const F_GETLK: i32 = 5;   // Get lock
    pub const F_SETLK: i32 = 6;   // Set lock (non-blocking)
    pub const F_SETLKW: i32 = 7;  // Set lock (blocking)
}

/// File lock (for fcntl byte-range locks)
#[derive(Clone, Debug)]
pub struct FileLock {
    /// Lock type
    pub lock_type: LockType,
    /// Start offset
    pub start: u64,
    /// Length (0 = to end of file)
    pub len: u64,
    /// Process holding the lock
    pub pid: Pid,
    /// Whence (SEEK_SET, SEEK_CUR, SEEK_END)
    pub whence: i16,
}

/// flock structure (POSIX)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Flock {
    pub l_type: i16,   // Lock type
    pub l_whence: i16, // How to interpret l_start
    pub l_start: i64,  // Offset where lock begins
    pub l_len: i64,    // Size of the locked area; 0 = EOF
    pub l_pid: i32,    // PID of process holding lock
}

impl Flock {
    pub fn lock_type(&self) -> LockType {
        match self.l_type {
            0 => LockType::Shared,    // F_RDLCK
            1 => LockType::Exclusive, // F_WRLCK
            2 => LockType::Unlock,    // F_UNLCK
            _ => LockType::Unlock,
        }
    }
}

/// File lock state
struct FileLockState {
    /// Whole-file locks (flock style)
    flock_holders: Vec<(Pid, LockType)>,
    /// Byte-range locks (fcntl style)
    range_locks: Vec<FileLock>,
    /// Wait queue for blocked lockers
    wait_queue: WaitQueue,
}

impl FileLockState {
    fn new() -> Self {
        Self {
            flock_holders: Vec::new(),
            range_locks: Vec::new(),
            wait_queue: WaitQueue::new(),
        }
    }
}

/// Global file lock table (inode -> lock state)
static FILE_LOCKS: RwLock<BTreeMap<u64, FileLockState>> = RwLock::new(BTreeMap::new());

/// Locking error
#[derive(Clone, Debug)]
pub enum LockError {
    /// Lock would block
    WouldBlock,
    /// Lock not held
    NotLocked,
    /// Invalid argument
    Invalid,
    /// Deadlock detected
    Deadlock,
    /// Interrupted
    Interrupted,
}

/// Apply flock to file
pub fn flock(inode: u64, operation: i32) -> Result<(), LockError> {
    let lock_type = if operation & flock_ops::LOCK_UN != 0 {
        LockType::Unlock
    } else if operation & flock_ops::LOCK_EX != 0 {
        LockType::Exclusive
    } else if operation & flock_ops::LOCK_SH != 0 {
        LockType::Shared
    } else {
        return Err(LockError::Invalid);
    };

    let non_blocking = operation & flock_ops::LOCK_NB != 0;

    let pid = crate::process::current()
        .map(|p| p.pid)
        .unwrap_or(Pid(0));

    loop {
        let result = try_flock(inode, lock_type, pid);

        match result {
            Ok(()) => return Ok(()),
            Err(LockError::WouldBlock) if !non_blocking => {
                // Block and retry
                let locks = FILE_LOCKS.read();
                if let Some(state) = locks.get(&inode) {
                    drop(locks);

                    // Wait on the queue
                    let locks = FILE_LOCKS.read();
                    if let Some(state) = locks.get(&inode) {
                        state.wait_queue.wait_interruptible();
                    }
                }
            }
            Err(e) => return Err(e),
        }
    }
}

/// Try to acquire flock (non-blocking)
fn try_flock(inode: u64, lock_type: LockType, pid: Pid) -> Result<(), LockError> {
    let mut locks = FILE_LOCKS.write();
    let state = locks.entry(inode).or_insert_with(FileLockState::new);

    match lock_type {
        LockType::Unlock => {
            // Remove our lock
            state.flock_holders.retain(|(p, _)| *p != pid);
            // Wake up waiters
            state.wait_queue.wake_all();
            Ok(())
        }

        LockType::Shared => {
            // Check for conflicting exclusive locks from other processes
            for (holder_pid, holder_type) in &state.flock_holders {
                if *holder_pid != pid && *holder_type == LockType::Exclusive {
                    return Err(LockError::WouldBlock);
                }
            }

            // Remove any existing lock from this process
            state.flock_holders.retain(|(p, _)| *p != pid);

            // Add shared lock
            state.flock_holders.push((pid, LockType::Shared));
            Ok(())
        }

        LockType::Exclusive => {
            // Check for any locks from other processes
            for (holder_pid, _) in &state.flock_holders {
                if *holder_pid != pid {
                    return Err(LockError::WouldBlock);
                }
            }

            // Remove any existing lock from this process
            state.flock_holders.retain(|(p, _)| *p != pid);

            // Add exclusive lock
            state.flock_holders.push((pid, LockType::Exclusive));
            Ok(())
        }
    }
}

/// Apply fcntl lock (byte-range)
pub fn fcntl_lock(inode: u64, cmd: i32, flock: &mut Flock) -> Result<(), LockError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .unwrap_or(Pid(0));

    match cmd {
        fcntl_cmd::F_GETLK => {
            // Check if lock would conflict
            let locks = FILE_LOCKS.read();

            if let Some(state) = locks.get(&inode) {
                let start = flock.l_start as u64;
                let len = if flock.l_len == 0 { u64::MAX } else { flock.l_len as u64 };
                let end = start.saturating_add(len);

                for lock in &state.range_locks {
                    if lock.pid == pid {
                        continue;
                    }

                    let lock_end = if lock.len == 0 {
                        u64::MAX
                    } else {
                        lock.start.saturating_add(lock.len)
                    };

                    // Check overlap
                    if start < lock_end && end > lock.start {
                        // Check compatibility
                        let compatible = match (flock.lock_type(), lock.lock_type) {
                            (LockType::Shared, LockType::Shared) => true,
                            _ => false,
                        };

                        if !compatible {
                            // Return the conflicting lock info
                            flock.l_type = match lock.lock_type {
                                LockType::Shared => 0,
                                LockType::Exclusive => 1,
                                LockType::Unlock => 2,
                            };
                            flock.l_whence = lock.whence;
                            flock.l_start = lock.start as i64;
                            flock.l_len = lock.len as i64;
                            flock.l_pid = lock.pid.0 as i32;
                            return Ok(());
                        }
                    }
                }
            }

            // No conflict - return unlock
            flock.l_type = 2; // F_UNLCK
            Ok(())
        }

        fcntl_cmd::F_SETLK => {
            // Non-blocking lock attempt
            apply_range_lock(inode, flock, pid, false)
        }

        fcntl_cmd::F_SETLKW => {
            // Blocking lock attempt
            apply_range_lock(inode, flock, pid, true)
        }

        _ => Err(LockError::Invalid),
    }
}

/// Apply byte-range lock
fn apply_range_lock(
    inode: u64,
    flock: &Flock,
    pid: Pid,
    blocking: bool,
) -> Result<(), LockError> {
    let lock_type = flock.lock_type();
    let start = flock.l_start as u64;
    let len = if flock.l_len == 0 { 0 } else { flock.l_len as u64 };

    loop {
        let result = try_range_lock(inode, lock_type, start, len, flock.l_whence, pid);

        match result {
            Ok(()) => return Ok(()),
            Err(LockError::WouldBlock) if blocking => {
                // Block and retry
                let locks = FILE_LOCKS.read();
                if let Some(state) = locks.get(&inode) {
                    drop(locks);

                    let locks = FILE_LOCKS.read();
                    if let Some(state) = locks.get(&inode) {
                        if !state.wait_queue.wait_interruptible() {
                            return Err(LockError::Interrupted);
                        }
                    }
                }
            }
            Err(e) => return Err(e),
        }
    }
}

/// Try to acquire byte-range lock
fn try_range_lock(
    inode: u64,
    lock_type: LockType,
    start: u64,
    len: u64,
    whence: i16,
    pid: Pid,
) -> Result<(), LockError> {
    let mut locks = FILE_LOCKS.write();
    let state = locks.entry(inode).or_insert_with(FileLockState::new);

    let end = if len == 0 { u64::MAX } else { start.saturating_add(len) };

    if lock_type == LockType::Unlock {
        // Remove matching locks from this process
        state.range_locks.retain(|lock| {
            if lock.pid != pid {
                return true;
            }

            let lock_end = if lock.len == 0 {
                u64::MAX
            } else {
                lock.start.saturating_add(lock.len)
            };

            // Remove if fully contained in unlock range
            !(start <= lock.start && end >= lock_end)
        });

        state.wait_queue.wake_all();
        return Ok(());
    }

    // Check for conflicts
    for lock in &state.range_locks {
        if lock.pid == pid {
            continue;
        }

        let lock_end = if lock.len == 0 {
            u64::MAX
        } else {
            lock.start.saturating_add(lock.len)
        };

        // Check overlap
        if start < lock_end && end > lock.start {
            // Check compatibility
            let compatible = match (lock_type, lock.lock_type) {
                (LockType::Shared, LockType::Shared) => true,
                _ => false,
            };

            if !compatible {
                return Err(LockError::WouldBlock);
            }
        }
    }

    // Remove any existing locks from this process in the range
    state.range_locks.retain(|lock| {
        if lock.pid != pid {
            return true;
        }

        let lock_end = if lock.len == 0 {
            u64::MAX
        } else {
            lock.start.saturating_add(lock.len)
        };

        // Keep if no overlap
        start >= lock_end || end <= lock.start
    });

    // Add new lock
    state.range_locks.push(FileLock {
        lock_type,
        start,
        len,
        pid,
        whence,
    });

    Ok(())
}

/// Release all locks held by a process
pub fn release_all(pid: Pid) {
    let mut locks = FILE_LOCKS.write();

    for (_, state) in locks.iter_mut() {
        let had_locks = !state.flock_holders.is_empty() || !state.range_locks.is_empty();

        state.flock_holders.retain(|(p, _)| *p != pid);
        state.range_locks.retain(|lock| lock.pid != pid);

        if had_locks {
            state.wait_queue.wake_all();
        }
    }
}

/// Check if file has any locks
pub fn is_locked(inode: u64) -> bool {
    let locks = FILE_LOCKS.read();

    if let Some(state) = locks.get(&inode) {
        !state.flock_holders.is_empty() || !state.range_locks.is_empty()
    } else {
        false
    }
}

/// Get lock count for debugging
pub fn lock_count(inode: u64) -> (usize, usize) {
    let locks = FILE_LOCKS.read();

    if let Some(state) = locks.get(&inode) {
        (state.flock_holders.len(), state.range_locks.len())
    } else {
        (0, 0)
    }
}

/// Initialize file locking subsystem
pub fn init() {
    crate::kprintln!("  File locking initialized");
}
