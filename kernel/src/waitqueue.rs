//! Wait Queues
//!
//! Provides blocking wait/wake functionality for processes.

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::Mutex;

use crate::process::Pid;

/// Wait queue ID
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaitQueueId(pub u64);

/// Global wait queue ID counter
static WAITQUEUE_ID: AtomicU64 = AtomicU64::new(1);

/// Waiter entry
#[derive(Clone, Debug)]
struct Waiter {
    pid: Pid,
    woken: AtomicBool,
}

/// Wait queue for blocking operations
pub struct WaitQueue {
    id: WaitQueueId,
    waiters: Mutex<VecDeque<Waiter>>,
}

impl WaitQueue {
    /// Create a new wait queue
    pub fn new() -> Self {
        Self {
            id: WaitQueueId(WAITQUEUE_ID.fetch_add(1, Ordering::SeqCst)),
            waiters: Mutex::new(VecDeque::new()),
        }
    }

    /// Get wait queue ID
    pub fn id(&self) -> WaitQueueId {
        self.id
    }

    /// Wait on this queue (blocking)
    pub fn wait(&self) {
        if let Some(proc) = crate::process::current() {
            let waiter = Waiter {
                pid: proc.pid,
                woken: AtomicBool::new(false),
            };

            self.waiters.lock().push_back(waiter.clone());

            // Block the current process
            proc.block(crate::process::BlockReason::WaitQueue);

            // Yield to scheduler
            crate::scheduler::schedule();

            // Remove ourselves from waiters when woken
            self.waiters.lock().retain(|w| w.pid != proc.pid);
        }
    }

    /// Wait with timeout (in milliseconds)
    pub fn wait_timeout(&self, timeout_ms: u64) -> bool {
        if let Some(proc) = crate::process::current() {
            let waiter = Waiter {
                pid: proc.pid,
                woken: AtomicBool::new(false),
            };

            self.waiters.lock().push_back(waiter.clone());

            // Block with timeout
            proc.sleep(timeout_ms);

            // Yield to scheduler
            crate::scheduler::schedule();

            // Check if we were woken or timed out
            let woken = waiter.woken.load(Ordering::SeqCst);

            // Remove ourselves from waiters
            self.waiters.lock().retain(|w| w.pid != proc.pid);

            woken
        } else {
            false
        }
    }

    /// Wait interruptible (can be interrupted by signals)
    pub fn wait_interruptible(&self) -> bool {
        if let Some(proc) = crate::process::current() {
            let waiter = Waiter {
                pid: proc.pid,
                woken: AtomicBool::new(false),
            };

            self.waiters.lock().push_back(waiter.clone());

            // Block interruptibly
            proc.block(crate::process::BlockReason::Interruptible);

            // Yield to scheduler
            crate::scheduler::schedule();

            // Check if we have pending signals
            let has_signals = crate::signal::has_pending(proc.pid);

            // Remove ourselves from waiters
            self.waiters.lock().retain(|w| w.pid != proc.pid);

            // Return true if woken by wake_up, false if interrupted
            waiter.woken.load(Ordering::SeqCst) && !has_signals
        } else {
            false
        }
    }

    /// Wake up one waiter
    pub fn wake_one(&self) -> bool {
        let mut waiters = self.waiters.lock();

        if let Some(waiter) = waiters.pop_front() {
            waiter.woken.store(true, Ordering::SeqCst);

            // Wake up the process
            if let Some(proc) = crate::process::get(waiter.pid) {
                proc.wake();
            }

            return true;
        }

        false
    }

    /// Wake up all waiters
    pub fn wake_all(&self) -> usize {
        let mut waiters = self.waiters.lock();
        let count = waiters.len();

        for waiter in waiters.drain(..) {
            waiter.woken.store(true, Ordering::SeqCst);

            if let Some(proc) = crate::process::get(waiter.pid) {
                proc.wake();
            }
        }

        count
    }

    /// Wake up N waiters
    pub fn wake_n(&self, n: usize) -> usize {
        let mut waiters = self.waiters.lock();
        let count = n.min(waiters.len());

        for _ in 0..count {
            if let Some(waiter) = waiters.pop_front() {
                waiter.woken.store(true, Ordering::SeqCst);

                if let Some(proc) = crate::process::get(waiter.pid) {
                    proc.wake();
                }
            }
        }

        count
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.waiters.lock().is_empty()
    }

    /// Get number of waiters
    pub fn len(&self) -> usize {
        self.waiters.lock().len()
    }

    /// Get list of waiting PIDs
    pub fn waiting_pids(&self) -> Vec<Pid> {
        self.waiters.lock().iter().map(|w| w.pid).collect()
    }
}

impl Default for WaitQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Condition variable (higher-level abstraction)
pub struct CondVar {
    queue: WaitQueue,
}

impl CondVar {
    /// Create new condition variable
    pub fn new() -> Self {
        Self {
            queue: WaitQueue::new(),
        }
    }

    /// Wait for condition (with mutex-like behavior)
    pub fn wait<T, F>(&self, guard: &mut T, condition: F)
    where
        F: Fn(&T) -> bool,
    {
        while !condition(guard) {
            self.queue.wait();
        }
    }

    /// Wait with timeout
    pub fn wait_timeout<T, F>(&self, guard: &mut T, timeout_ms: u64, condition: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        let start = crate::time::system_time_ms();

        while !condition(guard) {
            let elapsed = crate::time::system_time_ms() - start;
            if elapsed >= timeout_ms {
                return false;
            }

            let remaining = timeout_ms - elapsed;
            if !self.queue.wait_timeout(remaining) {
                return false;
            }
        }

        true
    }

    /// Signal one waiter
    pub fn signal(&self) {
        self.queue.wake_one();
    }

    /// Signal all waiters
    pub fn broadcast(&self) {
        self.queue.wake_all();
    }
}

impl Default for CondVar {
    fn default() -> Self {
        Self::new()
    }
}

/// Completion (one-shot synchronization)
pub struct Completion {
    done: AtomicBool,
    queue: WaitQueue,
}

impl Completion {
    /// Create new completion
    pub fn new() -> Self {
        Self {
            done: AtomicBool::new(false),
            queue: WaitQueue::new(),
        }
    }

    /// Wait for completion
    pub fn wait(&self) {
        while !self.done.load(Ordering::SeqCst) {
            self.queue.wait();
        }
    }

    /// Wait with timeout (returns true if completed)
    pub fn wait_timeout(&self, timeout_ms: u64) -> bool {
        if self.done.load(Ordering::SeqCst) {
            return true;
        }

        self.queue.wait_timeout(timeout_ms);
        self.done.load(Ordering::SeqCst)
    }

    /// Mark as complete and wake all waiters
    pub fn complete(&self) {
        self.done.store(true, Ordering::SeqCst);
        self.queue.wake_all();
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }

    /// Reset for reuse
    pub fn reset(&self) {
        self.done.store(false, Ordering::SeqCst);
    }
}

impl Default for Completion {
    fn default() -> Self {
        Self::new()
    }
}

/// Event (can be set/reset multiple times)
pub struct Event {
    signaled: AtomicBool,
    auto_reset: bool,
    queue: WaitQueue,
}

impl Event {
    /// Create new manual-reset event
    pub fn new_manual() -> Self {
        Self {
            signaled: AtomicBool::new(false),
            auto_reset: false,
            queue: WaitQueue::new(),
        }
    }

    /// Create new auto-reset event
    pub fn new_auto() -> Self {
        Self {
            signaled: AtomicBool::new(false),
            auto_reset: true,
            queue: WaitQueue::new(),
        }
    }

    /// Wait for event to be signaled
    pub fn wait(&self) {
        loop {
            if self.signaled.load(Ordering::SeqCst) {
                if self.auto_reset {
                    self.signaled.store(false, Ordering::SeqCst);
                }
                return;
            }
            self.queue.wait();
        }
    }

    /// Wait with timeout
    pub fn wait_timeout(&self, timeout_ms: u64) -> bool {
        if self.signaled.load(Ordering::SeqCst) {
            if self.auto_reset {
                self.signaled.store(false, Ordering::SeqCst);
            }
            return true;
        }

        self.queue.wait_timeout(timeout_ms);

        if self.signaled.load(Ordering::SeqCst) {
            if self.auto_reset {
                self.signaled.store(false, Ordering::SeqCst);
            }
            true
        } else {
            false
        }
    }

    /// Signal the event
    pub fn set(&self) {
        self.signaled.store(true, Ordering::SeqCst);
        if self.auto_reset {
            self.queue.wake_one();
        } else {
            self.queue.wake_all();
        }
    }

    /// Reset the event
    pub fn reset(&self) {
        self.signaled.store(false, Ordering::SeqCst);
    }

    /// Check if signaled
    pub fn is_set(&self) -> bool {
        self.signaled.load(Ordering::SeqCst)
    }
}

/// Initialize wait queue subsystem
pub fn init() {
    crate::kprintln!("  Wait queue subsystem initialized");
}
