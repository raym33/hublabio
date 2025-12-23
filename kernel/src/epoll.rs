//! Event Poll (epoll) - I/O Multiplexing
//!
//! Linux-compatible epoll interface for efficient I/O event notification.
//! Also includes poll() and select() implementations.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::{Mutex, RwLock};

use crate::process::Pid;

/// epoll events (bitmask)
pub mod events {
    /// Available for read
    pub const EPOLLIN: u32 = 0x001;
    /// Urgent data available
    pub const EPOLLPRI: u32 = 0x002;
    /// Available for write
    pub const EPOLLOUT: u32 = 0x004;
    /// Error condition
    pub const EPOLLERR: u32 = 0x008;
    /// Hang up
    pub const EPOLLHUP: u32 = 0x010;
    /// Invalid request
    pub const EPOLLNVAL: u32 = 0x020;
    /// Read hang up
    pub const EPOLLRDHUP: u32 = 0x2000;
    /// Exclusive wake up
    pub const EPOLLEXCLUSIVE: u32 = 1 << 28;
    /// Wake up once
    pub const EPOLLWAKEUP: u32 = 1 << 29;
    /// One-shot mode
    pub const EPOLLONESHOT: u32 = 1 << 30;
    /// Edge-triggered
    pub const EPOLLET: u32 = 1 << 31;
}

/// epoll operations
pub mod ops {
    pub const EPOLL_CTL_ADD: i32 = 1;
    pub const EPOLL_CTL_DEL: i32 = 2;
    pub const EPOLL_CTL_MOD: i32 = 3;
}

/// epoll_event structure
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct EpollEvent {
    /// Events mask
    pub events: u32,
    /// User data
    pub data: u64,
}

impl EpollEvent {
    pub fn new(events: u32, data: u64) -> Self {
        Self { events, data }
    }
}

/// Watched file descriptor entry
#[derive(Clone)]
struct EpollEntry {
    /// File descriptor
    fd: i32,
    /// Events to watch
    events: u32,
    /// User data
    data: u64,
    /// One-shot (disable after trigger)
    oneshot: bool,
    /// Edge-triggered mode
    edge_triggered: bool,
    /// Last reported events (for edge-triggered)
    last_events: u32,
}

/// epoll instance
pub struct EpollInstance {
    /// Instance ID
    id: u32,
    /// Owning process
    owner: Pid,
    /// Watched file descriptors
    entries: RwLock<BTreeMap<i32, EpollEntry>>,
    /// Ready file descriptors
    ready: Mutex<Vec<(i32, u32)>>,
    /// Wait queue
    waiters: crate::waitqueue::WaitQueue,
    /// Reference count
    refcount: AtomicU32,
}

impl EpollInstance {
    /// Create new epoll instance
    pub fn new(id: u32, owner: Pid) -> Self {
        Self {
            id,
            owner,
            entries: RwLock::new(BTreeMap::new()),
            ready: Mutex::new(Vec::new()),
            waiters: crate::waitqueue::WaitQueue::new(),
            refcount: AtomicU32::new(1),
        }
    }

    /// Add file descriptor to watch list
    pub fn add(&self, fd: i32, event: &EpollEvent) -> Result<(), EpollError> {
        let mut entries = self.entries.write();

        if entries.contains_key(&fd) {
            return Err(EpollError::Exists);
        }

        let entry = EpollEntry {
            fd,
            events: event.events,
            data: event.data,
            oneshot: event.events & events::EPOLLONESHOT != 0,
            edge_triggered: event.events & events::EPOLLET != 0,
            last_events: 0,
        };

        entries.insert(fd, entry);

        // Check if already ready
        self.check_fd_ready(fd);

        Ok(())
    }

    /// Modify watched events for file descriptor
    pub fn modify(&self, fd: i32, event: &EpollEvent) -> Result<(), EpollError> {
        let mut entries = self.entries.write();

        let entry = entries.get_mut(&fd).ok_or(EpollError::NotFound)?;

        entry.events = event.events;
        entry.data = event.data;
        entry.oneshot = event.events & events::EPOLLONESHOT != 0;
        entry.edge_triggered = event.events & events::EPOLLET != 0;

        Ok(())
    }

    /// Remove file descriptor from watch list
    pub fn delete(&self, fd: i32) -> Result<(), EpollError> {
        let mut entries = self.entries.write();

        if entries.remove(&fd).is_none() {
            return Err(EpollError::NotFound);
        }

        // Remove from ready list
        self.ready.lock().retain(|(f, _)| *f != fd);

        Ok(())
    }

    /// Wait for events
    pub fn wait(&self, events: &mut [EpollEvent], timeout_ms: i32) -> Result<usize, EpollError> {
        let deadline = if timeout_ms < 0 {
            None
        } else if timeout_ms == 0 {
            Some(0)
        } else {
            Some(crate::time::monotonic_ns() + (timeout_ms as u64) * 1_000_000)
        };

        loop {
            // Check all watched fds
            self.poll_all();

            // Get ready events
            let mut ready = self.ready.lock();
            let count = ready.len().min(events.len());

            if count > 0 {
                let entries = self.entries.read();

                for i in 0..count {
                    let (fd, revents) = ready[i];
                    if let Some(entry) = entries.get(&fd) {
                        events[i] = EpollEvent {
                            events: revents,
                            data: entry.data,
                        };
                    }
                }

                // Remove reported events
                ready.drain(..count);
                drop(ready);
                drop(entries);

                // Handle oneshot
                let mut entries = self.entries.write();
                for i in 0..count {
                    let fd = events[i].data as i32; // Assuming data contains fd
                    if let Some(entry) = entries.get_mut(&fd) {
                        if entry.oneshot {
                            entry.events = 0; // Disable
                        }
                    }
                }

                return Ok(count);
            }

            drop(ready);

            // Check timeout
            if let Some(dl) = deadline {
                if dl == 0 || crate::time::monotonic_ns() >= dl {
                    return Ok(0);
                }
            }

            // Wait for events
            if let Some(dl) = deadline {
                let remaining = dl.saturating_sub(crate::time::monotonic_ns());
                self.waiters.wait_timeout(remaining / 1_000_000);
            } else {
                self.waiters.wait();
            }

            // Check for signals
            if let Some(proc) = crate::process::current() {
                if crate::signal::has_pending_signals(proc.pid) {
                    return Err(EpollError::Interrupted);
                }
            }
        }
    }

    /// Poll all watched file descriptors
    fn poll_all(&self) {
        let entries = self.entries.read();
        let mut ready = self.ready.lock();

        for (fd, entry) in entries.iter() {
            if entry.events == 0 {
                continue; // Disabled (oneshot triggered)
            }

            let revents = poll_fd(*fd, entry.events);

            if revents != 0 {
                // Edge-triggered: only report if changed
                if entry.edge_triggered {
                    if revents != entry.last_events {
                        ready.push((*fd, revents));
                    }
                } else {
                    // Level-triggered: always report
                    if !ready.iter().any(|(f, _)| *f == *fd) {
                        ready.push((*fd, revents));
                    }
                }
            }
        }
    }

    /// Check if specific fd is ready
    fn check_fd_ready(&self, fd: i32) {
        let entries = self.entries.read();

        if let Some(entry) = entries.get(&fd) {
            let revents = poll_fd(fd, entry.events);
            if revents != 0 {
                let mut ready = self.ready.lock();
                if !ready.iter().any(|(f, _)| *f == fd) {
                    ready.push((fd, revents));
                    self.waiters.wake_all();
                }
            }
        }
    }

    /// Notify that fd has new events
    pub fn notify(&self, fd: i32, revents: u32) {
        let entries = self.entries.read();

        if let Some(entry) = entries.get(&fd) {
            let matched = entry.events & revents;
            if matched != 0 {
                let mut ready = self.ready.lock();
                if !ready.iter().any(|(f, _)| *f == fd) {
                    ready.push((fd, matched));
                    self.waiters.wake_all();
                }
            }
        }
    }
}

/// Poll a single file descriptor
fn poll_fd(fd: i32, events: u32) -> u32 {
    let mut revents = 0u32;

    // Check fd validity and get status
    if let Some(proc) = crate::process::current() {
        // Would check actual fd state through VFS
        // For now, simulate basic behavior

        // Check if fd is valid
        if fd < 0 || fd >= 1024 {
            return events::EPOLLNVAL;
        }

        // Check read/write availability (would query VFS)
        // Placeholder: assume always ready
        if events & events::EPOLLIN != 0 {
            revents |= events::EPOLLIN;
        }
        if events & events::EPOLLOUT != 0 {
            revents |= events::EPOLLOUT;
        }
    }

    revents
}

/// epoll error
#[derive(Clone, Copy, Debug)]
pub enum EpollError {
    /// fd already in set
    Exists,
    /// fd not found
    NotFound,
    /// Invalid argument
    Invalid,
    /// Too many fds
    TooMany,
    /// Interrupted by signal
    Interrupted,
    /// Bad fd
    BadFd,
    /// Permission denied
    Permission,
}

impl EpollError {
    pub fn to_errno(&self) -> i32 {
        match self {
            EpollError::Exists => -17,     // EEXIST
            EpollError::NotFound => -2,    // ENOENT
            EpollError::Invalid => -22,    // EINVAL
            EpollError::TooMany => -24,    // EMFILE
            EpollError::Interrupted => -4, // EINTR
            EpollError::BadFd => -9,       // EBADF
            EpollError::Permission => -1,  // EPERM
        }
    }
}

// ============================================================================
// Global State
// ============================================================================

/// All epoll instances
static EPOLL_INSTANCES: RwLock<BTreeMap<u32, Arc<EpollInstance>>> = RwLock::new(BTreeMap::new());

/// Next instance ID
static NEXT_EPOLL_ID: AtomicU32 = AtomicU32::new(0);

/// Create new epoll instance
pub fn epoll_create() -> Result<i32, EpollError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(EpollError::Permission)?;

    let id = NEXT_EPOLL_ID.fetch_add(1, Ordering::SeqCst);
    let instance = Arc::new(EpollInstance::new(id, pid));

    EPOLL_INSTANCES.write().insert(id, instance);

    // Return as fd (would be registered in process fd table)
    Ok(id as i32 + 1000) // Offset to avoid conflict with regular fds
}

/// Get epoll instance by fd
fn get_epoll(epfd: i32) -> Option<Arc<EpollInstance>> {
    if epfd < 1000 {
        return None;
    }
    let id = (epfd - 1000) as u32;
    EPOLL_INSTANCES.read().get(&id).cloned()
}

/// Close epoll instance
pub fn epoll_close(epfd: i32) -> Result<(), EpollError> {
    if epfd < 1000 {
        return Err(EpollError::BadFd);
    }
    let id = (epfd - 1000) as u32;
    EPOLL_INSTANCES
        .write()
        .remove(&id)
        .ok_or(EpollError::BadFd)?;
    Ok(())
}

// ============================================================================
// Syscalls
// ============================================================================

/// epoll_create syscall
pub fn sys_epoll_create(size: i32) -> isize {
    if size <= 0 {
        return -22; // EINVAL
    }
    match epoll_create() {
        Ok(fd) => fd as isize,
        Err(e) => e.to_errno() as isize,
    }
}

/// epoll_create1 syscall
pub fn sys_epoll_create1(flags: i32) -> isize {
    // EPOLL_CLOEXEC = 0x80000
    match epoll_create() {
        Ok(fd) => fd as isize,
        Err(e) => e.to_errno() as isize,
    }
}

/// epoll_ctl syscall
pub fn sys_epoll_ctl(epfd: i32, op: i32, fd: i32, event: *const EpollEvent) -> isize {
    let instance = match get_epoll(epfd) {
        Some(i) => i,
        None => return -9, // EBADF
    };

    match op {
        ops::EPOLL_CTL_ADD => {
            if event.is_null() {
                return -14; // EFAULT
            }
            let ev = unsafe { &*event };
            match instance.add(fd, ev) {
                Ok(()) => 0,
                Err(e) => e.to_errno() as isize,
            }
        }
        ops::EPOLL_CTL_DEL => match instance.delete(fd) {
            Ok(()) => 0,
            Err(e) => e.to_errno() as isize,
        },
        ops::EPOLL_CTL_MOD => {
            if event.is_null() {
                return -14; // EFAULT
            }
            let ev = unsafe { &*event };
            match instance.modify(fd, ev) {
                Ok(()) => 0,
                Err(e) => e.to_errno() as isize,
            }
        }
        _ => -22, // EINVAL
    }
}

/// epoll_wait syscall
pub fn sys_epoll_wait(epfd: i32, events: *mut EpollEvent, maxevents: i32, timeout: i32) -> isize {
    if maxevents <= 0 || events.is_null() {
        return -22; // EINVAL
    }

    let instance = match get_epoll(epfd) {
        Some(i) => i,
        None => return -9, // EBADF
    };

    let event_slice = unsafe { core::slice::from_raw_parts_mut(events, maxevents as usize) };

    match instance.wait(event_slice, timeout) {
        Ok(n) => n as isize,
        Err(e) => e.to_errno() as isize,
    }
}

/// epoll_pwait syscall
pub fn sys_epoll_pwait(
    epfd: i32,
    events: *mut EpollEvent,
    maxevents: i32,
    timeout: i32,
    sigmask: *const u64,
) -> isize {
    // Would set signal mask around wait
    sys_epoll_wait(epfd, events, maxevents, timeout)
}

// ============================================================================
// poll() Implementation
// ============================================================================

/// pollfd structure
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PollFd {
    /// File descriptor
    pub fd: i32,
    /// Events to poll for
    pub events: i16,
    /// Returned events
    pub revents: i16,
}

/// poll events
pub mod poll_events {
    pub const POLLIN: i16 = 0x0001;
    pub const POLLPRI: i16 = 0x0002;
    pub const POLLOUT: i16 = 0x0004;
    pub const POLLERR: i16 = 0x0008;
    pub const POLLHUP: i16 = 0x0010;
    pub const POLLNVAL: i16 = 0x0020;
    pub const POLLRDNORM: i16 = 0x0040;
    pub const POLLRDBAND: i16 = 0x0080;
    pub const POLLWRNORM: i16 = 0x0100;
    pub const POLLWRBAND: i16 = 0x0200;
}

/// poll syscall
pub fn sys_poll(fds: *mut PollFd, nfds: u32, timeout: i32) -> isize {
    if fds.is_null() && nfds > 0 {
        return -14; // EFAULT
    }

    let deadline = if timeout < 0 {
        None
    } else if timeout == 0 {
        Some(0)
    } else {
        Some(crate::time::monotonic_ns() + (timeout as u64) * 1_000_000)
    };

    let fds_slice = unsafe { core::slice::from_raw_parts_mut(fds, nfds as usize) };

    loop {
        let mut ready_count = 0i32;

        for pollfd in fds_slice.iter_mut() {
            pollfd.revents = 0;

            if pollfd.fd < 0 {
                continue;
            }

            // Check fd state
            let mut revents: i16 = 0;

            // Would check actual fd state through VFS
            // Placeholder: check basic availability
            if pollfd.events & poll_events::POLLIN != 0 {
                // Check if readable
                revents |= poll_events::POLLIN;
            }
            if pollfd.events & poll_events::POLLOUT != 0 {
                // Check if writable
                revents |= poll_events::POLLOUT;
            }

            if revents != 0 {
                pollfd.revents = revents;
                ready_count += 1;
            }
        }

        if ready_count > 0 {
            return ready_count as isize;
        }

        // Check timeout
        if let Some(dl) = deadline {
            if dl == 0 || crate::time::monotonic_ns() >= dl {
                return 0;
            }
        }

        // Sleep briefly
        crate::scheduler::schedule();

        // Check signals
        if let Some(proc) = crate::process::current() {
            if crate::signal::has_pending_signals(proc.pid) {
                return -4; // EINTR
            }
        }
    }
}

/// ppoll syscall
pub fn sys_ppoll(
    fds: *mut PollFd,
    nfds: u32,
    tmo_p: *const TimeSpec,
    sigmask: *const u64,
) -> isize {
    let timeout = if tmo_p.is_null() {
        -1
    } else {
        let ts = unsafe { &*tmo_p };
        ((ts.tv_sec * 1000) + (ts.tv_nsec / 1_000_000)) as i32
    };

    sys_poll(fds, nfds, timeout)
}

/// timespec structure
#[repr(C)]
pub struct TimeSpec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

// ============================================================================
// select() Implementation
// ============================================================================

/// fd_set for select
#[derive(Clone)]
#[repr(C)]
pub struct FdSet {
    /// Bitmask of file descriptors
    pub fds_bits: [u64; 16], // Support up to 1024 fds
}

impl FdSet {
    pub fn new() -> Self {
        Self { fds_bits: [0; 16] }
    }

    pub fn set(&mut self, fd: i32) {
        if fd >= 0 && fd < 1024 {
            let idx = fd as usize / 64;
            let bit = fd as usize % 64;
            self.fds_bits[idx] |= 1u64 << bit;
        }
    }

    pub fn clear(&mut self, fd: i32) {
        if fd >= 0 && fd < 1024 {
            let idx = fd as usize / 64;
            let bit = fd as usize % 64;
            self.fds_bits[idx] &= !(1u64 << bit);
        }
    }

    pub fn is_set(&self, fd: i32) -> bool {
        if fd >= 0 && fd < 1024 {
            let idx = fd as usize / 64;
            let bit = fd as usize % 64;
            (self.fds_bits[idx] & (1u64 << bit)) != 0
        } else {
            false
        }
    }

    pub fn zero(&mut self) {
        self.fds_bits = [0; 16];
    }
}

/// timeval structure
#[repr(C)]
pub struct TimeVal {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

/// select syscall
pub fn sys_select(
    nfds: i32,
    readfds: *mut FdSet,
    writefds: *mut FdSet,
    exceptfds: *mut FdSet,
    timeout: *mut TimeVal,
) -> isize {
    let deadline = if timeout.is_null() {
        None
    } else {
        let tv = unsafe { &*timeout };
        if tv.tv_sec == 0 && tv.tv_usec == 0 {
            Some(0)
        } else {
            Some(
                crate::time::monotonic_ns()
                    + (tv.tv_sec as u64) * 1_000_000_000
                    + (tv.tv_usec as u64) * 1000,
            )
        }
    };

    loop {
        let mut ready_count = 0i32;

        for fd in 0..nfds {
            // Check readfds
            if !readfds.is_null() {
                let rset = unsafe { &mut *readfds };
                if rset.is_set(fd) {
                    // Check if readable - placeholder
                    ready_count += 1;
                }
            }

            // Check writefds
            if !writefds.is_null() {
                let wset = unsafe { &mut *writefds };
                if wset.is_set(fd) {
                    // Check if writable - placeholder
                    ready_count += 1;
                }
            }

            // Check exceptfds
            if !exceptfds.is_null() {
                let eset = unsafe { &mut *exceptfds };
                if eset.is_set(fd) {
                    // Check for exceptions
                    eset.clear(fd); // No exceptions
                }
            }
        }

        if ready_count > 0 {
            return ready_count as isize;
        }

        // Check timeout
        if let Some(dl) = deadline {
            if dl == 0 || crate::time::monotonic_ns() >= dl {
                return 0;
            }
        }

        // Sleep
        crate::scheduler::schedule();

        // Check signals
        if let Some(proc) = crate::process::current() {
            if crate::signal::has_pending_signals(proc.pid) {
                return -4; // EINTR
            }
        }
    }
}

/// pselect6 syscall
pub fn sys_pselect6(
    nfds: i32,
    readfds: *mut FdSet,
    writefds: *mut FdSet,
    exceptfds: *mut FdSet,
    timeout: *const TimeSpec,
    sigmask: *const u64,
) -> isize {
    let tv = if timeout.is_null() {
        core::ptr::null_mut()
    } else {
        let ts = unsafe { &*timeout };
        let mut tv = TimeVal {
            tv_sec: ts.tv_sec,
            tv_usec: ts.tv_nsec / 1000,
        };
        &mut tv as *mut TimeVal
    };

    sys_select(nfds, readfds, writefds, exceptfds, tv)
}

/// Initialize epoll subsystem
pub fn init() {
    crate::kprintln!("  epoll/poll/select initialized");
}
