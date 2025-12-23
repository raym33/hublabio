//! Resource Limits (rlimit)
//!
//! POSIX resource limit enforcement.
//! Controls per-process resource consumption.

use alloc::collections::BTreeMap;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::RwLock;

use crate::process::Pid;

/// Resource limit types
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub enum Resource {
    /// CPU time in seconds
    Cpu = 0,
    /// Maximum file size (bytes)
    Fsize = 1,
    /// Data segment size (bytes)
    Data = 2,
    /// Stack size (bytes)
    Stack = 3,
    /// Core file size (bytes)
    Core = 4,
    /// Resident set size (bytes)
    Rss = 5,
    /// Number of processes
    Nproc = 6,
    /// Number of open files
    Nofile = 7,
    /// Locked-in-memory address space (bytes)
    Memlock = 8,
    /// Address space limit (bytes)
    As = 9,
    /// Number of file locks
    Locks = 10,
    /// Number of pending signals
    Sigpending = 11,
    /// POSIX message queue size (bytes)
    Msgqueue = 12,
    /// Max nice value
    Nice = 13,
    /// Max real-time priority
    Rtprio = 14,
    /// Max real-time timeout (microseconds)
    Rttime = 15,
}

impl Resource {
    /// Maximum resource type value
    pub const MAX: u32 = 16;

    /// Convert from u32
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Resource::Cpu),
            1 => Some(Resource::Fsize),
            2 => Some(Resource::Data),
            3 => Some(Resource::Stack),
            4 => Some(Resource::Core),
            5 => Some(Resource::Rss),
            6 => Some(Resource::Nproc),
            7 => Some(Resource::Nofile),
            8 => Some(Resource::Memlock),
            9 => Some(Resource::As),
            10 => Some(Resource::Locks),
            11 => Some(Resource::Sigpending),
            12 => Some(Resource::Msgqueue),
            13 => Some(Resource::Nice),
            14 => Some(Resource::Rtprio),
            15 => Some(Resource::Rttime),
            _ => None,
        }
    }

    /// Get resource name
    pub fn name(&self) -> &'static str {
        match self {
            Resource::Cpu => "cpu",
            Resource::Fsize => "fsize",
            Resource::Data => "data",
            Resource::Stack => "stack",
            Resource::Core => "core",
            Resource::Rss => "rss",
            Resource::Nproc => "nproc",
            Resource::Nofile => "nofile",
            Resource::Memlock => "memlock",
            Resource::As => "as",
            Resource::Locks => "locks",
            Resource::Sigpending => "sigpending",
            Resource::Msgqueue => "msgqueue",
            Resource::Nice => "nice",
            Resource::Rtprio => "rtprio",
            Resource::Rttime => "rttime",
        }
    }
}

/// Unlimited value
pub const RLIM_INFINITY: u64 = u64::MAX;

/// Resource limit values
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Rlimit {
    /// Soft limit (current limit)
    pub rlim_cur: u64,
    /// Hard limit (ceiling)
    pub rlim_max: u64,
}

impl Rlimit {
    /// Create unlimited limit
    pub const fn unlimited() -> Self {
        Self {
            rlim_cur: RLIM_INFINITY,
            rlim_max: RLIM_INFINITY,
        }
    }

    /// Create limit with same soft and hard values
    pub const fn new(limit: u64) -> Self {
        Self {
            rlim_cur: limit,
            rlim_max: limit,
        }
    }

    /// Create limit with soft and hard values
    pub const fn with_hard(soft: u64, hard: u64) -> Self {
        Self {
            rlim_cur: soft,
            rlim_max: hard,
        }
    }
}

impl Default for Rlimit {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Per-process resource limits
#[derive(Clone, Debug)]
pub struct ProcessLimits {
    limits: [Rlimit; Resource::MAX as usize],
}

impl ProcessLimits {
    /// Create with default limits
    pub fn new() -> Self {
        Self {
            limits: [
                // RLIMIT_CPU - unlimited
                Rlimit::unlimited(),
                // RLIMIT_FSIZE - unlimited
                Rlimit::unlimited(),
                // RLIMIT_DATA - unlimited (would be heap limit)
                Rlimit::unlimited(),
                // RLIMIT_STACK - 8MB default
                Rlimit::with_hard(8 * 1024 * 1024, RLIM_INFINITY),
                // RLIMIT_CORE - 0 (no core dumps by default)
                Rlimit::new(0),
                // RLIMIT_RSS - unlimited
                Rlimit::unlimited(),
                // RLIMIT_NPROC - 4096
                Rlimit::with_hard(4096, 4096),
                // RLIMIT_NOFILE - 1024 soft, 1048576 hard
                Rlimit::with_hard(1024, 1048576),
                // RLIMIT_MEMLOCK - 64KB
                Rlimit::with_hard(65536, 65536),
                // RLIMIT_AS - unlimited
                Rlimit::unlimited(),
                // RLIMIT_LOCKS - unlimited
                Rlimit::unlimited(),
                // RLIMIT_SIGPENDING - ~30000
                Rlimit::with_hard(30000, 30000),
                // RLIMIT_MSGQUEUE - 819200
                Rlimit::with_hard(819200, 819200),
                // RLIMIT_NICE - 0
                Rlimit::new(0),
                // RLIMIT_RTPRIO - 0
                Rlimit::new(0),
                // RLIMIT_RTTIME - unlimited
                Rlimit::unlimited(),
            ],
        }
    }

    /// Create init process limits (more permissive)
    pub fn for_init() -> Self {
        let mut limits = Self::new();
        limits.limits[Resource::Nproc as usize] = Rlimit::unlimited();
        limits.limits[Resource::Nofile as usize] = Rlimit::with_hard(1048576, 1048576);
        limits
    }

    /// Get limit for resource
    pub fn get(&self, resource: Resource) -> Rlimit {
        self.limits[resource as usize]
    }

    /// Set limit for resource
    pub fn set(&mut self, resource: Resource, limit: Rlimit) -> Result<(), RlimitError> {
        let current = &self.limits[resource as usize];

        // Cannot raise hard limit (unless privileged)
        if limit.rlim_max > current.rlim_max {
            return Err(RlimitError::PermissionDenied);
        }

        // Soft limit cannot exceed hard limit
        if limit.rlim_cur > limit.rlim_max {
            return Err(RlimitError::Invalid);
        }

        self.limits[resource as usize] = limit;
        Ok(())
    }

    /// Set limit (privileged - can raise hard limit)
    pub fn set_privileged(&mut self, resource: Resource, limit: Rlimit) -> Result<(), RlimitError> {
        if limit.rlim_cur > limit.rlim_max {
            return Err(RlimitError::Invalid);
        }

        self.limits[resource as usize] = limit;
        Ok(())
    }
}

impl Default for ProcessLimits {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-process resource usage tracking
pub struct ResourceUsage {
    /// CPU time used (nanoseconds)
    pub cpu_time: AtomicU64,
    /// Maximum resident set size (bytes)
    pub max_rss: AtomicU64,
    /// Current resident set size (bytes)
    pub current_rss: AtomicU64,
    /// Virtual memory size (bytes)
    pub virtual_size: AtomicU64,
    /// Number of open files
    pub open_files: AtomicU64,
    /// Number of file locks
    pub file_locks: AtomicU64,
    /// Number of pending signals
    pub pending_signals: AtomicU64,
    /// Data segment size
    pub data_size: AtomicU64,
    /// Stack size
    pub stack_size: AtomicU64,
    /// Locked memory
    pub locked_mem: AtomicU64,
}

impl ResourceUsage {
    pub const fn new() -> Self {
        Self {
            cpu_time: AtomicU64::new(0),
            max_rss: AtomicU64::new(0),
            current_rss: AtomicU64::new(0),
            virtual_size: AtomicU64::new(0),
            open_files: AtomicU64::new(0),
            file_locks: AtomicU64::new(0),
            pending_signals: AtomicU64::new(0),
            data_size: AtomicU64::new(0),
            stack_size: AtomicU64::new(0),
            locked_mem: AtomicU64::new(0),
        }
    }

    /// Update max RSS if current is higher
    pub fn update_max_rss(&self, current: u64) {
        self.current_rss.store(current, Ordering::Relaxed);
        let mut max = self.max_rss.load(Ordering::Relaxed);
        while current > max {
            match self.max_rss.compare_exchange_weak(
                max, current, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(new_max) => max = new_max,
            }
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self::new()
    }
}

/// Rlimit error
#[derive(Clone, Copy, Debug)]
pub enum RlimitError {
    /// Invalid resource
    InvalidResource,
    /// Invalid limit value
    Invalid,
    /// Permission denied (cannot raise hard limit)
    PermissionDenied,
    /// Resource limit exceeded
    LimitExceeded,
    /// Process not found
    NoProcess,
}

impl RlimitError {
    pub fn to_errno(&self) -> i32 {
        match self {
            RlimitError::InvalidResource => -22, // EINVAL
            RlimitError::Invalid => -22,         // EINVAL
            RlimitError::PermissionDenied => -1, // EPERM
            RlimitError::LimitExceeded => -24,   // EMFILE/ENFILE depending on context
            RlimitError::NoProcess => -3,        // ESRCH
        }
    }
}

// ============================================================================
// Global State
// ============================================================================

/// Per-process limits storage
static PROCESS_LIMITS: RwLock<BTreeMap<Pid, ProcessLimits>> = RwLock::new(BTreeMap::new());

/// Per-process usage tracking
static PROCESS_USAGE: RwLock<BTreeMap<Pid, ResourceUsage>> = RwLock::new(BTreeMap::new());

/// Get process limits
pub fn get_limits(pid: Pid) -> Option<ProcessLimits> {
    PROCESS_LIMITS.read().get(&pid).cloned()
}

/// Set process limits
pub fn set_limits(pid: Pid, limits: ProcessLimits) {
    PROCESS_LIMITS.write().insert(pid, limits);
}

/// Get limit for specific resource
pub fn getrlimit(pid: Pid, resource: Resource) -> Option<Rlimit> {
    PROCESS_LIMITS.read().get(&pid).map(|l| l.get(resource))
}

/// Set limit for specific resource
pub fn setrlimit(pid: Pid, resource: Resource, limit: Rlimit) -> Result<(), RlimitError> {
    let mut limits_map = PROCESS_LIMITS.write();
    let limits = limits_map.get_mut(&pid).ok_or(RlimitError::NoProcess)?;

    // Check if privileged
    let privileged = crate::capability::has_capability(pid, crate::capability::Capability::SysResource);

    if privileged {
        limits.set_privileged(resource, limit)
    } else {
        limits.set(resource, limit)
    }
}

/// Initialize limits for new process
pub fn init_process(pid: Pid, parent: Option<Pid>) {
    let limits = if let Some(ppid) = parent {
        // Inherit from parent
        PROCESS_LIMITS.read().get(&ppid).cloned().unwrap_or_default()
    } else if pid.0 == 1 {
        ProcessLimits::for_init()
    } else {
        ProcessLimits::new()
    };

    PROCESS_LIMITS.write().insert(pid, limits);
    PROCESS_USAGE.write().insert(pid, ResourceUsage::new());
}

/// Clean up process limits
pub fn cleanup_process(pid: Pid) {
    PROCESS_LIMITS.write().remove(&pid);
    PROCESS_USAGE.write().remove(&pid);
}

// ============================================================================
// Limit Checking
// ============================================================================

/// Check if opening a new file is allowed
pub fn check_nofile(pid: Pid) -> Result<(), RlimitError> {
    let limits = PROCESS_LIMITS.read();
    let usage = PROCESS_USAGE.read();

    let limit = limits.get(&pid).ok_or(RlimitError::NoProcess)?;
    let usage = usage.get(&pid).ok_or(RlimitError::NoProcess)?;

    let max_files = limit.get(Resource::Nofile).rlim_cur;
    let current = usage.open_files.load(Ordering::Relaxed);

    if current >= max_files {
        Err(RlimitError::LimitExceeded)
    } else {
        Ok(())
    }
}

/// Check if creating a new process is allowed
pub fn check_nproc(uid: u32) -> Result<(), RlimitError> {
    // Count processes for this UID
    let count = crate::process::count_by_uid(uid);

    // Get any process with this UID to check limits
    // In practice, would need to track per-user limits
    if count >= 4096 {
        Err(RlimitError::LimitExceeded)
    } else {
        Ok(())
    }
}

/// Check file size limit
pub fn check_fsize(pid: Pid, size: u64) -> Result<(), RlimitError> {
    let limits = PROCESS_LIMITS.read();
    let limit = limits.get(&pid).ok_or(RlimitError::NoProcess)?;

    let max_size = limit.get(Resource::Fsize).rlim_cur;

    if max_size != RLIM_INFINITY && size > max_size {
        Err(RlimitError::LimitExceeded)
    } else {
        Ok(())
    }
}

/// Check memory lock limit
pub fn check_memlock(pid: Pid, additional: u64) -> Result<(), RlimitError> {
    let limits = PROCESS_LIMITS.read();
    let usage = PROCESS_USAGE.read();

    let limit = limits.get(&pid).ok_or(RlimitError::NoProcess)?;
    let usage = usage.get(&pid).ok_or(RlimitError::NoProcess)?;

    let max_lock = limit.get(Resource::Memlock).rlim_cur;
    let current = usage.locked_mem.load(Ordering::Relaxed);

    if max_lock != RLIM_INFINITY && current + additional > max_lock {
        Err(RlimitError::LimitExceeded)
    } else {
        Ok(())
    }
}

/// Check address space limit
pub fn check_as(pid: Pid, additional: u64) -> Result<(), RlimitError> {
    let limits = PROCESS_LIMITS.read();
    let usage = PROCESS_USAGE.read();

    let limit = limits.get(&pid).ok_or(RlimitError::NoProcess)?;
    let usage = usage.get(&pid).ok_or(RlimitError::NoProcess)?;

    let max_as = limit.get(Resource::As).rlim_cur;
    let current = usage.virtual_size.load(Ordering::Relaxed);

    if max_as != RLIM_INFINITY && current + additional > max_as {
        Err(RlimitError::LimitExceeded)
    } else {
        Ok(())
    }
}

/// Check core dump size limit
pub fn get_core_limit(pid: Pid) -> u64 {
    PROCESS_LIMITS.read()
        .get(&pid)
        .map(|l| l.get(Resource::Core).rlim_cur)
        .unwrap_or(0)
}

/// Check stack size limit
pub fn get_stack_limit(pid: Pid) -> u64 {
    PROCESS_LIMITS.read()
        .get(&pid)
        .map(|l| l.get(Resource::Stack).rlim_cur)
        .unwrap_or(8 * 1024 * 1024)
}

// ============================================================================
// Usage Tracking
// ============================================================================

/// Record file open
pub fn record_file_open(pid: Pid) {
    if let Some(usage) = PROCESS_USAGE.read().get(&pid) {
        usage.open_files.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record file close
pub fn record_file_close(pid: Pid) {
    if let Some(usage) = PROCESS_USAGE.read().get(&pid) {
        usage.open_files.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Record CPU time
pub fn record_cpu_time(pid: Pid, ns: u64) {
    if let Some(usage) = PROCESS_USAGE.read().get(&pid) {
        let new_time = usage.cpu_time.fetch_add(ns, Ordering::Relaxed) + ns;

        // Check CPU limit
        if let Some(limit) = PROCESS_LIMITS.read().get(&pid) {
            let max_cpu = limit.get(Resource::Cpu).rlim_cur;
            if max_cpu != RLIM_INFINITY {
                let max_ns = max_cpu * 1_000_000_000;
                if new_time > max_ns {
                    // Send SIGXCPU
                    crate::signal::send_signal(pid, crate::signal::Signal::SIGXCPU);
                }
            }
        }
    }
}

/// Update RSS tracking
pub fn update_rss(pid: Pid, rss: u64) {
    if let Some(usage) = PROCESS_USAGE.read().get(&pid) {
        usage.update_max_rss(rss);
    }
}

// ============================================================================
// Syscall Interface
// ============================================================================

/// getrlimit syscall
pub fn sys_getrlimit(resource: u32, rlim_ptr: usize) -> isize {
    let resource = match Resource::from_u32(resource) {
        Some(r) => r,
        None => return -22, // EINVAL
    };

    let pid = match crate::process::current() {
        Some(p) => p.pid,
        None => return -3, // ESRCH
    };

    let rlimit = match getrlimit(pid, resource) {
        Some(r) => r,
        None => return -3,
    };

    unsafe {
        *(rlim_ptr as *mut Rlimit) = rlimit;
    }

    0
}

/// setrlimit syscall
pub fn sys_setrlimit(resource: u32, rlim_ptr: usize) -> isize {
    let resource = match Resource::from_u32(resource) {
        Some(r) => r,
        None => return -22, // EINVAL
    };

    let pid = match crate::process::current() {
        Some(p) => p.pid,
        None => return -3,
    };

    let rlimit = unsafe { *(rlim_ptr as *const Rlimit) };

    match setrlimit(pid, resource, rlimit) {
        Ok(()) => 0,
        Err(e) => e.to_errno() as isize,
    }
}

/// prlimit64 syscall (get and set)
pub fn sys_prlimit64(pid: i32, resource: u32, new_rlim: usize, old_rlim: usize) -> isize {
    let resource = match Resource::from_u32(resource) {
        Some(r) => r,
        None => return -22,
    };

    let target_pid = if pid == 0 {
        match crate::process::current() {
            Some(p) => p.pid,
            None => return -3,
        }
    } else {
        Pid(pid as u32)
    };

    // Get old value if requested
    if old_rlim != 0 {
        let rlimit = match getrlimit(target_pid, resource) {
            Some(r) => r,
            None => return -3,
        };
        unsafe {
            *(old_rlim as *mut Rlimit) = rlimit;
        }
    }

    // Set new value if provided
    if new_rlim != 0 {
        let rlimit = unsafe { *(new_rlim as *const Rlimit) };
        match setrlimit(target_pid, resource, rlimit) {
            Ok(()) => {}
            Err(e) => return e.to_errno() as isize,
        }
    }

    0
}

/// getrusage syscall
pub fn sys_getrusage(who: i32, usage_ptr: usize) -> isize {
    let pid = match crate::process::current() {
        Some(p) => p.pid,
        None => return -3,
    };

    let target_pid = match who {
        0 => pid,     // RUSAGE_SELF
        -1 => pid,    // RUSAGE_CHILDREN - would aggregate children
        -2 => pid,    // RUSAGE_THREAD
        _ => return -22,
    };

    let usage = match PROCESS_USAGE.read().get(&target_pid) {
        Some(u) => u,
        None => return -3,
    };

    // Fill in rusage structure
    #[repr(C)]
    struct Rusage {
        ru_utime: [i64; 2],  // user time
        ru_stime: [i64; 2],  // system time
        ru_maxrss: i64,
        ru_ixrss: i64,
        ru_idrss: i64,
        ru_isrss: i64,
        ru_minflt: i64,
        ru_majflt: i64,
        ru_nswap: i64,
        ru_inblock: i64,
        ru_oublock: i64,
        ru_msgsnd: i64,
        ru_msgrcv: i64,
        ru_nsignals: i64,
        ru_nvcsw: i64,
        ru_nivcsw: i64,
    }

    let cpu_ns = usage.cpu_time.load(Ordering::Relaxed);
    let cpu_sec = cpu_ns / 1_000_000_000;
    let cpu_usec = (cpu_ns % 1_000_000_000) / 1000;

    let rusage = Rusage {
        ru_utime: [cpu_sec as i64, cpu_usec as i64],
        ru_stime: [0, 0],
        ru_maxrss: (usage.max_rss.load(Ordering::Relaxed) / 1024) as i64,
        ru_ixrss: 0,
        ru_idrss: 0,
        ru_isrss: 0,
        ru_minflt: 0,
        ru_majflt: 0,
        ru_nswap: 0,
        ru_inblock: 0,
        ru_oublock: 0,
        ru_msgsnd: 0,
        ru_msgrcv: 0,
        ru_nsignals: 0,
        ru_nvcsw: 0,
        ru_nivcsw: 0,
    };

    unsafe {
        *(usage_ptr as *mut Rusage) = rusage;
    }

    0
}

/// Initialize rlimit subsystem
pub fn init() {
    crate::kprintln!("  Resource limits initialized");
}
