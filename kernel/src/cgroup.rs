//! Control Groups (cgroups v2)
//!
//! Resource limiting and accounting for process groups.
//! Implements unified hierarchy with cpu, memory, io, and pids controllers.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::{Mutex, RwLock};

use crate::process::Pid;

/// Cgroup ID
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CgroupId(pub u64);

static NEXT_CGROUP_ID: AtomicU64 = AtomicU64::new(1);

impl CgroupId {
    pub fn new() -> Self {
        Self(NEXT_CGROUP_ID.fetch_add(1, Ordering::SeqCst))
    }
}

/// Cgroup controller types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Controller {
    /// CPU time accounting and limiting
    Cpu,
    /// CPU set (affinity)
    Cpuset,
    /// Memory usage limiting
    Memory,
    /// Block I/O limiting
    Io,
    /// Process count limiting
    Pids,
    /// HugeTLB limiting
    HugeTlb,
    /// RDMA limiting
    Rdma,
    /// Misc controller
    Misc,
}

impl Controller {
    pub fn name(&self) -> &'static str {
        match self {
            Controller::Cpu => "cpu",
            Controller::Cpuset => "cpuset",
            Controller::Memory => "memory",
            Controller::Io => "io",
            Controller::Pids => "pids",
            Controller::HugeTlb => "hugetlb",
            Controller::Rdma => "rdma",
            Controller::Misc => "misc",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cpu" => Some(Controller::Cpu),
            "cpuset" => Some(Controller::Cpuset),
            "memory" => Some(Controller::Memory),
            "io" => Some(Controller::Io),
            "pids" => Some(Controller::Pids),
            "hugetlb" => Some(Controller::HugeTlb),
            "rdma" => Some(Controller::Rdma),
            "misc" => Some(Controller::Misc),
            _ => None,
        }
    }
}

// ============================================================================
// CPU Controller
// ============================================================================

/// CPU controller state
#[derive(Clone, Debug)]
pub struct CpuController {
    /// CPU weight (1-10000, default 100)
    pub weight: u32,
    /// CPU weight for nice 0 (used for nice calculations)
    pub weight_nice: i32,
    /// Maximum CPU bandwidth (quota per period)
    /// max = quota period (microseconds)
    pub max: Option<(u64, u64)>,
    /// Burst (extra bandwidth that can be used)
    pub burst: u64,
    /// Pressure stall information
    pub pressure: CpuPressure,
    /// Statistics
    pub stat: CpuStat,
}

#[derive(Clone, Debug, Default)]
pub struct CpuPressure {
    /// Some tasks stalled (percent)
    pub some_avg10: f32,
    pub some_avg60: f32,
    pub some_avg300: f32,
    pub some_total: u64,
    /// All tasks stalled
    pub full_avg10: f32,
    pub full_avg60: f32,
    pub full_avg300: f32,
    pub full_total: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CpuStat {
    /// Total CPU time used (microseconds)
    pub usage_usec: u64,
    /// User CPU time
    pub user_usec: u64,
    /// System CPU time
    pub system_usec: u64,
    /// Number of periods
    pub nr_periods: u64,
    /// Number of throttled periods
    pub nr_throttled: u64,
    /// Total throttled time
    pub throttled_usec: u64,
    /// Number of burst periods
    pub nr_bursts: u64,
    /// Total burst time
    pub burst_usec: u64,
}

impl Default for CpuController {
    fn default() -> Self {
        Self {
            weight: 100,
            weight_nice: 0,
            max: None, // No limit
            burst: 0,
            pressure: CpuPressure::default(),
            stat: CpuStat::default(),
        }
    }
}

impl CpuController {
    /// Check if process can use more CPU
    pub fn can_run(&self, used_usec: u64, period_usec: u64) -> bool {
        if let Some((quota, period)) = self.max {
            let allowed = (quota * period_usec) / period;
            used_usec < allowed + self.burst
        } else {
            true // No limit
        }
    }

    /// Record CPU usage
    pub fn charge(&mut self, usec: u64, is_user: bool) {
        self.stat.usage_usec += usec;
        if is_user {
            self.stat.user_usec += usec;
        } else {
            self.stat.system_usec += usec;
        }
    }
}

// ============================================================================
// Memory Controller
// ============================================================================

/// Memory controller state
#[derive(Clone, Debug)]
pub struct MemoryController {
    /// Current memory usage (bytes)
    pub current: u64,
    /// Memory usage limit (bytes), 0 = no limit
    pub max: u64,
    /// High watermark (starts reclaim)
    pub high: u64,
    /// Low watermark (protected from reclaim)
    pub low: u64,
    /// Minimum memory guarantee
    pub min: u64,
    /// OOM group - kill all tasks on OOM
    pub oom_group: bool,
    /// Swap usage
    pub swap_current: u64,
    /// Swap limit
    pub swap_max: u64,
    /// Statistics
    pub stat: MemoryStat,
    /// Events
    pub events: MemoryEvents,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryStat {
    /// Anonymous memory
    pub anon: u64,
    /// File cache
    pub file: u64,
    /// Kernel memory
    pub kernel: u64,
    /// Slab memory
    pub slab: u64,
    /// Socket buffers
    pub sock: u64,
    /// Shmem
    pub shmem: u64,
    /// Page tables
    pub pagetables: u64,
    /// Swap cached
    pub swapcached: u64,
    /// Pages paged in
    pub pgfault: u64,
    /// Major page faults
    pub pgmajfault: u64,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryEvents {
    /// Number of times low exceeded
    pub low: u64,
    /// Number of times high exceeded
    pub high: u64,
    /// Number of times max exceeded
    pub max: u64,
    /// Number of OOM events
    pub oom: u64,
    /// Number of OOM kills
    pub oom_kill: u64,
    /// Number of OOM kills grouped
    pub oom_group_kill: u64,
}

impl Default for MemoryController {
    fn default() -> Self {
        Self {
            current: 0,
            max: 0, // No limit
            high: u64::MAX,
            low: 0,
            min: 0,
            oom_group: false,
            swap_current: 0,
            swap_max: u64::MAX,
            stat: MemoryStat::default(),
            events: MemoryEvents::default(),
        }
    }
}

impl MemoryController {
    /// Try to charge memory
    pub fn try_charge(&mut self, bytes: u64) -> Result<(), CgroupError> {
        let new_usage = self.current.saturating_add(bytes);

        // Check hard limit
        if self.max > 0 && new_usage > self.max {
            self.events.max += 1;
            return Err(CgroupError::MemoryLimit);
        }

        // Check high watermark
        if new_usage > self.high {
            self.events.high += 1;
            // Trigger reclaim but allow
        }

        self.current = new_usage;
        Ok(())
    }

    /// Uncharge memory
    pub fn uncharge(&mut self, bytes: u64) {
        self.current = self.current.saturating_sub(bytes);
    }

    /// Check if under memory pressure
    pub fn under_pressure(&self) -> bool {
        self.current > self.high || (self.max > 0 && self.current > self.max * 90 / 100)
    }

    /// Get OOM score for this cgroup
    pub fn oom_score(&self) -> u64 {
        if self.max == 0 {
            return 0;
        }
        (self.current * 1000) / self.max
    }
}

// ============================================================================
// I/O Controller
// ============================================================================

/// I/O controller state
#[derive(Clone, Debug)]
pub struct IoController {
    /// Weight for proportional I/O (1-10000)
    pub weight: u32,
    /// Per-device limits
    pub limits: BTreeMap<(u32, u32), IoLimit>, // (major, minor) -> limit
    /// Statistics per device
    pub stat: BTreeMap<(u32, u32), IoStat>,
}

#[derive(Clone, Debug, Default)]
pub struct IoLimit {
    /// Read bytes per second
    pub rbps: Option<u64>,
    /// Write bytes per second
    pub wbps: Option<u64>,
    /// Read IOPS
    pub riops: Option<u64>,
    /// Write IOPS
    pub wiops: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct IoStat {
    /// Bytes read
    pub rbytes: u64,
    /// Bytes written
    pub wbytes: u64,
    /// Read operations
    pub rios: u64,
    /// Write operations
    pub wios: u64,
    /// Discard bytes
    pub dbytes: u64,
    /// Discard operations
    pub dios: u64,
}

impl Default for IoController {
    fn default() -> Self {
        Self {
            weight: 100,
            limits: BTreeMap::new(),
            stat: BTreeMap::new(),
        }
    }
}

impl IoController {
    /// Check if I/O is allowed
    pub fn can_io(&self, dev: (u32, u32), bytes: u64, is_write: bool) -> bool {
        if let Some(limit) = self.limits.get(&dev) {
            let limit_bps = if is_write { limit.wbps } else { limit.rbps };
            // Simplified check - real implementation would track rate over time
            if let Some(bps) = limit_bps {
                return bytes < bps;
            }
        }
        true
    }

    /// Record I/O
    pub fn charge_io(&mut self, dev: (u32, u32), bytes: u64, is_write: bool) {
        let stat = self.stat.entry(dev).or_default();
        if is_write {
            stat.wbytes += bytes;
            stat.wios += 1;
        } else {
            stat.rbytes += bytes;
            stat.rios += 1;
        }
    }
}

// ============================================================================
// PIDs Controller
// ============================================================================

/// PIDs controller state
#[derive(Clone, Debug)]
pub struct PidsController {
    /// Current number of processes
    pub current: u32,
    /// Maximum processes allowed
    pub max: u32, // 0 = no limit
    /// Events
    pub events: PidsEvents,
}

#[derive(Clone, Debug, Default)]
pub struct PidsEvents {
    /// Number of times fork was rejected
    pub max: u64,
}

impl Default for PidsController {
    fn default() -> Self {
        Self {
            current: 0,
            max: 0,
            events: PidsEvents::default(),
        }
    }
}

impl PidsController {
    /// Try to fork
    pub fn try_fork(&mut self) -> Result<(), CgroupError> {
        if self.max > 0 && self.current >= self.max {
            self.events.max += 1;
            return Err(CgroupError::PidsLimit);
        }
        self.current += 1;
        Ok(())
    }

    /// Process exited
    pub fn exit(&mut self) {
        self.current = self.current.saturating_sub(1);
    }
}

// ============================================================================
// Cgroup
// ============================================================================

/// A control group
pub struct Cgroup {
    /// Cgroup ID
    pub id: CgroupId,
    /// Path in hierarchy
    pub path: String,
    /// Parent cgroup
    pub parent: Option<Arc<Cgroup>>,
    /// Children
    children: RwLock<BTreeMap<String, Arc<Cgroup>>>,
    /// Member processes
    procs: RwLock<Vec<Pid>>,
    /// Enabled controllers
    controllers: RwLock<Vec<Controller>>,
    /// CPU controller
    pub cpu: Mutex<CpuController>,
    /// Memory controller
    pub memory: Mutex<MemoryController>,
    /// I/O controller
    pub io: Mutex<IoController>,
    /// PIDs controller
    pub pids: Mutex<PidsController>,
    /// Frozen state
    pub frozen: AtomicU32, // 0 = running, 1 = frozen
}

impl Cgroup {
    /// Create root cgroup
    pub fn root() -> Arc<Self> {
        Arc::new(Self {
            id: CgroupId::new(),
            path: String::from("/"),
            parent: None,
            children: RwLock::new(BTreeMap::new()),
            procs: RwLock::new(Vec::new()),
            controllers: RwLock::new(vec![
                Controller::Cpu,
                Controller::Memory,
                Controller::Io,
                Controller::Pids,
            ]),
            cpu: Mutex::new(CpuController::default()),
            memory: Mutex::new(MemoryController::default()),
            io: Mutex::new(IoController::default()),
            pids: Mutex::new(PidsController::default()),
            frozen: AtomicU32::new(0),
        })
    }

    /// Create child cgroup
    pub fn create_child(parent: Arc<Self>, name: &str) -> Result<Arc<Self>, CgroupError> {
        let path = if parent.path == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", parent.path, name)
        };

        // Check if already exists
        if parent.children.read().contains_key(name) {
            return Err(CgroupError::AlreadyExists);
        }

        let child = Arc::new(Self {
            id: CgroupId::new(),
            path,
            parent: Some(parent.clone()),
            children: RwLock::new(BTreeMap::new()),
            procs: RwLock::new(Vec::new()),
            controllers: RwLock::new(parent.controllers.read().clone()),
            cpu: Mutex::new(CpuController::default()),
            memory: Mutex::new(MemoryController::default()),
            io: Mutex::new(IoController::default()),
            pids: Mutex::new(PidsController::default()),
            frozen: AtomicU32::new(0),
        });

        parent
            .children
            .write()
            .insert(String::from(name), child.clone());
        Ok(child)
    }

    /// Delete cgroup (must be empty)
    pub fn delete(self: &Arc<Self>) -> Result<(), CgroupError> {
        // Check if empty
        if !self.procs.read().is_empty() {
            return Err(CgroupError::NotEmpty);
        }
        if !self.children.read().is_empty() {
            return Err(CgroupError::NotEmpty);
        }

        // Remove from parent
        if let Some(ref parent) = self.parent {
            let name = self.path.rsplit('/').next().unwrap_or("");
            parent.children.write().remove(name);
        }

        Ok(())
    }

    /// Add process to cgroup
    pub fn add_proc(&self, pid: Pid) -> Result<(), CgroupError> {
        // Check pids limit
        self.pids.lock().try_fork()?;

        self.procs.write().push(pid);
        Ok(())
    }

    /// Remove process from cgroup
    pub fn remove_proc(&self, pid: Pid) {
        self.procs.write().retain(|p| *p != pid);
        self.pids.lock().exit();
    }

    /// Get all processes (including children)
    pub fn get_all_procs(&self) -> Vec<Pid> {
        let mut procs = self.procs.read().clone();
        for child in self.children.read().values() {
            procs.extend(child.get_all_procs());
        }
        procs
    }

    /// Freeze cgroup
    pub fn freeze(&self) {
        self.frozen.store(1, Ordering::SeqCst);
        // TODO: Actually stop all processes
    }

    /// Thaw cgroup
    pub fn thaw(&self) {
        self.frozen.store(0, Ordering::SeqCst);
        // TODO: Resume all processes
    }

    /// Check if frozen
    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::SeqCst) != 0
    }

    /// Enable controller
    pub fn enable_controller(&self, controller: Controller) {
        let mut controllers = self.controllers.write();
        if !controllers.contains(&controller) {
            controllers.push(controller);
        }
    }

    /// Disable controller
    pub fn disable_controller(&self, controller: Controller) {
        self.controllers.write().retain(|c| *c != controller);
    }

    /// Check if controller is enabled
    pub fn has_controller(&self, controller: Controller) -> bool {
        self.controllers.read().contains(&controller)
    }

    /// Get child by name
    pub fn get_child(&self, name: &str) -> Option<Arc<Cgroup>> {
        self.children.read().get(name).cloned()
    }

    /// Try to charge memory
    pub fn try_charge_memory(&self, bytes: u64) -> Result<(), CgroupError> {
        // Check this cgroup and all parents
        self.memory.lock().try_charge(bytes)?;

        if let Some(ref parent) = self.parent {
            parent.try_charge_memory(bytes)?;
        }

        Ok(())
    }

    /// Uncharge memory
    pub fn uncharge_memory(&self, bytes: u64) {
        self.memory.lock().uncharge(bytes);

        if let Some(ref parent) = self.parent {
            parent.uncharge_memory(bytes);
        }
    }
}

// ============================================================================
// Cgroup Error
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub enum CgroupError {
    /// Cgroup not found
    NotFound,
    /// Cgroup already exists
    AlreadyExists,
    /// Cgroup not empty
    NotEmpty,
    /// Memory limit exceeded
    MemoryLimit,
    /// PIDs limit exceeded
    PidsLimit,
    /// I/O limit exceeded
    IoLimit,
    /// CPU limit exceeded
    CpuLimit,
    /// Invalid path
    InvalidPath,
    /// Permission denied
    PermissionDenied,
}

// ============================================================================
// Global State
// ============================================================================

/// Root cgroup
static ROOT_CGROUP: RwLock<Option<Arc<Cgroup>>> = RwLock::new(None);

/// Process to cgroup mapping
static PROCESS_CGROUPS: RwLock<BTreeMap<Pid, Arc<Cgroup>>> = RwLock::new(BTreeMap::new());

/// Initialize cgroup subsystem
pub fn init() {
    let root = Cgroup::root();
    *ROOT_CGROUP.write() = Some(root.clone());

    // Create default cgroups
    let _ = Cgroup::create_child(root.clone(), "system.slice");
    let _ = Cgroup::create_child(root.clone(), "user.slice");
    let _ = Cgroup::create_child(root, "init.scope");

    crate::kprintln!("  Cgroups v2 initialized (cpu, memory, io, pids controllers)");
}

/// Get root cgroup
pub fn root() -> Arc<Cgroup> {
    ROOT_CGROUP.read().as_ref().unwrap().clone()
}

/// Lookup cgroup by path
pub fn lookup(path: &str) -> Option<Arc<Cgroup>> {
    let root = root();
    if path == "/" {
        return Some(root);
    }

    let mut current = root;
    for component in path.trim_start_matches('/').split('/') {
        if component.is_empty() {
            continue;
        }
        current = current.get_child(component)?;
    }

    Some(current)
}

/// Create cgroup
pub fn create(path: &str) -> Result<Arc<Cgroup>, CgroupError> {
    let parent_path = path.rsplit_once('/').map(|(p, _)| p).unwrap_or("/");
    let name = path.rsplit('/').next().ok_or(CgroupError::InvalidPath)?;

    let parent = if parent_path.is_empty() {
        root()
    } else {
        lookup(parent_path).ok_or(CgroupError::NotFound)?
    };

    Cgroup::create_child(parent, name)
}

/// Delete cgroup
pub fn delete(path: &str) -> Result<(), CgroupError> {
    let cgroup = lookup(path).ok_or(CgroupError::NotFound)?;
    cgroup.delete()
}

/// Get process cgroup
pub fn get_process_cgroup(pid: Pid) -> Option<Arc<Cgroup>> {
    PROCESS_CGROUPS.read().get(&pid).cloned()
}

/// Set process cgroup
pub fn set_process_cgroup(pid: Pid, cgroup: Arc<Cgroup>) -> Result<(), CgroupError> {
    // Remove from old cgroup
    if let Some(old_cg) = PROCESS_CGROUPS.write().remove(&pid) {
        old_cg.remove_proc(pid);
    }

    // Add to new cgroup
    cgroup.add_proc(pid)?;
    PROCESS_CGROUPS.write().insert(pid, cgroup);

    Ok(())
}

/// Move process to cgroup by path
pub fn move_to_cgroup(pid: Pid, path: &str) -> Result<(), CgroupError> {
    let cgroup = lookup(path).ok_or(CgroupError::NotFound)?;
    set_process_cgroup(pid, cgroup)
}

/// Remove process from cgroups
pub fn cleanup_process(pid: Pid) {
    if let Some(cgroup) = PROCESS_CGROUPS.write().remove(&pid) {
        cgroup.remove_proc(pid);
    }
}

/// Fork: child inherits parent cgroup
pub fn fork_cgroup(parent: Pid, child: Pid) {
    if let Some(cgroup) = get_process_cgroup(parent) {
        let _ = set_process_cgroup(child, cgroup);
    } else {
        // Default to root cgroup
        let _ = set_process_cgroup(child, root());
    }
}

/// Check if process can allocate memory
pub fn can_alloc_memory(pid: Pid, bytes: u64) -> bool {
    if let Some(cgroup) = get_process_cgroup(pid) {
        cgroup.try_charge_memory(bytes).is_ok()
    } else {
        true
    }
}

/// Charge memory to process cgroup
pub fn charge_memory(pid: Pid, bytes: u64) -> Result<(), CgroupError> {
    if let Some(cgroup) = get_process_cgroup(pid) {
        cgroup.try_charge_memory(bytes)
    } else {
        Ok(())
    }
}

/// Uncharge memory from process cgroup
pub fn uncharge_memory(pid: Pid, bytes: u64) {
    if let Some(cgroup) = get_process_cgroup(pid) {
        cgroup.uncharge_memory(bytes);
    }
}

/// Check if process can fork
pub fn can_fork(pid: Pid) -> bool {
    if let Some(cgroup) = get_process_cgroup(pid) {
        cgroup.pids.lock().current < cgroup.pids.lock().max || cgroup.pids.lock().max == 0
    } else {
        true
    }
}

/// Generate cgroup.controllers content
pub fn generate_controllers(cgroup: &Cgroup) -> String {
    cgroup
        .controllers
        .read()
        .iter()
        .map(|c| c.name())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Generate cgroup.procs content
pub fn generate_procs(cgroup: &Cgroup) -> String {
    cgroup
        .procs
        .read()
        .iter()
        .map(|p| format!("{}", p.0))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate memory.current content
pub fn generate_memory_current(cgroup: &Cgroup) -> String {
    format!("{}", cgroup.memory.lock().current)
}

/// Generate memory.max content
pub fn generate_memory_max(cgroup: &Cgroup) -> String {
    let max = cgroup.memory.lock().max;
    if max == 0 {
        String::from("max")
    } else {
        format!("{}", max)
    }
}

/// Parse and set memory.max
pub fn set_memory_max(cgroup: &Cgroup, value: &str) -> Result<(), CgroupError> {
    let max = if value == "max" {
        0
    } else {
        value.parse().map_err(|_| CgroupError::InvalidPath)?
    };
    cgroup.memory.lock().max = max;
    Ok(())
}

/// Generate cpu.max content
pub fn generate_cpu_max(cgroup: &Cgroup) -> String {
    let cpu = cgroup.cpu.lock();
    if let Some((quota, period)) = cpu.max {
        format!("{} {}", quota, period)
    } else {
        String::from("max 100000")
    }
}

/// Parse and set cpu.max
pub fn set_cpu_max(cgroup: &Cgroup, value: &str) -> Result<(), CgroupError> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() != 2 {
        return Err(CgroupError::InvalidPath);
    }

    let quota = if parts[0] == "max" {
        return Ok(()); // No limit
    } else {
        parts[0].parse().map_err(|_| CgroupError::InvalidPath)?
    };
    let period: u64 = parts[1].parse().map_err(|_| CgroupError::InvalidPath)?;

    cgroup.cpu.lock().max = Some((quota, period));
    Ok(())
}

/// Generate pids.max content
pub fn generate_pids_max(cgroup: &Cgroup) -> String {
    let max = cgroup.pids.lock().max;
    if max == 0 {
        String::from("max")
    } else {
        format!("{}", max)
    }
}

/// Parse and set pids.max
pub fn set_pids_max(cgroup: &Cgroup, value: &str) -> Result<(), CgroupError> {
    let max = if value == "max" {
        0
    } else {
        value.parse().map_err(|_| CgroupError::InvalidPath)?
    };
    cgroup.pids.lock().max = max;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cgroup_hierarchy() {
        let root = Cgroup::root();
        let child = Cgroup::create_child(root.clone(), "test").unwrap();

        assert_eq!(child.path, "/test");
        assert!(root.get_child("test").is_some());
    }

    #[test]
    fn test_memory_limit() {
        let cgroup = Cgroup::root();
        cgroup.memory.lock().max = 1024;

        assert!(cgroup.try_charge_memory(512).is_ok());
        assert!(cgroup.try_charge_memory(1024).is_err());
    }

    #[test]
    fn test_pids_limit() {
        let cgroup = Cgroup::root();
        cgroup.pids.lock().max = 2;

        assert!(cgroup.pids.lock().try_fork().is_ok());
        assert!(cgroup.pids.lock().try_fork().is_ok());
        assert!(cgroup.pids.lock().try_fork().is_err());
    }
}
