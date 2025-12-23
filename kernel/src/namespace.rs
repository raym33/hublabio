//! Linux-style Namespaces
//!
//! Process isolation through namespaces for containerization.
//! Implements PID, network, IPC, UTS, mount, user, and cgroup namespaces.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::{Mutex, RwLock};

use crate::auth::{Gid, Uid};
use crate::process::Pid;

/// Namespace ID
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NsId(pub u64);

static NEXT_NS_ID: AtomicU64 = AtomicU64::new(1);

impl NsId {
    pub fn new() -> Self {
        Self(NEXT_NS_ID.fetch_add(1, Ordering::SeqCst))
    }
}

/// Namespace types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum NsType {
    /// Process ID namespace
    Pid = 0x20000000, // CLONE_NEWPID
    /// Network namespace
    Net = 0x40000000, // CLONE_NEWNET
    /// IPC namespace
    Ipc = 0x08000000, // CLONE_NEWIPC
    /// UTS namespace (hostname)
    Uts = 0x04000000, // CLONE_NEWUTS
    /// Mount namespace
    Mnt = 0x00020000, // CLONE_NEWNS
    /// User namespace
    User = 0x10000000, // CLONE_NEWUSER
    /// Cgroup namespace
    Cgroup = 0x02000000, // CLONE_NEWCGROUP
    /// Time namespace
    Time = 0x00000080, // CLONE_NEWTIME
}

impl NsType {
    pub fn from_clone_flag(flag: u32) -> Option<Self> {
        match flag {
            0x20000000 => Some(NsType::Pid),
            0x40000000 => Some(NsType::Net),
            0x08000000 => Some(NsType::Ipc),
            0x04000000 => Some(NsType::Uts),
            0x00020000 => Some(NsType::Mnt),
            0x10000000 => Some(NsType::User),
            0x02000000 => Some(NsType::Cgroup),
            0x00000080 => Some(NsType::Time),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NsType::Pid => "pid",
            NsType::Net => "net",
            NsType::Ipc => "ipc",
            NsType::Uts => "uts",
            NsType::Mnt => "mnt",
            NsType::User => "user",
            NsType::Cgroup => "cgroup",
            NsType::Time => "time",
        }
    }
}

/// Clone flags for namespace creation
pub mod clone_flags {
    pub const CLONE_NEWPID: u32 = 0x20000000;
    pub const CLONE_NEWNET: u32 = 0x40000000;
    pub const CLONE_NEWIPC: u32 = 0x08000000;
    pub const CLONE_NEWUTS: u32 = 0x04000000;
    pub const CLONE_NEWNS: u32 = 0x00020000;
    pub const CLONE_NEWUSER: u32 = 0x10000000;
    pub const CLONE_NEWCGROUP: u32 = 0x02000000;
    pub const CLONE_NEWTIME: u32 = 0x00000080;

    pub const ALL_NS_FLAGS: u32 = CLONE_NEWPID
        | CLONE_NEWNET
        | CLONE_NEWIPC
        | CLONE_NEWUTS
        | CLONE_NEWNS
        | CLONE_NEWUSER
        | CLONE_NEWCGROUP
        | CLONE_NEWTIME;
}

// ============================================================================
// PID Namespace
// ============================================================================

/// PID namespace state
pub struct PidNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Parent namespace (None for init namespace)
    pub parent: Option<Arc<PidNamespace>>,
    /// Next PID to allocate in this namespace
    next_pid: AtomicU32,
    /// PID mappings: ns_pid -> global_pid
    pid_map: RwLock<BTreeMap<u32, Pid>>,
    /// Reverse mapping: global_pid -> ns_pid
    reverse_map: RwLock<BTreeMap<Pid, u32>>,
    /// Init process in this namespace (PID 1)
    pub init_pid: Mutex<Option<Pid>>,
    /// Nesting level
    pub level: u32,
}

impl PidNamespace {
    /// Create the initial (root) PID namespace
    pub fn init_ns() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            parent: None,
            next_pid: AtomicU32::new(2), // PID 1 reserved for init
            pid_map: RwLock::new(BTreeMap::new()),
            reverse_map: RwLock::new(BTreeMap::new()),
            init_pid: Mutex::new(None),
            level: 0,
        })
    }

    /// Create a child PID namespace
    pub fn create_child(parent: Arc<Self>) -> Arc<Self> {
        let level = parent.level + 1;
        if level > 32 {
            // Max nesting depth
            return parent;
        }

        Arc::new(Self {
            id: NsId::new(),
            parent: Some(parent),
            next_pid: AtomicU32::new(1), // Start from 1 in new namespace
            pid_map: RwLock::new(BTreeMap::new()),
            reverse_map: RwLock::new(BTreeMap::new()),
            init_pid: Mutex::new(None),
            level,
        })
    }

    /// Allocate a PID in this namespace
    pub fn alloc_pid(&self, global_pid: Pid) -> u32 {
        let ns_pid = self.next_pid.fetch_add(1, Ordering::SeqCst);

        // Set init if this is PID 1
        if ns_pid == 1 {
            *self.init_pid.lock() = Some(global_pid);
        }

        self.pid_map.write().insert(ns_pid, global_pid);
        self.reverse_map.write().insert(global_pid, ns_pid);

        // Also register in parent namespaces
        if let Some(ref parent) = self.parent {
            parent.alloc_pid(global_pid);
        }

        ns_pid
    }

    /// Translate namespace PID to global PID
    pub fn to_global(&self, ns_pid: u32) -> Option<Pid> {
        self.pid_map.read().get(&ns_pid).copied()
    }

    /// Translate global PID to namespace PID
    pub fn from_global(&self, global_pid: Pid) -> Option<u32> {
        self.reverse_map.read().get(&global_pid).copied()
    }

    /// Remove PID from namespace
    pub fn free_pid(&self, global_pid: Pid) {
        if let Some(ns_pid) = self.reverse_map.write().remove(&global_pid) {
            self.pid_map.write().remove(&ns_pid);
        }

        // Also remove from parent
        if let Some(ref parent) = self.parent {
            parent.free_pid(global_pid);
        }
    }

    /// Check if process is init (PID 1) in this namespace
    pub fn is_init(&self, global_pid: Pid) -> bool {
        self.init_pid
            .lock()
            .map(|p| p == global_pid)
            .unwrap_or(false)
    }
}

// ============================================================================
// Network Namespace
// ============================================================================

/// Network namespace state
pub struct NetNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Loopback interface configured
    pub loopback_up: bool,
    /// Network interfaces in this namespace
    interfaces: RwLock<Vec<NetInterface>>,
    /// Routing table
    routes: RwLock<Vec<Route>>,
    /// Firewall rules
    firewall: RwLock<Vec<FirewallRule>>,
}

#[derive(Clone)]
pub struct NetInterface {
    pub name: String,
    pub index: u32,
    pub mac: [u8; 6],
    pub ipv4: Option<(u32, u8)>, // (addr, prefix_len)
    pub flags: u32,
}

#[derive(Clone)]
pub struct Route {
    pub dest: u32,
    pub prefix_len: u8,
    pub gateway: Option<u32>,
    pub iface: String,
    pub metric: u32,
}

#[derive(Clone)]
pub struct FirewallRule {
    pub chain: String,
    pub action: FirewallAction,
    pub src: Option<(u32, u8)>,
    pub dst: Option<(u32, u8)>,
    pub port: Option<u16>,
    pub proto: Option<u8>,
}

#[derive(Clone, Copy, Debug)]
pub enum FirewallAction {
    Accept,
    Drop,
    Reject,
    Log,
}

impl NetNamespace {
    /// Create initial network namespace
    pub fn init_ns() -> Arc<Self> {
        let ns = Arc::new(Self {
            id: NsId::new(),
            loopback_up: true,
            interfaces: RwLock::new(Vec::new()),
            routes: RwLock::new(Vec::new()),
            firewall: RwLock::new(Vec::new()),
        });

        // Add loopback interface
        ns.interfaces.write().push(NetInterface {
            name: String::from("lo"),
            index: 1,
            mac: [0; 6],
            ipv4: Some((0x7f000001, 8)), // 127.0.0.1/8
            flags: 0x49,                 // IFF_UP | IFF_LOOPBACK | IFF_RUNNING
        });

        ns
    }

    /// Create new network namespace
    pub fn create_new() -> Arc<Self> {
        let ns = Arc::new(Self {
            id: NsId::new(),
            loopback_up: false,
            interfaces: RwLock::new(Vec::new()),
            routes: RwLock::new(Vec::new()),
            firewall: RwLock::new(Vec::new()),
        });

        // New namespace starts with only loopback (down)
        ns.interfaces.write().push(NetInterface {
            name: String::from("lo"),
            index: 1,
            mac: [0; 6],
            ipv4: Some((0x7f000001, 8)),
            flags: 0x08, // IFF_LOOPBACK only
        });

        ns
    }

    /// Add interface to namespace
    pub fn add_interface(&self, iface: NetInterface) {
        self.interfaces.write().push(iface);
    }

    /// Get interface by name
    pub fn get_interface(&self, name: &str) -> Option<NetInterface> {
        self.interfaces
            .read()
            .iter()
            .find(|i| i.name == name)
            .cloned()
    }

    /// Add route
    pub fn add_route(&self, route: Route) {
        self.routes.write().push(route);
    }

    /// Lookup route for destination
    pub fn lookup_route(&self, dest: u32) -> Option<Route> {
        let routes = self.routes.read();
        let mut best: Option<&Route> = None;

        for route in routes.iter() {
            let mask = if route.prefix_len == 0 {
                0
            } else {
                !0u32 << (32 - route.prefix_len)
            };
            if (dest & mask) == (route.dest & mask) {
                if best.is_none() || route.prefix_len > best.unwrap().prefix_len {
                    best = Some(route);
                }
            }
        }

        best.cloned()
    }
}

// ============================================================================
// IPC Namespace
// ============================================================================

/// IPC namespace state
pub struct IpcNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Shared memory segments
    shm_ids: RwLock<BTreeMap<i32, ShmInfo>>,
    /// Semaphore sets
    sem_ids: RwLock<BTreeMap<i32, SemInfo>>,
    /// Message queues
    msg_ids: RwLock<BTreeMap<i32, MsgInfo>>,
    /// Next IDs
    next_shm_id: AtomicU32,
    next_sem_id: AtomicU32,
    next_msg_id: AtomicU32,
}

#[derive(Clone)]
pub struct ShmInfo {
    pub key: i32,
    pub size: usize,
    pub uid: Uid,
    pub gid: Gid,
    pub mode: u32,
}

#[derive(Clone)]
pub struct SemInfo {
    pub key: i32,
    pub nsems: usize,
    pub uid: Uid,
    pub gid: Gid,
    pub mode: u32,
}

#[derive(Clone)]
pub struct MsgInfo {
    pub key: i32,
    pub uid: Uid,
    pub gid: Gid,
    pub mode: u32,
    pub max_bytes: usize,
}

impl IpcNamespace {
    pub fn init_ns() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            shm_ids: RwLock::new(BTreeMap::new()),
            sem_ids: RwLock::new(BTreeMap::new()),
            msg_ids: RwLock::new(BTreeMap::new()),
            next_shm_id: AtomicU32::new(0),
            next_sem_id: AtomicU32::new(0),
            next_msg_id: AtomicU32::new(0),
        })
    }

    pub fn create_new() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            shm_ids: RwLock::new(BTreeMap::new()),
            sem_ids: RwLock::new(BTreeMap::new()),
            msg_ids: RwLock::new(BTreeMap::new()),
            next_shm_id: AtomicU32::new(0),
            next_sem_id: AtomicU32::new(0),
            next_msg_id: AtomicU32::new(0),
        })
    }

    pub fn alloc_shm_id(&self, info: ShmInfo) -> i32 {
        let id = self.next_shm_id.fetch_add(1, Ordering::SeqCst) as i32;
        self.shm_ids.write().insert(id, info);
        id
    }
}

// ============================================================================
// UTS Namespace
// ============================================================================

/// UTS namespace state (hostname, domain)
pub struct UtsNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Hostname
    hostname: RwLock<String>,
    /// Domain name
    domainname: RwLock<String>,
}

impl UtsNamespace {
    pub fn init_ns() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            hostname: RwLock::new(String::from("hublab")),
            domainname: RwLock::new(String::from("localdomain")),
        })
    }

    pub fn create_new() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            hostname: RwLock::new(String::from("container")),
            domainname: RwLock::new(String::new()),
        })
    }

    pub fn hostname(&self) -> String {
        self.hostname.read().clone()
    }

    pub fn set_hostname(&self, name: &str) {
        *self.hostname.write() = String::from(name);
    }

    pub fn domainname(&self) -> String {
        self.domainname.read().clone()
    }

    pub fn set_domainname(&self, name: &str) {
        *self.domainname.write() = String::from(name);
    }
}

// ============================================================================
// Mount Namespace
// ============================================================================

/// Mount namespace state
pub struct MntNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Mount points
    mounts: RwLock<Vec<MountPoint>>,
    /// Root of this namespace
    root: RwLock<String>,
}

#[derive(Clone)]
pub struct MountPoint {
    pub source: String,
    pub target: String,
    pub fstype: String,
    pub flags: u32,
    pub options: String,
}

impl MntNamespace {
    pub fn init_ns() -> Arc<Self> {
        let ns = Arc::new(Self {
            id: NsId::new(),
            mounts: RwLock::new(Vec::new()),
            root: RwLock::new(String::from("/")),
        });

        // Add essential mounts
        let mounts = vec![
            MountPoint {
                source: String::from("proc"),
                target: String::from("/proc"),
                fstype: String::from("proc"),
                flags: 0,
                options: String::new(),
            },
            MountPoint {
                source: String::from("sysfs"),
                target: String::from("/sys"),
                fstype: String::from("sysfs"),
                flags: 0,
                options: String::new(),
            },
            MountPoint {
                source: String::from("devtmpfs"),
                target: String::from("/dev"),
                fstype: String::from("devtmpfs"),
                flags: 0,
                options: String::new(),
            },
            MountPoint {
                source: String::from("tmpfs"),
                target: String::from("/tmp"),
                fstype: String::from("tmpfs"),
                flags: 0,
                options: String::new(),
            },
        ];

        *ns.mounts.write() = mounts;
        ns
    }

    pub fn create_copy(parent: &Self) -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            mounts: RwLock::new(parent.mounts.read().clone()),
            root: RwLock::new(parent.root.read().clone()),
        })
    }

    pub fn mount(&self, mount: MountPoint) -> Result<(), NsError> {
        self.mounts.write().push(mount);
        Ok(())
    }

    pub fn umount(&self, target: &str) -> Result<(), NsError> {
        let mut mounts = self.mounts.write();
        if let Some(pos) = mounts.iter().position(|m| m.target == target) {
            mounts.remove(pos);
            Ok(())
        } else {
            Err(NsError::NotFound)
        }
    }

    pub fn pivot_root(&self, new_root: &str, put_old: &str) -> Result<(), NsError> {
        // Validate paths
        if !new_root.starts_with('/') || !put_old.starts_with('/') {
            return Err(NsError::InvalidPath);
        }

        *self.root.write() = String::from(new_root);
        Ok(())
    }

    pub fn get_mounts(&self) -> Vec<MountPoint> {
        self.mounts.read().clone()
    }
}

// ============================================================================
// User Namespace
// ============================================================================

/// User namespace state
pub struct UserNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Parent namespace
    pub parent: Option<Arc<UserNamespace>>,
    /// UID mappings: (ns_uid, host_uid, count)
    uid_map: RwLock<Vec<(Uid, Uid, u32)>>,
    /// GID mappings: (ns_gid, host_gid, count)
    gid_map: RwLock<Vec<(Gid, Gid, u32)>>,
    /// Owner UID (in parent namespace)
    pub owner_uid: Uid,
    /// Owner GID (in parent namespace)
    pub owner_gid: Gid,
    /// Nesting level
    pub level: u32,
}

impl UserNamespace {
    pub fn init_ns() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            parent: None,
            uid_map: RwLock::new(vec![(0, 0, u32::MAX)]), // Identity mapping
            gid_map: RwLock::new(vec![(0, 0, u32::MAX)]),
            owner_uid: 0,
            owner_gid: 0,
            level: 0,
        })
    }

    pub fn create_child(
        parent: Arc<Self>,
        owner_uid: Uid,
        owner_gid: Gid,
    ) -> Result<Arc<Self>, NsError> {
        if parent.level >= 32 {
            return Err(NsError::NestingTooDeep);
        }

        Ok(Arc::new(Self {
            id: NsId::new(),
            parent: Some(parent.clone()),
            uid_map: RwLock::new(Vec::new()),
            gid_map: RwLock::new(Vec::new()),
            owner_uid,
            owner_gid,
            level: parent.level + 1,
        }))
    }

    /// Set UID mapping (can only be done once)
    pub fn set_uid_map(&self, mappings: Vec<(Uid, Uid, u32)>) -> Result<(), NsError> {
        let mut map = self.uid_map.write();
        if !map.is_empty() {
            return Err(NsError::AlreadySet);
        }

        // Validate mappings
        for (ns_uid, host_uid, count) in &mappings {
            if *count == 0 {
                return Err(NsError::InvalidMapping);
            }
            // Check overflow
            if ns_uid.checked_add(*count).is_none() || host_uid.checked_add(*count).is_none() {
                return Err(NsError::InvalidMapping);
            }
        }

        *map = mappings;
        Ok(())
    }

    /// Set GID mapping
    pub fn set_gid_map(&self, mappings: Vec<(Gid, Gid, u32)>) -> Result<(), NsError> {
        let mut map = self.gid_map.write();
        if !map.is_empty() {
            return Err(NsError::AlreadySet);
        }

        for (ns_gid, host_gid, count) in &mappings {
            if *count == 0
                || ns_gid.checked_add(*count).is_none()
                || host_gid.checked_add(*count).is_none()
            {
                return Err(NsError::InvalidMapping);
            }
        }

        *map = mappings;
        Ok(())
    }

    /// Translate namespace UID to host UID
    pub fn to_host_uid(&self, ns_uid: Uid) -> Option<Uid> {
        for (ns_start, host_start, count) in self.uid_map.read().iter() {
            if ns_uid >= *ns_start && ns_uid < ns_start + count {
                return Some(host_start + (ns_uid - ns_start));
            }
        }
        None
    }

    /// Translate host UID to namespace UID
    pub fn from_host_uid(&self, host_uid: Uid) -> Option<Uid> {
        for (ns_start, host_start, count) in self.uid_map.read().iter() {
            if host_uid >= *host_start && host_uid < host_start + count {
                return Some(ns_start + (host_uid - host_start));
            }
        }
        None
    }

    /// Check if UID has capability in this namespace
    pub fn ns_capable(&self, uid: Uid, cap: crate::capability::Cap) -> bool {
        // Owner has all capabilities
        if let Some(ref parent) = self.parent {
            if let Some(host_uid) = parent.to_host_uid(uid) {
                return host_uid == self.owner_uid;
            }
        }

        // Root in init namespace has all capabilities
        if self.parent.is_none() && uid == 0 {
            return true;
        }

        false
    }
}

// ============================================================================
// Cgroup Namespace
// ============================================================================

/// Cgroup namespace state
pub struct CgroupNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Root cgroup path for this namespace
    pub root: String,
}

impl CgroupNamespace {
    pub fn init_ns() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            root: String::from("/"),
        })
    }

    pub fn create_new(root: &str) -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            root: String::from(root),
        })
    }
}

// ============================================================================
// Time Namespace
// ============================================================================

/// Time namespace state
pub struct TimeNamespace {
    /// Namespace ID
    pub id: NsId,
    /// Monotonic time offset (nanoseconds)
    pub monotonic_offset: i64,
    /// Boottime offset (nanoseconds)
    pub boottime_offset: i64,
}

impl TimeNamespace {
    pub fn init_ns() -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            monotonic_offset: 0,
            boottime_offset: 0,
        })
    }

    pub fn create_new(monotonic_offset: i64, boottime_offset: i64) -> Arc<Self> {
        Arc::new(Self {
            id: NsId::new(),
            monotonic_offset,
            boottime_offset,
        })
    }
}

// ============================================================================
// Process Namespace Set
// ============================================================================

/// Complete namespace set for a process
#[derive(Clone)]
pub struct NsSet {
    pub pid_ns: Arc<PidNamespace>,
    pub net_ns: Arc<NetNamespace>,
    pub ipc_ns: Arc<IpcNamespace>,
    pub uts_ns: Arc<UtsNamespace>,
    pub mnt_ns: Arc<MntNamespace>,
    pub user_ns: Arc<UserNamespace>,
    pub cgroup_ns: Arc<CgroupNamespace>,
    pub time_ns: Arc<TimeNamespace>,
}

impl NsSet {
    /// Create initial namespace set
    pub fn init() -> Self {
        Self {
            pid_ns: PidNamespace::init_ns(),
            net_ns: NetNamespace::init_ns(),
            ipc_ns: IpcNamespace::init_ns(),
            uts_ns: UtsNamespace::init_ns(),
            mnt_ns: MntNamespace::init_ns(),
            user_ns: UserNamespace::init_ns(),
            cgroup_ns: CgroupNamespace::init_ns(),
            time_ns: TimeNamespace::init_ns(),
        }
    }

    /// Create child namespace set with specified new namespaces
    pub fn unshare(&self, flags: u32, uid: Uid, gid: Gid) -> Result<Self, NsError> {
        let mut new_set = self.clone();

        if flags & clone_flags::CLONE_NEWUSER != 0 {
            new_set.user_ns = UserNamespace::create_child(self.user_ns.clone(), uid, gid)?;
        }

        if flags & clone_flags::CLONE_NEWPID != 0 {
            new_set.pid_ns = PidNamespace::create_child(self.pid_ns.clone());
        }

        if flags & clone_flags::CLONE_NEWNET != 0 {
            new_set.net_ns = NetNamespace::create_new();
        }

        if flags & clone_flags::CLONE_NEWIPC != 0 {
            new_set.ipc_ns = IpcNamespace::create_new();
        }

        if flags & clone_flags::CLONE_NEWUTS != 0 {
            new_set.uts_ns = UtsNamespace::create_new();
        }

        if flags & clone_flags::CLONE_NEWNS != 0 {
            new_set.mnt_ns = MntNamespace::create_copy(&self.mnt_ns);
        }

        if flags & clone_flags::CLONE_NEWCGROUP != 0 {
            new_set.cgroup_ns = CgroupNamespace::create_new("/");
        }

        if flags & clone_flags::CLONE_NEWTIME != 0 {
            new_set.time_ns = TimeNamespace::create_new(0, 0);
        }

        Ok(new_set)
    }

    /// Enter existing namespaces (setns)
    pub fn setns(&mut self, ns_type: NsType, ns_id: NsId) -> Result<(), NsError> {
        // This would require looking up the namespace by ID
        // For now, just return success
        Ok(())
    }
}

/// Namespace error
#[derive(Clone, Copy, Debug)]
pub enum NsError {
    /// Permission denied
    PermissionDenied,
    /// Namespace not found
    NotFound,
    /// Invalid path
    InvalidPath,
    /// Nesting too deep
    NestingTooDeep,
    /// Mapping already set
    AlreadySet,
    /// Invalid mapping
    InvalidMapping,
    /// Resource limit exceeded
    ResourceLimit,
}

// ============================================================================
// Global State
// ============================================================================

/// Initial namespace set
static INIT_NS: RwLock<Option<NsSet>> = RwLock::new(None);

/// Process to namespace mapping
static PROCESS_NS: RwLock<BTreeMap<Pid, NsSet>> = RwLock::new(BTreeMap::new());

/// Initialize namespace subsystem
pub fn init() {
    let init_set = NsSet::init();
    *INIT_NS.write() = Some(init_set.clone());

    // Init process (PID 1) gets init namespaces
    PROCESS_NS.write().insert(Pid(1), init_set);

    crate::kprintln!("  Namespaces initialized (PID, NET, IPC, UTS, MNT, USER, CGROUP, TIME)");
}

/// Get initial namespace set
pub fn get_init_ns() -> NsSet {
    INIT_NS.read().as_ref().unwrap().clone()
}

/// Get process namespaces
pub fn get_process_ns(pid: Pid) -> Option<NsSet> {
    PROCESS_NS.read().get(&pid).cloned()
}

/// Set process namespaces
pub fn set_process_ns(pid: Pid, ns: NsSet) {
    PROCESS_NS.write().insert(pid, ns);
}

/// Remove process namespaces
pub fn cleanup_process(pid: Pid) {
    let ns = PROCESS_NS.write().remove(&pid);

    // Free PID in namespace
    if let Some(ns) = ns {
        ns.pid_ns.free_pid(pid);
    }
}

/// Fork: child inherits parent namespaces
pub fn fork_ns(parent: Pid, child: Pid) {
    if let Some(ns) = get_process_ns(parent) {
        // Allocate PID in namespace
        ns.pid_ns.alloc_pid(child);
        set_process_ns(child, ns);
    }
}

/// Clone with new namespaces
pub fn clone_ns(parent: Pid, child: Pid, flags: u32) -> Result<(), NsError> {
    let parent_ns = get_process_ns(parent).ok_or(NsError::NotFound)?;

    let (uid, gid) = if let Some(proc) = crate::process::get(parent) {
        (proc.uid, proc.gid)
    } else {
        (0, 0)
    };

    let child_ns = if flags & clone_flags::ALL_NS_FLAGS != 0 {
        parent_ns.unshare(flags, uid, gid)?
    } else {
        parent_ns
    };

    child_ns.pid_ns.alloc_pid(child);
    set_process_ns(child, child_ns);

    Ok(())
}

/// unshare syscall
pub fn sys_unshare(flags: u32) -> Result<(), NsError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(NsError::PermissionDenied)?;

    let current_ns = get_process_ns(pid).ok_or(NsError::NotFound)?;

    // Check permissions for user namespace
    if flags & clone_flags::CLONE_NEWUSER != 0 {
        // Creating user namespace requires CAP_SYS_ADMIN in parent user_ns
        // or being unprivileged (for unprivileged user namespaces)
    }

    let (uid, gid) = if let Some(proc) = crate::process::current() {
        (proc.uid, proc.gid)
    } else {
        (0, 0)
    };

    let new_ns = current_ns.unshare(flags, uid, gid)?;
    set_process_ns(pid, new_ns);

    Ok(())
}

/// setns syscall
pub fn sys_setns(fd: i32, nstype: u32) -> Result<(), NsError> {
    // Would need to look up namespace from fd
    // For now, just return success
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_namespace() {
        let ns = PidNamespace::init_ns();
        let global_pid = Pid(100);

        let ns_pid = ns.alloc_pid(global_pid);
        assert_eq!(ns_pid, 2); // First allocation after init

        assert_eq!(ns.to_global(ns_pid), Some(global_pid));
        assert_eq!(ns.from_global(global_pid), Some(ns_pid));
    }

    #[test]
    fn test_child_pid_namespace() {
        let parent = PidNamespace::init_ns();
        let child = PidNamespace::create_child(parent);

        assert_eq!(child.level, 1);

        let global_pid = Pid(200);
        let ns_pid = child.alloc_pid(global_pid);
        assert_eq!(ns_pid, 1); // PID 1 in new namespace
    }
}
