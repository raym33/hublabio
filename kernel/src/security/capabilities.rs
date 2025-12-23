//! Capability-Based Security
//!
//! Linux-style capabilities for fine-grained privilege control.
//! Processes can have specific capabilities instead of full root access.

use alloc::vec::Vec;
use core::fmt;

/// Individual capabilities
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Capability {
    /// Change file ownership (chown)
    Chown = 0,
    /// Override DAC (discretionary access control)
    DacOverride = 1,
    /// Read files ignoring read permission
    DacReadSearch = 2,
    /// Override file ownership restrictions
    Fowner = 3,
    /// Bypass file setuid/setgid restrictions
    Fsetid = 4,
    /// Send signals to any process
    Kill = 5,
    /// Set process GID
    Setgid = 6,
    /// Set process UID
    Setuid = 7,
    /// Set capabilities on files
    Setfcap = 8,
    /// Lock memory (mlock)
    IpcLock = 9,
    /// Override IPC ownership checks
    IpcOwner = 10,
    /// Load/unload kernel modules
    SysModule = 11,
    /// Use raw I/O ports
    SysRawio = 12,
    /// Use chroot
    SysChroot = 13,
    /// Trace any process (ptrace)
    SysPtrace = 14,
    /// Configure process accounting
    SysPacct = 15,
    /// System administration (reboot, etc.)
    SysAdmin = 16,
    /// Use reboot()
    SysBoot = 17,
    /// Set process nice value
    SysNice = 18,
    /// Override resource limits
    SysResource = 19,
    /// Set system time
    SysTime = 20,
    /// Configure TTY
    SysTtyConfig = 21,
    /// Create special files (mknod)
    Mknod = 22,
    /// Establish leases on files
    Lease = 23,
    /// Write audit log
    AuditWrite = 24,
    /// Configure audit subsystem
    AuditControl = 25,
    /// Set file capabilities
    Setpcap = 26,
    /// Bind to ports < 1024
    NetBindService = 27,
    /// Configure network interfaces
    NetAdmin = 28,
    /// Use raw sockets
    NetRaw = 29,
    /// Broadcast/multicast
    NetBroadcast = 30,
    /// Use MAC (mandatory access control)
    MacOverride = 31,
    /// Configure MAC
    MacAdmin = 32,
    /// Configure kernel syslog
    Syslog = 33,
    /// Use wake-on-lan
    WakeAlarm = 34,
    /// Block system suspend
    BlockSuspend = 35,
    /// Read audit log
    AuditRead = 36,
    /// Use perf events
    Perfmon = 37,
    /// Use BPF
    Bpf = 38,
    /// Checkpoint/restore
    CheckpointRestore = 39,

    // HubLab IO specific capabilities
    /// Load/manage AI models
    AiModel = 64,
    /// Run AI inference
    AiInference = 65,
    /// Access NPU/GPU hardware
    AiHardware = 66,
    /// Join distributed cluster
    ClusterJoin = 67,
    /// Manage cluster
    ClusterAdmin = 68,
    /// Access sensors
    SensorAccess = 69,
    /// Control GPIO
    GpioControl = 70,
    /// Access camera
    CameraAccess = 71,
    /// Access microphone
    MicrophoneAccess = 72,
    /// Use TTS/STT
    VoiceAccess = 73,
}

impl Capability {
    /// Get capability name
    pub fn name(&self) -> &'static str {
        match self {
            Capability::Chown => "CAP_CHOWN",
            Capability::DacOverride => "CAP_DAC_OVERRIDE",
            Capability::DacReadSearch => "CAP_DAC_READ_SEARCH",
            Capability::Fowner => "CAP_FOWNER",
            Capability::Fsetid => "CAP_FSETID",
            Capability::Kill => "CAP_KILL",
            Capability::Setgid => "CAP_SETGID",
            Capability::Setuid => "CAP_SETUID",
            Capability::Setfcap => "CAP_SETFCAP",
            Capability::IpcLock => "CAP_IPC_LOCK",
            Capability::IpcOwner => "CAP_IPC_OWNER",
            Capability::SysModule => "CAP_SYS_MODULE",
            Capability::SysRawio => "CAP_SYS_RAWIO",
            Capability::SysChroot => "CAP_SYS_CHROOT",
            Capability::SysPtrace => "CAP_SYS_PTRACE",
            Capability::SysPacct => "CAP_SYS_PACCT",
            Capability::SysAdmin => "CAP_SYS_ADMIN",
            Capability::SysBoot => "CAP_SYS_BOOT",
            Capability::SysNice => "CAP_SYS_NICE",
            Capability::SysResource => "CAP_SYS_RESOURCE",
            Capability::SysTime => "CAP_SYS_TIME",
            Capability::SysTtyConfig => "CAP_SYS_TTY_CONFIG",
            Capability::Mknod => "CAP_MKNOD",
            Capability::Lease => "CAP_LEASE",
            Capability::AuditWrite => "CAP_AUDIT_WRITE",
            Capability::AuditControl => "CAP_AUDIT_CONTROL",
            Capability::Setpcap => "CAP_SETPCAP",
            Capability::NetBindService => "CAP_NET_BIND_SERVICE",
            Capability::NetAdmin => "CAP_NET_ADMIN",
            Capability::NetRaw => "CAP_NET_RAW",
            Capability::NetBroadcast => "CAP_NET_BROADCAST",
            Capability::MacOverride => "CAP_MAC_OVERRIDE",
            Capability::MacAdmin => "CAP_MAC_ADMIN",
            Capability::Syslog => "CAP_SYSLOG",
            Capability::WakeAlarm => "CAP_WAKE_ALARM",
            Capability::BlockSuspend => "CAP_BLOCK_SUSPEND",
            Capability::AuditRead => "CAP_AUDIT_READ",
            Capability::Perfmon => "CAP_PERFMON",
            Capability::Bpf => "CAP_BPF",
            Capability::CheckpointRestore => "CAP_CHECKPOINT_RESTORE",
            Capability::AiModel => "CAP_AI_MODEL",
            Capability::AiInference => "CAP_AI_INFERENCE",
            Capability::AiHardware => "CAP_AI_HARDWARE",
            Capability::ClusterJoin => "CAP_CLUSTER_JOIN",
            Capability::ClusterAdmin => "CAP_CLUSTER_ADMIN",
            Capability::SensorAccess => "CAP_SENSOR_ACCESS",
            Capability::GpioControl => "CAP_GPIO_CONTROL",
            Capability::CameraAccess => "CAP_CAMERA_ACCESS",
            Capability::MicrophoneAccess => "CAP_MICROPHONE_ACCESS",
            Capability::VoiceAccess => "CAP_VOICE_ACCESS",
        }
    }

    /// Get all standard Linux capabilities
    pub fn linux_caps() -> &'static [Capability] {
        &[
            Capability::Chown,
            Capability::DacOverride,
            Capability::DacReadSearch,
            Capability::Fowner,
            Capability::Fsetid,
            Capability::Kill,
            Capability::Setgid,
            Capability::Setuid,
            Capability::Setfcap,
            Capability::IpcLock,
            Capability::IpcOwner,
            Capability::SysModule,
            Capability::SysRawio,
            Capability::SysChroot,
            Capability::SysPtrace,
            Capability::SysPacct,
            Capability::SysAdmin,
            Capability::SysBoot,
            Capability::SysNice,
            Capability::SysResource,
            Capability::SysTime,
            Capability::SysTtyConfig,
            Capability::Mknod,
            Capability::Lease,
            Capability::AuditWrite,
            Capability::AuditControl,
            Capability::Setpcap,
            Capability::NetBindService,
            Capability::NetAdmin,
            Capability::NetRaw,
            Capability::NetBroadcast,
            Capability::MacOverride,
            Capability::MacAdmin,
            Capability::Syslog,
            Capability::WakeAlarm,
            Capability::BlockSuspend,
            Capability::AuditRead,
            Capability::Perfmon,
            Capability::Bpf,
            Capability::CheckpointRestore,
        ]
    }

    /// Get HubLab IO specific capabilities
    pub fn hublab_caps() -> &'static [Capability] {
        &[
            Capability::AiModel,
            Capability::AiInference,
            Capability::AiHardware,
            Capability::ClusterJoin,
            Capability::ClusterAdmin,
            Capability::SensorAccess,
            Capability::GpioControl,
            Capability::CameraAccess,
            Capability::MicrophoneAccess,
            Capability::VoiceAccess,
        ]
    }
}

/// Set of capabilities
#[derive(Clone, Debug, Default)]
pub struct CapabilitySet {
    /// Bitmask for capabilities 0-63
    bits_low: u64,
    /// Bitmask for capabilities 64-127
    bits_high: u64,
}

impl CapabilitySet {
    /// Create empty capability set
    pub fn empty() -> Self {
        Self {
            bits_low: 0,
            bits_high: 0,
        }
    }

    /// Create capability set with all capabilities
    pub fn all() -> Self {
        Self {
            bits_low: u64::MAX,
            bits_high: u64::MAX,
        }
    }

    /// Create default set for unprivileged user
    pub fn user_default() -> Self {
        let mut set = Self::empty();
        // Users can run AI inference by default
        set.add(Capability::AiInference);
        // Users can use voice by default
        set.add(Capability::VoiceAccess);
        set
    }

    /// Create set for system service
    pub fn service_default() -> Self {
        let mut set = Self::user_default();
        set.add(Capability::NetBindService);
        set.add(Capability::AiModel);
        set.add(Capability::SensorAccess);
        set
    }

    /// Create set for AI service
    pub fn ai_service() -> Self {
        let mut set = Self::service_default();
        set.add(Capability::AiHardware);
        set.add(Capability::ClusterJoin);
        set.add(Capability::IpcLock); // For pinning model memory
        set
    }

    /// Add capability to set
    pub fn add(&mut self, cap: Capability) {
        let bit = cap as u32;
        if bit < 64 {
            self.bits_low |= 1u64 << bit;
        } else {
            self.bits_high |= 1u64 << (bit - 64);
        }
    }

    /// Remove capability from set
    pub fn remove(&mut self, cap: Capability) {
        let bit = cap as u32;
        if bit < 64 {
            self.bits_low &= !(1u64 << bit);
        } else {
            self.bits_high &= !(1u64 << (bit - 64));
        }
    }

    /// Check if set has capability
    pub fn has(&self, cap: Capability) -> bool {
        let bit = cap as u32;
        if bit < 64 {
            (self.bits_low & (1u64 << bit)) != 0
        } else {
            (self.bits_high & (1u64 << (bit - 64))) != 0
        }
    }

    /// Union of two sets
    pub fn union(&self, other: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            bits_low: self.bits_low | other.bits_low,
            bits_high: self.bits_high | other.bits_high,
        }
    }

    /// Intersection of two sets
    pub fn intersection(&self, other: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            bits_low: self.bits_low & other.bits_low,
            bits_high: self.bits_high & other.bits_high,
        }
    }

    /// Check if set is subset of another
    pub fn is_subset_of(&self, other: &CapabilitySet) -> bool {
        (self.bits_low & !other.bits_low) == 0 &&
        (self.bits_high & !other.bits_high) == 0
    }

    /// Check if set is empty
    pub fn is_empty(&self) -> bool {
        self.bits_low == 0 && self.bits_high == 0
    }

    /// Get list of capabilities in set
    pub fn list(&self) -> Vec<Capability> {
        let mut caps = Vec::new();

        for cap in Capability::linux_caps() {
            if self.has(*cap) {
                caps.push(*cap);
            }
        }

        for cap in Capability::hublab_caps() {
            if self.has(*cap) {
                caps.push(*cap);
            }
        }

        caps
    }

    /// Count capabilities in set
    pub fn count(&self) -> usize {
        (self.bits_low.count_ones() + self.bits_high.count_ones()) as usize
    }
}

impl fmt::Display for CapabilitySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let caps = self.list();
        if caps.is_empty() {
            write!(f, "(none)")
        } else {
            let names: Vec<&str> = caps.iter().map(|c| c.name()).collect();
            write!(f, "{}", names.join(", "))
        }
    }
}

/// Capability bounding set for a process
#[derive(Clone, Debug)]
pub struct CapabilityBounds {
    /// Effective capabilities (currently in use)
    pub effective: CapabilitySet,
    /// Permitted capabilities (can be gained)
    pub permitted: CapabilitySet,
    /// Inheritable capabilities (passed to children)
    pub inheritable: CapabilitySet,
    /// Ambient capabilities (auto-effective for unprivileged)
    pub ambient: CapabilitySet,
    /// Bounding set (maximum capabilities)
    pub bounding: CapabilitySet,
}

impl CapabilityBounds {
    /// Create bounds for root
    pub fn root() -> Self {
        Self {
            effective: CapabilitySet::all(),
            permitted: CapabilitySet::all(),
            inheritable: CapabilitySet::all(),
            ambient: CapabilitySet::empty(),
            bounding: CapabilitySet::all(),
        }
    }

    /// Create bounds for regular user
    pub fn user() -> Self {
        let user_caps = CapabilitySet::user_default();
        Self {
            effective: user_caps.clone(),
            permitted: user_caps.clone(),
            inheritable: CapabilitySet::empty(),
            ambient: user_caps.clone(),
            bounding: user_caps,
        }
    }

    /// Create bounds for service
    pub fn service() -> Self {
        let service_caps = CapabilitySet::service_default();
        Self {
            effective: service_caps.clone(),
            permitted: service_caps.clone(),
            inheritable: CapabilitySet::empty(),
            ambient: service_caps.clone(),
            bounding: service_caps,
        }
    }

    /// Drop capability from all sets
    pub fn drop(&mut self, cap: Capability) {
        self.effective.remove(cap);
        self.permitted.remove(cap);
        self.inheritable.remove(cap);
        self.ambient.remove(cap);
        self.bounding.remove(cap);
    }

    /// Raise capability (move from permitted to effective)
    pub fn raise(&mut self, cap: Capability) -> bool {
        if self.permitted.has(cap) && self.bounding.has(cap) {
            self.effective.add(cap);
            true
        } else {
            false
        }
    }

    /// Lower capability (remove from effective)
    pub fn lower(&mut self, cap: Capability) {
        self.effective.remove(cap);
    }

    /// Check if capability can be used
    pub fn can_use(&self, cap: Capability) -> bool {
        self.effective.has(cap)
    }

    /// Calculate capabilities for exec
    pub fn on_exec(&self, file_caps: &FileCaps) -> CapabilityBounds {
        // P'(ambient) = P(ambient)
        // P'(permitted) = (P(inheritable) & F(inheritable)) |
        //                 (F(permitted) & P(bounding)) | P(ambient)
        // P'(effective) = F(effective) ? P'(permitted) : P'(ambient)
        // P'(inheritable) = P(inheritable)

        let new_permitted = self.inheritable.intersection(&file_caps.inheritable)
            .union(&file_caps.permitted.intersection(&self.bounding))
            .union(&self.ambient);

        let new_effective = if file_caps.effective {
            new_permitted.clone()
        } else {
            self.ambient.clone()
        };

        CapabilityBounds {
            effective: new_effective.intersection(&self.bounding),
            permitted: new_permitted.intersection(&self.bounding),
            inheritable: self.inheritable.clone(),
            ambient: self.ambient.intersection(&new_permitted),
            bounding: self.bounding.clone(),
        }
    }
}

/// File capabilities (stored in extended attributes)
#[derive(Clone, Debug, Default)]
pub struct FileCaps {
    /// Permitted capabilities
    pub permitted: CapabilitySet,
    /// Inheritable capabilities
    pub inheritable: CapabilitySet,
    /// Raise effective on exec
    pub effective: bool,
    /// Root UID that owns these caps
    pub rootid: u32,
}

impl FileCaps {
    /// Create empty file caps
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create file caps for setuid-like behavior
    pub fn setuid_like(caps: CapabilitySet) -> Self {
        Self {
            permitted: caps,
            inheritable: CapabilitySet::empty(),
            effective: true,
            rootid: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_set() {
        let mut set = CapabilitySet::empty();
        assert!(set.is_empty());

        set.add(Capability::NetBindService);
        assert!(set.has(Capability::NetBindService));
        assert!(!set.has(Capability::SysAdmin));

        set.remove(Capability::NetBindService);
        assert!(!set.has(Capability::NetBindService));
    }

    #[test]
    fn test_capability_set_operations() {
        let mut set1 = CapabilitySet::empty();
        set1.add(Capability::NetBindService);
        set1.add(Capability::AiInference);

        let mut set2 = CapabilitySet::empty();
        set2.add(Capability::AiInference);
        set2.add(Capability::VoiceAccess);

        let union = set1.union(&set2);
        assert!(union.has(Capability::NetBindService));
        assert!(union.has(Capability::AiInference));
        assert!(union.has(Capability::VoiceAccess));

        let intersection = set1.intersection(&set2);
        assert!(!intersection.has(Capability::NetBindService));
        assert!(intersection.has(Capability::AiInference));
        assert!(!intersection.has(Capability::VoiceAccess));
    }

    #[test]
    fn test_user_default_caps() {
        let caps = CapabilitySet::user_default();
        assert!(caps.has(Capability::AiInference));
        assert!(caps.has(Capability::VoiceAccess));
        assert!(!caps.has(Capability::SysAdmin));
    }
}
