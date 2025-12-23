//! POSIX Capabilities System
//!
//! Fine-grained privilege management following POSIX 1003.1e draft.
//! Replaces the all-or-nothing root model with specific capabilities.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::RwLock;

use crate::process::Pid;

/// Capability identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum Cap {
    /// Allow changing file ownership (chown)
    Chown = 0,
    /// Bypass file permission checks (DAC = discretionary access control)
    DacOverride = 1,
    /// Bypass file read permission checks
    DacReadSearch = 2,
    /// Bypass file ownership restrictions on setuid/setgid
    Fowner = 3,
    /// Bypass permission checks for sending signals
    Kill = 4,
    /// Allow setting arbitrary process GIDs
    Setgid = 5,
    /// Allow setting arbitrary process UIDs
    Setuid = 6,
    /// Allow setting file capabilities
    Setfcap = 7,
    /// Allow privileged port binding (< 1024)
    NetBindService = 8,
    /// Allow raw socket access
    NetRaw = 9,
    /// Allow network administration
    NetAdmin = 10,
    /// Allow system administration operations
    SysAdmin = 11,
    /// Allow loading/unloading kernel modules
    SysModule = 12,
    /// Allow setting system time
    SysTime = 13,
    /// Allow rebooting/shutdown
    SysBootreboot = 14,
    /// Allow mounting/unmounting filesystems
    SysMountUmount = 15,
    /// Allow changing resource limits
    SysResource = 16,
    /// Allow ptrace of any process
    SysPtrace = 17,
    /// Allow nice value changes and scheduling policy
    SysNice = 18,
    /// Allow I/O port operations
    SysRawio = 19,
    /// Allow memory locking (mlock)
    IpcLock = 20,
    /// Allow IPC object ownership changes
    IpcOwner = 21,
    /// Allow audit configuration
    AuditControl = 22,
    /// Allow writing audit logs
    AuditWrite = 23,
    /// Allow chroot
    SysChroot = 24,
    /// Allow TTY configuration
    SysTtyConfig = 25,
    /// Allow lease on file descriptors
    Lease = 26,
    /// Allow setting process securebits
    Setpcap = 27,
    /// Allow MAC override (SELinux/AppArmor)
    MacOverride = 28,
    /// Allow MAC administration
    MacAdmin = 29,
    /// Allow syslog operations
    Syslog = 30,
    /// Allow kernel keyring operations
    WakeAlarm = 31,
    /// Allow blocking system suspend
    BlockSuspend = 32,
    /// Allow reading audit logs
    AuditRead = 33,
    /// Allow perfmon operations
    Perfmon = 34,
    /// Allow BPF operations
    Bpf = 35,
    /// Allow checkpoint/restore
    CheckpointRestore = 36,
}

impl Cap {
    /// Total number of capabilities
    pub const COUNT: usize = 37;

    /// Convert from u32
    pub fn from_u32(v: u32) -> Option<Cap> {
        if v < Self::COUNT as u32 {
            // Safe because we checked bounds
            Some(unsafe { core::mem::transmute(v) })
        } else {
            None
        }
    }

    /// Get capability name
    pub fn name(&self) -> &'static str {
        match self {
            Cap::Chown => "CAP_CHOWN",
            Cap::DacOverride => "CAP_DAC_OVERRIDE",
            Cap::DacReadSearch => "CAP_DAC_READ_SEARCH",
            Cap::Fowner => "CAP_FOWNER",
            Cap::Kill => "CAP_KILL",
            Cap::Setgid => "CAP_SETGID",
            Cap::Setuid => "CAP_SETUID",
            Cap::Setfcap => "CAP_SETFCAP",
            Cap::NetBindService => "CAP_NET_BIND_SERVICE",
            Cap::NetRaw => "CAP_NET_RAW",
            Cap::NetAdmin => "CAP_NET_ADMIN",
            Cap::SysAdmin => "CAP_SYS_ADMIN",
            Cap::SysModule => "CAP_SYS_MODULE",
            Cap::SysTime => "CAP_SYS_TIME",
            Cap::SysBootreboot => "CAP_SYS_BOOT",
            Cap::SysMountUmount => "CAP_SYS_MOUNT",
            Cap::SysResource => "CAP_SYS_RESOURCE",
            Cap::SysPtrace => "CAP_SYS_PTRACE",
            Cap::SysNice => "CAP_SYS_NICE",
            Cap::SysRawio => "CAP_SYS_RAWIO",
            Cap::IpcLock => "CAP_IPC_LOCK",
            Cap::IpcOwner => "CAP_IPC_OWNER",
            Cap::AuditControl => "CAP_AUDIT_CONTROL",
            Cap::AuditWrite => "CAP_AUDIT_WRITE",
            Cap::SysChroot => "CAP_SYS_CHROOT",
            Cap::SysTtyConfig => "CAP_SYS_TTY_CONFIG",
            Cap::Lease => "CAP_LEASE",
            Cap::Setpcap => "CAP_SETPCAP",
            Cap::MacOverride => "CAP_MAC_OVERRIDE",
            Cap::MacAdmin => "CAP_MAC_ADMIN",
            Cap::Syslog => "CAP_SYSLOG",
            Cap::WakeAlarm => "CAP_WAKE_ALARM",
            Cap::BlockSuspend => "CAP_BLOCK_SUSPEND",
            Cap::AuditRead => "CAP_AUDIT_READ",
            Cap::Perfmon => "CAP_PERFMON",
            Cap::Bpf => "CAP_BPF",
            Cap::CheckpointRestore => "CAP_CHECKPOINT_RESTORE",
        }
    }
}

/// Capability set stored as a bitmask
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapSet {
    bits: u64,
}

impl CapSet {
    /// Empty capability set
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    /// Full capability set (all capabilities)
    pub const fn full() -> Self {
        Self { bits: (1u64 << Cap::COUNT) - 1 }
    }

    /// Check if capability is present
    pub fn has(&self, cap: Cap) -> bool {
        self.bits & (1u64 << cap as u32) != 0
    }

    /// Add capability to set
    pub fn add(&mut self, cap: Cap) {
        self.bits |= 1u64 << cap as u32;
    }

    /// Remove capability from set
    pub fn remove(&mut self, cap: Cap) {
        self.bits &= !(1u64 << cap as u32);
    }

    /// Clear all capabilities
    pub fn clear(&mut self) {
        self.bits = 0;
    }

    /// Union of two sets
    pub fn union(&self, other: &CapSet) -> CapSet {
        CapSet { bits: self.bits | other.bits }
    }

    /// Intersection of two sets
    pub fn intersect(&self, other: &CapSet) -> CapSet {
        CapSet { bits: self.bits & other.bits }
    }

    /// Difference (self - other)
    pub fn difference(&self, other: &CapSet) -> CapSet {
        CapSet { bits: self.bits & !other.bits }
    }

    /// Check if set is empty
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// Count capabilities in set
    pub fn count(&self) -> usize {
        self.bits.count_ones() as usize
    }

    /// Iterate over capabilities
    pub fn iter(&self) -> impl Iterator<Item = Cap> + '_ {
        (0..Cap::COUNT as u32).filter_map(move |i| {
            if self.bits & (1u64 << i) != 0 {
                Cap::from_u32(i)
            } else {
                None
            }
        })
    }

    /// Create from raw bits
    pub fn from_bits(bits: u64) -> Self {
        Self { bits: bits & ((1u64 << Cap::COUNT) - 1) }
    }

    /// Get raw bits
    pub fn bits(&self) -> u64 {
        self.bits
    }
}

/// Process capability state
#[derive(Clone, Debug)]
pub struct ProcessCaps {
    /// Permitted: upper bound on capabilities
    pub permitted: CapSet,
    /// Effective: currently active capabilities
    pub effective: CapSet,
    /// Inheritable: capabilities preserved across exec
    pub inheritable: CapSet,
    /// Bounding set: limit on permitted after exec
    pub bounding: CapSet,
    /// Ambient: auto-added to permitted/effective on exec
    pub ambient: CapSet,
    /// Securebits flags
    pub securebits: SecureBits,
}

impl ProcessCaps {
    /// Create root capabilities (full access)
    pub fn root() -> Self {
        Self {
            permitted: CapSet::full(),
            effective: CapSet::full(),
            inheritable: CapSet::empty(),
            bounding: CapSet::full(),
            ambient: CapSet::empty(),
            securebits: SecureBits::default(),
        }
    }

    /// Create unprivileged capabilities (no access)
    pub fn unprivileged() -> Self {
        Self {
            permitted: CapSet::empty(),
            effective: CapSet::empty(),
            inheritable: CapSet::empty(),
            bounding: CapSet::full(),
            ambient: CapSet::empty(),
            securebits: SecureBits::default(),
        }
    }

    /// Check if process has capability in effective set
    pub fn has(&self, cap: Cap) -> bool {
        self.effective.has(cap)
    }

    /// Raise a capability (add to effective if permitted)
    pub fn raise(&mut self, cap: Cap) -> bool {
        if self.permitted.has(cap) {
            self.effective.add(cap);
            true
        } else {
            false
        }
    }

    /// Lower a capability (remove from effective)
    pub fn lower(&mut self, cap: Cap) {
        self.effective.remove(cap);
    }

    /// Drop capability from bounding set (permanent)
    pub fn drop_bounding(&mut self, cap: Cap) {
        self.bounding.remove(cap);
    }

    /// Set ambient capability (must be in permitted and inheritable)
    pub fn set_ambient(&mut self, cap: Cap) -> bool {
        if self.permitted.has(cap) && self.inheritable.has(cap) {
            self.ambient.add(cap);
            true
        } else {
            false
        }
    }

    /// Clear ambient capability
    pub fn clear_ambient(&mut self, cap: Cap) {
        self.ambient.remove(cap);
    }

    /// Apply capability transformation for exec
    pub fn transform_for_exec(&mut self, file_caps: Option<&FileCaps>) {
        let old_permitted = self.permitted;

        if let Some(fcaps) = file_caps {
            // P'(permitted) = (P(inheritable) & F(inheritable)) |
            //                 (F(permitted) & P(bounding)) | P'(ambient)
            self.permitted = self.inheritable.intersect(&fcaps.inheritable)
                .union(&fcaps.permitted.intersect(&self.bounding))
                .union(&self.ambient);

            // P'(effective) = F(effective) ? P'(permitted) : P'(ambient)
            if fcaps.effective {
                self.effective = self.permitted;
            } else {
                self.effective = self.ambient;
            }
        } else {
            // No file caps: only ambient survives
            self.permitted = self.ambient;
            self.effective = self.ambient;
        }

        // Handle securebits
        if self.securebits.no_root {
            // Root doesn't get special treatment
        } else {
            // Traditional root behavior could be here
        }

        // Inheritable unchanged
        // Bounding unchanged
    }

    /// Fork: child inherits all capability sets
    pub fn fork(&self) -> Self {
        self.clone()
    }
}

/// File capability state (stored in xattr)
#[derive(Clone, Debug, Default)]
pub struct FileCaps {
    /// Permitted capabilities granted by file
    pub permitted: CapSet,
    /// Inheritable capabilities required
    pub inheritable: CapSet,
    /// Whether to raise effective on exec
    pub effective: bool,
    /// Root UID for namespace-aware capabilities
    pub rootid: u32,
}

impl FileCaps {
    /// Parse from xattr data (VFS_CAP_REVISION_3 format)
    pub fn from_xattr(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let version = magic & 0xFF000000;

        match version {
            // VFS_CAP_REVISION_2 (0x02000000)
            0x02000000 => Self::parse_v2(data),
            // VFS_CAP_REVISION_3 (0x03000000)
            0x03000000 => Self::parse_v3(data),
            _ => None,
        }
    }

    fn parse_v2(data: &[u8]) -> Option<Self> {
        if data.len() < 20 {
            return None;
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let effective = (magic & 0x01) != 0;

        let permitted_lo = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let inheritable_lo = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let permitted_hi = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let inheritable_hi = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);

        Some(Self {
            permitted: CapSet::from_bits(
                (permitted_lo as u64) | ((permitted_hi as u64) << 32)
            ),
            inheritable: CapSet::from_bits(
                (inheritable_lo as u64) | ((inheritable_hi as u64) << 32)
            ),
            effective,
            rootid: 0,
        })
    }

    fn parse_v3(data: &[u8]) -> Option<Self> {
        if data.len() < 24 {
            return None;
        }

        let mut caps = Self::parse_v2(data)?;
        caps.rootid = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        Some(caps)
    }

    /// Serialize to xattr data
    pub fn to_xattr(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(24);

        // VFS_CAP_REVISION_3 with effective bit
        let magic: u32 = 0x03000000 | if self.effective { 1 } else { 0 };
        data.extend_from_slice(&magic.to_le_bytes());

        let permitted = self.permitted.bits();
        let inheritable = self.inheritable.bits();

        data.extend_from_slice(&(permitted as u32).to_le_bytes());
        data.extend_from_slice(&(inheritable as u32).to_le_bytes());
        data.extend_from_slice(&((permitted >> 32) as u32).to_le_bytes());
        data.extend_from_slice(&((inheritable >> 32) as u32).to_le_bytes());
        data.extend_from_slice(&self.rootid.to_le_bytes());

        data
    }
}

/// Securebits flags
#[derive(Clone, Copy, Debug, Default)]
pub struct SecureBits {
    /// Keep capabilities on UID 0 -> non-0 transition
    pub keep_caps: bool,
    /// Lock keep_caps
    pub keep_caps_locked: bool,
    /// Don't grant capabilities to root
    pub no_root: bool,
    /// Lock no_root
    pub no_root_locked: bool,
    /// Don't grant capabilities on exec of setuid-root
    pub no_setuid_fixup: bool,
    /// Lock no_setuid_fixup
    pub no_setuid_fixup_locked: bool,
    /// Disable ambient capabilities
    pub no_cap_ambient_raise: bool,
    /// Lock no_cap_ambient_raise
    pub no_cap_ambient_raise_locked: bool,
}

impl SecureBits {
    /// Parse from integer (prctl format)
    pub fn from_bits(bits: u32) -> Self {
        Self {
            keep_caps: bits & (1 << 4) != 0,
            keep_caps_locked: bits & (1 << 5) != 0,
            no_root: bits & (1 << 0) != 0,
            no_root_locked: bits & (1 << 1) != 0,
            no_setuid_fixup: bits & (1 << 2) != 0,
            no_setuid_fixup_locked: bits & (1 << 3) != 0,
            no_cap_ambient_raise: bits & (1 << 6) != 0,
            no_cap_ambient_raise_locked: bits & (1 << 7) != 0,
        }
    }

    /// Convert to integer
    pub fn to_bits(&self) -> u32 {
        let mut bits = 0u32;
        if self.keep_caps { bits |= 1 << 4; }
        if self.keep_caps_locked { bits |= 1 << 5; }
        if self.no_root { bits |= 1 << 0; }
        if self.no_root_locked { bits |= 1 << 1; }
        if self.no_setuid_fixup { bits |= 1 << 2; }
        if self.no_setuid_fixup_locked { bits |= 1 << 3; }
        if self.no_cap_ambient_raise { bits |= 1 << 6; }
        if self.no_cap_ambient_raise_locked { bits |= 1 << 7; }
        bits
    }
}

/// Global capability state for all processes
static PROCESS_CAPS: RwLock<alloc::collections::BTreeMap<Pid, ProcessCaps>> =
    RwLock::new(alloc::collections::BTreeMap::new());

/// Capability error
#[derive(Clone, Copy, Debug)]
pub enum CapError {
    /// Permission denied (EPERM)
    PermissionDenied,
    /// Invalid capability
    InvalidCapability,
    /// Operation not supported
    NotSupported,
}

/// Initialize capability for a new process
pub fn init_process(pid: Pid, is_root: bool) {
    let caps = if is_root {
        ProcessCaps::root()
    } else {
        ProcessCaps::unprivileged()
    };
    PROCESS_CAPS.write().insert(pid, caps);
}

/// Remove capability state for exited process
pub fn cleanup_process(pid: Pid) {
    PROCESS_CAPS.write().remove(&pid);
}

/// Get process capabilities
pub fn get_caps(pid: Pid) -> Option<ProcessCaps> {
    PROCESS_CAPS.read().get(&pid).cloned()
}

/// Set process capabilities (requires CAP_SETPCAP)
pub fn set_caps(pid: Pid, caps: ProcessCaps) -> Result<(), CapError> {
    // Verify caller has CAP_SETPCAP
    if let Some(current) = crate::process::current() {
        if !capable(current.pid, Cap::Setpcap) {
            return Err(CapError::PermissionDenied);
        }
    }

    PROCESS_CAPS.write().insert(pid, caps);
    Ok(())
}

/// Check if process has capability
pub fn capable(pid: Pid, cap: Cap) -> bool {
    PROCESS_CAPS.read()
        .get(&pid)
        .map(|c| c.has(cap))
        .unwrap_or(false)
}

/// Check if current process has capability
pub fn current_has(cap: Cap) -> bool {
    if let Some(proc) = crate::process::current() {
        capable(proc.pid, cap)
    } else {
        false
    }
}

/// Require capability or return error
pub fn require(cap: Cap) -> Result<(), CapError> {
    if current_has(cap) {
        Ok(())
    } else {
        Err(CapError::PermissionDenied)
    }
}

/// Drop capability from current process
pub fn drop_cap(cap: Cap) -> Result<(), CapError> {
    if let Some(proc) = crate::process::current() {
        let mut caps = PROCESS_CAPS.write();
        if let Some(pcaps) = caps.get_mut(&proc.pid) {
            pcaps.lower(cap);
            pcaps.permitted.remove(cap);
            return Ok(());
        }
    }
    Err(CapError::PermissionDenied)
}

/// capget syscall implementation
pub fn sys_capget(hdrp: *const CapUserHeader, datap: *mut CapUserData) -> Result<(), CapError> {
    if hdrp.is_null() {
        return Err(CapError::PermissionDenied);
    }

    let header = unsafe { &*hdrp };
    let pid = if header.pid == 0 {
        crate::process::current().map(|p| p.pid).ok_or(CapError::PermissionDenied)?
    } else {
        Pid(header.pid as u32)
    };

    let caps = get_caps(pid).ok_or(CapError::PermissionDenied)?;

    if !datap.is_null() {
        let data = unsafe { &mut *datap };
        data.effective = caps.effective.bits() as u32;
        data.permitted = caps.permitted.bits() as u32;
        data.inheritable = caps.inheritable.bits() as u32;
    }

    Ok(())
}

/// capset syscall implementation
pub fn sys_capset(hdrp: *const CapUserHeader, datap: *const CapUserData) -> Result<(), CapError> {
    if hdrp.is_null() || datap.is_null() {
        return Err(CapError::PermissionDenied);
    }

    let header = unsafe { &*hdrp };
    let data = unsafe { &*datap };

    let pid = if header.pid == 0 {
        crate::process::current().map(|p| p.pid).ok_or(CapError::PermissionDenied)?
    } else {
        Pid(header.pid as u32)
    };

    // Can only modify own capabilities without CAP_SETPCAP
    let current_pid = crate::process::current().map(|p| p.pid).ok_or(CapError::PermissionDenied)?;
    if pid != current_pid && !current_has(Cap::Setpcap) {
        return Err(CapError::PermissionDenied);
    }

    let mut all_caps = PROCESS_CAPS.write();
    let pcaps = all_caps.get_mut(&pid).ok_or(CapError::PermissionDenied)?;

    // New permitted must be subset of old permitted
    let new_permitted = CapSet::from_bits(data.permitted as u64);
    if !pcaps.permitted.intersect(&new_permitted).bits() == new_permitted.bits() {
        // Can only drop, not add
        if new_permitted.difference(&pcaps.permitted).bits() != 0 {
            return Err(CapError::PermissionDenied);
        }
    }

    // New effective must be subset of new permitted
    let new_effective = CapSet::from_bits(data.effective as u64);
    if new_effective.difference(&new_permitted).bits() != 0 {
        return Err(CapError::PermissionDenied);
    }

    // New inheritable: can add from bounding set with CAP_SETPCAP
    let new_inheritable = CapSet::from_bits(data.inheritable as u64);
    let added_inheritable = new_inheritable.difference(&pcaps.inheritable);
    if !added_inheritable.is_empty() {
        if !current_has(Cap::Setpcap) {
            return Err(CapError::PermissionDenied);
        }
        // Must be in bounding set
        if added_inheritable.difference(&pcaps.bounding).bits() != 0 {
            return Err(CapError::PermissionDenied);
        }
    }

    pcaps.permitted = new_permitted;
    pcaps.effective = new_effective;
    pcaps.inheritable = new_inheritable;

    Ok(())
}

/// Capability header for capget/capset
#[repr(C)]
pub struct CapUserHeader {
    pub version: u32,
    pub pid: i32,
}

/// Capability data for capget/capset
#[repr(C)]
pub struct CapUserData {
    pub effective: u32,
    pub permitted: u32,
    pub inheritable: u32,
}

/// prctl capability operations
pub mod prctl {
    use super::*;

    pub const PR_CAPBSET_READ: i32 = 23;
    pub const PR_CAPBSET_DROP: i32 = 24;
    pub const PR_CAP_AMBIENT: i32 = 47;
    pub const PR_CAP_AMBIENT_IS_SET: i32 = 1;
    pub const PR_CAP_AMBIENT_RAISE: i32 = 2;
    pub const PR_CAP_AMBIENT_LOWER: i32 = 3;
    pub const PR_CAP_AMBIENT_CLEAR_ALL: i32 = 4;

    /// Handle prctl capability operations
    pub fn handle(option: i32, arg2: usize, arg3: usize) -> Result<isize, CapError> {
        let pid = crate::process::current()
            .map(|p| p.pid)
            .ok_or(CapError::PermissionDenied)?;

        let mut all_caps = PROCESS_CAPS.write();
        let pcaps = all_caps.get_mut(&pid).ok_or(CapError::PermissionDenied)?;

        match option {
            PR_CAPBSET_READ => {
                let cap = Cap::from_u32(arg2 as u32).ok_or(CapError::InvalidCapability)?;
                Ok(if pcaps.bounding.has(cap) { 1 } else { 0 })
            }
            PR_CAPBSET_DROP => {
                let cap = Cap::from_u32(arg2 as u32).ok_or(CapError::InvalidCapability)?;
                if !pcaps.effective.has(Cap::Setpcap) {
                    return Err(CapError::PermissionDenied);
                }
                pcaps.drop_bounding(cap);
                // Also clear from ambient
                pcaps.clear_ambient(cap);
                Ok(0)
            }
            PR_CAP_AMBIENT => {
                let cap = Cap::from_u32(arg3 as u32).ok_or(CapError::InvalidCapability)?;
                match arg2 as i32 {
                    PR_CAP_AMBIENT_IS_SET => {
                        Ok(if pcaps.ambient.has(cap) { 1 } else { 0 })
                    }
                    PR_CAP_AMBIENT_RAISE => {
                        if pcaps.securebits.no_cap_ambient_raise {
                            return Err(CapError::PermissionDenied);
                        }
                        if pcaps.set_ambient(cap) {
                            Ok(0)
                        } else {
                            Err(CapError::PermissionDenied)
                        }
                    }
                    PR_CAP_AMBIENT_LOWER => {
                        pcaps.clear_ambient(cap);
                        Ok(0)
                    }
                    PR_CAP_AMBIENT_CLEAR_ALL => {
                        pcaps.ambient.clear();
                        Ok(0)
                    }
                    _ => Err(CapError::InvalidCapability),
                }
            }
            _ => Err(CapError::NotSupported),
        }
    }
}

/// Initialize capability subsystem
pub fn init() {
    crate::kprintln!("  Capability system initialized ({} capabilities)", Cap::COUNT);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capset_operations() {
        let mut set = CapSet::empty();
        assert!(set.is_empty());

        set.add(Cap::NetRaw);
        assert!(set.has(Cap::NetRaw));
        assert!(!set.has(Cap::SysAdmin));

        set.remove(Cap::NetRaw);
        assert!(!set.has(Cap::NetRaw));
    }

    #[test]
    fn test_root_caps() {
        let caps = ProcessCaps::root();
        assert!(caps.has(Cap::SysAdmin));
        assert!(caps.has(Cap::NetRaw));
        assert!(caps.has(Cap::Kill));
    }

    #[test]
    fn test_unprivileged_caps() {
        let caps = ProcessCaps::unprivileged();
        assert!(!caps.has(Cap::SysAdmin));
        assert!(!caps.has(Cap::NetRaw));
    }
}
