//! Security Subsystem
//!
//! Provides process sandboxing, capability-based security, and
//! code signing verification for HubLab IO.

pub mod sandbox;
pub mod capabilities;
pub mod signing;
pub mod audit;

pub use sandbox::*;
pub use capabilities::*;
pub use signing::*;
pub use audit::*;

use alloc::string::String;
use alloc::vec::Vec;

/// Security policy for the system
#[derive(Clone, Debug)]
pub struct SecurityPolicy {
    /// Enable sandbox by default for new processes
    pub sandbox_by_default: bool,
    /// Require signed binaries
    pub require_signed: bool,
    /// Enable audit logging
    pub audit_enabled: bool,
    /// Maximum privilege level for unprivileged processes
    pub max_unprivileged_caps: CapabilitySet,
    /// Trusted signing keys
    pub trusted_keys: Vec<PublicKey>,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            sandbox_by_default: true,
            require_signed: false, // Disabled for development
            audit_enabled: true,
            max_unprivileged_caps: CapabilitySet::user_default(),
            trusted_keys: Vec::new(),
        }
    }
}

/// Security context for a process
#[derive(Clone, Debug)]
pub struct SecurityContext {
    /// Process ID
    pub pid: u64,
    /// User ID
    pub uid: u32,
    /// Group ID
    pub gid: u32,
    /// Effective capabilities
    pub capabilities: CapabilitySet,
    /// Sandbox configuration
    pub sandbox: Option<SandboxConfig>,
    /// Security label (for MAC)
    pub label: Option<String>,
    /// Is privileged (root)
    pub privileged: bool,
}

impl SecurityContext {
    /// Create context for root process
    pub fn root() -> Self {
        Self {
            pid: 0,
            uid: 0,
            gid: 0,
            capabilities: CapabilitySet::all(),
            sandbox: None,
            label: Some(String::from("system_u:system_r:kernel_t")),
            privileged: true,
        }
    }

    /// Create context for regular user process
    pub fn user(pid: u64, uid: u32, gid: u32) -> Self {
        Self {
            pid,
            uid,
            gid,
            capabilities: CapabilitySet::user_default(),
            sandbox: Some(SandboxConfig::default()),
            label: Some(String::from("user_u:user_r:user_t")),
            privileged: false,
        }
    }

    /// Check if context has capability
    pub fn has_capability(&self, cap: Capability) -> bool {
        self.privileged || self.capabilities.has(cap)
    }

    /// Drop a capability
    pub fn drop_capability(&mut self, cap: Capability) {
        self.capabilities.remove(cap);
    }

    /// Check if can access file with given permissions
    pub fn can_access(&self, file_uid: u32, file_gid: u32, file_mode: u16, access: AccessMode) -> bool {
        if self.privileged {
            return true;
        }

        let mode_bits = match access {
            AccessMode::Read => 0o4,
            AccessMode::Write => 0o2,
            AccessMode::Execute => 0o1,
        };

        // Owner check
        if self.uid == file_uid {
            return (file_mode >> 6) & mode_bits != 0;
        }

        // Group check
        if self.gid == file_gid {
            return (file_mode >> 3) & mode_bits != 0;
        }

        // Other check
        file_mode & mode_bits != 0
    }
}

/// Access mode for permission checks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
    Execute,
}

/// Security error types
#[derive(Clone, Debug)]
pub enum SecurityError {
    /// Permission denied
    PermissionDenied,
    /// Missing capability
    MissingCapability(Capability),
    /// Sandbox violation
    SandboxViolation(String),
    /// Invalid signature
    InvalidSignature,
    /// Key not trusted
    UntrustedKey,
    /// Policy violation
    PolicyViolation(String),
    /// Audit failure
    AuditError(String),
}
