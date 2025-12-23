//! Process Sandboxing
//!
//! Provides seccomp-like syscall filtering and namespace isolation
//! for running untrusted code safely.

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;

/// Sandbox configuration
#[derive(Clone, Debug)]
pub struct SandboxConfig {
    /// Syscall filter
    pub syscall_filter: SyscallFilter,
    /// Filesystem restrictions
    pub fs_rules: Vec<FsRule>,
    /// Network restrictions
    pub network: NetworkPolicy,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Namespace configuration
    pub namespaces: NamespaceConfig,
    /// Allow AI operations
    pub allow_ai: bool,
    /// Allowed IPC endpoints
    pub allowed_ipc: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            syscall_filter: SyscallFilter::default_allow(),
            fs_rules: vec![
                FsRule::read_only("/"),
                FsRule::read_write("/tmp"),
                FsRule::read_write("/home"),
                FsRule::deny("/etc/shadow"),
                FsRule::deny("/etc/passwd"),
            ],
            network: NetworkPolicy::AllowOutbound,
            limits: ResourceLimits::default(),
            namespaces: NamespaceConfig::default(),
            allow_ai: true,
            allowed_ipc: vec![
                String::from("io.hublab.ai"),
                String::from("io.hublab.display"),
            ],
        }
    }
}

impl SandboxConfig {
    /// Create minimal sandbox for untrusted code
    pub fn minimal() -> Self {
        Self {
            syscall_filter: SyscallFilter::minimal(),
            fs_rules: vec![
                FsRule::deny("/"),
                FsRule::read_only("/lib"),
                FsRule::read_write("/tmp/sandbox"),
            ],
            network: NetworkPolicy::Deny,
            limits: ResourceLimits::strict(),
            namespaces: NamespaceConfig::full_isolation(),
            allow_ai: false,
            allowed_ipc: Vec::new(),
        }
    }

    /// Create sandbox for AI inference
    pub fn ai_inference() -> Self {
        Self {
            syscall_filter: SyscallFilter::default_allow(),
            fs_rules: vec![
                FsRule::read_only("/"),
                FsRule::read_only("/models"),
                FsRule::read_write("/tmp"),
            ],
            network: NetworkPolicy::AllowOutbound,
            limits: ResourceLimits {
                max_memory: 2 * 1024 * 1024 * 1024, // 2GB
                max_cpu_percent: 100,
                max_threads: 16,
                max_files: 64,
                max_file_size: 100 * 1024 * 1024, // 100MB
            },
            namespaces: NamespaceConfig::default(),
            allow_ai: true,
            allowed_ipc: vec![
                String::from("io.hublab.ai"),
            ],
        }
    }

    /// Create sandbox for web content
    pub fn web_content() -> Self {
        Self {
            syscall_filter: SyscallFilter::minimal(),
            fs_rules: vec![
                FsRule::deny("/"),
                FsRule::read_only("/usr/share/fonts"),
                FsRule::read_write("/tmp/browser"),
            ],
            network: NetworkPolicy::AllowOutbound,
            limits: ResourceLimits {
                max_memory: 512 * 1024 * 1024, // 512MB
                max_cpu_percent: 50,
                max_threads: 8,
                max_files: 32,
                max_file_size: 10 * 1024 * 1024, // 10MB
            },
            namespaces: NamespaceConfig::full_isolation(),
            allow_ai: false,
            allowed_ipc: vec![
                String::from("io.hublab.display"),
            ],
        }
    }
}

/// Syscall filtering policy
#[derive(Clone, Debug)]
pub struct SyscallFilter {
    /// Default action
    pub default_action: FilterAction,
    /// Rules for specific syscalls
    pub rules: Vec<SyscallRule>,
}

impl SyscallFilter {
    /// Create filter that allows most syscalls
    pub fn default_allow() -> Self {
        Self {
            default_action: FilterAction::Allow,
            rules: vec![
                // Block dangerous syscalls
                SyscallRule::deny("reboot"),
                SyscallRule::deny("kexec_load"),
                SyscallRule::deny("init_module"),
                SyscallRule::deny("delete_module"),
                SyscallRule::deny("acct"),
                SyscallRule::deny("swapon"),
                SyscallRule::deny("swapoff"),
                SyscallRule::deny("mount"),
                SyscallRule::deny("umount"),
                SyscallRule::deny("pivot_root"),
                SyscallRule::deny("ptrace"),
            ],
        }
    }

    /// Create minimal filter for untrusted code
    pub fn minimal() -> Self {
        Self {
            default_action: FilterAction::Deny,
            rules: vec![
                // Only allow basic operations
                SyscallRule::allow("read"),
                SyscallRule::allow("write"),
                SyscallRule::allow("open"),
                SyscallRule::allow("close"),
                SyscallRule::allow("stat"),
                SyscallRule::allow("fstat"),
                SyscallRule::allow("lseek"),
                SyscallRule::allow("mmap"),
                SyscallRule::allow("mprotect"),
                SyscallRule::allow("munmap"),
                SyscallRule::allow("brk"),
                SyscallRule::allow("exit"),
                SyscallRule::allow("exit_group"),
                SyscallRule::allow("clock_gettime"),
                SyscallRule::allow("gettimeofday"),
                SyscallRule::allow("getpid"),
                SyscallRule::allow("getuid"),
                SyscallRule::allow("getgid"),
            ],
        }
    }

    /// Check if syscall is allowed
    pub fn check(&self, syscall: &str) -> FilterAction {
        for rule in &self.rules {
            if rule.syscall == syscall {
                return rule.action;
            }
        }
        self.default_action
    }
}

/// Syscall filter rule
#[derive(Clone, Debug)]
pub struct SyscallRule {
    /// Syscall name
    pub syscall: String,
    /// Action to take
    pub action: FilterAction,
    /// Optional argument checks
    pub arg_checks: Vec<ArgCheck>,
}

impl SyscallRule {
    pub fn allow(syscall: &str) -> Self {
        Self {
            syscall: syscall.to_string(),
            action: FilterAction::Allow,
            arg_checks: Vec::new(),
        }
    }

    pub fn deny(syscall: &str) -> Self {
        Self {
            syscall: syscall.to_string(),
            action: FilterAction::Deny,
            arg_checks: Vec::new(),
        }
    }

    pub fn log(syscall: &str) -> Self {
        Self {
            syscall: syscall.to_string(),
            action: FilterAction::Log,
            arg_checks: Vec::new(),
        }
    }
}

/// Action for syscall filter
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterAction {
    /// Allow syscall
    Allow,
    /// Deny syscall (return EPERM)
    Deny,
    /// Kill process
    Kill,
    /// Log and allow
    Log,
    /// Trap to handler
    Trap,
}

/// Argument check for syscall filtering
#[derive(Clone, Debug)]
pub struct ArgCheck {
    /// Argument index (0-5)
    pub arg_index: usize,
    /// Comparison operation
    pub op: ArgOp,
    /// Value to compare
    pub value: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum ArgOp {
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    MaskedEqual(u64),
}

/// Filesystem access rule
#[derive(Clone, Debug)]
pub struct FsRule {
    /// Path pattern
    pub path: String,
    /// Access mode
    pub access: FsAccess,
}

impl FsRule {
    pub fn read_only(path: &str) -> Self {
        Self {
            path: path.to_string(),
            access: FsAccess::ReadOnly,
        }
    }

    pub fn read_write(path: &str) -> Self {
        Self {
            path: path.to_string(),
            access: FsAccess::ReadWrite,
        }
    }

    pub fn deny(path: &str) -> Self {
        Self {
            path: path.to_string(),
            access: FsAccess::Deny,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsAccess {
    Deny,
    ReadOnly,
    ReadWrite,
    Execute,
}

/// Network policy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkPolicy {
    /// No network access
    Deny,
    /// Allow outbound connections only
    AllowOutbound,
    /// Allow all network access
    AllowAll,
    /// Custom rules (TODO: implement)
    Custom,
}

/// Resource limits
#[derive(Clone, Debug)]
pub struct ResourceLimits {
    /// Maximum memory (bytes)
    pub max_memory: usize,
    /// Maximum CPU usage (percent)
    pub max_cpu_percent: u8,
    /// Maximum threads
    pub max_threads: usize,
    /// Maximum open files
    pub max_files: usize,
    /// Maximum file size (bytes)
    pub max_file_size: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            max_cpu_percent: 100,
            max_threads: 32,
            max_files: 256,
            max_file_size: 1024 * 1024 * 1024, // 1GB
        }
    }
}

impl ResourceLimits {
    pub fn strict() -> Self {
        Self {
            max_memory: 128 * 1024 * 1024, // 128MB
            max_cpu_percent: 25,
            max_threads: 4,
            max_files: 16,
            max_file_size: 10 * 1024 * 1024, // 10MB
        }
    }

    pub fn unlimited() -> Self {
        Self {
            max_memory: usize::MAX,
            max_cpu_percent: 100,
            max_threads: usize::MAX,
            max_files: usize::MAX,
            max_file_size: usize::MAX,
        }
    }
}

/// Namespace configuration
#[derive(Clone, Debug)]
pub struct NamespaceConfig {
    /// Use separate PID namespace
    pub pid_ns: bool,
    /// Use separate network namespace
    pub net_ns: bool,
    /// Use separate mount namespace
    pub mnt_ns: bool,
    /// Use separate user namespace
    pub user_ns: bool,
    /// Use separate UTS namespace
    pub uts_ns: bool,
    /// Use separate IPC namespace
    pub ipc_ns: bool,
    /// Use separate cgroup namespace
    pub cgroup_ns: bool,
}

impl Default for NamespaceConfig {
    fn default() -> Self {
        Self {
            pid_ns: false,
            net_ns: false,
            mnt_ns: true,  // Default to mount namespace
            user_ns: false,
            uts_ns: false,
            ipc_ns: false,
            cgroup_ns: false,
        }
    }
}

impl NamespaceConfig {
    pub fn none() -> Self {
        Self {
            pid_ns: false,
            net_ns: false,
            mnt_ns: false,
            user_ns: false,
            uts_ns: false,
            ipc_ns: false,
            cgroup_ns: false,
        }
    }

    pub fn full_isolation() -> Self {
        Self {
            pid_ns: true,
            net_ns: true,
            mnt_ns: true,
            user_ns: true,
            uts_ns: true,
            ipc_ns: true,
            cgroup_ns: true,
        }
    }
}

/// Sandbox instance for a process
pub struct Sandbox {
    /// Configuration
    config: SandboxConfig,
    /// Process ID
    pid: u64,
    /// Current resource usage
    resource_usage: ResourceUsage,
    /// Violation count
    violation_count: usize,
    /// Is active
    active: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ResourceUsage {
    pub memory: usize,
    pub cpu_time_us: u64,
    pub threads: usize,
    pub open_files: usize,
}

impl Sandbox {
    /// Create new sandbox
    pub fn new(pid: u64, config: SandboxConfig) -> Self {
        Self {
            config,
            pid,
            resource_usage: ResourceUsage::default(),
            violation_count: 0,
            active: true,
        }
    }

    /// Check if syscall is allowed
    pub fn check_syscall(&mut self, syscall: &str) -> Result<(), SandboxViolation> {
        if !self.active {
            return Ok(());
        }

        match self.config.syscall_filter.check(syscall) {
            FilterAction::Allow => Ok(()),
            FilterAction::Log => {
                log::info!("Sandbox[{}]: syscall {} (logged)", self.pid, syscall);
                Ok(())
            }
            FilterAction::Deny => {
                self.violation_count += 1;
                Err(SandboxViolation::SyscallDenied(syscall.to_string()))
            }
            FilterAction::Kill => {
                self.violation_count += 1;
                Err(SandboxViolation::FatalViolation(syscall.to_string()))
            }
            FilterAction::Trap => {
                Err(SandboxViolation::Trap(syscall.to_string()))
            }
        }
    }

    /// Check if filesystem access is allowed
    pub fn check_fs_access(&mut self, path: &str, write: bool) -> Result<(), SandboxViolation> {
        if !self.active {
            return Ok(());
        }

        // Find most specific matching rule
        let mut best_match: Option<&FsRule> = None;
        let mut best_len = 0;

        for rule in &self.config.fs_rules {
            if path.starts_with(&rule.path) && rule.path.len() > best_len {
                best_match = Some(rule);
                best_len = rule.path.len();
            }
        }

        match best_match {
            Some(rule) => match rule.access {
                FsAccess::Deny => {
                    self.violation_count += 1;
                    Err(SandboxViolation::FsAccessDenied(path.to_string()))
                }
                FsAccess::ReadOnly if write => {
                    self.violation_count += 1;
                    Err(SandboxViolation::FsWriteDenied(path.to_string()))
                }
                _ => Ok(()),
            },
            None => Ok(()), // No rule = allow
        }
    }

    /// Check if network access is allowed
    pub fn check_network(&mut self, outbound: bool) -> Result<(), SandboxViolation> {
        if !self.active {
            return Ok(());
        }

        match self.config.network {
            NetworkPolicy::AllowAll => Ok(()),
            NetworkPolicy::AllowOutbound if outbound => Ok(()),
            NetworkPolicy::AllowOutbound => {
                self.violation_count += 1;
                Err(SandboxViolation::NetworkDenied)
            }
            NetworkPolicy::Deny | NetworkPolicy::Custom => {
                self.violation_count += 1;
                Err(SandboxViolation::NetworkDenied)
            }
        }
    }

    /// Check if IPC endpoint is allowed
    pub fn check_ipc(&mut self, endpoint: &str) -> Result<(), SandboxViolation> {
        if !self.active {
            return Ok(());
        }

        if self.config.allowed_ipc.iter().any(|e| e == endpoint) {
            Ok(())
        } else {
            self.violation_count += 1;
            Err(SandboxViolation::IpcDenied(endpoint.to_string()))
        }
    }

    /// Update resource usage
    pub fn update_resources(&mut self, usage: ResourceUsage) -> Result<(), SandboxViolation> {
        self.resource_usage = usage;

        if self.resource_usage.memory > self.config.limits.max_memory {
            return Err(SandboxViolation::MemoryLimit);
        }

        if self.resource_usage.threads > self.config.limits.max_threads {
            return Err(SandboxViolation::ThreadLimit);
        }

        if self.resource_usage.open_files > self.config.limits.max_files {
            return Err(SandboxViolation::FileLimit);
        }

        Ok(())
    }

    /// Get violation count
    pub fn violations(&self) -> usize {
        self.violation_count
    }

    /// Disable sandbox (for debugging)
    pub fn disable(&mut self) {
        self.active = false;
    }

    /// Enable sandbox
    pub fn enable(&mut self) {
        self.active = true;
    }

    /// Check if active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Sandbox violation types
#[derive(Clone, Debug)]
pub enum SandboxViolation {
    SyscallDenied(String),
    FsAccessDenied(String),
    FsWriteDenied(String),
    NetworkDenied,
    IpcDenied(String),
    MemoryLimit,
    CpuLimit,
    ThreadLimit,
    FileLimit,
    FatalViolation(String),
    Trap(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_filter() {
        let filter = SyscallFilter::default_allow();
        assert_eq!(filter.check("read"), FilterAction::Allow);
        assert_eq!(filter.check("reboot"), FilterAction::Deny);
    }

    #[test]
    fn test_sandbox_fs_access() {
        let config = SandboxConfig::default();
        let mut sandbox = Sandbox::new(1, config);

        assert!(sandbox.check_fs_access("/tmp/test", true).is_ok());
        assert!(sandbox.check_fs_access("/etc/config", false).is_ok());
        assert!(sandbox.check_fs_access("/etc/shadow", false).is_err());
    }

    #[test]
    fn test_resource_limits() {
        let config = SandboxConfig::minimal();
        let mut sandbox = Sandbox::new(1, config);

        let usage = ResourceUsage {
            memory: 256 * 1024 * 1024, // 256MB > 128MB limit
            ..Default::default()
        };

        assert!(sandbox.update_resources(usage).is_err());
    }
}
