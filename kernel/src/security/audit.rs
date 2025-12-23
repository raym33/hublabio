//! Security Audit Logging
//!
//! Records security-relevant events for monitoring and forensics.
//! Provides structured logging with tamper-evident features.

use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::format;
use core::sync::atomic::{AtomicU64, Ordering};

/// Audit event types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditEventType {
    // Process events
    ProcessCreate,
    ProcessExit,
    ProcessExec,
    ProcessSignal,

    // Authentication
    LoginSuccess,
    LoginFailure,
    Logout,
    PrivilegeEscalation,
    CapabilityChange,

    // File operations
    FileOpen,
    FileCreate,
    FileDelete,
    FileModify,
    FilePermissionChange,
    FileOwnerChange,

    // Network
    NetworkConnect,
    NetworkBind,
    NetworkAccept,
    FirewallBlock,

    // Security
    SandboxViolation,
    SignatureVerify,
    SignatureFailure,
    PolicyViolation,
    AccessDenied,

    // System
    SystemBoot,
    SystemShutdown,
    ModuleLoad,
    ModuleUnload,
    ConfigChange,

    // AI specific
    AiModelLoad,
    AiModelUnload,
    AiInferenceStart,
    AiClusterJoin,
    AiClusterLeave,
}

impl AuditEventType {
    /// Get event type name
    pub fn name(&self) -> &'static str {
        match self {
            AuditEventType::ProcessCreate => "PROCESS_CREATE",
            AuditEventType::ProcessExit => "PROCESS_EXIT",
            AuditEventType::ProcessExec => "PROCESS_EXEC",
            AuditEventType::ProcessSignal => "PROCESS_SIGNAL",
            AuditEventType::LoginSuccess => "LOGIN_SUCCESS",
            AuditEventType::LoginFailure => "LOGIN_FAILURE",
            AuditEventType::Logout => "LOGOUT",
            AuditEventType::PrivilegeEscalation => "PRIV_ESCALATION",
            AuditEventType::CapabilityChange => "CAP_CHANGE",
            AuditEventType::FileOpen => "FILE_OPEN",
            AuditEventType::FileCreate => "FILE_CREATE",
            AuditEventType::FileDelete => "FILE_DELETE",
            AuditEventType::FileModify => "FILE_MODIFY",
            AuditEventType::FilePermissionChange => "FILE_PERM_CHANGE",
            AuditEventType::FileOwnerChange => "FILE_OWNER_CHANGE",
            AuditEventType::NetworkConnect => "NET_CONNECT",
            AuditEventType::NetworkBind => "NET_BIND",
            AuditEventType::NetworkAccept => "NET_ACCEPT",
            AuditEventType::FirewallBlock => "FW_BLOCK",
            AuditEventType::SandboxViolation => "SANDBOX_VIOLATION",
            AuditEventType::SignatureVerify => "SIG_VERIFY",
            AuditEventType::SignatureFailure => "SIG_FAILURE",
            AuditEventType::PolicyViolation => "POLICY_VIOLATION",
            AuditEventType::AccessDenied => "ACCESS_DENIED",
            AuditEventType::SystemBoot => "SYSTEM_BOOT",
            AuditEventType::SystemShutdown => "SYSTEM_SHUTDOWN",
            AuditEventType::ModuleLoad => "MODULE_LOAD",
            AuditEventType::ModuleUnload => "MODULE_UNLOAD",
            AuditEventType::ConfigChange => "CONFIG_CHANGE",
            AuditEventType::AiModelLoad => "AI_MODEL_LOAD",
            AuditEventType::AiModelUnload => "AI_MODEL_UNLOAD",
            AuditEventType::AiInferenceStart => "AI_INFERENCE",
            AuditEventType::AiClusterJoin => "AI_CLUSTER_JOIN",
            AuditEventType::AiClusterLeave => "AI_CLUSTER_LEAVE",
        }
    }

    /// Get severity level
    pub fn severity(&self) -> AuditSeverity {
        match self {
            AuditEventType::LoginFailure |
            AuditEventType::PrivilegeEscalation |
            AuditEventType::SandboxViolation |
            AuditEventType::SignatureFailure |
            AuditEventType::PolicyViolation |
            AuditEventType::AccessDenied => AuditSeverity::Warning,

            AuditEventType::SystemBoot |
            AuditEventType::SystemShutdown => AuditSeverity::Critical,

            _ => AuditSeverity::Info,
        }
    }
}

/// Severity levels
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditSeverity {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Critical = 4,
}

impl AuditSeverity {
    pub fn name(&self) -> &'static str {
        match self {
            AuditSeverity::Debug => "DEBUG",
            AuditSeverity::Info => "INFO",
            AuditSeverity::Warning => "WARN",
            AuditSeverity::Error => "ERROR",
            AuditSeverity::Critical => "CRIT",
        }
    }
}

/// Audit event
#[derive(Clone, Debug)]
pub struct AuditEvent {
    /// Sequence number
    pub seq: u64,
    /// Timestamp (microseconds since boot)
    pub timestamp: u64,
    /// Event type
    pub event_type: AuditEventType,
    /// Severity
    pub severity: AuditSeverity,
    /// Process ID
    pub pid: u64,
    /// User ID
    pub uid: u32,
    /// Success/failure
    pub success: bool,
    /// Subject (who did it)
    pub subject: String,
    /// Object (what was affected)
    pub object: String,
    /// Additional details
    pub details: String,
    /// Source IP (for network events)
    pub source_ip: Option<String>,
    /// Destination IP (for network events)
    pub dest_ip: Option<String>,
}

impl AuditEvent {
    /// Create new event
    pub fn new(event_type: AuditEventType, pid: u64, uid: u32) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);

        Self {
            seq: SEQ.fetch_add(1, Ordering::SeqCst),
            timestamp: 0, // TODO: get actual timestamp
            event_type,
            severity: event_type.severity(),
            pid,
            uid,
            success: true,
            subject: String::new(),
            object: String::new(),
            details: String::new(),
            source_ip: None,
            dest_ip: None,
        }
    }

    /// Set success/failure
    pub fn success(mut self, success: bool) -> Self {
        self.success = success;
        if !success && self.severity < AuditSeverity::Warning {
            self.severity = AuditSeverity::Warning;
        }
        self
    }

    /// Set subject
    pub fn subject(mut self, subject: &str) -> Self {
        self.subject = subject.to_string();
        self
    }

    /// Set object
    pub fn object(mut self, object: &str) -> Self {
        self.object = object.to_string();
        self
    }

    /// Set details
    pub fn details(mut self, details: &str) -> Self {
        self.details = details.to_string();
        self
    }

    /// Set network info
    pub fn network(mut self, source: &str, dest: &str) -> Self {
        self.source_ip = Some(source.to_string());
        self.dest_ip = Some(dest.to_string());
        self
    }

    /// Set severity override
    pub fn severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Format as log line
    pub fn to_log_line(&self) -> String {
        let status = if self.success { "OK" } else { "FAIL" };

        format!(
            "[{}] seq={} type={} pid={} uid={} status={} subj=\"{}\" obj=\"{}\" {}",
            self.severity.name(),
            self.seq,
            self.event_type.name(),
            self.pid,
            self.uid,
            status,
            self.subject,
            self.object,
            self.details,
        )
    }

    /// Format as JSON (for structured logging)
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"seq":{},"ts":{},"type":"{}","sev":"{}","pid":{},"uid":{},"success":{},"subject":"{}","object":"{}","details":"{}"}}"#,
            self.seq,
            self.timestamp,
            self.event_type.name(),
            self.severity.name(),
            self.pid,
            self.uid,
            self.success,
            self.subject,
            self.object,
            self.details,
        )
    }
}

/// Audit configuration
#[derive(Clone, Debug)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Minimum severity to log
    pub min_severity: AuditSeverity,
    /// Maximum events in memory buffer
    pub buffer_size: usize,
    /// Log to console
    pub log_console: bool,
    /// Log to file
    pub log_file: bool,
    /// Log file path
    pub log_path: String,
    /// Use JSON format
    pub json_format: bool,
    /// Event types to log (empty = all)
    pub event_filter: Vec<AuditEventType>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_severity: AuditSeverity::Info,
            buffer_size: 10000,
            log_console: true,
            log_file: true,
            log_path: String::from("/var/log/audit.log"),
            json_format: false,
            event_filter: Vec::new(), // Log all
        }
    }
}

/// Audit logger
pub struct AuditLogger {
    /// Configuration
    config: AuditConfig,
    /// Event buffer
    buffer: VecDeque<AuditEvent>,
    /// Event count by type
    event_counts: [u64; 64],
    /// Dropped events (buffer overflow)
    dropped: u64,
}

impl AuditLogger {
    /// Create new logger
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            event_counts: [0; 64],
            dropped: 0,
        }
    }

    /// Log an event
    pub fn log(&mut self, event: AuditEvent) {
        if !self.config.enabled {
            return;
        }

        // Check severity filter
        if event.severity < self.config.min_severity {
            return;
        }

        // Check event type filter
        if !self.config.event_filter.is_empty() &&
           !self.config.event_filter.contains(&event.event_type) {
            return;
        }

        // Update counts
        let type_idx = (event.event_type as usize) % 64;
        self.event_counts[type_idx] += 1;

        // Output to console
        if self.config.log_console {
            let line = if self.config.json_format {
                event.to_json()
            } else {
                event.to_log_line()
            };
            log::info!("AUDIT: {}", line);
        }

        // Add to buffer
        if self.buffer.len() >= self.config.buffer_size {
            self.buffer.pop_front();
            self.dropped += 1;
        }
        self.buffer.push_back(event);

        // TODO: Write to file
    }

    /// Log process creation
    pub fn log_process_create(&mut self, pid: u64, uid: u32, parent_pid: u64, command: &str) {
        let event = AuditEvent::new(AuditEventType::ProcessCreate, pid, uid)
            .subject(&format!("pid={}", parent_pid))
            .object(command)
            .details(&format!("ppid={}", parent_pid));
        self.log(event);
    }

    /// Log process exec
    pub fn log_exec(&mut self, pid: u64, uid: u32, path: &str, success: bool) {
        let event = AuditEvent::new(AuditEventType::ProcessExec, pid, uid)
            .object(path)
            .success(success);
        self.log(event);
    }

    /// Log file access
    pub fn log_file_open(&mut self, pid: u64, uid: u32, path: &str, flags: u32, success: bool) {
        let event = AuditEvent::new(AuditEventType::FileOpen, pid, uid)
            .object(path)
            .details(&format!("flags={:#x}", flags))
            .success(success);
        self.log(event);
    }

    /// Log network connection
    pub fn log_network_connect(&mut self, pid: u64, uid: u32, addr: &str, port: u16, success: bool) {
        let event = AuditEvent::new(AuditEventType::NetworkConnect, pid, uid)
            .object(&format!("{}:{}", addr, port))
            .success(success);
        self.log(event);
    }

    /// Log sandbox violation
    pub fn log_sandbox_violation(&mut self, pid: u64, uid: u32, violation: &str) {
        let event = AuditEvent::new(AuditEventType::SandboxViolation, pid, uid)
            .details(violation)
            .success(false)
            .severity(AuditSeverity::Warning);
        self.log(event);
    }

    /// Log access denied
    pub fn log_access_denied(&mut self, pid: u64, uid: u32, resource: &str, reason: &str) {
        let event = AuditEvent::new(AuditEventType::AccessDenied, pid, uid)
            .object(resource)
            .details(reason)
            .success(false);
        self.log(event);
    }

    /// Log signature verification
    pub fn log_signature(&mut self, pid: u64, uid: u32, path: &str, key_id: &str, valid: bool) {
        let event_type = if valid {
            AuditEventType::SignatureVerify
        } else {
            AuditEventType::SignatureFailure
        };

        let event = AuditEvent::new(event_type, pid, uid)
            .object(path)
            .details(&format!("key={}", key_id))
            .success(valid);
        self.log(event);
    }

    /// Log AI model load
    pub fn log_ai_model_load(&mut self, pid: u64, uid: u32, model_path: &str, success: bool) {
        let event = AuditEvent::new(AuditEventType::AiModelLoad, pid, uid)
            .object(model_path)
            .success(success);
        self.log(event);
    }

    /// Log login attempt
    pub fn log_login(&mut self, uid: u32, username: &str, source: &str, success: bool) {
        let event_type = if success {
            AuditEventType::LoginSuccess
        } else {
            AuditEventType::LoginFailure
        };

        let event = AuditEvent::new(event_type, 0, uid)
            .subject(username)
            .details(&format!("from={}", source))
            .success(success);
        self.log(event);
    }

    /// Get recent events
    pub fn recent_events(&self, count: usize) -> Vec<&AuditEvent> {
        self.buffer.iter().rev().take(count).collect()
    }

    /// Search events
    pub fn search(&self, event_type: Option<AuditEventType>, pid: Option<u64>, uid: Option<u32>) -> Vec<&AuditEvent> {
        self.buffer.iter()
            .filter(|e| {
                event_type.map_or(true, |t| e.event_type == t) &&
                pid.map_or(true, |p| e.pid == p) &&
                uid.map_or(true, |u| e.uid == u)
            })
            .collect()
    }

    /// Get event count
    pub fn event_count(&self) -> usize {
        self.buffer.len()
    }

    /// Get dropped count
    pub fn dropped_count(&self) -> u64 {
        self.dropped
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> AuditStats {
        AuditStats {
            total_events: self.buffer.len() as u64,
            dropped_events: self.dropped,
            events_by_severity: [
                self.buffer.iter().filter(|e| e.severity == AuditSeverity::Debug).count() as u64,
                self.buffer.iter().filter(|e| e.severity == AuditSeverity::Info).count() as u64,
                self.buffer.iter().filter(|e| e.severity == AuditSeverity::Warning).count() as u64,
                self.buffer.iter().filter(|e| e.severity == AuditSeverity::Error).count() as u64,
                self.buffer.iter().filter(|e| e.severity == AuditSeverity::Critical).count() as u64,
            ],
            failed_events: self.buffer.iter().filter(|e| !e.success).count() as u64,
        }
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new(AuditConfig::default())
    }
}

/// Audit statistics
#[derive(Clone, Debug, Default)]
pub struct AuditStats {
    pub total_events: u64,
    pub dropped_events: u64,
    pub events_by_severity: [u64; 5],
    pub failed_events: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event() {
        let event = AuditEvent::new(AuditEventType::ProcessCreate, 100, 1000)
            .subject("init")
            .object("/bin/bash")
            .success(true);

        assert_eq!(event.pid, 100);
        assert_eq!(event.uid, 1000);
        assert!(event.success);
    }

    #[test]
    fn test_audit_logger() {
        let mut logger = AuditLogger::default();

        logger.log_process_create(1, 0, 0, "/sbin/init");
        logger.log_exec(100, 1000, "/bin/bash", true);
        logger.log_file_open(100, 1000, "/etc/passwd", 0, true);

        assert_eq!(logger.event_count(), 3);
    }

    #[test]
    fn test_audit_search() {
        let mut logger = AuditLogger::default();

        logger.log_process_create(1, 0, 0, "/sbin/init");
        logger.log_exec(100, 1000, "/bin/bash", true);
        logger.log_exec(101, 1000, "/bin/ls", true);

        let results = logger.search(Some(AuditEventType::ProcessExec), None, None);
        assert_eq!(results.len(), 2);

        let results = logger.search(None, Some(100), None);
        assert_eq!(results.len(), 1);
    }
}
