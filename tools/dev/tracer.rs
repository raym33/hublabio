//! System Call Tracer
//!
//! strace-like functionality for HubLab IO.
//! Traces system calls, signals, and IPC messages.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;

/// System call entry
#[derive(Clone, Debug)]
pub struct SyscallEntry {
    /// Syscall number
    pub number: u32,
    /// Syscall name
    pub name: String,
    /// Arguments
    pub args: Vec<SyscallArg>,
    /// Timestamp (entry)
    pub entry_time: u64,
    /// Timestamp (exit)
    pub exit_time: Option<u64>,
    /// Return value
    pub result: Option<i64>,
    /// Error name (if failed)
    pub error: Option<String>,
    /// Process ID
    pub pid: u32,
    /// Thread ID
    pub tid: u32,
}

/// Syscall argument
#[derive(Clone, Debug)]
pub enum SyscallArg {
    /// Integer value
    Int(i64),
    /// Unsigned integer
    UInt(u64),
    /// Pointer address
    Ptr(u64),
    /// String value
    String(String),
    /// Buffer with size
    Buffer { addr: u64, size: usize, data: Option<Vec<u8>> },
    /// File descriptor
    Fd(i32),
    /// Flags (with symbolic names)
    Flags { value: u64, names: Vec<String> },
    /// Struct (name and fields)
    Struct { name: String, fields: Vec<(String, String)> },
}

impl SyscallArg {
    /// Format argument for display
    pub fn format(&self) -> String {
        match self {
            SyscallArg::Int(v) => format!("{}", v),
            SyscallArg::UInt(v) => format!("{}", v),
            SyscallArg::Ptr(v) => {
                if *v == 0 { String::from("NULL") } else { format!("{:#x}", v) }
            }
            SyscallArg::String(s) => format!("\"{}\"", s.escape_default()),
            SyscallArg::Buffer { addr, size, data } => {
                if let Some(d) = data {
                    if d.len() <= 32 {
                        format!("{:?}", d)
                    } else {
                        format!("{:#x}[{}]", addr, size)
                    }
                } else {
                    format!("{:#x}[{}]", addr, size)
                }
            }
            SyscallArg::Fd(fd) => format!("{}", fd),
            SyscallArg::Flags { value, names } => {
                if names.is_empty() {
                    format!("{:#x}", value)
                } else {
                    names.join("|")
                }
            }
            SyscallArg::Struct { name, fields } => {
                let field_str: Vec<String> = fields.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect();
                format!("{{{}={{{}}}}}", name, field_str.join(", "))
            }
        }
    }
}

/// Signal event
#[derive(Clone, Debug)]
pub struct SignalEvent {
    /// Signal number
    pub signum: u32,
    /// Signal name
    pub name: String,
    /// Sender PID (if known)
    pub sender_pid: Option<u32>,
    /// Timestamp
    pub timestamp: u64,
    /// Action taken
    pub action: SignalAction,
}

/// Signal action
#[derive(Clone, Debug)]
pub enum SignalAction {
    /// Default action
    Default,
    /// Ignored
    Ignored,
    /// Caught by handler
    Caught(u64),
    /// Blocked
    Blocked,
}

/// IPC event
#[derive(Clone, Debug)]
pub struct IpcEvent {
    /// Event type
    pub event_type: IpcEventType,
    /// Channel/endpoint ID
    pub channel_id: u64,
    /// Message size
    pub size: usize,
    /// Timestamp
    pub timestamp: u64,
    /// Source PID
    pub src_pid: u32,
    /// Destination PID
    pub dst_pid: Option<u32>,
}

/// IPC event type
#[derive(Clone, Debug)]
pub enum IpcEventType {
    /// Message sent
    Send,
    /// Message received
    Receive,
    /// Channel created
    Create,
    /// Channel closed
    Close,
    /// Endpoint registered
    Register,
    /// Endpoint connected
    Connect,
}

/// Tracer configuration
#[derive(Clone, Debug)]
pub struct TracerConfig {
    /// Trace syscalls
    pub trace_syscalls: bool,
    /// Trace signals
    pub trace_signals: bool,
    /// Trace IPC
    pub trace_ipc: bool,
    /// Follow forks
    pub follow_forks: bool,
    /// Maximum string length to capture
    pub max_string_len: usize,
    /// Syscalls to trace (empty = all)
    pub syscall_filter: Vec<String>,
    /// Output timestamps
    pub show_timestamps: bool,
}

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            trace_syscalls: true,
            trace_signals: true,
            trace_ipc: false,
            follow_forks: true,
            max_string_len: 256,
            syscall_filter: Vec::new(),
            show_timestamps: true,
        }
    }
}

/// Syscall tracer
pub struct SyscallTracer {
    /// Configuration
    config: TracerConfig,
    /// Target process ID
    target_pid: Option<u32>,
    /// Traced processes (for follow_forks)
    traced_pids: Vec<u32>,
    /// Syscall log
    syscalls: Vec<SyscallEntry>,
    /// Signal log
    signals: Vec<SignalEvent>,
    /// IPC log
    ipc_events: Vec<IpcEvent>,
    /// Pending syscalls (pid -> entry)
    pending: BTreeMap<u32, SyscallEntry>,
    /// Syscall statistics
    stats: BTreeMap<String, SyscallStats>,
    /// Running
    running: bool,
}

/// Syscall statistics
#[derive(Clone, Debug, Default)]
pub struct SyscallStats {
    /// Call count
    pub count: u64,
    /// Total time (microseconds)
    pub total_time: u64,
    /// Error count
    pub errors: u64,
}

impl SyscallTracer {
    /// Create new tracer
    pub fn new(config: TracerConfig) -> Self {
        Self {
            config,
            target_pid: None,
            traced_pids: Vec::new(),
            syscalls: Vec::new(),
            signals: Vec::new(),
            ipc_events: Vec::new(),
            pending: BTreeMap::new(),
            stats: BTreeMap::new(),
            running: false,
        }
    }

    /// Start tracing a process
    pub fn trace(&mut self, pid: u32) -> Result<(), TracerError> {
        self.target_pid = Some(pid);
        self.traced_pids.push(pid);
        self.running = true;
        self.syscalls.clear();
        self.signals.clear();
        self.ipc_events.clear();
        self.pending.clear();
        self.stats.clear();

        // TODO: Actually attach to process using ptrace-like mechanism
        Ok(())
    }

    /// Stop tracing
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Check if should trace this syscall
    fn should_trace_syscall(&self, name: &str) -> bool {
        if self.config.syscall_filter.is_empty() {
            return true;
        }
        self.config.syscall_filter.iter().any(|f| f == name)
    }

    /// Record syscall entry
    pub fn on_syscall_entry(&mut self, pid: u32, tid: u32, number: u32, args: &[u64], timestamp: u64) {
        if !self.running || !self.traced_pids.contains(&pid) {
            return;
        }

        let name = syscall_name(number);
        if !self.should_trace_syscall(&name) {
            return;
        }

        let parsed_args = self.parse_args(number, args);

        let entry = SyscallEntry {
            number,
            name: name.clone(),
            args: parsed_args,
            entry_time: timestamp,
            exit_time: None,
            result: None,
            error: None,
            pid,
            tid,
        };

        self.pending.insert(tid, entry);
    }

    /// Record syscall exit
    pub fn on_syscall_exit(&mut self, pid: u32, tid: u32, result: i64, timestamp: u64) {
        if let Some(mut entry) = self.pending.remove(&tid) {
            entry.exit_time = Some(timestamp);
            entry.result = Some(result);

            if result < 0 {
                entry.error = Some(errno_name(-result as u32));
            }

            // Update statistics
            let stats = self.stats.entry(entry.name.clone()).or_default();
            stats.count += 1;
            if let Some(exit) = entry.exit_time {
                stats.total_time += exit - entry.entry_time;
            }
            if result < 0 {
                stats.errors += 1;
            }

            // Handle fork/clone for follow_forks
            if self.config.follow_forks {
                if entry.name == "fork" || entry.name == "clone" || entry.name == "vfork" {
                    if result > 0 {
                        self.traced_pids.push(result as u32);
                    }
                }
            }

            self.syscalls.push(entry);
        }
    }

    /// Record signal
    pub fn on_signal(&mut self, pid: u32, signum: u32, sender: Option<u32>, timestamp: u64) {
        if !self.running || !self.config.trace_signals {
            return;
        }

        self.signals.push(SignalEvent {
            signum,
            name: signal_name(signum),
            sender_pid: sender,
            timestamp,
            action: SignalAction::Default,
        });
    }

    /// Record IPC event
    pub fn on_ipc(&mut self, event: IpcEvent) {
        if !self.running || !self.config.trace_ipc {
            return;
        }
        self.ipc_events.push(event);
    }

    /// Parse syscall arguments
    fn parse_args(&self, number: u32, args: &[u64]) -> Vec<SyscallArg> {
        // Parse based on syscall number
        // This is a simplified version - real implementation would have
        // detailed argument type information for each syscall
        match number {
            // read(fd, buf, count)
            0 => vec![
                SyscallArg::Fd(args.get(0).map(|&v| v as i32).unwrap_or(0)),
                SyscallArg::Ptr(args.get(1).copied().unwrap_or(0)),
                SyscallArg::UInt(args.get(2).copied().unwrap_or(0)),
            ],
            // write(fd, buf, count)
            1 => vec![
                SyscallArg::Fd(args.get(0).map(|&v| v as i32).unwrap_or(0)),
                SyscallArg::Ptr(args.get(1).copied().unwrap_or(0)),
                SyscallArg::UInt(args.get(2).copied().unwrap_or(0)),
            ],
            // open(path, flags, mode)
            2 => vec![
                SyscallArg::String(String::from("<path>")), // Would read from process memory
                SyscallArg::Flags {
                    value: args.get(1).copied().unwrap_or(0),
                    names: parse_open_flags(args.get(1).copied().unwrap_or(0)),
                },
                SyscallArg::UInt(args.get(2).copied().unwrap_or(0)),
            ],
            // Default: treat all as unsigned integers
            _ => args.iter().map(|&v| SyscallArg::UInt(v)).collect(),
        }
    }

    /// Format a syscall entry for output
    pub fn format_syscall(&self, entry: &SyscallEntry) -> String {
        let args_str: Vec<String> = entry.args.iter().map(|a| a.format()).collect();

        let mut output = String::new();

        if self.config.show_timestamps {
            output.push_str(&format!("{:>10} ", entry.entry_time));
        }

        output.push_str(&format!(
            "{}({}) = ",
            entry.name,
            args_str.join(", ")
        ));

        if let Some(result) = entry.result {
            if result < 0 {
                output.push_str(&format!("-1 {} ({})",
                    entry.error.as_deref().unwrap_or("UNKNOWN"),
                    -result
                ));
            } else {
                output.push_str(&format!("{}", result));
            }
        } else {
            output.push_str("?");
        }

        if let (Some(entry_t), Some(exit_t)) = (Some(entry.entry_time), entry.exit_time) {
            output.push_str(&format!(" <{} us>", exit_t - entry_t));
        }

        output
    }

    /// Get syscall log
    pub fn syscalls(&self) -> &[SyscallEntry] {
        &self.syscalls
    }

    /// Get signal log
    pub fn signals(&self) -> &[SignalEvent] {
        &self.signals
    }

    /// Get statistics
    pub fn statistics(&self) -> &BTreeMap<String, SyscallStats> {
        &self.stats
    }

    /// Generate statistics summary
    pub fn stats_summary(&self) -> String {
        let mut output = String::new();
        output.push_str("% time     seconds  usecs/call     calls    errors syscall\n");
        output.push_str("------ ----------- ----------- --------- --------- --------\n");

        let total_time: u64 = self.stats.values().map(|s| s.total_time).sum();

        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

        for (name, stats) in sorted {
            let pct = if total_time > 0 {
                (stats.total_time as f64 / total_time as f64) * 100.0
            } else {
                0.0
            };
            let avg = if stats.count > 0 {
                stats.total_time / stats.count
            } else {
                0
            };

            output.push_str(&format!(
                "{:6.2} {:11.6} {:11} {:9} {:9} {}\n",
                pct,
                stats.total_time as f64 / 1_000_000.0,
                avg,
                stats.count,
                if stats.errors > 0 { format!("{}", stats.errors) } else { String::new() },
                name
            ));
        }

        output
    }
}

impl Default for SyscallTracer {
    fn default() -> Self {
        Self::new(TracerConfig::default())
    }
}

/// Tracer errors
#[derive(Clone, Debug)]
pub enum TracerError {
    /// Already tracing
    AlreadyTracing,
    /// Process not found
    ProcessNotFound,
    /// Permission denied
    PermissionDenied,
    /// Attach failed
    AttachFailed,
}

/// Get syscall name from number (AArch64)
fn syscall_name(number: u32) -> String {
    match number {
        0 => String::from("read"),
        1 => String::from("write"),
        2 => String::from("open"),
        3 => String::from("close"),
        4 => String::from("stat"),
        5 => String::from("fstat"),
        6 => String::from("lstat"),
        7 => String::from("poll"),
        8 => String::from("lseek"),
        9 => String::from("mmap"),
        10 => String::from("mprotect"),
        11 => String::from("munmap"),
        12 => String::from("brk"),
        56 => String::from("clone"),
        57 => String::from("fork"),
        58 => String::from("vfork"),
        59 => String::from("execve"),
        60 => String::from("exit"),
        61 => String::from("wait4"),
        62 => String::from("kill"),
        // AI-specific syscalls (HubLab IO extensions)
        500 => String::from("ai_load"),
        501 => String::from("ai_generate"),
        502 => String::from("ai_unload"),
        _ => format!("syscall_{}", number),
    }
}

/// Get signal name
fn signal_name(signum: u32) -> String {
    match signum {
        1 => String::from("SIGHUP"),
        2 => String::from("SIGINT"),
        3 => String::from("SIGQUIT"),
        4 => String::from("SIGILL"),
        5 => String::from("SIGTRAP"),
        6 => String::from("SIGABRT"),
        7 => String::from("SIGBUS"),
        8 => String::from("SIGFPE"),
        9 => String::from("SIGKILL"),
        10 => String::from("SIGUSR1"),
        11 => String::from("SIGSEGV"),
        12 => String::from("SIGUSR2"),
        13 => String::from("SIGPIPE"),
        14 => String::from("SIGALRM"),
        15 => String::from("SIGTERM"),
        17 => String::from("SIGCHLD"),
        18 => String::from("SIGCONT"),
        19 => String::from("SIGSTOP"),
        20 => String::from("SIGTSTP"),
        _ => format!("SIG{}", signum),
    }
}

/// Get errno name
fn errno_name(errno: u32) -> String {
    match errno {
        1 => String::from("EPERM"),
        2 => String::from("ENOENT"),
        3 => String::from("ESRCH"),
        4 => String::from("EINTR"),
        5 => String::from("EIO"),
        9 => String::from("EBADF"),
        11 => String::from("EAGAIN"),
        12 => String::from("ENOMEM"),
        13 => String::from("EACCES"),
        14 => String::from("EFAULT"),
        17 => String::from("EEXIST"),
        22 => String::from("EINVAL"),
        _ => format!("errno_{}", errno),
    }
}

/// Parse open() flags
fn parse_open_flags(flags: u64) -> Vec<String> {
    let mut names = Vec::new();

    let access = flags & 3;
    match access {
        0 => names.push(String::from("O_RDONLY")),
        1 => names.push(String::from("O_WRONLY")),
        2 => names.push(String::from("O_RDWR")),
        _ => {}
    }

    if flags & 0x40 != 0 { names.push(String::from("O_CREAT")); }
    if flags & 0x80 != 0 { names.push(String::from("O_EXCL")); }
    if flags & 0x200 != 0 { names.push(String::from("O_TRUNC")); }
    if flags & 0x400 != 0 { names.push(String::from("O_APPEND")); }
    if flags & 0x800 != 0 { names.push(String::from("O_NONBLOCK")); }

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_name() {
        assert_eq!(syscall_name(0), "read");
        assert_eq!(syscall_name(1), "write");
        assert_eq!(syscall_name(500), "ai_load");
    }

    #[test]
    fn test_signal_name() {
        assert_eq!(signal_name(9), "SIGKILL");
        assert_eq!(signal_name(15), "SIGTERM");
    }

    #[test]
    fn test_open_flags() {
        let flags = parse_open_flags(0x42); // O_RDWR | O_CREAT
        assert!(flags.contains(&String::from("O_RDWR")));
        assert!(flags.contains(&String::from("O_CREAT")));
    }
}
