//! System Inspector
//!
//! Runtime inspection tools for processes, memory, and system state.
//! Similar to /proc filesystem exploration tools.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;

/// Process information
#[derive(Clone, Debug)]
pub struct ProcessInfo {
    /// Process ID
    pub pid: u32,
    /// Parent PID
    pub ppid: u32,
    /// Process name
    pub name: String,
    /// Command line
    pub cmdline: Vec<String>,
    /// State
    pub state: ProcessState,
    /// Priority
    pub priority: i32,
    /// Nice value
    pub nice: i32,
    /// Number of threads
    pub num_threads: u32,
    /// Start time
    pub start_time: u64,
    /// CPU time (user)
    pub user_time: u64,
    /// CPU time (system)
    pub system_time: u64,
    /// Virtual memory size
    pub vsize: u64,
    /// Resident set size
    pub rss: u64,
    /// User ID
    pub uid: u32,
    /// Group ID
    pub gid: u32,
}

/// Process state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessState {
    Running,
    Sleeping,
    DiskSleep,
    Stopped,
    Zombie,
    Dead,
}

impl ProcessState {
    /// Get state character
    pub fn as_char(&self) -> char {
        match self {
            ProcessState::Running => 'R',
            ProcessState::Sleeping => 'S',
            ProcessState::DiskSleep => 'D',
            ProcessState::Stopped => 'T',
            ProcessState::Zombie => 'Z',
            ProcessState::Dead => 'X',
        }
    }
}

/// Thread information
#[derive(Clone, Debug)]
pub struct ThreadInfo {
    /// Thread ID
    pub tid: u32,
    /// Parent PID
    pub pid: u32,
    /// Thread name
    pub name: String,
    /// State
    pub state: ProcessState,
    /// CPU affinity mask
    pub cpu_affinity: u64,
    /// Stack base address
    pub stack_base: u64,
    /// Stack size
    pub stack_size: usize,
}

/// Memory mapping
#[derive(Clone, Debug)]
pub struct MemoryMapping {
    /// Start address
    pub start: u64,
    /// End address
    pub end: u64,
    /// Readable
    pub read: bool,
    /// Writable
    pub write: bool,
    /// Executable
    pub execute: bool,
    /// Private (vs shared)
    pub private: bool,
    /// File offset
    pub offset: u64,
    /// Device
    pub device: String,
    /// Inode
    pub inode: u64,
    /// Path name
    pub pathname: String,
}

impl MemoryMapping {
    /// Format permissions
    pub fn perms_string(&self) -> String {
        format!("{}{}{}{}",
            if self.read { 'r' } else { '-' },
            if self.write { 'w' } else { '-' },
            if self.execute { 'x' } else { '-' },
            if self.private { 'p' } else { 's' }
        )
    }

    /// Get size in bytes
    pub fn size(&self) -> u64 {
        self.end - self.start
    }
}

/// File descriptor information
#[derive(Clone, Debug)]
pub struct FdInfo {
    /// FD number
    pub fd: i32,
    /// Target path
    pub path: String,
    /// FD type
    pub fd_type: FdType,
    /// Flags
    pub flags: u32,
    /// Position (for files)
    pub position: Option<u64>,
}

/// File descriptor type
#[derive(Clone, Debug)]
pub enum FdType {
    Regular,
    Directory,
    CharDevice,
    BlockDevice,
    Pipe,
    Socket,
    Symlink,
    Unknown,
}

/// System memory information
#[derive(Clone, Debug, Default)]
pub struct MemInfo {
    /// Total memory
    pub total: u64,
    /// Free memory
    pub free: u64,
    /// Available memory
    pub available: u64,
    /// Buffer memory
    pub buffers: u64,
    /// Cached memory
    pub cached: u64,
    /// Swap total
    pub swap_total: u64,
    /// Swap free
    pub swap_free: u64,
    /// Dirty pages
    pub dirty: u64,
    /// Writeback
    pub writeback: u64,
    /// Mapped memory
    pub mapped: u64,
    /// Slab memory
    pub slab: u64,
    /// Kernel stack
    pub kernel_stack: u64,
    /// Page tables
    pub page_tables: u64,
}

impl MemInfo {
    /// Get used memory
    pub fn used(&self) -> u64 {
        self.total - self.free - self.buffers - self.cached
    }

    /// Get memory usage percentage
    pub fn usage_percent(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.used() as f32 / self.total as f32) * 100.0
    }
}

/// CPU information
#[derive(Clone, Debug, Default)]
pub struct CpuInfo {
    /// CPU number
    pub cpu: u32,
    /// Model name
    pub model: String,
    /// Clock speed (MHz)
    pub mhz: f32,
    /// Cache size
    pub cache_size: u64,
    /// User time
    pub user: u64,
    /// Nice time
    pub nice: u64,
    /// System time
    pub system: u64,
    /// Idle time
    pub idle: u64,
    /// I/O wait time
    pub iowait: u64,
    /// IRQ time
    pub irq: u64,
    /// Soft IRQ time
    pub softirq: u64,
}

impl CpuInfo {
    /// Get total time
    pub fn total(&self) -> u64 {
        self.user + self.nice + self.system + self.idle + self.iowait + self.irq + self.softirq
    }

    /// Get active time
    pub fn active(&self) -> u64 {
        self.user + self.nice + self.system + self.irq + self.softirq
    }

    /// Get usage percentage
    pub fn usage_percent(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.active() as f32 / total as f32) * 100.0
    }
}

/// Network interface information
#[derive(Clone, Debug)]
pub struct NetInterface {
    /// Interface name
    pub name: String,
    /// Bytes received
    pub rx_bytes: u64,
    /// Packets received
    pub rx_packets: u64,
    /// RX errors
    pub rx_errors: u64,
    /// RX drops
    pub rx_drops: u64,
    /// Bytes transmitted
    pub tx_bytes: u64,
    /// Packets transmitted
    pub tx_packets: u64,
    /// TX errors
    pub tx_errors: u64,
    /// TX drops
    pub tx_drops: u64,
}

/// Disk information
#[derive(Clone, Debug)]
pub struct DiskInfo {
    /// Device name
    pub name: String,
    /// Reads completed
    pub reads: u64,
    /// Sectors read
    pub read_sectors: u64,
    /// Read time (ms)
    pub read_time: u64,
    /// Writes completed
    pub writes: u64,
    /// Sectors written
    pub write_sectors: u64,
    /// Write time (ms)
    pub write_time: u64,
    /// I/Os in progress
    pub in_progress: u64,
}

/// System inspector
pub struct SystemInspector {
    /// Cached process list
    processes: BTreeMap<u32, ProcessInfo>,
    /// Last update time
    last_update: u64,
}

impl SystemInspector {
    /// Create new inspector
    pub fn new() -> Self {
        Self {
            processes: BTreeMap::new(),
            last_update: 0,
        }
    }

    /// Get process information
    pub fn process(&self, pid: u32) -> Option<ProcessInfo> {
        // TODO: Read from /proc/{pid}/stat, /proc/{pid}/status, etc.
        self.processes.get(&pid).cloned()
    }

    /// List all processes
    pub fn list_processes(&mut self) -> Vec<ProcessInfo> {
        // TODO: Scan /proc for process directories
        self.processes.values().cloned().collect()
    }

    /// Get process threads
    pub fn threads(&self, pid: u32) -> Vec<ThreadInfo> {
        // TODO: Read from /proc/{pid}/task/
        Vec::new()
    }

    /// Get process memory maps
    pub fn memory_maps(&self, pid: u32) -> Vec<MemoryMapping> {
        // TODO: Read from /proc/{pid}/maps
        Vec::new()
    }

    /// Get process file descriptors
    pub fn file_descriptors(&self, pid: u32) -> Vec<FdInfo> {
        // TODO: Read from /proc/{pid}/fd/
        Vec::new()
    }

    /// Get system memory info
    pub fn memory_info(&self) -> MemInfo {
        // TODO: Read from /proc/meminfo
        MemInfo::default()
    }

    /// Get CPU info
    pub fn cpu_info(&self) -> Vec<CpuInfo> {
        // TODO: Read from /proc/cpuinfo and /proc/stat
        Vec::new()
    }

    /// Get network interfaces
    pub fn network_interfaces(&self) -> Vec<NetInterface> {
        // TODO: Read from /proc/net/dev
        Vec::new()
    }

    /// Get disk info
    pub fn disk_info(&self) -> Vec<DiskInfo> {
        // TODO: Read from /proc/diskstats
        Vec::new()
    }

    /// Get load average
    pub fn load_average(&self) -> (f32, f32, f32) {
        // TODO: Read from /proc/loadavg
        (0.0, 0.0, 0.0)
    }

    /// Get uptime
    pub fn uptime(&self) -> (u64, u64) {
        // TODO: Read from /proc/uptime
        // Returns (uptime_seconds, idle_seconds)
        (0, 0)
    }

    /// Format process list (like ps)
    pub fn format_process_list(&mut self) -> String {
        let mut output = String::new();
        output.push_str("  PID  PPID S PRI  NI    VSZ    RSS %CPU %MEM COMMAND\n");

        let mem = self.memory_info();
        let procs = self.list_processes();

        for p in procs {
            let cpu_pct = 0.0f32; // Would need time-based calculation
            let mem_pct = if mem.total > 0 {
                (p.rss as f32 / mem.total as f32) * 100.0
            } else {
                0.0
            };

            output.push_str(&format!(
                "{:5} {:5} {} {:3} {:3} {:6} {:6} {:4.1} {:4.1} {}\n",
                p.pid,
                p.ppid,
                p.state.as_char(),
                p.priority,
                p.nice,
                p.vsize / 1024,
                p.rss / 1024,
                cpu_pct,
                mem_pct,
                p.name
            ));
        }

        output
    }

    /// Format memory info (like free)
    pub fn format_memory_info(&self) -> String {
        let m = self.memory_info();

        let mut output = String::new();
        output.push_str("              total        used        free      shared  buff/cache   available\n");
        output.push_str(&format!(
            "Mem:    {:11} {:11} {:11} {:11} {:11} {:11}\n",
            m.total,
            m.used(),
            m.free,
            0, // shared
            m.buffers + m.cached,
            m.available
        ));
        output.push_str(&format!(
            "Swap:   {:11} {:11} {:11}\n",
            m.swap_total,
            m.swap_total - m.swap_free,
            m.swap_free
        ));

        output
    }

    /// Format CPU info (like top header)
    pub fn format_cpu_info(&self) -> String {
        let cpus = self.cpu_info();
        if cpus.is_empty() {
            return String::from("No CPU info available\n");
        }

        let mut output = String::new();

        // Aggregate CPU stats
        let total_user: u64 = cpus.iter().map(|c| c.user).sum();
        let total_system: u64 = cpus.iter().map(|c| c.system).sum();
        let total_idle: u64 = cpus.iter().map(|c| c.idle).sum();
        let total_iowait: u64 = cpus.iter().map(|c| c.iowait).sum();
        let total: u64 = cpus.iter().map(|c| c.total()).sum();

        let to_pct = |v: u64| -> f32 {
            if total == 0 { 0.0 } else { (v as f32 / total as f32) * 100.0 }
        };

        output.push_str(&format!(
            "%Cpu(s): {:5.1} us, {:5.1} sy, {:5.1} id, {:5.1} wa\n",
            to_pct(total_user),
            to_pct(total_system),
            to_pct(total_idle),
            to_pct(total_iowait)
        ));

        for cpu in &cpus {
            output.push_str(&format!(
                "%Cpu{}: {:5.1} us, {:5.1} sy, {:5.1} id\n",
                cpu.cpu,
                if cpu.total() == 0 { 0.0 } else { (cpu.user as f32 / cpu.total() as f32) * 100.0 },
                if cpu.total() == 0 { 0.0 } else { (cpu.system as f32 / cpu.total() as f32) * 100.0 },
                if cpu.total() == 0 { 0.0 } else { (cpu.idle as f32 / cpu.total() as f32) * 100.0 },
            ));
        }

        output
    }

    /// Format memory maps (like pmap)
    pub fn format_memory_maps(&self, pid: u32) -> String {
        let maps = self.memory_maps(pid);
        let mut output = String::new();

        output.push_str(&format!("{}:  \n", pid));
        output.push_str("Address           Kbytes Mode  Mapping\n");

        let mut total_kb = 0u64;

        for map in &maps {
            let kb = map.size() / 1024;
            total_kb += kb;

            output.push_str(&format!(
                "{:016x} {:7} {} {}\n",
                map.start,
                kb,
                map.perms_string(),
                if map.pathname.is_empty() { "[anon]" } else { &map.pathname }
            ));
        }

        output.push_str(&format!("total {:15}K\n", total_kb));

        output
    }

    /// Format file descriptors (like lsof)
    pub fn format_file_descriptors(&self, pid: u32) -> String {
        let fds = self.file_descriptors(pid);
        let mut output = String::new();

        output.push_str("COMMAND   PID   FD   TYPE NAME\n");

        for fd in &fds {
            let fd_type = match fd.fd_type {
                FdType::Regular => "REG",
                FdType::Directory => "DIR",
                FdType::CharDevice => "CHR",
                FdType::BlockDevice => "BLK",
                FdType::Pipe => "FIFO",
                FdType::Socket => "sock",
                FdType::Symlink => "LINK",
                FdType::Unknown => "unknown",
            };

            output.push_str(&format!(
                "          {:5} {:4} {:6} {}\n",
                pid,
                fd.fd,
                fd_type,
                fd.path
            ));
        }

        output
    }

    /// Format network interfaces
    pub fn format_network(&self) -> String {
        let ifaces = self.network_interfaces();
        let mut output = String::new();

        output.push_str("Interface        RX bytes   RX pkts   TX bytes   TX pkts\n");

        for iface in &ifaces {
            output.push_str(&format!(
                "{:<15} {:10} {:9} {:10} {:9}\n",
                iface.name,
                iface.rx_bytes,
                iface.rx_packets,
                iface.tx_bytes,
                iface.tx_packets
            ));
        }

        output
    }

    /// Check if process exists
    pub fn process_exists(&self, pid: u32) -> bool {
        // TODO: Check /proc/{pid}
        self.processes.contains_key(&pid)
    }

    /// Kill process
    pub fn kill_process(&self, pid: u32, signal: u32) -> Result<(), InspectorError> {
        if !self.process_exists(pid) {
            return Err(InspectorError::ProcessNotFound);
        }
        // TODO: Actually send signal
        Ok(())
    }
}

impl Default for SystemInspector {
    fn default() -> Self {
        Self::new()
    }
}

/// Inspector errors
#[derive(Clone, Debug)]
pub enum InspectorError {
    /// Process not found
    ProcessNotFound,
    /// Permission denied
    PermissionDenied,
    /// Read error
    ReadError,
}

/// Process tree node
#[derive(Clone, Debug)]
pub struct ProcessTreeNode {
    /// Process info
    pub info: ProcessInfo,
    /// Children
    pub children: Vec<ProcessTreeNode>,
}

/// Build process tree
pub fn build_process_tree(processes: &[ProcessInfo]) -> Vec<ProcessTreeNode> {
    let mut roots = Vec::new();
    let by_pid: BTreeMap<u32, &ProcessInfo> = processes.iter()
        .map(|p| (p.pid, p))
        .collect();

    fn add_children(parent_pid: u32, by_pid: &BTreeMap<u32, &ProcessInfo>, processes: &[ProcessInfo]) -> Vec<ProcessTreeNode> {
        let mut children = Vec::new();

        for p in processes {
            if p.ppid == parent_pid && p.pid != parent_pid {
                children.push(ProcessTreeNode {
                    info: p.clone(),
                    children: add_children(p.pid, by_pid, processes),
                });
            }
        }

        children
    }

    // Find root processes (ppid = 0 or ppid = self)
    for p in processes {
        if p.ppid == 0 || p.pid == p.ppid {
            roots.push(ProcessTreeNode {
                info: p.clone(),
                children: add_children(p.pid, &by_pid, processes),
            });
        }
    }

    roots
}

/// Format process tree (like pstree)
pub fn format_process_tree(roots: &[ProcessTreeNode]) -> String {
    fn format_node(node: &ProcessTreeNode, prefix: &str, is_last: bool) -> String {
        let mut output = String::new();

        let connector = if is_last { "└─" } else { "├─" };
        output.push_str(&format!("{}{}{}\n", prefix, connector, node.info.name));

        let child_prefix = if is_last {
            format!("{}  ", prefix)
        } else {
            format!("{}│ ", prefix)
        };

        for (i, child) in node.children.iter().enumerate() {
            let is_last_child = i == node.children.len() - 1;
            output.push_str(&format_node(child, &child_prefix, is_last_child));
        }

        output
    }

    let mut output = String::new();

    for (i, root) in roots.iter().enumerate() {
        output.push_str(&format!("{}\n", root.info.name));
        for (j, child) in root.children.iter().enumerate() {
            let is_last = j == root.children.len() - 1;
            output.push_str(&format_node(child, "", is_last));
        }
        if i < roots.len() - 1 {
            output.push('\n');
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_mapping_perms() {
        let map = MemoryMapping {
            start: 0x1000,
            end: 0x2000,
            read: true,
            write: true,
            execute: false,
            private: true,
            offset: 0,
            device: String::new(),
            inode: 0,
            pathname: String::new(),
        };

        assert_eq!(map.perms_string(), "rw-p");
        assert_eq!(map.size(), 0x1000);
    }

    #[test]
    fn test_process_state_char() {
        assert_eq!(ProcessState::Running.as_char(), 'R');
        assert_eq!(ProcessState::Sleeping.as_char(), 'S');
        assert_eq!(ProcessState::Zombie.as_char(), 'Z');
    }
}
