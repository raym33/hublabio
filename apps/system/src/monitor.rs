//! System Monitor Application
//!
//! Real-time system monitoring for HubLab IO.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// CPU core information
#[derive(Clone, Debug)]
pub struct CpuCore {
    /// Core ID
    pub id: u32,
    /// Current usage percentage (0-100)
    pub usage: f32,
    /// Current frequency in MHz
    pub frequency: u32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Is online
    pub online: bool,
}

/// CPU information
#[derive(Clone, Debug)]
pub struct CpuInfo {
    /// CPU model name
    pub model: String,
    /// Architecture
    pub arch: String,
    /// Number of cores
    pub cores: u32,
    /// Per-core information
    pub core_info: Vec<CpuCore>,
    /// Overall usage
    pub total_usage: f32,
    /// Load average (1, 5, 15 min)
    pub load_avg: (f32, f32, f32),
}

/// Memory information
#[derive(Clone, Debug)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total: u64,
    /// Used memory in bytes
    pub used: u64,
    /// Free memory in bytes
    pub free: u64,
    /// Cached memory in bytes
    pub cached: u64,
    /// Buffer memory in bytes
    pub buffers: u64,
    /// Swap total in bytes
    pub swap_total: u64,
    /// Swap used in bytes
    pub swap_used: u64,
}

impl MemoryInfo {
    /// Usage percentage
    pub fn usage_percent(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.used as f64 / self.total as f64 * 100.0) as f32
    }

    /// Format bytes for display
    pub fn format_bytes(bytes: u64) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

/// Network interface info
#[derive(Clone, Debug)]
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    /// IP address
    pub ip_addr: Option<String>,
    /// MAC address
    pub mac_addr: String,
    /// Is up
    pub is_up: bool,
    /// Bytes received
    pub rx_bytes: u64,
    /// Bytes transmitted
    pub tx_bytes: u64,
    /// Packets received
    pub rx_packets: u64,
    /// Packets transmitted
    pub tx_packets: u64,
    /// RX errors
    pub rx_errors: u64,
    /// TX errors
    pub tx_errors: u64,
    /// RX speed (bytes/sec)
    pub rx_speed: u64,
    /// TX speed (bytes/sec)
    pub tx_speed: u64,
}

/// Disk information
#[derive(Clone, Debug)]
pub struct DiskInfo {
    /// Device path
    pub device: String,
    /// Mount point
    pub mount_point: String,
    /// Filesystem type
    pub fs_type: String,
    /// Total size in bytes
    pub total: u64,
    /// Used space in bytes
    pub used: u64,
    /// Free space in bytes
    pub free: u64,
    /// Read bytes
    pub read_bytes: u64,
    /// Write bytes
    pub write_bytes: u64,
    /// Read speed (bytes/sec)
    pub read_speed: u64,
    /// Write speed (bytes/sec)
    pub write_speed: u64,
}

impl DiskInfo {
    /// Usage percentage
    pub fn usage_percent(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.used as f64 / self.total as f64 * 100.0) as f32
    }
}

/// Process state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessState {
    Running,
    Sleeping,
    Waiting,
    Stopped,
    Zombie,
    Dead,
}

impl ProcessState {
    pub fn as_char(&self) -> char {
        match self {
            Self::Running => 'R',
            Self::Sleeping => 'S',
            Self::Waiting => 'D',
            Self::Stopped => 'T',
            Self::Zombie => 'Z',
            Self::Dead => 'X',
        }
    }
}

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
    pub cmdline: String,
    /// State
    pub state: ProcessState,
    /// User ID
    pub uid: u32,
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub mem_usage: u64,
    /// Memory usage percentage
    pub mem_percent: f32,
    /// Number of threads
    pub threads: u32,
    /// Nice value
    pub nice: i32,
    /// Priority
    pub priority: i32,
    /// Start time (unix timestamp)
    pub start_time: u64,
    /// CPU time (user + system) in ms
    pub cpu_time: u64,
}

/// Sort order for processes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessSort {
    Pid,
    Name,
    Cpu,
    Memory,
    State,
}

/// System monitor configuration
#[derive(Clone, Debug)]
pub struct MonitorConfig {
    /// Update interval in milliseconds
    pub update_interval_ms: u32,
    /// Show kernel threads
    pub show_kernel_threads: bool,
    /// Show per-CPU usage
    pub show_per_cpu: bool,
    /// Process sort order
    pub process_sort: ProcessSort,
    /// Sort descending
    pub sort_descending: bool,
    /// CPU usage history size
    pub history_size: usize,
    /// Show temperatures
    pub show_temperatures: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 1000,
            show_kernel_threads: false,
            show_per_cpu: true,
            process_sort: ProcessSort::Cpu,
            sort_descending: true,
            history_size: 60,
            show_temperatures: true,
        }
    }
}

/// System monitor state
pub struct SystemMonitor {
    /// Configuration
    config: MonitorConfig,
    /// CPU info
    cpu_info: Option<CpuInfo>,
    /// Memory info
    memory_info: Option<MemoryInfo>,
    /// Network interfaces
    network_interfaces: Vec<NetworkInterface>,
    /// Disk information
    disks: Vec<DiskInfo>,
    /// Process list
    processes: Vec<ProcessInfo>,
    /// CPU usage history
    cpu_history: Vec<f32>,
    /// Memory usage history
    mem_history: Vec<f32>,
    /// Selected process index
    selected_process: usize,
    /// Current view tab
    current_tab: MonitorTab,
    /// Uptime in seconds
    uptime: u64,
}

/// Monitor view tabs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonitorTab {
    Overview,
    Processes,
    Cpu,
    Memory,
    Network,
    Disks,
}

impl SystemMonitor {
    /// Create new system monitor
    pub fn new() -> Self {
        Self {
            config: MonitorConfig::default(),
            cpu_info: None,
            memory_info: None,
            network_interfaces: Vec::new(),
            disks: Vec::new(),
            processes: Vec::new(),
            cpu_history: Vec::new(),
            mem_history: Vec::new(),
            selected_process: 0,
            current_tab: MonitorTab::Overview,
            uptime: 0,
        }
    }

    /// Update all system information
    pub fn update(&mut self) {
        self.update_cpu();
        self.update_memory();
        self.update_network();
        self.update_disks();
        self.update_processes();
        self.update_uptime();

        // Update history
        if let Some(ref cpu) = self.cpu_info {
            self.cpu_history.push(cpu.total_usage);
            if self.cpu_history.len() > self.config.history_size {
                self.cpu_history.remove(0);
            }
        }

        if let Some(ref mem) = self.memory_info {
            self.mem_history.push(mem.usage_percent());
            if self.mem_history.len() > self.config.history_size {
                self.mem_history.remove(0);
            }
        }
    }

    /// Update CPU info
    fn update_cpu(&mut self) {
        // Mock data - would read from /proc/stat in real implementation
        let cores = 4;
        let mut core_info = Vec::new();

        for i in 0..cores {
            core_info.push(CpuCore {
                id: i,
                usage: 10.0 + (i as f32) * 5.0, // Mock
                frequency: 1800,
                temperature: 45.0 + (i as f32) * 2.0,
                online: true,
            });
        }

        let total_usage: f32 = core_info.iter().map(|c| c.usage).sum::<f32>() / cores as f32;

        self.cpu_info = Some(CpuInfo {
            model: String::from("ARM Cortex-A76"),
            arch: String::from("aarch64"),
            cores,
            core_info,
            total_usage,
            load_avg: (0.5, 0.3, 0.2),
        });
    }

    /// Update memory info
    fn update_memory(&mut self) {
        // Mock data - would read from /proc/meminfo
        let total = 4 * 1024 * 1024 * 1024; // 4GB
        let used = 1500 * 1024 * 1024; // 1.5GB

        self.memory_info = Some(MemoryInfo {
            total,
            used,
            free: total - used,
            cached: 512 * 1024 * 1024,
            buffers: 128 * 1024 * 1024,
            swap_total: 0,
            swap_used: 0,
        });
    }

    /// Update network info
    fn update_network(&mut self) {
        // Mock data
        self.network_interfaces = alloc::vec![
            NetworkInterface {
                name: String::from("eth0"),
                ip_addr: Some(String::from("192.168.1.100")),
                mac_addr: String::from("dc:a6:32:xx:xx:xx"),
                is_up: true,
                rx_bytes: 1024 * 1024 * 100,
                tx_bytes: 1024 * 1024 * 50,
                rx_packets: 100000,
                tx_packets: 50000,
                rx_errors: 0,
                tx_errors: 0,
                rx_speed: 1024 * 100,
                tx_speed: 1024 * 50,
            },
            NetworkInterface {
                name: String::from("wlan0"),
                ip_addr: Some(String::from("192.168.1.101")),
                mac_addr: String::from("dc:a6:32:yy:yy:yy"),
                is_up: true,
                rx_bytes: 1024 * 1024 * 200,
                tx_bytes: 1024 * 1024 * 100,
                rx_packets: 200000,
                tx_packets: 100000,
                rx_errors: 5,
                tx_errors: 2,
                rx_speed: 1024 * 200,
                tx_speed: 1024 * 100,
            },
        ];
    }

    /// Update disk info
    fn update_disks(&mut self) {
        // Mock data
        self.disks = alloc::vec![
            DiskInfo {
                device: String::from("/dev/mmcblk0p2"),
                mount_point: String::from("/"),
                fs_type: String::from("ext4"),
                total: 32 * 1024 * 1024 * 1024,
                used: 8 * 1024 * 1024 * 1024,
                free: 24 * 1024 * 1024 * 1024,
                read_bytes: 1024 * 1024 * 500,
                write_bytes: 1024 * 1024 * 200,
                read_speed: 1024 * 50,
                write_speed: 1024 * 20,
            },
            DiskInfo {
                device: String::from("/dev/mmcblk0p1"),
                mount_point: String::from("/boot"),
                fs_type: String::from("fat32"),
                total: 256 * 1024 * 1024,
                used: 64 * 1024 * 1024,
                free: 192 * 1024 * 1024,
                read_bytes: 1024 * 1024 * 10,
                write_bytes: 1024 * 1024 * 2,
                read_speed: 0,
                write_speed: 0,
            },
        ];
    }

    /// Update process list
    fn update_processes(&mut self) {
        // Mock data - would read from /proc/[pid]
        self.processes = alloc::vec![
            ProcessInfo {
                pid: 1,
                ppid: 0,
                name: String::from("init"),
                cmdline: String::from("/sbin/init"),
                state: ProcessState::Sleeping,
                uid: 0,
                cpu_usage: 0.1,
                mem_usage: 2 * 1024 * 1024,
                mem_percent: 0.05,
                threads: 1,
                nice: 0,
                priority: 20,
                start_time: 0,
                cpu_time: 1000,
            },
            ProcessInfo {
                pid: 2,
                ppid: 1,
                name: String::from("ai_service"),
                cmdline: String::from("/usr/bin/ai_service --model tiny"),
                state: ProcessState::Sleeping,
                uid: 1000,
                cpu_usage: 15.5,
                mem_usage: 512 * 1024 * 1024,
                mem_percent: 12.5,
                threads: 4,
                nice: 0,
                priority: 20,
                start_time: 100,
                cpu_time: 50000,
            },
            ProcessInfo {
                pid: 3,
                ppid: 1,
                name: String::from("shell"),
                cmdline: String::from("/usr/bin/hublabsh"),
                state: ProcessState::Running,
                uid: 1000,
                cpu_usage: 2.0,
                mem_usage: 8 * 1024 * 1024,
                mem_percent: 0.2,
                threads: 1,
                nice: 0,
                priority: 20,
                start_time: 200,
                cpu_time: 5000,
            },
            ProcessInfo {
                pid: 4,
                ppid: 1,
                name: String::from("network"),
                cmdline: String::from("/usr/bin/networkd"),
                state: ProcessState::Sleeping,
                uid: 0,
                cpu_usage: 0.5,
                mem_usage: 16 * 1024 * 1024,
                mem_percent: 0.4,
                threads: 2,
                nice: 0,
                priority: 20,
                start_time: 50,
                cpu_time: 2000,
            },
        ];

        // Sort processes
        self.sort_processes();
    }

    /// Sort process list
    fn sort_processes(&mut self) {
        self.processes.sort_by(|a, b| {
            let cmp = match self.config.process_sort {
                ProcessSort::Pid => a.pid.cmp(&b.pid),
                ProcessSort::Name => a.name.cmp(&b.name),
                ProcessSort::Cpu => a
                    .cpu_usage
                    .partial_cmp(&b.cpu_usage)
                    .unwrap_or(core::cmp::Ordering::Equal),
                ProcessSort::Memory => a.mem_usage.cmp(&b.mem_usage),
                ProcessSort::State => (a.state as u8).cmp(&(b.state as u8)),
            };

            if self.config.sort_descending {
                cmp.reverse()
            } else {
                cmp
            }
        });
    }

    /// Update uptime
    fn update_uptime(&mut self) {
        // Would read from /proc/uptime
        self.uptime += 1; // Mock: increment by 1 second
    }

    /// Format uptime for display
    pub fn format_uptime(&self) -> String {
        let days = self.uptime / 86400;
        let hours = (self.uptime % 86400) / 3600;
        let minutes = (self.uptime % 3600) / 60;
        let seconds = self.uptime % 60;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}s", seconds)
        }
    }

    /// Get CPU info
    pub fn cpu_info(&self) -> Option<&CpuInfo> {
        self.cpu_info.as_ref()
    }

    /// Get memory info
    pub fn memory_info(&self) -> Option<&MemoryInfo> {
        self.memory_info.as_ref()
    }

    /// Get network interfaces
    pub fn network_interfaces(&self) -> &[NetworkInterface] {
        &self.network_interfaces
    }

    /// Get disks
    pub fn disks(&self) -> &[DiskInfo] {
        &self.disks
    }

    /// Get processes
    pub fn processes(&self) -> &[ProcessInfo] {
        &self.processes
    }

    /// Get CPU history
    pub fn cpu_history(&self) -> &[f32] {
        &self.cpu_history
    }

    /// Get memory history
    pub fn memory_history(&self) -> &[f32] {
        &self.mem_history
    }

    /// Get selected process
    pub fn selected_process(&self) -> Option<&ProcessInfo> {
        self.processes.get(self.selected_process)
    }

    /// Select next process
    pub fn select_next_process(&mut self) {
        if self.selected_process < self.processes.len().saturating_sub(1) {
            self.selected_process += 1;
        }
    }

    /// Select previous process
    pub fn select_prev_process(&mut self) {
        if self.selected_process > 0 {
            self.selected_process -= 1;
        }
    }

    /// Kill selected process
    pub fn kill_process(&mut self, signal: u8) -> Result<(), MonitorError> {
        if let Some(proc) = self.selected_process() {
            // Would call kill syscall here
            log::info!("Kill process {} with signal {}", proc.pid, signal);
            Ok(())
        } else {
            Err(MonitorError::NoSelection)
        }
    }

    /// Get current tab
    pub fn current_tab(&self) -> MonitorTab {
        self.current_tab
    }

    /// Set current tab
    pub fn set_tab(&mut self, tab: MonitorTab) {
        self.current_tab = tab;
    }

    /// Get config
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Get mutable config
    pub fn config_mut(&mut self) -> &mut MonitorConfig {
        &mut self.config
    }

    /// Get uptime
    pub fn uptime(&self) -> u64 {
        self.uptime
    }

    /// Get process count summary
    pub fn process_summary(&self) -> (usize, usize, usize) {
        let total = self.processes.len();
        let running = self
            .processes
            .iter()
            .filter(|p| p.state == ProcessState::Running)
            .count();
        let sleeping = self
            .processes
            .iter()
            .filter(|p| p.state == ProcessState::Sleeping)
            .count();
        (total, running, sleeping)
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitor errors
#[derive(Clone, Debug)]
pub enum MonitorError {
    NoSelection,
    PermissionDenied,
    ProcessNotFound,
}
