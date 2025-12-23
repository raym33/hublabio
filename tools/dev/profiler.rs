//! Performance Profiler
//!
//! CPU sampling profiler and performance analysis tools for HubLab IO.
//! Supports flame graphs, hot function detection, and memory profiling.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Sample rate (samples per second)
pub const DEFAULT_SAMPLE_RATE: u32 = 1000;

/// Maximum stack depth for sampling
pub const MAX_STACK_DEPTH: usize = 64;

/// CPU sample
#[derive(Clone, Debug)]
pub struct CpuSample {
    /// Timestamp (microseconds)
    pub timestamp: u64,
    /// Process ID
    pub pid: u32,
    /// Thread ID
    pub tid: u32,
    /// CPU number
    pub cpu: u32,
    /// Stack trace (addresses)
    pub stack: Vec<u64>,
    /// Was in kernel mode
    pub kernel: bool,
}

/// Memory allocation event
#[derive(Clone, Debug)]
pub struct AllocEvent {
    /// Timestamp
    pub timestamp: u64,
    /// Allocation address
    pub address: u64,
    /// Size in bytes
    pub size: usize,
    /// Is free (vs alloc)
    pub is_free: bool,
    /// Call stack
    pub stack: Vec<u64>,
}

/// Function statistics
#[derive(Clone, Debug, Default)]
pub struct FunctionStats {
    /// Function name
    pub name: String,
    /// Function address
    pub address: u64,
    /// Self samples (time in this function)
    pub self_samples: u64,
    /// Total samples (including callees)
    pub total_samples: u64,
    /// Self time percentage
    pub self_percent: f32,
    /// Total time percentage
    pub total_percent: f32,
}

/// Call graph node
#[derive(Clone, Debug)]
pub struct CallNode {
    /// Function address
    pub address: u64,
    /// Function name (if known)
    pub name: Option<String>,
    /// Call count
    pub count: u64,
    /// Children (callees)
    pub children: Vec<CallNode>,
}

/// Memory profile statistics
#[derive(Clone, Debug, Default)]
pub struct MemoryStats {
    /// Total allocations
    pub total_allocs: u64,
    /// Total frees
    pub total_frees: u64,
    /// Current allocations
    pub current_allocs: u64,
    /// Peak allocations
    pub peak_allocs: u64,
    /// Total bytes allocated
    pub total_bytes: u64,
    /// Current bytes allocated
    pub current_bytes: u64,
    /// Peak bytes allocated
    pub peak_bytes: u64,
    /// Allocation sizes histogram
    pub size_histogram: BTreeMap<usize, u64>,
}

/// Profile data
#[derive(Clone, Debug)]
pub struct ProfileData {
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp
    pub end_time: u64,
    /// Total samples collected
    pub total_samples: u64,
    /// Samples per second achieved
    pub actual_sample_rate: f32,
    /// CPU samples
    pub cpu_samples: Vec<CpuSample>,
    /// Memory events
    pub memory_events: Vec<AllocEvent>,
    /// Function statistics
    pub function_stats: Vec<FunctionStats>,
    /// Memory statistics
    pub memory_stats: MemoryStats,
}

/// Profiler state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfilerState {
    /// Not started
    Idle,
    /// Actively sampling
    Running,
    /// Paused
    Paused,
    /// Stopped, data ready
    Stopped,
}

/// Profiler configuration
#[derive(Clone, Debug)]
pub struct ProfilerConfig {
    /// CPU sampling rate
    pub sample_rate: u32,
    /// Enable CPU profiling
    pub cpu_profile: bool,
    /// Enable memory profiling
    pub memory_profile: bool,
    /// Target process (None = all)
    pub target_pid: Option<u32>,
    /// Include kernel samples
    pub include_kernel: bool,
    /// Maximum samples to collect
    pub max_samples: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            cpu_profile: true,
            memory_profile: false,
            target_pid: None,
            include_kernel: true,
            max_samples: 100_000,
        }
    }
}

/// CPU Profiler
pub struct Profiler {
    /// Configuration
    config: ProfilerConfig,
    /// Current state
    state: ProfilerState,
    /// Collected samples
    samples: Vec<CpuSample>,
    /// Memory events
    memory_events: Vec<AllocEvent>,
    /// Symbol table for address resolution
    symbols: BTreeMap<u64, String>,
    /// Start timestamp
    start_time: u64,
    /// Sample counter
    sample_count: AtomicU64,
    /// Running flag
    running: AtomicBool,
}

impl Profiler {
    /// Create new profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            state: ProfilerState::Idle,
            samples: Vec::new(),
            memory_events: Vec::new(),
            symbols: BTreeMap::new(),
            start_time: 0,
            sample_count: AtomicU64::new(0),
            running: AtomicBool::new(false),
        }
    }

    /// Start profiling
    pub fn start(&mut self) -> Result<(), ProfilerError> {
        if self.state == ProfilerState::Running {
            return Err(ProfilerError::AlreadyRunning);
        }

        self.samples.clear();
        self.memory_events.clear();
        self.sample_count.store(0, Ordering::SeqCst);
        self.start_time = self.get_time_us();
        self.state = ProfilerState::Running;
        self.running.store(true, Ordering::SeqCst);

        // TODO: Set up timer interrupt for sampling

        Ok(())
    }

    /// Stop profiling
    pub fn stop(&mut self) -> Result<ProfileData, ProfilerError> {
        if self.state != ProfilerState::Running && self.state != ProfilerState::Paused {
            return Err(ProfilerError::NotRunning);
        }

        self.running.store(false, Ordering::SeqCst);
        self.state = ProfilerState::Stopped;

        let end_time = self.get_time_us();
        let duration = (end_time - self.start_time) as f64 / 1_000_000.0;
        let total_samples = self.sample_count.load(Ordering::SeqCst);

        // Compute function statistics
        let function_stats = self.compute_function_stats();
        let memory_stats = self.compute_memory_stats();

        Ok(ProfileData {
            start_time: self.start_time,
            end_time,
            total_samples,
            actual_sample_rate: (total_samples as f64 / duration) as f32,
            cpu_samples: self.samples.clone(),
            memory_events: self.memory_events.clone(),
            function_stats,
            memory_stats,
        })
    }

    /// Pause profiling
    pub fn pause(&mut self) -> Result<(), ProfilerError> {
        if self.state != ProfilerState::Running {
            return Err(ProfilerError::NotRunning);
        }
        self.state = ProfilerState::Paused;
        Ok(())
    }

    /// Resume profiling
    pub fn resume(&mut self) -> Result<(), ProfilerError> {
        if self.state != ProfilerState::Paused {
            return Err(ProfilerError::NotPaused);
        }
        self.state = ProfilerState::Running;
        Ok(())
    }

    /// Record a CPU sample (called from timer interrupt)
    pub fn record_sample(&mut self, sample: CpuSample) {
        if !self.running.load(Ordering::SeqCst) {
            return;
        }

        if let Some(pid) = self.config.target_pid {
            if sample.pid != pid {
                return;
            }
        }

        if !self.config.include_kernel && sample.kernel {
            return;
        }

        if self.samples.len() < self.config.max_samples {
            self.samples.push(sample);
            self.sample_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Record memory allocation event
    pub fn record_alloc(&mut self, event: AllocEvent) {
        if !self.config.memory_profile || !self.running.load(Ordering::SeqCst) {
            return;
        }

        self.memory_events.push(event);
    }

    /// Load symbols for address resolution
    pub fn load_symbols(&mut self, symbols: BTreeMap<u64, String>) {
        self.symbols = symbols;
    }

    /// Add symbol
    pub fn add_symbol(&mut self, address: u64, name: &str) {
        self.symbols.insert(address, String::from(name));
    }

    /// Resolve address to symbol name
    pub fn resolve_address(&self, address: u64) -> Option<&String> {
        // Find symbol at or before address
        self.symbols.range(..=address).next_back().map(|(_, name)| name)
    }

    /// Compute function statistics from samples
    fn compute_function_stats(&self) -> Vec<FunctionStats> {
        let total_samples = self.samples.len() as u64;
        if total_samples == 0 {
            return Vec::new();
        }

        let mut self_counts: BTreeMap<u64, u64> = BTreeMap::new();
        let mut total_counts: BTreeMap<u64, u64> = BTreeMap::new();

        for sample in &self.samples {
            if let Some(&top) = sample.stack.first() {
                *self_counts.entry(top).or_insert(0) += 1;
            }

            for &addr in &sample.stack {
                *total_counts.entry(addr).or_insert(0) += 1;
            }
        }

        let mut stats: Vec<FunctionStats> = self_counts.keys().map(|&addr| {
            let self_samples = self_counts.get(&addr).copied().unwrap_or(0);
            let total = total_counts.get(&addr).copied().unwrap_or(0);

            FunctionStats {
                name: self.resolve_address(addr)
                    .cloned()
                    .unwrap_or_else(|| format!("{:#x}", addr)),
                address: addr,
                self_samples,
                total_samples: total,
                self_percent: (self_samples as f32 / total_samples as f32) * 100.0,
                total_percent: (total as f32 / total_samples as f32) * 100.0,
            }
        }).collect();

        // Sort by self samples descending
        stats.sort_by(|a, b| b.self_samples.cmp(&a.self_samples));

        stats
    }

    /// Compute memory statistics
    fn compute_memory_stats(&self) -> MemoryStats {
        let mut stats = MemoryStats::default();
        let mut current_bytes = 0u64;
        let mut allocations: BTreeMap<u64, usize> = BTreeMap::new();

        for event in &self.memory_events {
            if event.is_free {
                stats.total_frees += 1;
                if let Some(size) = allocations.remove(&event.address) {
                    current_bytes -= size as u64;
                    stats.current_allocs -= 1;
                }
            } else {
                stats.total_allocs += 1;
                stats.current_allocs += 1;
                stats.total_bytes += event.size as u64;
                current_bytes += event.size as u64;
                allocations.insert(event.address, event.size);

                // Update histogram
                let bucket = event.size.next_power_of_two();
                *stats.size_histogram.entry(bucket).or_insert(0) += 1;
            }

            stats.current_bytes = current_bytes;
            if current_bytes > stats.peak_bytes {
                stats.peak_bytes = current_bytes;
            }
            if stats.current_allocs > stats.peak_allocs {
                stats.peak_allocs = stats.current_allocs;
            }
        }

        stats
    }

    /// Generate flame graph data (folded stack format)
    pub fn flame_graph_folded(&self) -> String {
        let mut stacks: BTreeMap<String, u64> = BTreeMap::new();

        for sample in &self.samples {
            let mut stack_str = String::new();

            // Build stack string (bottom-up)
            for (i, &addr) in sample.stack.iter().rev().enumerate() {
                if i > 0 {
                    stack_str.push(';');
                }
                if let Some(name) = self.resolve_address(addr) {
                    stack_str.push_str(name);
                } else {
                    stack_str.push_str(&format!("{:#x}", addr));
                }
            }

            *stacks.entry(stack_str).or_insert(0) += 1;
        }

        let mut output = String::new();
        for (stack, count) in stacks {
            output.push_str(&format!("{} {}\n", stack, count));
        }
        output
    }

    /// Generate top functions report
    pub fn top_functions(&self, n: usize) -> String {
        let stats = self.compute_function_stats();
        let mut output = String::new();

        output.push_str("   Self%  Total%  Function\n");
        output.push_str("  ------  ------  --------\n");

        for stat in stats.iter().take(n) {
            output.push_str(&format!(
                "  {:5.1}%  {:5.1}%  {}\n",
                stat.self_percent, stat.total_percent, stat.name
            ));
        }

        output
    }

    /// Generate call graph
    pub fn call_graph(&self) -> CallNode {
        let mut root = CallNode {
            address: 0,
            name: Some(String::from("[root]")),
            count: self.samples.len() as u64,
            children: Vec::new(),
        };

        for sample in &self.samples {
            self.add_to_call_graph(&mut root, &sample.stack);
        }

        root
    }

    /// Add sample to call graph
    fn add_to_call_graph(&self, node: &mut CallNode, stack: &[u64]) {
        if stack.is_empty() {
            return;
        }

        let addr = stack[stack.len() - 1];
        let rest = &stack[..stack.len() - 1];

        // Find or create child
        if let Some(child) = node.children.iter_mut().find(|c| c.address == addr) {
            child.count += 1;
            self.add_to_call_graph(child, rest);
        } else {
            let mut child = CallNode {
                address: addr,
                name: self.resolve_address(addr).cloned(),
                count: 1,
                children: Vec::new(),
            };
            self.add_to_call_graph(&mut child, rest);
            node.children.push(child);
        }
    }

    /// Get current time in microseconds
    fn get_time_us(&self) -> u64 {
        // TODO: Get actual time
        0
    }

    /// Get profiler state
    pub fn state(&self) -> ProfilerState {
        self.state
    }

    /// Get sample count
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::SeqCst)
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new(ProfilerConfig::default())
    }
}

/// Profiler errors
#[derive(Clone, Debug)]
pub enum ProfilerError {
    /// Profiler is already running
    AlreadyRunning,
    /// Profiler is not running
    NotRunning,
    /// Profiler is not paused
    NotPaused,
    /// Buffer full
    BufferFull,
    /// Permission denied
    PermissionDenied,
}

/// Tracing profiler for detailed timing
pub struct TracingProfiler {
    /// Trace events
    events: Vec<TraceEvent>,
    /// Start time
    start_time: u64,
    /// Running
    running: bool,
}

/// Trace event
#[derive(Clone, Debug)]
pub struct TraceEvent {
    /// Event name
    pub name: String,
    /// Category
    pub category: String,
    /// Phase (B=begin, E=end, X=complete, I=instant)
    pub phase: char,
    /// Timestamp (microseconds from start)
    pub timestamp: u64,
    /// Duration (for complete events)
    pub duration: Option<u64>,
    /// Process ID
    pub pid: u32,
    /// Thread ID
    pub tid: u32,
    /// Additional arguments
    pub args: Option<String>,
}

impl TracingProfiler {
    /// Create new tracing profiler
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            start_time: 0,
            running: false,
        }
    }

    /// Start tracing
    pub fn start(&mut self) {
        self.events.clear();
        self.start_time = 0; // TODO: get time
        self.running = true;
    }

    /// Stop tracing
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Record begin event
    pub fn begin(&mut self, name: &str, category: &str) {
        if !self.running {
            return;
        }
        self.events.push(TraceEvent {
            name: String::from(name),
            category: String::from(category),
            phase: 'B',
            timestamp: 0, // TODO
            duration: None,
            pid: 0,
            tid: 0,
            args: None,
        });
    }

    /// Record end event
    pub fn end(&mut self, name: &str, category: &str) {
        if !self.running {
            return;
        }
        self.events.push(TraceEvent {
            name: String::from(name),
            category: String::from(category),
            phase: 'E',
            timestamp: 0,
            duration: None,
            pid: 0,
            tid: 0,
            args: None,
        });
    }

    /// Record instant event
    pub fn instant(&mut self, name: &str, category: &str) {
        if !self.running {
            return;
        }
        self.events.push(TraceEvent {
            name: String::from(name),
            category: String::from(category),
            phase: 'I',
            timestamp: 0,
            duration: None,
            pid: 0,
            tid: 0,
            args: None,
        });
    }

    /// Export to Chrome trace format (JSON)
    pub fn to_chrome_trace(&self) -> String {
        let mut output = String::from("{\"traceEvents\":[\n");

        for (i, event) in self.events.iter().enumerate() {
            if i > 0 {
                output.push_str(",\n");
            }
            output.push_str(&format!(
                "{{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"{}\",\"ts\":{},\"pid\":{},\"tid\":{}}}",
                event.name, event.category, event.phase, event.timestamp, event.pid, event.tid
            ));
        }

        output.push_str("\n]}");
        output
    }
}

impl Default for TracingProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::default();
        assert_eq!(profiler.state(), ProfilerState::Idle);
    }

    #[test]
    fn test_function_stats() {
        let mut profiler = Profiler::default();
        profiler.add_symbol(0x1000, "main");
        profiler.add_symbol(0x2000, "foo");

        profiler.samples.push(CpuSample {
            timestamp: 0,
            pid: 1,
            tid: 1,
            cpu: 0,
            stack: vec![0x1000],
            kernel: false,
        });

        profiler.samples.push(CpuSample {
            timestamp: 1000,
            pid: 1,
            tid: 1,
            cpu: 0,
            stack: vec![0x1000, 0x2000],
            kernel: false,
        });

        let stats = profiler.compute_function_stats();
        assert!(!stats.is_empty());
    }
}
