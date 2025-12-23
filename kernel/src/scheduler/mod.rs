//! AI-Enhanced SMP Scheduler
//!
//! Implements an AI-assisted scheduler with SMP support that uses neural network
//! predictions to optimize process scheduling decisions across multiple CPU cores.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use spin::{Mutex, RwLock};

use crate::process::{Pid, ProcessState};
use crate::smp;

pub mod policy;
pub mod ai;

/// Time slice in microseconds (default 10ms)
const DEFAULT_TIME_SLICE: u64 = 10_000;

/// Number of priority levels
const PRIORITY_LEVELS: usize = 64;

/// Per-CPU scheduler state
#[repr(C, align(64))] // Cache-line aligned
struct PerCpuScheduler {
    /// Current running PID on this CPU
    current_pid: AtomicU64,
    /// Local run queue for this CPU (for load balancing)
    local_queue: Mutex<VecDeque<Pid>>,
    /// Number of processes in local queue
    local_count: AtomicU32,
    /// Last time load balancing was done
    last_balance: AtomicU64,
}

impl PerCpuScheduler {
    const fn new() -> Self {
        Self {
            current_pid: AtomicU64::new(0),
            local_queue: Mutex::new(VecDeque::new()),
            local_count: AtomicU32::new(0),
            last_balance: AtomicU64::new(0),
        }
    }
}

/// Per-CPU schedulers
static PER_CPU_SCHED: [PerCpuScheduler; smp::MAX_CPUS] = {
    const INIT: PerCpuScheduler = PerCpuScheduler::new();
    [INIT; smp::MAX_CPUS]
};

/// Global scheduler instance
static SCHEDULER: RwLock<Option<Scheduler>> = RwLock::new(None);

/// Current running process (legacy - use per-CPU instead)
static CURRENT_PID: AtomicU64 = AtomicU64::new(0);

/// Scheduler enabled flag
static SCHEDULER_ENABLED: AtomicBool = AtomicBool::new(false);

/// Scheduler ticks
static TICKS: AtomicU64 = AtomicU64::new(0);

/// Load balance interval (in ticks)
const LOAD_BALANCE_INTERVAL: u64 = 100;

/// Process priority
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    pub const IDLE: Self = Self(0);
    pub const LOW: Self = Self(16);
    pub const NORMAL: Self = Self(32);
    pub const HIGH: Self = Self(48);
    pub const REALTIME: Self = Self(63);

    /// AI inference priority (high but not realtime)
    pub const AI_INFERENCE: Self = Self(56);
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Scheduling policy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Normal time-sharing
    Normal,
    /// Real-time FIFO
    RealTimeFifo,
    /// Real-time round-robin
    RealTimeRoundRobin,
    /// AI workload optimized
    AiOptimized,
    /// Batch processing
    Batch,
    /// Idle (only when nothing else to run)
    Idle,
}

/// Thread scheduling information
#[derive(Clone, Debug)]
pub struct SchedInfo {
    pub pid: Pid,
    pub priority: Priority,
    pub policy: SchedulingPolicy,
    pub time_slice: u64,
    pub time_used: u64,
    pub last_run: u64,
    pub cpu_affinity: u64,  // Bitmask of allowed CPUs
    pub last_cpu: u32,       // Last CPU this process ran on
    pub preferred_cpu: Option<u32>, // Preferred CPU for cache affinity
    pub ai_predicted_runtime: Option<u64>,
    pub migrations: u64,     // Number of times migrated between CPUs
}

impl SchedInfo {
    pub fn new(pid: Pid) -> Self {
        Self {
            pid,
            priority: Priority::default(),
            policy: SchedulingPolicy::Normal,
            time_slice: DEFAULT_TIME_SLICE,
            time_used: 0,
            last_run: 0,
            cpu_affinity: !0, // All CPUs
            last_cpu: 0,
            preferred_cpu: None,
            ai_predicted_runtime: None,
            migrations: 0,
        }
    }

    /// Check if process can run on given CPU
    pub fn can_run_on(&self, cpu: u32) -> bool {
        (self.cpu_affinity & (1 << cpu)) != 0
    }
}

/// Per-priority run queue
struct RunQueue {
    queue: VecDeque<Pid>,
}

impl RunQueue {
    const fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    fn push(&mut self, pid: Pid) {
        if !self.queue.contains(&pid) {
            self.queue.push_back(pid);
        }
    }

    fn pop(&mut self) -> Option<Pid> {
        self.queue.pop_front()
    }

    fn remove(&mut self, pid: Pid) {
        self.queue.retain(|&p| p != pid);
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// The main scheduler
pub struct Scheduler {
    /// Run queues indexed by priority
    run_queues: [RunQueue; PRIORITY_LEVELS],
    /// Process scheduling info
    processes: BTreeMap<Pid, SchedInfo>,
    /// Blocked processes
    blocked: BTreeMap<Pid, BlockReason>,
    /// AI model loaded flag
    ai_enabled: bool,
    /// Statistics per CPU
    per_cpu_stats: [SchedulerStats; smp::MAX_CPUS],
    /// Global statistics
    stats: SchedulerStats,
    /// Total number of runnable processes
    runnable_count: usize,
}

/// Why a process is blocked
#[derive(Clone, Debug)]
pub enum BlockReason {
    Sleeping { until: u64 },
    WaitingForIo,
    WaitingForIpc,
    WaitingForChild,
    WaitingForMutex,
    WaitingForSemaphore,
}

/// Scheduler statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerStats {
    pub context_switches: u64,
    pub idle_ticks: u64,
    pub total_ticks: u64,
    pub ai_predictions: u64,
    pub ai_prediction_hits: u64,
    pub load_balances: u64,
    pub migrations: u64,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new() -> Self {
        Self {
            run_queues: core::array::from_fn(|_| RunQueue::new()),
            processes: BTreeMap::new(),
            blocked: BTreeMap::new(),
            ai_enabled: false,
            per_cpu_stats: [SchedulerStats::default(); smp::MAX_CPUS],
            stats: SchedulerStats::default(),
            runnable_count: 0,
        }
    }

    /// Add a process to the scheduler
    pub fn add_process(&mut self, pid: Pid, priority: Priority, policy: SchedulingPolicy) {
        let mut info = SchedInfo::new(pid);
        info.priority = priority;
        info.policy = policy;

        // AI optimization: adjust time slice based on policy
        info.time_slice = match policy {
            SchedulingPolicy::AiOptimized => DEFAULT_TIME_SLICE * 2, // Longer for AI
            SchedulingPolicy::RealTimeFifo => u64::MAX, // Run until blocked
            SchedulingPolicy::Batch => DEFAULT_TIME_SLICE * 4,
            _ => DEFAULT_TIME_SLICE,
        };

        self.processes.insert(pid, info);
        self.run_queues[priority.0 as usize].push(pid);
        self.runnable_count += 1;
    }

    /// Remove a process from the scheduler
    pub fn remove_process(&mut self, pid: Pid) {
        if let Some(info) = self.processes.remove(&pid) {
            self.run_queues[info.priority.0 as usize].remove(pid);
            if self.runnable_count > 0 {
                self.runnable_count -= 1;
            }
        }
        self.blocked.remove(&pid);
    }

    /// Block a process
    pub fn block(&mut self, pid: Pid, reason: BlockReason) {
        if let Some(info) = self.processes.get(&pid) {
            self.run_queues[info.priority.0 as usize].remove(pid);
            self.blocked.insert(pid, reason);
            if self.runnable_count > 0 {
                self.runnable_count -= 1;
            }
        }
    }

    /// Unblock a process
    pub fn unblock(&mut self, pid: Pid) {
        if self.blocked.remove(&pid).is_some() {
            if let Some(info) = self.processes.get(&pid) {
                self.run_queues[info.priority.0 as usize].push(pid);
                self.runnable_count += 1;

                // If we have multiple CPUs and more runnable than CPUs,
                // wake an idle CPU to pick up work
                let online_cpus = smp::online_count();
                if online_cpus > 1 && self.runnable_count > 1 {
                    wake_idle_cpu();
                }
            }
        }
    }

    /// Check and unblock sleeping processes
    fn check_sleeping(&mut self, current_time: u64) {
        let mut to_wake = Vec::new();

        for (pid, reason) in &self.blocked {
            if let BlockReason::Sleeping { until } = reason {
                if current_time >= *until {
                    to_wake.push(*pid);
                }
            }
        }

        for pid in to_wake {
            self.unblock(pid);
        }
    }

    /// Pick the next process to run (considers CPU affinity)
    pub fn pick_next(&mut self) -> Option<Pid> {
        let cpu = smp::cpu_id();
        self.pick_next_for_cpu(cpu)
    }

    /// Pick the next process to run for a specific CPU
    pub fn pick_next_for_cpu(&mut self, cpu: u32) -> Option<Pid> {
        // Search from highest to lowest priority
        for priority in (0..PRIORITY_LEVELS).rev() {
            // Try to find a process that can run on this CPU
            let queue = &mut self.run_queues[priority];

            // First pass: look for a process that last ran on this CPU (cache affinity)
            for i in 0..queue.queue.len() {
                let pid = queue.queue[i];
                if let Some(info) = self.processes.get(&pid) {
                    if info.can_run_on(cpu) && info.last_cpu == cpu {
                        queue.queue.remove(i);
                        if self.runnable_count > 0 {
                            self.runnable_count -= 1;
                        }
                        return Some(pid);
                    }
                }
            }

            // Second pass: any process that can run on this CPU
            for i in 0..queue.queue.len() {
                let pid = queue.queue[i];
                if let Some(info) = self.processes.get(&pid) {
                    if info.can_run_on(cpu) {
                        queue.queue.remove(i);
                        if self.runnable_count > 0 {
                            self.runnable_count -= 1;
                        }

                        // If AI is enabled, consider predictions
                        if self.ai_enabled {
                            if let Some(better_pid) = self.ai_optimize_choice(pid, priority, cpu) {
                                // Put the original back
                                queue.queue.push_back(pid);
                                self.runnable_count += 1;
                                return Some(better_pid);
                            }
                        }

                        return Some(pid);
                    }
                }
            }
        }
        None
    }

    /// Use AI to optimize scheduling decision
    fn ai_optimize_choice(&mut self, default_pid: Pid, priority: usize, cpu: u32) -> Option<Pid> {
        self.stats.ai_predictions += 1;

        // Simple heuristic for now (would use actual AI model)
        // Prefer processes with shorter predicted runtime that can run on this CPU
        let default_info = self.processes.get(&default_pid)?;
        let default_runtime = default_info.ai_predicted_runtime.unwrap_or(u64::MAX);

        for other_pid in &self.run_queues[priority].queue {
            if let Some(other_info) = self.processes.get(other_pid) {
                if !other_info.can_run_on(cpu) {
                    continue;
                }
                if let Some(other_runtime) = other_info.ai_predicted_runtime {
                    if other_runtime < default_runtime {
                        self.stats.ai_prediction_hits += 1;
                        return Some(*other_pid);
                    }
                }
            }
        }

        None
    }

    /// Perform load balancing across CPUs
    pub fn load_balance(&mut self, cpu: u32) {
        let online_cpus = smp::online_count();
        if online_cpus <= 1 {
            return;
        }

        self.stats.load_balances += 1;

        // Calculate average load
        let avg_load = self.runnable_count / online_cpus;
        if avg_load == 0 {
            return;
        }

        // Get this CPU's local queue count
        let local_count = PER_CPU_SCHED[cpu as usize].local_count.load(Ordering::Acquire) as usize;

        // If we're below average, try to pull work from busier CPUs
        if local_count < avg_load {
            let needed = avg_load - local_count;
            self.pull_tasks(cpu, needed);
        }
    }

    /// Pull tasks from busier CPUs
    fn pull_tasks(&mut self, target_cpu: u32, count: usize) {
        let mut pulled = 0;

        // Find the busiest CPU
        let mut busiest_cpu = 0u32;
        let mut busiest_count = 0u32;

        for cpu in 0..smp::MAX_CPUS as u32 {
            if cpu == target_cpu || !smp::is_online(cpu) {
                continue;
            }

            let count = PER_CPU_SCHED[cpu as usize].local_count.load(Ordering::Acquire);
            if count > busiest_count {
                busiest_count = count;
                busiest_cpu = cpu;
            }
        }

        if busiest_count <= 1 {
            return; // No work to pull
        }

        // Pull from global queue instead (simpler for now)
        // In a real implementation, we'd migrate tasks between CPUs
        for priority in (0..PRIORITY_LEVELS).rev() {
            if pulled >= count {
                break;
            }

            let queue = &mut self.run_queues[priority];
            for i in (0..queue.queue.len()).rev() {
                if pulled >= count {
                    break;
                }

                let pid = queue.queue[i];
                if let Some(info) = self.processes.get_mut(&pid) {
                    // Can this process run on target CPU?
                    if info.can_run_on(target_cpu) {
                        // Track migration
                        info.migrations += 1;
                        info.last_cpu = target_cpu;
                        self.stats.migrations += 1;
                        pulled += 1;
                    }
                }
            }
        }
    }

    /// Called on timer interrupt
    pub fn tick(&mut self) {
        self.tick_for_cpu(smp::cpu_id())
    }

    /// Called on timer interrupt for a specific CPU
    pub fn tick_for_cpu(&mut self, cpu: u32) {
        self.stats.total_ticks += 1;
        self.per_cpu_stats[cpu as usize].total_ticks += 1;

        let current_time = TICKS.load(Ordering::Acquire);
        self.check_sleeping(current_time);

        // Check if current process has exhausted its time slice
        let current = PER_CPU_SCHED[cpu as usize].current_pid.load(Ordering::Acquire);
        if current != 0 {
            if let Some(info) = self.processes.get_mut(&Pid(current)) {
                info.time_used += 1;
                info.last_run = current_time;
                info.last_cpu = cpu;

                if info.time_used >= info.time_slice {
                    // Time slice exhausted, reschedule
                    info.time_used = 0;
                    self.run_queues[info.priority.0 as usize].push(Pid(current));
                    self.runnable_count += 1;

                    // Signal need for reschedule
                    smp::current().need_resched.store(true, Ordering::Release);
                }
            }
        } else {
            self.stats.idle_ticks += 1;
            self.per_cpu_stats[cpu as usize].idle_ticks += 1;
        }

        // Periodic load balancing
        let last_balance = PER_CPU_SCHED[cpu as usize].last_balance.load(Ordering::Acquire);
        if current_time - last_balance >= LOAD_BALANCE_INTERVAL {
            PER_CPU_SCHED[cpu as usize].last_balance.store(current_time, Ordering::Release);
            self.load_balance(cpu);
        }
    }

    /// Change process priority
    pub fn set_priority(&mut self, pid: Pid, new_priority: Priority) {
        if let Some(info) = self.processes.get_mut(&pid) {
            let old_priority = info.priority;
            info.priority = new_priority;

            // Move between queues if running
            if !self.blocked.contains_key(&pid) {
                self.run_queues[old_priority.0 as usize].remove(pid);
                self.run_queues[new_priority.0 as usize].push(pid);
            }
        }
    }

    /// Enable AI-assisted scheduling
    pub fn enable_ai(&mut self) {
        self.ai_enabled = true;
        crate::kinfo!("AI-assisted scheduling enabled");
    }

    /// Get statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }
}

/// Initialize the scheduler
pub fn init() {
    let scheduler = Scheduler::new();
    *SCHEDULER.write() = Some(scheduler);

    crate::kprintln!("  Scheduler initialized ({} priority levels)", PRIORITY_LEVELS);
}

/// Load AI model for scheduling optimization
pub fn load_ai_model(addr: usize, size: usize) {
    // Parse and load the AI model
    crate::kprintln!("  Loading scheduler AI model...");

    if let Some(ref mut sched) = *SCHEDULER.write() {
        // In a real implementation, we would load the GGUF model here
        sched.enable_ai();
    }
}

/// Add a process to the scheduler
pub fn add(pid: Pid, priority: Priority) {
    if let Some(ref mut sched) = *SCHEDULER.write() {
        sched.add_process(pid, priority, SchedulingPolicy::Normal);
    }
}

/// Remove a process from the scheduler
pub fn remove(pid: Pid) {
    if let Some(ref mut sched) = *SCHEDULER.write() {
        sched.remove_process(pid);
    }
}

/// Block a process
pub fn block(pid: Pid, reason: BlockReason) {
    if let Some(ref mut sched) = *SCHEDULER.write() {
        sched.block(pid, reason);
    }
}

/// Unblock a process
pub fn unblock(pid: Pid) {
    if let Some(ref mut sched) = *SCHEDULER.write() {
        sched.unblock(pid);
    }
}

/// Wake a process (alias for unblock, commonly used)
pub fn wake(pid: Pid) {
    unblock(pid);
}

/// Sleep for a duration (in ticks)
pub fn sleep(ticks: u64) {
    let current = current_pid();
    let until = TICKS.load(Ordering::Acquire) + ticks;
    block(current, BlockReason::Sleeping { until });
    yield_now();
}

/// Yield the current process
pub fn yield_now() {
    schedule();
}

/// Get current running PID
pub fn current_pid() -> Pid {
    Pid(CURRENT_PID.load(Ordering::Acquire))
}

/// Timer tick handler
pub fn tick() {
    TICKS.fetch_add(1, Ordering::AcqRel);

    if SCHEDULER_ENABLED.load(Ordering::Acquire) {
        if let Some(ref mut sched) = *SCHEDULER.write() {
            sched.tick();
        }
    }
}

/// Pick next process and context switch
pub fn schedule() {
    if !SCHEDULER_ENABLED.load(Ordering::Acquire) {
        return;
    }

    // Disable interrupts during scheduling to prevent race conditions
    let was_enabled = crate::arch::interrupts_enabled();
    if was_enabled {
        crate::arch::disable_interrupts();
    }

    let current_pid_val = CURRENT_PID.load(Ordering::Acquire);
    let current_task = crate::task::current();

    let next_pid = {
        let mut sched = SCHEDULER.write();
        sched.as_mut().and_then(|s| {
            s.stats.context_switches += 1;
            s.pick_next()
        })
    };

    if let Some(pid) = next_pid {
        // Check if we're switching to a different process
        if pid.0 != current_pid_val {
            CURRENT_PID.store(pid.0, Ordering::Release);

            // Perform actual context switch
            if let Some(from_task) = current_task {
                if let Some(to_task) = get_task_for_pid(pid) {
                    // Re-enable interrupts after context switch returns
                    // (the new context will have its own interrupt state)
                    unsafe {
                        crate::task::context_switch(&from_task, &to_task);
                    }
                    // Restore interrupt state when we return
                    if was_enabled {
                        crate::arch::enable_interrupts();
                    }
                    return;
                }
            }

            // No from task (first run) - just set up the new task
            if let Some(to_task) = get_task_for_pid(pid) {
                crate::task::set_current(&to_task);

                // Switch to the task's address space
                let memory = to_task.process.memory.lock();
                if memory.page_table != 0 {
                    unsafe {
                        core::arch::asm!(
                            "msr ttbr0_el1, {0}",
                            "isb",
                            in(reg) memory.page_table,
                        );
                    }
                }
            }
        }
    } else {
        CURRENT_PID.store(0, Ordering::Release);
    }

    // Restore interrupt state
    if was_enabled {
        crate::arch::enable_interrupts();
    }

    // Idle - wait for interrupt (with interrupts enabled)
    if next_pid.is_none() {
        crate::arch::halt();
    }
}

/// Get task for a given PID
fn get_task_for_pid(pid: Pid) -> Option<alloc::sync::Arc<crate::task::Task>> {
    // Look up the task associated with this process
    // In our design, we use the process's main task
    if let Some(process) = crate::process::get(pid) {
        let main_tid = process.main_tid;
        return crate::task::get(crate::task::TaskId(main_tid.0));
    }
    None
}

/// Main scheduler loop (called from kernel_main, never returns)
pub fn run() -> ! {
    SCHEDULER_ENABLED.store(true, Ordering::Release);

    crate::kprintln!("[SCHED] Scheduler running");

    loop {
        schedule();
        crate::arch::halt();
    }
}

/// Get scheduler statistics
pub fn stats() -> Option<SchedulerStats> {
    SCHEDULER.read().as_ref().map(|s| s.stats.clone())
}

/// Pick next process for a specific CPU (called by secondary CPUs)
pub fn pick_next_for_cpu(cpu: u32) -> Option<Pid> {
    if let Some(ref mut sched) = *SCHEDULER.write() {
        return sched.pick_next_for_cpu(cpu);
    }
    None
}

/// Run a task on the specified CPU (called from secondary CPU idle loop)
pub fn run_task(cpu: u32) {
    if let Some(pid) = pick_next_for_cpu(cpu) {
        // Update per-CPU current PID
        PER_CPU_SCHED[cpu as usize].current_pid.store(pid.0, Ordering::Release);

        // Perform context switch
        if let Some(to_task) = get_task_for_pid(pid) {
            crate::task::set_current(&to_task);

            // Switch to the task's address space
            let memory = to_task.process.memory.lock();
            if memory.page_table != 0 {
                unsafe {
                    core::arch::asm!(
                        "msr ttbr0_el1, {0}",
                        "isb",
                        in(reg) memory.page_table,
                    );
                }
            }
            drop(memory);

            // Update scheduler stats
            if let Some(ref mut sched) = *SCHEDULER.write() {
                sched.per_cpu_stats[cpu as usize].context_switches += 1;
                sched.stats.context_switches += 1;
            }
        }
    }
}

/// Set CPU affinity for a process
pub fn set_affinity(pid: Pid, mask: u64) {
    if let Some(ref mut sched) = *SCHEDULER.write() {
        if let Some(info) = sched.processes.get_mut(&pid) {
            info.cpu_affinity = mask;
        }
    }
}

/// Get CPU affinity for a process
pub fn get_affinity(pid: Pid) -> u64 {
    if let Some(ref sched) = *SCHEDULER.read() {
        if let Some(info) = sched.processes.get(&pid) {
            return info.cpu_affinity;
        }
    }
    !0 // Default: all CPUs
}

/// Wake an idle CPU to pick up work
fn wake_idle_cpu() {
    // Find an idle CPU and send it a reschedule IPI
    for cpu in 0..smp::MAX_CPUS as u32 {
        if cpu == smp::cpu_id() {
            continue;
        }

        if let Some(pcpu) = smp::get(cpu) {
            if pcpu.get_state() == smp::CpuState::Idle {
                smp::reschedule_cpu(cpu);
                return;
            }
        }
    }
}

/// Get per-CPU scheduler stats
pub fn cpu_stats(cpu: u32) -> Option<SchedulerStats> {
    if cpu as usize >= smp::MAX_CPUS {
        return None;
    }
    SCHEDULER.read().as_ref().map(|s| s.per_cpu_stats[cpu as usize])
}

/// Get current PID for a specific CPU
pub fn current_pid_on(cpu: u32) -> Pid {
    if cpu as usize >= smp::MAX_CPUS {
        return Pid(0);
    }
    Pid(PER_CPU_SCHED[cpu as usize].current_pid.load(Ordering::Acquire))
}

/// Migrate a process to a specific CPU
pub fn migrate(pid: Pid, target_cpu: u32) -> bool {
    if target_cpu as usize >= smp::MAX_CPUS || !smp::is_online(target_cpu) {
        return false;
    }

    if let Some(ref mut sched) = *SCHEDULER.write() {
        if let Some(info) = sched.processes.get_mut(&pid) {
            // Check if process can run on target CPU
            if !info.can_run_on(target_cpu) {
                return false;
            }

            // Set preferred CPU
            info.preferred_cpu = Some(target_cpu);
            info.last_cpu = target_cpu;
            info.migrations += 1;
            sched.stats.migrations += 1;

            // If process is currently running on another CPU, send reschedule IPI
            for cpu in 0..smp::MAX_CPUS as u32 {
                if PER_CPU_SCHED[cpu as usize].current_pid.load(Ordering::Acquire) == pid.0 {
                    if cpu != target_cpu {
                        smp::reschedule_cpu(cpu);
                    }
                    break;
                }
            }

            return true;
        }
    }

    false
}

/// Get total number of runnable processes
pub fn runnable_count() -> usize {
    SCHEDULER.read().as_ref().map(|s| s.runnable_count).unwrap_or(0)
}

/// Called when a process exits - update per-CPU state if needed
pub fn on_exit(pid: Pid) {
    // Clear from any per-CPU current PID
    for cpu in 0..smp::MAX_CPUS {
        let pcpu_sched = &PER_CPU_SCHED[cpu];
        let _ = pcpu_sched.current_pid.compare_exchange(
            pid.0,
            0,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
    }
}
