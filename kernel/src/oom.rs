//! Out-Of-Memory (OOM) Killer
//!
//! Handles memory pressure by killing processes to free memory.
//! Implements Linux-style badness scoring and victim selection.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::{Mutex, RwLock};

use crate::process::Pid;
use crate::signal::Signal;

/// OOM killer state
pub struct OomKiller {
    /// Whether OOM killer is enabled
    enabled: AtomicBool,
    /// Panic on OOM instead of killing
    panic_on_oom: AtomicBool,
    /// Number of OOM kills
    kill_count: AtomicU64,
    /// Last kill timestamp
    last_kill: AtomicU64,
    /// Protected processes (never kill)
    protected: RwLock<Vec<Pid>>,
    /// OOM score adjustments per process
    score_adj: RwLock<BTreeMap<Pid, i32>>,
}

impl OomKiller {
    /// Create new OOM killer
    pub const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            panic_on_oom: AtomicBool::new(false),
            kill_count: AtomicU64::new(0),
            last_kill: AtomicU64::new(0),
            protected: RwLock::new(Vec::new()),
            score_adj: RwLock::new(BTreeMap::new()),
        }
    }

    /// Enable/disable OOM killer
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    /// Set panic on OOM
    pub fn set_panic_on_oom(&self, panic: bool) {
        self.panic_on_oom.store(panic, Ordering::SeqCst);
    }

    /// Add process to protected list
    pub fn protect(&self, pid: Pid) {
        let mut protected = self.protected.write();
        if !protected.contains(&pid) {
            protected.push(pid);
        }
    }

    /// Remove process from protected list
    pub fn unprotect(&self, pid: Pid) {
        self.protected.write().retain(|p| *p != pid);
    }

    /// Set OOM score adjustment (-1000 to 1000)
    pub fn set_score_adj(&self, pid: Pid, adj: i32) {
        let adj = adj.clamp(-1000, 1000);
        self.score_adj.write().insert(pid, adj);
    }

    /// Get OOM score adjustment
    pub fn get_score_adj(&self, pid: Pid) -> i32 {
        self.score_adj.read().get(&pid).copied().unwrap_or(0)
    }

    /// Check if process is protected
    fn is_protected(&self, pid: Pid) -> bool {
        // Init process is always protected
        if pid.0 == 1 {
            return true;
        }

        // Kernel threads (PID 2 in Linux, but we don't have them)
        // Check protected list
        self.protected.read().contains(&pid)
    }

    /// Calculate badness score for a process
    fn calculate_badness(&self, pid: Pid) -> u64 {
        // Get process info
        let proc = match crate::process::get(pid) {
            Some(p) => p,
            None => return 0,
        };

        // Protected processes get 0 (never kill)
        if self.is_protected(pid) {
            return 0;
        }

        // Score adjustment of -1000 means never kill
        let adj = self.get_score_adj(pid);
        if adj == -1000 {
            return 0;
        }

        // Base score: proportion of total memory used
        let total_memory = crate::memory::total_memory();
        let proc_memory = proc.memory_usage();

        // Calculate base points (0-1000)
        let base_points = if total_memory > 0 {
            ((proc_memory as u64 * 1000) / total_memory as u64) as i64
        } else {
            0
        };

        // Apply adjustment
        let adjusted = base_points + adj as i64;

        // Clamp to valid range
        if adjusted <= 0 {
            1 // Minimum score of 1 if not protected
        } else if adjusted > 1000 {
            1000
        } else {
            adjusted as u64
        }
    }

    /// Select victim process to kill
    fn select_victim(&self) -> Option<Pid> {
        let mut best_pid: Option<Pid> = None;
        let mut best_score: u64 = 0;

        // Iterate all processes
        for proc in crate::process::all_processes() {
            let score = self.calculate_badness(proc.pid);

            if score > best_score {
                best_score = score;
                best_pid = Some(proc.pid);
            }
        }

        // Only kill if score is significant
        if best_score > 0 {
            best_pid
        } else {
            None
        }
    }

    /// Kill a process
    fn kill_process(&self, pid: Pid) -> bool {
        crate::kerror!("OOM killer: Killing process {} (badness={})",
                      pid.0, self.calculate_badness(pid));

        // Send SIGKILL
        crate::signal::send_signal(pid, Signal::Kill);

        // Update statistics
        self.kill_count.fetch_add(1, Ordering::SeqCst);
        self.last_kill.store(
            crate::time::monotonic_ns() / 1_000_000_000,
            Ordering::SeqCst
        );

        true
    }

    /// Invoke OOM killer
    pub fn invoke(&self) -> bool {
        if !self.enabled.load(Ordering::SeqCst) {
            crate::kerror!("OOM killer: Disabled, cannot free memory");
            return false;
        }

        if self.panic_on_oom.load(Ordering::SeqCst) {
            panic!("Out of memory and panic_on_oom is set");
        }

        crate::kwarn!("OOM killer invoked - selecting victim...");

        // Try to select and kill a victim
        if let Some(victim) = self.select_victim() {
            return self.kill_process(victim);
        }

        crate::kerror!("OOM killer: No suitable victim found");
        false
    }

    /// Get kill count
    pub fn kill_count(&self) -> u64 {
        self.kill_count.load(Ordering::SeqCst)
    }
}

/// Global OOM killer instance
static OOM_KILLER: OomKiller = OomKiller::new();

/// Memory pressure levels
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    /// No pressure - plenty of memory
    None = 0,
    /// Low pressure - some memory pressure, start reclaim
    Low = 1,
    /// Medium pressure - significant pressure, aggressive reclaim
    Medium = 2,
    /// Critical pressure - very low memory, consider OOM
    Critical = 3,
}

/// Memory pressure thresholds (percentage of total)
pub mod thresholds {
    /// Low pressure when free memory falls below this percentage
    pub const LOW: usize = 20;
    /// Medium pressure threshold
    pub const MEDIUM: usize = 10;
    /// Critical pressure threshold
    pub const CRITICAL: usize = 5;
}

/// Check current memory pressure level
pub fn check_pressure() -> MemoryPressure {
    let total = crate::memory::total_memory();
    let free = crate::memory::free_memory();

    if total == 0 {
        return MemoryPressure::None;
    }

    let free_percent = (free * 100) / total;

    if free_percent < thresholds::CRITICAL {
        MemoryPressure::Critical
    } else if free_percent < thresholds::MEDIUM {
        MemoryPressure::Medium
    } else if free_percent < thresholds::LOW {
        MemoryPressure::Low
    } else {
        MemoryPressure::None
    }
}

/// Handle memory pressure
pub fn handle_pressure(pressure: MemoryPressure) {
    match pressure {
        MemoryPressure::None => {
            // All good
        }
        MemoryPressure::Low => {
            // Start background reclaim
            crate::kdebug!("Memory pressure: Low - starting reclaim");
            // TODO: Trigger page cache reclaim
        }
        MemoryPressure::Medium => {
            // Aggressive reclaim
            crate::kwarn!("Memory pressure: Medium - aggressive reclaim");
            // TODO: More aggressive reclaim, shrink caches
        }
        MemoryPressure::Critical => {
            // OOM likely
            crate::kerror!("Memory pressure: Critical - may invoke OOM killer");

            // Try reclaim first
            // TODO: Synchronous reclaim

            // If still critical, invoke OOM killer
            if check_pressure() >= MemoryPressure::Critical {
                invoke_oom_killer();
            }
        }
    }
}

/// Invoke the OOM killer
pub fn invoke_oom_killer() -> bool {
    OOM_KILLER.invoke()
}

/// Set OOM score adjustment for a process
pub fn set_oom_score_adj(pid: Pid, adj: i32) {
    OOM_KILLER.set_score_adj(pid, adj);
}

/// Get OOM score for a process
pub fn get_oom_score(pid: Pid) -> u64 {
    OOM_KILLER.calculate_badness(pid)
}

/// Protect a process from OOM killer
pub fn protect_process(pid: Pid) {
    OOM_KILLER.protect(pid);
}

/// Unprotect a process
pub fn unprotect_process(pid: Pid) {
    OOM_KILLER.unprotect(pid);
}

/// Enable/disable OOM killer
pub fn set_enabled(enabled: bool) {
    OOM_KILLER.set_enabled(enabled);
}

/// Set panic on OOM
pub fn set_panic_on_oom(panic: bool) {
    OOM_KILLER.set_panic_on_oom(panic);
}

/// Clean up when process exits
pub fn cleanup_process(pid: Pid) {
    OOM_KILLER.score_adj.write().remove(&pid);
    OOM_KILLER.protected.write().retain(|p| *p != pid);
}

/// Get OOM kill statistics
pub fn get_stats() -> OomStats {
    OomStats {
        kill_count: OOM_KILLER.kill_count.load(Ordering::SeqCst),
        last_kill: OOM_KILLER.last_kill.load(Ordering::SeqCst),
        enabled: OOM_KILLER.enabled.load(Ordering::SeqCst),
        panic_on_oom: OOM_KILLER.panic_on_oom.load(Ordering::SeqCst),
    }
}

/// OOM statistics
#[derive(Clone, Debug)]
pub struct OomStats {
    pub kill_count: u64,
    pub last_kill: u64,
    pub enabled: bool,
    pub panic_on_oom: bool,
}

/// Generate /proc/sys/vm/oom_kill_allocating_task content
pub fn generate_oom_kill_allocating_task() -> String {
    // We always try to find a good victim, not necessarily the allocating task
    String::from("0")
}

/// Generate /proc/sys/vm/panic_on_oom content
pub fn generate_panic_on_oom() -> String {
    if OOM_KILLER.panic_on_oom.load(Ordering::SeqCst) {
        String::from("1")
    } else {
        String::from("0")
    }
}

/// Generate /proc/[pid]/oom_score content
pub fn generate_oom_score(pid: Pid) -> String {
    format!("{}", get_oom_score(pid))
}

/// Generate /proc/[pid]/oom_score_adj content
pub fn generate_oom_score_adj(pid: Pid) -> String {
    format!("{}", OOM_KILLER.get_score_adj(pid))
}

/// Parse and set /proc/[pid]/oom_score_adj
pub fn parse_oom_score_adj(pid: Pid, value: &str) -> Result<(), &'static str> {
    let adj: i32 = value.trim().parse().map_err(|_| "Invalid number")?;
    if adj < -1000 || adj > 1000 {
        return Err("Value must be between -1000 and 1000");
    }
    set_oom_score_adj(pid, adj);
    Ok(())
}

/// Memory allocation with OOM handling
pub fn alloc_with_oom_handling<F, T>(alloc_fn: F) -> Option<T>
where
    F: Fn() -> Option<T>,
{
    // First attempt
    if let Some(result) = alloc_fn() {
        return Some(result);
    }

    // Check pressure and possibly reclaim
    let pressure = check_pressure();
    handle_pressure(pressure);

    // Second attempt
    if let Some(result) = alloc_fn() {
        return Some(result);
    }

    // Try OOM killer
    if invoke_oom_killer() {
        // Give some time for the killed process to exit
        // In a real implementation, we'd wait for memory to be freed
        for _ in 0..10 {
            if let Some(result) = alloc_fn() {
                return Some(result);
            }
            // Small delay
            crate::arch::delay_us(1000);
        }
    }

    // Failed even after OOM killer
    None
}

/// Cgroup-aware OOM killer
pub mod cgroup {
    use super::*;

    /// Kill processes in a cgroup when it exceeds memory limit
    pub fn oom_cgroup(cgroup_path: &str) -> bool {
        let cgroup = match crate::cgroup::lookup(cgroup_path) {
            Some(cg) => cg,
            None => return false,
        };

        // Get all processes in cgroup
        let procs = cgroup.get_all_procs();
        if procs.is_empty() {
            return false;
        }

        // Find worst offender within cgroup
        let mut best_pid: Option<Pid> = None;
        let mut best_score: u64 = 0;

        for pid in procs {
            // Skip protected
            if OOM_KILLER.is_protected(pid) {
                continue;
            }

            let score = OOM_KILLER.calculate_badness(pid);
            if score > best_score {
                best_score = score;
                best_pid = Some(pid);
            }
        }

        if let Some(victim) = best_pid {
            crate::kerror!("Cgroup OOM: Killing {} in {} (badness={})",
                          victim.0, cgroup_path, best_score);

            // Check if oom_group is set
            if cgroup.memory.lock().oom_group {
                // Kill all processes in cgroup
                for pid in cgroup.get_all_procs() {
                    if !OOM_KILLER.is_protected(pid) {
                        crate::signal::send_signal(pid, Signal::Kill);
                    }
                }
                cgroup.memory.lock().events.oom_group_kill += 1;
            } else {
                crate::signal::send_signal(victim, Signal::Kill);
            }

            cgroup.memory.lock().events.oom_kill += 1;
            return true;
        }

        false
    }
}

/// Initialize OOM killer
pub fn init() {
    // Protect init process
    OOM_KILLER.protect(Pid(1));

    crate::kprintln!("  OOM killer initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_adj() {
        let killer = OomKiller::new();

        killer.set_score_adj(Pid(100), 500);
        assert_eq!(killer.get_score_adj(Pid(100)), 500);

        killer.set_score_adj(Pid(100), -1000);
        assert_eq!(killer.get_score_adj(Pid(100)), -1000);

        // Clamp test
        killer.set_score_adj(Pid(100), 2000);
        assert_eq!(killer.get_score_adj(Pid(100)), 1000);
    }

    #[test]
    fn test_protection() {
        let killer = OomKiller::new();

        killer.protect(Pid(100));
        assert!(killer.is_protected(Pid(100)));

        killer.unprotect(Pid(100));
        assert!(!killer.is_protected(Pid(100)));

        // Init is always protected
        assert!(killer.is_protected(Pid(1)));
    }
}
