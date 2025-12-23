//! Scheduling Policies
//!
//! Different scheduling algorithms and their implementations.

use super::{Priority, SchedulingPolicy, SchedInfo, Pid};
use alloc::vec::Vec;

/// Completely Fair Scheduler (CFS) inspired policy
pub struct CfsPolicy {
    /// Virtual runtime for each process
    vruntime: alloc::collections::BTreeMap<Pid, u64>,
    /// Weight based on priority (nice value)
    weights: alloc::collections::BTreeMap<Pid, u32>,
}

impl CfsPolicy {
    pub fn new() -> Self {
        Self {
            vruntime: alloc::collections::BTreeMap::new(),
            weights: alloc::collections::BTreeMap::new(),
        }
    }

    /// Calculate weight from priority (nice-like)
    fn priority_to_weight(priority: Priority) -> u32 {
        // Higher priority = higher weight = more CPU time
        match priority.0 {
            0..=15 => 1,
            16..=31 => 2,
            32..=47 => 4,
            48..=55 => 8,
            56..=63 => 16,
            _ => 4,
        }
    }

    /// Add a process
    pub fn add(&mut self, pid: Pid, priority: Priority) {
        let weight = Self::priority_to_weight(priority);
        self.weights.insert(pid, weight);
        self.vruntime.insert(pid, 0);
    }

    /// Remove a process
    pub fn remove(&mut self, pid: Pid) {
        self.weights.remove(&pid);
        self.vruntime.remove(&pid);
    }

    /// Update virtual runtime after running
    pub fn update_vruntime(&mut self, pid: Pid, delta: u64) {
        if let (Some(vr), Some(weight)) = (self.vruntime.get_mut(&pid), self.weights.get(&pid)) {
            // Lower weight = vruntime increases faster = less CPU time
            *vr += delta * 16 / (*weight as u64);
        }
    }

    /// Pick the process with lowest vruntime
    pub fn pick_next(&self, candidates: &[Pid]) -> Option<Pid> {
        candidates.iter()
            .filter_map(|pid| {
                self.vruntime.get(pid).map(|vr| (*pid, *vr))
            })
            .min_by_key(|(_, vr)| *vr)
            .map(|(pid, _)| pid)
    }
}

/// Real-time FIFO policy
pub struct FifoPolicy {
    queue: alloc::collections::VecDeque<Pid>,
}

impl FifoPolicy {
    pub fn new() -> Self {
        Self {
            queue: alloc::collections::VecDeque::new(),
        }
    }

    pub fn add(&mut self, pid: Pid) {
        if !self.queue.contains(&pid) {
            self.queue.push_back(pid);
        }
    }

    pub fn remove(&mut self, pid: Pid) {
        self.queue.retain(|&p| p != pid);
    }

    pub fn pick_next(&mut self) -> Option<Pid> {
        self.queue.pop_front()
    }
}

/// Round-robin policy with time slices
pub struct RoundRobinPolicy {
    queue: alloc::collections::VecDeque<Pid>,
    time_slice: u64,
}

impl RoundRobinPolicy {
    pub fn new(time_slice: u64) -> Self {
        Self {
            queue: alloc::collections::VecDeque::new(),
            time_slice,
        }
    }

    pub fn add(&mut self, pid: Pid) {
        if !self.queue.contains(&pid) {
            self.queue.push_back(pid);
        }
    }

    pub fn remove(&mut self, pid: Pid) {
        self.queue.retain(|&p| p != pid);
    }

    pub fn pick_next(&mut self) -> Option<Pid> {
        self.queue.pop_front()
    }

    pub fn requeue(&mut self, pid: Pid) {
        self.queue.push_back(pid);
    }

    pub fn time_slice(&self) -> u64 {
        self.time_slice
    }
}

/// AI-optimized scheduling policy
pub struct AiPolicy {
    /// Predicted runtimes for processes
    predictions: alloc::collections::BTreeMap<Pid, u64>,
    /// Historical data for learning
    history: Vec<(Pid, u64, u64)>, // (pid, predicted, actual)
    /// Base policy for fallback
    base_policy: CfsPolicy,
}

impl AiPolicy {
    pub fn new() -> Self {
        Self {
            predictions: alloc::collections::BTreeMap::new(),
            history: Vec::new(),
            base_policy: CfsPolicy::new(),
        }
    }

    /// Add a prediction for a process
    pub fn add_prediction(&mut self, pid: Pid, predicted_runtime: u64) {
        self.predictions.insert(pid, predicted_runtime);
    }

    /// Record actual runtime for learning
    pub fn record_actual(&mut self, pid: Pid, predicted: u64, actual: u64) {
        self.history.push((pid, predicted, actual));

        // Keep history bounded
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    /// Shortest Job First based on AI predictions
    pub fn pick_next(&self, candidates: &[Pid]) -> Option<Pid> {
        candidates.iter()
            .filter_map(|pid| {
                self.predictions.get(pid).map(|rt| (*pid, *rt))
            })
            .min_by_key(|(_, rt)| *rt)
            .map(|(pid, _)| pid)
    }

    /// Calculate prediction accuracy
    pub fn accuracy(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }

        let mut correct = 0;
        for (_, predicted, actual) in &self.history {
            // Consider correct if within 20%
            let diff = if *predicted > *actual {
                predicted - actual
            } else {
                actual - predicted
            };

            if diff <= *actual / 5 {
                correct += 1;
            }
        }

        correct as f32 / self.history.len() as f32
    }
}

/// Batch processing policy
pub struct BatchPolicy {
    queue: Vec<(Pid, Priority)>,
}

impl BatchPolicy {
    pub fn new() -> Self {
        Self {
            queue: Vec::new(),
        }
    }

    pub fn add(&mut self, pid: Pid, priority: Priority) {
        self.queue.push((pid, priority));
        self.queue.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by priority descending
    }

    pub fn remove(&mut self, pid: Pid) {
        self.queue.retain(|(p, _)| *p != pid);
    }

    pub fn pick_next(&mut self) -> Option<Pid> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0).0)
        }
    }
}
