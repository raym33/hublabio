//! =============================================================================
//! HUBLABIO AI-ENHANCED SCHEDULER
//! =============================================================================
//! A process scheduler that uses a tiny transformer model to predict
//! optimal scheduling decisions, improving efficiency on edge devices.
//! =============================================================================

#![allow(dead_code)]

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Task identifier
pub type TaskId = u64;

/// Priority levels (0 = highest, 255 = lowest)
pub type Priority = u8;

/// CPU core identifier
pub type CoreId = u8;

/// Global task ID counter
static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

/// Task state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Ready to run
    Ready,
    /// Currently running
    Running,
    /// Blocked waiting for I/O
    Blocked,
    /// Sleeping for a duration
    Sleeping,
    /// Terminated
    Dead,
}

/// Task information
#[derive(Debug)]
pub struct Task {
    /// Unique task identifier
    pub id: TaskId,
    /// Task name
    pub name: [u8; 32],
    /// Current state
    pub state: TaskState,
    /// Base priority
    pub priority: Priority,
    /// CPU affinity mask
    pub affinity: u64,
    /// Stack pointer
    pub sp: usize,
    /// Program counter
    pub pc: usize,
    /// CPU registers (x0-x30)
    pub regs: [u64; 31],
    /// Total CPU time used (nanoseconds)
    pub cpu_time: u64,
    /// Last scheduled timestamp
    pub last_run: u64,
    /// AI prediction features
    pub features: TaskFeatures,
}

/// Features for AI prediction
#[derive(Debug, Default, Clone)]
pub struct TaskFeatures {
    /// Average CPU burst length (microseconds)
    pub avg_burst_us: f32,
    /// Memory access rate (accesses per microsecond)
    pub mem_access_rate: f32,
    /// I/O wait ratio (0.0 - 1.0)
    pub io_wait_ratio: f32,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f32,
    /// Last N burst lengths (for prediction)
    pub burst_history: [f32; 8],
    /// Number of context switches
    pub context_switches: u32,
}

/// AI prediction result
#[derive(Debug, Clone)]
pub struct TaskPrediction {
    /// Predicted next burst length (microseconds)
    pub predicted_burst_us: f32,
    /// Predicted memory pressure (0.0 - 1.0)
    pub memory_pressure: f32,
    /// Predicted I/O wait (0.0 - 1.0)
    pub io_probability: f32,
    /// Optimal core assignment
    pub optimal_core: CoreId,
    /// Scheduling priority adjustment
    pub priority_delta: i8,
    /// Confidence in prediction (0.0 - 1.0)
    pub confidence: f32,
}

/// Tiny transformer for task prediction
pub struct TinyTransformer {
    /// Embedding weights
    embed_weights: [[f32; 16]; 8],
    /// Attention weights (Q, K, V)
    attention_qkv: [[f32; 16]; 48],
    /// Feed-forward weights
    ff_weights: [[f32; 16]; 32],
    /// Output projection
    output_weights: [[f32; 6]; 16],
}

impl TinyTransformer {
    /// Create a new transformer with random weights
    /// (In production, these would be pre-trained)
    pub fn new() -> Self {
        Self {
            embed_weights: [[0.1; 16]; 8],
            attention_qkv: [[0.05; 16]; 48],
            ff_weights: [[0.02; 16]; 32],
            output_weights: [[0.1; 6]; 16],
        }
    }

    /// Load pre-trained weights
    pub fn load_weights(&mut self, data: &[u8]) {
        // Parse weight data
        // Format: magic (4 bytes) + version (4 bytes) + weights
        if data.len() < 8 {
            return;
        }

        // Simplified - real implementation would properly deserialize
    }

    /// Predict task behavior
    pub fn predict(&self, features: &TaskFeatures) -> TaskPrediction {
        // Convert features to input vector
        let input = [
            features.avg_burst_us / 1000.0,        // Normalize to ms
            features.mem_access_rate / 100.0,       // Normalize
            features.io_wait_ratio,
            features.cache_hit_rate,
            features.burst_history[0] / 1000.0,
            features.burst_history[1] / 1000.0,
            features.burst_history[2] / 1000.0,
            features.burst_history[3] / 1000.0,
        ];

        // Embedding layer
        let mut embedded = [0.0f32; 16];
        for (i, &val) in input.iter().enumerate() {
            for (j, &weight) in self.embed_weights[i].iter().enumerate() {
                embedded[j] += val * weight;
            }
        }

        // ReLU activation
        for val in embedded.iter_mut() {
            *val = val.max(0.0);
        }

        // Simplified attention (just use embedded directly)
        let attended = embedded;

        // Feed-forward layer
        let mut ff_out = [0.0f32; 16];
        for i in 0..16 {
            for j in 0..16 {
                ff_out[i] += attended[j] * self.ff_weights[i][j];
            }
            ff_out[i] = ff_out[i].max(0.0); // ReLU
        }

        // Output projection (6 outputs)
        let mut output = [0.0f32; 6];
        for i in 0..6 {
            for j in 0..16 {
                output[i] += ff_out[j] * self.output_weights[j][i];
            }
        }

        // Interpret outputs
        TaskPrediction {
            predicted_burst_us: (output[0] * 1000.0).max(0.0),
            memory_pressure: output[1].clamp(0.0, 1.0),
            io_probability: output[2].clamp(0.0, 1.0),
            optimal_core: (output[3].abs() as u8) % 4,
            priority_delta: (output[4] * 10.0) as i8,
            confidence: sigmoid(output[5]),
        }
    }
}

/// Sigmoid activation
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// AI-Enhanced Scheduler
pub struct AiScheduler {
    /// Ready queues (one per priority level)
    ready_queues: [VecDeque<TaskId>; 256],
    /// All tasks
    tasks: BTreeMap<TaskId, Task>,
    /// AI prediction model
    model: TinyTransformer,
    /// Predictions cache
    predictions: BTreeMap<TaskId, TaskPrediction>,
    /// Current running task per core
    running: [Option<TaskId>; 8],
    /// Number of CPU cores
    num_cores: usize,
    /// Scheduler statistics
    stats: SchedulerStats,
    /// Use AI predictions
    ai_enabled: bool,
}

/// Scheduler statistics
#[derive(Debug, Default)]
pub struct SchedulerStats {
    /// Total context switches
    pub context_switches: u64,
    /// Total scheduling decisions
    pub decisions: u64,
    /// AI predictions made
    pub ai_predictions: u64,
    /// Prediction accuracy (rolling average)
    pub prediction_accuracy: f32,
    /// Average wait time (microseconds)
    pub avg_wait_time_us: f32,
    /// CPU utilization per core
    pub cpu_utilization: [f32; 8],
}

impl AiScheduler {
    /// Create a new AI scheduler
    pub fn new(num_cores: usize) -> Self {
        Self {
            ready_queues: core::array::from_fn(|_| VecDeque::new()),
            tasks: BTreeMap::new(),
            model: TinyTransformer::new(),
            predictions: BTreeMap::new(),
            running: [None; 8],
            num_cores: num_cores.min(8),
            stats: SchedulerStats::default(),
            ai_enabled: true,
        }
    }

    /// Load AI model weights
    pub fn load_model(&mut self, weights: &[u8]) {
        self.model.load_weights(weights);
    }

    /// Enable/disable AI predictions
    pub fn set_ai_enabled(&mut self, enabled: bool) {
        self.ai_enabled = enabled;
    }

    /// Create a new task
    pub fn create_task(
        &mut self,
        name: &[u8],
        entry_point: usize,
        stack: usize,
        priority: Priority,
    ) -> TaskId {
        let id = NEXT_TASK_ID.fetch_add(1, Ordering::SeqCst);

        let mut task_name = [0u8; 32];
        let len = name.len().min(31);
        task_name[..len].copy_from_slice(&name[..len]);

        let task = Task {
            id,
            name: task_name,
            state: TaskState::Ready,
            priority,
            affinity: u64::MAX, // All cores
            sp: stack,
            pc: entry_point,
            regs: [0; 31],
            cpu_time: 0,
            last_run: 0,
            features: TaskFeatures::default(),
        };

        self.tasks.insert(id, task);
        self.ready_queues[priority as usize].push_back(id);

        id
    }

    /// Schedule next task for a core
    pub fn schedule(&mut self, core: CoreId) -> Option<TaskId> {
        let core = core as usize;
        if core >= self.num_cores {
            return None;
        }

        self.stats.decisions += 1;

        // Get current task (if any)
        let current = self.running[core];

        // If AI is enabled, use predictions
        if self.ai_enabled {
            self.update_predictions();
            return self.select_optimal_task(core);
        }

        // Fallback to simple priority-based scheduling
        self.select_highest_priority(core)
    }

    /// Update AI predictions for all ready tasks
    fn update_predictions(&mut self) {
        for (priority_queue) in self.ready_queues.iter() {
            for &task_id in priority_queue.iter() {
                if let Some(task) = self.tasks.get(&task_id) {
                    if !self.predictions.contains_key(&task_id) {
                        let pred = self.model.predict(&task.features);
                        self.predictions.insert(task_id, pred);
                        self.stats.ai_predictions += 1;
                    }
                }
            }
        }
    }

    /// Select optimal task using AI predictions
    fn select_optimal_task(&mut self, core: usize) -> Option<TaskId> {
        let mut best_task: Option<TaskId> = None;
        let mut best_score = f32::MIN;

        // Iterate through ready queues
        for (priority, queue) in self.ready_queues.iter().enumerate() {
            for &task_id in queue.iter() {
                let task = match self.tasks.get(&task_id) {
                    Some(t) => t,
                    None => continue,
                };

                // Check affinity
                if (task.affinity & (1 << core)) == 0 {
                    continue;
                }

                // Calculate score
                let mut score = 255.0 - priority as f32; // Base priority

                // Add AI prediction adjustments
                if let Some(pred) = self.predictions.get(&task_id) {
                    // Prefer tasks that will run on this core optimally
                    if pred.optimal_core == core as u8 {
                        score += 10.0 * pred.confidence;
                    }

                    // Prefer short bursts (better responsiveness)
                    if pred.predicted_burst_us < 1000.0 {
                        score += 5.0;
                    }

                    // Avoid I/O bound tasks if possible
                    score -= pred.io_probability * 3.0;

                    // Adjust by priority delta
                    score += pred.priority_delta as f32;
                }

                if score > best_score {
                    best_score = score;
                    best_task = Some(task_id);
                }
            }
        }

        // Remove from ready queue and mark as running
        if let Some(task_id) = best_task {
            if let Some(task) = self.tasks.get_mut(&task_id) {
                // Find and remove from ready queue
                let priority = task.priority as usize;
                if let Some(pos) = self.ready_queues[priority]
                    .iter()
                    .position(|&id| id == task_id)
                {
                    self.ready_queues[priority].remove(pos);
                }

                task.state = TaskState::Running;
                self.running[core] = Some(task_id);
                self.stats.context_switches += 1;
            }
        }

        best_task
    }

    /// Simple priority-based selection (fallback)
    fn select_highest_priority(&mut self, core: usize) -> Option<TaskId> {
        for (priority, queue) in self.ready_queues.iter_mut().enumerate() {
            while let Some(task_id) = queue.pop_front() {
                if let Some(task) = self.tasks.get_mut(&task_id) {
                    if (task.affinity & (1 << core)) != 0 {
                        task.state = TaskState::Running;
                        self.running[core] = Some(task_id);
                        self.stats.context_switches += 1;
                        return Some(task_id);
                    }
                    // Put back if wrong affinity
                    queue.push_back(task_id);
                }
            }
        }
        None
    }

    /// Yield current task
    pub fn yield_task(&mut self, core: CoreId) {
        let core = core as usize;
        if let Some(task_id) = self.running[core].take() {
            if let Some(task) = self.tasks.get_mut(&task_id) {
                task.state = TaskState::Ready;
                task.features.context_switches += 1;
                self.ready_queues[task.priority as usize].push_back(task_id);
            }
            // Clear prediction (will be recalculated)
            self.predictions.remove(&task_id);
        }
    }

    /// Block current task
    pub fn block_task(&mut self, core: CoreId) {
        let core = core as usize;
        if let Some(task_id) = self.running[core].take() {
            if let Some(task) = self.tasks.get_mut(&task_id) {
                task.state = TaskState::Blocked;

                // Update I/O wait ratio for prediction
                let ratio = &mut task.features.io_wait_ratio;
                *ratio = *ratio * 0.9 + 0.1; // Exponential moving average
            }
        }
    }

    /// Unblock a task
    pub fn unblock_task(&mut self, task_id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            if task.state == TaskState::Blocked {
                task.state = TaskState::Ready;
                self.ready_queues[task.priority as usize].push_back(task_id);
            }
        }
    }

    /// Record CPU burst completion (for learning)
    pub fn record_burst(&mut self, task_id: TaskId, burst_us: u64) {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            let features = &mut task.features;

            // Shift burst history
            for i in (1..8).rev() {
                features.burst_history[i] = features.burst_history[i - 1];
            }
            features.burst_history[0] = burst_us as f32;

            // Update average
            let alpha = 0.2;
            features.avg_burst_us = features.avg_burst_us * (1.0 - alpha)
                + burst_us as f32 * alpha;

            // Check prediction accuracy
            if let Some(pred) = self.predictions.get(&task_id) {
                let error = (pred.predicted_burst_us - burst_us as f32).abs();
                let accuracy = 1.0 - (error / burst_us as f32).min(1.0);

                self.stats.prediction_accuracy = self.stats.prediction_accuracy * 0.99
                    + accuracy * 0.01;
            }
        }

        // Invalidate prediction
        self.predictions.remove(&task_id);
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Get task by ID
    pub fn get_task(&self, id: TaskId) -> Option<&Task> {
        self.tasks.get(&id)
    }

    /// Terminate a task
    pub fn terminate_task(&mut self, task_id: TaskId) {
        // Remove from running
        for running in self.running.iter_mut() {
            if *running == Some(task_id) {
                *running = None;
            }
        }

        // Remove from ready queues
        for queue in self.ready_queues.iter_mut() {
            queue.retain(|&id| id != task_id);
        }

        // Mark as dead
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.state = TaskState::Dead;
        }

        // Remove prediction
        self.predictions.remove(&task_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = AiScheduler::new(4);
        assert_eq!(scheduler.num_cores, 4);
        assert!(scheduler.ai_enabled);
    }

    #[test]
    fn test_task_creation() {
        let mut scheduler = AiScheduler::new(4);
        let task_id = scheduler.create_task(b"test_task", 0x1000, 0x2000, 128);
        assert!(task_id > 0);
        assert!(scheduler.get_task(task_id).is_some());
    }

    #[test]
    fn test_prediction() {
        let model = TinyTransformer::new();
        let features = TaskFeatures {
            avg_burst_us: 500.0,
            mem_access_rate: 10.0,
            io_wait_ratio: 0.1,
            cache_hit_rate: 0.9,
            burst_history: [500.0, 480.0, 520.0, 490.0, 510.0, 0.0, 0.0, 0.0],
            context_switches: 10,
        };

        let pred = model.predict(&features);
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        assert!(pred.predicted_burst_us >= 0.0);
    }
}
