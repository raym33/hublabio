//! HubLab IO AI Runtime
//!
//! Provides AI inference capabilities for the operating system.
//! Supports GGUF models, distributed inference, and MoE-R patterns.

#![no_std]

extern crate alloc;

pub mod ai;
pub mod distributed;
pub mod moe;

/// Runtime version
pub const VERSION: &str = "0.1.0";

/// Maximum model size in memory (2GB default)
pub const MAX_MODEL_SIZE: usize = 2 * 1024 * 1024 * 1024;

/// Maximum context length
pub const MAX_CONTEXT_LENGTH: usize = 8192;

/// Maximum batch size
pub const MAX_BATCH_SIZE: usize = 32;

/// Runtime configuration
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Maximum memory for models
    pub max_memory: usize,
    /// Number of threads for inference
    pub num_threads: usize,
    /// Enable distributed inference
    pub distributed: bool,
    /// Enable MoE-R routing
    pub moe_enabled: bool,
    /// Default generation temperature
    pub temperature: f32,
    /// Default top-p sampling
    pub top_p: f32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_memory: MAX_MODEL_SIZE,
            num_threads: 4,
            distributed: false,
            moe_enabled: false,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

/// Initialize the runtime with configuration
pub fn init(config: RuntimeConfig) {
    log::info!("HubLab IO Runtime v{} initializing...", VERSION);
    log::info!("  Max memory: {} MB", config.max_memory / (1024 * 1024));
    log::info!("  Threads: {}", config.num_threads);
    log::info!("  Distributed: {}", config.distributed);
    log::info!("  MoE-R: {}", config.moe_enabled);
}
