//! AI Module
//!
//! Core AI inference functionality including NPU acceleration.

pub mod inference;
pub mod npu;
pub mod quantization;
pub mod sampling;
pub mod tokenizer;

pub use inference::*;
pub use npu::{NpuBackend, NpuError, NpuInfo, NpuManager, NpuType};
