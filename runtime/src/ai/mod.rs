//! AI Module
//!
//! Core AI inference functionality including NPU acceleration.

pub mod inference;
pub mod tokenizer;
pub mod sampling;
pub mod quantization;
pub mod npu;

pub use inference::*;
pub use npu::{NpuManager, NpuBackend, NpuInfo, NpuType, NpuError};
