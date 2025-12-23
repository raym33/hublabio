//! NPU Acceleration
//!
//! Hardware acceleration for AI inference using NPU/AI accelerators.
//! Supports Raspberry Pi AI Kit (Hailo-8L), EdgeTPU, and other accelerators.

pub mod hailo;
pub mod backend;

pub use backend::*;
pub use hailo::*;

use alloc::string::String;
use alloc::vec::Vec;

/// NPU device types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuType {
    /// Hailo-8L (Raspberry Pi AI Kit)
    Hailo8L,
    /// Hailo-8
    Hailo8,
    /// Google Edge TPU
    EdgeTpu,
    /// Intel Neural Compute Stick
    Myriad,
    /// Rockchip NPU
    RockchipNpu,
    /// Custom/unknown
    Custom,
}

/// NPU device information
#[derive(Clone, Debug)]
pub struct NpuInfo {
    /// Device type
    pub device_type: NpuType,
    /// Device name
    pub name: String,
    /// Compute capability (TOPS)
    pub tops: f32,
    /// Memory size (bytes)
    pub memory: usize,
    /// Driver version
    pub driver_version: String,
    /// Firmware version
    pub firmware_version: String,
    /// Device path
    pub device_path: String,
}

/// NPU capabilities
#[derive(Clone, Debug, Default)]
pub struct NpuCapabilities {
    /// Supported data types
    pub dtypes: Vec<NpuDtype>,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum model size (bytes)
    pub max_model_size: usize,
    /// Supports dynamic shapes
    pub dynamic_shapes: bool,
    /// Supports quantized models
    pub quantized: bool,
    /// Supported quantization levels
    pub quant_levels: Vec<u8>,
}

/// NPU data types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuDtype {
    Float32,
    Float16,
    Int8,
    Uint8,
    Int16,
}

/// NPU error types
#[derive(Clone, Debug)]
pub enum NpuError {
    /// Device not found
    DeviceNotFound,
    /// Device busy
    DeviceBusy,
    /// Driver error
    DriverError(String),
    /// Model load failed
    ModelLoadFailed,
    /// Model format not supported
    UnsupportedFormat,
    /// Out of memory
    OutOfMemory,
    /// Inference error
    InferenceError(String),
    /// Timeout
    Timeout,
    /// Permission denied
    PermissionDenied,
}

/// NPU model format
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFormat {
    /// Hailo compiled model (.hef)
    Hef,
    /// TensorFlow Lite
    TfLite,
    /// ONNX
    Onnx,
    /// Custom/raw format
    Custom,
}

/// Detect available NPU devices
pub fn detect_devices() -> Vec<NpuInfo> {
    let mut devices = Vec::new();

    // Check for Hailo devices
    if let Some(hailo) = hailo::detect_hailo() {
        devices.push(hailo);
    }

    // TODO: Check for other NPU types

    devices
}

/// Get best available NPU for workload
pub fn select_best_device(required_tops: f32, model_size: usize) -> Option<NpuInfo> {
    let devices = detect_devices();

    devices.into_iter()
        .filter(|d| d.tops >= required_tops && d.memory >= model_size)
        .max_by(|a, b| a.tops.partial_cmp(&b.tops).unwrap())
}
