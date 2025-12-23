//! NPU Backend Abstraction
//!
//! Generic backend interface for different NPU accelerators.
//! Provides unified API for model loading, inference, and resource management.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::hailo::{HailoDevice, HailoModel};
use super::{ModelFormat, NpuCapabilities, NpuDtype, NpuError, NpuInfo, NpuType};

/// Inference result
#[derive(Clone, Debug)]
pub struct InferenceResult {
    /// Output tensors
    pub outputs: Vec<TensorData>,
    /// Inference time (microseconds)
    pub inference_time_us: u64,
    /// Preprocessing time (microseconds)
    pub preprocess_time_us: u64,
    /// Postprocessing time (microseconds)
    pub postprocess_time_us: u64,
}

/// Tensor data container
#[derive(Clone, Debug)]
pub struct TensorData {
    /// Tensor name
    pub name: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: NpuDtype,
    /// Raw data bytes
    pub data: Vec<u8>,
}

impl TensorData {
    /// Create new tensor
    pub fn new(name: &str, shape: Vec<usize>, dtype: NpuDtype, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            shape,
            dtype,
            data,
        }
    }

    /// Get data as f32 slice
    pub fn as_f32(&self) -> Option<Vec<f32>> {
        if self.dtype != NpuDtype::Float32 {
            return None;
        }
        Some(
            self.data
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Get data as u8 slice
    pub fn as_u8(&self) -> &[u8] {
        &self.data
    }

    /// Get total element count
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get byte size
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

/// NPU backend trait - implemented by all NPU drivers
pub trait NpuBackend: Send + Sync {
    /// Get device info
    fn info(&self) -> &NpuInfo;

    /// Get device capabilities
    fn capabilities(&self) -> NpuCapabilities;

    /// Check if device is available
    fn is_available(&self) -> bool;

    /// Load model from bytes
    fn load_model(&mut self, data: &[u8], format: ModelFormat) -> Result<ModelHandle, NpuError>;

    /// Unload model
    fn unload_model(&mut self, handle: ModelHandle) -> Result<(), NpuError>;

    /// Run inference
    fn infer(
        &mut self,
        handle: ModelHandle,
        inputs: &[TensorData],
    ) -> Result<InferenceResult, NpuError>;

    /// Get model input info
    fn get_input_info(&self, handle: ModelHandle) -> Result<Vec<TensorInfo>, NpuError>;

    /// Get model output info
    fn get_output_info(&self, handle: ModelHandle) -> Result<Vec<TensorInfo>, NpuError>;

    /// Reset device
    fn reset(&mut self) -> Result<(), NpuError>;

    /// Get statistics
    fn stats(&self) -> BackendStats;
}

/// Model handle for tracking loaded models
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelHandle(u64);

impl ModelHandle {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn id(&self) -> u64 {
        self.0
    }
}

/// Tensor metadata
#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: NpuDtype,
    /// Is quantized
    pub quantized: bool,
    /// Quantization scale (for INT8)
    pub scale: Option<f32>,
    /// Quantization zero point
    pub zero_point: Option<i32>,
}

/// Backend statistics
#[derive(Clone, Debug, Default)]
pub struct BackendStats {
    /// Total inferences
    pub total_inferences: u64,
    /// Successful inferences
    pub successful_inferences: u64,
    /// Failed inferences
    pub failed_inferences: u64,
    /// Total inference time (microseconds)
    pub total_inference_time_us: u64,
    /// Average inference time (microseconds)
    pub avg_inference_time_us: u64,
    /// Models loaded
    pub models_loaded: usize,
    /// Memory used (bytes)
    pub memory_used: usize,
}

/// Unified NPU manager
pub struct NpuManager {
    /// Available backends
    backends: Vec<Box<dyn NpuBackend>>,
    /// Active backend index
    active_backend: Option<usize>,
    /// Model counter for handles
    model_counter: AtomicU64,
    /// Is initialized
    initialized: AtomicBool,
}

impl NpuManager {
    /// Create new NPU manager
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            active_backend: None,
            model_counter: AtomicU64::new(0),
            initialized: AtomicBool::new(false),
        }
    }

    /// Initialize and detect NPU devices
    pub fn init(&mut self) -> Result<(), NpuError> {
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        // Try to open Hailo device
        if let Ok(hailo) = HailoDevice::open(super::hailo::HAILO_DEVICE_PATH) {
            self.backends.push(Box::new(HailoBackend::new(hailo)));
        }

        // TODO: Initialize other backends (EdgeTPU, etc.)

        if !self.backends.is_empty() {
            self.active_backend = Some(0);
        }

        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Get available devices
    pub fn devices(&self) -> Vec<&NpuInfo> {
        self.backends.iter().map(|b| b.info()).collect()
    }

    /// Select backend by type
    pub fn select_backend(&mut self, npu_type: NpuType) -> Result<(), NpuError> {
        for (i, backend) in self.backends.iter().enumerate() {
            if backend.info().device_type == npu_type {
                self.active_backend = Some(i);
                return Ok(());
            }
        }
        Err(NpuError::DeviceNotFound)
    }

    /// Get active backend
    pub fn active(&mut self) -> Result<&mut dyn NpuBackend, NpuError> {
        let idx = self.active_backend.ok_or(NpuError::DeviceNotFound)?;
        Ok(self.backends[idx].as_mut())
    }

    /// Load model on active backend
    pub fn load_model(
        &mut self,
        data: &[u8],
        format: ModelFormat,
    ) -> Result<ModelHandle, NpuError> {
        let backend = self.active()?;
        backend.load_model(data, format)
    }

    /// Run inference on active backend
    pub fn infer(
        &mut self,
        handle: ModelHandle,
        inputs: &[TensorData],
    ) -> Result<InferenceResult, NpuError> {
        let backend = self.active()?;
        backend.infer(handle, inputs)
    }

    /// Check if any NPU is available
    pub fn has_npu(&self) -> bool {
        !self.backends.is_empty()
    }

    /// Get total compute capability (TOPS)
    pub fn total_tops(&self) -> f32 {
        self.backends.iter().map(|b| b.info().tops).sum()
    }
}

impl Default for NpuManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Hailo backend implementation
pub struct HailoBackend {
    device: HailoDevice,
    loaded_models: Vec<(ModelHandle, HailoModel)>,
    stats: BackendStats,
}

impl HailoBackend {
    pub fn new(device: HailoDevice) -> Self {
        Self {
            device,
            loaded_models: Vec::new(),
            stats: BackendStats::default(),
        }
    }

    fn next_handle(&mut self) -> ModelHandle {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        ModelHandle::new(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    fn find_model(&self, handle: ModelHandle) -> Option<usize> {
        self.loaded_models.iter().position(|(h, _)| *h == handle)
    }
}

impl NpuBackend for HailoBackend {
    fn info(&self) -> &NpuInfo {
        self.device.info()
    }

    fn capabilities(&self) -> NpuCapabilities {
        self.device.capabilities()
    }

    fn is_available(&self) -> bool {
        self.device.is_open()
    }

    fn load_model(&mut self, data: &[u8], format: ModelFormat) -> Result<ModelHandle, NpuError> {
        if format != ModelFormat::Hef {
            return Err(NpuError::UnsupportedFormat);
        }

        let model = HailoModel::from_hef(data)?;
        let handle = self.next_handle();
        self.loaded_models.push((handle, model));
        self.stats.models_loaded = self.loaded_models.len();

        Ok(handle)
    }

    fn unload_model(&mut self, handle: ModelHandle) -> Result<(), NpuError> {
        if let Some(idx) = self.find_model(handle) {
            self.loaded_models.remove(idx);
            self.stats.models_loaded = self.loaded_models.len();
            Ok(())
        } else {
            Err(NpuError::ModelLoadFailed)
        }
    }

    fn infer(
        &mut self,
        handle: ModelHandle,
        inputs: &[TensorData],
    ) -> Result<InferenceResult, NpuError> {
        let idx = self
            .find_model(handle)
            .ok_or_else(|| NpuError::InferenceError(String::from("Model not loaded")))?;

        let (_handle, model) = &self.loaded_models[idx];

        // Load model onto device if not already loaded
        if !model.is_loaded() {
            // Clone model for device loading
            let model_clone = self.loaded_models[idx].1.network_data.clone();
            let reloaded = HailoModel::from_hef(&model_clone)?;
            self.device.load_model(reloaded)?;
        }

        // Prepare input refs
        let input_refs: Vec<&[u8]> = inputs.iter().map(|t| t.data.as_slice()).collect();

        // Run inference
        let start = 0u64; // TODO: Get actual timestamp
        let outputs = self.device.infer(&input_refs)?;
        let end = 0u64; // TODO: Get actual timestamp

        // Update stats
        self.stats.total_inferences += 1;
        self.stats.successful_inferences += 1;
        let inference_time = end.saturating_sub(start);
        self.stats.total_inference_time_us += inference_time;
        if self.stats.total_inferences > 0 {
            self.stats.avg_inference_time_us =
                self.stats.total_inference_time_us / self.stats.total_inferences;
        }

        // Convert outputs to TensorData
        let output_tensors: Vec<TensorData> = model
            .outputs()
            .iter()
            .zip(outputs.into_iter())
            .map(|(info, data)| TensorData {
                name: info.name.clone(),
                shape: info.shape.clone(),
                dtype: info.dtype,
                data,
            })
            .collect();

        Ok(InferenceResult {
            outputs: output_tensors,
            inference_time_us: inference_time,
            preprocess_time_us: 0,
            postprocess_time_us: 0,
        })
    }

    fn get_input_info(&self, handle: ModelHandle) -> Result<Vec<TensorInfo>, NpuError> {
        let idx = self
            .find_model(handle)
            .ok_or_else(|| NpuError::InferenceError(String::from("Model not loaded")))?;

        let (_, model) = &self.loaded_models[idx];
        Ok(model
            .inputs()
            .iter()
            .map(|t| TensorInfo {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype,
                quantized: t.dtype == NpuDtype::Int8 || t.dtype == NpuDtype::Uint8,
                scale: None,
                zero_point: None,
            })
            .collect())
    }

    fn get_output_info(&self, handle: ModelHandle) -> Result<Vec<TensorInfo>, NpuError> {
        let idx = self
            .find_model(handle)
            .ok_or_else(|| NpuError::InferenceError(String::from("Model not loaded")))?;

        let (_, model) = &self.loaded_models[idx];
        Ok(model
            .outputs()
            .iter()
            .map(|t| TensorInfo {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype,
                quantized: t.dtype == NpuDtype::Int8 || t.dtype == NpuDtype::Uint8,
                scale: None,
                zero_point: None,
            })
            .collect())
    }

    fn reset(&mut self) -> Result<(), NpuError> {
        self.device.unload_model();
        self.loaded_models.clear();
        self.stats = BackendStats::default();
        Ok(())
    }

    fn stats(&self) -> BackendStats {
        self.stats.clone()
    }
}

/// Image preprocessing utilities
pub struct ImagePreprocessor {
    /// Target width
    pub target_width: usize,
    /// Target height
    pub target_height: usize,
    /// Number of channels
    pub channels: usize,
    /// Normalization mean
    pub mean: [f32; 3],
    /// Normalization std
    pub std: [f32; 3],
    /// Output format (NHWC or NCHW)
    pub format: ImageFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    NHWC,
    NCHW,
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self {
            target_width: 224,
            target_height: 224,
            channels: 3,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            format: ImageFormat::NHWC,
        }
    }
}

impl ImagePreprocessor {
    /// Create preprocessor for ImageNet models
    pub fn imagenet() -> Self {
        Self::default()
    }

    /// Create preprocessor for YOLO models
    pub fn yolo(size: usize) -> Self {
        Self {
            target_width: size,
            target_height: size,
            channels: 3,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            format: ImageFormat::NCHW,
        }
    }

    /// Preprocess image to tensor
    pub fn preprocess(&self, image: &[u8], width: usize, height: usize) -> TensorData {
        let mut output = vec![0u8; self.target_width * self.target_height * self.channels];

        // Simple nearest-neighbor resize (TODO: bilinear interpolation)
        for y in 0..self.target_height {
            for x in 0..self.target_width {
                let src_x = (x * width) / self.target_width;
                let src_y = (y * height) / self.target_height;
                let src_idx = (src_y * width + src_x) * self.channels;
                let dst_idx = (y * self.target_width + x) * self.channels;

                for c in 0..self.channels {
                    if src_idx + c < image.len() {
                        // Apply normalization: (pixel / 255 - mean) / std
                        let pixel = image[src_idx + c] as f32 / 255.0;
                        let normalized = (pixel - self.mean[c]) / self.std[c];
                        // Convert back to uint8 range [0, 255]
                        let value = ((normalized + 2.0) * 64.0).clamp(0.0, 255.0) as u8;
                        output[dst_idx + c] = value;
                    }
                }
            }
        }

        TensorData::new(
            "input",
            vec![1, self.target_height, self.target_width, self.channels],
            NpuDtype::Uint8,
            output,
        )
    }
}

/// Classification postprocessor
pub struct ClassificationPostprocessor {
    /// Number of classes
    pub num_classes: usize,
    /// Top-k results
    pub top_k: usize,
    /// Class labels (optional)
    pub labels: Option<Vec<String>>,
}

impl ClassificationPostprocessor {
    pub fn new(num_classes: usize, top_k: usize) -> Self {
        Self {
            num_classes,
            top_k,
            labels: None,
        }
    }

    /// Set class labels
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Postprocess classification output
    pub fn postprocess(&self, output: &TensorData) -> Vec<ClassificationResult> {
        let scores = output.as_f32().unwrap_or_default();

        // Get top-k indices
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        indexed
            .into_iter()
            .take(self.top_k)
            .map(|(idx, score)| {
                let label = self.labels.as_ref().and_then(|l| l.get(idx)).cloned();
                ClassificationResult {
                    class_id: idx,
                    score,
                    label,
                }
            })
            .collect()
    }
}

/// Classification result
#[derive(Clone, Debug)]
pub struct ClassificationResult {
    /// Class index
    pub class_id: usize,
    /// Confidence score
    pub score: f32,
    /// Class label (if available)
    pub label: Option<String>,
}

/// Detection postprocessor for YOLO-style outputs
pub struct DetectionPostprocessor {
    /// Input image width
    pub input_width: usize,
    /// Input image height
    pub input_height: usize,
    /// Confidence threshold
    pub conf_threshold: f32,
    /// NMS IoU threshold
    pub nms_threshold: f32,
    /// Number of classes
    pub num_classes: usize,
}

impl DetectionPostprocessor {
    pub fn new(input_width: usize, input_height: usize, num_classes: usize) -> Self {
        Self {
            input_width,
            input_height,
            conf_threshold: 0.25,
            nms_threshold: 0.45,
            num_classes,
        }
    }

    /// Postprocess detection output
    pub fn postprocess(&self, output: &TensorData) -> Vec<DetectionResult> {
        let mut detections = Vec::new();

        // Parse YOLO output format
        // Output shape: [1, num_detections, 5 + num_classes]
        // Format: [x, y, w, h, obj_conf, class_probs...]

        let data = output.as_f32().unwrap_or_default();
        let detection_size = 5 + self.num_classes;
        let num_detections = data.len() / detection_size;

        for i in 0..num_detections {
            let base = i * detection_size;
            if base + detection_size > data.len() {
                break;
            }

            let x = data[base];
            let y = data[base + 1];
            let w = data[base + 2];
            let h = data[base + 3];
            let obj_conf = data[base + 4];

            if obj_conf < self.conf_threshold {
                continue;
            }

            // Find best class
            let mut best_class = 0;
            let mut best_prob = 0.0f32;
            for c in 0..self.num_classes {
                let prob = data[base + 5 + c];
                if prob > best_prob {
                    best_prob = prob;
                    best_class = c;
                }
            }

            let confidence = obj_conf * best_prob;
            if confidence < self.conf_threshold {
                continue;
            }

            // Convert to corner format (x1, y1, x2, y2)
            let x1 = (x - w / 2.0).clamp(0.0, 1.0);
            let y1 = (y - h / 2.0).clamp(0.0, 1.0);
            let x2 = (x + w / 2.0).clamp(0.0, 1.0);
            let y2 = (y + h / 2.0).clamp(0.0, 1.0);

            detections.push(DetectionResult {
                bbox: BoundingBox { x1, y1, x2, y2 },
                class_id: best_class,
                confidence,
                label: None,
            });
        }

        // Apply NMS
        self.nms(&mut detections);

        detections
    }

    /// Non-maximum suppression
    fn nms(&self, detections: &mut Vec<DetectionResult>) {
        // Sort by confidence
        detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut keep = vec![true; detections.len()];

        for i in 0..detections.len() {
            if !keep[i] {
                continue;
            }

            for j in (i + 1)..detections.len() {
                if !keep[j] {
                    continue;
                }

                if detections[i].class_id == detections[j].class_id {
                    let iou = detections[i].bbox.iou(&detections[j].bbox);
                    if iou > self.nms_threshold {
                        keep[j] = false;
                    }
                }
            }
        }

        let mut idx = 0;
        detections.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
    }
}

/// Detection result
#[derive(Clone, Debug)]
pub struct DetectionResult {
    /// Bounding box (normalized coordinates)
    pub bbox: BoundingBox,
    /// Class index
    pub class_id: usize,
    /// Confidence score
    pub confidence: f32,
    /// Class label (if available)
    pub label: Option<String>,
}

/// Bounding box
#[derive(Clone, Copy, Debug)]
pub struct BoundingBox {
    /// Top-left x (normalized)
    pub x1: f32,
    /// Top-left y (normalized)
    pub y1: f32,
    /// Bottom-right x (normalized)
    pub x2: f32,
    /// Bottom-right y (normalized)
    pub y2: f32,
}

impl BoundingBox {
    /// Calculate area
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    /// Calculate IoU with another box
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);

        if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
            return 0.0;
        }

        let inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
        let union_area = self.area() + other.area() - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }

    /// Convert to pixel coordinates
    pub fn to_pixels(&self, width: usize, height: usize) -> (usize, usize, usize, usize) {
        (
            (self.x1 * width as f32) as usize,
            (self.y1 * height as f32) as usize,
            (self.x2 * width as f32) as usize,
            (self.y2 * height as f32) as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = TensorData::new("test", vec![2, 2], NpuDtype::Float32, bytes);

        assert_eq!(tensor.element_count(), 4);
        let f32_data = tensor.as_f32().unwrap();
        assert_eq!(f32_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_bounding_box_iou() {
        let box1 = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.5,
            y2: 0.5,
        };
        let box2 = BoundingBox {
            x1: 0.25,
            y1: 0.25,
            x2: 0.75,
            y2: 0.75,
        };

        let iou = box1.iou(&box2);
        // Intersection: 0.25 * 0.25 = 0.0625
        // Union: 0.25 + 0.25 - 0.0625 = 0.4375
        // IoU: 0.0625 / 0.4375 ≈ 0.143
        assert!(iou > 0.14 && iou < 0.15);
    }

    #[test]
    fn test_npu_manager() {
        let manager = NpuManager::new();
        assert!(!manager.has_npu()); // No devices in test environment
    }
}
