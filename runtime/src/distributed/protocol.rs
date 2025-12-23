//! Distributed Inference Protocol
//!
//! Binary protocol for communication between cluster nodes during
//! distributed AI inference. Handles activation passing, synchronization,
//! and model coordination.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use super::NodeId;

/// Protocol version
pub const PROTOCOL_VERSION: u8 = 1;

/// Magic bytes for protocol identification
pub const PROTOCOL_MAGIC: [u8; 4] = [0x48, 0x4C, 0x49, 0x4F]; // "HLIO"

/// Maximum message size (16MB)
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Message types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Ping/pong for latency measurement
    Ping = 0x01,
    Pong = 0x02,

    /// Model distribution
    LoadModel = 0x10,
    ModelLoaded = 0x11,
    UnloadModel = 0x12,

    /// Layer assignment
    AssignLayers = 0x20,
    LayersReady = 0x21,

    /// Inference requests
    InferenceRequest = 0x30,
    InferenceResponse = 0x31,

    /// Activation transfer
    SendActivations = 0x40,
    ActivationsReceived = 0x41,

    /// Synchronization
    SyncRequest = 0x50,
    SyncResponse = 0x51,
    Barrier = 0x52,
    BarrierAck = 0x53,

    /// Error handling
    Error = 0xF0,

    /// Unknown/invalid
    Unknown = 0xFF,
}

impl From<u8> for MessageType {
    fn from(v: u8) -> Self {
        match v {
            0x01 => MessageType::Ping,
            0x02 => MessageType::Pong,
            0x10 => MessageType::LoadModel,
            0x11 => MessageType::ModelLoaded,
            0x12 => MessageType::UnloadModel,
            0x20 => MessageType::AssignLayers,
            0x21 => MessageType::LayersReady,
            0x30 => MessageType::InferenceRequest,
            0x31 => MessageType::InferenceResponse,
            0x40 => MessageType::SendActivations,
            0x41 => MessageType::ActivationsReceived,
            0x50 => MessageType::SyncRequest,
            0x51 => MessageType::SyncResponse,
            0x52 => MessageType::Barrier,
            0x53 => MessageType::BarrierAck,
            0xF0 => MessageType::Error,
            _ => MessageType::Unknown,
        }
    }
}

/// Message header (16 bytes)
#[derive(Clone, Debug)]
pub struct MessageHeader {
    /// Protocol magic
    pub magic: [u8; 4],
    /// Protocol version
    pub version: u8,
    /// Message type
    pub msg_type: MessageType,
    /// Reserved for future use
    pub reserved: [u8; 2],
    /// Sender node ID
    pub sender: NodeId,
    /// Payload length
    pub payload_len: u32,
}

impl MessageHeader {
    /// Create new header
    pub fn new(msg_type: MessageType, sender: NodeId, payload_len: u32) -> Self {
        Self {
            magic: PROTOCOL_MAGIC,
            version: PROTOCOL_VERSION,
            msg_type,
            reserved: [0, 0],
            sender,
            payload_len,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version;
        bytes[5] = self.msg_type as u8;
        bytes[6..8].copy_from_slice(&self.reserved);
        bytes[8..16].copy_from_slice(&self.sender.0.to_le_bytes());
        // Note: payload_len is implicitly known from message content
        bytes
    }

    /// Parse header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }

        let magic: [u8; 4] = bytes[0..4].try_into().ok()?;
        if magic != PROTOCOL_MAGIC {
            return None;
        }

        let version = bytes[4];
        if version != PROTOCOL_VERSION {
            return None;
        }

        let msg_type = MessageType::from(bytes[5]);
        let reserved: [u8; 2] = bytes[6..8].try_into().ok()?;
        let sender_bytes: [u8; 8] = bytes[8..16].try_into().ok()?;
        let sender = NodeId(u64::from_le_bytes(sender_bytes));

        Some(Self {
            magic,
            version,
            msg_type,
            reserved,
            sender,
            payload_len: 0, // Will be set when reading payload
        })
    }

    /// Header size in bytes
    pub const fn size() -> usize {
        16
    }
}

/// Activation data for transfer between nodes
#[derive(Clone, Debug)]
pub struct ActivationData {
    /// Request ID for tracking
    pub request_id: u64,
    /// Source layer index
    pub layer_idx: u32,
    /// Batch size
    pub batch_size: u32,
    /// Sequence length
    pub seq_len: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Data type (0=f32, 1=f16, 2=bf16)
    pub dtype: u8,
    /// Compressed flag
    pub compressed: bool,
    /// Activation tensor data
    pub data: Vec<u8>,
}

impl ActivationData {
    /// Serialize activation data
    pub fn serialize(&self) -> Vec<u8> {
        let header_size = 8 + 4 + 4 + 4 + 4 + 1 + 1 + 4; // 30 bytes header
        let mut bytes = Vec::with_capacity(header_size + self.data.len());

        bytes.extend_from_slice(&self.request_id.to_le_bytes());
        bytes.extend_from_slice(&self.layer_idx.to_le_bytes());
        bytes.extend_from_slice(&self.batch_size.to_le_bytes());
        bytes.extend_from_slice(&self.seq_len.to_le_bytes());
        bytes.extend_from_slice(&self.hidden_dim.to_le_bytes());
        bytes.push(self.dtype);
        bytes.push(self.compressed as u8);
        bytes.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.data);

        bytes
    }

    /// Deserialize activation data
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 30 {
            return None;
        }

        let request_id = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let layer_idx = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let batch_size = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
        let seq_len = u32::from_le_bytes(bytes[16..20].try_into().ok()?);
        let hidden_dim = u32::from_le_bytes(bytes[20..24].try_into().ok()?);
        let dtype = bytes[24];
        let compressed = bytes[25] != 0;
        let data_len = u32::from_le_bytes(bytes[26..30].try_into().ok()?) as usize;

        if bytes.len() < 30 + data_len {
            return None;
        }

        let data = bytes[30..30 + data_len].to_vec();

        Some(Self {
            request_id,
            layer_idx,
            batch_size,
            seq_len,
            hidden_dim,
            dtype,
            compressed,
            data,
        })
    }

    /// Calculate data size in bytes
    pub fn data_size(&self) -> usize {
        let element_size = match self.dtype {
            0 => 4, // f32
            1 => 2, // f16
            2 => 2, // bf16
            _ => 4,
        };
        (self.batch_size as usize)
            * (self.seq_len as usize)
            * (self.hidden_dim as usize)
            * element_size
    }
}

/// Inference request
#[derive(Clone, Debug)]
pub struct InferenceRequest {
    /// Unique request ID
    pub request_id: u64,
    /// Model ID
    pub model_id: String,
    /// Input token IDs
    pub tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
}

impl InferenceRequest {
    /// Serialize request
    pub fn serialize(&self) -> Vec<u8> {
        let model_bytes = self.model_id.as_bytes();
        let tokens_len = self.tokens.len() * 4;
        let size = 8 + 4 + model_bytes.len() + 4 + tokens_len + 4 + 4 + 4;

        let mut bytes = Vec::with_capacity(size);

        bytes.extend_from_slice(&self.request_id.to_le_bytes());
        bytes.extend_from_slice(&(model_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(model_bytes);
        bytes.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for token in &self.tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        bytes.extend_from_slice(&self.max_tokens.to_le_bytes());
        bytes.extend_from_slice(&self.temperature.to_le_bytes());
        bytes.extend_from_slice(&self.top_p.to_le_bytes());

        bytes
    }

    /// Deserialize request
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }

        let request_id = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let model_len = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;

        if bytes.len() < 12 + model_len + 4 {
            return None;
        }

        let model_id = String::from_utf8(bytes[12..12 + model_len].to_vec()).ok()?;
        let offset = 12 + model_len;

        let tokens_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let offset = offset + 4;

        if bytes.len() < offset + tokens_len * 4 + 12 {
            return None;
        }

        let mut tokens = Vec::with_capacity(tokens_len);
        for i in 0..tokens_len {
            let idx = offset + i * 4;
            let token = u32::from_le_bytes(bytes[idx..idx + 4].try_into().ok()?);
            tokens.push(token);
        }

        let offset = offset + tokens_len * 4;
        let max_tokens = u32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
        let temperature = f32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?);
        let top_p = f32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().ok()?);

        Some(Self {
            request_id,
            model_id,
            tokens,
            max_tokens,
            temperature,
            top_p,
        })
    }
}

/// Inference response
#[derive(Clone, Debug)]
pub struct InferenceResponse {
    /// Request ID (matches request)
    pub request_id: u64,
    /// Generated token IDs
    pub tokens: Vec<u32>,
    /// Total time in microseconds
    pub time_us: u64,
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Success flag
    pub success: bool,
    /// Error message (if not success)
    pub error: Option<String>,
}

impl InferenceResponse {
    /// Serialize response
    pub fn serialize(&self) -> Vec<u8> {
        let tokens_len = self.tokens.len() * 4;
        let error_bytes = self
            .error
            .as_ref()
            .map(|e| e.as_bytes().to_vec())
            .unwrap_or_default();

        let size = 8 + 4 + tokens_len + 8 + 4 + 1 + 4 + error_bytes.len();
        let mut bytes = Vec::with_capacity(size);

        bytes.extend_from_slice(&self.request_id.to_le_bytes());
        bytes.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for token in &self.tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        bytes.extend_from_slice(&self.time_us.to_le_bytes());
        bytes.extend_from_slice(&self.tokens_per_second.to_le_bytes());
        bytes.push(self.success as u8);
        bytes.extend_from_slice(&(error_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&error_bytes);

        bytes
    }

    /// Deserialize response
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }

        let request_id = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        let tokens_len = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;

        let offset = 12;
        if bytes.len() < offset + tokens_len * 4 + 17 {
            return None;
        }

        let mut tokens = Vec::with_capacity(tokens_len);
        for i in 0..tokens_len {
            let idx = offset + i * 4;
            let token = u32::from_le_bytes(bytes[idx..idx + 4].try_into().ok()?);
            tokens.push(token);
        }

        let offset = offset + tokens_len * 4;
        let time_us = u64::from_le_bytes(bytes[offset..offset + 8].try_into().ok()?);
        let tokens_per_second = f32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().ok()?);
        let success = bytes[offset + 12] != 0;
        let error_len =
            u32::from_le_bytes(bytes[offset + 13..offset + 17].try_into().ok()?) as usize;

        let error = if error_len > 0 && bytes.len() >= offset + 17 + error_len {
            String::from_utf8(bytes[offset + 17..offset + 17 + error_len].to_vec()).ok()
        } else {
            None
        };

        Some(Self {
            request_id,
            tokens,
            time_us,
            tokens_per_second,
            success,
            error,
        })
    }
}

/// Layer assignment message
#[derive(Clone, Debug)]
pub struct LayerAssignment {
    /// Model ID
    pub model_id: String,
    /// Start layer (inclusive)
    pub start_layer: u32,
    /// End layer (exclusive)
    pub end_layer: u32,
    /// Memory required
    pub memory_required: u64,
}

impl LayerAssignment {
    /// Serialize assignment
    pub fn serialize(&self) -> Vec<u8> {
        let model_bytes = self.model_id.as_bytes();
        let size = 4 + model_bytes.len() + 4 + 4 + 8;

        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(&(model_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(model_bytes);
        bytes.extend_from_slice(&self.start_layer.to_le_bytes());
        bytes.extend_from_slice(&self.end_layer.to_le_bytes());
        bytes.extend_from_slice(&self.memory_required.to_le_bytes());

        bytes
    }

    /// Deserialize assignment
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }

        let model_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;

        if bytes.len() < 4 + model_len + 16 {
            return None;
        }

        let model_id = String::from_utf8(bytes[4..4 + model_len].to_vec()).ok()?;
        let offset = 4 + model_len;

        let start_layer = u32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
        let end_layer = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?);
        let memory_required = u64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().ok()?);

        Some(Self {
            model_id,
            start_layer,
            end_layer,
            memory_required,
        })
    }
}

/// Protocol error types
#[derive(Clone, Debug)]
pub enum ProtocolError {
    /// Invalid magic bytes
    InvalidMagic,
    /// Version mismatch
    VersionMismatch,
    /// Message too large
    MessageTooLarge,
    /// Invalid message type
    InvalidMessageType,
    /// Deserialization failed
    DeserializationError,
    /// Connection error
    ConnectionError(String),
    /// Timeout
    Timeout,
}

/// Connection state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not connected
    Disconnected,
    /// Connection in progress
    Connecting,
    /// Connected and ready
    Connected,
    /// Connection error
    Error,
}

/// Protocol connection
pub struct ProtocolConnection {
    /// Remote node ID
    pub remote_id: Option<NodeId>,
    /// Connection state
    pub state: ConnectionState,
    /// Remote address
    pub address: String,
    /// Remote port
    pub port: u16,
    /// Pending request ID
    pub pending_request: Option<u64>,
    /// Round-trip time in microseconds
    pub rtt_us: u64,
}

impl ProtocolConnection {
    /// Create new connection
    pub fn new(address: &str, port: u16) -> Self {
        Self {
            remote_id: None,
            state: ConnectionState::Disconnected,
            address: String::from(address),
            port,
            pending_request: None,
            rtt_us: 0,
        }
    }

    /// Mark as connected
    pub fn mark_connected(&mut self, remote_id: NodeId) {
        self.remote_id = Some(remote_id);
        self.state = ConnectionState::Connected;
    }

    /// Mark as error
    pub fn mark_error(&mut self) {
        self.state = ConnectionState::Error;
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.state == ConnectionState::Connected
    }
}

/// Request tracker for managing in-flight requests
pub struct RequestTracker {
    /// Next request ID
    next_id: u64,
    /// In-flight requests: request_id -> (sent_time, target_node)
    in_flight: Vec<(u64, u64, NodeId)>,
    /// Request timeout in microseconds
    timeout_us: u64,
}

impl RequestTracker {
    /// Create new tracker
    pub fn new(timeout_us: u64) -> Self {
        Self {
            next_id: 1,
            in_flight: Vec::new(),
            timeout_us,
        }
    }

    /// Generate new request ID
    pub fn new_request(&mut self, target: NodeId, current_time: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.in_flight.push((id, current_time, target));
        id
    }

    /// Mark request as complete
    pub fn complete(&mut self, request_id: u64) -> Option<u64> {
        if let Some(idx) = self
            .in_flight
            .iter()
            .position(|(id, _, _)| *id == request_id)
        {
            let (_, sent_time, _) = self.in_flight.remove(idx);
            return Some(sent_time);
        }
        None
    }

    /// Check for timed out requests
    pub fn check_timeouts(&mut self, current_time: u64) -> Vec<(u64, NodeId)> {
        let mut timed_out = Vec::new();

        self.in_flight.retain(|(id, sent_time, target)| {
            if current_time - sent_time > self.timeout_us {
                timed_out.push((*id, *target));
                false
            } else {
                true
            }
        });

        timed_out
    }

    /// Get number of in-flight requests
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = MessageHeader::new(MessageType::Ping, NodeId(42), 100);
        let bytes = header.to_bytes();
        let parsed = MessageHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.msg_type, MessageType::Ping);
        assert_eq!(parsed.sender.0, 42);
    }

    #[test]
    fn test_activation_roundtrip() {
        let act = ActivationData {
            request_id: 1,
            layer_idx: 5,
            batch_size: 1,
            seq_len: 128,
            hidden_dim: 2048,
            dtype: 0,
            compressed: false,
            data: vec![0u8; 1024],
        };

        let bytes = act.serialize();
        let parsed = ActivationData::deserialize(&bytes).unwrap();

        assert_eq!(parsed.request_id, 1);
        assert_eq!(parsed.layer_idx, 5);
        assert_eq!(parsed.data.len(), 1024);
    }

    #[test]
    fn test_inference_request_roundtrip() {
        let req = InferenceRequest {
            request_id: 123,
            model_id: String::from("test-model"),
            tokens: vec![1, 2, 3, 4, 5],
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
        };

        let bytes = req.serialize();
        let parsed = InferenceRequest::deserialize(&bytes).unwrap();

        assert_eq!(parsed.request_id, 123);
        assert_eq!(parsed.model_id, "test-model");
        assert_eq!(parsed.tokens, vec![1, 2, 3, 4, 5]);
    }
}
