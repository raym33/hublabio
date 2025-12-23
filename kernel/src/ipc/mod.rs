//! Inter-Process Communication (IPC)
//!
//! Provides message passing and shared memory primitives for the microkernel.
//! All service communication happens through these IPC mechanisms.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use alloc::string::String;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::{Mutex, RwLock};

/// Maximum message size (64 KB)
pub const MAX_MESSAGE_SIZE: usize = 64 * 1024;

/// Maximum pending messages per channel
pub const MAX_PENDING_MESSAGES: usize = 256;

/// Channel ID counter
static CHANNEL_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Global channel registry
static CHANNELS: RwLock<BTreeMap<ChannelId, Arc<Channel>>> = RwLock::new(BTreeMap::new());

/// Named endpoints for service discovery
static ENDPOINTS: RwLock<BTreeMap<String, ChannelId>> = RwLock::new(BTreeMap::new());

/// Channel identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChannelId(pub u64);

/// Process identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessId(pub u64);

/// Message header
#[derive(Clone, Debug)]
pub struct MessageHeader {
    /// Message type/operation code
    pub msg_type: u32,
    /// Flags (async, priority, etc.)
    pub flags: u32,
    /// Sender process ID
    pub sender: ProcessId,
    /// Message sequence number
    pub seq: u64,
    /// Payload length
    pub payload_len: usize,
}

/// Complete message
#[derive(Clone)]
pub struct Message {
    pub header: MessageHeader,
    pub payload: Vec<u8>,
}

/// Channel endpoint (one side of a bidirectional channel)
pub struct ChannelEndpoint {
    /// Which side of the channel this is
    side: ChannelSide,
    /// Reference to the channel
    channel: Arc<Channel>,
}

/// Which side of the channel
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChannelSide {
    A,
    B,
}

/// Bidirectional communication channel
pub struct Channel {
    id: ChannelId,
    /// Messages from A to B
    queue_a_to_b: Mutex<Vec<Message>>,
    /// Messages from B to A
    queue_b_to_a: Mutex<Vec<Message>>,
    /// Sequence counter
    seq_counter: AtomicU64,
    /// Is side A connected
    connected_a: Mutex<bool>,
    /// Is side B connected
    connected_b: Mutex<bool>,
}

impl Channel {
    /// Create a new channel
    fn new(id: ChannelId) -> Self {
        Self {
            id,
            queue_a_to_b: Mutex::new(Vec::new()),
            queue_b_to_a: Mutex::new(Vec::new()),
            seq_counter: AtomicU64::new(0),
            connected_a: Mutex::new(true),
            connected_b: Mutex::new(true),
        }
    }

    /// Send a message
    fn send(&self, side: ChannelSide, msg: Message) -> Result<(), IpcError> {
        let queue = match side {
            ChannelSide::A => &self.queue_a_to_b,
            ChannelSide::B => &self.queue_b_to_a,
        };

        let mut queue = queue.lock();

        if queue.len() >= MAX_PENDING_MESSAGES {
            return Err(IpcError::QueueFull);
        }

        if msg.payload.len() > MAX_MESSAGE_SIZE {
            return Err(IpcError::MessageTooLarge);
        }

        queue.push(msg);

        // TODO: Wake up waiting receiver

        Ok(())
    }

    /// Receive a message (non-blocking)
    fn try_receive(&self, side: ChannelSide) -> Option<Message> {
        let queue = match side {
            ChannelSide::A => &self.queue_b_to_a,
            ChannelSide::B => &self.queue_a_to_b,
        };

        let mut queue = queue.lock();
        if queue.is_empty() {
            None
        } else {
            Some(queue.remove(0))
        }
    }

    /// Get next sequence number
    fn next_seq(&self) -> u64 {
        self.seq_counter.fetch_add(1, Ordering::SeqCst)
    }
}

impl ChannelEndpoint {
    /// Send a message
    pub fn send(&self, msg_type: u32, payload: &[u8]) -> Result<(), IpcError> {
        let header = MessageHeader {
            msg_type,
            flags: 0,
            sender: ProcessId(0), // TODO: Get current process ID
            seq: self.channel.next_seq(),
            payload_len: payload.len(),
        };

        let msg = Message {
            header,
            payload: payload.to_vec(),
        };

        self.channel.send(self.side, msg)
    }

    /// Send with flags
    pub fn send_with_flags(&self, msg_type: u32, flags: u32, payload: &[u8]) -> Result<(), IpcError> {
        let header = MessageHeader {
            msg_type,
            flags,
            sender: ProcessId(0),
            seq: self.channel.next_seq(),
            payload_len: payload.len(),
        };

        let msg = Message {
            header,
            payload: payload.to_vec(),
        };

        self.channel.send(self.side, msg)
    }

    /// Try to receive a message (non-blocking)
    pub fn try_receive(&self) -> Option<Message> {
        self.channel.try_receive(self.side)
    }

    /// Get channel ID
    pub fn channel_id(&self) -> ChannelId {
        self.channel.id
    }
}

/// IPC errors
#[derive(Debug)]
pub enum IpcError {
    /// Message queue is full
    QueueFull,
    /// Message too large
    MessageTooLarge,
    /// Channel not found
    ChannelNotFound,
    /// Endpoint not connected
    NotConnected,
    /// Invalid operation
    InvalidOperation,
    /// Permission denied
    PermissionDenied,
}

/// Initialize IPC subsystem
pub fn init() {
    crate::kprintln!("  IPC channels initialized");
}

/// Create a new channel pair
pub fn create_channel() -> (ChannelEndpoint, ChannelEndpoint) {
    let id = ChannelId(CHANNEL_COUNTER.fetch_add(1, Ordering::SeqCst));
    let channel = Arc::new(Channel::new(id));

    CHANNELS.write().insert(id, channel.clone());

    let endpoint_a = ChannelEndpoint {
        side: ChannelSide::A,
        channel: channel.clone(),
    };

    let endpoint_b = ChannelEndpoint {
        side: ChannelSide::B,
        channel,
    };

    (endpoint_a, endpoint_b)
}

/// Register a named endpoint for service discovery
pub fn register_endpoint(name: &str, channel_id: ChannelId) -> Result<(), IpcError> {
    let mut endpoints = ENDPOINTS.write();

    if endpoints.contains_key(name) {
        return Err(IpcError::InvalidOperation);
    }

    endpoints.insert(String::from(name), channel_id);
    Ok(())
}

/// Look up a named endpoint
pub fn lookup_endpoint(name: &str) -> Option<ChannelId> {
    ENDPOINTS.read().get(name).copied()
}

/// Standard service names
pub mod services {
    /// AI inference service
    pub const AI_SERVICE: &str = "io.hublab.ai";
    /// File system service
    pub const FS_SERVICE: &str = "io.hublab.fs";
    /// Network service
    pub const NET_SERVICE: &str = "io.hublab.net";
    /// Display service
    pub const DISPLAY_SERVICE: &str = "io.hublab.display";
    /// Input service
    pub const INPUT_SERVICE: &str = "io.hublab.input";
    /// Audio service
    pub const AUDIO_SERVICE: &str = "io.hublab.audio";
    /// Power management service
    pub const POWER_SERVICE: &str = "io.hublab.power";
}

/// Message types for system services
pub mod msg_types {
    // AI Service
    pub const AI_LOAD_MODEL: u32 = 0x0100;
    pub const AI_GENERATE: u32 = 0x0101;
    pub const AI_TOKENIZE: u32 = 0x0102;
    pub const AI_EMBED: u32 = 0x0103;

    // File System
    pub const FS_OPEN: u32 = 0x0200;
    pub const FS_READ: u32 = 0x0201;
    pub const FS_WRITE: u32 = 0x0202;
    pub const FS_CLOSE: u32 = 0x0203;
    pub const FS_STAT: u32 = 0x0204;

    // Network
    pub const NET_CONNECT: u32 = 0x0300;
    pub const NET_SEND: u32 = 0x0301;
    pub const NET_RECV: u32 = 0x0302;
    pub const NET_CLOSE: u32 = 0x0303;

    // Display
    pub const DISP_CREATE_SURFACE: u32 = 0x0400;
    pub const DISP_UPDATE: u32 = 0x0401;
    pub const DISP_PRESENT: u32 = 0x0402;

    // Generic
    pub const REPLY_OK: u32 = 0xFFFF_0000;
    pub const REPLY_ERROR: u32 = 0xFFFF_0001;
}
