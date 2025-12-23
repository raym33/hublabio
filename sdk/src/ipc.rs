//! IPC Module
//!
//! Inter-process communication for applications.

#[cfg(feature = "no_std")]
use alloc::{string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

/// IPC error types
#[derive(Debug)]
pub enum IpcError {
    /// Channel not found
    ChannelNotFound,
    /// Connection refused
    ConnectionRefused,
    /// Message too large
    MessageTooLarge,
    /// Timeout
    Timeout,
    /// Permission denied
    PermissionDenied,
}

/// IPC channel
pub struct Channel {
    id: u64,
    connected: bool,
}

impl Channel {
    /// Connect to a named service
    pub fn connect(service_name: &str) -> Result<Self, IpcError> {
        // TODO: Connect via kernel IPC
        Ok(Self {
            id: 1,
            connected: true,
        })
    }

    /// Send a message
    pub fn send(&self, msg: &Message) -> Result<(), IpcError> {
        if !self.connected {
            return Err(IpcError::ChannelNotFound);
        }
        // TODO: Send via kernel
        Ok(())
    }

    /// Receive a message (blocking)
    pub fn receive(&self) -> Result<Message, IpcError> {
        if !self.connected {
            return Err(IpcError::ChannelNotFound);
        }
        // TODO: Receive via kernel
        Ok(Message::empty())
    }

    /// Try to receive a message (non-blocking)
    pub fn try_receive(&self) -> Result<Option<Message>, IpcError> {
        if !self.connected {
            return Err(IpcError::ChannelNotFound);
        }
        // TODO: Try receive via kernel
        Ok(None)
    }

    /// Close the channel
    pub fn close(&mut self) {
        self.connected = false;
    }

    /// Get channel ID
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// IPC message
#[derive(Debug, Clone)]
pub struct Message {
    /// Message type
    pub msg_type: u32,
    /// Payload data
    pub data: Vec<u8>,
}

impl Message {
    /// Create an empty message
    pub fn empty() -> Self {
        Self {
            msg_type: 0,
            data: Vec::new(),
        }
    }

    /// Create a new message
    pub fn new(msg_type: u32, data: Vec<u8>) -> Self {
        Self { msg_type, data }
    }

    /// Create from string
    pub fn from_string(msg_type: u32, s: &str) -> Self {
        Self {
            msg_type,
            data: s.as_bytes().to_vec(),
        }
    }

    /// Get data as string
    pub fn as_string(&self) -> Option<String> {
        String::from_utf8(self.data.clone()).ok()
    }
}
