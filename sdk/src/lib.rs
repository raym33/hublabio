//! HubLab IO SDK
//!
//! Software Development Kit for building native HubLab IO applications.
//!
//! # Features
//!
//! - **AI Integration**: Native access to AI inference
//! - **IPC**: Simple inter-process communication
//! - **UI**: Widget toolkit for TUI applications
//! - **Storage**: File and database access
//! - **Network**: HTTP and P2P networking
//!
//! # Example
//!
//! ```rust,ignore
//! use hublabio_sdk::prelude::*;
//!
//! #[hublab_app]
//! fn main() -> Result<()> {
//!     let ai = AiClient::connect()?;
//!
//!     let response = ai.generate("Hello, world!")?;
//!     println!("{}", response);
//!
//!     Ok(())
//! }
//! ```

#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

pub mod ai;
pub mod fs;
pub mod ipc;
pub mod net;
pub mod sys;
pub mod ui;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::ai::{AiClient, AiError};
    pub use crate::fs::{File, FileSystem};
    pub use crate::ipc::{Channel, Message};
    pub use crate::sys::{ProcessInfo, System};
    pub use crate::ui::{App, View, Widget};
    pub use crate::Result;
}

/// SDK version
pub const VERSION: &str = "0.1.0";

/// Result type for SDK operations
pub type Result<T> = core::result::Result<T, Error>;

/// SDK error types
#[derive(Debug)]
pub enum Error {
    /// AI error
    Ai(ai::AiError),
    /// IPC error
    Ipc(ipc::IpcError),
    /// File system error
    Fs(fs::FsError),
    /// Network error
    Net(net::NetError),
    /// System error
    Sys(sys::SysError),
    /// Generic error
    Other(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Ai(e) => write!(f, "AI error: {:?}", e),
            Error::Ipc(e) => write!(f, "IPC error: {:?}", e),
            Error::Fs(e) => write!(f, "FS error: {:?}", e),
            Error::Net(e) => write!(f, "Network error: {:?}", e),
            Error::Sys(e) => write!(f, "System error: {:?}", e),
            Error::Other(s) => write!(f, "{}", s),
        }
    }
}

/// Application manifest
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppManifest {
    /// Application name
    pub name: String,
    /// Version string
    pub version: String,
    /// Description
    pub description: String,
    /// Author
    pub author: String,
    /// Required permissions
    pub permissions: Vec<Permission>,
    /// Entry point
    pub entry: String,
    /// Icon path
    pub icon: Option<String>,
    /// Categories
    pub categories: Vec<String>,
}

/// Application permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Permission {
    /// File system access
    FileSystem,
    /// Network access
    Network,
    /// AI inference
    AiInference,
    /// Camera access
    Camera,
    /// Microphone access
    Microphone,
    /// Location access
    Location,
    /// Bluetooth access
    Bluetooth,
    /// System settings
    SystemSettings,
    /// Background execution
    Background,
    /// Notifications
    Notifications,
}

#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(feature = "no_std")]
use alloc::string::String;
#[cfg(feature = "no_std")]
use alloc::vec::Vec;
