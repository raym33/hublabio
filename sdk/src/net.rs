//! Network Module
//!
//! HTTP and P2P networking.

#[cfg(feature = "no_std")]
use alloc::{collections::BTreeMap, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{collections::BTreeMap, string::String, vec::Vec};

/// Network error
#[derive(Debug)]
pub enum NetError {
    ConnectionFailed,
    Timeout,
    InvalidUrl,
    DnsError,
    TlsError,
    IoError,
}

/// HTTP client
pub struct HttpClient;

impl HttpClient {
    /// Create a new HTTP client
    pub fn new() -> Self {
        Self
    }

    /// Perform a GET request
    pub fn get(&self, url: &str) -> Result<Response, NetError> {
        self.request(Request {
            method: Method::Get,
            url: String::from(url),
            headers: BTreeMap::new(),
            body: None,
        })
    }

    /// Perform a POST request
    pub fn post(&self, url: &str, body: &[u8]) -> Result<Response, NetError> {
        self.request(Request {
            method: Method::Post,
            url: String::from(url),
            headers: BTreeMap::new(),
            body: Some(body.to_vec()),
        })
    }

    /// Perform a request
    pub fn request(&self, request: Request) -> Result<Response, NetError> {
        // TODO: Send via network service
        Ok(Response {
            status: 200,
            headers: BTreeMap::new(),
            body: Vec::new(),
        })
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP method
#[derive(Debug, Clone, Copy)]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
}

/// HTTP request
#[derive(Debug, Clone)]
pub struct Request {
    pub method: Method,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Option<Vec<u8>>,
}

/// HTTP response
#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl Response {
    /// Get body as string
    pub fn text(&self) -> Option<String> {
        String::from_utf8(self.body.clone()).ok()
    }

    /// Check if status is success (2xx)
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }
}

/// P2P peer
pub struct Peer {
    id: String,
    address: String,
    port: u16,
}

impl Peer {
    /// Connect to a peer
    pub fn connect(address: &str, port: u16) -> Result<Self, NetError> {
        // TODO: Connect via P2P service
        Ok(Self {
            id: String::from("peer-1"),
            address: String::from(address),
            port,
        })
    }

    /// Send data to peer
    pub fn send(&self, data: &[u8]) -> Result<(), NetError> {
        // TODO: Send via P2P service
        Ok(())
    }

    /// Receive data from peer
    pub fn receive(&self) -> Result<Vec<u8>, NetError> {
        // TODO: Receive via P2P service
        Ok(Vec::new())
    }

    /// Get peer ID
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// P2P discovery
pub struct Discovery;

impl Discovery {
    /// Discover peers on local network
    pub fn discover_local() -> Result<Vec<PeerInfo>, NetError> {
        // TODO: mDNS discovery
        Ok(Vec::new())
    }
}

/// Peer information
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: String,
    pub name: String,
    pub address: String,
    pub port: u16,
    pub capabilities: Vec<String>,
}
