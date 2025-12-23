//! Cluster Discovery and Management
//!
//! Implements mDNS-based peer discovery, heartbeat monitoring,
//! and automatic cluster formation for distributed AI inference.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::{new_node_id, NodeId, NodeInfo, NodeStatus};

/// Service type for mDNS discovery
pub const MDNS_SERVICE_TYPE: &str = "_hublabio._tcp.local";

/// Default cluster port
pub const DEFAULT_PORT: u16 = 5000;

/// Heartbeat interval in milliseconds
pub const HEARTBEAT_INTERVAL_MS: u64 = 5000;

/// Node timeout in milliseconds (3 missed heartbeats)
pub const NODE_TIMEOUT_MS: u64 = 15000;

/// Discovery message types
#[derive(Clone, Debug)]
pub enum DiscoveryMessage {
    /// Announce presence on network
    Announce {
        node_id: NodeId,
        address: String,
        port: u16,
        memory: usize,
        compute: f32,
    },
    /// Heartbeat to maintain presence
    Heartbeat {
        node_id: NodeId,
        status: NodeStatus,
        available_memory: usize,
        load: f32,
    },
    /// Query for other nodes
    Query { from: NodeId },
    /// Response to query
    QueryResponse { nodes: Vec<NodeInfo> },
    /// Node leaving cluster
    Leave { node_id: NodeId },
}

/// Cluster discovery service
pub struct ClusterDiscovery {
    /// Local node information
    local_node: NodeInfo,
    /// Discovered peers
    peers: BTreeMap<NodeId, PeerInfo>,
    /// Last heartbeat timestamps
    last_seen: BTreeMap<NodeId, u64>,
    /// Discovery enabled
    enabled: AtomicBool,
    /// Cluster name
    cluster_name: String,
    /// Discovery mode
    mode: DiscoveryMode,
}

/// Peer information with connection state
#[derive(Clone, Debug)]
pub struct PeerInfo {
    /// Node information
    pub info: NodeInfo,
    /// Connection established
    pub connected: bool,
    /// Latency in microseconds
    pub latency_us: u64,
    /// Last successful communication
    pub last_success: u64,
    /// Error count
    pub error_count: u32,
}

/// Discovery modes
#[derive(Clone, Debug)]
pub enum DiscoveryMode {
    /// Automatic discovery via mDNS
    Auto,
    /// Static list of peers
    Static(Vec<String>),
    /// Manual peer addition only
    Manual,
}

impl ClusterDiscovery {
    /// Create new cluster discovery service
    pub fn new(cluster_name: &str, mode: DiscoveryMode) -> Self {
        let local_id = new_node_id();

        Self {
            local_node: NodeInfo {
                id: local_id,
                address: String::from("0.0.0.0"),
                port: DEFAULT_PORT,
                memory: 0,
                available_memory: 0,
                compute: 0.0,
                status: NodeStatus::Available,
                layers: None,
            },
            peers: BTreeMap::new(),
            last_seen: BTreeMap::new(),
            enabled: AtomicBool::new(false),
            cluster_name: String::from(cluster_name),
            mode,
        }
    }

    /// Set local node information
    pub fn set_local_info(&mut self, address: &str, port: u16, memory: usize, compute: f32) {
        self.local_node.address = String::from(address);
        self.local_node.port = port;
        self.local_node.memory = memory;
        self.local_node.available_memory = memory;
        self.local_node.compute = compute;
    }

    /// Start discovery service
    pub fn start(&self) {
        self.enabled.store(true, Ordering::SeqCst);
    }

    /// Stop discovery service
    pub fn stop(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    /// Check if discovery is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Get local node ID
    pub fn local_id(&self) -> NodeId {
        self.local_node.id
    }

    /// Get local node info
    pub fn local_info(&self) -> &NodeInfo {
        &self.local_node
    }

    /// Create announce message
    pub fn create_announce(&self) -> DiscoveryMessage {
        DiscoveryMessage::Announce {
            node_id: self.local_node.id,
            address: self.local_node.address.clone(),
            port: self.local_node.port,
            memory: self.local_node.memory,
            compute: self.local_node.compute,
        }
    }

    /// Create heartbeat message
    pub fn create_heartbeat(&self, load: f32) -> DiscoveryMessage {
        DiscoveryMessage::Heartbeat {
            node_id: self.local_node.id,
            status: self.local_node.status,
            available_memory: self.local_node.available_memory,
            load,
        }
    }

    /// Process received discovery message
    pub fn process_message(
        &mut self,
        msg: DiscoveryMessage,
        current_time: u64,
    ) -> Option<DiscoveryMessage> {
        match msg {
            DiscoveryMessage::Announce {
                node_id,
                address,
                port,
                memory,
                compute,
            } => {
                if node_id != self.local_node.id {
                    let info = NodeInfo {
                        id: node_id,
                        address,
                        port,
                        memory,
                        available_memory: memory,
                        compute,
                        status: NodeStatus::Available,
                        layers: None,
                    };

                    self.peers.insert(
                        node_id,
                        PeerInfo {
                            info,
                            connected: false,
                            latency_us: 0,
                            last_success: current_time,
                            error_count: 0,
                        },
                    );
                    self.last_seen.insert(node_id, current_time);

                    // Respond with our announce
                    return Some(self.create_announce());
                }
            }

            DiscoveryMessage::Heartbeat {
                node_id,
                status,
                available_memory,
                load: _,
            } => {
                if let Some(peer) = self.peers.get_mut(&node_id) {
                    peer.info.status = status;
                    peer.info.available_memory = available_memory;
                    peer.last_success = current_time;
                    self.last_seen.insert(node_id, current_time);
                }
            }

            DiscoveryMessage::Query { from } => {
                if from != self.local_node.id {
                    let nodes: Vec<NodeInfo> =
                        self.peers.values().map(|p| p.info.clone()).collect();
                    return Some(DiscoveryMessage::QueryResponse { nodes });
                }
            }

            DiscoveryMessage::QueryResponse { nodes } => {
                for info in nodes {
                    if info.id != self.local_node.id && !self.peers.contains_key(&info.id) {
                        self.peers.insert(
                            info.id,
                            PeerInfo {
                                info,
                                connected: false,
                                latency_us: 0,
                                last_success: current_time,
                                error_count: 0,
                            },
                        );
                    }
                }
            }

            DiscoveryMessage::Leave { node_id } => {
                self.peers.remove(&node_id);
                self.last_seen.remove(&node_id);
            }
        }

        None
    }

    /// Check for timed out nodes
    pub fn check_timeouts(&mut self, current_time: u64) -> Vec<NodeId> {
        let mut timed_out = Vec::new();

        for (&node_id, &last_seen) in &self.last_seen {
            if current_time - last_seen > NODE_TIMEOUT_MS {
                timed_out.push(node_id);
            }
        }

        for node_id in &timed_out {
            self.peers.remove(node_id);
            self.last_seen.remove(node_id);
        }

        timed_out
    }

    /// Get all discovered peers
    pub fn peers(&self) -> impl Iterator<Item = &PeerInfo> {
        self.peers.values()
    }

    /// Get peer by ID
    pub fn get_peer(&self, id: NodeId) -> Option<&PeerInfo> {
        self.peers.get(&id)
    }

    /// Get peer count
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Mark peer as connected
    pub fn mark_connected(&mut self, id: NodeId, latency_us: u64) {
        if let Some(peer) = self.peers.get_mut(&id) {
            peer.connected = true;
            peer.latency_us = latency_us;
            peer.error_count = 0;
        }
    }

    /// Mark peer connection error
    pub fn mark_error(&mut self, id: NodeId) {
        if let Some(peer) = self.peers.get_mut(&id) {
            peer.error_count += 1;
            if peer.error_count >= 3 {
                peer.connected = false;
            }
        }
    }

    /// Get total cluster memory
    pub fn cluster_memory(&self) -> usize {
        let peer_memory: usize = self
            .peers
            .values()
            .filter(|p| p.connected)
            .map(|p| p.info.memory)
            .sum();
        self.local_node.memory + peer_memory
    }

    /// Get total cluster compute (TFLOPS)
    pub fn cluster_compute(&self) -> f32 {
        let peer_compute: f32 = self
            .peers
            .values()
            .filter(|p| p.connected)
            .map(|p| p.info.compute)
            .sum();
        self.local_node.compute + peer_compute
    }

    /// Add static peer
    pub fn add_static_peer(&mut self, address: &str, port: u16) {
        let id = new_node_id();
        let info = NodeInfo {
            id,
            address: String::from(address),
            port,
            memory: 0,
            available_memory: 0,
            compute: 0.0,
            status: NodeStatus::Available,
            layers: None,
        };

        self.peers.insert(
            id,
            PeerInfo {
                info,
                connected: false,
                latency_us: 0,
                last_success: 0,
                error_count: 0,
            },
        );
    }

    /// Update local available memory
    pub fn update_available_memory(&mut self, available: usize) {
        self.local_node.available_memory = available;
    }

    /// Update local status
    pub fn update_status(&mut self, status: NodeStatus) {
        self.local_node.status = status;
    }
}

/// mDNS packet builder for service discovery
pub struct MdnsBuilder {
    buffer: Vec<u8>,
}

impl MdnsBuilder {
    /// Create new mDNS builder
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(512),
        }
    }

    /// Build service announcement packet
    pub fn build_announcement(&mut self, hostname: &str, port: u16) -> &[u8] {
        self.buffer.clear();

        // Transaction ID (2 bytes)
        self.buffer.extend_from_slice(&[0x00, 0x00]);

        // Flags: Standard response, authoritative (2 bytes)
        self.buffer.extend_from_slice(&[0x84, 0x00]);

        // Questions, Answers, Authority, Additional (2 bytes each)
        self.buffer.extend_from_slice(&[0x00, 0x00]); // Questions
        self.buffer.extend_from_slice(&[0x00, 0x01]); // Answers
        self.buffer.extend_from_slice(&[0x00, 0x00]); // Authority
        self.buffer.extend_from_slice(&[0x00, 0x00]); // Additional

        // Answer section: SRV record
        self.write_name(hostname);
        self.buffer.extend_from_slice(&[0x00, 0x21]); // Type: SRV
        self.buffer.extend_from_slice(&[0x80, 0x01]); // Class: IN, cache flush
        self.buffer.extend_from_slice(&[0x00, 0x00, 0x00, 0x78]); // TTL: 120s

        // SRV data length (placeholder)
        let len_pos = self.buffer.len();
        self.buffer.extend_from_slice(&[0x00, 0x00]);

        // SRV data: priority, weight, port, target
        self.buffer.extend_from_slice(&[0x00, 0x00]); // Priority
        self.buffer.extend_from_slice(&[0x00, 0x00]); // Weight
        self.buffer.extend_from_slice(&port.to_be_bytes()); // Port

        let data_start = self.buffer.len();
        self.write_name(hostname);

        // Update data length
        let data_len = (self.buffer.len() - data_start + 6) as u16;
        self.buffer[len_pos] = (data_len >> 8) as u8;
        self.buffer[len_pos + 1] = data_len as u8;

        &self.buffer
    }

    /// Build service query packet
    pub fn build_query(&mut self) -> &[u8] {
        self.buffer.clear();

        // Transaction ID
        self.buffer.extend_from_slice(&[0x00, 0x00]);

        // Flags: Standard query
        self.buffer.extend_from_slice(&[0x00, 0x00]);

        // Questions, Answers, Authority, Additional
        self.buffer.extend_from_slice(&[0x00, 0x01]); // 1 Question
        self.buffer.extend_from_slice(&[0x00, 0x00]);
        self.buffer.extend_from_slice(&[0x00, 0x00]);
        self.buffer.extend_from_slice(&[0x00, 0x00]);

        // Query: _hublabio._tcp.local
        self.write_name(MDNS_SERVICE_TYPE);
        self.buffer.extend_from_slice(&[0x00, 0x0C]); // Type: PTR
        self.buffer.extend_from_slice(&[0x00, 0x01]); // Class: IN

        &self.buffer
    }

    /// Write DNS name encoding
    fn write_name(&mut self, name: &str) {
        for part in name.split('.') {
            let len = part.len() as u8;
            self.buffer.push(len);
            self.buffer.extend_from_slice(part.as_bytes());
        }
        self.buffer.push(0x00); // Null terminator
    }
}

impl Default for MdnsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster health monitor
pub struct ClusterHealth {
    /// Health check interval in ms
    pub interval_ms: u64,
    /// Last check timestamp
    pub last_check: u64,
    /// Healthy node threshold
    pub healthy_threshold: f32,
}

impl ClusterHealth {
    /// Create new health monitor
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            last_check: 0,
            healthy_threshold: 0.75,
        }
    }

    /// Check if health check is due
    pub fn is_check_due(&self, current_time: u64) -> bool {
        current_time - self.last_check >= self.interval_ms
    }

    /// Update last check time
    pub fn mark_checked(&mut self, current_time: u64) {
        self.last_check = current_time;
    }

    /// Calculate cluster health score
    pub fn calculate_score(&self, discovery: &ClusterDiscovery) -> f32 {
        let total = discovery.peer_count();
        if total == 0 {
            return 1.0; // Single node is healthy
        }

        let connected: usize = discovery
            .peers()
            .filter(|p| p.connected && p.info.status != NodeStatus::Error)
            .count();

        connected as f32 / total as f32
    }

    /// Check if cluster is healthy
    pub fn is_healthy(&self, discovery: &ClusterDiscovery) -> bool {
        self.calculate_score(discovery) >= self.healthy_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_creation() {
        let discovery = ClusterDiscovery::new("test", DiscoveryMode::Manual);
        assert!(!discovery.is_enabled());
        assert_eq!(discovery.peer_count(), 0);
    }

    #[test]
    fn test_announce_processing() {
        let mut discovery = ClusterDiscovery::new("test", DiscoveryMode::Manual);
        let other_id = NodeId(999);

        let msg = DiscoveryMessage::Announce {
            node_id: other_id,
            address: String::from("192.168.1.100"),
            port: 5000,
            memory: 4 * 1024 * 1024 * 1024,
            compute: 2.0,
        };

        discovery.process_message(msg, 0);

        assert_eq!(discovery.peer_count(), 1);
        assert!(discovery.get_peer(other_id).is_some());
    }

    #[test]
    fn test_timeout_detection() {
        let mut discovery = ClusterDiscovery::new("test", DiscoveryMode::Manual);

        let msg = DiscoveryMessage::Announce {
            node_id: NodeId(999),
            address: String::from("192.168.1.100"),
            port: 5000,
            memory: 4 * 1024 * 1024 * 1024,
            compute: 2.0,
        };

        discovery.process_message(msg, 0);

        // Check timeout after NODE_TIMEOUT_MS + 1
        let timed_out = discovery.check_timeouts(NODE_TIMEOUT_MS + 1);
        assert_eq!(timed_out.len(), 1);
        assert_eq!(discovery.peer_count(), 0);
    }
}
