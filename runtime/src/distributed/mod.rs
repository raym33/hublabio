//! Distributed Inference
//!
//! Enables splitting large models across multiple devices.
//! Based on concepts from Jupiter project's distributed AI.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

pub mod cluster;
pub mod protocol;
pub mod tensor_parallel;

/// Node identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Node ID counter
static NODE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate new node ID
pub fn new_node_id() -> NodeId {
    NodeId(NODE_COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// Cluster node information
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// Unique node ID
    pub id: NodeId,
    /// Node hostname or address
    pub address: String,
    /// Port number
    pub port: u16,
    /// Total memory in bytes
    pub memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability (TFLOPS estimate)
    pub compute: f32,
    /// Node status
    pub status: NodeStatus,
    /// Assigned layer range (start, end)
    pub layers: Option<(usize, usize)>,
}

/// Node status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is available
    Available,
    /// Node is loading model
    Loading,
    /// Node is ready for inference
    Ready,
    /// Node is processing a request
    Busy,
    /// Node is offline
    Offline,
    /// Node reported an error
    Error,
}

/// Layer assignment for distributed inference
#[derive(Clone, Debug)]
pub struct LayerAssignment {
    /// Node responsible for this layer range
    pub node_id: NodeId,
    /// Start layer (inclusive)
    pub start_layer: usize,
    /// End layer (exclusive)
    pub end_layer: usize,
    /// Memory required for these layers
    pub memory_required: usize,
}

/// Distributed model configuration
#[derive(Clone, Debug)]
pub struct DistributedConfig {
    /// Model identifier
    pub model_id: String,
    /// Total number of layers
    pub num_layers: usize,
    /// Layer assignments
    pub assignments: Vec<LayerAssignment>,
    /// Pipeline parallelism degree
    pub pipeline_parallel: usize,
    /// Tensor parallelism degree
    pub tensor_parallel: usize,
    /// Overlap communication with computation
    pub overlap_comm: bool,
}

/// Cluster manager
pub struct ClusterManager {
    /// Local node ID
    local_id: NodeId,
    /// Known nodes
    nodes: BTreeMap<NodeId, NodeInfo>,
    /// Current distributed config
    config: Option<DistributedConfig>,
}

impl ClusterManager {
    /// Create a new cluster manager
    pub fn new() -> Self {
        let local_id = new_node_id();
        Self {
            local_id,
            nodes: BTreeMap::new(),
            config: None,
        }
    }

    /// Get local node ID
    pub fn local_id(&self) -> NodeId {
        self.local_id
    }

    /// Add a node to the cluster
    pub fn add_node(&mut self, info: NodeInfo) {
        self.nodes.insert(info.id, info);
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.remove(&id);
    }

    /// Get node info
    pub fn get_node(&self, id: NodeId) -> Option<&NodeInfo> {
        self.nodes.get(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &NodeInfo> {
        self.nodes.values()
    }

    /// Get total cluster memory
    pub fn total_memory(&self) -> usize {
        self.nodes.values().map(|n| n.memory).sum()
    }

    /// Get available cluster memory
    pub fn available_memory(&self) -> usize {
        self.nodes.values().map(|n| n.available_memory).sum()
    }

    /// Compute layer assignments for a model
    pub fn compute_assignments(
        &self,
        num_layers: usize,
        memory_per_layer: usize,
    ) -> Vec<LayerAssignment> {
        let mut assignments = Vec::new();

        // Collect available nodes sorted by memory
        let mut available_nodes: Vec<_> = self.nodes
            .values()
            .filter(|n| n.status == NodeStatus::Available || n.status == NodeStatus::Ready)
            .collect();

        available_nodes.sort_by_key(|n| core::cmp::Reverse(n.available_memory));

        if available_nodes.is_empty() {
            return assignments;
        }

        // Simple assignment: divide layers by memory ratio
        let total_memory: usize = available_nodes.iter().map(|n| n.available_memory).sum();
        let mut current_layer = 0;

        for node in available_nodes {
            if current_layer >= num_layers {
                break;
            }

            let memory_ratio = node.available_memory as f64 / total_memory as f64;
            let num_node_layers = ((num_layers as f64) * memory_ratio).ceil() as usize;
            let end_layer = (current_layer + num_node_layers).min(num_layers);

            if end_layer > current_layer {
                assignments.push(LayerAssignment {
                    node_id: node.id,
                    start_layer: current_layer,
                    end_layer,
                    memory_required: (end_layer - current_layer) * memory_per_layer,
                });
                current_layer = end_layer;
            }
        }

        assignments
    }

    /// Set distributed configuration
    pub fn set_config(&mut self, config: DistributedConfig) {
        self.config = Some(config);
    }

    /// Get current configuration
    pub fn config(&self) -> Option<&DistributedConfig> {
        self.config.as_ref()
    }

    /// Check if distributed inference is active
    pub fn is_distributed(&self) -> bool {
        self.config.is_some() && self.nodes.len() > 1
    }
}

impl Default for ClusterManager {
    fn default() -> Self {
        Self::new()
    }
}
