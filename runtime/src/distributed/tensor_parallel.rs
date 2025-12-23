//! Tensor Parallelism for Distributed Inference
//!
//! Implements tensor splitting and merging for parallel execution
//! across multiple devices. Supports column and row parallelism
//! for attention and feed-forward layers.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp;

/// Parallelism strategy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelismStrategy {
    /// Split along columns (output features)
    ColumnParallel,
    /// Split along rows (input features)
    RowParallel,
    /// Split along attention heads
    HeadParallel,
    /// No parallelism (full tensor on each device)
    Replicated,
}

/// Tensor partition information
#[derive(Clone, Debug)]
pub struct TensorPartition {
    /// Global tensor shape
    pub global_shape: Vec<usize>,
    /// Local partition shape
    pub local_shape: Vec<usize>,
    /// Partition index (0..num_partitions-1)
    pub partition_idx: usize,
    /// Total number of partitions
    pub num_partitions: usize,
    /// Parallelism strategy used
    pub strategy: ParallelismStrategy,
    /// Dimension being split
    pub split_dim: usize,
}

impl TensorPartition {
    /// Create column-parallel partition
    pub fn column_parallel(shape: &[usize], rank: usize, world_size: usize) -> Self {
        assert!(
            shape.len() >= 2,
            "Need at least 2D tensor for column parallelism"
        );
        let split_dim = shape.len() - 1; // Last dimension
        let global_cols = shape[split_dim];

        let cols_per_rank = (global_cols + world_size - 1) / world_size;
        let start_col = rank * cols_per_rank;
        let end_col = cmp::min(start_col + cols_per_rank, global_cols);
        let local_cols = end_col - start_col;

        let mut local_shape = shape.to_vec();
        local_shape[split_dim] = local_cols;

        Self {
            global_shape: shape.to_vec(),
            local_shape,
            partition_idx: rank,
            num_partitions: world_size,
            strategy: ParallelismStrategy::ColumnParallel,
            split_dim,
        }
    }

    /// Create row-parallel partition
    pub fn row_parallel(shape: &[usize], rank: usize, world_size: usize) -> Self {
        assert!(
            shape.len() >= 2,
            "Need at least 2D tensor for row parallelism"
        );
        let split_dim = shape.len() - 2; // Second-to-last dimension
        let global_rows = shape[split_dim];

        let rows_per_rank = (global_rows + world_size - 1) / world_size;
        let start_row = rank * rows_per_rank;
        let end_row = cmp::min(start_row + rows_per_rank, global_rows);
        let local_rows = end_row - start_row;

        let mut local_shape = shape.to_vec();
        local_shape[split_dim] = local_rows;

        Self {
            global_shape: shape.to_vec(),
            local_shape,
            partition_idx: rank,
            num_partitions: world_size,
            strategy: ParallelismStrategy::RowParallel,
            split_dim,
        }
    }

    /// Create head-parallel partition for attention
    pub fn head_parallel(
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Self {
        let heads_per_rank = (num_heads + world_size - 1) / world_size;
        let start_head = rank * heads_per_rank;
        let end_head = cmp::min(start_head + heads_per_rank, num_heads);
        let local_heads = end_head - start_head;

        Self {
            global_shape: vec![batch, seq_len, num_heads, head_dim],
            local_shape: vec![batch, seq_len, local_heads, head_dim],
            partition_idx: rank,
            num_partitions: world_size,
            strategy: ParallelismStrategy::HeadParallel,
            split_dim: 2, // Split along head dimension
        }
    }

    /// Create replicated partition (no splitting)
    pub fn replicated(shape: &[usize], rank: usize, world_size: usize) -> Self {
        Self {
            global_shape: shape.to_vec(),
            local_shape: shape.to_vec(),
            partition_idx: rank,
            num_partitions: world_size,
            strategy: ParallelismStrategy::Replicated,
            split_dim: 0,
        }
    }

    /// Get local element count
    pub fn local_numel(&self) -> usize {
        self.local_shape.iter().product()
    }

    /// Get global element count
    pub fn global_numel(&self) -> usize {
        self.global_shape.iter().product()
    }

    /// Get offset in global tensor for this partition
    pub fn global_offset(&self) -> usize {
        if self.strategy == ParallelismStrategy::Replicated {
            return 0;
        }

        let dim_size = self.global_shape[self.split_dim];
        let chunk_size = (dim_size + self.num_partitions - 1) / self.num_partitions;
        let offset_in_dim = self.partition_idx * chunk_size;

        // Calculate linear offset
        let mut offset = offset_in_dim;
        let mut stride = 1;

        for i in (self.split_dim + 1..self.global_shape.len()).rev() {
            stride *= self.global_shape[i];
        }

        offset * stride
    }
}

/// Tensor splitter for distributing tensors across devices
pub struct TensorSplitter {
    /// World size (number of devices)
    world_size: usize,
    /// Local rank
    rank: usize,
}

impl TensorSplitter {
    /// Create new tensor splitter
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { world_size, rank }
    }

    /// Split tensor along specified dimension
    pub fn split(&self, data: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
        if self.world_size == 1 {
            return data.to_vec();
        }

        let dim_size = shape[dim];
        let chunk_size = (dim_size + self.world_size - 1) / self.world_size;
        let start = self.rank * chunk_size;
        let end = cmp::min(start + chunk_size, dim_size);

        self.slice_along_dim(data, shape, dim, start, end)
    }

    /// Slice tensor along dimension
    fn slice_along_dim(
        &self,
        data: &[f32],
        shape: &[usize],
        dim: usize,
        start: usize,
        end: usize,
    ) -> Vec<f32> {
        let slice_size = end - start;
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();
        let dim_stride = shape[dim] * inner_size;

        let mut result = Vec::with_capacity(outer_size * slice_size * inner_size);

        for outer in 0..outer_size {
            let outer_offset = outer * dim_stride;
            for i in start..end {
                let src_offset = outer_offset + i * inner_size;
                result.extend_from_slice(&data[src_offset..src_offset + inner_size]);
            }
        }

        result
    }

    /// Gather tensor parts from all devices (all-gather)
    pub fn all_gather(&self, local_data: &[f32], partition: &TensorPartition) -> Vec<f32> {
        // In a real implementation, this would perform network communication
        // For now, this is a placeholder that assumes single-device
        if self.world_size == 1 {
            return local_data.to_vec();
        }

        // Placeholder: would need actual network gather
        Vec::with_capacity(partition.global_numel())
    }

    /// Reduce tensor parts across all devices (all-reduce)
    pub fn all_reduce_sum(&self, data: &mut [f32]) {
        // In a real implementation, this would perform network communication
        // For now, this is a no-op for single device
        if self.world_size == 1 {
            return;
        }

        // Placeholder: would need actual network reduce
    }

    /// Scatter tensor to all devices
    pub fn scatter(&self, data: &[f32], shape: &[usize], dim: usize) -> Vec<Vec<f32>> {
        let dim_size = shape[dim];
        let chunk_size = (dim_size + self.world_size - 1) / self.world_size;

        let mut parts = Vec::with_capacity(self.world_size);

        for rank in 0..self.world_size {
            let start = rank * chunk_size;
            let end = cmp::min(start + chunk_size, dim_size);
            let part = self.slice_along_dim(data, shape, dim, start, end);
            parts.push(part);
        }

        parts
    }
}

/// Attention tensor parallelism helper
pub struct AttentionParallel {
    /// Number of heads per device
    heads_per_device: usize,
    /// Total attention heads
    total_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// World size
    world_size: usize,
    /// Local rank
    rank: usize,
}

impl AttentionParallel {
    /// Create new attention parallelism helper
    pub fn new(num_heads: usize, head_dim: usize, rank: usize, world_size: usize) -> Self {
        let heads_per_device = (num_heads + world_size - 1) / world_size;

        Self {
            heads_per_device,
            total_heads: num_heads,
            head_dim,
            world_size,
            rank,
        }
    }

    /// Get local head range
    pub fn local_head_range(&self) -> (usize, usize) {
        let start = self.rank * self.heads_per_device;
        let end = cmp::min(start + self.heads_per_device, self.total_heads);
        (start, end)
    }

    /// Get number of local heads
    pub fn local_heads(&self) -> usize {
        let (start, end) = self.local_head_range();
        end - start
    }

    /// Split QKV weights for this rank
    pub fn split_qkv_weights(&self, qkv: &[f32], hidden_size: usize) -> Vec<f32> {
        // QKV shape: [hidden_size, 3 * num_heads * head_dim]
        // Split to: [hidden_size, 3 * local_heads * head_dim]

        let (start_head, end_head) = self.local_head_range();
        let local_heads = end_head - start_head;

        let total_qkv_dim = 3 * self.total_heads * self.head_dim;
        let local_qkv_dim = 3 * local_heads * self.head_dim;

        let mut result = Vec::with_capacity(hidden_size * local_qkv_dim);

        for row in 0..hidden_size {
            let row_offset = row * total_qkv_dim;

            // For each of Q, K, V
            for qkv_idx in 0..3 {
                let qkv_offset = qkv_idx * self.total_heads * self.head_dim;

                for head in start_head..end_head {
                    let head_offset = head * self.head_dim;
                    let src_offset = row_offset + qkv_offset + head_offset;
                    result.extend_from_slice(&qkv[src_offset..src_offset + self.head_dim]);
                }
            }
        }

        result
    }

    /// Split output projection weights
    pub fn split_output_weights(&self, output: &[f32], hidden_size: usize) -> Vec<f32> {
        // Output shape: [num_heads * head_dim, hidden_size]
        // Split to: [local_heads * head_dim, hidden_size]

        let (start_head, end_head) = self.local_head_range();
        let local_heads = end_head - start_head;

        let total_input_dim = self.total_heads * self.head_dim;
        let local_input_dim = local_heads * self.head_dim;

        let mut result = Vec::with_capacity(local_input_dim * hidden_size);

        for head in start_head..end_head {
            let head_start = head * self.head_dim;
            for h in 0..self.head_dim {
                let row = head_start + h;
                let row_offset = row * hidden_size;
                result.extend_from_slice(&output[row_offset..row_offset + hidden_size]);
            }
        }

        result
    }
}

/// Feed-forward tensor parallelism helper
pub struct FFNParallel {
    /// Intermediate dimension per device
    intermediate_per_device: usize,
    /// Total intermediate dimension
    total_intermediate: usize,
    /// World size
    world_size: usize,
    /// Local rank
    rank: usize,
}

impl FFNParallel {
    /// Create new FFN parallelism helper
    pub fn new(intermediate_size: usize, rank: usize, world_size: usize) -> Self {
        let intermediate_per_device = (intermediate_size + world_size - 1) / world_size;

        Self {
            intermediate_per_device,
            total_intermediate: intermediate_size,
            world_size,
            rank,
        }
    }

    /// Get local intermediate range
    pub fn local_range(&self) -> (usize, usize) {
        let start = self.rank * self.intermediate_per_device;
        let end = cmp::min(
            start + self.intermediate_per_device,
            self.total_intermediate,
        );
        (start, end)
    }

    /// Get local intermediate size
    pub fn local_size(&self) -> usize {
        let (start, end) = self.local_range();
        end - start
    }

    /// Split gate/up projection (column parallel)
    pub fn split_gate_up(&self, weights: &[f32], hidden_size: usize) -> Vec<f32> {
        // Shape: [hidden_size, intermediate_size]
        // Split columns
        let (start, end) = self.local_range();
        let local_size = end - start;

        let mut result = Vec::with_capacity(hidden_size * local_size);

        for row in 0..hidden_size {
            let row_offset = row * self.total_intermediate;
            result.extend_from_slice(&weights[row_offset + start..row_offset + end]);
        }

        result
    }

    /// Split down projection (row parallel)
    pub fn split_down(&self, weights: &[f32], hidden_size: usize) -> Vec<f32> {
        // Shape: [intermediate_size, hidden_size]
        // Split rows
        let (start, end) = self.local_range();
        let local_size = end - start;

        let mut result = Vec::with_capacity(local_size * hidden_size);

        for row in start..end {
            let row_offset = row * hidden_size;
            result.extend_from_slice(&weights[row_offset..row_offset + hidden_size]);
        }

        result
    }
}

/// Communication pattern for tensor parallel operations
#[derive(Clone, Debug)]
pub enum CommPattern {
    /// All-gather: collect all parts into full tensor
    AllGather { dim: usize },
    /// All-reduce: sum across all devices
    AllReduceSum,
    /// Scatter: distribute tensor parts
    Scatter { dim: usize },
    /// No communication needed
    None,
}

/// Determine required communication for layer
pub fn get_layer_comm_pattern(
    layer_type: &str,
    strategy: ParallelismStrategy,
) -> (CommPattern, CommPattern) {
    match (layer_type, strategy) {
        // Column parallel: no input comm, all-gather output
        ("qkv_proj", ParallelismStrategy::ColumnParallel) => {
            (CommPattern::None, CommPattern::AllGather { dim: 2 })
        }
        ("gate_proj", ParallelismStrategy::ColumnParallel)
        | ("up_proj", ParallelismStrategy::ColumnParallel) => {
            (CommPattern::None, CommPattern::AllGather { dim: 1 })
        }

        // Row parallel: scatter input, all-reduce output
        ("o_proj", ParallelismStrategy::RowParallel)
        | ("down_proj", ParallelismStrategy::RowParallel) => {
            (CommPattern::Scatter { dim: 1 }, CommPattern::AllReduceSum)
        }

        // Head parallel attention
        ("attention", ParallelismStrategy::HeadParallel) => {
            (CommPattern::None, CommPattern::AllGather { dim: 2 })
        }

        _ => (CommPattern::None, CommPattern::None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_partition() {
        let shape = [512, 2048];
        let partition = TensorPartition::column_parallel(&shape, 0, 4);

        assert_eq!(partition.local_shape, vec![512, 512]);
        assert_eq!(partition.split_dim, 1);
    }

    #[test]
    fn test_row_partition() {
        let shape = [512, 2048];
        let partition = TensorPartition::row_parallel(&shape, 1, 4);

        assert_eq!(partition.local_shape, vec![128, 2048]);
        assert_eq!(partition.partition_idx, 1);
    }

    #[test]
    fn test_head_partition() {
        let partition = TensorPartition::head_parallel(1, 128, 32, 64, 0, 4);

        assert_eq!(partition.local_shape, vec![1, 128, 8, 64]);
        assert_eq!(partition.global_shape, vec![1, 128, 32, 64]);
    }

    #[test]
    fn test_tensor_split() {
        let splitter = TensorSplitter::new(0, 2);
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let shape = [2, 8];

        let part = splitter.split(&data, &shape, 1);
        assert_eq!(part.len(), 8); // 2 * 4
    }

    #[test]
    fn test_attention_parallel() {
        let attn = AttentionParallel::new(32, 64, 1, 4);

        let (start, end) = attn.local_head_range();
        assert_eq!(start, 8);
        assert_eq!(end, 16);
        assert_eq!(attn.local_heads(), 8);
    }

    #[test]
    fn test_ffn_parallel() {
        let ffn = FFNParallel::new(11008, 0, 4);

        let (start, end) = ffn.local_range();
        assert_eq!(start, 0);
        assert_eq!(end, 2752);
        assert_eq!(ffn.local_size(), 2752);
    }
}
