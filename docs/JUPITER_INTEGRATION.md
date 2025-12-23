# Jupiter Integration in HubLab IO

This document explains where Jupiter-derived code is used in the HubLab IO codebase and what purpose each component serves.

## Overview

HubLab IO inherits several key AI architecture concepts from the Jupiter project:

1. **MoE-R (Mixture of Real Experts)** - Expert routing and synthesis system
2. **Distributed Inference** - Multi-node model splitting and coordination
3. **Quantization Support** - GGUF format with 23 quantization types
4. **AI-Enhanced Scheduling** - Neural network-based process scheduling

## Code Locations

### 1. MoE-R System

**Location:** `runtime/src/moe/mod.rs`

The Mixture of Real Experts system enables multiple specialized AI models to collaborate on tasks. This is a direct adaptation of Jupiter's MoE-R swarm architecture.

#### Components

| Component | Line | Purpose |
|-----------|------|---------|
| `ExpertId` | 13 | Unique identifier for each expert model |
| `ExpertDomain` | 17-25 | Domain specialization (keywords, thresholds) |
| `ExpertConfig` | 28-44 | Expert configuration (model path, priority, node) |
| `Router` | 57-181 | Routes queries to appropriate experts |
| `RoutingStrategy` | 67-77 | Selection strategies (TopK, Threshold, Single, All) |
| `Synthesizer` | 184-262 | Combines multiple expert responses |
| `SynthesisStrategy` | 190-202 | Combination methods (Best, Voting, Concatenate, etc.) |
| `Swarm` | 266-301 | Main orchestrator for the MoE-R system |

#### Usage Example

```rust
use hublabio_runtime::moe::{Swarm, RoutingStrategy, SynthesisStrategy, ExpertConfig, ExpertDomain, ExpertId};

// Create a swarm with TopK routing and Best synthesis
let mut swarm = Swarm::new(
    RoutingStrategy::TopK(3),
    SynthesisStrategy::Best,
);

// Register experts
swarm.router.register(ExpertConfig {
    id: ExpertId(1),
    name: String::from("python-expert"),
    description: String::from("Python code specialist"),
    domain: ExpertDomain {
        name: String::from("python"),
        keywords: vec!["python".into(), "django".into(), "flask".into()],
        threshold: 0.5,
    },
    model_path: String::from("/models/python-expert.gguf"),
    node_id: None,
    priority: 10,
});

// Route a query
let selected_experts = swarm.process("How do I use Django ORM?");
```

#### Routing Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `TopK(n)` | Select top N experts by relevance | General multi-expert queries |
| `Threshold(t)` | Select experts above confidence threshold | Quality-focused responses |
| `Single` | Route to single best expert | Simple, domain-specific queries |
| `All` | Use all available experts | Comprehensive analysis |

#### Synthesis Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `Best` | Return highest-confidence response | Speed-critical applications |
| `WeightedAverage` | Blend responses by confidence | Numeric/classification tasks |
| `Voting` | Use majority agreement | Factual queries |
| `Concatenate` | Combine all responses | Multi-perspective analysis |
| `Summarize` | Create summary of all responses | Complex queries |

---

### 2. Distributed Inference

**Location:** `runtime/src/distributed/mod.rs`

Enables splitting large models across multiple devices (e.g., Raspberry Pi cluster). Based on Jupiter's distributed AI concepts.

#### Components

| Component | Line | Purpose |
|-----------|------|---------|
| `NodeId` | 17 | Unique node identifier |
| `NodeInfo` | 28-46 | Node metadata (address, memory, compute capability) |
| `NodeStatus` | 49-63 | Node state (Available, Ready, Busy, Offline, Error) |
| `LayerAssignment` | 66-76 | Maps model layers to specific nodes |
| `DistributedConfig` | 79-93 | Configuration for distributed execution |
| `ClusterManager` | 96-218 | Manages cluster nodes and layer assignments |

#### Additional Modules

- `runtime/src/distributed/cluster.rs` - Cluster discovery and heartbeat
- `runtime/src/distributed/protocol.rs` - Communication protocol
- `runtime/src/distributed/tensor_parallel.rs` - Tensor splitting across nodes

#### Usage Example

```rust
use hublabio_runtime::distributed::{ClusterManager, NodeInfo, NodeStatus, NodeId};

// Create cluster manager
let mut cluster = ClusterManager::new();

// Add nodes (e.g., Raspberry Pis on local network)
cluster.add_node(NodeInfo {
    id: NodeId(1),
    address: String::from("192.168.1.101"),
    port: 5000,
    memory: 4 * 1024 * 1024 * 1024, // 4GB
    available_memory: 3 * 1024 * 1024 * 1024,
    compute: 2.0, // TFLOPS estimate
    status: NodeStatus::Available,
    layers: None,
});

cluster.add_node(NodeInfo {
    id: NodeId(2),
    address: String::from("192.168.1.102"),
    port: 5000,
    memory: 8 * 1024 * 1024 * 1024, // 8GB
    available_memory: 6 * 1024 * 1024 * 1024,
    compute: 4.0,
    status: NodeStatus::Available,
    layers: None,
});

// Compute layer assignments for a 32-layer model
let assignments = cluster.compute_assignments(32, 256 * 1024 * 1024);
// Node 1 gets fewer layers (less memory)
// Node 2 gets more layers (more memory)
```

#### Layer Distribution Algorithm

The `compute_assignments` function distributes model layers proportionally based on available memory:

```
Node Memory Ratio = Node Available Memory / Total Cluster Memory
Node Layers = Total Layers × Node Memory Ratio
```

This ensures larger memory nodes handle more layers, optimizing cluster utilization.

---

### 3. Inference Engine with Distributed Support

**Location:** `runtime/ai/inference/engine.rs`

The main inference engine supports both local and distributed inference modes.

#### Key Components

| Component | Line | Purpose |
|-----------|------|---------|
| `InferenceEngine` | 269-280 | Main local inference engine |
| `DistributedEngine` | 617-720 | Coordinates distributed inference across peers |
| `PeerNode` | 629-642 | Represents a peer in the distributed network |
| `ModelConfig` | 136-158 | Model architecture configuration |
| `GenerationConfig` | 284-310 | Text generation parameters |

#### Architecture Support

The engine supports multiple model architectures (line 161-169):

- Llama
- Qwen2
- Phi3
- Gemma
- Mistral
- Custom

#### Distributed Engine Flow

```
1. Coordinator receives prompt
2. Tokenizes input
3. Each node computes its assigned layers
4. Activations passed between nodes (pipeline)
5. Final logits collected at coordinator
6. Sampling and decoding at coordinator
7. Response returned
```

---

### 4. Quantization Support

**Location:** `runtime/src/ai/quantization.rs`

Supports 23 quantization formats for efficient model storage and inference.

#### Supported Formats

| Type | Bits/Weight | Description |
|------|-------------|-------------|
| `F32` | 32.0 | Full precision float |
| `F16` | 16.0 | Half precision float |
| `Q4_0` | 4.5 | 4-bit quantization |
| `Q4_1` | 4.5 | 4-bit with min value |
| `Q5_0` | 5.5 | 5-bit quantization |
| `Q5_1` | 5.5 | 5-bit with min value |
| `Q8_0` | 8.5 | 8-bit quantization |
| `Q8_1` | 8.5 | 8-bit with sum |
| `Q2_K` | 2.6 | 2-bit k-quant |
| `Q3_K` | 3.4 | 3-bit k-quant |
| `Q4_K` | 4.5 | 4-bit k-quant |
| `Q5_K` | 5.5 | 5-bit k-quant |
| `Q6_K` | 6.6 | 6-bit k-quant |
| `Q8_K` | 8.5 | 8-bit k-quant |
| `IQ2_XXS` | 2.1 | 2-bit importance quantization |
| `IQ2_XS` | 2.3 | 2-bit importance (extra small) |
| `IQ2_S` | 2.5 | 2-bit importance (small) |
| `IQ3_XXS` | 3.1 | 3-bit importance quantization |
| `IQ3_S` | 3.4 | 3-bit importance (small) |
| `IQ4_NL` | 4.3 | 4-bit importance (non-linear) |
| `IQ4_XS` | 4.3 | 4-bit importance (extra small) |
| `IQ1_S` | 1.6 | 1-bit importance quantization |

#### Block Structures

The quantization uses block-based compression:

- **Q4_0 Block** (18 bytes): 2-byte scale + 16-byte quantized values (32 weights)
- **Q8_0 Block** (34 bytes): 2-byte scale + 32-byte quantized values (32 weights)

#### Dequantization Functions

```rust
// Q4_0 dequantization
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32; 32]) {
    let scale = f16_to_f32(block.scale);
    for (i, &byte) in block.quants.iter().enumerate() {
        let low = (byte & 0x0F) as i8 - 8;
        let high = (byte >> 4) as i8 - 8;
        output[i * 2] = (low as f32) * scale;
        output[i * 2 + 1] = (high as f32) * scale;
    }
}
```

---

### 5. AI-Enhanced Scheduler

**Location:** `kernel/src/scheduler/ai.rs`

Uses a small neural network to predict process behavior and optimize scheduling decisions.

#### Components

| Component | Line | Purpose |
|-----------|------|---------|
| `ProcessFeatures` | 15-33 | Input features for prediction |
| `SchedulingPrediction` | 65-77 | NN output (runtime, time slice, etc.) |
| `SchedulerNN` | 80-172 | Neural network (8→16→4 architecture) |
| `predict()` | 202-209 | Make scheduling prediction |
| `heuristic_predict()` | 218-252 | Fallback when AI unavailable |

#### Neural Network Architecture

```
Input Layer (8 neurons):
  - cpu_time (normalized)
  - io_ops (normalized)
  - memory_usage (MB)
  - ipc_count
  - idle_time
  - avg_runtime
  - is_ai_workload (0/1)
  - priority (normalized)

Hidden Layer (16 neurons, ReLU activation)

Output Layer (4 neurons):
  - predicted_runtime
  - io_probability (sigmoid)
  - recommended_time_slice
  - boost_priority (threshold)
```

#### Model Loading

The scheduler AI model is loaded from GGUF format:

```rust
pub fn load_model(addr: usize, size: usize) -> Result<(), &'static str> {
    let data = unsafe {
        core::slice::from_raw_parts(addr as *const u8, size)
    };
    unsafe {
        if let Some(ref mut nn) = SCHEDULER_NN {
            nn.load_from_gguf(data)?;
        }
    }
    Ok(())
}
```

---

### 6. Tokenizer

**Location:** `runtime/src/ai/tokenizer.rs`

BPE tokenizer implementation compatible with GGUF models.

#### Features

- BPE (Byte-Pair Encoding) tokenization
- Special token handling (BOS, EOS, PAD, UNK)
- GGUF metadata parsing
- Llama-style space handling (▁ prefix)

#### Special Tokens

| Token | ID | Purpose |
|-------|------|---------|
| `<pad>` | 0 | Padding |
| `<s>` | 1 | Beginning of sequence |
| `</s>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown token |

---

### 7. Sampling Strategies

**Location:** `runtime/src/ai/sampling.rs`

Implements various sampling strategies for text generation.

#### Presets

| Preset | Temperature | Top-p | Top-k | Use Case |
|--------|-------------|-------|-------|----------|
| `greedy()` | 0.0 | 1.0 | 1 | Deterministic output |
| `precise()` | 0.2 | 0.8 | 20 | Code generation |
| `default()` | 0.7 | 0.9 | 40 | Balanced generation |
| `creative()` | 1.0 | 0.95 | 50 | Creative writing |

#### Penalties

- **Repetition Penalty**: Reduces probability of repeated tokens
- **Frequency Penalty**: Linear penalty based on token count
- **Presence Penalty**: Binary penalty for any token occurrence

---

### 8. SDK AI Module

**Location:** `sdk/src/ai.rs`

User-facing API for AI capabilities.

#### Classes

| Class | Purpose |
|-------|---------|
| `AiClient` | Connect to AI service, generate text |
| `ChatSession` | Multi-turn conversation management |
| `GenerateOptions` | Generation parameters |

#### Usage

```rust
use hublabio_sdk::ai::{AiClient, ChatSession, GenerateOptions};

// Simple generation
let mut client = AiClient::connect()?;
client.load_model("/models/qwen2-0.5b-q4.gguf")?;
let response = client.generate("Hello, world!")?;

// Chat session
let mut chat = ChatSession::new(client);
chat.set_system_prompt("You are a helpful assistant.");
let response = chat.chat("What is 2+2?")?;
```

---

### 9. Runtime Configuration

**Location:** `runtime/src/lib.rs`

Global runtime configuration that enables/disables Jupiter-derived features.

```rust
pub struct RuntimeConfig {
    /// Maximum memory for models
    pub max_memory: usize,
    /// Number of threads for inference
    pub num_threads: usize,
    /// Enable distributed inference
    pub distributed: bool,       // Jupiter distributed
    /// Enable MoE-R routing
    pub moe_enabled: bool,       // Jupiter MoE-R
    /// Default generation temperature
    pub temperature: f32,
    /// Default top-p sampling
    pub top_p: f32,
}
```

---

## Integration Points

### Where MoE-R is Used

1. **AI Chat Service** - Routes user queries to specialized experts
2. **Code Assistance** - Selects language-specific models
3. **System Tasks** - Combines general and domain experts

### Where Distributed Inference is Used

1. **Large Model Loading** - Splits 7B+ models across Pi cluster
2. **Expert Distribution** - Places different experts on different nodes
3. **Load Balancing** - Distributes inference load across available hardware

### Where AI Scheduling is Used

1. **Process Prioritization** - Predicts CPU/IO behavior
2. **Time Slice Allocation** - Optimizes quantum based on workload
3. **AI Workload Boost** - Prioritizes inference processes

---

## Configuration Files

### System Configuration (`config/system.toml`)

```toml
[kernel]
log_level = "info"
heap_size = "64M"

[scheduler]
ai_enabled = true
model_path = "/models/scheduler.gguf"

[runtime]
distributed = false  # Enable for cluster
moe_enabled = true   # Enable MoE-R
num_threads = 4
max_memory = "2G"

[ai]
default_model = "/models/qwen2-0.5b-q4.gguf"
temperature = 0.7
top_p = 0.9
```

### Distributed Configuration (`config/distributed.toml`)

```toml
[cluster]
enabled = true
discovery = "multicast"  # or "static"
port = 5000

[nodes]
# Static node configuration
[[nodes.static]]
address = "192.168.1.101"
port = 5000

[[nodes.static]]
address = "192.168.1.102"
port = 5000

[pipeline]
overlap_comm = true      # Overlap communication with computation
tensor_parallel = 1      # Tensor parallelism degree
pipeline_parallel = 2    # Pipeline parallelism degree
```

---

## Memory Requirements

### Single Node

| Model Size | Quantization | RAM Required |
|------------|--------------|--------------|
| 0.5B | Q4_K | ~400MB |
| 1B | Q4_K | ~800MB |
| 3B | Q4_K | ~2GB |
| 7B | Q4_K | ~4GB |

### Distributed (2 Nodes)

| Model Size | Quantization | Per Node RAM |
|------------|--------------|--------------|
| 3B | Q4_K | ~1GB each |
| 7B | Q4_K | ~2GB each |
| 13B | Q4_K | ~4GB each |

---

## Performance Considerations

### MoE-R Optimization

1. **Expert Caching**: Keep frequently-used experts in memory
2. **Lazy Loading**: Load experts on-demand
3. **Keyword Indexing**: O(1) keyword lookup for routing

### Distributed Optimization

1. **Pipeline Parallelism**: Overlap layer computation across nodes
2. **Tensor Parallelism**: Split attention heads across nodes
3. **Communication Overlap**: Hide network latency with computation

### Quantization Trade-offs

| Format | Speed | Quality | Memory |
|--------|-------|---------|--------|
| Q8_K | Fast | High | High |
| Q4_K | Medium | Good | Medium |
| Q2_K | Fast | Acceptable | Low |
| IQ2_XXS | Fast | Lower | Lowest |

---

## Future Enhancements

Based on Jupiter concepts not yet fully implemented:

1. **Speculative Decoding** - Use small draft model for speedup
2. **Dynamic Expert Loading** - Load experts based on current queries
3. **Cross-Node KV Cache** - Share KV cache across distributed nodes
4. **Expert Fine-tuning** - Online adaptation of expert weights

---

## Related Documentation

- [Architecture Overview](ARCHITECTURE.md) - System architecture
- [Model Training Guide](MODEL_TRAINING.md) - Training custom models
- [Getting Started](GETTING_STARTED.md) - Setup and usage
- [Prototype Status](PROTOTYPE.md) - Implementation progress

---

## Code References Summary

| File | Jupiter Feature | Lines |
|------|-----------------|-------|
| `runtime/src/moe/mod.rs` | MoE-R System | 1-302 |
| `runtime/src/distributed/mod.rs` | Distributed Inference | 1-219 |
| `runtime/ai/inference/engine.rs` | Inference Engine | 1-746 |
| `runtime/src/ai/quantization.rs` | Quantization | 1-273 |
| `runtime/src/ai/sampling.rs` | Sampling Strategies | 1-245 |
| `runtime/src/ai/tokenizer.rs` | BPE Tokenizer | 1-212 |
| `kernel/src/scheduler/ai.rs` | AI Scheduling | 1-253 |
| `sdk/src/ai.rs` | User API | 1-233 |
| `runtime/src/lib.rs` | Runtime Config | 1-64 |
