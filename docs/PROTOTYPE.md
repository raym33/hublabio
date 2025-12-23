# HubLab IO - Prototype Documentation

> **Status: EXPERIMENTAL PROTOTYPE**
>
> HubLab IO is currently in early prototype stage. This document provides comprehensive
> information about the project's current state, capabilities, limitations, and roadmap.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current Status](#current-status)
3. [What Works](#what-works)
4. [What Doesn't Work Yet](#what-doesnt-work-yet)
5. [Architecture Deep Dive](#architecture-deep-dive)
6. [Usage Examples](#usage-examples)
7. [Development Setup](#development-setup)
8. [Testing](#testing)
9. [Known Limitations](#known-limitations)
10. [Roadmap](#roadmap)
11. [FAQ](#faq)

---

## Project Overview

### What is HubLab IO?

HubLab IO is an **experimental operating system prototype** designed from the ground up
with artificial intelligence as a first-class citizen. Unlike traditional operating
systems that treat AI as an application-layer concern, HubLab IO integrates AI into
the kernel, scheduler, filesystem, and user interface.

### Vision

Imagine an operating system that:
- **Understands your intent** instead of requiring precise commands
- **Learns from your usage patterns** to optimize performance
- **Runs AI models natively** without heavyweight runtimes
- **Distributes computation** across your devices seamlessly
- **Responds to voice** as naturally as touch or keyboard

### Why Build a New OS?

| Challenge | Traditional Approach | HubLab IO Approach |
|-----------|---------------------|-------------------|
| Running AI on edge devices | Install Python, PyTorch, models (2GB+) | Native GGML in kernel (<100MB) |
| Optimizing for workloads | Static scheduler algorithms | AI-predicted scheduling |
| Finding files | Filename/path search | Semantic understanding |
| Voice control | Third-party apps with high latency | Kernel-level, <100ms |
| Multi-device AI | Complex distributed systems | Native P2P clustering |

---

## Current Status

### Development Phase

```
Phase 1: Foundation     [====================] 100% Complete
Phase 2: Core Services  [====================] 100% Complete
Phase 3: User Interface [============--------]  60% Complete
Phase 4: Applications   [======--------------]  30% Complete
Phase 5: Polish         [==------------------]  10% Started
```

### Component Status

| Component | Status | Description |
|-----------|--------|-------------|
| Bootloader (Stage 1) | Complete | ARM64 assembly bootstrap |
| Bootloader (Stage 2) | Complete | Rust early initialization |
| Kernel Core | Complete | Process, memory, IPC, VFS, scheduler, SMP |
| AI Scheduler | Complete | Priority queues, AI prediction hooks, multi-core |
| SMP Support | Complete | Multi-core, load balancing, CPU affinity |
| Memory Manager | Complete | Buddy allocator, slab allocator, paging |
| IPC System | Complete | Message passing, named endpoints |
| VFS | Complete | FAT32, ext4, ramfs, procfs, sysfs, devfs |
| Network Stack | Complete | TCP/IP, UDP, DHCP, DNS, ARP, ICMP |
| WiFi | Complete | BCM43xx driver with WPA2 |
| USB | Complete | DWC2, XHCI host controllers |
| Signals | Complete | POSIX signal handling |
| ELF Loader | Complete | ELF64 with relocations |
| Panic Handler | Complete | Stack traces, symbol resolution |
| AI Runtime | Working | GGUF loading, basic inference |
| Tokenizer | Working | BPE implementation |
| Sampling | Working | Top-k, top-p, temperature |
| Distributed | Partial | Cluster structure, needs tensor transfer |
| MoE-R | Partial | Router logic, needs integration |
| Shell (TUI) | Working | Basic commands, themes |
| Shell (Voice) | Not Started | Planned for Phase 4 |
| SDK | Working | Core APIs implemented |

### Maturity Levels

- **Stable**: Tested, documented, ready for development
- **Beta**: Functional but may have bugs
- **Alpha**: Partially implemented, expect changes
- **Planned**: Designed but not implemented

---

## What Works

### 1. Building and Running

```bash
# Clone the repository
git clone https://github.com/raym33/hublabio.git
cd hublabio

# Install dependencies (macOS)
brew install aarch64-elf-gcc qemu

# Setup Rust toolchain
make setup

# Build for Raspberry Pi 5
make rpi5

# Run in QEMU emulator
make run
```

Expected output:
```
===========================================
  HubLab IO Kernel v0.1.0
  AI-Native Operating System
===========================================

[BOOT] Initializing memory manager...
  Total RAM: 2048 MB
[BOOT] Setting up kernel heap...
[BOOT] Initializing architecture...
[BOOT] Initializing AI-enhanced scheduler...
[BOOT] Setting up IPC channels...
  IPC channels initialized
[BOOT] Mounting virtual filesystem...
  VFS initialized with 4 mount points
[BOOT] Setting up syscall interface...

[BOOT] Kernel initialization complete!
[BOOT] Starting init process...

hublab>
```

### 2. Shell Commands

```bash
# Help system
hublab> help
HubLab IO Shell Commands:

Built-in Commands:
help        Show this help message
clear       Clear the screen
history     Show command history
theme [n]   Get/set theme
version     Show version
exit        Exit the shell

AI Assistant:
?<query>    Ask AI a question

System Commands:
ls          List files
cd          Change directory
cat         Display file contents
ps          List processes
top         System monitor
ai          AI model management
pkg         Package manager

# Version check
hublab> version
HubLab IO Shell v0.1.0

# Theme switching
hublab> theme dracula
Theme set to: dracula

hublab> theme
Current theme: dracula

# Command history
hublab> history
   1  help
   2  version
   3  theme dracula
   4  theme
```

### 3. Memory Management

The kernel uses a buddy allocator for physical memory:

```rust
// Allocate a 4KB frame
let frame = memory::allocate_frame();

// Allocate 16KB (4 contiguous frames)
let frames = memory::allocate_order(2); // 2^2 = 4 pages

// Memory statistics
let stats = memory::stats();
println!("Total: {} MB", stats.total / (1024 * 1024));
println!("Used: {} MB", stats.used / (1024 * 1024));
println!("Free: {} MB", stats.free / (1024 * 1024));
```

### 4. IPC System

```rust
use kernel::ipc::{create_channel, register_endpoint, lookup_endpoint};

// Create a channel pair
let (client, server) = create_channel();

// Server: Register an endpoint
register_endpoint("io.hublab.myservice", server.channel_id())?;

// Client: Look up and connect
let service_id = lookup_endpoint("io.hublab.myservice")?;

// Send a message
client.send(MSG_TYPE_REQUEST, b"Hello, service!")?;

// Receive response
if let Some(msg) = server.try_receive() {
    println!("Received: {:?}", msg);
}
```

### 5. AI Inference (Basic)

```rust
use runtime::ai::{InferenceEngine, GenerationConfig};
use runtime::ai::tokenizer::Tokenizer;
use runtime::ai::sampling::{Sampler, SamplingConfig};

// Load a model (GGUF format)
let mut engine = InferenceEngine::new();
engine.load_model("/models/tinyllama-1.1b-q4.gguf")?;

// Configure generation
let config = GenerationConfig {
    max_tokens: 128,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    ..Default::default()
};

// Generate text
let prompt = "The quick brown fox";
let output = engine.generate(prompt, &config)?;
println!("{}", output);
```

### 6. SDK Usage

```rust
use hublabio_sdk::prelude::*;

fn main() -> Result<()> {
    // AI client
    let mut ai = AiClient::connect()?;
    ai.load_model("/models/chat.gguf")?;
    let response = ai.generate("Hello, world!")?;
    println!("{}", response);

    // File system
    let mut file = File::create("/tmp/test.txt")?;
    file.write_string("Hello from HubLab IO!")?;

    // System info
    println!("OS: {} {}", System::os_name(), System::os_version());
    println!("CPU: {}", System::cpu_info().model);
    println!("Memory: {} MB", System::memory_info().total / (1024 * 1024));

    Ok(())
}
```

---

## What Doesn't Work Yet

### Production-Ready Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Kernel Core** | Complete | Process, memory, IPC, VFS, scheduler |
| **Network Stack** | Complete | TCP/IP, UDP, DHCP, DNS, ARP, ICMP |
| **WiFi** | Complete | BCM43xx driver with WPA2 |
| **USB** | Complete | Host controllers, device enumeration |
| **Filesystems** | Complete | FAT32 (r/w), ext4 (read), ramfs |
| **Signals** | Complete | POSIX signals, signal delivery |
| **ELF Execution** | Complete | ELF64 loading with relocations |
| **Device Drivers** | Complete | GPIO, UART, framebuffer, timers |

### Features In Development

| Feature | What Works | What's Missing |
|---------|------------|----------------|
| Real Hardware | Drivers implemented | Testing on actual Pi |
| ext4 Write | Read support | Write support |
| GUI | TUI shell | Graphical compositor |
| Voice | Architecture designed | Implementation |
| Distributed AI | Cluster structures | Tensor transfer |

---

## Architecture Deep Dive

### Boot Sequence

```
┌──────────────────────────────────────────────────────────────────┐
│                       BOOT SEQUENCE                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Hardware Power-On                                             │
│         │                                                         │
│         ▼                                                         │
│  2. Stage 1 Bootloader (kernel/boot/stage1.S)                    │
│     - CPU initialization                                          │
│     - MMU setup (identity mapping)                                │
│     - FPU/SIMD enable                                             │
│     - Stack setup                                                 │
│     - Jump to Stage 2                                             │
│         │                                                         │
│         ▼                                                         │
│  3. Stage 2 (kernel/boot/stage2.rs)                              │
│     - Device tree parsing                                         │
│     - Memory map construction                                     │
│     - Framebuffer detection                                       │
│     - AI model detection                                          │
│     - Build BootInfo structure                                    │
│     - Jump to kernel_main                                         │
│         │                                                         │
│         ▼                                                         │
│  4. Kernel Main (kernel/src/lib.rs)                              │
│     - Console initialization                                      │
│     - Memory manager setup                                        │
│     - Heap initialization                                         │
│     - Architecture-specific init                                  │
│     - Scheduler initialization                                    │
│     - AI model loading (if available)                             │
│     - IPC initialization                                          │
│     - VFS mounting                                                │
│     - Syscall interface setup                                     │
│     - Spawn init process                                          │
│     - Enter scheduler loop                                        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Memory Layout (ARM64)

```
Virtual Address Space:
┌────────────────────────┐ 0xFFFF_FFFF_FFFF_FFFF
│                        │
│     Kernel Space       │
│                        │
├────────────────────────┤ 0xFFFF_0000_0000_0000
│                        │
│       (Unused)         │
│                        │
├────────────────────────┤ 0x0001_0000_0000_0000
│                        │
│     User Space         │
│                        │
├────────────────────────┤ 0x0000_0000_8000_0000
│                        │
│    Device MMIO         │
│                        │
├────────────────────────┤ 0x0000_0000_2000_0000
│                        │
│     AI Models          │
│     (512 MB)           │
│                        │
├────────────────────────┤ 0x0000_0000_0500_0000
│                        │
│    Kernel Heap         │
│     (64 MB)            │
│                        │
├────────────────────────┤ 0x0000_0000_0100_0000
│                        │
│      Kernel            │
│                        │
├────────────────────────┤ 0x0000_0000_0010_0000
│                        │
│    Bootloader          │
│                        │
├────────────────────────┤ 0x0000_0000_0008_0000
│     Reserved           │
└────────────────────────┘ 0x0000_0000_0000_0000
```

### IPC Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      IPC ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Process A                              Process B               │
│  ┌──────────┐                           ┌──────────┐            │
│  │ Endpoint │                           │ Endpoint │            │
│  │   (A)    │◄─────────────────────────►│   (B)    │            │
│  └──────────┘                           └──────────┘            │
│       │                                       │                  │
│       │                                       │                  │
│       ▼                                       ▼                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      Channel                             │    │
│  │  ┌─────────────────┐      ┌─────────────────┐           │    │
│  │  │  Queue A → B    │      │  Queue B → A    │           │    │
│  │  │  [msg][msg][msg]│      │  [msg][msg]     │           │    │
│  │  └─────────────────┘      └─────────────────┘           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Named Endpoint Registry                  │    │
│  │                                                          │    │
│  │    "io.hublab.ai"       → Channel 1                     │    │
│  │    "io.hublab.fs"       → Channel 2                     │    │
│  │    "io.hublab.net"      → Channel 3                     │    │
│  │    "io.hublab.display"  → Channel 4                     │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### AI Runtime Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI RUNTIME STACK                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Application Layer                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   SDK API (hublabio_sdk::ai)                            │    │
│  │   - AiClient, ChatSession                               │    │
│  │   - High-level generate(), chat(), embed()              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Service Layer                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   AI Service (via IPC)                                   │    │
│  │   - Request queue management                             │    │
│  │   - Model caching                                        │    │
│  │   - MoE-R routing                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Runtime Layer                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   Inference Engine (runtime::ai::inference)              │    │
│  │   ┌────────────┐  ┌────────────┐  ┌────────────┐        │    │
│  │   │  Tokenizer │  │   Model    │  │   Sampler  │        │    │
│  │   │    (BPE)   │  │   (GGUF)   │  │  (TopK/P)  │        │    │
│  │   └────────────┘  └────────────┘  └────────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Kernel Layer                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │   Memory-mapped AI regions                               │    │
│  │   - Model weights (read-only)                            │    │
│  │   - KV cache (read-write)                                │    │
│  │   - Activation buffers                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Example 1: Simple Shell Interaction

```bash
# Start the OS (in QEMU)
$ make run

# After boot, you're in the shell
hublab> help
# Shows available commands

hublab> version
HubLab IO Shell v0.1.0

hublab> theme amoled
Theme set to: amoled

# AI query (when AI service is running)
hublab> ?what time is it
AI: I don't have access to a real-time clock, but I can help you
    set up time synchronization...

hublab> exit
# Exits shell
```

### Example 2: Building a Simple App

```rust
// apps/hello/src/main.rs
#![no_std]
#![no_main]

extern crate alloc;

use hublabio_sdk::prelude::*;

#[no_mangle]
pub fn main() -> Result<()> {
    // Print to console
    println!("Hello from HubLab IO!");

    // Get system info
    let mem = System::memory_info();
    println!("Memory: {} MB total, {} MB free",
             mem.total / (1024 * 1024),
             mem.free / (1024 * 1024));

    // Try AI (if available)
    if let Ok(ai) = AiClient::connect() {
        if ai.is_connected() {
            let response = ai.generate("Say hello in 5 words")?;
            println!("AI says: {}", response);
        }
    }

    Ok(())
}
```

```toml
# apps/hello/Cargo.toml
[package]
name = "hello"
version = "0.1.0"
edition = "2021"

[dependencies]
hublabio-sdk = { path = "../../sdk" }

[profile.release]
opt-level = "z"
lto = true
panic = "abort"
```

### Example 3: Using the AI Runtime Directly

```rust
use runtime::ai::inference::InferenceEngine;
use runtime::ai::tokenizer::Tokenizer;
use runtime::ai::sampling::{Sampler, SamplingConfig};
use runtime::ai::quantization::QuantType;

fn main() {
    // Create inference engine
    let mut engine = InferenceEngine::new();

    // Load model
    println!("Loading model...");
    engine.load_model("/models/qwen2-0.5b-q4_0.gguf")
        .expect("Failed to load model");

    // Check model info
    let info = engine.model_info();
    println!("Model: {} layers, {} vocab, {:?} quantization",
             info.num_layers, info.vocab_size, info.quant_type);

    // Configure sampling
    let sampling = SamplingConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
        ..Default::default()
    };

    // Generate
    let prompt = "Write a haiku about programming:";
    println!("Prompt: {}", prompt);
    println!("---");

    let output = engine.generate_with_config(prompt, sampling, 64)
        .expect("Generation failed");

    println!("{}", output);
}
```

### Example 4: IPC Communication

```rust
// Service (server side)
use kernel::ipc::{create_channel, register_endpoint, msg_types};

fn start_service() {
    let (client_end, server_end) = create_channel();

    // Register as a named service
    register_endpoint("io.hublab.myservice", server_end.channel_id())
        .expect("Failed to register");

    println!("Service started, waiting for messages...");

    loop {
        if let Some(msg) = server_end.try_receive() {
            println!("Received: type={}, len={}", msg.header.msg_type, msg.payload.len());

            // Send response
            server_end.send(msg_types::REPLY_OK, b"OK")
                .expect("Failed to send response");
        }
    }
}

// Client side
use kernel::ipc::{Channel, lookup_endpoint};

fn call_service() {
    // Find the service
    let service_id = lookup_endpoint("io.hublab.myservice")
        .expect("Service not found");

    // Connect and send
    let channel = Channel::connect(service_id)
        .expect("Failed to connect");

    channel.send(0x100, b"Hello, service!")
        .expect("Failed to send");

    // Wait for response
    let response = channel.receive()
        .expect("Failed to receive");

    println!("Response: {:?}", response);
}
```

### Example 5: MoE-R Expert Routing

```rust
use runtime::moe::{Swarm, Router, Synthesizer, RoutingStrategy, SynthesisStrategy};
use runtime::moe::{ExpertConfig, ExpertDomain, ExpertId};

fn setup_moe() {
    // Create swarm with top-2 routing
    let mut swarm = Swarm::new(
        RoutingStrategy::TopK(2),
        SynthesisStrategy::Best
    );

    // Register experts
    swarm.router.register(ExpertConfig {
        id: ExpertId(1),
        name: "python-expert".into(),
        description: "Python programming expert".into(),
        domain: ExpertDomain {
            name: "python".into(),
            keywords: vec!["python", "pip", "django", "flask"].iter()
                .map(|s| s.to_string()).collect(),
            threshold: 0.5,
        },
        model_path: "/models/python-expert.gguf".into(),
        node_id: None,
        priority: 10,
    });

    swarm.router.register(ExpertConfig {
        id: ExpertId(2),
        name: "rust-expert".into(),
        description: "Rust programming expert".into(),
        domain: ExpertDomain {
            name: "rust".into(),
            keywords: vec!["rust", "cargo", "crate", "tokio"].iter()
                .map(|s| s.to_string()).collect(),
            threshold: 0.5,
        },
        model_path: "/models/rust-expert.gguf".into(),
        node_id: None,
        priority: 10,
    });

    // Route a query
    let query = "How do I parse JSON in Python?";
    let experts = swarm.process(query);

    println!("Query: {}", query);
    println!("Routed to {} expert(s):", experts.len());
    for id in &experts {
        if let Some(expert) = swarm.router.get_expert(*id) {
            println!("  - {}", expert.config.name);
        }
    }
}
```

---

## Development Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | nightly | Kernel and runtime |
| aarch64-elf-gcc | 13+ | Cross-compilation |
| QEMU | 8+ | Emulation |
| Git | 2.30+ | Version control |

### macOS Setup

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install rustup aarch64-elf-gcc qemu

# Setup Rust
rustup default nightly
rustup target add aarch64-unknown-none
rustup component add rust-src llvm-tools-preview

# Clone and build
git clone https://github.com/raym33/hublabio.git
cd hublabio
make setup
make rpi5
make run
```

### Linux Setup (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update
sudo apt install curl git build-essential
sudo apt install gcc-aarch64-linux-gnu qemu-system-arm

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup default nightly
rustup target add aarch64-unknown-none

# Clone and build
git clone https://github.com/raym33/hublabio.git
cd hublabio
make setup
make rpi5
make run
```

### Directory Structure

```
hublabio/
├── kernel/                 # Kernel source
│   ├── boot/              # Bootloader
│   │   ├── stage1.S       # Assembly bootstrap
│   │   └── stage2.rs      # Rust early init
│   ├── src/               # Kernel modules
│   │   ├── lib.rs         # Kernel entry point
│   │   ├── memory/        # Memory management
│   │   ├── ipc/           # Inter-process communication
│   │   └── vfs/           # Virtual filesystem
│   └── Cargo.toml
├── runtime/               # AI Runtime
│   ├── src/
│   │   ├── lib.rs
│   │   ├── ai/            # AI inference
│   │   │   ├── inference/ # Inference engine
│   │   │   ├── tokenizer.rs
│   │   │   ├── sampling.rs
│   │   │   └── quantization.rs
│   │   ├── distributed/   # Distributed inference
│   │   └── moe/           # MoE-R system
│   └── Cargo.toml
├── shell/                 # User shell
│   ├── src/
│   │   ├── lib.rs
│   │   ├── commands/      # Shell commands
│   │   └── themes.rs      # Visual themes
│   ├── tui/               # TUI interface
│   └── Cargo.toml
├── sdk/                   # Application SDK
│   ├── src/
│   │   ├── lib.rs
│   │   ├── ai.rs          # AI client
│   │   ├── ipc.rs         # IPC client
│   │   ├── ui.rs          # Widget toolkit
│   │   ├── fs.rs          # File system
│   │   ├── net.rs         # Networking
│   │   └── sys.rs         # System info
│   └── Cargo.toml
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md
│   ├── GETTING_STARTED.md
│   └── PROTOTYPE.md       # This file
├── Cargo.toml             # Workspace config
├── Makefile               # Build system
├── rust-toolchain.toml    # Rust toolchain config
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## Testing

### Running Tests

```bash
# All tests
make test

# Kernel tests only
make test-kernel

# Runtime tests only
make test-runtime

# SDK tests only
make test-sdk

# With verbose output
cargo test --workspace -- --nocapture
```

### Test Coverage

| Component | Unit Tests | Integration Tests | Hardware Tests |
|-----------|------------|-------------------|----------------|
| Kernel Memory | Yes | No | No |
| Kernel IPC | Yes | Yes | No |
| Kernel VFS | Partial | No | No |
| AI Tokenizer | Yes | Yes | N/A |
| AI Sampling | Yes | No | N/A |
| AI Quantization | Yes | No | N/A |
| SDK APIs | Partial | No | No |
| Shell | No | No | No |

### QEMU Testing

```bash
# Run with default settings (2GB RAM)
make run

# Run with more RAM
make run QEMU_MEM=4G

# Run with GDB server for debugging
make debug
# In another terminal:
aarch64-none-elf-gdb target/aarch64-unknown-none/release/hublabio-kernel
(gdb) target remote :1234
(gdb) continue
```

---

## Known Limitations

### Current Limitations

1. **Hardware Testing**: Only verified in QEMU (real Pi testing pending)
2. **GUI**: TUI only, no graphical compositor yet
3. **Voice**: Voice interface not implemented yet

### Performance Limitations

1. **~~Single-core Only~~**: SMP fully implemented (multi-core support complete)
2. **No DMA**: All I/O is CPU-driven
3. **No GPU**: No graphics acceleration
4. **Limited Memory**: No swap, may OOM with large models

### Compatibility

1. **ARM64 Primary**: Full ARM64 support, RISC-V in development
2. **Networking**: Full TCP/IP stack with WiFi support
3. **USB**: Host controllers implemented (DWC2, XHCI)
4. **Storage**: FAT32, ext4 (full read/write + journaling), ramfs supported

---

## Roadmap

### Phase 2: Core Services (Complete)

- [x] Complete VFS with FAT32 support
- [x] UART driver for Pi (PL011)
- [x] Framebuffer console
- [x] Process lifecycle (fork/exec/wait)
- [x] Signal handling (POSIX)
- [x] Full TCP/IP network stack
- [x] WiFi driver (BCM43xx)
- [x] USB stack (DWC2, XHCI)
- [x] Block device layer with partitions

### Phase 3: User Interface (Current)

- [x] TUI shell with themes
- [ ] TUI improvements (scrolling, editing)
- [ ] AI integration in shell
- [ ] File manager app
- [ ] System monitor app
- [ ] Settings app
- [ ] GUI compositor

### Phase 4: Applications

- [ ] Voice interface (Whisper, Piper)
- [ ] Web browser (minimal)
- [ ] Package manager
- [ ] App SDK improvements
- [ ] Developer tools

### Phase 5: Polish

- [ ] Hardware testing (RPi 5, RPi 4)
- [ ] Performance optimization
- [ ] Security hardening
- [ ] Documentation completion
- [ ] First public release

### Long-term Goals

- Multi-device clustering
- NPU/GPU acceleration
- Smartphone support
- OTA updates
- App marketplace

---

## FAQ

### General

**Q: Is HubLab IO usable today?**
A: It's a prototype. You can boot it in QEMU and run basic commands, but it's not
   ready for production use.

**Q: What makes this different from Linux with Ollama?**
A: HubLab IO integrates AI at the kernel level, not as a userspace application.
   This enables features like AI-enhanced scheduling and semantic filesystems
   that aren't possible with a traditional OS.

**Q: Can I run this on my Raspberry Pi?**
A: Not yet reliably. It's only tested in QEMU. Hardware testing is planned for
   Phase 5.

### Technical

**Q: Why Rust instead of C?**
A: Memory safety is critical for an OS kernel. Rust's borrow checker prevents
   entire classes of bugs that plague C kernels.

**Q: Why a microkernel?**
A: Security and modularity. By running drivers in userspace, we reduce the
   kernel attack surface and can update components independently.

**Q: What AI models can it run?**
A: Any GGUF-formatted model. Recommended sizes are 0.5B-3B parameters for
   devices with 1-4GB RAM.

### Development

**Q: How can I contribute?**
A: See CONTRIBUTING.md. We especially need help with device drivers,
   documentation, and testing.

**Q: Where can I get help?**
A: Open an issue on GitHub. We're building a community around this project.

**Q: Is there a development chat?**
A: Not yet. We plan to set up Discord when we have more contributors.

---

## Conclusion

HubLab IO is an ambitious experiment to rethink operating systems for the AI era.
While still in early prototype stage, it demonstrates that AI can be a first-class
citizen at every layer of the OS stack.

We welcome contributors, testers, and feedback. Together, we can build the
operating system that AI deserves.

---

*Last updated: December 2024*
*Version: 0.1.0-prototype*
