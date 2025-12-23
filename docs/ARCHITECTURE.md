# HubLab IO Architecture

This document describes the architecture of HubLab IO, an AI-native operating system.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATIONS                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │  Shell  │ │ AI Chat │ │ Files   │ │ Settings│ │ Custom  │              │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│       └──────────┬┴──────────┬┴──────────┬┴──────────┬┘                    │
│                  │           │           │           │                      │
│  ┌───────────────┴───────────┴───────────┴───────────┴────────────────────┐│
│  │                         HubLab IO SDK                                   ││
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                      ││
│  │  │ AI  │ │ IPC │ │ UI  │ │ FS  │ │ Net │ │ Sys │                      ││
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                      ││
│  └───────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                              SERVICES (Userspace)                            │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │    AI     │ │   File    │ │  Network  │ │  Display  │ │   Input   │    │
│  │  Service  │ │  System   │ │  Service  │ │  Service  │ │  Service  │    │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘    │
│        └─────────────┴───────┬─────┴─────────────┴─────────────┘          │
│                              │                                              │
│                       ┌──────┴──────┐                                       │
│                       │     IPC     │                                       │
│                       └──────┬──────┘                                       │
├──────────────────────────────┼──────────────────────────────────────────────┤
│                      KERNEL (Microkernel)                                   │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │ Scheduler │ │  Memory   │ │    IPC    │ │    VFS    │ │  Syscall  │    │
│  │   (AI)    │ │  Manager  │ │  Channels │ │           │ │ Interface │    │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                      HARDWARE ABSTRACTION LAYER                             │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │   GPIO    │ │   UART    │ │    SPI    │ │    I2C    │ │    USB    │    │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Kernel

The kernel is a microkernel written in Rust, designed for minimal attack surface and maximum reliability.

### Components

#### AI-Enhanced Scheduler
The scheduler uses a tiny transformer model (~1M parameters) to predict:
- Task runtime estimates
- Optimal priority adjustments
- Memory access patterns

```rust
pub struct AiScheduler {
    ready_queues: [VecDeque<TaskId>; 256],
    tasks: BTreeMap<TaskId, Task>,
    model: TinyTransformer,
    predictions: BTreeMap<TaskId, TaskPrediction>,
}
```

#### Memory Manager
Uses a buddy allocator for physical memory with support for:
- 4KB/16KB/64KB pages
- NUMA awareness
- Memory-mapped AI model regions

#### IPC System
Message-passing IPC with:
- Named endpoints for service discovery
- Async message queues
- Zero-copy transfers for large data

#### VFS
Virtual filesystem providing:
- Unified namespace
- Mount points for various filesystems
- Special filesystems (/proc, /sys, /dev)

## Runtime

The AI runtime provides inference capabilities.

### Inference Engine
- GGUF/GGML model loading
- Quantization support (Q2 to Q8)
- Distributed inference across nodes

### MoE-R Integration
Multiple expert models collaborating:
- Router for query classification
- Synthesizer for response combination
- Distributed expert execution

## Services

All device drivers and system services run in userspace.

### AI Service
- Model loading and management
- Inference request handling
- Distributed cluster coordination

### File System Service
- Actual filesystem implementations
- Block device access
- Caching and buffering

### Network Service
- TCP/IP stack
- P2P networking
- HTTP client/server

## SDK

Provides APIs for application development.

### Modules
- `ai`: AI inference access
- `ipc`: Inter-process communication
- `ui`: Widget toolkit
- `fs`: File operations
- `net`: Networking
- `sys`: System information

## Boot Process

1. **Stage 1 (Assembly)**: MMU init, stack setup, jump to Rust
2. **Stage 2 (Rust)**: Device tree parsing, memory map, load kernel
3. **Kernel Init**: Memory, scheduler, IPC, VFS initialization
4. **Service Startup**: AI, FS, Network services
5. **Shell Launch**: User interface ready

## Hardware Support

### Supported Platforms
| Platform | SoC | RAM | Status |
|----------|-----|-----|--------|
| Raspberry Pi 5 | BCM2712 | 4-8GB | Primary |
| Raspberry Pi 4 | BCM2711 | 2-8GB | Supported |
| Pi Zero 2 W | RP3A0 | 512MB | Limited |
| PinePhone Pro | RK3399S | 4GB | Planned |
| RISC-V boards | Various | Varies | Planned |

## Memory Layout

```
0x0000_0000_0000_0000  +------------------+
                       |     Reserved     |
0x0000_0000_0008_0000  +------------------+
                       |    Bootloader    |
0x0000_0000_0010_0000  +------------------+
                       |      Kernel      |
0x0000_0000_0100_0000  +------------------+
                       |   Kernel Heap    |
0x0000_0000_0500_0000  +------------------+
                       |    AI Models     |
0x0000_0000_2000_0000  +------------------+
                       |   User Space     |
0x0000_FFFF_FFFF_FFFF  +------------------+
```

## Security Model

- Microkernel isolation
- Capability-based security
- Permission system for apps
- Sandboxed userspace services

## Future Directions

- GPU acceleration for inference
- Voice-first interface
- Mesh networking for distributed AI
- Hardware security module support
