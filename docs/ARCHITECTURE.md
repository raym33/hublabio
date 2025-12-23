# HubLab IO Architecture

This document describes the architecture of HubLab IO, an AI-native operating system.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATIONS                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  Shell  │ │ AI Chat │ │  Files  │ │ Monitor │ │ Settings│ │   Pkg   │  │
│  │TUI/GUI/ │ │         │ │ Manager │ │         │ │         │ │ Manager │  │
│  │ Voice   │ │         │ │         │ │         │ │         │ │         │  │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │
│       └──────────┬┴──────────┬┴──────────┬┴──────────┬┴──────────┬┘        │
│                  │           │           │           │           │          │
│  ┌───────────────┴───────────┴───────────┴───────────┴───────────┴────────┐│
│  │                         HubLab IO SDK                                   ││
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             ││
│  │  │ AI  │ │ IPC │ │ UI  │ │ FS  │ │ Net │ │ Sys │ │ Pkg │             ││
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘             ││
│  └───────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                              SERVICES (Userspace)                            │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │    AI     │ │   File    │ │  Network  │ │  Display  │ │   Voice   │    │
│  │  Service  │ │  System   │ │  Service  │ │  Service  │ │  Service  │    │
│  │  (GGML)   │ │  (VFS)    │ │  (TCP/IP) │ │  (GUI)    │ │ (STT/TTS) │    │
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
│  │ (AI/SMP)  │ │  Manager  │ │  Channels │ │           │ │ Interface │    │
│  │           │ │  (Buddy)  │ │           │ │           │ │  (50+)    │    │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                      HARDWARE ABSTRACTION LAYER                             │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │   GPIO    │ │   UART    │ │   WiFi    │ │    USB    │ │   Audio   │    │
│  │  BCM2711  │ │   PL011   │ │  BCM43xx  │ │ DWC2/XHCI │ │   I2S     │    │
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
- Core affinity for SMP systems

```rust
pub struct AiScheduler {
    ready_queues: [VecDeque<TaskId>; 256],
    tasks: BTreeMap<TaskId, Task>,
    model: TinyTransformer,
    predictions: BTreeMap<TaskId, TaskPrediction>,
    per_cpu: PerCpuData,  // SMP support
}
```

#### Memory Manager
Uses a buddy allocator for physical memory with support for:
- 4KB/16KB/64KB pages
- Slab allocator for kernel objects
- Memory-mapped AI model regions
- Per-CPU memory pools

#### IPC System
Message-passing IPC with:
- Named endpoints for service discovery
- Async message queues
- Zero-copy transfers for large data
- Pipes and FIFOs

#### VFS
Virtual filesystem providing:
- Unified namespace
- Mount points for various filesystems (FAT32, ext4, ramfs)
- Special filesystems (/proc, /sys, /dev)
- Block cache with LRU eviction

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

## Shell

Three interface modes available:

### TUI (Terminal User Interface)
- ANSI escape code rendering
- Theme support (Material Dark, AMOLED, Light)
- Line editor with cursor/history/completion
- Scrollable views and lists

### GUI (Graphical User Interface)
- Window compositor with z-ordering
- Widget toolkit (buttons, labels, text inputs)
- Touch and mouse input support
- Framebuffer rendering

### Voice Interface
- Wake word detection ("Hey HubLab")
- Speech-to-text (Whisper-style)
- Text-to-speech (Piper-style)
- Dialogue management

## Services

All device drivers and system services run in userspace.

### AI Service
- Model loading and management
- Inference request handling
- Distributed cluster coordination

### File System Service
- Actual filesystem implementations (FAT32, ext4)
- Block device access
- Caching and buffering

### Network Service
- Full TCP/IP stack
- WiFi (BCM43xx with WPA2)
- DNS resolver with caching
- DHCP client

### Voice Service
- Audio capture and playback
- Speech recognition
- Speech synthesis
- Wake word detection

## Applications

### Core Apps
- **File Manager**: Navigation, clipboard, bookmarks, search
- **System Monitor**: CPU, memory, processes, network statistics
- **Settings**: System preferences, display, sound, AI, security

### Package Manager
- Package installation and removal
- Repository management
- Dependency resolution
- Version management

## SDK

Provides APIs for application development.

### Modules
- `ai`: AI inference access
- `ipc`: Inter-process communication
- `ui`: Widget toolkit
- `fs`: File operations
- `net`: Networking
- `sys`: System information
- `pkg`: Package management

## Boot Process

1. **Stage 1 (Assembly)**: MMU init, stack setup, jump to Rust
2. **Stage 2 (Rust)**: Device tree parsing, memory map, load kernel
3. **Kernel Init**: Memory, scheduler, IPC, VFS initialization
4. **Service Startup**: AI, FS, Network, Voice services
5. **Shell Launch**: TUI/GUI/Voice interface ready

## Hardware Support

### Supported Platforms
| Platform | SoC | RAM | Status |
|----------|-----|-----|--------|
| Raspberry Pi 5 | BCM2712 | 4-8GB | Primary |
| Raspberry Pi 4 | BCM2711 | 2-8GB | Supported |
| Pi Zero 2 W | RP3A0 | 512MB | Limited |
| PinePhone Pro | RK3399S | 4GB | Planned |
| RISC-V boards | Various | Varies | Planned |

### Driver Support
| Driver | Status | Description |
|--------|--------|-------------|
| GPIO | Complete | BCM2711/BCM2837 |
| UART | Complete | PL011 |
| Framebuffer | Complete | BCM VideoCore |
| WiFi | Complete | BCM43xx with WPA2 |
| USB | Complete | DWC2, XHCI |
| Ethernet | Complete | BCM GENET |
| Audio | Complete | I2S interface |

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
- Process namespacing

## Component Status

| Component | Status |
|-----------|--------|
| Kernel Core | Complete |
| AI Scheduler | Complete |
| Memory Manager | Complete |
| IPC System | Complete |
| VFS | Complete |
| Network Stack | Complete |
| USB Stack | Complete |
| TUI Shell | Complete |
| GUI Compositor | Complete |
| Voice Interface | Complete |
| File Manager | Complete |
| System Monitor | Complete |
| Settings App | Complete |
| Package Manager | Complete |

## Future Directions

- MoE-R distributed experts
- NPU/GPU acceleration for inference
- Mesh networking for distributed AI
- Hardware security module support
- OTA updates
- App marketplace

## Related Documentation

- [Getting Started](GETTING_STARTED.md) - Setup and first steps
- [Prototype Status](PROTOTYPE.md) - Current implementation status
- [Model Training](MODEL_TRAINING.md) - Training custom AI models
- [Jupiter Integration](JUPITER_INTEGRATION.md) - MoE-R and distributed AI code locations
- [Contributing](../CONTRIBUTING.md) - How to contribute
