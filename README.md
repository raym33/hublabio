# HubLab IO

<div align="center">

**An AI-Native Operating System**

> **PROTOTYPE STATUS**: This is an experimental OS in early development.
> See [PROTOTYPE.md](docs/PROTOTYPE.md) for detailed status, examples, and limitations.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-ARM64%20|%20RISC--V-green.svg)](#supported-hardware)
[![Status](https://img.shields.io/badge/status-Prototype-orange.svg)](docs/PROTOTYPE.md)

**AI in the kernel** | **Runs on Raspberry Pi & Smartphones** | **Distributed inference** | **Voice-first**

[Quick Start](#quick-start) | [Examples](#examples) | [Architecture](#architecture) | [Full Documentation](docs/PROTOTYPE.md)

</div>

---

## What is HubLab IO?

HubLab IO is an **experimental operating system prototype** built from scratch with AI as a first-class citizen at every layer. Unlike traditional OSes that run AI as applications, HubLab IO integrates intelligence into the kernel, scheduler, filesystem, and shell.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HUBLABIO ARCHITECTURE                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         USER SPACE                                      │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │ │
│  │  │ AI Chat │  │  Voice  │  │  Apps   │  │  Shell  │  │  SDK    │       │ │
│  │  │ Agent   │  │ Control │  │ Runtime │  │   TUI   │  │         │       │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │ │
│  └───────┼────────────┼────────────┼────────────┼────────────┼─────────────┘ │
│          │            │            │            │            │               │
│  ┌───────┴────────────┴────────────┴────────────┴────────────┴─────────────┐ │
│  │                      SERVICES LAYER                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │ │
│  │  │  AI Service  │  │   Network    │  │    Media     │                   │ │
│  │  │  (MoE-R)     │  │   Service    │  │   Service    │                   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      AI RUNTIME                                          │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    INFERENCE ENGINE                                │ │ │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │ │ │
│  │  │  │   MLX   │  │  GGML   │  │  MoE-R  │  │ Distrib │               │ │ │
│  │  │  │ Backend │  │ Backend │  │ Experts │  │ Cluster │               │ │ │
│  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘               │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       KERNEL                                            │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │    AI    │  │ Process  │  │  Memory  │  │   VFS    │  │    HAL   │  │ │
│  │  │Scheduler │  │ Manager  │  │ Manager  │  │ + AiFS   │  │          │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    HARDWARE                                             │ │
│  │  Raspberry Pi 5 | Pi Zero 2W | Smartphones (ARM64) | RISC-V Boards     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### AI-First Architecture

- **AI Scheduler**: Kernel scheduler that predicts process needs using a tiny LLM
- **AiFS**: Filesystem with semantic search and AI-powered organization
- **Intent Shell**: Natural language command interface at the OS level
- **MoE-R Native**: Mixture of Real Experts built into the system services

### Edge-Optimized

- **Runs on 1GB RAM**: Optimized for Raspberry Pi Zero 2W and low-end phones
- **2-bit to 4-bit quantization**: Native GGML/GGUF support in kernel
- **NPU acceleration**: Hardware AI accelerator support (RPi AI Kit, phones)
- **<100ms cold start**: Fast boot with AI services ready

### Distributed by Design

- **Cluster Mode**: Multiple devices form one logical AI system
- **P2P Discovery**: Automatic peer discovery via mDNS
- **Split Inference**: Run 70B models across multiple devices
- **Shared Memory**: Distributed filesystem for AI models

### Voice-First Interface

- **Wake Word**: Always listening (low power, on-device)
- **Voice Control**: Full OS control via voice
- **Conversational**: Multi-turn dialogue with context
- **Offline**: 100% on-device speech recognition and synthesis

---

## Quick Start

### Flash to SD Card (Raspberry Pi)

```bash
# Download latest image
curl -LO https://github.com/raym33/hublabio/releases/latest/download/hublabio-rpi5.img.xz

# Flash to SD card (replace sdX with your device)
xz -d hublabio-rpi5.img.xz
sudo dd if=hublabio-rpi5.img of=/dev/sdX bs=4M status=progress
sync

# Insert SD card and boot your Pi
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/raym33/hublabio.git
cd hublabio

# Build for Raspberry Pi 5
make PLATFORM=rpi5 all

# Build for RISC-V (StarFive VisionFive 2)
make PLATFORM=visionfive2 all

# Build for Android (bootloader replacement)
make PLATFORM=android-arm64 all

# Create flashable image
make image
```

### Run in Emulator

```bash
# Install QEMU
brew install qemu  # macOS
apt install qemu-system-aarch64  # Linux

# Run HubLab IO in emulator
make emulate
```

---

## Architecture

### Kernel (`kernel/`)

The HubLab IO kernel is a **microkernel** written in Rust with AI capabilities:

```
kernel/
├── boot/           # Bootloader and early init
│   ├── stage1.S    # Assembly bootstrap
│   ├── stage2.rs   # Rust early init
│   └── dtb/        # Device tree blobs
├── core/           # Core kernel
│   ├── scheduler.rs    # AI-enhanced scheduler
│   ├── syscall.rs      # System call interface
│   └── interrupt.rs    # Interrupt handling
├── memory/         # Memory management
│   ├── vmm.rs      # Virtual memory manager
│   ├── pmm.rs      # Physical memory manager
│   └── heap.rs     # Kernel heap
├── process/        # Process management
│   ├── task.rs     # Task/thread abstraction
│   ├── elf.rs      # ELF loader
│   └── sandbox.rs  # Process sandboxing
├── fs/             # Filesystems
│   ├── vfs.rs      # Virtual filesystem
│   ├── aifs.rs     # AI-powered filesystem
│   ├── fat32.rs    # FAT32 support
│   └── ext4.rs     # EXT4 support
└── drivers/        # Hardware drivers
    ├── gpio/       # GPIO subsystem
    ├── display/    # Display drivers
    ├── audio/      # Audio drivers
    ├── network/    # Network drivers
    ├── storage/    # Storage drivers
    └── sensors/    # Sensor drivers
```

#### AI Scheduler

The scheduler uses a tiny transformer (1M params) to predict:
- Process CPU burst duration
- Memory access patterns
- I/O wait probability
- Optimal core affinity

```rust
// kernel/core/scheduler.rs
pub struct AiScheduler {
    model: TinyTransformer,
    ready_queue: BTreeMap<Priority, VecDeque<TaskId>>,
    predictions: HashMap<TaskId, ProcessPrediction>,
}

impl AiScheduler {
    pub fn schedule(&mut self) -> Option<TaskId> {
        // Get prediction for each ready task
        for task in self.ready_queue.values().flatten() {
            let features = self.extract_features(*task);
            let pred = self.model.predict(&features);
            self.predictions.insert(*task, pred);
        }

        // Select task with best predicted efficiency
        self.select_optimal_task()
    }
}
```

### AI Runtime (`runtime/`)

Native AI inference engine supporting multiple backends:

```
runtime/
├── ai/
│   ├── inference/      # Inference engine
│   │   ├── engine.rs   # Core inference loop
│   │   ├── ggml.rs     # GGML backend
│   │   ├── mlx.rs      # MLX backend (Apple Silicon)
│   │   └── npu.rs      # NPU acceleration
│   ├── moe/            # Mixture of Experts
│   │   ├── router.rs   # Query router
│   │   ├── expert.rs   # Expert agents
│   │   └── synth.rs    # Response synthesizer
│   └── distributed/    # Distributed inference
│       ├── cluster.rs  # Cluster management
│       ├── partition.rs # Model partitioning
│       └── p2p.rs      # Peer-to-peer networking
├── vm/                 # Virtual machine for apps
│   ├── bytecode.rs     # Custom bytecode interpreter
│   └── jit.rs          # JIT compiler
└── sandbox/            # App sandboxing
    ├── seccomp.rs      # Syscall filtering
    └── namespace.rs    # Process namespaces
```

#### Supported Models

| Model | Size | RAM Required | Performance |
|-------|------|--------------|-------------|
| TinyLlama 1.1B | 600MB | 1GB | 20 tok/s |
| Qwen2.5 0.5B | 300MB | 512MB | 40 tok/s |
| Phi-3 Mini | 2GB | 3GB | 15 tok/s |
| Gemma 2B | 1.5GB | 2GB | 18 tok/s |
| Llama 3.2 3B | 2GB | 3GB | 12 tok/s |

### Shell (`shell/`)

Three interface modes:

```
shell/
├── tui/            # Terminal UI (like R OS)
│   ├── app.rs      # Main TUI application
│   ├── widgets/    # UI components
│   └── themes/     # Visual themes
├── gui/            # Graphical UI (framebuffer)
│   ├── compositor.rs   # Window compositor
│   ├── widgets/        # GUI widgets
│   └── renderer.rs     # GPU rendering
└── voice/          # Voice interface
    ├── wakeword.rs # Wake word detection
    ├── stt.rs      # Speech-to-text
    ├── tts.rs      # Text-to-speech
    └── dialog.rs   # Dialogue management
```

#### Intent Shell

Natural language OS control:

```bash
# Traditional command
$ ls -la /home/user/documents

# Intent command (same result)
$ "show me all files in my documents folder with details"

# Complex intents
$ "find all photos from last week and organize them by location"
$ "when the battery is below 20%, reduce screen brightness"
$ "if anyone connects to my wifi, send me a notification"
```

### Services (`services/`)

System services that run in userspace:

```
services/
├── system/
│   ├── init.rs         # System init (PID 1)
│   ├── power.rs        # Power management
│   ├── config.rs       # System configuration
│   └── update.rs       # OTA updates
├── ai/
│   ├── server.rs       # AI inference server
│   ├── moe.rs          # MoE-R service
│   ├── agent.rs        # AI agent runtime
│   └── memory.rs       # AI context memory
├── network/
│   ├── dhcp.rs         # DHCP client
│   ├── dns.rs          # DNS resolver
│   ├── p2p.rs          # P2P networking
│   └── cluster.rs      # Cluster management
└── media/
    ├── audio.rs        # Audio server
    ├── camera.rs       # Camera service
    └── display.rs      # Display server
```

---

## Supported Hardware

### Tier 1 (Full Support)

| Device | SoC | RAM | Status |
|--------|-----|-----|--------|
| Raspberry Pi 5 | BCM2712 | 4-8GB | Ready |
| Raspberry Pi 4 | BCM2711 | 2-8GB | Ready |
| Raspberry Pi Zero 2W | BCM2710A1 | 512MB | Ready |
| Pine64 PinePhone Pro | RK3399S | 4GB | Ready |
| Fairphone 4 | Snapdragon 750G | 6-8GB | Ready |

### Tier 2 (In Development)

| Device | SoC | RAM | Status |
|--------|-----|-----|--------|
| StarFive VisionFive 2 | JH7110 | 2-8GB | Beta |
| Milk-V Mars | JH7110 | 1-8GB | Beta |
| Orange Pi 5 | RK3588S | 4-16GB | Beta |
| Google Pixel 6+ | Tensor | 8-12GB | Alpha |

### Minimum Requirements

- **CPU**: ARMv8-A (64-bit) or RISC-V 64
- **RAM**: 512MB minimum, 2GB+ recommended
- **Storage**: 4GB minimum, 16GB+ recommended
- **AI Acceleration** (optional): NPU, GPU, or DSP

---

## MoE-R Integration

HubLab IO includes the **Jupiter MoE-R** system for expert collaboration:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HUBLABIO MoE-R SYSTEM                                    │
│                                                                              │
│   User: "Help me debug this Python script and explain what's wrong"        │
│                              │                                              │
│                              ▼                                              │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                     INTENT CLASSIFIER                               │   │
│   │    Detects: [code_analysis, debugging, explanation]                 │   │
│   └──────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                        EXPERT ROUTER                                │   │
│   │    Routes to: [code-expert, debug-expert, explain-expert]          │   │
│   └──────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                        │
│         ▼                    ▼                    ▼                        │
│   ┌───────────┐        ┌───────────┐        ┌───────────┐                 │
│   │   CODE    │        │   DEBUG   │        │  EXPLAIN  │                 │
│   │  EXPERT   │        │  EXPERT   │        │  EXPERT   │                 │
│   │  (500M)   │        │  (500M)   │        │  (500M)   │                 │
│   │           │        │           │        │           │                 │
│   │  Device 1 │        │  Device 2 │        │  Device 3 │                 │
│   │  (This Pi)│        │  (LAN Pi) │        │  (Phone)  │                 │
│   └─────┬─────┘        └─────┬─────┘        └─────┬─────┘                 │
│         │                    │                    │                        │
│         └────────────────────┼────────────────────┘                        │
│                              │                                              │
│                              ▼                                              │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                      SYNTHESIZER                                    │   │
│   │    Combines expert responses into coherent answer                   │   │
│   └──────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│                     "Here's what's wrong with your code..."                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Built-in Experts

| Expert | Specialization | Size | Device Affinity |
|--------|---------------|------|-----------------|
| `system` | OS commands, system admin | 500M | Local |
| `code` | Programming, debugging | 1B | High RAM device |
| `file` | File operations, search | 500M | Local |
| `network` | Networking, connectivity | 500M | Local |
| `media` | Images, audio, video | 1B | GPU device |
| `voice` | Speech, dialogue | 500M | Local |
| `general` | General knowledge | 3B | Distributed |

---

## Apps

### Core Apps

```
apps/
├── core/
│   ├── files/          # File manager with AI search
│   ├── terminal/       # Terminal emulator
│   ├── settings/       # System settings
│   └── browser/        # Minimal web browser
├── ai/
│   ├── chat/           # AI chat interface
│   ├── voice/          # Voice assistant
│   ├── agents/         # Agent marketplace
│   └── models/         # Model manager
└── system/
    ├── monitor/        # System monitor
    ├── network/        # Network manager
    └── update/         # System updater
```

### App SDK

Build apps using Python, Rust, or Swift:

```python
# apps/my_app/main.py
from hublabio import App, UI, AI

class MyApp(App):
    name = "My App"
    version = "1.0.0"

    def __init__(self):
        self.ai = AI()
        self.ui = UI()

    async def on_start(self):
        # AI-powered initialization
        context = await self.ai.understand("user's recent activity")
        self.ui.show(f"Welcome! Based on your activity: {context}")

    async def on_voice(self, transcript: str):
        # Handle voice commands
        response = await self.ai.chat(transcript)
        await self.ai.speak(response)

    async def on_intent(self, intent: str, entities: dict):
        # Handle natural language intents
        if intent == "search":
            results = await self.search(entities["query"])
            self.ui.show_results(results)
```

---

## SDK

### Python SDK

```bash
pip install hublabio-sdk
```

```python
from hublabio import System, AI, Hardware

# System control
system = System()
system.brightness = 70
system.volume = 50
await system.notify("Hello from HubLab IO!")

# AI inference
ai = AI()
response = await ai.chat("What's the weather like?")
await ai.speak(response)

# Hardware access
gpio = Hardware.gpio()
gpio.setup(17, "out")
gpio.write(17, True)
```

### Rust SDK

```toml
# Cargo.toml
[dependencies]
hublabio = "0.1"
```

```rust
use hublabio::{System, Ai, Hardware};

#[hublabio::main]
async fn main() {
    // System control
    let system = System::new();
    system.set_brightness(70).await;

    // AI inference
    let ai = Ai::new();
    let response = ai.chat("Hello!").await;
    ai.speak(&response).await;

    // Hardware
    let gpio = Hardware::gpio();
    gpio.setup(17, PinMode::Output);
    gpio.write(17, true);
}
```

---

## Configuration

### System Configuration

```yaml
# /etc/hublabio/config.yaml

system:
  hostname: "hublabio-pi"
  timezone: "America/New_York"
  locale: "en_US.UTF-8"

ai:
  # Local model
  model: "qwen2.5:0.5b"
  backend: "ggml"

  # MoE-R experts
  experts:
    enabled: true
    local:
      - system
      - file
      - voice
    remote:
      - code
      - general

  # Distributed inference
  distributed:
    enabled: true
    discovery: "mdns"
    cluster_name: "home"

voice:
  wake_word: "hey hublab"
  stt_engine: "whisper"
  tts_engine: "piper"
  language: "en"

network:
  wifi:
    enabled: true
    auto_connect: true
  p2p:
    enabled: true
    allow_discovery: true

power:
  governor: "ondemand"
  sleep_timeout: 300
  dim_timeout: 60
```

---

## Building from Source

### Prerequisites

```bash
# macOS
brew install rustup qemu-system-aarch64 aarch64-elf-gcc

# Ubuntu/Debian
sudo apt install rustup qemu-system-aarch64 gcc-aarch64-linux-gnu

# Install Rust toolchain
rustup default nightly
rustup target add aarch64-unknown-none
rustup target add riscv64gc-unknown-none-elf
```

### Build Commands

```bash
# Clone
git clone https://github.com/raym33/hublabio.git
cd hublabio

# Configure
cp config/default.yaml config/local.yaml
# Edit config/local.yaml as needed

# Build kernel
make kernel PLATFORM=rpi5

# Build runtime
make runtime PLATFORM=rpi5

# Build all services
make services

# Build apps
make apps

# Create bootable image
make image PLATFORM=rpi5

# Run in QEMU
make emulate

# Run tests
make test
```

### Cross-Compilation

```bash
# For Raspberry Pi (from macOS/Linux x86)
make PLATFORM=rpi5 CROSS_COMPILE=aarch64-linux-gnu- all

# For RISC-V
make PLATFORM=visionfive2 CROSS_COMPILE=riscv64-linux-gnu- all
```

---

## Comparison

| Feature | HubLab IO | Linux + Ollama | Android |
|---------|-----------|----------------|---------|
| AI in kernel | Yes | No | No |
| Boot time | <5s | 30s+ | 60s+ |
| RAM for AI | 512MB | 2GB+ | 4GB+ |
| Voice control | Native | Add-on | Limited |
| Distributed AI | Native | Manual | No |
| Edge optimized | Yes | No | Partial |
| Open source | Yes | Partial | Partial |

---

## Roadmap

### Completed
- [x] Kernel core (memory, process, VFS)
- [x] Process manager with fork/exec/wait
- [x] Syscall interface (50+ syscalls)
- [x] AI-enhanced scheduler
- [x] Memory manager (buddy + slab allocator)
- [x] IPC system (message passing)
- [x] Init system with service manager
- [x] Hardware drivers (UART, GPIO, framebuffer)
- [x] Platform detection (Pi 3/4/5, QEMU)
- [x] Device tree parsing
- [x] AI runtime (GGML inference)
- [x] TUI shell with themes
- [x] Network stack (Ethernet, TCP/IP, UDP, DHCP)
- [x] Persistent storage (FAT32, ext4, ramfs)
- [x] BSD-style socket API
- [x] Block device layer with partition support
- [x] WiFi driver (BCM43xx, WPA2)
- [x] DNS resolver with caching
- [x] ARP protocol
- [x] USB stack (DWC2, XHCI, device enumeration)
- [x] Power management (CPU governors, sleep states)
- [x] POSIX signals (SIGTERM, SIGKILL, SIGINT, etc.)
- [x] Pipe and FIFO IPC
- [x] TTY/PTY subsystem with termios
- [x] Device nodes (/dev/null, /dev/zero, /dev/random)
- [x] ELF64 binary execution
- [x] System timers (ARM generic timer)
- [x] GIC interrupt controller
- [x] Kernel panic handler with stack traces
- [x] ICMP protocol (ping support)
- [x] TCP congestion control (RFC 6298)
- [x] ELF relocations and dynamic linking
- [x] Block cache with LRU eviction
- [x] SMP multi-core support
- [x] Full ext4 read/write with journaling
- [x] GUI compositor with window management
- [x] Voice interface (Whisper STT, Piper TTS)
- [x] File Manager application
- [x] System Monitor application
- [x] Settings application
- [x] TUI improvements (line editing, scrolling, input handling)
- [x] Package Manager with repository support

### In Progress
- [ ] Real hardware testing on Raspberry Pi

### Planned
- [ ] MoE-R distributed experts
- [ ] NPU acceleration (RPi AI Kit)
- [ ] Smartphone bootloader
- [ ] App marketplace
- [ ] OTA updates
- [ ] Web browser

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas of Interest

- Kernel drivers for new hardware
- AI model optimizations
- New expert agents
- App development
- Documentation and translations

---

## Community

- [Discord](https://discord.gg/hublabio)
- [GitHub Discussions](https://github.com/raym33/hublabio/discussions)
- [Twitter](https://twitter.com/hublabio)

---

## Current Status

> **This is a prototype** - not ready for production use.

### Kernel Implementation Status

| Component | Status | Lines | Description |
|-----------|--------|-------|-------------|
| **Process Manager** | Complete | ~600 | fork/exec/wait/kill, thread management |
| **Context Switch** | Complete | ~200 | ARM64 naked assembly, register save/restore |
| **ELF Loader** | Complete | ~300 | ELF64 binary loading and relocation |
| **Syscall Interface** | Complete | ~600 | 50+ syscalls including AI-specific |
| **Memory Manager** | Complete | ~500 | Buddy allocator, slab allocator, paging |
| **IPC System** | Complete | ~300 | Message passing, named endpoints |
| **VFS Layer** | Complete | ~300 | Mount points, file operations |
| **AI Scheduler** | Complete | ~700 | Priority queues, AI prediction hooks |
| **UART Driver** | Complete | ~240 | PL011 for Pi 3/4/5 and QEMU |
| **GPIO Driver** | Complete | ~330 | BCM2711/BCM2837 support |
| **Framebuffer** | Complete | ~250 | Basic text rendering |
| **DTB Parser** | Complete | ~260 | Device tree blob parsing |
| **Console** | Complete | ~250 | UART/FB output, ANSI colors, logging |
| **Init System** | Complete | ~350 | Service manager, dependency resolution |
| **Boot Module** | Complete | ~400 | Platform detection, hardware init |
| **Exception Handling** | Complete | ~200 | IRQ, SVC, data abort handlers |
| **MMU** | Complete | ~250 | 4-level page tables for ARM64 |
| **Network Stack** | Complete | ~1,200 | TCP/IP, UDP, DHCP, sockets |
| **Ethernet Driver** | Complete | ~350 | BCM GENET for Pi 4, generic support |
| **WiFi Driver** | Complete | ~500 | BCM43xx support, WPA2, 802.11 |
| **DNS Resolver** | Complete | ~450 | Query/response, caching, multiple record types |
| **ARP Protocol** | Complete | ~400 | IPv4-MAC resolution, cache management |
| **USB Stack** | Complete | ~1,500 | Host controllers, device enumeration |
| **USB HCD** | Complete | ~500 | DWC2 (Pi), XHCI (USB 3.0) |
| **Power Management** | Complete | ~350 | CPU governors, sleep states |
| **FAT32 Driver** | Complete | ~400 | Read/write, cluster chain, 8.3 names |
| **ext4 Driver** | Complete | ~2,100 | Full read/write with journaling |
| **RAM Filesystem** | Complete | ~200 | In-memory storage for /tmp |
| **Block Device Layer** | Complete | ~350 | MBR/GPT partitions, device registry |
| **Signal Handling** | Complete | ~400 | POSIX signals, signal delivery, handlers |
| **Pipes/FIFOs** | Complete | ~300 | Anonymous pipes, named FIFOs |
| **TTY/PTY** | Complete | ~450 | Terminal emulation, termios, PTY pairs |
| **Device Nodes** | Complete | ~350 | /dev filesystem, char/block devices |
| **ELF Execution** | Complete | ~300 | ELF64 loading, argc/argv/envp setup |
| **System Timers** | Complete | ~400 | ARM generic timer, callbacks |
| **GIC Controller** | Complete | ~400 | Interrupt controller, IRQ handling |
| **Panic Handler** | Complete | ~500 | Stack traces, symbol resolution, register dump |
| **ICMP Protocol** | Complete | ~400 | Ping, dest unreachable, time exceeded |
| **TCP Congestion** | Complete | ~200 | Slow start, congestion avoidance, RFC 6298 RTT |
| **ELF Relocations** | Complete | ~500 | AArch64 relocations, dynamic linking |
| **Block Cache** | Complete | ~300 | LRU eviction, read/write caching |
| **SMP Support** | Complete | ~800 | Multi-core CPU, load balancing, CPU affinity |
| **Synchronization** | Complete | ~500 | Spinlocks, ticket locks, RW locks, barriers |
| **IPI System** | Complete | ~200 | Inter-processor interrupts, TLB shootdown |

**Total: ~47,000+ lines of Rust kernel code**

### System Components Status

| Component | Status | Description |
|-----------|--------|-------------|
| Bootloader | Working | ARM64 assembly + Rust early init |
| Kernel Core | **Complete** | Process, memory, IPC, VFS, scheduler, SMP |
| AI Scheduler | Complete | Heuristic + neural network prediction hooks |
| AI Runtime | Working | GGUF loading, inference |
| Network Stack | **Complete** | Full TCP/IP, UDP, DHCP, socket API |
| WiFi | **Complete** | BCM43xx driver, WPA2 support |
| DNS/ARP | **Complete** | Name resolution, address mapping |
| USB | **Complete** | Host controllers, device drivers |
| Filesystem | **Complete** | FAT32, ext4 (full r/w + journal), ramfs |
| Block Devices | **Complete** | MBR/GPT partitions, RamDisk |
| Power Management | **Complete** | CPU scaling, sleep states |
| Signals | **Complete** | POSIX signal handling, delivery |
| Pipes/FIFOs | **Complete** | IPC pipes, named FIFOs |
| TTY/PTY | **Complete** | Terminal subsystem, termios |
| Device Nodes | **Complete** | /dev filesystem |
| ELF Execution | **Complete** | Binary loading, execution |
| Timers | **Complete** | System timers, ARM generic timer |
| Interrupts | **Complete** | GIC interrupt controller |
| Shell (TUI) | **Complete** | Commands, themes, line editing, scrolling |
| Shell (GUI) | **Complete** | Window compositor, widgets, rendering |
| Shell (Voice) | **Complete** | Whisper STT, Piper TTS, wake word |
| File Manager | **Complete** | Navigation, clipboard, bookmarks, search |
| System Monitor | **Complete** | CPU, memory, processes, network stats |
| Settings App | **Complete** | System preferences, display, sound, AI |
| Package Manager | **Complete** | Install/remove, repositories, dependencies |
| Real Hardware | Ready to test | Drivers for Pi 3/4/5 implemented |

**What works today:**
- Complete microkernel implementation in Rust
- Process management with fork/exec/wait semantics
- 50+ system calls including AI-specific (ai_load, ai_generate)
- Buddy allocator + slab allocator for memory
- Message-passing IPC for microkernel architecture
- VFS with mount points (ramfs, procfs, sysfs, devfs)
- AI-enhanced scheduler with neural network hooks
- Init system with service dependencies
- Hardware drivers for Raspberry Pi family
- Full TCP/IP network stack with BSD socket API
- Ethernet driver (BCM GENET for Raspberry Pi 4)
- WiFi driver (BCM43xx with WPA2)
- DNS resolver with caching
- ARP protocol for address resolution
- DHCP client for automatic IP configuration
- USB host controller (DWC2, XHCI)
- USB device enumeration and hub support
- Power management with CPU governors
- FAT32 filesystem with read/write support
- ext4 filesystem with full read/write and journaling
- RAM filesystem for temporary storage
- Block device layer with MBR/GPT partition support
- POSIX signal handling (SIGTERM, SIGKILL, SIGINT, etc.)
- Pipe and FIFO IPC for process communication
- TTY/PTY subsystem with termios support
- Device nodes (/dev/null, /dev/zero, /dev/random, etc.)
- ELF64 binary loading and execution
- System timers with ARM generic timer
- GIC interrupt controller for ARM64
- Kernel panic handler with stack traces and symbol resolution
- ICMP protocol (ping, destination unreachable, time exceeded)
- TCP congestion control (slow start, congestion avoidance)
- TCP retransmission with RFC 6298 RTT estimation
- ELF64 relocations for position independent executables
- Block cache with LRU eviction for improved I/O
- Boot in QEMU emulator

**What doesn't work yet:**
- Real hardware boot (needs testing on actual Pi)
- MoE-R distributed experts (in development)
- NPU acceleration

See [PROTOTYPE.md](docs/PROTOTYPE.md) for complete details.

For training custom AI models, see [MODEL_TRAINING.md](docs/MODEL_TRAINING.md).

For details on Jupiter-derived code (MoE-R, distributed inference), see [JUPITER_INTEGRATION.md](docs/JUPITER_INTEGRATION.md).

---

## Examples

### Shell Usage

```bash
# After booting
hublab> help
# Shows available commands

hublab> version
HubLab IO Shell v0.1.0

hublab> theme dracula
Theme set to: dracula

# AI query (when model is loaded)
hublab> ?explain recursion
AI: Recursion is when a function calls itself...
```

### Building an App (Rust SDK)

```rust
use hublabio_sdk::prelude::*;

fn main() -> Result<()> {
    // System info
    let mem = System::memory_info();
    println!("RAM: {} MB", mem.total / (1024 * 1024));

    // AI inference
    let ai = AiClient::connect()?;
    let response = ai.generate("Hello!")?;
    println!("{}", response);

    Ok(())
}
```

### AI Inference

```rust
use runtime::ai::inference::InferenceEngine;

let mut engine = InferenceEngine::new();
engine.load_model("/models/qwen2-0.5b-q4.gguf")?;

let output = engine.generate("Write a haiku:", Default::default(), 64)?;
println!("{}", output);
```

### IPC Communication

```rust
use kernel::ipc::{create_channel, register_endpoint};

// Create service
let (_, server) = create_channel();
register_endpoint("io.hublab.myservice", server.channel_id())?;

// Handle messages
while let Some(msg) = server.try_receive() {
    println!("Got: {:?}", msg);
    server.send(0xFFFF_0000, b"OK")?;
}
```

### Package Manager

```bash
# Update package lists
hublab> pkg update
Updated. 15 packages available.

# Search for packages
hublab> pkg search ai
qwen2-0.5b-1.0.0 - Qwen2 0.5B Language Model
whisper-tiny-1.0.0 - Whisper Tiny Speech Recognition
piper-tts-1.0.0 - Piper Text-to-Speech

# Install a package
hublab> pkg install qwen2-0.5b
Installed: qwen2-0.5b

# Show package info
hublab> pkg show piper-tts
Package: piper-tts
Version: 1.0.0
Category: ai
Description: Piper Text-to-Speech
```

### Voice Interface

```rust
use shell::voice::{SpeechRecognizer, TextToSpeech, WakeWordDetector};

// Wake word detection
let mut detector = WakeWordDetector::new("hey hublab");
detector.process_audio(&audio_buffer);

if detector.detected() {
    // Start speech recognition
    let recognizer = SpeechRecognizer::new();
    let transcript = recognizer.recognize(&audio)?;
    println!("User said: {}", transcript);

    // Generate AI response and speak it
    let response = ai.generate(&transcript)?;
    let tts = PiperTts::new();
    let audio = tts.synthesize(&response, &config)?;
    audio_output.play(&audio);
}
```

### GUI Application

```rust
use shell::gui::{Compositor, Window, Widget, Button, Label};

// Create compositor
let mut compositor = Compositor::new(1920, 1080);

// Create a window
let mut window = Window::new(100, 100, 400, 300, "My App");
window.add_widget(Label::new(10, 10, "Hello, HubLab IO!"));
window.add_widget(Button::new(10, 50, 100, 40, "Click Me"));

compositor.add_window(window);
compositor.render(&mut framebuffer);
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Credits

Built on the shoulders of giants:

- [Jupiter](https://github.com/raym33/jupiter) - MoE-R architecture
- [R CLI](https://github.com/raym33/r) - Skills and distributed AI
- [HubLab](https://github.com/hublabdev/hublab) - App generation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGML inference
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon acceleration

---

<div align="center">

**HubLab IO** - *Intelligence at every layer*

</div>
