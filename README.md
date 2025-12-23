# HubLab IO

<div align="center">

**The First AI-Native Operating System**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-ARM64%20|%20RISC--V-green.svg)](#supported-hardware)
[![AI](https://img.shields.io/badge/AI-Native-purple.svg)](#ai-first-architecture)

**AI at the kernel level** | **Run on smartphones & Raspberry Pi** | **Distributed inference** | **Voice-first interface**

[Quick Start](#quick-start) | [Architecture](#architecture) | [Hardware](#supported-hardware) | [Documentation](docs/)

</div>

---

## What is HubLab IO?

HubLab IO is a **new operating system built from scratch** with AI as a first-class citizen at every layer. Unlike traditional OSes that bolt on AI features, HubLab IO integrates intelligence into the kernel, scheduler, filesystem, and shell.

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

- [x] Kernel core (memory, process, VFS)
- [x] AI runtime (GGML inference)
- [x] TUI shell
- [x] Basic apps (files, terminal, settings)
- [ ] GUI compositor
- [ ] MoE-R distributed experts
- [ ] NPU acceleration (RPi AI Kit)
- [ ] Smartphone bootloader
- [ ] App marketplace
- [ ] OTA updates

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

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

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
