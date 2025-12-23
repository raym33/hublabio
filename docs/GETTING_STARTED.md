# Getting Started with HubLab IO

This guide will help you set up HubLab IO on your Raspberry Pi or other ARM device.

## Requirements

### Hardware
- Raspberry Pi 5 (recommended), Pi 4, or Pi Zero 2 W
- MicroSD card (32GB+ recommended)
- USB-C power supply
- HDMI display or serial console

### Software
- Rust nightly toolchain
- ARM cross-compiler
- QEMU (for testing)

## Installation

### 1. Install Dependencies

#### macOS
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install ARM toolchain
brew install aarch64-elf-gcc qemu

# Clone repository
git clone https://github.com/raym33/hublabio.git
cd hublabio
```

#### Linux
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install ARM toolchain (Ubuntu/Debian)
sudo apt install gcc-aarch64-linux-gnu qemu-system-arm

# Clone repository
git clone https://github.com/raym33/hublabio.git
cd hublabio
```

### 2. Setup Toolchain

```bash
# Setup development environment
make setup
```

This installs:
- `aarch64-unknown-none` target
- `rust-src` component
- `cargo-binutils`

### 3. Build

```bash
# Build for Raspberry Pi 5
make rpi5

# Or build for another device
make rpi4
make rpi-zero2
```

### 4. Test in Emulator

```bash
# Run in QEMU
make run
```

You should see:
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
[BOOT] Mounting virtual filesystem...
[BOOT] Setting up syscall interface...

[BOOT] Kernel initialization complete!
[BOOT] Starting init process...

hublab>
```

### 5. Flash to SD Card

```bash
# On macOS (replace disk2 with your SD card)
make flash SDCARD=/dev/disk2

# On Linux
sudo dd if=out/rpi5/hublabio.img of=/dev/sdX bs=4M
```

### 6. Boot on Hardware

1. Insert SD card into Raspberry Pi
2. Connect display and power
3. Wait for boot sequence
4. Interact via shell

## First Steps

### Shell Commands

```bash
# Show help
help

# List files
ls /

# Check system info
uname -a

# Ask AI a question
?what is the meaning of life

# Check processes
ps

# Exit shell
exit
```

### AI Chat

Start an AI chat session:

```bash
# Enter AI mode
ai

# Or ask directly
?explain quantum computing
```

### Package Manager

```bash
# List installed packages
pkg list

# Search for packages
pkg search editor

# Install a package
pkg install nano
```

## Development

### Create a New App

1. Create app directory:
```bash
mkdir -p apps/myapp
cd apps/myapp
```

2. Create `Cargo.toml`:
```toml
[package]
name = "myapp"
version = "0.1.0"
edition = "2021"

[dependencies]
hublabio-sdk = { path = "../../sdk" }
```

3. Create `src/main.rs`:
```rust
use hublabio_sdk::prelude::*;

fn main() -> Result<()> {
    // Connect to AI
    let ai = AiClient::connect()?;

    // Generate response
    let response = ai.generate("Hello, HubLab IO!")?;
    println!("{}", response);

    Ok(())
}
```

4. Build:
```bash
cargo build --target aarch64-unknown-none
```

### Debugging

```bash
# Start QEMU with GDB server
make debug

# In another terminal
aarch64-none-elf-gdb target/aarch64-unknown-none/debug/hublabio-kernel
(gdb) target remote :1234
(gdb) continue
```

## Configuration

### System Config

Edit `config/system.toml`:
```toml
[kernel]
log_level = "info"
heap_size = "64M"

[scheduler]
ai_enabled = true
model_path = "/models/scheduler.gguf"

[display]
resolution = "1920x1080"
theme = "material-dark"
```

### AI Models

Place GGUF models in `/models/`:
```
/models/
├── scheduler.gguf      # Scheduler AI (1M params)
├── chat.gguf          # Chat assistant
└── embeddings.gguf    # For semantic search
```

## Troubleshooting

### Kernel Panic
- Check memory configuration
- Verify device tree is correct
- Enable verbose boot: `make run VERBOSE=1`

### No Display Output
- Try serial console instead
- Check HDMI cable
- Verify framebuffer init in device tree

### AI Not Working
- Ensure model file exists
- Check available memory
- Verify model format (GGUF only)

## Next Steps

- [Architecture Overview](ARCHITECTURE.md)
- [SDK Documentation](SDK.md)
- [Contributing Guide](../CONTRIBUTING.md)
