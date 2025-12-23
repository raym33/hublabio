# Getting Started with HubLab IO

This guide will help you set up HubLab IO on your Raspberry Pi or other ARM device.

## Requirements

### Hardware
- Raspberry Pi 5 (recommended), Pi 4, or Pi Zero 2 W
- MicroSD card (32GB+ recommended)
- USB-C power supply
- HDMI display or serial console
- Optional: USB microphone and speakers for voice interface

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

### Theme Switching

```bash
# Change to AMOLED theme
hublab> theme amoled
Theme set to: amoled

# Change to light theme
hublab> theme light
Theme set to: light

# Change back to Material Dark
hublab> theme material-dark
Theme set to: material-dark
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
# Update package lists
pkg update

# List available packages
pkg list

# Search for packages
pkg search editor

# Install a package
pkg install nano

# Show package info
pkg show qwen2-0.5b

# List installed packages
pkg list-installed

# Remove a package
pkg remove nano
```

### Voice Commands

If you have a microphone connected:

```bash
# Say "Hey HubLab" to activate
# Then speak your command

# Example voice commands:
# "Open file manager"
# "What's the weather like?"
# "Set brightness to 50 percent"
```

### Using the GUI

```bash
# Launch GUI mode
gui

# Navigation:
# - Use arrow keys or touch to navigate
# - Click/tap to select apps
# - Swipe down for notifications
```

## System Apps

### File Manager

```bash
# Open file manager
files

# Keyboard shortcuts:
# Enter    - Open file/folder
# Backspace - Go back
# Ctrl+C   - Copy
# Ctrl+V   - Paste
# Ctrl+X   - Cut
# Del      - Delete
# Ctrl+B   - Add bookmark
# /        - Search
```

### System Monitor

```bash
# Open system monitor
monitor

# Views:
# 1 - CPU usage
# 2 - Memory usage
# 3 - Processes
# 4 - Network
# 5 - Disk
```

### Settings

```bash
# Open settings
settings

# Categories:
# - General (hostname, timezone, locale)
# - Display (brightness, theme, font size)
# - Sound (volume, TTS voice)
# - Network (WiFi, Bluetooth)
# - AI (model, temperature, context)
# - Security (password, encryption)
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

### Using the Package Manager API

```rust
use hublabio_pkg::{PackageManager, init};

fn main() {
    // Initialize package manager
    let mut pm = init("/");

    // Update package lists
    pm.update().expect("Update failed");

    // Search for packages
    let results = pm.search("ai");
    for pkg in results {
        println!("{} - {}", pkg.full_name(), pkg.description);
    }

    // Install a package
    pm.install("qwen2-0.5b").expect("Install failed");
}
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

[voice]
wake_word = "hey hublab"
stt_engine = "whisper"
tts_engine = "piper"
```

### AI Models

Place GGUF models in `/models/`:
```
/models/
├── scheduler.gguf      # Scheduler AI (1M params)
├── chat.gguf          # Chat assistant
├── whisper-tiny.bin   # Speech recognition
├── piper-en.onnx      # Text-to-speech
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

### Voice Not Responding
- Check microphone connection
- Verify audio drivers loaded
- Try: `hublab> voice test`

### Package Manager Errors
- Run `pkg update` first
- Check network connectivity
- Verify repository URLs

## Next Steps

- [Architecture Overview](ARCHITECTURE.md)
- [Model Training Guide](MODEL_TRAINING.md) - Train custom 1B-8B models
- [Jupiter Integration](JUPITER_INTEGRATION.md) - MoE-R and distributed AI code details
- [Prototype Status](PROTOTYPE.md)
- [Contributing Guide](../CONTRIBUTING.md)

## Quick Reference

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `version` | Show version info |
| `theme <name>` | Change visual theme |
| `?<query>` | Ask AI a question |
| `pkg <cmd>` | Package management |
| `files` | Open file manager |
| `monitor` | Open system monitor |
| `settings` | Open settings |
| `gui` | Switch to GUI mode |
| `voice` | Toggle voice control |
| `exit` | Exit shell |
