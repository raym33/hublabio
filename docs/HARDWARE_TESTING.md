# Hardware Testing Guide

> **Status: Hardware Testing Not Yet Performed**
>
> This guide explains how to test HubLab IO on real Raspberry Pi hardware.
> Community contributions for hardware testing are greatly appreciated!

---

## Table of Contents

1. [Required Hardware](#required-hardware)
2. [Preparing the SD Card](#preparing-the-sd-card)
3. [Serial Console Setup](#serial-console-setup)
4. [Boot Testing](#boot-testing)
5. [Peripheral Testing](#peripheral-testing)
6. [AI/NPU Testing](#ainpu-testing)
7. [Reporting Results](#reporting-results)

---

## Required Hardware

### Minimum Setup

| Item | Purpose | Notes |
|------|---------|-------|
| Raspberry Pi 5 or 4 | Target device | 4GB+ RAM recommended |
| MicroSD Card | Boot media | 16GB+ Class 10 |
| USB-TTL Serial Adapter | Console access | 3.3V logic level |
| Power Supply | Power | Official RPi PSU recommended |
| SD Card Reader | Flashing | USB or built-in |

### Recommended Additions

| Item | Purpose | Notes |
|------|---------|-------|
| HDMI Monitor | Framebuffer testing | Any HDMI display |
| USB Keyboard | Input testing | Standard USB HID |
| Ethernet Cable | Network testing | For wired networking |
| Raspberry Pi AI Kit | NPU testing | Hailo-8L accelerator |

### Optional

| Item | Purpose |
|------|---------|
| USB Mouse | GUI testing |
| WiFi Network | Wireless testing |
| USB Storage | USB mass storage testing |
| GPIO LEDs/Buttons | GPIO testing |

---

## Preparing the SD Card

### Method 1: Using Flash Script (Recommended)

```bash
# Build the kernel
make rpi5  # or make rpi4

# Flash to SD card (replace /dev/sdX with your device)
# WARNING: This will erase the SD card!
sudo ./tools/flash/create_sd.sh /dev/sdX
```

### Method 2: Manual Preparation

1. **Format the SD card:**

```bash
# Create partitions
sudo fdisk /dev/sdX << EOF
o
n
p
1

+256M
t
c
n
p
2


w
EOF

# Format partitions
sudo mkfs.vfat -F 32 /dev/sdX1
sudo mkfs.ext4 /dev/sdX2
```

2. **Copy boot files:**

```bash
# Mount boot partition
sudo mount /dev/sdX1 /mnt

# Copy kernel and config
sudo cp target/aarch64-unknown-none/release/kernel8.img /mnt/
sudo cp boot/config.txt /mnt/
sudo cp boot/cmdline.txt /mnt/

# Copy firmware (from RPi firmware repo)
sudo cp firmware/bootcode.bin /mnt/
sudo cp firmware/start4.elf /mnt/
sudo cp firmware/fixup4.dat /mnt/

sudo umount /mnt
```

3. **Prepare root filesystem:**

```bash
# Mount root partition
sudo mount /dev/sdX2 /mnt

# Create directory structure
sudo mkdir -p /mnt/{bin,etc,home,models,proc,sys,tmp,var}

# Copy any required files
sudo cp -r rootfs/* /mnt/

sudo umount /mnt
```

### Boot Configuration Files

**config.txt:**

```ini
# HubLab IO Boot Configuration
arm_64bit=1
kernel=kernel8.img
enable_uart=1
dtparam=audio=on
gpu_mem=128

# For RPi 5
[pi5]
kernel=kernel8.img

# For RPi 4
[pi4]
kernel=kernel8.img
```

**cmdline.txt:**

```
console=serial0,115200 root=/dev/mmcblk0p2 rootwait
```

---

## Serial Console Setup

### Hardware Connection

Connect USB-TTL adapter to RPi GPIO header:

```
USB-TTL    Raspberry Pi
-------    ------------
GND   -->  Pin 6  (GND)
TXD   -->  Pin 10 (GPIO15/RXD)
RXD   -->  Pin 8  (GPIO14/TXD)

DO NOT CONNECT 5V/VCC - Power Pi separately!
```

### Software Setup

**macOS:**

```bash
# Find device
ls /dev/tty.usbserial*

# Connect with screen
screen /dev/tty.usbserial-* 115200

# Or with minicom
brew install minicom
minicom -D /dev/tty.usbserial-* -b 115200
```

**Linux:**

```bash
# Find device
ls /dev/ttyUSB*

# Connect with screen
sudo screen /dev/ttyUSB0 115200

# Or with minicom
sudo apt install minicom
sudo minicom -D /dev/ttyUSB0 -b 115200
```

**Windows:**

1. Install PuTTY or similar
2. Find COM port in Device Manager
3. Connect: 115200 baud, 8N1

### Exiting Serial Console

- **screen:** Press `Ctrl+A` then `K`, confirm with `Y`
- **minicom:** Press `Ctrl+A` then `X`

---

## Boot Testing

### Expected Boot Sequence

```
===========================================
  HubLab IO Kernel v0.1.0
  AI-Native Operating System
===========================================

[BOOT] Initializing memory manager...
  Total RAM: 4096 MB
[BOOT] Setting up kernel heap...
[BOOT] Initializing architecture...
  CPU: Cortex-A76 x4
[BOOT] Initializing AI-enhanced scheduler...
  Scheduler initialized with 4 cores
[BOOT] Setting up IPC channels...
  IPC channels initialized
[BOOT] Mounting virtual filesystem...
  VFS initialized with 4 mount points
[BOOT] Setting up syscall interface...

[BOOT] Kernel initialization complete!
[BOOT] Starting init process...

hublab>
```

### Boot Test Checklist

| Test | Command | Expected |
|------|---------|----------|
| Kernel loads | - | Boot messages appear |
| Memory detected | `meminfo` | Shows total/free RAM |
| CPUs detected | `cpuinfo` | Shows all cores |
| Shell works | `help` | Shows commands |
| Filesystem | `ls /` | Lists directories |

### Common Boot Issues

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| No output | Wrong baud rate | Verify 115200 |
| No output | TX/RX swapped | Swap wires |
| Rainbow screen | Kernel not found | Check kernel8.img |
| Kernel panic | Memory issue | Check RAM |
| Hangs at boot | Driver issue | Enable debug output |

---

## Peripheral Testing

### HDMI Display

1. Connect HDMI monitor before power-on
2. Should see framebuffer console
3. Test with: `clear` and `echo test`

**Test checklist:**

- [ ] Display detected at boot
- [ ] Text visible on screen
- [ ] Resolution correct
- [ ] Colors display properly

### USB Keyboard

1. Connect USB keyboard to any USB port
2. Should work immediately after boot

**Test checklist:**

- [ ] Keyboard detected
- [ ] Keys register correctly
- [ ] Special keys work (arrows, etc.)

### Ethernet Networking

```bash
# Check interface
hublab> ifconfig

# Test connectivity
hublab> ping 8.8.8.8

# DHCP (if implemented)
hublab> dhcp eth0
```

**Test checklist:**

- [ ] Interface detected
- [ ] Link established
- [ ] IP obtained (DHCP)
- [ ] Ping works
- [ ] DNS resolution works

### WiFi (BCM43xx)

```bash
# Scan networks
hublab> wifi scan

# Connect
hublab> wifi connect "SSID" "password"

# Status
hublab> wifi status
```

**Test checklist:**

- [ ] WiFi chip detected
- [ ] Scan shows networks
- [ ] WPA2 connection works
- [ ] Internet access works

### GPIO

```bash
# Export pin
hublab> gpio export 17 out

# Set high
hublab> gpio set 17 1

# Read pin
hublab> gpio read 17
```

**Test with LED:**

1. Connect LED + resistor to GPIO17 and GND
2. Run `gpio set 17 1` - LED should light
3. Run `gpio set 17 0` - LED should turn off

---

## AI/NPU Testing

### CPU-Based AI Inference

```bash
# Load model
hublab> ai load /models/tinyllama-1.1b-q4.gguf

# Generate text
hublab> ai generate "Hello, I am"

# Check performance
hublab> ai stats
```

**Expected metrics (RPi 5):**

| Model | Size | Tokens/sec |
|-------|------|------------|
| TinyLlama 1.1B Q4 | 668MB | ~2-5 |
| Qwen2 0.5B Q4 | 394MB | ~5-10 |
| Phi-2 Q4 | 1.5GB | ~1-2 |

### NPU Testing (Hailo-8L)

Requires Raspberry Pi AI Kit installed.

```bash
# Check NPU detection
hublab> npu info

# Expected output:
# Device: Hailo-8L
# Compute: 13 TOPS
# Status: Available

# Load HEF model
hublab> npu load /models/yolov8n.hef

# Run inference
hublab> npu infer /images/test.jpg

# Performance
hublab> npu stats
```

**Expected NPU metrics:**

| Task | Model | FPS |
|------|-------|-----|
| Object Detection | YOLOv8n | 30+ |
| Classification | ResNet50 | 100+ |
| Pose Estimation | YOLOv8n-pose | 25+ |

### AI Benchmark Suite

```bash
# Run full benchmark
hublab> ai benchmark

# Output:
# Model Loading: 2.3s
# Tokenization: 0.1ms/token
# Inference: 150ms/token
# Memory Usage: 1.2GB
```

---

## Reporting Results

### Test Report Template

When reporting hardware test results, please include:

```markdown
## Hardware Test Report

### System Information
- **Device:** Raspberry Pi 5 4GB
- **Kernel Version:** v0.1.0
- **Test Date:** 2024-12-XX

### Boot Test
- [ ] Serial console output: PASS/FAIL
- [ ] HDMI output: PASS/FAIL
- [ ] Boot time: XX seconds

### Peripheral Tests
- [ ] USB Keyboard: PASS/FAIL
- [ ] Ethernet: PASS/FAIL
- [ ] WiFi: PASS/FAIL/N/A
- [ ] GPIO: PASS/FAIL

### AI Tests
- [ ] Model loading: PASS/FAIL
- [ ] Inference: PASS/FAIL
- [ ] Performance: XX tokens/sec

### NPU Tests (if applicable)
- [ ] Detection: PASS/FAIL
- [ ] Model loading: PASS/FAIL
- [ ] Performance: XX FPS

### Issues Found
1. [Description of any issues]

### Notes
[Any additional observations]
```

### How to Submit

1. Create a GitHub Issue with label `hardware-test`
2. Use the template above
3. Attach serial console logs if relevant
4. Include photos if helpful

### Test Result Database

We maintain a compatibility database:

| Device | RAM | Status | Tester | Date |
|--------|-----|--------|--------|------|
| RPi 5 | 4GB | Untested | - | - |
| RPi 5 | 8GB | Untested | - | - |
| RPi 4 | 4GB | Untested | - | - |
| RPi 4 | 8GB | Untested | - | - |
| RPi 3B+ | 1GB | Untested | - | - |

---

## Troubleshooting

### Debug Build

For more verbose output:

```bash
# Build with debug symbols
make rpi5 DEBUG=1

# Enable kernel debug output
# In config.txt add:
kernel_debug=1
```

### Memory Debugging

```bash
# At shell
hublab> meminfo -v

# Shows detailed memory map
```

### Driver Issues

```bash
# List loaded drivers
hublab> lsmod

# Driver debug
hublab> dmesg | grep driver_name
```

---

## Contributing

### Priority Tests Needed

1. **RPi 5 Boot Test** - Verify basic boot on RPi 5
2. **Network Test** - Verify Ethernet/WiFi work
3. **AI Performance** - Benchmark inference speed
4. **NPU Integration** - Test Hailo-8L (requires AI Kit)

### Test Lab Setup

If you want to set up a dedicated test environment:

1. Multiple RPi models (3, 4, 5)
2. Various RAM configurations
3. Network switch for automated testing
4. Serial console multiplexer
5. Power control (for automated reboot)

---

*Last updated: December 2024*
*Contributions welcome!*
