# HubLab IO - AI-Native Operating System
# Makefile for building the OS

# ============================================================================
# Configuration
# ============================================================================

# Target architecture (arm64, riscv64)
ARCH ?= arm64

# Build type (debug, release)
BUILD ?= release

# Target device (rpi5, rpi4, rpi-zero2, pinephone, generic)
DEVICE ?= rpi5

# Rust target triple
ifeq ($(ARCH),arm64)
    TARGET = aarch64-unknown-none
else ifeq ($(ARCH),riscv64)
    TARGET = riscv64gc-unknown-none-elf
endif

# Build directory
BUILD_DIR = target/$(TARGET)/$(BUILD)

# Output directory
OUT_DIR = out/$(DEVICE)

# ============================================================================
# Tools
# ============================================================================

CARGO = cargo
RUSTC = rustc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump

ifeq ($(ARCH),arm64)
    CROSS_COMPILE ?= aarch64-none-elf-
else ifeq ($(ARCH),riscv64)
    CROSS_COMPILE ?= riscv64-unknown-elf-
endif

# ============================================================================
# Flags
# ============================================================================

CARGOFLAGS = --target $(TARGET)
ifeq ($(BUILD),release)
    CARGOFLAGS += --release
endif

ASFLAGS = -march=armv8-a

LDFLAGS = -T kernel/linker-$(ARCH).ld -nostdlib

# ============================================================================
# Targets
# ============================================================================

.PHONY: all kernel runtime shell sdk clean run debug test docs

all: kernel runtime shell sdk image

# Build kernel
kernel:
	@echo "Building kernel for $(ARCH)..."
	$(CARGO) build -p hublabio-kernel $(CARGOFLAGS)

# Build runtime
runtime:
	@echo "Building AI runtime..."
	$(CARGO) build -p hublabio-runtime $(CARGOFLAGS)

# Build shell
shell:
	@echo "Building shell..."
	$(CARGO) build -p hublabio-shell $(CARGOFLAGS)

# Build SDK
sdk:
	@echo "Building SDK..."
	$(CARGO) build -p hublabio-sdk $(CARGOFLAGS)

# Build bootloader
bootloader:
	@echo "Building bootloader..."
	@mkdir -p $(OUT_DIR)
	$(AS) $(ASFLAGS) kernel/boot/stage1.S -o $(OUT_DIR)/stage1.o
	$(LD) $(LDFLAGS) $(OUT_DIR)/stage1.o -o $(OUT_DIR)/bootloader.elf
	$(OBJCOPY) -O binary $(OUT_DIR)/bootloader.elf $(OUT_DIR)/bootloader.bin

# Create disk image
image: kernel bootloader
	@echo "Creating disk image for $(DEVICE)..."
	@mkdir -p $(OUT_DIR)
	@./tools/mkimage.sh $(DEVICE) $(OUT_DIR)

# ============================================================================
# Device-specific targets
# ============================================================================

# Raspberry Pi 5
rpi5: DEVICE=rpi5
rpi5: all

# Raspberry Pi 4
rpi4: DEVICE=rpi4
rpi4: all

# Raspberry Pi Zero 2 W
rpi-zero2: DEVICE=rpi-zero2
rpi-zero2: all

# PinePhone Pro
pinephone: DEVICE=pinephone
pinephone: all

# ============================================================================
# Run and Debug
# ============================================================================

# Run in QEMU
run: kernel
	@echo "Starting QEMU..."
	qemu-system-aarch64 \
		-M virt \
		-cpu cortex-a72 \
		-m 2G \
		-nographic \
		-kernel $(BUILD_DIR)/hublabio-kernel

# Debug with GDB
debug: kernel
	@echo "Starting QEMU with GDB server..."
	qemu-system-aarch64 \
		-M virt \
		-cpu cortex-a72 \
		-m 2G \
		-nographic \
		-kernel $(BUILD_DIR)/hublabio-kernel \
		-s -S

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "Running tests..."
	$(CARGO) test --workspace

test-kernel:
	$(CARGO) test -p hublabio-kernel

test-runtime:
	$(CARGO) test -p hublabio-runtime

test-sdk:
	$(CARGO) test -p hublabio-sdk

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "Generating documentation..."
	$(CARGO) doc --workspace --no-deps

docs-open: docs
	@echo "Opening documentation..."
	open target/doc/hublabio_kernel/index.html

# ============================================================================
# Utilities
# ============================================================================

# Format code
fmt:
	$(CARGO) fmt --all

# Lint code
lint:
	$(CARGO) clippy --workspace -- -D warnings

# Check code
check:
	$(CARGO) check --workspace

# Clean build artifacts
clean:
	$(CARGO) clean
	rm -rf out/

# Flash to SD card (macOS)
flash: image
	@echo "Flashing to SD card..."
	@echo "Please specify SD card device (e.g., make flash SDCARD=/dev/disk2)"
	@test -n "$(SDCARD)" || (echo "Error: SDCARD not specified" && exit 1)
	diskutil unmountDisk $(SDCARD)
	sudo dd if=$(OUT_DIR)/hublabio.img of=$(SDCARD) bs=4m
	diskutil eject $(SDCARD)
	@echo "Done! Insert SD card and boot."

# ============================================================================
# Dependencies
# ============================================================================

# Install Rust toolchain
setup:
	@echo "Setting up development environment..."
	rustup target add $(TARGET)
	rustup component add rust-src llvm-tools-preview
	@echo "Installing cargo-binutils..."
	cargo install cargo-binutils
	@echo "Done!"

# ============================================================================
# Help
# ============================================================================

help:
	@echo "HubLab IO Build System"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build everything"
	@echo "  kernel     - Build kernel only"
	@echo "  runtime    - Build AI runtime only"
	@echo "  shell      - Build shell only"
	@echo "  sdk        - Build SDK only"
	@echo "  image      - Create bootable image"
	@echo "  run        - Run in QEMU"
	@echo "  debug      - Debug with QEMU+GDB"
	@echo "  test       - Run all tests"
	@echo "  docs       - Generate documentation"
	@echo "  clean      - Clean build artifacts"
	@echo "  flash      - Flash to SD card"
	@echo "  setup      - Install development dependencies"
	@echo ""
	@echo "Device targets:"
	@echo "  rpi5       - Raspberry Pi 5"
	@echo "  rpi4       - Raspberry Pi 4"
	@echo "  rpi-zero2  - Raspberry Pi Zero 2 W"
	@echo "  pinephone  - PinePhone Pro"
	@echo ""
	@echo "Options:"
	@echo "  ARCH=arm64|riscv64  - Target architecture (default: arm64)"
	@echo "  BUILD=debug|release - Build type (default: release)"
	@echo "  DEVICE=...          - Target device (default: rpi5)"
