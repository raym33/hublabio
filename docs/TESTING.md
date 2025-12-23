# HubLab IO Testing Guide

> **Status: Tests Pending Implementation**
>
> This document outlines the testing strategy and instructions for HubLab IO.
> Hardware testing has not yet been performed - contributions welcome!

---

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Hardware Testing](#hardware-testing)
5. [Contributing Tests](#contributing-tests)
6. [Known Gaps](#known-gaps)

---

## Testing Overview

### Current Status

| Test Category | Status | Notes |
|---------------|--------|-------|
| Unit Tests | Partial | Basic tests in modules |
| Integration Tests | Not Started | Requires test harness |
| QEMU Tests | Partial | Boot tests work |
| Hardware Tests | Not Started | Need physical devices |
| Performance Tests | Not Started | Need benchmarks |
| Security Tests | Not Started | Need audit framework |

### Test Infrastructure Needed

- [ ] CI/CD pipeline (GitHub Actions)
- [ ] QEMU-based integration tests
- [ ] Hardware test automation (RPi)
- [ ] Code coverage reporting
- [ ] Benchmark suite
- [ ] Fuzzing infrastructure

---

## Test Categories

### 1. Unit Tests

Unit tests are embedded in source files using Rust's `#[cfg(test)]` attribute.

**Locations with existing tests:**

```
kernel/src/memory/         - Memory allocator tests
kernel/src/ipc/            - IPC channel tests
runtime/src/ai/tokenizer.rs    - Tokenizer tests
runtime/src/ai/sampling.rs     - Sampling algorithm tests
runtime/src/ai/quantization.rs - Quantization tests
runtime/src/moe/           - MoE-R routing tests
runtime/src/ai/npu/        - NPU backend tests
apps/core/browser/         - HTML parser tests
```

**Running unit tests:**

```bash
# All unit tests (requires host toolchain)
cargo test --workspace --lib

# Specific crate
cargo test -p hublabio-kernel --lib
cargo test -p hublabio-runtime --lib

# With output
cargo test --workspace -- --nocapture

# Single test
cargo test test_name -- --nocapture
```

**Note:** Some tests require `no_std` compatible test harness which is not yet configured.

### 2. Integration Tests

Integration tests verify component interactions.

**Needed tests:**

- [ ] Kernel boot sequence
- [ ] Process creation and scheduling
- [ ] IPC message passing
- [ ] VFS operations
- [ ] Network stack
- [ ] AI inference pipeline
- [ ] Distributed inference cluster

**Proposed structure:**

```
tests/
├── integration/
│   ├── boot_test.rs        - Kernel boot verification
│   ├── process_test.rs     - Process lifecycle
│   ├── ipc_test.rs         - IPC functionality
│   ├── vfs_test.rs         - Filesystem operations
│   ├── network_test.rs     - TCP/IP stack
│   └── ai_test.rs          - AI inference
├── e2e/
│   ├── shell_test.rs       - Shell commands
│   └── app_test.rs         - Application launch
└── fixtures/
    ├── test_model.gguf     - Small test model
    └── test_files/         - Test data
```

### 3. QEMU Tests

QEMU provides emulation for testing without hardware.

**Current QEMU setup:**

```bash
# Run in QEMU (basic boot test)
make run

# Run with GDB for debugging
make debug

# Run with specific memory
make run QEMU_MEM=4G
```

**Needed QEMU tests:**

```bash
# Proposed test commands
make test-boot      # Verify kernel boots
make test-shell     # Test shell commands
make test-ai        # Test AI inference
make test-network   # Test networking (with QEMU network)
```

**QEMU test script example:**

```bash
#!/bin/bash
# tests/qemu/boot_test.sh

TIMEOUT=30
EXPECTED_OUTPUT="HubLab IO Kernel"

# Start QEMU with timeout
timeout $TIMEOUT qemu-system-aarch64 \
    -machine virt \
    -cpu cortex-a72 \
    -m 2G \
    -kernel target/aarch64-unknown-none/release/hublabio-kernel \
    -nographic \
    -serial mon:stdio \
    2>&1 | tee /tmp/qemu_output.log &

QEMU_PID=$!
sleep 10

# Check output
if grep -q "$EXPECTED_OUTPUT" /tmp/qemu_output.log; then
    echo "PASS: Kernel boot successful"
    kill $QEMU_PID 2>/dev/null
    exit 0
else
    echo "FAIL: Expected output not found"
    kill $QEMU_PID 2>/dev/null
    exit 1
fi
```

### 4. Hardware Tests

Hardware testing requires physical Raspberry Pi devices.

**Supported hardware:**

| Device | Status | Notes |
|--------|--------|-------|
| Raspberry Pi 5 | Untested | Primary target |
| Raspberry Pi 4 | Untested | Secondary target |
| Raspberry Pi 3 | Untested | Limited support |
| QEMU virt | Working | Emulation only |

**Hardware test checklist:**

- [ ] Boot from SD card
- [ ] UART console output
- [ ] HDMI framebuffer
- [ ] USB keyboard/mouse
- [ ] Ethernet networking
- [ ] WiFi (BCM43xx)
- [ ] GPIO operations
- [ ] AI inference performance
- [ ] NPU acceleration (with AI Kit)

See [HARDWARE_TESTING.md](HARDWARE_TESTING.md) for detailed instructions.

---

## Running Tests

### Prerequisites

```bash
# Install Rust nightly
rustup default nightly
rustup target add aarch64-unknown-none

# Install QEMU
brew install qemu           # macOS
sudo apt install qemu-system-arm  # Linux

# Install cross-compiler
brew install aarch64-elf-gcc      # macOS
sudo apt install gcc-aarch64-linux-gnu  # Linux
```

### Quick Test Commands

```bash
# Build everything
make all

# Run unit tests (host)
make test

# Run in QEMU
make run

# Run with debugging
make debug
```

### Makefile Test Targets (TODO)

These targets need to be implemented:

```makefile
# Proposed Makefile additions

test: test-unit test-integration

test-unit:
	cargo test --workspace --lib

test-integration:
	./tests/run_integration.sh

test-qemu:
	./tests/qemu/run_all.sh

test-hardware:
	@echo "Hardware tests require physical device"
	@echo "See docs/HARDWARE_TESTING.md"

test-coverage:
	cargo tarpaulin --workspace --out Html

test-bench:
	cargo bench --workspace
```

---

## Hardware Testing

### Setting Up Test Environment

1. **Prepare SD Card:**

```bash
# Build kernel
make rpi5

# Create bootable SD card
./tools/flash/create_sd.sh /dev/sdX
```

2. **Connect Serial Console:**

```bash
# Using screen
screen /dev/ttyUSB0 115200

# Using minicom
minicom -D /dev/ttyUSB0 -b 115200
```

3. **Boot and Observe:**

```
Expected boot output:
===========================================
  HubLab IO Kernel v0.1.0
  AI-Native Operating System
===========================================

[BOOT] Initializing memory manager...
[BOOT] Setting up kernel heap...
...
hublab>
```

### Test Procedures

See [HARDWARE_TESTING.md](HARDWARE_TESTING.md) for complete procedures.

---

## Contributing Tests

### How to Add Unit Tests

```rust
// In any source file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        // Arrange
        let input = setup_test_data();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected_value);
    }
}
```

### How to Add Integration Tests

1. Create test file in `tests/integration/`
2. Use the test harness (TBD)
3. Document test purpose and requirements
4. Add to CI pipeline

### Test Naming Convention

```
test_<module>_<function>_<scenario>

Examples:
- test_memory_allocate_single_page
- test_ipc_send_large_message
- test_tokenizer_unicode_handling
- test_scheduler_priority_preemption
```

### Test Documentation

Each test should include:

```rust
/// Test: [Brief description]
///
/// Verifies that [behavior being tested].
///
/// Prerequisites:
/// - [Any setup required]
///
/// Expected: [Expected outcome]
#[test]
fn test_example() {
    // ...
}
```

---

## Known Gaps

### Critical Missing Tests

1. **Kernel Boot** - No automated boot verification
2. **Memory Safety** - No MIRI or sanitizer tests
3. **Concurrency** - No multi-core stress tests
4. **Network** - No TCP/IP conformance tests
5. **Security** - No penetration testing

### Test Infrastructure Needs

| Need | Priority | Effort |
|------|----------|--------|
| CI/CD Pipeline | High | Medium |
| QEMU Test Harness | High | Medium |
| Hardware Test Lab | Medium | High |
| Fuzzing Setup | Medium | Medium |
| Benchmark Suite | Low | Low |

### How to Help

1. **Easy wins:**
   - Add unit tests to existing modules
   - Document test procedures
   - Report bugs found during testing

2. **Medium effort:**
   - Set up GitHub Actions CI
   - Create QEMU test scripts
   - Write integration tests

3. **Major contributions:**
   - Hardware test automation
   - Fuzzing infrastructure
   - Performance benchmarking

---

## Test Results

### Latest Test Run

```
Status: No automated test results yet

Manual testing performed:
- [x] QEMU boot (verified)
- [x] Shell commands (basic)
- [ ] Hardware boot (not tested)
- [ ] AI inference (not tested)
- [ ] Network stack (not tested)
```

### Continuous Integration

CI pipeline not yet configured. Proposed workflow:

```yaml
# .github/workflows/test.yml (TODO)
name: Test

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo test --workspace --lib

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make setup
      - run: make rpi5

  qemu-test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - run: sudo apt install qemu-system-arm
      - run: make test-qemu
```

---

## Contact

For questions about testing:
- Open an issue on GitHub
- Tag with `testing` label

---

*Last updated: December 2024*
