//! Address Space Layout Randomization (ASLR)
//!
//! Randomizes memory layout to prevent exploitation.
//! Implements stack, heap, mmap, and executable randomization.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// ASLR entropy source using architectural features
pub struct AslrEntropy {
    /// Current state for PRNG
    state: AtomicU64,
    /// Counter for additional entropy
    counter: AtomicU64,
}

impl AslrEntropy {
    pub const fn new() -> Self {
        Self {
            state: AtomicU64::new(0x5DEECE66D),
            counter: AtomicU64::new(0),
        }
    }

    /// Initialize with hardware entropy if available
    pub fn init(&self) {
        // Try to get entropy from timer
        let timer = crate::arch::read_timer();

        // Mix in some architectural randomness
        let mut seed = timer;

        // Read some memory addresses for additional entropy
        let stack_addr = &seed as *const u64 as u64;
        seed ^= stack_addr.rotate_left(17);

        // Counter value
        seed ^= self.counter.fetch_add(1, Ordering::SeqCst);

        self.state.store(seed, Ordering::SeqCst);
    }

    /// Get random u64
    pub fn random_u64(&self) -> u64 {
        // Linear congruential generator with mixing
        loop {
            let old = self.state.load(Ordering::SeqCst);
            let new = old
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);

            if self.state.compare_exchange(
                old, new, Ordering::SeqCst, Ordering::SeqCst
            ).is_ok() {
                // Additional mixing
                let mixed = new ^ (new >> 33);
                let mixed = mixed.wrapping_mul(0xff51afd7ed558ccd);
                let mixed = mixed ^ (mixed >> 33);
                let mixed = mixed.wrapping_mul(0xc4ceb9fe1a85ec53);
                return mixed ^ (mixed >> 33);
            }
        }
    }

    /// Get random value in range [0, max)
    pub fn random_range(&self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.random_u64() % max
    }

    /// Get random aligned address in range
    pub fn random_aligned(&self, min: usize, max: usize, align: usize) -> usize {
        if max <= min || align == 0 {
            return min;
        }

        let range = (max - min) / align;
        if range == 0 {
            return min;
        }

        let offset = self.random_range(range as u64) as usize;
        min + (offset * align)
    }
}

/// Global entropy source
static ENTROPY: AslrEntropy = AslrEntropy::new();

/// ASLR configuration
pub struct AslrConfig {
    /// ASLR enabled
    pub enabled: AtomicBool,
    /// Stack randomization enabled
    pub stack: AtomicBool,
    /// Heap (brk) randomization enabled
    pub heap: AtomicBool,
    /// mmap randomization enabled
    pub mmap: AtomicBool,
    /// Executable base randomization (PIE)
    pub exec: AtomicBool,
    /// VDSO randomization
    pub vdso: AtomicBool,
}

impl AslrConfig {
    pub const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            stack: AtomicBool::new(true),
            heap: AtomicBool::new(true),
            mmap: AtomicBool::new(true),
            exec: AtomicBool::new(true),
            vdso: AtomicBool::new(true),
        }
    }

    /// Check if ASLR is globally enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Set ASLR mode (0 = off, 1 = conservative, 2 = full)
    pub fn set_mode(&self, mode: u32) {
        match mode {
            0 => {
                self.enabled.store(false, Ordering::SeqCst);
                self.stack.store(false, Ordering::SeqCst);
                self.heap.store(false, Ordering::SeqCst);
                self.mmap.store(false, Ordering::SeqCst);
                self.exec.store(false, Ordering::SeqCst);
                self.vdso.store(false, Ordering::SeqCst);
            }
            1 => {
                // Conservative: only stack and libraries
                self.enabled.store(true, Ordering::SeqCst);
                self.stack.store(true, Ordering::SeqCst);
                self.heap.store(false, Ordering::SeqCst);
                self.mmap.store(true, Ordering::SeqCst);
                self.exec.store(false, Ordering::SeqCst);
                self.vdso.store(true, Ordering::SeqCst);
            }
            _ => {
                // Full ASLR
                self.enabled.store(true, Ordering::SeqCst);
                self.stack.store(true, Ordering::SeqCst);
                self.heap.store(true, Ordering::SeqCst);
                self.mmap.store(true, Ordering::SeqCst);
                self.exec.store(true, Ordering::SeqCst);
                self.vdso.store(true, Ordering::SeqCst);
            }
        }
    }

    /// Get current mode
    pub fn mode(&self) -> u32 {
        if !self.enabled.load(Ordering::SeqCst) {
            0
        } else if !self.heap.load(Ordering::SeqCst) {
            1
        } else {
            2
        }
    }
}

/// Global ASLR configuration
static CONFIG: AslrConfig = AslrConfig::new();

/// Memory layout constants for AArch64
pub mod layout {
    /// User address space start
    pub const USER_START: usize = 0x0000_0000_0000;
    /// User address space end
    pub const USER_END: usize = 0x0000_FFFF_FFFF_FFFF;

    /// Stack region
    pub const STACK_TOP_MAX: usize = 0x0000_7FFF_FFFF_0000;
    pub const STACK_TOP_MIN: usize = 0x0000_7FFF_0000_0000;

    /// Stack randomization bits (16MB range)
    pub const STACK_RANDOM_BITS: usize = 24;

    /// Heap region
    pub const HEAP_START_MIN: usize = 0x0000_0000_1000_0000;
    pub const HEAP_START_MAX: usize = 0x0000_0000_2000_0000;

    /// Heap randomization bits (256MB range)
    pub const HEAP_RANDOM_BITS: usize = 28;

    /// mmap region
    pub const MMAP_START_MIN: usize = 0x0000_7000_0000_0000;
    pub const MMAP_START_MAX: usize = 0x0000_7F00_0000_0000;

    /// mmap randomization bits
    pub const MMAP_RANDOM_BITS: usize = 28;

    /// Executable base for PIE
    pub const EXEC_BASE_MIN: usize = 0x0000_0000_0040_0000;
    pub const EXEC_BASE_MAX: usize = 0x0000_0000_0100_0000;

    /// Page size for alignment
    pub const PAGE_SIZE: usize = 4096;
}

/// Randomized memory layout for a process
#[derive(Clone, Debug)]
pub struct MemoryLayout {
    /// Stack top address
    pub stack_top: usize,
    /// Stack size
    pub stack_size: usize,
    /// Stack guard page
    pub stack_guard: usize,
    /// Heap start (brk base)
    pub heap_start: usize,
    /// Current heap end
    pub heap_end: usize,
    /// mmap base address
    pub mmap_base: usize,
    /// Executable base (for PIE)
    pub exec_base: usize,
    /// VDSO address
    pub vdso_addr: usize,
    /// Stack entropy (bytes of random offset)
    pub stack_entropy: usize,
}

impl MemoryLayout {
    /// Create a new randomized memory layout
    pub fn new_randomized() -> Self {
        let config = &CONFIG;

        // Stack randomization
        let stack_top = if config.stack.load(Ordering::SeqCst) && config.is_enabled() {
            let range = layout::STACK_TOP_MAX - layout::STACK_TOP_MIN;
            let random_offset = ENTROPY.random_range((range / layout::PAGE_SIZE) as u64) as usize;
            layout::STACK_TOP_MAX - (random_offset * layout::PAGE_SIZE)
        } else {
            layout::STACK_TOP_MAX
        };

        // Additional per-exec stack entropy (within a page)
        let stack_entropy = if config.stack.load(Ordering::SeqCst) && config.is_enabled() {
            // 8-byte aligned random offset within 256 bytes
            (ENTROPY.random_range(32) as usize) * 8
        } else {
            0
        };

        // Heap randomization
        let heap_start = if config.heap.load(Ordering::SeqCst) && config.is_enabled() {
            ENTROPY.random_aligned(
                layout::HEAP_START_MIN,
                layout::HEAP_START_MAX,
                layout::PAGE_SIZE,
            )
        } else {
            layout::HEAP_START_MIN
        };

        // mmap base randomization
        let mmap_base = if config.mmap.load(Ordering::SeqCst) && config.is_enabled() {
            ENTROPY.random_aligned(
                layout::MMAP_START_MIN,
                layout::MMAP_START_MAX,
                layout::PAGE_SIZE,
            )
        } else {
            layout::MMAP_START_MIN
        };

        // Executable base (PIE)
        let exec_base = if config.exec.load(Ordering::SeqCst) && config.is_enabled() {
            ENTROPY.random_aligned(
                layout::EXEC_BASE_MIN,
                layout::EXEC_BASE_MAX,
                layout::PAGE_SIZE,
            )
        } else {
            layout::EXEC_BASE_MIN
        };

        // VDSO
        let vdso_addr = if config.vdso.load(Ordering::SeqCst) && config.is_enabled() {
            // Place VDSO near mmap region
            let vdso_min = mmap_base - 0x10000;
            let vdso_max = mmap_base;
            ENTROPY.random_aligned(vdso_min, vdso_max, layout::PAGE_SIZE)
        } else {
            mmap_base - layout::PAGE_SIZE
        };

        // Default stack size: 8MB
        let stack_size = 8 * 1024 * 1024;
        let stack_guard = stack_top - stack_size - layout::PAGE_SIZE;

        Self {
            stack_top,
            stack_size,
            stack_guard,
            heap_start,
            heap_end: heap_start,
            mmap_base,
            exec_base,
            vdso_addr,
            stack_entropy,
        }
    }

    /// Create non-randomized layout (for compatibility)
    pub fn new_fixed() -> Self {
        Self {
            stack_top: layout::STACK_TOP_MAX,
            stack_size: 8 * 1024 * 1024,
            stack_guard: layout::STACK_TOP_MAX - 8 * 1024 * 1024 - layout::PAGE_SIZE,
            heap_start: layout::HEAP_START_MIN,
            heap_end: layout::HEAP_START_MIN,
            mmap_base: layout::MMAP_START_MIN,
            exec_base: layout::EXEC_BASE_MIN,
            vdso_addr: layout::MMAP_START_MIN - layout::PAGE_SIZE,
            stack_entropy: 0,
        }
    }

    /// Fork: child gets similar layout with slight variations
    pub fn fork(&self) -> Self {
        let mut child = self.clone();

        // Add slight variation to stack pointer for child
        if CONFIG.stack.load(Ordering::SeqCst) && CONFIG.is_enabled() {
            child.stack_entropy = (ENTROPY.random_range(32) as usize) * 8;
        }

        child
    }

    /// Exec: completely new randomized layout
    pub fn exec(&self) -> Self {
        Self::new_randomized()
    }

    /// Get randomized stack pointer for program start
    pub fn initial_stack_pointer(&self) -> usize {
        self.stack_top - self.stack_entropy
    }

    /// Allocate mmap region
    pub fn mmap_alloc(&mut self, size: usize, hint: Option<usize>) -> usize {
        let aligned_size = (size + layout::PAGE_SIZE - 1) & !(layout::PAGE_SIZE - 1);

        if let Some(addr) = hint {
            // Check if hint is valid
            if addr >= layout::USER_START && addr + aligned_size <= self.stack_guard {
                return addr;
            }
        }

        // Randomize within mmap region if ASLR enabled
        if CONFIG.mmap.load(Ordering::SeqCst) && CONFIG.is_enabled() {
            // Random offset within available range
            let max_addr = self.stack_guard.saturating_sub(aligned_size);
            if max_addr > self.mmap_base {
                return ENTROPY.random_aligned(
                    self.mmap_base,
                    max_addr,
                    layout::PAGE_SIZE,
                );
            }
        }

        // Fallback: use mmap_base and move it up
        let addr = self.mmap_base;
        self.mmap_base += aligned_size;
        addr
    }

    /// Validate address is in user space
    pub fn is_user_address(&self, addr: usize) -> bool {
        addr >= layout::USER_START && addr < layout::USER_END
    }

    /// Validate range is in user space
    pub fn is_user_range(&self, addr: usize, len: usize) -> bool {
        addr >= layout::USER_START &&
        addr < layout::USER_END &&
        addr.checked_add(len).map(|end| end <= layout::USER_END).unwrap_or(false)
    }
}

/// Get random bytes
pub fn random_bytes(buf: &mut [u8]) {
    let mut remaining = buf.len();
    let mut offset = 0;

    while remaining > 0 {
        let random = ENTROPY.random_u64();
        let bytes = random.to_le_bytes();
        let to_copy = remaining.min(8);

        buf[offset..offset + to_copy].copy_from_slice(&bytes[..to_copy]);
        offset += to_copy;
        remaining -= to_copy;
    }
}

/// Get random u64
pub fn random_u64() -> u64 {
    ENTROPY.random_u64()
}

/// Get random value in range
pub fn random_range(max: u64) -> u64 {
    ENTROPY.random_range(max)
}

/// Set ASLR mode
pub fn set_mode(mode: u32) {
    CONFIG.set_mode(mode);
}

/// Get ASLR mode
pub fn get_mode() -> u32 {
    CONFIG.mode()
}

/// Check if ASLR is enabled
pub fn is_enabled() -> bool {
    CONFIG.is_enabled()
}

/// Generate /proc/sys/kernel/randomize_va_space content
pub fn generate_randomize_va_space() -> String {
    format!("{}", CONFIG.mode())
}

/// Parse /proc/sys/kernel/randomize_va_space
pub fn parse_randomize_va_space(value: &str) -> Result<(), &'static str> {
    let mode: u32 = value.trim().parse().map_err(|_| "Invalid number")?;
    if mode > 2 {
        return Err("Value must be 0, 1, or 2");
    }
    set_mode(mode);
    Ok(())
}

/// Stack canary for stack smashing protection
pub struct StackCanary(u64);

impl StackCanary {
    /// Generate new stack canary
    pub fn new() -> Self {
        Self(ENTROPY.random_u64())
    }

    /// Get canary value
    pub fn value(&self) -> u64 {
        self.0
    }

    /// Check canary
    pub fn check(&self, value: u64) -> bool {
        self.0 == value
    }
}

/// Initialize ASLR subsystem
pub fn init() {
    ENTROPY.init();

    // Enable full ASLR by default
    CONFIG.set_mode(2);

    crate::kprintln!("  ASLR initialized (mode=2, full randomization)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy() {
        let e = AslrEntropy::new();
        e.init();

        let r1 = e.random_u64();
        let r2 = e.random_u64();
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_random_range() {
        let e = AslrEntropy::new();
        e.init();

        for _ in 0..100 {
            let r = e.random_range(100);
            assert!(r < 100);
        }
    }

    #[test]
    fn test_memory_layout() {
        let layout1 = MemoryLayout::new_randomized();
        let layout2 = MemoryLayout::new_randomized();

        // Layouts should be different (with high probability)
        assert!(
            layout1.stack_top != layout2.stack_top ||
            layout1.heap_start != layout2.heap_start ||
            layout1.mmap_base != layout2.mmap_base
        );
    }

    #[test]
    fn test_stack_canary() {
        let canary = StackCanary::new();
        assert!(canary.check(canary.value()));
        assert!(!canary.check(0));
    }
}
