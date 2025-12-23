//! Random Number Generator and Entropy Pool
//!
//! Cryptographically secure random number generation with entropy harvesting.
//! Compatible with /dev/random and /dev/urandom semantics.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use spin::Mutex;

/// Entropy pool size in bytes
const POOL_SIZE: usize = 512;

/// Minimum entropy bits for blocking read
const MIN_ENTROPY_BITS: u32 = 128;

/// Maximum entropy bits in pool
const MAX_ENTROPY_BITS: u32 = POOL_SIZE as u32 * 8;

/// Entropy pool
pub struct EntropyPool {
    /// Pool data
    data: [u8; POOL_SIZE],
    /// Current write position
    write_pos: usize,
    /// Estimated entropy bits
    entropy_bits: AtomicU32,
    /// Pool initialized
    initialized: AtomicBool,
    /// Generation counter (for reseeding detection)
    generation: AtomicU64,
}

impl EntropyPool {
    const fn new() -> Self {
        Self {
            data: [0; POOL_SIZE],
            write_pos: 0,
            entropy_bits: AtomicU32::new(0),
            initialized: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        }
    }

    /// Add entropy to pool
    fn add_entropy(&mut self, data: &[u8], bits: u32) {
        // Mix data into pool using simple XOR and rotation
        for &byte in data {
            self.data[self.write_pos] ^= byte;
            self.write_pos = (self.write_pos + 1) % POOL_SIZE;

            // Also mix with neighboring bytes
            let prev = (self.write_pos + POOL_SIZE - 1) % POOL_SIZE;
            let next = (self.write_pos + 1) % POOL_SIZE;
            self.data[prev] = self.data[prev].wrapping_add(byte.rotate_left(3));
            self.data[next] = self.data[next].wrapping_add(byte.rotate_right(5));
        }

        // Update entropy estimate
        let current = self.entropy_bits.load(Ordering::Relaxed);
        let new_bits = (current + bits).min(MAX_ENTROPY_BITS);
        self.entropy_bits.store(new_bits, Ordering::Relaxed);

        // Mark as initialized if we have enough entropy
        if new_bits >= MIN_ENTROPY_BITS && !self.initialized.load(Ordering::Relaxed) {
            self.initialized.store(true, Ordering::SeqCst);
            crate::kinfo!("Random: entropy pool initialized ({} bits)", new_bits);
        }
    }

    /// Extract random bytes from pool
    fn extract(&mut self, buf: &mut [u8]) {
        // Use ChaCha20-like PRNG seeded from pool
        let mut state = ChaChaState::new(&self.data[..32]);

        for chunk in buf.chunks_mut(64) {
            let block = state.block();
            let len = chunk.len();
            chunk.copy_from_slice(&block[..len]);
            state.counter += 1;
        }

        // Mix output back into pool to prevent backtracking
        self.add_entropy(buf, 0);

        // Decrement entropy estimate (conservative)
        let extracted = (buf.len() * 4).min(self.entropy_bits.load(Ordering::Relaxed) as usize);
        self.entropy_bits
            .fetch_sub(extracted as u32, Ordering::Relaxed);

        // Increment generation
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current entropy estimate
    fn entropy_available(&self) -> u32 {
        self.entropy_bits.load(Ordering::Relaxed)
    }

    /// Check if pool is initialized
    fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }
}

/// Global entropy pool
static POOL: Mutex<EntropyPool> = Mutex::new(EntropyPool::new());

/// Wait queue for blocking reads
static WAITERS: crate::waitqueue::WaitQueue = crate::waitqueue::WaitQueue::new();

/// Statistics
static STATS: RandomStats = RandomStats::new();

struct RandomStats {
    bytes_generated: AtomicU64,
    entropy_added: AtomicU64,
    pool_reseeds: AtomicU64,
}

impl RandomStats {
    const fn new() -> Self {
        Self {
            bytes_generated: AtomicU64::new(0),
            entropy_added: AtomicU64::new(0),
            pool_reseeds: AtomicU64::new(0),
        }
    }
}

// ============================================================================
// ChaCha20-based PRNG
// ============================================================================

/// ChaCha20 state
struct ChaChaState {
    state: [u32; 16],
    counter: u64,
}

impl ChaChaState {
    fn new(key: &[u8]) -> Self {
        let mut state = [0u32; 16];

        // Constants "expand 32-byte k"
        state[0] = 0x61707865;
        state[1] = 0x3320646e;
        state[2] = 0x79622d32;
        state[3] = 0x6b206574;

        // Key
        for i in 0..8 {
            let idx = i * 4;
            if idx + 3 < key.len() {
                state[4 + i] =
                    u32::from_le_bytes([key[idx], key[idx + 1], key[idx + 2], key[idx + 3]]);
            }
        }

        // Counter and nonce
        state[12] = 0;
        state[13] = 0;
        state[14] = 0;
        state[15] = 0;

        Self { state, counter: 0 }
    }

    fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        state[a] = state[a].wrapping_add(state[b]);
        state[d] ^= state[a];
        state[d] = state[d].rotate_left(16);

        state[c] = state[c].wrapping_add(state[d]);
        state[b] ^= state[c];
        state[b] = state[b].rotate_left(12);

        state[a] = state[a].wrapping_add(state[b]);
        state[d] ^= state[a];
        state[d] = state[d].rotate_left(8);

        state[c] = state[c].wrapping_add(state[d]);
        state[b] ^= state[c];
        state[b] = state[b].rotate_left(7);
    }

    fn block(&self) -> [u8; 64] {
        let mut working = self.state;

        // Set counter
        working[12] = self.counter as u32;
        working[13] = (self.counter >> 32) as u32;

        // 20 rounds (10 double-rounds)
        for _ in 0..10 {
            // Column rounds
            Self::quarter_round(&mut working, 0, 4, 8, 12);
            Self::quarter_round(&mut working, 1, 5, 9, 13);
            Self::quarter_round(&mut working, 2, 6, 10, 14);
            Self::quarter_round(&mut working, 3, 7, 11, 15);

            // Diagonal rounds
            Self::quarter_round(&mut working, 0, 5, 10, 15);
            Self::quarter_round(&mut working, 1, 6, 11, 12);
            Self::quarter_round(&mut working, 2, 7, 8, 13);
            Self::quarter_round(&mut working, 3, 4, 9, 14);
        }

        // Add original state
        for i in 0..16 {
            working[i] = working[i].wrapping_add(self.state[i]);
        }

        // Serialize to bytes
        let mut output = [0u8; 64];
        for i in 0..16 {
            let bytes = working[i].to_le_bytes();
            output[i * 4] = bytes[0];
            output[i * 4 + 1] = bytes[1];
            output[i * 4 + 2] = bytes[2];
            output[i * 4 + 3] = bytes[3];
        }

        output
    }
}

// ============================================================================
// Entropy Sources
// ============================================================================

/// Add entropy from hardware timer jitter
pub fn add_timer_entropy() {
    let timestamp = crate::time::monotonic_ns();
    let data = timestamp.to_le_bytes();

    // Timer jitter provides ~1-4 bits per sample
    POOL.lock().add_entropy(&data, 2);
    STATS.entropy_added.fetch_add(8, Ordering::Relaxed);
}

/// Add entropy from interrupt timing
pub fn add_interrupt_entropy(irq: u32) {
    let timestamp = crate::time::monotonic_ns();
    let data = [
        (timestamp & 0xFF) as u8,
        ((timestamp >> 8) & 0xFF) as u8,
        (irq & 0xFF) as u8,
        ((irq >> 8) & 0xFF) as u8,
    ];

    // Interrupt timing provides ~2-4 bits
    POOL.lock().add_entropy(&data, 3);
    STATS.entropy_added.fetch_add(4, Ordering::Relaxed);

    // Wake any waiters if we have enough entropy
    if POOL.lock().entropy_available() >= MIN_ENTROPY_BITS {
        WAITERS.wake_all();
    }
}

/// Add entropy from keyboard input
pub fn add_keyboard_entropy(scancode: u8, timestamp: u64) {
    let data = [
        scancode,
        (timestamp & 0xFF) as u8,
        ((timestamp >> 8) & 0xFF) as u8,
        ((timestamp >> 16) & 0xFF) as u8,
    ];

    // Keyboard timing provides ~4-8 bits
    POOL.lock().add_entropy(&data, 6);
    STATS.entropy_added.fetch_add(4, Ordering::Relaxed);

    if POOL.lock().entropy_available() >= MIN_ENTROPY_BITS {
        WAITERS.wake_all();
    }
}

/// Add entropy from mouse movement
pub fn add_mouse_entropy(x: i32, y: i32, timestamp: u64) {
    let data = [
        (x & 0xFF) as u8,
        ((x >> 8) & 0xFF) as u8,
        (y & 0xFF) as u8,
        ((y >> 8) & 0xFF) as u8,
        (timestamp & 0xFF) as u8,
        ((timestamp >> 8) & 0xFF) as u8,
    ];

    // Mouse movement provides ~4-6 bits
    POOL.lock().add_entropy(&data, 5);
    STATS.entropy_added.fetch_add(6, Ordering::Relaxed);

    if POOL.lock().entropy_available() >= MIN_ENTROPY_BITS {
        WAITERS.wake_all();
    }
}

/// Add entropy from network packet timing
pub fn add_network_entropy(packet_hash: u32, timestamp: u64) {
    let data = [
        (packet_hash & 0xFF) as u8,
        ((packet_hash >> 8) & 0xFF) as u8,
        ((packet_hash >> 16) & 0xFF) as u8,
        ((packet_hash >> 24) & 0xFF) as u8,
        (timestamp & 0xFF) as u8,
        ((timestamp >> 8) & 0xFF) as u8,
    ];

    // Network timing provides ~2-4 bits
    POOL.lock().add_entropy(&data, 3);
    STATS.entropy_added.fetch_add(6, Ordering::Relaxed);
}

/// Add entropy from disk I/O timing
pub fn add_disk_entropy(sector: u64, latency_ns: u64) {
    let data = [
        (sector & 0xFF) as u8,
        ((sector >> 8) & 0xFF) as u8,
        (latency_ns & 0xFF) as u8,
        ((latency_ns >> 8) & 0xFF) as u8,
    ];

    // Disk timing provides ~2-4 bits
    POOL.lock().add_entropy(&data, 3);
    STATS.entropy_added.fetch_add(4, Ordering::Relaxed);
}

/// Add raw entropy data (from hardware RNG)
pub fn add_raw_entropy(data: &[u8], bits: u32) {
    POOL.lock().add_entropy(data, bits);
    STATS
        .entropy_added
        .fetch_add(data.len() as u64, Ordering::Relaxed);

    if POOL.lock().entropy_available() >= MIN_ENTROPY_BITS {
        WAITERS.wake_all();
    }
}

/// Try to gather entropy from hardware sources
fn gather_hardware_entropy() {
    // Try ARM RNDR instruction
    #[cfg(target_arch = "aarch64")]
    {
        let mut random: u64;
        let success: u64;
        unsafe {
            core::arch::asm!(
                "mrs {0}, rndr",
                "cset {1}, ne",
                out(reg) random,
                out(reg) success,
                options(nomem, nostack)
            );
        }
        if success != 0 {
            add_raw_entropy(&random.to_le_bytes(), 64);
        }
    }

    // Fallback: use timer
    add_timer_entropy();
}

// ============================================================================
// Random Generation Interface
// ============================================================================

/// Get random bytes (blocking, high quality - /dev/random)
pub fn get_random_bytes_blocking(buf: &mut [u8]) -> Result<usize, RandomError> {
    loop {
        {
            let pool = POOL.lock();
            if pool.is_initialized() && pool.entropy_available() >= buf.len() as u32 {
                break;
            }
        }

        // Gather more entropy
        gather_hardware_entropy();

        // Wait for entropy
        WAITERS.wait_timeout(100); // 100ms timeout

        // Check for signals
        if let Some(proc) = crate::process::current() {
            if crate::signal::has_pending_signals(proc.pid) {
                return Err(RandomError::Interrupted);
            }
        }
    }

    let mut pool = POOL.lock();
    pool.extract(buf);

    STATS
        .bytes_generated
        .fetch_add(buf.len() as u64, Ordering::Relaxed);

    Ok(buf.len())
}

/// Get random bytes (non-blocking, /dev/urandom)
pub fn get_random_bytes(buf: &mut [u8]) -> usize {
    let mut pool = POOL.lock();

    // If not initialized, seed from hardware
    if !pool.is_initialized() {
        drop(pool);
        gather_hardware_entropy();
        gather_hardware_entropy();
        gather_hardware_entropy();
        pool = POOL.lock();
    }

    pool.extract(buf);

    STATS
        .bytes_generated
        .fetch_add(buf.len() as u64, Ordering::Relaxed);

    buf.len()
}

/// Get a random u32
pub fn get_random_u32() -> u32 {
    let mut buf = [0u8; 4];
    get_random_bytes(&mut buf);
    u32::from_ne_bytes(buf)
}

/// Get a random u64
pub fn get_random_u64() -> u64 {
    let mut buf = [0u8; 8];
    get_random_bytes(&mut buf);
    u64::from_ne_bytes(buf)
}

/// Get a random u32 in range [0, max)
pub fn get_random_u32_bounded(max: u32) -> u32 {
    if max == 0 {
        return 0;
    }

    // Rejection sampling for uniform distribution
    let threshold = u32::MAX - (u32::MAX % max);
    loop {
        let r = get_random_u32();
        if r < threshold {
            return r % max;
        }
    }
}

/// Fill buffer with random bytes (getrandom syscall)
pub fn getrandom(buf: &mut [u8], flags: u32) -> Result<usize, RandomError> {
    const GRND_NONBLOCK: u32 = 0x0001;
    const GRND_RANDOM: u32 = 0x0002;
    const GRND_INSECURE: u32 = 0x0004;

    if flags & GRND_RANDOM != 0 {
        // /dev/random behavior
        if flags & GRND_NONBLOCK != 0 {
            let pool = POOL.lock();
            if !pool.is_initialized() || pool.entropy_available() < buf.len() as u32 {
                return Err(RandomError::WouldBlock);
            }
        }
        return get_random_bytes_blocking(buf);
    }

    // /dev/urandom behavior
    if flags & GRND_NONBLOCK != 0 {
        let pool = POOL.lock();
        if !pool.is_initialized() && flags & GRND_INSECURE == 0 {
            return Err(RandomError::WouldBlock);
        }
    }

    Ok(get_random_bytes(buf))
}

/// Random error
#[derive(Clone, Copy, Debug)]
pub enum RandomError {
    WouldBlock,
    Interrupted,
    Invalid,
}

impl RandomError {
    pub fn to_errno(&self) -> i32 {
        match self {
            RandomError::WouldBlock => -11, // EAGAIN
            RandomError::Interrupted => -4, // EINTR
            RandomError::Invalid => -22,    // EINVAL
        }
    }
}

// ============================================================================
// /dev/random and /dev/urandom Interface
// ============================================================================

/// Read from /dev/random (blocking)
pub fn read_dev_random(buf: &mut [u8]) -> Result<usize, RandomError> {
    get_random_bytes_blocking(buf)
}

/// Read from /dev/urandom (non-blocking)
pub fn read_dev_urandom(buf: &mut [u8]) -> usize {
    get_random_bytes(buf)
}

/// Write to /dev/random (add entropy)
pub fn write_dev_random(data: &[u8]) -> usize {
    // User-provided data is not trusted for entropy estimation
    add_raw_entropy(data, 0);
    data.len()
}

/// IOCTL for /dev/random
pub fn ioctl_random(cmd: u32, arg: usize) -> Result<i32, RandomError> {
    const RNDGETENTCNT: u32 = 0x80045200;
    const RNDADDTOENTCNT: u32 = 0x40045201;
    const RNDGETPOOL: u32 = 0x80085202;
    const RNDADDENTROPY: u32 = 0x40085203;
    const RNDZAPENTCNT: u32 = 0x5204;
    const RNDCLEARPOOL: u32 = 0x5206;
    const RNDRESEEDCRNG: u32 = 0x5207;

    match cmd {
        RNDGETENTCNT => {
            let bits = POOL.lock().entropy_available();
            unsafe {
                *(arg as *mut u32) = bits;
            }
            Ok(0)
        }
        RNDADDTOENTCNT => {
            // Requires CAP_SYS_ADMIN
            if !crate::capability::has_current_capability(crate::capability::Capability::SysAdmin) {
                return Err(RandomError::Invalid);
            }
            let bits = unsafe { *(arg as *const u32) };
            let current = POOL.lock().entropy_bits.load(Ordering::Relaxed);
            POOL.lock().entropy_bits.store(
                current.saturating_add(bits).min(MAX_ENTROPY_BITS),
                Ordering::Relaxed,
            );
            Ok(0)
        }
        RNDRESEEDCRNG => {
            // Force reseed
            gather_hardware_entropy();
            STATS.pool_reseeds.fetch_add(1, Ordering::Relaxed);
            Ok(0)
        }
        RNDZAPENTCNT | RNDCLEARPOOL => {
            // Requires CAP_SYS_ADMIN
            if !crate::capability::has_current_capability(crate::capability::Capability::SysAdmin) {
                return Err(RandomError::Invalid);
            }
            POOL.lock().entropy_bits.store(0, Ordering::Relaxed);
            Ok(0)
        }
        _ => Err(RandomError::Invalid),
    }
}

// ============================================================================
// Syscall Interface
// ============================================================================

/// getrandom syscall
pub fn sys_getrandom(buf: usize, count: usize, flags: u32) -> isize {
    if buf == 0 || count == 0 {
        return 0;
    }

    let slice = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, count) };

    match getrandom(slice, flags) {
        Ok(n) => n as isize,
        Err(e) => e.to_errno() as isize,
    }
}

/// Get entropy information
pub fn entropy_avail() -> u32 {
    POOL.lock().entropy_available()
}

/// Get statistics
pub fn get_stats() -> (u64, u64, u64) {
    (
        STATS.bytes_generated.load(Ordering::Relaxed),
        STATS.entropy_added.load(Ordering::Relaxed),
        STATS.pool_reseeds.load(Ordering::Relaxed),
    )
}

/// Generate /proc/sys/kernel/random info
pub fn generate_proc_info() -> alloc::string::String {
    let pool = POOL.lock();
    let (bytes, entropy, reseeds) = get_stats();

    alloc::format!(
        "entropy_avail: {}\n\
         poolsize: {}\n\
         read_wakeup_threshold: {}\n\
         write_wakeup_threshold: {}\n\
         bytes_generated: {}\n\
         entropy_added: {}\n\
         reseeds: {}\n\
         initialized: {}\n",
        pool.entropy_available(),
        POOL_SIZE * 8,
        MIN_ENTROPY_BITS,
        MIN_ENTROPY_BITS,
        bytes,
        entropy,
        reseeds,
        pool.is_initialized(),
    )
}

/// Initialize random subsystem
pub fn init() {
    // Gather initial entropy
    for _ in 0..10 {
        gather_hardware_entropy();
    }

    crate::kprintln!(
        "  Random/entropy pool initialized ({} bits)",
        POOL.lock().entropy_available()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_generation() {
        let mut buf1 = [0u8; 32];
        let mut buf2 = [0u8; 32];

        get_random_bytes(&mut buf1);
        get_random_bytes(&mut buf2);

        // Should be different (extremely high probability)
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn test_bounded_random() {
        for _ in 0..1000 {
            let r = get_random_u32_bounded(100);
            assert!(r < 100);
        }
    }

    #[test]
    fn test_chacha_block() {
        let state = ChaChaState::new(&[0u8; 32]);
        let block = state.block();
        // Verify it produces output (not zeros)
        assert!(block.iter().any(|&b| b != 0));
    }
}
