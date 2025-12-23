//! Time and Timer Subsystem
//!
//! System time, timers, and clock management.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;

/// System tick counter (incremented by timer interrupt)
static TICKS: AtomicU64 = AtomicU64::new(0);

/// Timer frequency in Hz
static TIMER_FREQ: AtomicU64 = AtomicU64::new(1000); // 1000 Hz = 1ms tick

/// Boot timestamp (nanoseconds since epoch, if RTC available)
static BOOT_TIME: AtomicU64 = AtomicU64::new(0);

/// Get current tick count
pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

/// Get uptime in milliseconds
pub fn uptime_ms() -> u64 {
    let ticks = TICKS.load(Ordering::Relaxed);
    let freq = TIMER_FREQ.load(Ordering::Relaxed);
    (ticks * 1000) / freq
}

/// Get uptime in seconds
pub fn uptime_secs() -> u64 {
    uptime_ms() / 1000
}

/// Get uptime as Duration-like struct
#[derive(Clone, Copy, Debug)]
pub struct Duration {
    pub secs: u64,
    pub nanos: u32,
}

impl Duration {
    pub const ZERO: Self = Self { secs: 0, nanos: 0 };

    pub fn from_millis(ms: u64) -> Self {
        Self {
            secs: ms / 1000,
            nanos: ((ms % 1000) * 1_000_000) as u32,
        }
    }

    pub fn from_secs(secs: u64) -> Self {
        Self { secs, nanos: 0 }
    }

    pub fn as_millis(&self) -> u64 {
        self.secs * 1000 + (self.nanos / 1_000_000) as u64
    }
}

pub fn uptime() -> Duration {
    let ms = uptime_ms();
    Duration::from_millis(ms)
}

/// Timestamp (seconds and nanoseconds since Unix epoch)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    pub secs: u64,
    pub nanos: u32,
}

impl Timestamp {
    pub const ZERO: Self = Self { secs: 0, nanos: 0 };

    pub fn now() -> Self {
        let boot = BOOT_TIME.load(Ordering::Relaxed);
        let uptime_ns = uptime_ms() * 1_000_000;
        let total_ns = boot + uptime_ns;

        Self {
            secs: total_ns / 1_000_000_000,
            nanos: (total_ns % 1_000_000_000) as u32,
        }
    }
}

/// Timer callback type
pub type TimerCallback = fn(usize);

/// Timer entry
struct TimerEntry {
    expires_at: u64, // Tick count when timer expires
    callback: TimerCallback,
    data: usize,
    periodic: Option<u64>, // Period in ticks (for repeating timers)
    id: u64,
}

impl PartialEq for TimerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.expires_at == other.expires_at
    }
}

impl Eq for TimerEntry {}

impl PartialOrd for TimerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimerEntry {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Reverse ordering for min-heap behavior
        other.expires_at.cmp(&self.expires_at)
    }
}

/// Timer queue
static TIMER_QUEUE: Mutex<BinaryHeap<TimerEntry>> = Mutex::new(BinaryHeap::new());
static NEXT_TIMER_ID: AtomicU64 = AtomicU64::new(1);

/// Schedule a one-shot timer
pub fn schedule_timer(delay_ms: u64, callback: TimerCallback, data: usize) -> u64 {
    let id = NEXT_TIMER_ID.fetch_add(1, Ordering::SeqCst);
    let freq = TIMER_FREQ.load(Ordering::Relaxed);
    let delay_ticks = (delay_ms * freq) / 1000;
    let expires_at = ticks() + delay_ticks;

    TIMER_QUEUE.lock().push(TimerEntry {
        expires_at,
        callback,
        data,
        periodic: None,
        id,
    });

    id
}

/// Schedule a periodic timer
pub fn schedule_periodic(period_ms: u64, callback: TimerCallback, data: usize) -> u64 {
    let id = NEXT_TIMER_ID.fetch_add(1, Ordering::SeqCst);
    let freq = TIMER_FREQ.load(Ordering::Relaxed);
    let period_ticks = (period_ms * freq) / 1000;
    let expires_at = ticks() + period_ticks;

    TIMER_QUEUE.lock().push(TimerEntry {
        expires_at,
        callback,
        data,
        periodic: Some(period_ticks),
        id,
    });

    id
}

/// Cancel a timer
pub fn cancel_timer(id: u64) {
    let mut queue = TIMER_QUEUE.lock();
    let entries: Vec<_> = core::mem::take(&mut *queue).into_vec();
    for entry in entries {
        if entry.id != id {
            queue.push(entry);
        }
    }
}

/// Process expired timers (called from timer interrupt)
pub fn process_timers() {
    let current_tick = ticks();
    let mut to_reschedule = Vec::new();

    {
        let mut queue = TIMER_QUEUE.lock();

        while let Some(entry) = queue.peek() {
            if entry.expires_at > current_tick {
                break;
            }

            let entry = queue.pop().unwrap();

            // Call the callback
            (entry.callback)(entry.data);

            // Reschedule if periodic
            if let Some(period) = entry.periodic {
                to_reschedule.push(TimerEntry {
                    expires_at: current_tick + period,
                    callback: entry.callback,
                    data: entry.data,
                    periodic: Some(period),
                    id: entry.id,
                });
            }
        }

        for entry in to_reschedule {
            queue.push(entry);
        }
    }
}

/// Tick handler (called from timer interrupt)
pub fn tick() {
    TICKS.fetch_add(1, Ordering::Relaxed);
    process_timers();
}

/// Sleep for specified milliseconds (busy wait)
pub fn sleep_ms(ms: u64) {
    let start = ticks();
    let freq = TIMER_FREQ.load(Ordering::Relaxed);
    let wait_ticks = (ms * freq) / 1000;

    while ticks() - start < wait_ticks {
        core::hint::spin_loop();
    }
}

/// Delay for specified microseconds (busy wait)
pub fn delay_us(us: u64) {
    // Use ARM system counter for precise timing
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let freq: u64;
        core::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq);

        let start: u64;
        core::arch::asm!("mrs {}, cntpct_el0", out(reg) start);

        let wait = (us * freq) / 1_000_000;

        loop {
            let current: u64;
            core::arch::asm!("mrs {}, cntpct_el0", out(reg) current);
            if current - start >= wait {
                break;
            }
            core::hint::spin_loop();
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback: convert to ms and use tick-based sleep
        sleep_ms(us / 1000);
    }
}

/// ARM Generic Timer support
pub mod arm_timer {
    use super::*;

    /// Timer base addresses
    pub const BCM2837_TIMER: usize = 0x3F003000; // Pi 3
    pub const BCM2711_TIMER: usize = 0xFE003000; // Pi 4

    /// System timer registers
    const CS: usize = 0x00; // Control/Status
    const CLO: usize = 0x04; // Counter lower 32 bits
    const CHI: usize = 0x08; // Counter upper 32 bits
    const C0: usize = 0x0C; // Compare 0
    const C1: usize = 0x10; // Compare 1
    const C2: usize = 0x14; // Compare 2
    const C3: usize = 0x18; // Compare 3

    /// Read timer counter (64-bit)
    pub fn read_counter(base: usize) -> u64 {
        unsafe {
            let lo = core::ptr::read_volatile((base + CLO) as *const u32) as u64;
            let hi = core::ptr::read_volatile((base + CHI) as *const u32) as u64;
            (hi << 32) | lo
        }
    }

    /// Set compare register
    pub fn set_compare(base: usize, channel: u8, value: u32) {
        let offset = match channel {
            0 => C0,
            1 => C1,
            2 => C2,
            3 => C3,
            _ => return,
        };
        unsafe {
            core::ptr::write_volatile((base + offset) as *mut u32, value);
        }
    }

    /// Clear interrupt for channel
    pub fn clear_interrupt(base: usize, channel: u8) {
        unsafe {
            core::ptr::write_volatile((base + CS) as *mut u32, 1 << channel);
        }
    }

    /// Initialize timer interrupt
    pub fn init_interrupt(base: usize, channel: u8, interval_us: u32) {
        let counter = read_counter(base) as u32;
        set_compare(base, channel, counter.wrapping_add(interval_us));
        clear_interrupt(base, channel);
    }

    /// Handle timer interrupt
    pub fn handle_interrupt(base: usize, channel: u8, interval_us: u32) {
        clear_interrupt(base, channel);

        let counter = read_counter(base) as u32;
        set_compare(base, channel, counter.wrapping_add(interval_us));

        tick();
    }
}

/// Generic Timer (ARM architected timer)
pub mod generic_timer {
    /// Read counter frequency
    pub fn frequency() -> u64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let freq: u64;
            core::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq);
            freq
        }

        #[cfg(not(target_arch = "aarch64"))]
        1_000_000 // Fallback
    }

    /// Read physical counter
    pub fn counter() -> u64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cnt: u64;
            core::arch::asm!("mrs {}, cntpct_el0", out(reg) cnt);
            cnt
        }

        #[cfg(not(target_arch = "aarch64"))]
        0
    }

    /// Enable timer interrupt
    pub fn enable(interval_ticks: u64) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Set compare value
            core::arch::asm!("msr cntp_tval_el0, {}", in(reg) interval_ticks);
            // Enable timer
            core::arch::asm!("msr cntp_ctl_el0, {}", in(reg) 1u64);
        }
    }

    /// Disable timer
    pub fn disable() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("msr cntp_ctl_el0, {}", in(reg) 0u64);
        }
    }
}

/// RTC (Real-Time Clock) support
pub mod rtc {
    use super::Timestamp;

    /// Read RTC time (platform-specific)
    pub fn read() -> Option<Timestamp> {
        // Would read from hardware RTC
        // Pi doesn't have RTC by default, would need external module
        None
    }

    /// Set RTC time
    pub fn write(_time: Timestamp) -> Result<(), &'static str> {
        Err("RTC not available")
    }

    /// Initialize RTC
    pub fn init() {
        if let Some(time) = read() {
            super::BOOT_TIME.store(
                time.secs * 1_000_000_000 + time.nanos as u64,
                core::sync::atomic::Ordering::SeqCst,
            );
        }
    }
}

/// Initialize time subsystem
pub fn init() {
    // Set timer frequency based on platform
    TIMER_FREQ.store(1000, Ordering::SeqCst); // 1000 Hz

    // Initialize RTC if available
    rtc::init();

    crate::kprintln!("  Time subsystem initialized");
}
