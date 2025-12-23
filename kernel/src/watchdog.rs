//! Watchdog Timer
//!
//! Hardware and software watchdog support for system stability.
//! Detects lockups and triggers recovery actions.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use spin::Mutex;

/// Watchdog configuration
pub struct WatchdogConfig {
    /// Timeout in seconds before reset
    pub timeout: u32,
    /// Pre-timeout warning (seconds before timeout)
    pub pretimeout: u32,
    /// Action on timeout
    pub action: WatchdogAction,
    /// Whether watchdog is enabled
    pub enabled: bool,
    /// Maximum timeout supported by hardware
    pub max_timeout: u32,
    /// Minimum timeout supported
    pub min_timeout: u32,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            timeout: 60,
            pretimeout: 10,
            action: WatchdogAction::Reset,
            enabled: false,
            max_timeout: 600,
            min_timeout: 1,
        }
    }
}

/// Action to take on watchdog timeout
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatchdogAction {
    /// Do nothing (testing)
    None,
    /// Panic the kernel
    Panic,
    /// Soft reboot
    Reboot,
    /// Hard reset
    Reset,
    /// Power off
    PowerOff,
}

/// Watchdog state
pub struct Watchdog {
    /// Configuration
    config: Mutex<WatchdogConfig>,
    /// Last ping time (nanoseconds)
    last_ping: AtomicU64,
    /// Running state
    running: AtomicBool,
    /// Expired flag
    expired: AtomicBool,
    /// Boot time (for uptime calculation)
    boot_time: AtomicU64,
    /// Number of timeouts
    timeout_count: AtomicU32,
}

impl Watchdog {
    pub const fn new() -> Self {
        Self {
            config: Mutex::new(WatchdogConfig {
                timeout: 60,
                pretimeout: 10,
                action: WatchdogAction::Reset,
                enabled: false,
                max_timeout: 600,
                min_timeout: 1,
            }),
            last_ping: AtomicU64::new(0),
            running: AtomicBool::new(false),
            expired: AtomicBool::new(false),
            boot_time: AtomicU64::new(0),
            timeout_count: AtomicU32::new(0),
        }
    }

    /// Start the watchdog
    pub fn start(&self) {
        self.last_ping.store(current_time_ns(), Ordering::SeqCst);
        self.running.store(true, Ordering::SeqCst);
        self.expired.store(false, Ordering::SeqCst);

        let config = self.config.lock();
        crate::kinfo!("Watchdog started (timeout={}s)", config.timeout);
    }

    /// Stop the watchdog
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        crate::kinfo!("Watchdog stopped");
    }

    /// Ping (keep-alive)
    pub fn ping(&self) {
        if self.running.load(Ordering::SeqCst) {
            self.last_ping.store(current_time_ns(), Ordering::SeqCst);
            self.expired.store(false, Ordering::SeqCst);
        }
    }

    /// Set timeout in seconds
    pub fn set_timeout(&self, seconds: u32) -> Result<(), WatchdogError> {
        let mut config = self.config.lock();

        if seconds < config.min_timeout || seconds > config.max_timeout {
            return Err(WatchdogError::InvalidTimeout);
        }

        config.timeout = seconds;
        Ok(())
    }

    /// Get timeout in seconds
    pub fn get_timeout(&self) -> u32 {
        self.config.lock().timeout
    }

    /// Set pretimeout
    pub fn set_pretimeout(&self, seconds: u32) -> Result<(), WatchdogError> {
        let mut config = self.config.lock();

        if seconds >= config.timeout {
            return Err(WatchdogError::InvalidTimeout);
        }

        config.pretimeout = seconds;
        Ok(())
    }

    /// Set action
    pub fn set_action(&self, action: WatchdogAction) {
        self.config.lock().action = action;
    }

    /// Check watchdog (called from timer interrupt)
    pub fn check(&self) {
        if !self.running.load(Ordering::SeqCst) {
            return;
        }

        let now = current_time_ns();
        let last = self.last_ping.load(Ordering::SeqCst);
        let config = self.config.lock();

        let elapsed_ns = now.saturating_sub(last);
        let elapsed_s = elapsed_ns / 1_000_000_000;

        // Check for pretimeout warning
        if config.pretimeout > 0 && elapsed_s >= (config.timeout - config.pretimeout) as u64 {
            if !self.expired.load(Ordering::SeqCst) {
                crate::kwarn!(
                    "Watchdog: {}s until timeout!",
                    config.timeout as u64 - elapsed_s
                );
            }
        }

        // Check for timeout
        if elapsed_s >= config.timeout as u64 {
            if !self.expired.swap(true, Ordering::SeqCst) {
                self.timeout_count.fetch_add(1, Ordering::SeqCst);
                self.handle_timeout(config.action);
            }
        }
    }

    /// Handle watchdog timeout
    fn handle_timeout(&self, action: WatchdogAction) {
        crate::kerror!("Watchdog timeout! Action: {:?}", action);

        match action {
            WatchdogAction::None => {
                crate::kwarn!("Watchdog: No action configured");
            }
            WatchdogAction::Panic => {
                panic!("Watchdog timeout - system hung");
            }
            WatchdogAction::Reboot => {
                crate::kerror!("Watchdog: Initiating soft reboot");
                // Attempt clean shutdown
                soft_reboot();
            }
            WatchdogAction::Reset => {
                crate::kerror!("Watchdog: Initiating hard reset");
                hard_reset();
            }
            WatchdogAction::PowerOff => {
                crate::kerror!("Watchdog: Powering off");
                power_off();
            }
        }
    }

    /// Get time remaining until timeout (seconds)
    pub fn time_remaining(&self) -> Option<u32> {
        if !self.running.load(Ordering::SeqCst) {
            return None;
        }

        let now = current_time_ns();
        let last = self.last_ping.load(Ordering::SeqCst);
        let config = self.config.lock();

        let elapsed_s = (now.saturating_sub(last) / 1_000_000_000) as u32;

        if elapsed_s >= config.timeout {
            Some(0)
        } else {
            Some(config.timeout - elapsed_s)
        }
    }

    /// Get status
    pub fn status(&self) -> WatchdogStatus {
        let config = self.config.lock();
        WatchdogStatus {
            running: self.running.load(Ordering::SeqCst),
            timeout: config.timeout,
            pretimeout: config.pretimeout,
            action: config.action,
            time_remaining: self.time_remaining(),
            timeout_count: self.timeout_count.load(Ordering::SeqCst),
        }
    }
}

/// Watchdog status
#[derive(Clone, Debug)]
pub struct WatchdogStatus {
    pub running: bool,
    pub timeout: u32,
    pub pretimeout: u32,
    pub action: WatchdogAction,
    pub time_remaining: Option<u32>,
    pub timeout_count: u32,
}

/// Watchdog error
#[derive(Clone, Copy, Debug)]
pub enum WatchdogError {
    InvalidTimeout,
    NotSupported,
    Busy,
}

/// Global watchdog instance
static WATCHDOG: Watchdog = Watchdog::new();

// ============================================================================
// Soft Lockup Detector
// ============================================================================

/// Per-CPU soft lockup state
pub struct SoftLockupDetector {
    /// Timestamp of last scheduler tick
    last_tick: AtomicU64,
    /// Threshold in seconds
    threshold: AtomicU32,
    /// Whether detector is enabled
    enabled: AtomicBool,
    /// Number of detected lockups
    lockup_count: AtomicU32,
}

impl SoftLockupDetector {
    pub const fn new() -> Self {
        Self {
            last_tick: AtomicU64::new(0),
            threshold: AtomicU32::new(20), // 20 second default
            enabled: AtomicBool::new(true),
            lockup_count: AtomicU32::new(0),
        }
    }

    /// Touch - called from scheduler tick
    pub fn touch(&self) {
        self.last_tick.store(current_time_ns(), Ordering::SeqCst);
    }

    /// Check for soft lockup
    pub fn check(&self) {
        if !self.enabled.load(Ordering::SeqCst) {
            return;
        }

        let now = current_time_ns();
        let last = self.last_tick.load(Ordering::SeqCst);
        let threshold = self.threshold.load(Ordering::SeqCst);

        let elapsed_s = (now.saturating_sub(last) / 1_000_000_000) as u32;

        if elapsed_s > threshold && last > 0 {
            self.lockup_count.fetch_add(1, Ordering::SeqCst);
            crate::kerror!("Soft lockup detected! CPU stuck for {} seconds", elapsed_s);

            // Dump state
            crate::arch::dump_state();

            // Optionally panic
            // panic!("Soft lockup - CPU stuck");
        }
    }

    /// Set threshold
    pub fn set_threshold(&self, seconds: u32) {
        self.threshold.store(seconds, Ordering::SeqCst);
    }

    /// Enable/disable
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }
}

/// Global soft lockup detector
static SOFT_LOCKUP: SoftLockupDetector = SoftLockupDetector::new();

// ============================================================================
// Hard Lockup Detector (NMI-based)
// ============================================================================

/// Hard lockup detector using NMI/performance counter
pub struct HardLockupDetector {
    /// Hrtimer counter
    hrtimer_count: AtomicU64,
    /// Last seen count
    last_count: AtomicU64,
    /// Threshold (missed hrtimer interrupts)
    threshold: AtomicU32,
    /// Enabled flag
    enabled: AtomicBool,
    /// Lockup count
    lockup_count: AtomicU32,
}

impl HardLockupDetector {
    pub const fn new() -> Self {
        Self {
            hrtimer_count: AtomicU64::new(0),
            last_count: AtomicU64::new(0),
            threshold: AtomicU32::new(10),
            enabled: AtomicBool::new(false), // Disabled by default
            lockup_count: AtomicU32::new(0),
        }
    }

    /// Increment hrtimer count (called from hrtimer interrupt)
    pub fn tick(&self) {
        self.hrtimer_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Check for hard lockup (called from NMI)
    pub fn check(&self) {
        if !self.enabled.load(Ordering::SeqCst) {
            return;
        }

        let current = self.hrtimer_count.load(Ordering::SeqCst);
        let last = self.last_count.swap(current, Ordering::SeqCst);

        if current == last {
            // Hrtimer hasn't fired - possible hard lockup
            let threshold = self.threshold.load(Ordering::SeqCst);
            // Would need actual tracking of missed ticks
            self.lockup_count.fetch_add(1, Ordering::SeqCst);
            crate::kerror!("Hard lockup suspected - interrupts blocked");
        }
    }
}

/// Global hard lockup detector
static HARD_LOCKUP: HardLockupDetector = HardLockupDetector::new();

// ============================================================================
// System Functions
// ============================================================================

fn current_time_ns() -> u64 {
    crate::time::monotonic_ns()
}

fn soft_reboot() {
    // Try to cleanly shutdown
    crate::kinfo!("Initiating soft reboot...");

    // Send SIGTERM to all processes
    // Sync filesystems
    // Unmount

    // Then hard reset
    hard_reset();
}

fn hard_reset() {
    crate::kerror!("Hard reset!");
    crate::arch::reboot();
}

fn power_off() {
    crate::kinfo!("Powering off...");
    // Platform-specific power off
    // For now, just halt
    loop {
        crate::arch::halt();
    }
}

// ============================================================================
// Public Interface
// ============================================================================

/// Start watchdog
pub fn start() {
    WATCHDOG.start();
}

/// Stop watchdog
pub fn stop() {
    WATCHDOG.stop();
}

/// Ping watchdog
pub fn ping() {
    WATCHDOG.ping();
}

/// Set watchdog timeout
pub fn set_timeout(seconds: u32) -> Result<(), WatchdogError> {
    WATCHDOG.set_timeout(seconds)
}

/// Get watchdog timeout
pub fn get_timeout() -> u32 {
    WATCHDOG.get_timeout()
}

/// Set watchdog action
pub fn set_action(action: WatchdogAction) {
    WATCHDOG.set_action(action);
}

/// Get watchdog status
pub fn status() -> WatchdogStatus {
    WATCHDOG.status()
}

/// Check watchdog (call from timer)
pub fn check() {
    WATCHDOG.check();
    SOFT_LOCKUP.check();
}

/// Touch soft lockup detector (call from scheduler)
pub fn touch_softlockup() {
    SOFT_LOCKUP.touch();
}

/// Generate /dev/watchdog info
pub fn generate_info() -> alloc::string::String {
    let status = WATCHDOG.status();
    alloc::format!(
        "running: {}\n\
         timeout: {}\n\
         pretimeout: {}\n\
         time_remaining: {:?}\n\
         timeout_count: {}\n",
        status.running,
        status.timeout,
        status.pretimeout,
        status.time_remaining,
        status.timeout_count
    )
}

/// Initialize watchdog subsystem
pub fn init() {
    WATCHDOG
        .boot_time
        .store(current_time_ns(), Ordering::SeqCst);

    // Initialize soft lockup detector
    SOFT_LOCKUP.touch();

    crate::kprintln!("  Watchdog timer initialized (soft/hard lockup detection)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_config() {
        let wd = Watchdog::new();

        assert_eq!(wd.get_timeout(), 60);

        wd.set_timeout(30).unwrap();
        assert_eq!(wd.get_timeout(), 30);

        assert!(wd.set_timeout(0).is_err());
        assert!(wd.set_timeout(1000).is_err());
    }

    #[test]
    fn test_watchdog_lifecycle() {
        let wd = Watchdog::new();

        assert!(!wd.status().running);

        wd.start();
        assert!(wd.status().running);

        wd.ping();

        wd.stop();
        assert!(!wd.status().running);
    }
}
