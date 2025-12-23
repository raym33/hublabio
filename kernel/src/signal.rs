//! Signal Handling
//!
//! POSIX-style signals for process communication and control.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};

/// Signal numbers (POSIX-compatible)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Signal {
    SIGHUP = 1,     // Hangup
    SIGINT = 2,     // Interrupt (Ctrl+C)
    SIGQUIT = 3,    // Quit
    SIGILL = 4,     // Illegal instruction
    SIGTRAP = 5,    // Trace trap
    SIGABRT = 6,    // Abort
    SIGBUS = 7,     // Bus error
    SIGFPE = 8,     // Floating point exception
    SIGKILL = 9,    // Kill (cannot be caught)
    SIGUSR1 = 10,   // User signal 1
    SIGSEGV = 11,   // Segmentation fault
    SIGUSR2 = 12,   // User signal 2
    SIGPIPE = 13,   // Broken pipe
    SIGALRM = 14,   // Alarm clock
    SIGTERM = 15,   // Termination
    SIGSTKFLT = 16, // Stack fault
    SIGCHLD = 17,   // Child stopped or terminated
    SIGCONT = 18,   // Continue
    SIGSTOP = 19,   // Stop (cannot be caught)
    SIGTSTP = 20,   // Terminal stop (Ctrl+Z)
    SIGTTIN = 21,   // Background read from tty
    SIGTTOU = 22,   // Background write to tty
    SIGURG = 23,    // Urgent condition on socket
    SIGXCPU = 24,   // CPU time limit exceeded
    SIGXFSZ = 25,   // File size limit exceeded
    SIGVTALRM = 26, // Virtual alarm clock
    SIGPROF = 27,   // Profiling timer expired
    SIGWINCH = 28,  // Window resize
    SIGIO = 29,     // I/O possible
    SIGPWR = 30,    // Power failure
    SIGSYS = 31,    // Bad system call
}

impl Signal {
    pub fn from_num(num: u8) -> Option<Self> {
        match num {
            1 => Some(Self::SIGHUP),
            2 => Some(Self::SIGINT),
            3 => Some(Self::SIGQUIT),
            4 => Some(Self::SIGILL),
            5 => Some(Self::SIGTRAP),
            6 => Some(Self::SIGABRT),
            7 => Some(Self::SIGBUS),
            8 => Some(Self::SIGFPE),
            9 => Some(Self::SIGKILL),
            10 => Some(Self::SIGUSR1),
            11 => Some(Self::SIGSEGV),
            12 => Some(Self::SIGUSR2),
            13 => Some(Self::SIGPIPE),
            14 => Some(Self::SIGALRM),
            15 => Some(Self::SIGTERM),
            16 => Some(Self::SIGSTKFLT),
            17 => Some(Self::SIGCHLD),
            18 => Some(Self::SIGCONT),
            19 => Some(Self::SIGSTOP),
            20 => Some(Self::SIGTSTP),
            21 => Some(Self::SIGTTIN),
            22 => Some(Self::SIGTTOU),
            23 => Some(Self::SIGURG),
            24 => Some(Self::SIGXCPU),
            25 => Some(Self::SIGXFSZ),
            26 => Some(Self::SIGVTALRM),
            27 => Some(Self::SIGPROF),
            28 => Some(Self::SIGWINCH),
            29 => Some(Self::SIGIO),
            30 => Some(Self::SIGPWR),
            31 => Some(Self::SIGSYS),
            _ => None,
        }
    }

    pub fn as_num(&self) -> u8 {
        *self as u8
    }

    /// Check if signal can be caught or ignored
    pub fn can_catch(&self) -> bool {
        !matches!(self, Signal::SIGKILL | Signal::SIGSTOP)
    }

    /// Check if signal terminates by default
    pub fn terminates(&self) -> bool {
        matches!(
            self,
            Signal::SIGHUP
                | Signal::SIGINT
                | Signal::SIGQUIT
                | Signal::SIGILL
                | Signal::SIGTRAP
                | Signal::SIGABRT
                | Signal::SIGBUS
                | Signal::SIGFPE
                | Signal::SIGKILL
                | Signal::SIGSEGV
                | Signal::SIGPIPE
                | Signal::SIGALRM
                | Signal::SIGTERM
                | Signal::SIGUSR1
                | Signal::SIGUSR2
                | Signal::SIGSTKFLT
                | Signal::SIGXCPU
                | Signal::SIGXFSZ
                | Signal::SIGSYS
        )
    }

    /// Check if signal generates core dump
    pub fn core_dumps(&self) -> bool {
        matches!(
            self,
            Signal::SIGQUIT
                | Signal::SIGILL
                | Signal::SIGABRT
                | Signal::SIGFPE
                | Signal::SIGSEGV
                | Signal::SIGBUS
                | Signal::SIGTRAP
                | Signal::SIGXCPU
                | Signal::SIGXFSZ
                | Signal::SIGSYS
        )
    }

    /// Check if signal stops process
    pub fn stops(&self) -> bool {
        matches!(
            self,
            Signal::SIGSTOP | Signal::SIGTSTP | Signal::SIGTTIN | Signal::SIGTTOU
        )
    }
}

/// Signal action
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalAction {
    /// Default action
    Default,
    /// Ignore signal
    Ignore,
    /// Custom handler (function pointer as usize)
    Handler(usize),
    /// Terminate process
    Terminate,
    /// Stop process
    Stop,
    /// Continue process
    Continue,
}

/// Signal mask (bitmap of blocked signals)
#[derive(Clone, Copy, Debug, Default)]
pub struct SignalMask(pub u64);

impl SignalMask {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn block(&mut self, sig: Signal) {
        self.0 |= 1 << sig.as_num();
    }

    pub fn unblock(&mut self, sig: Signal) {
        self.0 &= !(1 << sig.as_num());
    }

    pub fn is_blocked(&self, sig: Signal) -> bool {
        (self.0 & (1 << sig.as_num())) != 0
    }

    pub fn block_all(&mut self) {
        // Don't block SIGKILL and SIGSTOP
        self.0 = !0u64 & !(1 << Signal::SIGKILL.as_num()) & !(1 << Signal::SIGSTOP.as_num());
    }

    pub fn unblock_all(&mut self) {
        self.0 = 0;
    }
}

/// Pending signals for a process
#[derive(Clone, Debug, Default)]
pub struct PendingSignals {
    mask: u64,
    info: BTreeMap<u8, SignalInfo>,
}

/// Signal information
#[derive(Clone, Debug)]
pub struct SignalInfo {
    pub signal: Signal,
    pub sender_pid: u32,
    pub code: i32,
    pub value: usize,
}

impl PendingSignals {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pending signal
    pub fn add(&mut self, info: SignalInfo) {
        let num = info.signal.as_num();
        self.mask |= 1 << num;
        self.info.insert(num, info);
    }

    /// Check if any signals are pending
    pub fn any_pending(&self) -> bool {
        self.mask != 0
    }

    /// Check if specific signal is pending
    pub fn is_pending(&self, sig: Signal) -> bool {
        (self.mask & (1 << sig.as_num())) != 0
    }

    /// Get next pending signal (not blocked)
    pub fn next(&mut self, blocked: &SignalMask) -> Option<SignalInfo> {
        let deliverable = self.mask & !blocked.0;

        if deliverable == 0 {
            return None;
        }

        // Find lowest numbered pending signal
        let num = deliverable.trailing_zeros() as u8;
        self.mask &= !(1 << num);
        self.info.remove(&num)
    }

    /// Clear a specific signal
    pub fn clear(&mut self, sig: Signal) {
        let num = sig.as_num();
        self.mask &= !(1 << num);
        self.info.remove(&num);
    }

    /// Clear all signals
    pub fn clear_all(&mut self) {
        self.mask = 0;
        self.info.clear();
    }
}

/// Signal handlers for a process
#[derive(Clone, Debug)]
pub struct SignalHandlers {
    actions: [SignalAction; 32],
}

impl Default for SignalHandlers {
    fn default() -> Self {
        Self {
            actions: [SignalAction::Default; 32],
        }
    }
}

impl SignalHandlers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get action for signal
    pub fn get(&self, sig: Signal) -> SignalAction {
        let num = sig.as_num() as usize;
        if num < 32 {
            self.actions[num]
        } else {
            SignalAction::Default
        }
    }

    /// Set action for signal
    pub fn set(&mut self, sig: Signal, action: SignalAction) -> Result<(), &'static str> {
        if !sig.can_catch() && action != SignalAction::Default {
            return Err("Cannot change handler for SIGKILL or SIGSTOP");
        }

        let num = sig.as_num() as usize;
        if num < 32 {
            self.actions[num] = action;
            Ok(())
        } else {
            Err("Invalid signal number")
        }
    }

    /// Reset to default
    pub fn reset(&mut self, sig: Signal) {
        let num = sig.as_num() as usize;
        if num < 32 {
            self.actions[num] = SignalAction::Default;
        }
    }

    /// Reset all handlers (called on exec)
    pub fn reset_all(&mut self) {
        for action in &mut self.actions {
            if *action == SignalAction::Handler(_) {
                *action = SignalAction::Default;
            }
        }
    }
}

/// Per-process signal state
pub struct SignalState {
    pub handlers: SignalHandlers,
    pub pending: PendingSignals,
    pub blocked: SignalMask,
    pub in_handler: bool,
}

impl Default for SignalState {
    fn default() -> Self {
        Self {
            handlers: SignalHandlers::new(),
            pending: PendingSignals::new(),
            blocked: SignalMask::new(),
            in_handler: false,
        }
    }
}

impl SignalState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Global signal statistics
static SIGNALS_SENT: AtomicU64 = AtomicU64::new(0);
static SIGNALS_DELIVERED: AtomicU64 = AtomicU64::new(0);

/// Send signal to process
pub fn send_signal(pid: u32, sig: Signal, sender_pid: u32) -> Result<(), &'static str> {
    SIGNALS_SENT.fetch_add(1, Ordering::Relaxed);

    // Would look up process and add to its pending signals
    crate::kdebug!("Signal: Sending {:?} to pid {}", sig, pid);

    // This would integrate with the process manager
    // For now, just log it

    Ok(())
}

/// Send signal to process group
pub fn send_signal_group(pgid: u32, sig: Signal, sender_pid: u32) -> Result<(), &'static str> {
    crate::kdebug!("Signal: Sending {:?} to process group {}", sig, pgid);
    Ok(())
}

/// Deliver pending signals to current process
pub fn deliver_signals(_state: &mut SignalState) -> Option<Signal> {
    // Would be called before returning to userspace
    // Check for pending signals and handle them

    None
}

/// Signal statistics
pub fn stats() -> (u64, u64) {
    (
        SIGNALS_SENT.load(Ordering::Relaxed),
        SIGNALS_DELIVERED.load(Ordering::Relaxed),
    )
}

/// Initialize signal subsystem
pub fn init() {
    crate::kprintln!("  Signal subsystem initialized");
}
