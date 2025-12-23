//! GPIO abstraction

/// GPIO pin mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinMode {
    Input,
    Output,
    AltFunc(u8),
}

/// GPIO pull configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pull {
    None,
    Up,
    Down,
}

/// GPIO pin
pub struct Pin {
    number: u8,
    base: usize,
}

impl Pin {
    /// Create new pin
    pub const fn new(number: u8, base: usize) -> Self {
        Self { number, base }
    }

    /// Set pin mode
    pub fn set_mode(&self, mode: PinMode) {
        // Platform-specific implementation
        let _ = mode;
    }

    /// Set pull configuration
    pub fn set_pull(&self, pull: Pull) {
        // Platform-specific implementation
        let _ = pull;
    }

    /// Read pin value
    pub fn read(&self) -> bool {
        // Platform-specific implementation
        false
    }

    /// Write pin value
    pub fn write(&self, high: bool) {
        // Platform-specific implementation
        let _ = high;
    }

    /// Toggle pin
    pub fn toggle(&self) {
        let current = self.read();
        self.write(!current);
    }

    /// Get pin number
    pub fn number(&self) -> u8 {
        self.number
    }
}

/// GPIO controller
pub struct GpioController {
    base: usize,
}

impl GpioController {
    /// Create new GPIO controller
    pub const fn new(base: usize) -> Self {
        Self { base }
    }

    /// Get pin
    pub fn pin(&self, number: u8) -> Pin {
        Pin::new(number, self.base)
    }
}
