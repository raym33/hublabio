//! UART abstraction

use core::fmt::{self, Write};

/// UART instance
pub struct Uart {
    base: usize,
}

impl Uart {
    /// Create new UART at base address
    pub const fn new(base: usize) -> Self {
        Self { base }
    }

    /// Initialize UART
    pub fn init(&self, baud: u32) {
        // Platform-specific initialization would go here
        let _ = baud;
    }

    /// Write a byte
    pub fn write_byte(&self, byte: u8) {
        // Platform-specific write
        unsafe {
            core::ptr::write_volatile(self.base as *mut u8, byte);
        }
    }

    /// Read a byte (blocking)
    pub fn read_byte(&self) -> u8 {
        // Platform-specific read
        unsafe { core::ptr::read_volatile(self.base as *const u8) }
    }

    /// Check if data available
    pub fn data_available(&self) -> bool {
        // Platform-specific check
        false
    }

    /// Write string
    pub fn write_str(&self, s: &str) {
        for byte in s.bytes() {
            if byte == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(byte);
        }
    }
}

impl Write for Uart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        Uart::write_str(self, s);
        Ok(())
    }
}

/// Global debug UART (set during init)
static mut DEBUG_UART: Option<Uart> = None;

/// Initialize debug UART
pub fn init_debug(base: usize, baud: u32) {
    let uart = Uart::new(base);
    uart.init(baud);
    unsafe {
        DEBUG_UART = Some(uart);
    }
}

/// Print to debug UART
pub fn debug_print(s: &str) {
    unsafe {
        if let Some(ref uart) = DEBUG_UART {
            uart.write_str(s);
        }
    }
}
