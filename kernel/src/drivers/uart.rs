//! UART Driver
//!
//! Universal Asynchronous Receiver/Transmitter driver.
//! Supports:
//! - BCM2711/BCM2712 (Raspberry Pi 4/5) - PL011 UART
//! - Generic 16550 UART
//! - QEMU virt machine UART

use core::fmt::{self, Write};
use core::ptr::{read_volatile, write_volatile};
use spin::Mutex;

/// UART base addresses for different platforms
pub mod base {
    /// Raspberry Pi 4 (BCM2711) PL011 UART0
    pub const BCM2711_UART0: usize = 0xFE201000;

    /// Raspberry Pi 5 (BCM2712) PL011 UART0
    pub const BCM2712_UART0: usize = 0x107D001000;

    /// QEMU virt machine PL011
    pub const QEMU_VIRT: usize = 0x0900_0000;

    /// Raspberry Pi 3 (BCM2837) PL011
    pub const BCM2837_UART0: usize = 0x3F201000;
}

/// PL011 UART registers (offsets from base)
pub mod pl011 {
    pub const DR: usize = 0x00;      // Data register
    pub const FR: usize = 0x18;      // Flag register
    pub const IBRD: usize = 0x24;    // Integer baud rate divisor
    pub const FBRD: usize = 0x28;    // Fractional baud rate divisor
    pub const LCRH: usize = 0x2C;    // Line control register
    pub const CR: usize = 0x30;      // Control register
    pub const IMSC: usize = 0x38;    // Interrupt mask set/clear
    pub const ICR: usize = 0x44;     // Interrupt clear register

    // Flag register bits
    pub const FR_TXFF: u32 = 1 << 5;   // Transmit FIFO full
    pub const FR_RXFE: u32 = 1 << 4;   // Receive FIFO empty
    pub const FR_BUSY: u32 = 1 << 3;   // UART busy

    // Line control register bits
    pub const LCRH_WLEN_8: u32 = 0b11 << 5;  // 8-bit words
    pub const LCRH_FEN: u32 = 1 << 4;        // Enable FIFOs

    // Control register bits
    pub const CR_UARTEN: u32 = 1 << 0;  // UART enable
    pub const CR_TXE: u32 = 1 << 8;     // Transmit enable
    pub const CR_RXE: u32 = 1 << 9;     // Receive enable
}

/// Global UART instance
pub static UART: Mutex<Option<Uart>> = Mutex::new(None);

/// UART driver
pub struct Uart {
    base: usize,
    initialized: bool,
}

impl Uart {
    /// Create a new UART driver (uninitialized)
    pub const fn new(base: usize) -> Self {
        Self {
            base,
            initialized: false,
        }
    }

    /// Read a register
    fn read_reg(&self, offset: usize) -> u32 {
        unsafe { read_volatile((self.base + offset) as *const u32) }
    }

    /// Write a register
    fn write_reg(&self, offset: usize, value: u32) {
        unsafe { write_volatile((self.base + offset) as *mut u32, value) }
    }

    /// Initialize PL011 UART
    pub fn init_pl011(&mut self, baud: u32, clock: u32) {
        // Disable UART
        self.write_reg(pl011::CR, 0);

        // Wait for UART to be idle
        while self.read_reg(pl011::FR) & pl011::FR_BUSY != 0 {}

        // Calculate baud rate divisor
        // Divisor = UART_CLK / (16 * baud)
        let divisor = clock / (16 * baud);
        let fractional = ((clock % (16 * baud)) * 64 + baud / 2) / baud;

        self.write_reg(pl011::IBRD, divisor);
        self.write_reg(pl011::FBRD, fractional);

        // Set 8N1, enable FIFOs
        self.write_reg(pl011::LCRH, pl011::LCRH_WLEN_8 | pl011::LCRH_FEN);

        // Clear all interrupts
        self.write_reg(pl011::ICR, 0x7FF);

        // Disable all interrupts
        self.write_reg(pl011::IMSC, 0);

        // Enable UART, TX, and RX
        self.write_reg(pl011::CR, pl011::CR_UARTEN | pl011::CR_TXE | pl011::CR_RXE);

        self.initialized = true;
    }

    /// Initialize for QEMU virt machine (minimal init)
    pub fn init_qemu(&mut self) {
        // QEMU PL011 is already configured, just enable it
        self.write_reg(pl011::CR, pl011::CR_UARTEN | pl011::CR_TXE | pl011::CR_RXE);
        self.initialized = true;
    }

    /// Write a single byte
    pub fn write_byte(&self, byte: u8) {
        if !self.initialized {
            return;
        }

        // Wait until TX FIFO is not full
        while self.read_reg(pl011::FR) & pl011::FR_TXFF != 0 {}

        // Write byte to data register
        self.write_reg(pl011::DR, byte as u32);
    }

    /// Read a single byte (blocking)
    pub fn read_byte(&self) -> u8 {
        if !self.initialized {
            return 0;
        }

        // Wait until RX FIFO is not empty
        while self.read_reg(pl011::FR) & pl011::FR_RXFE != 0 {}

        // Read byte from data register
        (self.read_reg(pl011::DR) & 0xFF) as u8
    }

    /// Try to read a byte (non-blocking)
    pub fn try_read_byte(&self) -> Option<u8> {
        if !self.initialized {
            return None;
        }

        if self.read_reg(pl011::FR) & pl011::FR_RXFE != 0 {
            None
        } else {
            Some((self.read_reg(pl011::DR) & 0xFF) as u8)
        }
    }

    /// Write a string
    pub fn write_str(&self, s: &str) {
        for byte in s.bytes() {
            if byte == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(byte);
        }
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Write for Uart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        Uart::write_str(self, s);
        Ok(())
    }
}

/// Initialize UART with auto-detection
pub fn init() {
    let mut uart = Uart::new(base::QEMU_VIRT);

    // For now, assume QEMU virt machine
    // In real hardware, we'd detect from device tree
    uart.init_qemu();

    *UART.lock() = Some(uart);
}

/// Initialize UART with specific base address
pub fn init_at(base_addr: usize, baud: u32, clock: u32) {
    let mut uart = Uart::new(base_addr);
    uart.init_pl011(baud, clock);
    *UART.lock() = Some(uart);
}

/// Write a string to UART
pub fn write(s: &str) {
    if let Some(ref uart) = *UART.lock() {
        uart.write_str(s);
    }
}

/// Write a single character
pub fn putc(c: u8) {
    if let Some(ref uart) = *UART.lock() {
        if c == b'\n' {
            uart.write_byte(b'\r');
        }
        uart.write_byte(c);
    }
}

/// Read a single character (blocking)
pub fn getc() -> u8 {
    loop {
        if let Some(ref uart) = *UART.lock() {
            return uart.read_byte();
        }
    }
}

/// Try to read a character (non-blocking)
pub fn try_getc() -> Option<u8> {
    if let Some(ref uart) = *UART.lock() {
        uart.try_read_byte()
    } else {
        None
    }
}

/// Check if UART is ready
pub fn is_ready() -> bool {
    UART.lock().as_ref().map(|u| u.is_initialized()).unwrap_or(false)
}
