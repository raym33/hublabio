//! GPIO Driver
//!
//! General Purpose Input/Output driver for Raspberry Pi.

use core::ptr::{read_volatile, write_volatile};
use spin::Mutex;

/// GPIO base addresses
pub mod base {
    /// Raspberry Pi 4 (BCM2711)
    pub const BCM2711: usize = 0xFE200000;

    /// Raspberry Pi 5 (BCM2712)
    pub const BCM2712: usize = 0x107D517C00;

    /// Raspberry Pi 3 (BCM2837)
    pub const BCM2837: usize = 0x3F200000;
}

/// GPIO register offsets
pub mod regs {
    pub const GPFSEL0: usize = 0x00;   // GPIO Function Select 0
    pub const GPFSEL1: usize = 0x04;
    pub const GPFSEL2: usize = 0x08;
    pub const GPFSEL3: usize = 0x0C;
    pub const GPFSEL4: usize = 0x10;
    pub const GPFSEL5: usize = 0x14;

    pub const GPSET0: usize = 0x1C;    // GPIO Pin Output Set 0
    pub const GPSET1: usize = 0x20;

    pub const GPCLR0: usize = 0x28;    // GPIO Pin Output Clear 0
    pub const GPCLR1: usize = 0x2C;

    pub const GPLEV0: usize = 0x34;    // GPIO Pin Level 0
    pub const GPLEV1: usize = 0x38;

    pub const GPEDS0: usize = 0x40;    // GPIO Event Detect Status 0
    pub const GPEDS1: usize = 0x44;

    pub const GPREN0: usize = 0x4C;    // GPIO Rising Edge Detect Enable 0
    pub const GPREN1: usize = 0x50;

    pub const GPFEN0: usize = 0x58;    // GPIO Falling Edge Detect Enable 0
    pub const GPFEN1: usize = 0x5C;

    pub const GPPUD: usize = 0x94;     // GPIO Pull-up/down Enable
    pub const GPPUDCLK0: usize = 0x98; // GPIO Pull-up/down Enable Clock 0
    pub const GPPUDCLK1: usize = 0x9C;

    // BCM2711 (Pi 4) has different pull-up/down registers
    pub const GPIO_PUP_PDN_CNTRL0: usize = 0xE4;
    pub const GPIO_PUP_PDN_CNTRL1: usize = 0xE8;
    pub const GPIO_PUP_PDN_CNTRL2: usize = 0xEC;
    pub const GPIO_PUP_PDN_CNTRL3: usize = 0xF0;
}

/// GPIO pin function
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PinFunction {
    Input = 0b000,
    Output = 0b001,
    Alt0 = 0b100,
    Alt1 = 0b101,
    Alt2 = 0b110,
    Alt3 = 0b111,
    Alt4 = 0b011,
    Alt5 = 0b010,
}

/// GPIO pull-up/down mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PullMode {
    None = 0,
    PullUp = 1,
    PullDown = 2,
}

/// Global GPIO instance
pub static GPIO: Mutex<Option<Gpio>> = Mutex::new(None);

/// GPIO driver
pub struct Gpio {
    base: usize,
    is_bcm2711: bool,
}

impl Gpio {
    /// Create a new GPIO driver
    pub const fn new(base: usize, is_bcm2711: bool) -> Self {
        Self { base, is_bcm2711 }
    }

    /// Read a register
    fn read_reg(&self, offset: usize) -> u32 {
        unsafe { read_volatile((self.base + offset) as *const u32) }
    }

    /// Write a register
    fn write_reg(&self, offset: usize, value: u32) {
        unsafe { write_volatile((self.base + offset) as *mut u32, value) }
    }

    /// Set pin function
    pub fn set_function(&self, pin: u32, function: PinFunction) {
        if pin >= 58 {
            return;
        }

        let reg_offset = regs::GPFSEL0 + ((pin / 10) * 4) as usize;
        let shift = (pin % 10) * 3;

        let mut val = self.read_reg(reg_offset);
        val &= !(0b111 << shift);
        val |= (function as u32) << shift;
        self.write_reg(reg_offset, val);
    }

    /// Get pin function
    pub fn get_function(&self, pin: u32) -> PinFunction {
        if pin >= 58 {
            return PinFunction::Input;
        }

        let reg_offset = regs::GPFSEL0 + ((pin / 10) * 4) as usize;
        let shift = (pin % 10) * 3;

        let val = self.read_reg(reg_offset);
        let func = ((val >> shift) & 0b111) as u8;

        match func {
            0b000 => PinFunction::Input,
            0b001 => PinFunction::Output,
            0b100 => PinFunction::Alt0,
            0b101 => PinFunction::Alt1,
            0b110 => PinFunction::Alt2,
            0b111 => PinFunction::Alt3,
            0b011 => PinFunction::Alt4,
            0b010 => PinFunction::Alt5,
            _ => PinFunction::Input,
        }
    }

    /// Set pin output high
    pub fn set_high(&self, pin: u32) {
        if pin >= 58 {
            return;
        }

        let reg_offset = if pin < 32 { regs::GPSET0 } else { regs::GPSET1 };
        let bit = pin % 32;

        self.write_reg(reg_offset, 1 << bit);
    }

    /// Set pin output low
    pub fn set_low(&self, pin: u32) {
        if pin >= 58 {
            return;
        }

        let reg_offset = if pin < 32 { regs::GPCLR0 } else { regs::GPCLR1 };
        let bit = pin % 32;

        self.write_reg(reg_offset, 1 << bit);
    }

    /// Read pin level
    pub fn read(&self, pin: u32) -> bool {
        if pin >= 58 {
            return false;
        }

        let reg_offset = if pin < 32 { regs::GPLEV0 } else { regs::GPLEV1 };
        let bit = pin % 32;

        (self.read_reg(reg_offset) >> bit) & 1 != 0
    }

    /// Set pull-up/down mode (BCM2711+ style)
    pub fn set_pull(&self, pin: u32, mode: PullMode) {
        if pin >= 58 {
            return;
        }

        if self.is_bcm2711 {
            // BCM2711 uses different registers
            let reg_offset = regs::GPIO_PUP_PDN_CNTRL0 + ((pin / 16) * 4) as usize;
            let shift = (pin % 16) * 2;

            let mut val = self.read_reg(reg_offset);
            val &= !(0b11 << shift);
            val |= (mode as u32) << shift;
            self.write_reg(reg_offset, val);
        } else {
            // BCM2837 style (older Pis)
            self.write_reg(regs::GPPUD, mode as u32);

            // Wait 150 cycles
            for _ in 0..150 {
                core::hint::spin_loop();
            }

            let clk_offset = if pin < 32 { regs::GPPUDCLK0 } else { regs::GPPUDCLK1 };
            let bit = pin % 32;
            self.write_reg(clk_offset, 1 << bit);

            // Wait 150 cycles
            for _ in 0..150 {
                core::hint::spin_loop();
            }

            self.write_reg(regs::GPPUD, 0);
            self.write_reg(clk_offset, 0);
        }
    }

    /// Enable rising edge detect
    pub fn enable_rising_edge(&self, pin: u32) {
        if pin >= 58 {
            return;
        }

        let reg_offset = if pin < 32 { regs::GPREN0 } else { regs::GPREN1 };
        let bit = pin % 32;

        let val = self.read_reg(reg_offset) | (1 << bit);
        self.write_reg(reg_offset, val);
    }

    /// Enable falling edge detect
    pub fn enable_falling_edge(&self, pin: u32) {
        if pin >= 58 {
            return;
        }

        let reg_offset = if pin < 32 { regs::GPFEN0 } else { regs::GPFEN1 };
        let bit = pin % 32;

        let val = self.read_reg(reg_offset) | (1 << bit);
        self.write_reg(reg_offset, val);
    }

    /// Check and clear event detect status
    pub fn clear_event(&self, pin: u32) -> bool {
        if pin >= 58 {
            return false;
        }

        let reg_offset = if pin < 32 { regs::GPEDS0 } else { regs::GPEDS1 };
        let bit = pin % 32;

        let had_event = (self.read_reg(reg_offset) >> bit) & 1 != 0;

        if had_event {
            // Write 1 to clear
            self.write_reg(reg_offset, 1 << bit);
        }

        had_event
    }
}

/// Initialize GPIO for Raspberry Pi 4
pub fn init_bcm2711() {
    *GPIO.lock() = Some(Gpio::new(base::BCM2711, true));
}

/// Initialize GPIO for Raspberry Pi 3
pub fn init_bcm2837() {
    *GPIO.lock() = Some(Gpio::new(base::BCM2837, false));
}

/// Set pin as output
pub fn set_output(pin: u32) {
    if let Some(ref gpio) = *GPIO.lock() {
        gpio.set_function(pin, PinFunction::Output);
    }
}

/// Set pin as input
pub fn set_input(pin: u32) {
    if let Some(ref gpio) = *GPIO.lock() {
        gpio.set_function(pin, PinFunction::Input);
    }
}

/// Write to output pin
pub fn write(pin: u32, high: bool) {
    if let Some(ref gpio) = *GPIO.lock() {
        if high {
            gpio.set_high(pin);
        } else {
            gpio.set_low(pin);
        }
    }
}

/// Read from input pin
pub fn read(pin: u32) -> bool {
    if let Some(ref gpio) = *GPIO.lock() {
        gpio.read(pin)
    } else {
        false
    }
}

/// Toggle output pin
pub fn toggle(pin: u32) {
    let current = read(pin);
    write(pin, !current);
}

/// Set pull-up/down mode
pub fn set_pull(pin: u32, mode: PullMode) {
    if let Some(ref gpio) = *GPIO.lock() {
        gpio.set_pull(pin, mode);
    }
}

/// Check if GPIO is initialized
pub fn is_ready() -> bool {
    GPIO.lock().is_some()
}
