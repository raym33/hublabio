//! Device Drivers
//!
//! Hardware drivers for various platforms.

pub mod dtb;
pub mod framebuffer;
pub mod gpio;
pub mod input;
pub mod power;
pub mod sdmmc;
pub mod uart;
pub mod usb;

/// Initialize all drivers
pub fn init() {
    crate::kprintln!("  Initializing drivers...");
    usb::init();
    power::init();
    input::init();
    sdmmc::init();
}

/// Driver trait that all drivers implement
pub trait Driver: Send + Sync {
    /// Get driver name
    fn name(&self) -> &'static str;

    /// Initialize the driver
    fn init(&mut self) -> Result<(), DriverError>;

    /// Probe for hardware
    fn probe(&mut self) -> bool;
}

/// Driver errors
#[derive(Debug)]
pub enum DriverError {
    /// Device not found
    NotFound,
    /// Device busy
    Busy,
    /// Invalid parameter
    InvalidParam,
    /// I/O error
    IoError,
    /// Not supported
    NotSupported,
    /// Already initialized
    AlreadyInitialized,
}
