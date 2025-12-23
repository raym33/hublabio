//! Boot Module
//!
//! Handles early hardware initialization and boot sequence.

use crate::{BootInfo, MemoryMap, MemoryRegion, MemoryKind, FramebufferInfo};

pub mod platform;
pub mod cmdline;

/// Boot stages
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootStage {
    /// Early boot (no memory allocator)
    Early,
    /// Memory initialized
    Memory,
    /// Drivers initialized
    Drivers,
    /// Services starting
    Services,
    /// Fully booted
    Complete,
}

/// Current boot stage
static mut BOOT_STAGE: BootStage = BootStage::Early;

/// Get current boot stage
pub fn stage() -> BootStage {
    unsafe { BOOT_STAGE }
}

/// Advance to next boot stage
pub fn advance_stage(new_stage: BootStage) {
    unsafe {
        BOOT_STAGE = new_stage;
    }
    crate::kdebug!("Boot stage: {:?}", new_stage);
}

/// Supported platforms
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    /// QEMU virt machine
    QemuVirt,
    /// Raspberry Pi 3 (BCM2837)
    RaspberryPi3,
    /// Raspberry Pi 4 (BCM2711)
    RaspberryPi4,
    /// Raspberry Pi 5 (BCM2712)
    RaspberryPi5,
    /// Raspberry Pi Zero 2 W
    RaspberryPiZero2W,
    /// Generic ARM64
    GenericArm64,
    /// Unknown
    Unknown,
}

/// Detected platform
static mut DETECTED_PLATFORM: Platform = Platform::Unknown;

/// Get detected platform
pub fn platform() -> Platform {
    unsafe { DETECTED_PLATFORM }
}

/// Detect the hardware platform
pub fn detect_platform(dtb_addr: usize) -> Platform {
    let platform = if dtb_addr == 0 {
        // No DTB, assume QEMU
        Platform::QemuVirt
    } else {
        // Parse DTB to identify platform
        detect_from_dtb(dtb_addr)
    };

    unsafe {
        DETECTED_PLATFORM = platform;
    }

    platform
}

/// Detect platform from Device Tree
fn detect_from_dtb(dtb_addr: usize) -> Platform {
    // Try to find compatible string
    if let Some((_, _)) = crate::drivers::dtb::find_compatible(dtb_addr, "brcm,bcm2712") {
        return Platform::RaspberryPi5;
    }
    if let Some((_, _)) = crate::drivers::dtb::find_compatible(dtb_addr, "brcm,bcm2711") {
        return Platform::RaspberryPi4;
    }
    if let Some((_, _)) = crate::drivers::dtb::find_compatible(dtb_addr, "brcm,bcm2837") {
        // Could be Pi 3 or Zero 2 W
        return Platform::RaspberryPi3;
    }

    Platform::GenericArm64
}

/// Initialize platform-specific hardware
pub fn init_platform(platform: Platform) {
    match platform {
        Platform::QemuVirt => {
            crate::kprintln!("  Platform: QEMU virt");
            crate::drivers::uart::init();
        }
        Platform::RaspberryPi3 => {
            crate::kprintln!("  Platform: Raspberry Pi 3");
            crate::drivers::gpio::init_bcm2837();
            init_rpi_uart(crate::drivers::uart::base::BCM2837_UART0);
        }
        Platform::RaspberryPi4 => {
            crate::kprintln!("  Platform: Raspberry Pi 4");
            crate::drivers::gpio::init_bcm2711();
            init_rpi_uart(crate::drivers::uart::base::BCM2711_UART0);
        }
        Platform::RaspberryPi5 => {
            crate::kprintln!("  Platform: Raspberry Pi 5");
            // Pi 5 has different initialization
            init_rpi5();
        }
        Platform::RaspberryPiZero2W => {
            crate::kprintln!("  Platform: Raspberry Pi Zero 2 W");
            crate::drivers::gpio::init_bcm2837();
            init_rpi_uart(crate::drivers::uart::base::BCM2837_UART0);
        }
        Platform::GenericArm64 => {
            crate::kprintln!("  Platform: Generic ARM64");
            crate::drivers::uart::init();
        }
        Platform::Unknown => {
            crate::kprintln!("  Platform: Unknown (using defaults)");
            crate::drivers::uart::init();
        }
    }
}

/// Initialize Raspberry Pi UART
fn init_rpi_uart(base: usize) {
    // Default Pi UART clock is 48MHz, baud 115200
    crate::drivers::uart::init_at(base, 115200, 48_000_000);
}

/// Initialize Raspberry Pi 5
fn init_rpi5() {
    // Pi 5 uses RP1 chip for peripherals
    // TODO: Implement RP1 initialization
    crate::drivers::uart::init_at(
        crate::drivers::uart::base::BCM2712_UART0,
        115200,
        48_000_000,
    );
}

/// Memory sizes for different Pi models
pub fn get_memory_size(platform: Platform) -> usize {
    match platform {
        Platform::RaspberryPi5 => 8 * 1024 * 1024 * 1024, // Up to 8GB
        Platform::RaspberryPi4 => 4 * 1024 * 1024 * 1024, // Up to 8GB, assume 4GB
        Platform::RaspberryPi3 => 1024 * 1024 * 1024,     // 1GB
        Platform::RaspberryPiZero2W => 512 * 1024 * 1024, // 512MB
        Platform::QemuVirt => 128 * 1024 * 1024,          // QEMU default
        _ => 256 * 1024 * 1024,                            // Conservative default
    }
}

/// Create a simple boot info for early boot
pub fn create_early_boot_info(dtb_addr: usize) -> BootInfo {
    static EMPTY_REGIONS: [MemoryRegion; 0] = [];
    static EMPTY_MAP: MemoryMap = MemoryMap {
        entries: &EMPTY_REGIONS,
    };

    BootInfo {
        magic: 0x4855424C_41423130, // "HUBLAB10"
        memory_map: EMPTY_MAP,
        framebuffer: None,
        dtb_address: dtb_addr,
        cmdline: "",
        ai_model_addr: None,
        ai_model_size: 0,
    }
}

/// Early boot sequence (before memory allocator)
pub fn early_init(dtb_addr: usize) {
    // Detect platform
    let platform = detect_platform(dtb_addr);

    // Initialize platform hardware
    init_platform(platform);

    advance_stage(BootStage::Early);
}

/// Boot banner
pub fn print_banner() {
    crate::kprintln!();
    crate::kprintln!("  _   _       _     _           _      _____ ____  ");
    crate::kprintln!(" | | | |_   _| |__ | |     __ _| |__  |_   _/ ___| ");
    crate::kprintln!(" | |_| | | | | '_ \\| |    / _` | '_ \\   | || |  _  ");
    crate::kprintln!(" |  _  | |_| | |_) | |___| (_| | |_) |  | || |_| | ");
    crate::kprintln!(" |_| |_|\\__,_|_.__/|_____/\\__,_|_.__/   |_| \\____| ");
    crate::kprintln!();
    crate::kprintln!("  AI-Native Operating System v{}", crate::VERSION);
    crate::kprintln!("  Platform: {:?}", platform());
    crate::kprintln!();
}

/// Hardware capabilities
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Number of CPU cores
    pub cores: usize,
    /// Has FPU/SIMD
    pub has_fpu: bool,
    /// Has crypto extensions
    pub has_crypto: bool,
    /// Has NPU/AI accelerator
    pub has_npu: bool,
    /// Total RAM in bytes
    pub ram_size: usize,
    /// Has GPU
    pub has_gpu: bool,
    /// Has network
    pub has_network: bool,
    /// Has WiFi
    pub has_wifi: bool,
    /// Has Bluetooth
    pub has_bluetooth: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            cores: 1,
            has_fpu: true,
            has_crypto: false,
            has_npu: false,
            ram_size: 256 * 1024 * 1024,
            has_gpu: false,
            has_network: false,
            has_wifi: false,
            has_bluetooth: false,
        }
    }
}

/// Detect hardware capabilities
pub fn detect_capabilities(platform: Platform) -> Capabilities {
    let mut caps = Capabilities::default();

    match platform {
        Platform::RaspberryPi5 => {
            caps.cores = 4;
            caps.has_crypto = true;
            caps.ram_size = 8 * 1024 * 1024 * 1024;
            caps.has_gpu = true;
            caps.has_network = true;
            caps.has_wifi = true;
            caps.has_bluetooth = true;
        }
        Platform::RaspberryPi4 => {
            caps.cores = 4;
            caps.has_crypto = true;
            caps.ram_size = 4 * 1024 * 1024 * 1024;
            caps.has_gpu = true;
            caps.has_network = true;
            caps.has_wifi = true;
            caps.has_bluetooth = true;
        }
        Platform::RaspberryPi3 => {
            caps.cores = 4;
            caps.ram_size = 1024 * 1024 * 1024;
            caps.has_gpu = true;
            caps.has_network = true;
            caps.has_wifi = true;
            caps.has_bluetooth = true;
        }
        Platform::RaspberryPiZero2W => {
            caps.cores = 4;
            caps.ram_size = 512 * 1024 * 1024;
            caps.has_wifi = true;
            caps.has_bluetooth = true;
        }
        Platform::QemuVirt => {
            caps.cores = 4;
            caps.ram_size = 128 * 1024 * 1024;
        }
        _ => {}
    }

    caps
}

/// Print detected capabilities
pub fn print_capabilities(caps: &Capabilities) {
    crate::kprintln!("  Hardware Capabilities:");
    crate::kprintln!("    CPU Cores: {}", caps.cores);
    crate::kprintln!("    RAM: {} MB", caps.ram_size / (1024 * 1024));
    crate::kprintln!("    FPU: {}", if caps.has_fpu { "Yes" } else { "No" });
    crate::kprintln!("    Crypto: {}", if caps.has_crypto { "Yes" } else { "No" });
    crate::kprintln!("    GPU: {}", if caps.has_gpu { "Yes" } else { "No" });
    crate::kprintln!("    Network: {}", if caps.has_network { "Yes" } else { "No" });
    crate::kprintln!("    WiFi: {}", if caps.has_wifi { "Yes" } else { "No" });
}
