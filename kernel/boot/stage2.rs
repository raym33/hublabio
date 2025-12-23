//! =============================================================================
//! HUBLABIO BOOTLOADER - STAGE 2
//! =============================================================================
//! Rust-based early initialization after assembly bootstrap.
//! Sets up memory, initializes hardware, and starts the kernel.
//! =============================================================================

#![no_std]
#![no_main]
#![feature(naked_functions)]
#![feature(asm_const)]

use core::panic::PanicInfo;

mod uart;
mod memory;
mod dtb;

/// Kernel entry point address
const KERNEL_BASE: usize = 0xFFFF_0000_0008_0000;

/// Boot information passed to kernel
#[repr(C)]
pub struct BootInfo {
    /// Magic number for validation
    pub magic: u64,
    /// DTB (Device Tree Blob) address
    pub dtb_addr: usize,
    /// DTB size
    pub dtb_size: usize,
    /// Total RAM available
    pub total_ram: usize,
    /// Kernel command line
    pub cmdline: [u8; 256],
    /// Framebuffer address (if available)
    pub fb_addr: usize,
    /// Framebuffer width
    pub fb_width: u32,
    /// Framebuffer height
    pub fb_height: u32,
    /// Framebuffer pitch (bytes per line)
    pub fb_pitch: u32,
    /// AI model location (if preloaded)
    pub ai_model_addr: usize,
    /// AI model size
    pub ai_model_size: usize,
}

impl BootInfo {
    const MAGIC: u64 = 0x48554C_4249_4F5F31; // "HULBIO_1"

    pub fn new() -> Self {
        Self {
            magic: Self::MAGIC,
            dtb_addr: 0,
            dtb_size: 0,
            total_ram: 0,
            cmdline: [0; 256],
            fb_addr: 0,
            fb_width: 0,
            fb_height: 0,
            fb_pitch: 0,
            ai_model_addr: 0,
            ai_model_size: 0,
        }
    }
}

/// Stage 2 main entry point (called from assembly)
#[no_mangle]
pub extern "C" fn stage2_main() -> ! {
    // Initialize early UART for debug output
    uart::init();
    print_banner();

    log!("[BOOT] HubLab IO Stage 2 starting...");

    // Parse device tree
    let dtb_addr = unsafe { core::ptr::read_volatile(0x4000_0000 as *const usize) };
    log!("[BOOT] DTB at 0x{:016x}", dtb_addr);

    // Initialize memory management
    log!("[BOOT] Initializing memory...");
    let total_ram = memory::init(dtb_addr);
    log!("[BOOT] Total RAM: {} MB", total_ram / (1024 * 1024));

    // Initialize framebuffer (if available)
    log!("[BOOT] Checking framebuffer...");
    let (fb_addr, fb_width, fb_height, fb_pitch) = init_framebuffer();
    if fb_addr != 0 {
        log!("[BOOT] Framebuffer: {}x{} at 0x{:016x}", fb_width, fb_height, fb_addr);
        show_boot_logo(fb_addr, fb_width, fb_height, fb_pitch);
    }

    // Check for preloaded AI model
    log!("[BOOT] Checking for AI model...");
    let (ai_addr, ai_size) = find_ai_model();
    if ai_size > 0 {
        log!("[BOOT] AI model found: {} MB at 0x{:016x}", ai_size / (1024 * 1024), ai_addr);
    }

    // Prepare boot info for kernel
    let mut boot_info = BootInfo::new();
    boot_info.dtb_addr = dtb_addr;
    boot_info.dtb_size = dtb::get_size(dtb_addr);
    boot_info.total_ram = total_ram;
    boot_info.fb_addr = fb_addr;
    boot_info.fb_width = fb_width;
    boot_info.fb_height = fb_height;
    boot_info.fb_pitch = fb_pitch;
    boot_info.ai_model_addr = ai_addr;
    boot_info.ai_model_size = ai_size;

    // Copy command line
    parse_cmdline(&mut boot_info.cmdline);

    // Load and jump to kernel
    log!("[BOOT] Loading kernel...");
    let kernel_entry = load_kernel();

    log!("[BOOT] Jumping to kernel at 0x{:016x}...", kernel_entry);
    log!("================================================");

    // Jump to kernel with boot info
    unsafe {
        let kernel_main: extern "C" fn(*const BootInfo) -> ! =
            core::mem::transmute(kernel_entry);
        kernel_main(&boot_info);
    }
}

/// Print boot banner
fn print_banner() {
    log!("");
    log!("================================================");
    log!("  _   _       _     _           _       ___ ___  ");
    log!(" | | | |_   _| |__ | |     __ _| |__   |_ _/ _ \\ ");
    log!(" | |_| | | | | '_ \\| |    / _` | '_ \\   | | | | |");
    log!(" |  _  | |_| | |_) | |___| (_| | |_) |  | | |_| |");
    log!(" |_| |_|\\__,_|_.__/|______\\__,_|_.__/  |___\\___/ ");
    log!("");
    log!("  AI-Native Operating System v0.1.0");
    log!("================================================");
    log!("");
}

/// Initialize framebuffer via mailbox
fn init_framebuffer() -> (usize, u32, u32, u32) {
    // Raspberry Pi mailbox interface for framebuffer
    #[cfg(target_arch = "aarch64")]
    {
        // Request 1920x1080 framebuffer (or best available)
        let mailbox = 0x3F00_B880 as *mut u32; // BCM2837 mailbox

        // Simplified - real implementation would use proper mailbox protocol
        // For now, return zeros to indicate no framebuffer
        (0, 0, 0, 0)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        (0, 0, 0, 0)
    }
}

/// Show boot logo on framebuffer
fn show_boot_logo(fb: usize, width: u32, height: u32, pitch: u32) {
    // Simple logo - just a gradient for now
    let fb_ptr = fb as *mut u32;

    for y in 0..height {
        for x in 0..width {
            let offset = (y * pitch / 4 + x) as isize;

            // Purple gradient (HubLab IO brand color)
            let r = (x * 128 / width) as u8 + 50;
            let g = 0u8;
            let b = (y * 200 / height) as u8 + 55;

            let pixel = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);

            unsafe {
                core::ptr::write_volatile(fb_ptr.offset(offset), pixel);
            }
        }
    }
}

/// Find preloaded AI model in memory
fn find_ai_model() -> (usize, usize) {
    // Look for GGUF magic number in known locations
    const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

    let search_locations = [
        0x1000_0000usize, // 256MB
        0x2000_0000usize, // 512MB
        0x4000_0000usize, // 1GB
    ];

    for addr in search_locations {
        let magic = unsafe { core::ptr::read_volatile(addr as *const u32) };
        if magic == GGUF_MAGIC {
            // Read model size from header
            let size = unsafe { core::ptr::read_volatile((addr + 8) as *const u64) };
            return (addr, size as usize);
        }
    }

    (0, 0)
}

/// Parse kernel command line
fn parse_cmdline(buffer: &mut [u8; 256]) {
    // Default command line
    let default = b"console=ttyS0,115200 ai.model=auto";
    let len = default.len().min(255);
    buffer[..len].copy_from_slice(&default[..len]);
}

/// Load kernel from storage
fn load_kernel() -> usize {
    // For now, assume kernel is at a fixed location
    // Real implementation would load from SD card or network
    KERNEL_BASE
}

/// Exception handler (called from assembly)
#[no_mangle]
pub extern "C" fn handle_exception(context: *const u64) {
    let esr = unsafe { core::arch::asm!("mrs {}, esr_el1", out(reg) -> u64) };
    let far = unsafe { core::arch::asm!("mrs {}, far_el1", out(reg) -> u64) };

    log!("[EXCEPTION] ESR: 0x{:016x}, FAR: 0x{:016x}", esr, far);
    loop {
        unsafe { core::arch::asm!("wfe") };
    }
}

/// Panic handler
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    log!("[PANIC] {}", info);
    loop {
        unsafe { core::arch::asm!("wfe") };
    }
}

/// Simple logging macro
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        uart::print(format_args!($($arg)*));
        uart::print(format_args!("\n"));
    };
}

/// UART module for early debug output
mod uart {
    use core::fmt::{self, Write};

    const UART0_BASE: usize = 0x3F20_1000; // BCM2837 UART0

    struct Uart;

    impl Write for Uart {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            for byte in s.bytes() {
                put_char(byte);
            }
            Ok(())
        }
    }

    pub fn init() {
        // UART should already be initialized by firmware
        // Just ensure it's in a known state
    }

    pub fn print(args: fmt::Arguments) {
        let mut uart = Uart;
        let _ = uart.write_fmt(args);
    }

    fn put_char(c: u8) {
        unsafe {
            let uart = UART0_BASE as *mut u32;
            // Wait for TX FIFO not full
            while (core::ptr::read_volatile(uart.offset(5)) & 0x20) == 0 {}
            // Write character
            core::ptr::write_volatile(uart, c as u32);
        }
    }
}

/// Memory initialization module
mod memory {
    pub fn init(dtb_addr: usize) -> usize {
        // Parse DTB for memory size
        // Simplified - real implementation would properly parse DTB

        // Default to 1GB if we can't determine
        1024 * 1024 * 1024
    }
}

/// Device Tree Blob parsing
mod dtb {
    const FDT_MAGIC: u32 = 0xd00dfeed;

    pub fn get_size(addr: usize) -> usize {
        let header = addr as *const u32;
        let magic = unsafe { u32::from_be(core::ptr::read_volatile(header)) };

        if magic != FDT_MAGIC {
            return 0;
        }

        let total_size = unsafe {
            u32::from_be(core::ptr::read_volatile(header.offset(1)))
        };

        total_size as usize
    }
}
