//! HubLab IO Kernel
//!
//! A microkernel designed for AI-first computing on ARM64 and RISC-V devices.
//! Features an AI-enhanced scheduler, minimal attack surface, and native
//! support for neural network inference at the kernel level.

#![no_std]
#![feature(alloc_error_handler)]
#![feature(naked_functions)]
#![feature(asm_const)]

extern crate alloc;

pub mod arch;
pub mod boot;
pub mod console;
pub mod drivers;
pub mod fs;
pub mod init;
pub mod ipc;
pub mod memory;
pub mod net;
pub mod process;
pub mod scheduler;
pub mod syscall;
pub mod vfs;

use core::panic::PanicInfo;
use alloc::alloc::{GlobalAlloc, Layout};
use linked_list_allocator::LockedHeap;

/// Global kernel heap allocator
#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

/// Kernel version
pub const VERSION: &str = "0.1.0";

/// Kernel name
pub const NAME: &str = "HubLab IO";

/// Boot information passed from bootloader
#[repr(C)]
pub struct BootInfo {
    /// Magic number for validation
    pub magic: u64,
    /// Physical memory map
    pub memory_map: MemoryMap,
    /// Framebuffer info (if available)
    pub framebuffer: Option<FramebufferInfo>,
    /// Device tree blob address
    pub dtb_address: usize,
    /// Command line arguments
    pub cmdline: &'static str,
    /// AI model base address (if preloaded)
    pub ai_model_addr: Option<usize>,
    /// AI model size
    pub ai_model_size: usize,
}

/// Memory map from bootloader
#[repr(C)]
pub struct MemoryMap {
    pub entries: &'static [MemoryRegion],
}

/// Single memory region
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MemoryRegion {
    pub base: usize,
    pub size: usize,
    pub kind: MemoryKind,
}

/// Memory region type
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    Available = 0,
    Reserved = 1,
    AcpiReclaimable = 2,
    Kernel = 3,
    Framebuffer = 4,
    AiModel = 5,
}

/// Framebuffer information
#[repr(C)]
pub struct FramebufferInfo {
    pub address: usize,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u8,
}

/// Kernel entry point (called from stage2)
#[no_mangle]
pub extern "C" fn kernel_main(boot_info: &'static BootInfo) -> ! {
    // Initialize console for early output
    console::init();

    kprintln!("===========================================");
    kprintln!("  {} Kernel v{}", NAME, VERSION);
    kprintln!("  AI-Native Operating System");
    kprintln!("===========================================");
    kprintln!();

    // Validate boot info
    const BOOT_MAGIC: u64 = 0x4855424C_41423130; // "HUBLAB10" in hex
    if boot_info.magic != BOOT_MAGIC {
        panic!("Invalid boot info magic number");
    }

    // Initialize memory subsystem
    kprintln!("[BOOT] Initializing memory manager...");
    memory::init(&boot_info.memory_map);

    // Initialize heap allocator
    kprintln!("[BOOT] Setting up kernel heap...");
    unsafe {
        let heap_start = memory::KERNEL_HEAP_START;
        let heap_size = memory::KERNEL_HEAP_SIZE;
        ALLOCATOR.lock().init(heap_start as *mut u8, heap_size);
    }

    // Initialize architecture-specific features
    kprintln!("[BOOT] Initializing architecture...");
    arch::init();

    // Parse device tree
    if boot_info.dtb_address != 0 {
        kprintln!("[BOOT] Parsing device tree at 0x{:x}...", boot_info.dtb_address);
        drivers::dtb::parse(boot_info.dtb_address);
    }

    // Initialize scheduler
    kprintln!("[BOOT] Initializing AI-enhanced scheduler...");
    scheduler::init();

    // Load AI model if available
    if let Some(model_addr) = boot_info.ai_model_addr {
        kprintln!("[BOOT] Loading scheduler AI model ({} KB)...",
                 boot_info.ai_model_size / 1024);
        scheduler::load_ai_model(model_addr, boot_info.ai_model_size);
    }

    // Initialize IPC
    kprintln!("[BOOT] Setting up IPC channels...");
    ipc::init();

    // Initialize VFS
    kprintln!("[BOOT] Mounting virtual filesystem...");
    vfs::init();

    // Initialize filesystem layer
    kprintln!("[BOOT] Initializing filesystem drivers...");
    fs::init();

    // Initialize network stack
    kprintln!("[BOOT] Initializing network stack...");
    net::init();

    // Initialize framebuffer if available
    if let Some(ref fb) = boot_info.framebuffer {
        kprintln!("[BOOT] Framebuffer: {}x{} @ 0x{:x}",
                 fb.width, fb.height, fb.address);
        drivers::framebuffer::init(fb);
    }

    // Initialize syscall interface
    kprintln!("[BOOT] Setting up syscall interface...");
    syscall::init();

    kprintln!();
    kprintln!("[BOOT] Kernel initialization complete!");
    kprintln!("[BOOT] Starting init process...");
    kprintln!();

    // Start the init process (PID 1)
    process::spawn_init();

    // Enter scheduler loop (never returns)
    scheduler::run()
}

/// Panic handler
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    kprintln!();
    kprintln!("!!! KERNEL PANIC !!!");
    kprintln!("{}", info);
    kprintln!();

    // Try to dump registers and backtrace
    arch::dump_state();

    // Halt the system
    loop {
        arch::halt();
    }
}

/// Allocation error handler
#[alloc_error_handler]
fn alloc_error(layout: Layout) -> ! {
    panic!("Allocation error: {:?}", layout);
}

/// Kernel print macro
#[macro_export]
macro_rules! kprint {
    ($($arg:tt)*) => ($crate::console::_print(format_args!($($arg)*)));
}

/// Kernel println macro
#[macro_export]
macro_rules! kprintln {
    () => ($crate::kprint!("\n"));
    ($($arg:tt)*) => ($crate::kprint!("{}\n", format_args!($($arg)*)));
}

/// Debug logging macro
#[macro_export]
macro_rules! kdebug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        $crate::kprintln!("[DEBUG] {}", format_args!($($arg)*));
    };
}

/// Info logging macro
#[macro_export]
macro_rules! kinfo {
    ($($arg:tt)*) => {
        $crate::kprintln!("[INFO] {}", format_args!($($arg)*));
    };
}

/// Warning logging macro
#[macro_export]
macro_rules! kwarn {
    ($($arg:tt)*) => {
        $crate::kprintln!("[WARN] {}", format_args!($($arg)*));
    };
}

/// Error logging macro
#[macro_export]
macro_rules! kerror {
    ($($arg:tt)*) => {
        $crate::kprintln!("[ERROR] {}", format_args!($($arg)*));
    };
}
