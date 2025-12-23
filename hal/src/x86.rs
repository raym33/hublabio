//! x86_64-specific implementations

/// Initialize x86 HAL
pub fn init() {
    // Set up IDT
    // Initialize APIC
}

/// Get current CPU ID
pub fn cpu_id() -> usize {
    // Read from APIC ID
    let cpuid: u32;
    unsafe {
        core::arch::asm!(
            "mov eax, 1",
            "cpuid",
            out("ebx") cpuid,
            out("eax") _,
            out("ecx") _,
            out("edx") _,
        );
    }
    ((cpuid >> 24) & 0xFF) as usize
}

/// I/O port operations
pub mod io {
    /// Read byte from port
    pub unsafe fn inb(port: u16) -> u8 {
        let value: u8;
        core::arch::asm!("in al, dx", out("al") value, in("dx") port);
        value
    }

    /// Write byte to port
    pub unsafe fn outb(port: u16, value: u8) {
        core::arch::asm!("out dx, al", in("dx") port, in("al") value);
    }

    /// Read word from port
    pub unsafe fn inw(port: u16) -> u16 {
        let value: u16;
        core::arch::asm!("in ax, dx", out("ax") value, in("dx") port);
        value
    }

    /// Write word to port
    pub unsafe fn outw(port: u16, value: u16) {
        core::arch::asm!("out dx, ax", in("dx") port, in("ax") value);
    }

    /// Read dword from port
    pub unsafe fn inl(port: u16) -> u32 {
        let value: u32;
        core::arch::asm!("in eax, dx", out("eax") value, in("dx") port);
        value
    }

    /// Write dword to port
    pub unsafe fn outl(port: u16, value: u32) {
        core::arch::asm!("out dx, eax", in("dx") port, in("eax") value);
    }
}

/// Serial port (COM1)
pub mod serial {
    use super::io;

    const COM1: u16 = 0x3F8;

    pub fn init() {
        unsafe {
            io::outb(COM1 + 1, 0x00); // Disable interrupts
            io::outb(COM1 + 3, 0x80); // Enable DLAB
            io::outb(COM1 + 0, 0x03); // 38400 baud
            io::outb(COM1 + 1, 0x00);
            io::outb(COM1 + 3, 0x03); // 8N1
            io::outb(COM1 + 2, 0xC7); // Enable FIFO
            io::outb(COM1 + 4, 0x0B); // Enable IRQs
        }
    }

    pub fn write(byte: u8) {
        unsafe {
            while io::inb(COM1 + 5) & 0x20 == 0 {}
            io::outb(COM1, byte);
        }
    }

    pub fn read() -> u8 {
        unsafe {
            while io::inb(COM1 + 5) & 0x01 == 0 {}
            io::inb(COM1)
        }
    }
}
