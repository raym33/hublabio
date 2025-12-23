//! Memory management abstractions

/// Physical address
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhysAddr(pub u64);

/// Virtual address
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VirtAddr(pub u64);

impl PhysAddr {
    pub const fn new(addr: u64) -> Self {
        Self(addr)
    }

    pub const fn as_u64(&self) -> u64 {
        self.0
    }
}

impl VirtAddr {
    pub const fn new(addr: u64) -> Self {
        Self(addr)
    }

    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.0 as *const T
    }

    pub fn as_mut_ptr<T>(&self) -> *mut T {
        self.0 as *mut T
    }
}

/// Memory attributes
#[derive(Clone, Copy, Debug)]
pub struct MemoryAttributes {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
    pub cacheable: bool,
    pub device: bool,
}

impl MemoryAttributes {
    pub const NORMAL: Self = Self {
        readable: true,
        writable: true,
        executable: false,
        cacheable: true,
        device: false,
    };

    pub const DEVICE: Self = Self {
        readable: true,
        writable: true,
        executable: false,
        cacheable: false,
        device: true,
    };

    pub const CODE: Self = Self {
        readable: true,
        writable: false,
        executable: true,
        cacheable: true,
        device: false,
    };

    pub const RODATA: Self = Self {
        readable: true,
        writable: false,
        executable: false,
        cacheable: true,
        device: false,
    };
}

/// Flush TLB for address
pub fn flush_tlb(addr: VirtAddr) {
    #[cfg(feature = "arm64")]
    unsafe {
        core::arch::asm!("dsb ishst");
        core::arch::asm!("tlbi vaae1is, {}", in(reg) addr.0 >> 12);
        core::arch::asm!("dsb ish");
        core::arch::asm!("isb");
    }

    #[cfg(feature = "x86")]
    unsafe {
        core::arch::asm!("invlpg [{}]", in(reg) addr.0);
    }

    #[cfg(feature = "riscv")]
    unsafe {
        core::arch::asm!("sfence.vma {}, zero", in(reg) addr.0);
    }

    #[cfg(not(any(feature = "arm64", feature = "riscv", feature = "x86")))]
    let _ = addr;
}

/// Flush entire TLB
pub fn flush_tlb_all() {
    #[cfg(feature = "arm64")]
    unsafe {
        core::arch::asm!("dsb ishst");
        core::arch::asm!("tlbi vmalle1is");
        core::arch::asm!("dsb ish");
        core::arch::asm!("isb");
    }

    #[cfg(feature = "x86")]
    unsafe {
        let cr3: u64;
        core::arch::asm!("mov {}, cr3", out(reg) cr3);
        core::arch::asm!("mov cr3, {}", in(reg) cr3);
    }

    #[cfg(feature = "riscv")]
    unsafe {
        core::arch::asm!("sfence.vma zero, zero");
    }
}
