//! Device Nodes (/dev)
//!
//! Character and block device management.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::{Mutex, RwLock};
use core::sync::atomic::{AtomicU32, Ordering};

/// Device type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    Character,
    Block,
}

/// Device number (major, minor)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceNumber {
    pub major: u16,
    pub minor: u16,
}

impl DeviceNumber {
    pub fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    pub fn to_dev_t(&self) -> u32 {
        ((self.major as u32) << 16) | (self.minor as u32)
    }

    pub fn from_dev_t(dev: u32) -> Self {
        Self {
            major: (dev >> 16) as u16,
            minor: dev as u16,
        }
    }
}

/// Standard major device numbers
pub mod major {
    pub const MEM: u16 = 1;      // Memory devices
    pub const TTY: u16 = 4;      // TTY devices
    pub const CONSOLE: u16 = 5;  // Console
    pub const LP: u16 = 6;       // Printer
    pub const LOOP: u16 = 7;     // Loopback
    pub const SCSI_DISK: u16 = 8; // SCSI disks
    pub const PTY_MASTER: u16 = 128; // PTY masters
    pub const PTY_SLAVE: u16 = 136;  // PTY slaves
    pub const USB: u16 = 180;    // USB devices
    pub const MISC: u16 = 10;    // Misc devices
    pub const INPUT: u16 = 13;   // Input devices
    pub const SOUND: u16 = 14;   // Sound devices
    pub const FB: u16 = 29;      // Framebuffer
    pub const MTD: u16 = 90;     // MTD devices
    pub const MMC: u16 = 179;    // MMC/SD cards
}

/// Device operations trait
pub trait DeviceOps: Send + Sync {
    /// Open device
    fn open(&self, minor: u16, flags: u32) -> Result<(), DeviceError>;

    /// Close device
    fn close(&self, minor: u16) -> Result<(), DeviceError>;

    /// Read from device
    fn read(&self, minor: u16, buf: &mut [u8], offset: usize) -> Result<usize, DeviceError>;

    /// Write to device
    fn write(&self, minor: u16, buf: &[u8], offset: usize) -> Result<usize, DeviceError>;

    /// Device-specific control
    fn ioctl(&self, minor: u16, cmd: u32, arg: usize) -> Result<usize, DeviceError>;

    /// Poll for events
    fn poll(&self, minor: u16) -> PollEvents;
}

/// Poll events
#[derive(Clone, Copy, Debug, Default)]
pub struct PollEvents {
    pub readable: bool,
    pub writable: bool,
    pub error: bool,
    pub hangup: bool,
}

/// Device errors
#[derive(Clone, Debug)]
pub enum DeviceError {
    NotFound,
    Busy,
    NoPermission,
    InvalidArg,
    IoError,
    NotSupported,
    WouldBlock,
}

/// Device node
pub struct DeviceNode {
    pub name: String,
    pub dev_type: DeviceType,
    pub number: DeviceNumber,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
}

/// Registered devices
static CHAR_DEVICES: RwLock<BTreeMap<u16, Arc<dyn DeviceOps>>> = RwLock::new(BTreeMap::new());
static BLOCK_DEVICES: RwLock<BTreeMap<u16, Arc<dyn DeviceOps>>> = RwLock::new(BTreeMap::new());

/// Device nodes in /dev
static DEV_NODES: RwLock<BTreeMap<String, DeviceNode>> = RwLock::new(BTreeMap::new());

/// Register character device driver
pub fn register_chrdev(major: u16, ops: Arc<dyn DeviceOps>) -> Result<(), DeviceError> {
    let mut devices = CHAR_DEVICES.write();
    if devices.contains_key(&major) {
        return Err(DeviceError::Busy);
    }
    devices.insert(major, ops);
    crate::kdebug!("Registered char device major {}", major);
    Ok(())
}

/// Register block device driver
pub fn register_blkdev(major: u16, ops: Arc<dyn DeviceOps>) -> Result<(), DeviceError> {
    let mut devices = BLOCK_DEVICES.write();
    if devices.contains_key(&major) {
        return Err(DeviceError::Busy);
    }
    devices.insert(major, ops);
    crate::kdebug!("Registered block device major {}", major);
    Ok(())
}

/// Unregister character device
pub fn unregister_chrdev(major: u16) {
    CHAR_DEVICES.write().remove(&major);
}

/// Unregister block device
pub fn unregister_blkdev(major: u16) {
    BLOCK_DEVICES.write().remove(&major);
}

/// Create device node
pub fn mknod(path: &str, dev_type: DeviceType, major: u16, minor: u16, mode: u32) -> Result<(), DeviceError> {
    let node = DeviceNode {
        name: String::from(path),
        dev_type,
        number: DeviceNumber::new(major, minor),
        mode,
        uid: 0,
        gid: 0,
    };

    DEV_NODES.write().insert(String::from(path), node);
    Ok(())
}

/// Remove device node
pub fn unlink(path: &str) -> Result<(), DeviceError> {
    DEV_NODES.write().remove(path).ok_or(DeviceError::NotFound)?;
    Ok(())
}

/// Get device node
pub fn get_node(path: &str) -> Option<DeviceNode> {
    DEV_NODES.read().get(path).map(|n| DeviceNode {
        name: n.name.clone(),
        dev_type: n.dev_type,
        number: n.number,
        mode: n.mode,
        uid: n.uid,
        gid: n.gid,
    })
}

/// List all device nodes
pub fn list_nodes() -> Vec<String> {
    DEV_NODES.read().keys().cloned().collect()
}

/// Open device by path
pub fn open(path: &str, flags: u32) -> Result<DeviceHandle, DeviceError> {
    let node = get_node(path).ok_or(DeviceError::NotFound)?;

    let ops = match node.dev_type {
        DeviceType::Character => {
            CHAR_DEVICES.read().get(&node.number.major).cloned()
        }
        DeviceType::Block => {
            BLOCK_DEVICES.read().get(&node.number.major).cloned()
        }
    };

    let ops = ops.ok_or(DeviceError::NotFound)?;
    ops.open(node.number.minor, flags)?;

    Ok(DeviceHandle {
        dev_type: node.dev_type,
        number: node.number,
        ops,
        offset: 0,
    })
}

/// Device handle for I/O
pub struct DeviceHandle {
    dev_type: DeviceType,
    number: DeviceNumber,
    ops: Arc<dyn DeviceOps>,
    offset: usize,
}

impl DeviceHandle {
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, DeviceError> {
        let count = self.ops.read(self.number.minor, buf, self.offset)?;
        self.offset += count;
        Ok(count)
    }

    pub fn write(&mut self, buf: &[u8]) -> Result<usize, DeviceError> {
        let count = self.ops.write(self.number.minor, buf, self.offset)?;
        self.offset += count;
        Ok(count)
    }

    pub fn ioctl(&self, cmd: u32, arg: usize) -> Result<usize, DeviceError> {
        self.ops.ioctl(self.number.minor, cmd, arg)
    }

    pub fn poll(&self) -> PollEvents {
        self.ops.poll(self.number.minor)
    }

    pub fn seek(&mut self, offset: usize) {
        self.offset = offset;
    }
}

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        let _ = self.ops.close(self.number.minor);
    }
}

/// Null device (/dev/null)
struct NullDevice;

impl DeviceOps for NullDevice {
    fn open(&self, _minor: u16, _flags: u32) -> Result<(), DeviceError> {
        Ok(())
    }

    fn close(&self, _minor: u16) -> Result<(), DeviceError> {
        Ok(())
    }

    fn read(&self, _minor: u16, _buf: &mut [u8], _offset: usize) -> Result<usize, DeviceError> {
        Ok(0) // EOF
    }

    fn write(&self, _minor: u16, buf: &[u8], _offset: usize) -> Result<usize, DeviceError> {
        Ok(buf.len()) // Discard
    }

    fn ioctl(&self, _minor: u16, _cmd: u32, _arg: usize) -> Result<usize, DeviceError> {
        Ok(0)
    }

    fn poll(&self, _minor: u16) -> PollEvents {
        PollEvents {
            readable: true,
            writable: true,
            error: false,
            hangup: false,
        }
    }
}

/// Zero device (/dev/zero)
struct ZeroDevice;

impl DeviceOps for ZeroDevice {
    fn open(&self, _minor: u16, _flags: u32) -> Result<(), DeviceError> {
        Ok(())
    }

    fn close(&self, _minor: u16) -> Result<(), DeviceError> {
        Ok(())
    }

    fn read(&self, _minor: u16, buf: &mut [u8], _offset: usize) -> Result<usize, DeviceError> {
        buf.fill(0);
        Ok(buf.len())
    }

    fn write(&self, _minor: u16, buf: &[u8], _offset: usize) -> Result<usize, DeviceError> {
        Ok(buf.len())
    }

    fn ioctl(&self, _minor: u16, _cmd: u32, _arg: usize) -> Result<usize, DeviceError> {
        Ok(0)
    }

    fn poll(&self, _minor: u16) -> PollEvents {
        PollEvents {
            readable: true,
            writable: true,
            error: false,
            hangup: false,
        }
    }
}

/// Random device (/dev/random, /dev/urandom)
struct RandomDevice;

impl DeviceOps for RandomDevice {
    fn open(&self, _minor: u16, _flags: u32) -> Result<(), DeviceError> {
        Ok(())
    }

    fn close(&self, _minor: u16) -> Result<(), DeviceError> {
        Ok(())
    }

    fn read(&self, _minor: u16, buf: &mut [u8], _offset: usize) -> Result<usize, DeviceError> {
        // Simple PRNG (would use real entropy in production)
        static SEED: AtomicU32 = AtomicU32::new(0x12345678);

        for byte in buf.iter_mut() {
            let mut s = SEED.load(Ordering::Relaxed);
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            SEED.store(s, Ordering::Relaxed);
            *byte = s as u8;
        }

        Ok(buf.len())
    }

    fn write(&self, _minor: u16, buf: &[u8], _offset: usize) -> Result<usize, DeviceError> {
        // Could add to entropy pool
        Ok(buf.len())
    }

    fn ioctl(&self, _minor: u16, _cmd: u32, _arg: usize) -> Result<usize, DeviceError> {
        Ok(0)
    }

    fn poll(&self, _minor: u16) -> PollEvents {
        PollEvents {
            readable: true,
            writable: true,
            error: false,
            hangup: false,
        }
    }
}

/// Initialize /dev filesystem
pub fn init() {
    // Register memory devices
    let mem_ops = Arc::new(NullDevice);
    let _ = register_chrdev(major::MEM, mem_ops);

    // Create standard device nodes
    let _ = mknod("/dev/null", DeviceType::Character, major::MEM, 3, 0o666);
    let _ = mknod("/dev/zero", DeviceType::Character, major::MEM, 5, 0o666);
    let _ = mknod("/dev/random", DeviceType::Character, major::MEM, 8, 0o666);
    let _ = mknod("/dev/urandom", DeviceType::Character, major::MEM, 9, 0o666);

    // Console devices
    let _ = mknod("/dev/tty", DeviceType::Character, major::TTY, 0, 0o666);
    let _ = mknod("/dev/console", DeviceType::Character, major::CONSOLE, 1, 0o600);
    let _ = mknod("/dev/tty0", DeviceType::Character, major::TTY, 0, 0o600);

    crate::kprintln!("  Device subsystem initialized");
}
