//! Filesystem Implementations
//!
//! Persistent filesystem drivers for HubLab IO.

pub mod block;
pub mod ext4;
pub mod fat32;
pub mod ramfs;

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::RwLock;

use crate::vfs::{FilePermissions, FileStat, FileType, OpenFlags, VfsError};

/// Filesystem trait
pub trait Filesystem: Send + Sync {
    /// Get filesystem type name
    fn name(&self) -> &'static str;

    /// Check if filesystem is read-only
    fn is_readonly(&self) -> bool;

    /// Get total size in bytes
    fn total_size(&self) -> u64;

    /// Get free space in bytes
    fn free_space(&self) -> u64;

    /// Open a file
    fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, VfsError>;

    /// Read from file
    fn read(&self, handle: &FileHandle, buf: &mut [u8]) -> Result<usize, VfsError>;

    /// Write to file
    fn write(&self, handle: &FileHandle, buf: &[u8]) -> Result<usize, VfsError>;

    /// Close file
    fn close(&self, handle: FileHandle) -> Result<(), VfsError>;

    /// Get file metadata
    fn stat(&self, path: &str) -> Result<FileStat, VfsError>;

    /// List directory
    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError>;

    /// Create directory
    fn mkdir(&self, path: &str) -> Result<(), VfsError>;

    /// Remove file
    fn unlink(&self, path: &str) -> Result<(), VfsError>;

    /// Remove directory
    fn rmdir(&self, path: &str) -> Result<(), VfsError>;

    /// Rename file
    fn rename(&self, from: &str, to: &str) -> Result<(), VfsError>;

    /// Sync filesystem
    fn sync(&self) -> Result<(), VfsError>;
}

/// File handle
#[derive(Clone)]
pub struct FileHandle {
    pub inode: u64,
    pub position: u64,
    pub flags: OpenFlags,
}

/// Directory entry
#[derive(Clone, Debug)]
pub struct DirEntry {
    pub name: String,
    pub file_type: FileType,
    pub inode: u64,
}

/// Block device trait
pub trait BlockDevice: Send + Sync {
    /// Get block size
    fn block_size(&self) -> usize;

    /// Get total blocks
    fn total_blocks(&self) -> u64;

    /// Read a block
    fn read_block(&self, block: u64, buf: &mut [u8]) -> Result<(), VfsError>;

    /// Write a block
    fn write_block(&self, block: u64, buf: &[u8]) -> Result<(), VfsError>;

    /// Sync device
    fn sync(&self) -> Result<(), VfsError>;
}

/// Mounted filesystem
pub struct MountedFs {
    pub fs: Arc<dyn Filesystem>,
    pub mount_point: String,
    pub device: String,
}

/// Global mounted filesystems
static MOUNTED: RwLock<Vec<MountedFs>> = RwLock::new(Vec::new());

/// Mount a filesystem
pub fn mount(device: &str, mount_point: &str, fs_type: &str) -> Result<(), VfsError> {
    crate::kinfo!("Mounting {} on {} ({})", device, mount_point, fs_type);

    let fs: Arc<dyn Filesystem> = match fs_type {
        "ramfs" => Arc::new(ramfs::RamFs::new()),
        "fat32" | "vfat" => {
            // Would need block device
            return Err(VfsError::NotSupported);
        }
        "ext4" => {
            return Err(VfsError::NotSupported);
        }
        _ => return Err(VfsError::NotSupported),
    };

    MOUNTED.write().push(MountedFs {
        fs,
        mount_point: String::from(mount_point),
        device: String::from(device),
    });

    Ok(())
}

/// Unmount filesystem
pub fn unmount(mount_point: &str) -> Result<(), VfsError> {
    let mut mounted = MOUNTED.write();

    if let Some(pos) = mounted.iter().position(|m| m.mount_point == mount_point) {
        let mfs = &mounted[pos];
        mfs.fs.sync()?;
        mounted.remove(pos);
        crate::kinfo!("Unmounted {}", mount_point);
        Ok(())
    } else {
        Err(VfsError::NotMounted)
    }
}

/// Find filesystem for path
pub fn find_fs(path: &str) -> Option<(Arc<dyn Filesystem>, String)> {
    let mounted = MOUNTED.read();

    let mut best_match: Option<&MountedFs> = None;
    let mut best_len = 0;

    for mfs in mounted.iter() {
        if path.starts_with(&mfs.mount_point) && mfs.mount_point.len() > best_len {
            best_match = Some(mfs);
            best_len = mfs.mount_point.len();
        }
    }

    best_match.map(|mfs| {
        let relative = if mfs.mount_point == "/" {
            path.to_string()
        } else {
            path[mfs.mount_point.len()..].to_string()
        };
        (mfs.fs.clone(), relative)
    })
}

/// Initialize filesystem subsystem
pub fn init() {
    // Mount root ramfs
    let _ = mount("none", "/", "ramfs");

    // Create essential directories
    if let Some((fs, _)) = find_fs("/") {
        let _ = fs.mkdir("/dev");
        let _ = fs.mkdir("/proc");
        let _ = fs.mkdir("/sys");
        let _ = fs.mkdir("/tmp");
        let _ = fs.mkdir("/home");
        let _ = fs.mkdir("/etc");
        let _ = fs.mkdir("/var");
        let _ = fs.mkdir("/models");
    }

    crate::kprintln!("  Filesystem layer initialized");
}
