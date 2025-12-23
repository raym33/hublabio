//! Virtual File System (VFS)
//!
//! Provides a unified interface for all file operations.
//! Actual filesystem implementations run in userspace services.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::RwLock;

/// File descriptor counter
static FD_COUNTER: AtomicU64 = AtomicU64::new(3); // 0, 1, 2 reserved for stdin/out/err

/// Mount points
static MOUNTS: RwLock<BTreeMap<String, MountPoint>> = RwLock::new(BTreeMap::new());

/// Open file descriptors
static OPEN_FILES: RwLock<BTreeMap<u64, OpenFile>> = RwLock::new(BTreeMap::new());

/// File descriptor
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileDescriptor(pub u64);

/// File types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileType {
    Regular,
    Directory,
    SymLink,
    CharDevice,
    BlockDevice,
    Pipe,
    Socket,
}

/// File permissions
#[derive(Clone, Copy, Debug)]
pub struct FilePermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

/// File metadata
#[derive(Clone, Debug)]
pub struct FileStat {
    pub file_type: FileType,
    pub size: u64,
    pub permissions: FilePermissions,
    pub created: u64,
    pub modified: u64,
    pub accessed: u64,
    pub inode: u64,
}

/// Open file flags
#[derive(Clone, Copy, Debug)]
pub struct OpenFlags {
    pub read: bool,
    pub write: bool,
    pub create: bool,
    pub truncate: bool,
    pub append: bool,
}

impl OpenFlags {
    pub const READ: Self = Self {
        read: true, write: false, create: false, truncate: false, append: false,
    };

    pub const WRITE: Self = Self {
        read: false, write: true, create: true, truncate: true, append: false,
    };

    pub const READ_WRITE: Self = Self {
        read: true, write: true, create: false, truncate: false, append: false,
    };

    pub const APPEND: Self = Self {
        read: false, write: true, create: true, truncate: false, append: true,
    };
}

/// An open file
struct OpenFile {
    path: String,
    flags: OpenFlags,
    position: u64,
    mount_point: String,
}

/// Mount point
struct MountPoint {
    /// Filesystem type (e.g., "ramfs", "ext4", "fat32")
    fs_type: String,
    /// IPC channel to filesystem service
    service_channel: crate::ipc::ChannelId,
    /// Read-only mount
    read_only: bool,
}

/// VFS errors
#[derive(Debug)]
pub enum VfsError {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    NotADirectory,
    IsADirectory,
    NotEmpty,
    InvalidPath,
    IoError,
    NoSpace,
    ReadOnly,
    InvalidFd,
    NotMounted,
}

/// Initialize VFS
pub fn init() {
    // Create initial mount points
    let mut mounts = MOUNTS.write();

    // Root filesystem (ramfs initially)
    mounts.insert(String::from("/"), MountPoint {
        fs_type: String::from("ramfs"),
        service_channel: crate::ipc::ChannelId(0), // Special: kernel-internal
        read_only: false,
    });

    // Proc filesystem
    mounts.insert(String::from("/proc"), MountPoint {
        fs_type: String::from("procfs"),
        service_channel: crate::ipc::ChannelId(0),
        read_only: true,
    });

    // Sys filesystem
    mounts.insert(String::from("/sys"), MountPoint {
        fs_type: String::from("sysfs"),
        service_channel: crate::ipc::ChannelId(0),
        read_only: true,
    });

    // Dev filesystem
    mounts.insert(String::from("/dev"), MountPoint {
        fs_type: String::from("devfs"),
        service_channel: crate::ipc::ChannelId(0),
        read_only: false,
    });

    crate::kprintln!("  VFS initialized with {} mount points", mounts.len());
}

/// Open a file
pub fn open(path: &str, flags: OpenFlags) -> Result<FileDescriptor, VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }

    // Find the mount point
    let mount_point = find_mount_point(path)?;

    // TODO: Send open request to filesystem service via IPC

    let fd = FD_COUNTER.fetch_add(1, Ordering::SeqCst);

    OPEN_FILES.write().insert(fd, OpenFile {
        path: String::from(path),
        flags,
        position: 0,
        mount_point,
    });

    Ok(FileDescriptor(fd))
}

/// Read from a file
pub fn read(fd: FileDescriptor, buf: &mut [u8]) -> Result<usize, VfsError> {
    let files = OPEN_FILES.read();
    let file = files.get(&fd.0).ok_or(VfsError::InvalidFd)?;

    if !file.flags.read {
        return Err(VfsError::PermissionDenied);
    }

    // TODO: Send read request to filesystem service via IPC

    Ok(0)
}

/// Write to a file
pub fn write(fd: FileDescriptor, buf: &[u8]) -> Result<usize, VfsError> {
    let files = OPEN_FILES.read();
    let file = files.get(&fd.0).ok_or(VfsError::InvalidFd)?;

    if !file.flags.write {
        return Err(VfsError::PermissionDenied);
    }

    // TODO: Send write request to filesystem service via IPC

    Ok(buf.len())
}

/// Close a file
pub fn close(fd: FileDescriptor) -> Result<(), VfsError> {
    OPEN_FILES.write().remove(&fd.0).ok_or(VfsError::InvalidFd)?;
    Ok(())
}

/// Get file status
pub fn stat(path: &str) -> Result<FileStat, VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }

    // TODO: Query filesystem service

    Ok(FileStat {
        file_type: FileType::Regular,
        size: 0,
        permissions: FilePermissions { read: true, write: true, execute: false },
        created: 0,
        modified: 0,
        accessed: 0,
        inode: 0,
    })
}

/// Find the mount point for a path
fn find_mount_point(path: &str) -> Result<String, VfsError> {
    let mounts = MOUNTS.read();

    // Find longest matching mount point
    let mut best_match = None;
    let mut best_len = 0;

    for mount_path in mounts.keys() {
        if path.starts_with(mount_path.as_str()) && mount_path.len() > best_len {
            best_match = Some(mount_path.clone());
            best_len = mount_path.len();
        }
    }

    best_match.ok_or(VfsError::NotMounted)
}

/// Mount a filesystem
pub fn mount(
    source: &str,
    target: &str,
    fs_type: &str,
    read_only: bool,
    service_channel: crate::ipc::ChannelId,
) -> Result<(), VfsError> {
    let mut mounts = MOUNTS.write();

    if mounts.contains_key(target) {
        return Err(VfsError::AlreadyExists);
    }

    mounts.insert(String::from(target), MountPoint {
        fs_type: String::from(fs_type),
        service_channel,
        read_only,
    });

    crate::kprintln!("[VFS] Mounted {} on {} ({})", source, target, fs_type);

    Ok(())
}

/// Unmount a filesystem
pub fn unmount(target: &str) -> Result<(), VfsError> {
    let mut mounts = MOUNTS.write();

    if target == "/" {
        return Err(VfsError::PermissionDenied); // Can't unmount root
    }

    mounts.remove(target).ok_or(VfsError::NotMounted)?;

    crate::kprintln!("[VFS] Unmounted {}", target);

    Ok(())
}

/// List directory contents
pub fn readdir(path: &str) -> Result<Vec<String>, VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }

    // TODO: Query filesystem service

    Ok(Vec::new())
}
