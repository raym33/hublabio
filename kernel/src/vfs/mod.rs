//! Virtual File System (VFS)
//!
//! Provides a unified interface for all file operations.
//! Implements ramfs for in-memory storage.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::{Mutex, RwLock};

/// File descriptor counter
static FD_COUNTER: AtomicU64 = AtomicU64::new(3); // 0, 1, 2 reserved for stdin/out/err

/// Inode counter
static INODE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Global filesystem
static ROOT_FS: RwLock<Option<RamFs>> = RwLock::new(None);

/// Open file descriptors
static OPEN_FILES: RwLock<BTreeMap<u32, OpenFile>> = RwLock::new(BTreeMap::new());

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

/// File stat structure (POSIX compatible)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Stat {
    pub st_dev: u64,
    pub st_ino: u64,
    pub st_mode: u32,
    pub st_nlink: u32,
    pub st_uid: u32,
    pub st_gid: u32,
    pub st_rdev: u64,
    pub st_size: i64,
    pub st_blksize: i64,
    pub st_blocks: i64,
    pub st_atime: i64,
    pub st_mtime: i64,
    pub st_ctime: i64,
}

impl Stat {
    pub fn new_file(size: u64, inode: u64) -> Self {
        Self {
            st_dev: 1,
            st_ino: inode,
            st_mode: 0o100644, // Regular file
            st_nlink: 1,
            st_uid: 0,
            st_gid: 0,
            st_rdev: 0,
            st_size: size as i64,
            st_blksize: 4096,
            st_blocks: ((size + 511) / 512) as i64,
            st_atime: 0,
            st_mtime: 0,
            st_ctime: 0,
        }
    }

    pub fn new_dir(inode: u64) -> Self {
        Self {
            st_dev: 1,
            st_ino: inode,
            st_mode: 0o40755, // Directory
            st_nlink: 2,
            st_uid: 0,
            st_gid: 0,
            st_rdev: 0,
            st_size: 4096,
            st_blksize: 4096,
            st_blocks: 8,
            st_atime: 0,
            st_mtime: 0,
            st_ctime: 0,
        }
    }

    pub fn is_dir(&self) -> bool {
        (self.st_mode & 0o170000) == 0o40000
    }

    pub fn is_file(&self) -> bool {
        (self.st_mode & 0o170000) == 0o100000
    }
}

/// Directory entry
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DirEntry {
    pub d_ino: u64,
    pub d_off: i64,
    pub d_reclen: u16,
    pub d_type: u8,
    pub d_name: [u8; 256],
}

impl DirEntry {
    pub fn new(name: &str, inode: u64, file_type: u8) -> Self {
        let mut entry = Self {
            d_ino: inode,
            d_off: 0,
            d_reclen: core::mem::size_of::<DirEntry>() as u16,
            d_type: file_type,
            d_name: [0; 256],
        };
        let bytes = name.as_bytes();
        let len = bytes.len().min(255);
        entry.d_name[..len].copy_from_slice(&bytes[..len]);
        entry
    }
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

/// An open file
struct OpenFile {
    path: String,
    flags: u32,
    position: u64,
    inode: u64,
}

/// RAM filesystem inode
#[derive(Clone)]
struct Inode {
    id: u64,
    file_type: FileType,
    data: Vec<u8>,
    children: BTreeMap<String, u64>, // name -> inode_id (for directories)
    parent: Option<u64>,
}

/// RAM filesystem
struct RamFs {
    inodes: BTreeMap<u64, Inode>,
    root_inode: u64,
}

impl RamFs {
    fn new() -> Self {
        let root_inode = INODE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut inodes = BTreeMap::new();

        let mut root = Inode {
            id: root_inode,
            file_type: FileType::Directory,
            data: Vec::new(),
            children: BTreeMap::new(),
            parent: None,
        };

        // Create standard directories
        let dirs = ["bin", "dev", "etc", "home", "proc", "sys", "tmp", "var"];
        for dir in dirs {
            let inode_id = INODE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let inode = Inode {
                id: inode_id,
                file_type: FileType::Directory,
                data: Vec::new(),
                children: BTreeMap::new(),
                parent: Some(root_inode),
            };
            inodes.insert(inode_id, inode);
            root.children.insert(String::from(dir), inode_id);
        }

        inodes.insert(root_inode, root);

        Self { inodes, root_inode }
    }

    fn resolve_path(&self, path: &str) -> Result<u64, VfsError> {
        if path == "/" {
            return Ok(self.root_inode);
        }

        let mut current = self.root_inode;

        for component in path.trim_start_matches('/').split('/') {
            if component.is_empty() || component == "." {
                continue;
            }

            if component == ".." {
                if let Some(inode) = self.inodes.get(&current) {
                    current = inode.parent.unwrap_or(self.root_inode);
                }
                continue;
            }

            let inode = self.inodes.get(&current).ok_or(VfsError::NotFound)?;
            if inode.file_type != FileType::Directory {
                return Err(VfsError::NotADirectory);
            }

            current = *inode.children.get(component).ok_or(VfsError::NotFound)?;
        }

        Ok(current)
    }

    fn create_file(&mut self, path: &str, is_dir: bool) -> Result<u64, VfsError> {
        let (parent_path, name) = split_path(path);

        let parent_inode_id = self.resolve_path(&parent_path)?;
        let parent_inode = self.inodes.get_mut(&parent_inode_id).ok_or(VfsError::NotFound)?;

        if parent_inode.file_type != FileType::Directory {
            return Err(VfsError::NotADirectory);
        }

        if parent_inode.children.contains_key(&name) {
            return Err(VfsError::AlreadyExists);
        }

        let inode_id = INODE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let inode = Inode {
            id: inode_id,
            file_type: if is_dir { FileType::Directory } else { FileType::Regular },
            data: Vec::new(),
            children: BTreeMap::new(),
            parent: Some(parent_inode_id),
        };

        self.inodes.insert(inode_id, inode);

        // Update parent
        let parent_inode = self.inodes.get_mut(&parent_inode_id).unwrap();
        parent_inode.children.insert(name, inode_id);

        Ok(inode_id)
    }

    fn read(&self, inode_id: u64, offset: usize, buf: &mut [u8]) -> Result<usize, VfsError> {
        let inode = self.inodes.get(&inode_id).ok_or(VfsError::NotFound)?;

        if inode.file_type == FileType::Directory {
            return Err(VfsError::IsADirectory);
        }

        if offset >= inode.data.len() {
            return Ok(0);
        }

        let available = inode.data.len() - offset;
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&inode.data[offset..offset + to_read]);

        Ok(to_read)
    }

    fn write(&mut self, inode_id: u64, offset: usize, data: &[u8]) -> Result<usize, VfsError> {
        let inode = self.inodes.get_mut(&inode_id).ok_or(VfsError::NotFound)?;

        if inode.file_type == FileType::Directory {
            return Err(VfsError::IsADirectory);
        }

        // Extend if needed
        if offset + data.len() > inode.data.len() {
            inode.data.resize(offset + data.len(), 0);
        }

        inode.data[offset..offset + data.len()].copy_from_slice(data);

        Ok(data.len())
    }

    fn stat(&self, inode_id: u64) -> Result<Stat, VfsError> {
        let inode = self.inodes.get(&inode_id).ok_or(VfsError::NotFound)?;

        Ok(match inode.file_type {
            FileType::Directory => Stat::new_dir(inode_id),
            _ => Stat::new_file(inode.data.len() as u64, inode_id),
        })
    }

    fn readdir(&self, inode_id: u64) -> Result<Vec<(String, u64, u8)>, VfsError> {
        let inode = self.inodes.get(&inode_id).ok_or(VfsError::NotFound)?;

        if inode.file_type != FileType::Directory {
            return Err(VfsError::NotADirectory);
        }

        let mut entries = Vec::new();

        // Add . and ..
        entries.push((String::from("."), inode_id, 4)); // DT_DIR
        entries.push((String::from(".."), inode.parent.unwrap_or(inode_id), 4));

        for (name, &child_id) in &inode.children {
            let child = self.inodes.get(&child_id);
            let dtype = match child.map(|c| c.file_type) {
                Some(FileType::Directory) => 4, // DT_DIR
                Some(FileType::Regular) => 8,   // DT_REG
                Some(FileType::SymLink) => 10,  // DT_LNK
                _ => 0, // DT_UNKNOWN
            };
            entries.push((name.clone(), child_id, dtype));
        }

        Ok(entries)
    }

    fn unlink(&mut self, path: &str) -> Result<(), VfsError> {
        let (parent_path, name) = split_path(path);

        let parent_inode_id = self.resolve_path(&parent_path)?;
        let inode_id = {
            let parent_inode = self.inodes.get(&parent_inode_id).ok_or(VfsError::NotFound)?;
            *parent_inode.children.get(&name).ok_or(VfsError::NotFound)?
        };

        // Check if directory is empty
        if let Some(inode) = self.inodes.get(&inode_id) {
            if inode.file_type == FileType::Directory && !inode.children.is_empty() {
                return Err(VfsError::NotEmpty);
            }
        }

        // Remove from parent
        if let Some(parent) = self.inodes.get_mut(&parent_inode_id) {
            parent.children.remove(&name);
        }

        // Remove inode
        self.inodes.remove(&inode_id);

        Ok(())
    }
}

fn split_path(path: &str) -> (String, String) {
    let path = path.trim_end_matches('/');
    if let Some(pos) = path.rfind('/') {
        let parent = if pos == 0 { "/" } else { &path[..pos] };
        let name = &path[pos + 1..];
        (String::from(parent), String::from(name))
    } else {
        (String::from("/"), String::from(path))
    }
}

/// Initialize VFS
pub fn init() {
    let fs = RamFs::new();
    *ROOT_FS.write() = Some(fs);
    crate::kprintln!("  VFS initialized (ramfs)");
}

/// Open a file
pub fn open(path: &str, flags: u32, mode: u32) -> Result<u32, VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }

    let mut fs = ROOT_FS.write();
    let fs = fs.as_mut().ok_or(VfsError::NotMounted)?;

    // O_CREAT = 0o100
    let create = (flags & 0o100) != 0;

    let inode_id = match fs.resolve_path(path) {
        Ok(id) => id,
        Err(VfsError::NotFound) if create => {
            fs.create_file(path, false)?
        }
        Err(e) => return Err(e),
    };

    // O_TRUNC = 0o1000
    if (flags & 0o1000) != 0 {
        if let Some(inode) = fs.inodes.get_mut(&inode_id) {
            inode.data.clear();
        }
    }

    let fd = FD_COUNTER.fetch_add(1, Ordering::SeqCst) as u32;
    let _ = mode; // mode is used for creation permissions

    OPEN_FILES.write().insert(fd, OpenFile {
        path: String::from(path),
        flags,
        position: 0,
        inode: inode_id,
    });

    Ok(fd)
}

/// Read from a file descriptor
pub fn read(fd: u32, buf: &mut [u8]) -> Result<usize, VfsError> {
    let (inode_id, position) = {
        let files = OPEN_FILES.read();
        let file = files.get(&fd).ok_or(VfsError::InvalidFd)?;
        (file.inode, file.position as usize)
    };

    let fs = ROOT_FS.read();
    let fs = fs.as_ref().ok_or(VfsError::NotMounted)?;
    let bytes_read = fs.read(inode_id, position, buf)?;

    // Update position
    if bytes_read > 0 {
        let mut files = OPEN_FILES.write();
        if let Some(file) = files.get_mut(&fd) {
            file.position += bytes_read as u64;
        }
    }

    Ok(bytes_read)
}

/// Write to a file descriptor
pub fn write(fd: u32, data: &[u8]) -> Result<usize, VfsError> {
    let (inode_id, position, is_append) = {
        let files = OPEN_FILES.read();
        let file = files.get(&fd).ok_or(VfsError::InvalidFd)?;
        let is_append = (file.flags & 0o2000) != 0; // O_APPEND
        (file.inode, file.position as usize, is_append)
    };

    let mut fs = ROOT_FS.write();
    let fs = fs.as_mut().ok_or(VfsError::NotMounted)?;

    let write_pos = if is_append {
        fs.inodes.get(&inode_id).map(|i| i.data.len()).unwrap_or(0)
    } else {
        position
    };

    let bytes_written = fs.write(inode_id, write_pos, data)?;

    // Update position
    if bytes_written > 0 {
        let mut files = OPEN_FILES.write();
        if let Some(file) = files.get_mut(&fd) {
            file.position = (write_pos + bytes_written) as u64;
        }
    }

    Ok(bytes_written)
}

/// Seek in a file
pub fn seek(fd: u32, offset: i64, whence: u32) -> Result<u64, VfsError> {
    let mut files = OPEN_FILES.write();
    let file = files.get_mut(&fd).ok_or(VfsError::InvalidFd)?;

    let fs = ROOT_FS.read();
    let fs = fs.as_ref().ok_or(VfsError::NotMounted)?;

    let size = fs.inodes.get(&file.inode)
        .map(|i| i.data.len() as i64)
        .unwrap_or(0);

    let new_pos = match whence {
        0 => offset, // SEEK_SET
        1 => file.position as i64 + offset, // SEEK_CUR
        2 => size + offset, // SEEK_END
        _ => return Err(VfsError::IoError),
    };

    if new_pos < 0 {
        return Err(VfsError::IoError);
    }

    file.position = new_pos as u64;
    Ok(file.position)
}

/// Close a file descriptor
pub fn close(fd: u32) -> Result<(), VfsError> {
    OPEN_FILES.write().remove(&fd).ok_or(VfsError::InvalidFd)?;
    Ok(())
}

/// Get file status by path
pub fn stat(path: &str) -> Result<Stat, VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }

    let fs = ROOT_FS.read();
    let fs = fs.as_ref().ok_or(VfsError::NotMounted)?;

    let inode_id = fs.resolve_path(path)?;
    fs.stat(inode_id)
}

/// Get file status by fd
pub fn fstat(fd: u32) -> Result<Stat, VfsError> {
    let files = OPEN_FILES.read();
    let file = files.get(&fd).ok_or(VfsError::InvalidFd)?;

    let fs = ROOT_FS.read();
    let fs = fs.as_ref().ok_or(VfsError::NotMounted)?;

    fs.stat(file.inode)
}

/// Create directory
pub fn mkdir(path: &str, mode: u32) -> Result<(), VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }
    let _ = mode;

    let mut fs = ROOT_FS.write();
    let fs = fs.as_mut().ok_or(VfsError::NotMounted)?;

    fs.create_file(path, true)?;
    Ok(())
}

/// Remove directory
pub fn rmdir(path: &str) -> Result<(), VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }

    let mut fs = ROOT_FS.write();
    let fs = fs.as_mut().ok_or(VfsError::NotMounted)?;

    let inode_id = fs.resolve_path(path)?;
    let inode = fs.inodes.get(&inode_id).ok_or(VfsError::NotFound)?;

    if inode.file_type != FileType::Directory {
        return Err(VfsError::NotADirectory);
    }

    if !inode.children.is_empty() {
        return Err(VfsError::NotEmpty);
    }

    fs.unlink(path)
}

/// Remove file
pub fn unlink(path: &str) -> Result<(), VfsError> {
    if !path.starts_with('/') {
        return Err(VfsError::InvalidPath);
    }

    let mut fs = ROOT_FS.write();
    let fs = fs.as_mut().ok_or(VfsError::NotMounted)?;

    let inode_id = fs.resolve_path(path)?;
    let inode = fs.inodes.get(&inode_id).ok_or(VfsError::NotFound)?;

    if inode.file_type == FileType::Directory {
        return Err(VfsError::IsADirectory);
    }

    fs.unlink(path)
}

/// Read directory entries
pub fn readdir(fd: u32, dirent_ptr: usize, count: usize) -> Result<usize, VfsError> {
    let inode_id = {
        let files = OPEN_FILES.read();
        let file = files.get(&fd).ok_or(VfsError::InvalidFd)?;
        file.inode
    };

    let fs = ROOT_FS.read();
    let fs = fs.as_ref().ok_or(VfsError::NotMounted)?;

    let entries = fs.readdir(inode_id)?;

    // Copy entries to user buffer
    let entry_size = core::mem::size_of::<DirEntry>();
    let max_entries = count / entry_size;
    let mut written = 0;

    for (i, (name, inode, dtype)) in entries.iter().enumerate() {
        if i >= max_entries {
            break;
        }

        let entry = DirEntry::new(name, *inode, *dtype);
        unsafe {
            let dst = (dirent_ptr + i * entry_size) as *mut DirEntry;
            *dst = entry;
        }
        written += entry_size;
    }

    Ok(written)
}

/// Read file content (helper)
pub fn read_file(path: &str) -> Result<Vec<u8>, VfsError> {
    let fd = open(path, 0, 0)?; // O_RDONLY
    let stat = fstat(fd)?;
    let mut buf = vec![0u8; stat.st_size as usize];
    read(fd, &mut buf)?;
    close(fd)?;
    Ok(buf)
}

/// Write file content (helper)
pub fn write_file(path: &str, data: &[u8]) -> Result<(), VfsError> {
    let fd = open(path, 0o1101, 0o644)?; // O_WRONLY | O_CREAT | O_TRUNC
    write(fd, data)?;
    close(fd)?;
    Ok(())
}

/// Check if path exists
pub fn exists(path: &str) -> bool {
    stat(path).is_ok()
}

/// List directory (helper)
pub fn list_dir(path: &str) -> Result<Vec<String>, VfsError> {
    let fd = open(path, 0o200000, 0)?; // O_DIRECTORY

    let fs = ROOT_FS.read();
    let fs = fs.as_ref().ok_or(VfsError::NotMounted)?;

    let files = OPEN_FILES.read();
    let file = files.get(&fd).ok_or(VfsError::InvalidFd)?;

    let entries = fs.readdir(file.inode)?;
    drop(files);
    drop(fs);

    close(fd)?;

    Ok(entries.into_iter().map(|(name, _, _)| name).collect())
}
