//! RAM Filesystem
//!
//! In-memory filesystem for temporary storage.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::RwLock;

use super::{Filesystem, FileHandle, DirEntry};
use crate::vfs::{FileType, FileStat, FilePermissions, OpenFlags, VfsError};

/// Inode counter
static INODE_COUNTER: AtomicU64 = AtomicU64::new(2); // 1 reserved for root

/// RAM filesystem inode
#[derive(Clone)]
struct RamInode {
    inode: u64,
    file_type: FileType,
    data: Vec<u8>,
    children: BTreeMap<String, u64>,
    permissions: FilePermissions,
    created: u64,
    modified: u64,
}

impl RamInode {
    fn new_file() -> Self {
        Self {
            inode: INODE_COUNTER.fetch_add(1, Ordering::SeqCst),
            file_type: FileType::Regular,
            data: Vec::new(),
            children: BTreeMap::new(),
            permissions: FilePermissions {
                read: true,
                write: true,
                execute: false,
            },
            created: 0,
            modified: 0,
        }
    }

    fn new_dir() -> Self {
        let mut inode = Self::new_file();
        inode.file_type = FileType::Directory;
        inode
    }
}

/// RAM filesystem
pub struct RamFs {
    inodes: RwLock<BTreeMap<u64, RamInode>>,
    root: u64,
}

impl RamFs {
    /// Create a new RAM filesystem
    pub fn new() -> Self {
        let mut inodes = BTreeMap::new();

        // Create root directory
        let root = RamInode {
            inode: 1,
            file_type: FileType::Directory,
            data: Vec::new(),
            children: BTreeMap::new(),
            permissions: FilePermissions {
                read: true,
                write: true,
                execute: true,
            },
            created: 0,
            modified: 0,
        };

        inodes.insert(1, root);

        Self {
            inodes: RwLock::new(inodes),
            root: 1,
        }
    }

    /// Find inode by path
    fn find_inode(&self, path: &str) -> Result<u64, VfsError> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return Ok(self.root);
        }

        let inodes = self.inodes.read();
        let mut current = self.root;

        for part in parts {
            let inode = inodes.get(&current).ok_or(VfsError::NotFound)?;

            if inode.file_type != FileType::Directory {
                return Err(VfsError::NotADirectory);
            }

            current = *inode.children.get(part).ok_or(VfsError::NotFound)?;
        }

        Ok(current)
    }

    /// Find parent directory and name
    fn find_parent(&self, path: &str) -> Result<(u64, String), VfsError> {
        let path = path.trim_end_matches('/');
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return Err(VfsError::InvalidPath);
        }

        let name = parts.last().unwrap().to_string();
        let parent_path = parts[..parts.len() - 1].join("/");

        let parent_ino = if parent_path.is_empty() {
            self.root
        } else {
            self.find_inode(&format!("/{}", parent_path))?
        };

        Ok((parent_ino, name))
    }
}

impl Filesystem for RamFs {
    fn name(&self) -> &'static str {
        "ramfs"
    }

    fn is_readonly(&self) -> bool {
        false
    }

    fn total_size(&self) -> u64 {
        // Report available memory
        crate::memory::free_memory() as u64
    }

    fn free_space(&self) -> u64 {
        crate::memory::free_memory() as u64
    }

    fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, VfsError> {
        let ino = match self.find_inode(path) {
            Ok(ino) => ino,
            Err(VfsError::NotFound) if flags.create => {
                // Create file
                let (parent_ino, name) = self.find_parent(path)?;
                let new_inode = RamInode::new_file();
                let new_ino = new_inode.inode;

                let mut inodes = self.inodes.write();
                inodes.insert(new_ino, new_inode);

                if let Some(parent) = inodes.get_mut(&parent_ino) {
                    parent.children.insert(name, new_ino);
                }

                new_ino
            }
            Err(e) => return Err(e),
        };

        let inodes = self.inodes.read();
        let inode = inodes.get(&ino).ok_or(VfsError::NotFound)?;

        if inode.file_type == FileType::Directory {
            return Err(VfsError::IsADirectory);
        }

        Ok(FileHandle {
            inode: ino,
            position: if flags.append { inode.data.len() as u64 } else { 0 },
            flags,
        })
    }

    fn read(&self, handle: &FileHandle, buf: &mut [u8]) -> Result<usize, VfsError> {
        let inodes = self.inodes.read();
        let inode = inodes.get(&handle.inode).ok_or(VfsError::NotFound)?;

        let pos = handle.position as usize;
        if pos >= inode.data.len() {
            return Ok(0);
        }

        let available = inode.data.len() - pos;
        let to_read = available.min(buf.len());

        buf[..to_read].copy_from_slice(&inode.data[pos..pos + to_read]);

        Ok(to_read)
    }

    fn write(&self, handle: &FileHandle, buf: &[u8]) -> Result<usize, VfsError> {
        if !handle.flags.write {
            return Err(VfsError::PermissionDenied);
        }

        let mut inodes = self.inodes.write();
        let inode = inodes.get_mut(&handle.inode).ok_or(VfsError::NotFound)?;

        let pos = handle.position as usize;

        // Extend if needed
        if pos > inode.data.len() {
            inode.data.resize(pos, 0);
        }

        // Write data
        let end = pos + buf.len();
        if end > inode.data.len() {
            inode.data.resize(end, 0);
        }

        inode.data[pos..end].copy_from_slice(buf);

        Ok(buf.len())
    }

    fn close(&self, _handle: FileHandle) -> Result<(), VfsError> {
        Ok(())
    }

    fn stat(&self, path: &str) -> Result<FileStat, VfsError> {
        let ino = self.find_inode(path)?;
        let inodes = self.inodes.read();
        let inode = inodes.get(&ino).ok_or(VfsError::NotFound)?;

        Ok(FileStat {
            file_type: inode.file_type,
            size: inode.data.len() as u64,
            permissions: inode.permissions,
            created: inode.created,
            modified: inode.modified,
            accessed: inode.modified,
            inode: ino,
        })
    }

    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError> {
        let ino = self.find_inode(path)?;
        let inodes = self.inodes.read();
        let inode = inodes.get(&ino).ok_or(VfsError::NotFound)?;

        if inode.file_type != FileType::Directory {
            return Err(VfsError::NotADirectory);
        }

        let mut entries = Vec::new();

        for (name, &child_ino) in &inode.children {
            if let Some(child) = inodes.get(&child_ino) {
                entries.push(DirEntry {
                    name: name.clone(),
                    file_type: child.file_type,
                    inode: child_ino,
                });
            }
        }

        Ok(entries)
    }

    fn mkdir(&self, path: &str) -> Result<(), VfsError> {
        let (parent_ino, name) = self.find_parent(path)?;

        let new_inode = RamInode::new_dir();
        let new_ino = new_inode.inode;

        let mut inodes = self.inodes.write();

        // Check if already exists
        if let Some(parent) = inodes.get(&parent_ino) {
            if parent.children.contains_key(&name) {
                return Err(VfsError::AlreadyExists);
            }
        }

        inodes.insert(new_ino, new_inode);

        if let Some(parent) = inodes.get_mut(&parent_ino) {
            parent.children.insert(name, new_ino);
        }

        Ok(())
    }

    fn unlink(&self, path: &str) -> Result<(), VfsError> {
        let ino = self.find_inode(path)?;
        let (parent_ino, name) = self.find_parent(path)?;

        let mut inodes = self.inodes.write();

        // Check it's a file
        if let Some(inode) = inodes.get(&ino) {
            if inode.file_type == FileType::Directory {
                return Err(VfsError::IsADirectory);
            }
        }

        // Remove from parent
        if let Some(parent) = inodes.get_mut(&parent_ino) {
            parent.children.remove(&name);
        }

        // Remove inode
        inodes.remove(&ino);

        Ok(())
    }

    fn rmdir(&self, path: &str) -> Result<(), VfsError> {
        let ino = self.find_inode(path)?;
        let (parent_ino, name) = self.find_parent(path)?;

        let mut inodes = self.inodes.write();

        // Check it's an empty directory
        if let Some(inode) = inodes.get(&ino) {
            if inode.file_type != FileType::Directory {
                return Err(VfsError::NotADirectory);
            }
            if !inode.children.is_empty() {
                return Err(VfsError::NotEmpty);
            }
        }

        // Remove from parent
        if let Some(parent) = inodes.get_mut(&parent_ino) {
            parent.children.remove(&name);
        }

        // Remove inode
        inodes.remove(&ino);

        Ok(())
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), VfsError> {
        let from_ino = self.find_inode(from)?;
        let (from_parent, from_name) = self.find_parent(from)?;
        let (to_parent, to_name) = self.find_parent(to)?;

        let mut inodes = self.inodes.write();

        // Remove from old parent
        if let Some(parent) = inodes.get_mut(&from_parent) {
            parent.children.remove(&from_name);
        }

        // Add to new parent
        if let Some(parent) = inodes.get_mut(&to_parent) {
            parent.children.insert(to_name, from_ino);
        }

        Ok(())
    }

    fn sync(&self) -> Result<(), VfsError> {
        Ok(()) // Nothing to sync for RAM fs
    }
}

impl Default for RamFs {
    fn default() -> Self {
        Self::new()
    }
}
