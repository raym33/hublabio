//! ext4 Filesystem Driver
//!
//! Read support for ext4 partitions.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::RwLock;

use super::{Filesystem, FileHandle, DirEntry, BlockDevice};
use crate::vfs::{FileType, FileStat, FilePermissions, OpenFlags, VfsError};

/// ext4 superblock offset
const SUPERBLOCK_OFFSET: u64 = 1024;

/// ext4 Superblock
#[repr(C)]
#[derive(Clone, Copy)]
struct Ext4Superblock {
    s_inodes_count: u32,
    s_blocks_count_lo: u32,
    s_r_blocks_count_lo: u32,
    s_free_blocks_count_lo: u32,
    s_free_inodes_count: u32,
    s_first_data_block: u32,
    s_log_block_size: u32,
    s_log_cluster_size: u32,
    s_blocks_per_group: u32,
    s_clusters_per_group: u32,
    s_inodes_per_group: u32,
    s_mtime: u32,
    s_wtime: u32,
    s_mnt_count: u16,
    s_max_mnt_count: u16,
    s_magic: u16,
    s_state: u16,
    s_errors: u16,
    s_minor_rev_level: u16,
    s_lastcheck: u32,
    s_checkinterval: u32,
    s_creator_os: u32,
    s_rev_level: u32,
    s_def_resuid: u16,
    s_def_resgid: u16,
    // Extended fields
    s_first_ino: u32,
    s_inode_size: u16,
    s_block_group_nr: u16,
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_volume_name: [u8; 16],
    // More fields...
}

/// ext4 magic number
const EXT4_MAGIC: u16 = 0xEF53;

/// ext4 inode
#[repr(C)]
#[derive(Clone, Copy)]
struct Ext4Inode {
    i_mode: u16,
    i_uid: u16,
    i_size_lo: u32,
    i_atime: u32,
    i_ctime: u32,
    i_mtime: u32,
    i_dtime: u32,
    i_gid: u16,
    i_links_count: u16,
    i_blocks_lo: u32,
    i_flags: u32,
    i_osd1: u32,
    i_block: [u32; 15],  // Block pointers
    i_generation: u32,
    i_file_acl_lo: u32,
    i_size_high: u32,
    i_obso_faddr: u32,
    i_osd2: [u8; 12],
    i_extra_isize: u16,
    i_checksum_hi: u16,
    i_ctime_extra: u32,
    i_mtime_extra: u32,
    i_atime_extra: u32,
    i_crtime: u32,
    i_crtime_extra: u32,
    i_version_hi: u32,
    i_projid: u32,
}

/// Inode file types
mod inode_type {
    pub const FIFO: u16 = 0x1000;
    pub const CHARDEV: u16 = 0x2000;
    pub const DIR: u16 = 0x4000;
    pub const BLOCKDEV: u16 = 0x6000;
    pub const FILE: u16 = 0x8000;
    pub const SYMLINK: u16 = 0xA000;
    pub const SOCKET: u16 = 0xC000;
}

/// ext4 directory entry
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Ext4DirEntry {
    inode: u32,
    rec_len: u16,
    name_len: u8,
    file_type: u8,
    // name follows
}

/// Directory entry file types
mod dir_type {
    pub const UNKNOWN: u8 = 0;
    pub const REG_FILE: u8 = 1;
    pub const DIR: u8 = 2;
    pub const CHRDEV: u8 = 3;
    pub const BLKDEV: u8 = 4;
    pub const FIFO: u8 = 5;
    pub const SOCK: u8 = 6;
    pub const SYMLINK: u8 = 7;
}

/// Block group descriptor
#[repr(C)]
#[derive(Clone, Copy)]
struct Ext4GroupDesc {
    bg_block_bitmap_lo: u32,
    bg_inode_bitmap_lo: u32,
    bg_inode_table_lo: u32,
    bg_free_blocks_count_lo: u16,
    bg_free_inodes_count_lo: u16,
    bg_used_dirs_count_lo: u16,
    bg_flags: u16,
    bg_exclude_bitmap_lo: u32,
    bg_block_bitmap_csum_lo: u16,
    bg_inode_bitmap_csum_lo: u16,
    bg_itable_unused_lo: u16,
    bg_checksum: u16,
    // 64-bit fields
    bg_block_bitmap_hi: u32,
    bg_inode_bitmap_hi: u32,
    bg_inode_table_hi: u32,
    bg_free_blocks_count_hi: u16,
    bg_free_inodes_count_hi: u16,
    bg_used_dirs_count_hi: u16,
    bg_itable_unused_hi: u16,
    bg_exclude_bitmap_hi: u32,
    bg_block_bitmap_csum_hi: u16,
    bg_inode_bitmap_csum_hi: u16,
    bg_reserved: u32,
}

/// Root inode number
const ROOT_INODE: u32 = 2;

/// ext4 filesystem
pub struct Ext4Fs {
    device: Arc<dyn BlockDevice>,
    block_size: u32,
    inode_size: u32,
    inodes_per_group: u32,
    blocks_per_group: u32,
    group_desc_size: u32,
    total_inodes: u32,
    total_blocks: u64,
    first_data_block: u32,
    inode_cache: RwLock<BTreeMap<u32, Ext4Inode>>,
}

impl Ext4Fs {
    /// Mount ext4 filesystem
    pub fn mount(device: Arc<dyn BlockDevice>) -> Result<Self, VfsError> {
        let dev_block_size = device.block_size();

        // Read superblock
        let sb_block = SUPERBLOCK_OFFSET / dev_block_size as u64;
        let sb_offset = (SUPERBLOCK_OFFSET % dev_block_size as u64) as usize;

        let mut buf = alloc::vec![0u8; dev_block_size];
        device.read_block(sb_block, &mut buf)?;

        let sb = unsafe {
            core::ptr::read_unaligned(buf[sb_offset..].as_ptr() as *const Ext4Superblock)
        };

        // Validate magic
        if sb.s_magic != EXT4_MAGIC {
            return Err(VfsError::InvalidPath);
        }

        let block_size = 1024u32 << sb.s_log_block_size;
        let inode_size = sb.s_inode_size as u32;

        crate::kinfo!(
            "ext4: {} MB, block size {} bytes, inode size {} bytes",
            (sb.s_blocks_count_lo as u64 * block_size as u64) / (1024 * 1024),
            block_size,
            inode_size
        );

        Ok(Self {
            device,
            block_size,
            inode_size,
            inodes_per_group: sb.s_inodes_per_group,
            blocks_per_group: sb.s_blocks_per_group,
            group_desc_size: 64, // ext4 with 64-bit support
            total_inodes: sb.s_inodes_count,
            total_blocks: sb.s_blocks_count_lo as u64,
            first_data_block: sb.s_first_data_block,
            inode_cache: RwLock::new(BTreeMap::new()),
        })
    }

    /// Read a block
    fn read_block(&self, block: u64) -> Result<Vec<u8>, VfsError> {
        let dev_block_size = self.device.block_size() as u64;
        let blocks_per_fs_block = self.block_size as u64 / dev_block_size;

        let mut data = alloc::vec![0u8; self.block_size as usize];

        for i in 0..blocks_per_fs_block {
            let offset = (i * dev_block_size) as usize;
            self.device.read_block(
                block * blocks_per_fs_block + i,
                &mut data[offset..offset + dev_block_size as usize],
            )?;
        }

        Ok(data)
    }

    /// Get block group descriptor
    fn read_group_desc(&self, group: u32) -> Result<Ext4GroupDesc, VfsError> {
        let desc_per_block = self.block_size / self.group_desc_size;
        let block = self.first_data_block as u64 + 1 + (group / desc_per_block) as u64;
        let offset = ((group % desc_per_block) * self.group_desc_size) as usize;

        let data = self.read_block(block)?;

        Ok(unsafe {
            core::ptr::read_unaligned(data[offset..].as_ptr() as *const Ext4GroupDesc)
        })
    }

    /// Read inode
    fn read_inode(&self, ino: u32) -> Result<Ext4Inode, VfsError> {
        // Check cache
        if let Some(&inode) = self.inode_cache.read().get(&ino) {
            return Ok(inode);
        }

        let group = (ino - 1) / self.inodes_per_group;
        let index = (ino - 1) % self.inodes_per_group;

        let gd = self.read_group_desc(group)?;
        let inode_table = gd.bg_inode_table_lo as u64 |
            ((gd.bg_inode_table_hi as u64) << 32);

        let inodes_per_block = self.block_size / self.inode_size;
        let block = inode_table + (index / inodes_per_block) as u64;
        let offset = ((index % inodes_per_block) * self.inode_size) as usize;

        let data = self.read_block(block)?;

        let inode = unsafe {
            core::ptr::read_unaligned(data[offset..].as_ptr() as *const Ext4Inode)
        };

        // Cache it
        self.inode_cache.write().insert(ino, inode);

        Ok(inode)
    }

    /// Read inode data blocks
    fn read_inode_data(&self, inode: &Ext4Inode, offset: u64, buf: &mut [u8]) -> Result<usize, VfsError> {
        let size = inode.i_size_lo as u64 | ((inode.i_size_high as u64) << 32);

        if offset >= size {
            return Ok(0);
        }

        let to_read = (size - offset).min(buf.len() as u64) as usize;
        let mut total_read = 0;

        // For simplicity, only handle direct blocks (first 12 blocks)
        let start_block = (offset / self.block_size as u64) as usize;
        let offset_in_block = (offset % self.block_size as u64) as usize;

        for i in start_block..12 {
            if total_read >= to_read {
                break;
            }

            let block_num = inode.i_block[i];
            if block_num == 0 {
                break;
            }

            let data = self.read_block(block_num as u64)?;

            let start = if i == start_block { offset_in_block } else { 0 };
            let end = (start + to_read - total_read).min(self.block_size as usize);
            let len = end - start;

            buf[total_read..total_read + len].copy_from_slice(&data[start..end]);
            total_read += len;
        }

        Ok(total_read)
    }

    /// Read directory
    fn read_directory(&self, ino: u32) -> Result<Vec<(u32, String, u8)>, VfsError> {
        let inode = self.read_inode(ino)?;

        if inode.i_mode & inode_type::DIR == 0 {
            return Err(VfsError::NotADirectory);
        }

        let size = inode.i_size_lo as usize;
        let mut data = alloc::vec![0u8; size];
        self.read_inode_data(&inode, 0, &mut data)?;

        let mut entries = Vec::new();
        let mut pos = 0;

        while pos + 8 <= data.len() {
            let entry = unsafe {
                core::ptr::read_unaligned(data[pos..].as_ptr() as *const Ext4DirEntry)
            };

            if entry.inode == 0 || entry.rec_len == 0 {
                break;
            }

            if entry.name_len > 0 {
                let name_start = pos + 8;
                let name_end = name_start + entry.name_len as usize;

                if name_end <= data.len() {
                    let name = core::str::from_utf8(&data[name_start..name_end])
                        .unwrap_or("")
                        .to_string();

                    entries.push((entry.inode, name, entry.file_type));
                }
            }

            pos += entry.rec_len as usize;
        }

        Ok(entries)
    }

    /// Find inode by path
    fn find_inode(&self, path: &str) -> Result<u32, VfsError> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        let mut current_ino = ROOT_INODE;

        for part in parts {
            let entries = self.read_directory(current_ino)?;

            let found = entries.iter().find(|(_, name, _)| name == part);

            match found {
                Some((ino, _, _)) => {
                    current_ino = *ino;
                }
                None => return Err(VfsError::NotFound),
            }
        }

        Ok(current_ino)
    }
}

impl Filesystem for Ext4Fs {
    fn name(&self) -> &'static str {
        "ext4"
    }

    fn is_readonly(&self) -> bool {
        true // Read-only for now
    }

    fn total_size(&self) -> u64 {
        self.total_blocks * self.block_size as u64
    }

    fn free_space(&self) -> u64 {
        self.total_size() / 2
    }

    fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, VfsError> {
        if flags.write || flags.create {
            return Err(VfsError::ReadOnly);
        }

        let ino = self.find_inode(path)?;
        let inode = self.read_inode(ino)?;

        if inode.i_mode & inode_type::DIR != 0 {
            return Err(VfsError::IsADirectory);
        }

        Ok(FileHandle {
            inode: ino as u64,
            position: 0,
            flags,
        })
    }

    fn read(&self, handle: &FileHandle, buf: &mut [u8]) -> Result<usize, VfsError> {
        let inode = self.read_inode(handle.inode as u32)?;
        self.read_inode_data(&inode, handle.position, buf)
    }

    fn write(&self, _handle: &FileHandle, _buf: &[u8]) -> Result<usize, VfsError> {
        Err(VfsError::ReadOnly)
    }

    fn close(&self, _handle: FileHandle) -> Result<(), VfsError> {
        Ok(())
    }

    fn stat(&self, path: &str) -> Result<FileStat, VfsError> {
        let ino = self.find_inode(path)?;
        let inode = self.read_inode(ino)?;

        let file_type = match inode.i_mode & 0xF000 {
            inode_type::DIR => FileType::Directory,
            inode_type::SYMLINK => FileType::SymLink,
            inode_type::CHARDEV => FileType::CharDevice,
            inode_type::BLOCKDEV => FileType::BlockDevice,
            inode_type::FIFO => FileType::Pipe,
            inode_type::SOCKET => FileType::Socket,
            _ => FileType::Regular,
        };

        Ok(FileStat {
            file_type,
            size: inode.i_size_lo as u64 | ((inode.i_size_high as u64) << 32),
            permissions: FilePermissions {
                read: inode.i_mode & 0o400 != 0,
                write: inode.i_mode & 0o200 != 0,
                execute: inode.i_mode & 0o100 != 0,
            },
            created: inode.i_crtime as u64,
            modified: inode.i_mtime as u64,
            accessed: inode.i_atime as u64,
            inode: ino as u64,
        })
    }

    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError> {
        let ino = self.find_inode(path)?;
        let entries = self.read_directory(ino)?;

        Ok(entries
            .into_iter()
            .map(|(ino, name, file_type)| DirEntry {
                name,
                file_type: match file_type {
                    dir_type::DIR => FileType::Directory,
                    dir_type::SYMLINK => FileType::SymLink,
                    dir_type::CHRDEV => FileType::CharDevice,
                    dir_type::BLKDEV => FileType::BlockDevice,
                    _ => FileType::Regular,
                },
                inode: ino as u64,
            })
            .collect())
    }

    fn mkdir(&self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::ReadOnly)
    }

    fn unlink(&self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::ReadOnly)
    }

    fn rmdir(&self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::ReadOnly)
    }

    fn rename(&self, _from: &str, _to: &str) -> Result<(), VfsError> {
        Err(VfsError::ReadOnly)
    }

    fn sync(&self) -> Result<(), VfsError> {
        Ok(())
    }
}
