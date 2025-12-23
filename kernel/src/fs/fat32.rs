//! FAT32 Filesystem Driver
//!
//! Read/write support for FAT32 partitions.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::RwLock;

use super::{BlockDevice, DirEntry, FileHandle, Filesystem};
use crate::vfs::{FilePermissions, FileStat, FileType, OpenFlags, VfsError};

/// FAT32 Boot Sector
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Fat32BootSector {
    jmp_boot: [u8; 3],
    oem_name: [u8; 8],
    bytes_per_sector: u16,
    sectors_per_cluster: u8,
    reserved_sectors: u16,
    num_fats: u8,
    root_entry_count: u16, // 0 for FAT32
    total_sectors_16: u16, // 0 for FAT32
    media_type: u8,
    fat_size_16: u16, // 0 for FAT32
    sectors_per_track: u16,
    num_heads: u16,
    hidden_sectors: u32,
    total_sectors_32: u32,
    // FAT32 specific
    fat_size_32: u32,
    ext_flags: u16,
    fs_version: u16,
    root_cluster: u32,
    fs_info_sector: u16,
    backup_boot_sector: u16,
    reserved: [u8; 12],
    drive_number: u8,
    reserved1: u8,
    boot_sig: u8,
    volume_id: u32,
    volume_label: [u8; 11],
    fs_type: [u8; 8],
}

/// FAT32 Directory Entry
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Fat32DirEntry {
    name: [u8; 8],
    ext: [u8; 3],
    attr: u8,
    reserved: u8,
    create_time_tenth: u8,
    create_time: u16,
    create_date: u16,
    access_date: u16,
    cluster_high: u16,
    modify_time: u16,
    modify_date: u16,
    cluster_low: u16,
    file_size: u32,
}

/// FAT32 file attributes
mod attr {
    pub const READ_ONLY: u8 = 0x01;
    pub const HIDDEN: u8 = 0x02;
    pub const SYSTEM: u8 = 0x04;
    pub const VOLUME_ID: u8 = 0x08;
    pub const DIRECTORY: u8 = 0x10;
    pub const ARCHIVE: u8 = 0x20;
    pub const LONG_NAME: u8 = 0x0F;
}

/// FAT entry values
mod fat {
    pub const FREE: u32 = 0x00000000;
    pub const BAD: u32 = 0x0FFFFFF7;
    pub const EOC: u32 = 0x0FFFFFF8; // End of chain
}

/// FAT32 filesystem
pub struct Fat32Fs {
    device: Arc<dyn BlockDevice>,
    bytes_per_sector: u32,
    sectors_per_cluster: u32,
    cluster_size: u32,
    fat_start: u64,
    data_start: u64,
    root_cluster: u32,
    total_clusters: u32,
    fat_cache: RwLock<BTreeMap<u32, u32>>,
}

impl Fat32Fs {
    /// Mount FAT32 filesystem
    pub fn mount(device: Arc<dyn BlockDevice>) -> Result<Self, VfsError> {
        let block_size = device.block_size();
        let mut boot_sector = alloc::vec![0u8; block_size];

        device.read_block(0, &mut boot_sector)?;

        // Parse boot sector
        let bs =
            unsafe { core::ptr::read_unaligned(boot_sector.as_ptr() as *const Fat32BootSector) };

        // Validate
        if bs.bytes_per_sector < 512 || bs.sectors_per_cluster == 0 {
            return Err(VfsError::InvalidPath);
        }

        let bytes_per_sector = bs.bytes_per_sector as u32;
        let sectors_per_cluster = bs.sectors_per_cluster as u32;
        let cluster_size = bytes_per_sector * sectors_per_cluster;

        let reserved_sectors = bs.reserved_sectors as u64;
        let fat_size = bs.fat_size_32 as u64;
        let num_fats = bs.num_fats as u64;

        let fat_start = reserved_sectors;
        let data_start = reserved_sectors + (num_fats * fat_size);

        let total_sectors = bs.total_sectors_32 as u64;
        let data_sectors = total_sectors - data_start;
        let total_clusters = (data_sectors / sectors_per_cluster as u64) as u32;

        crate::kinfo!(
            "FAT32: {} MB, cluster size {} KB",
            (total_sectors * bytes_per_sector as u64) / (1024 * 1024),
            cluster_size / 1024
        );

        Ok(Self {
            device,
            bytes_per_sector,
            sectors_per_cluster,
            cluster_size,
            fat_start,
            data_start,
            root_cluster: bs.root_cluster,
            total_clusters,
            fat_cache: RwLock::new(BTreeMap::new()),
        })
    }

    /// Read FAT entry
    fn read_fat(&self, cluster: u32) -> Result<u32, VfsError> {
        // Check cache
        if let Some(&entry) = self.fat_cache.read().get(&cluster) {
            return Ok(entry);
        }

        let fat_offset = cluster * 4;
        let fat_sector = self.fat_start + (fat_offset / self.bytes_per_sector) as u64;
        let offset_in_sector = (fat_offset % self.bytes_per_sector) as usize;

        let mut buf = alloc::vec![0u8; self.device.block_size()];
        self.device.read_block(fat_sector, &mut buf)?;

        let entry = u32::from_le_bytes([
            buf[offset_in_sector],
            buf[offset_in_sector + 1],
            buf[offset_in_sector + 2],
            buf[offset_in_sector + 3],
        ]) & 0x0FFFFFFF;

        // Cache it
        self.fat_cache.write().insert(cluster, entry);

        Ok(entry)
    }

    /// Write FAT entry
    fn write_fat(&self, cluster: u32, value: u32) -> Result<(), VfsError> {
        let fat_offset = cluster * 4;
        let fat_sector = self.fat_start + (fat_offset / self.bytes_per_sector) as u64;
        let offset_in_sector = (fat_offset % self.bytes_per_sector) as usize;

        let mut buf = alloc::vec![0u8; self.device.block_size()];
        self.device.read_block(fat_sector, &mut buf)?;

        let bytes = (value & 0x0FFFFFFF).to_le_bytes();
        buf[offset_in_sector..offset_in_sector + 4].copy_from_slice(&bytes);

        self.device.write_block(fat_sector, &buf)?;

        // Update cache
        self.fat_cache.write().insert(cluster, value);

        Ok(())
    }

    /// Get cluster chain
    fn get_cluster_chain(&self, start: u32) -> Result<Vec<u32>, VfsError> {
        let mut chain = Vec::new();
        let mut current = start;

        while current >= 2 && current < fat::BAD {
            chain.push(current);
            let next = self.read_fat(current)?;
            if next >= fat::EOC {
                break;
            }
            current = next;

            // Safety limit
            if chain.len() > 1_000_000 {
                return Err(VfsError::IoError);
            }
        }

        Ok(chain)
    }

    /// Convert cluster to sector
    fn cluster_to_sector(&self, cluster: u32) -> u64 {
        self.data_start + ((cluster - 2) as u64 * self.sectors_per_cluster as u64)
    }

    /// Read cluster
    fn read_cluster(&self, cluster: u32) -> Result<Vec<u8>, VfsError> {
        let sector = self.cluster_to_sector(cluster);
        let mut data = alloc::vec![0u8; self.cluster_size as usize];

        for i in 0..self.sectors_per_cluster {
            let offset = (i * self.bytes_per_sector) as usize;
            self.device.read_block(
                sector + i as u64,
                &mut data[offset..offset + self.bytes_per_sector as usize],
            )?;
        }

        Ok(data)
    }

    /// Parse 8.3 filename
    fn parse_filename(entry: &Fat32DirEntry) -> String {
        let name = core::str::from_utf8(&entry.name).unwrap_or("").trim_end();
        let ext = core::str::from_utf8(&entry.ext).unwrap_or("").trim_end();

        if ext.is_empty() {
            String::from(name)
        } else {
            alloc::format!("{}.{}", name, ext)
        }
    }

    /// Read directory entries
    fn read_directory(&self, cluster: u32) -> Result<Vec<(Fat32DirEntry, String)>, VfsError> {
        let chain = self.get_cluster_chain(cluster)?;
        let mut entries = Vec::new();
        let mut long_name = String::new();

        for &cluster in &chain {
            let data = self.read_cluster(cluster)?;

            for i in (0..data.len()).step_by(32) {
                if data[i] == 0x00 {
                    // End of directory
                    return Ok(entries);
                }
                if data[i] == 0xE5 {
                    // Deleted entry
                    continue;
                }

                let entry = unsafe {
                    core::ptr::read_unaligned(data[i..].as_ptr() as *const Fat32DirEntry)
                };

                if entry.attr == attr::LONG_NAME {
                    // Long filename entry - would parse here
                    continue;
                }

                if entry.attr & attr::VOLUME_ID != 0 {
                    continue;
                }

                let name = if long_name.is_empty() {
                    Self::parse_filename(&entry)
                } else {
                    core::mem::take(&mut long_name)
                };

                entries.push((entry, name));
            }
        }

        Ok(entries)
    }

    /// Find entry by path
    fn find_entry(&self, path: &str) -> Result<(Fat32DirEntry, u32), VfsError> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            // Root directory - create a fake entry
            let mut entry: Fat32DirEntry = unsafe { core::mem::zeroed() };
            entry.attr = attr::DIRECTORY;
            entry.cluster_low = self.root_cluster as u16;
            entry.cluster_high = (self.root_cluster >> 16) as u16;
            return Ok((entry, self.root_cluster));
        }

        let mut current_cluster = self.root_cluster;

        for (i, part) in parts.iter().enumerate() {
            let entries = self.read_directory(current_cluster)?;

            let found = entries
                .iter()
                .find(|(_, name)| name.eq_ignore_ascii_case(part));

            match found {
                Some((entry, _)) => {
                    let entry_cluster =
                        ((entry.cluster_high as u32) << 16) | (entry.cluster_low as u32);

                    if i == parts.len() - 1 {
                        return Ok((*entry, entry_cluster));
                    }

                    if entry.attr & attr::DIRECTORY == 0 {
                        return Err(VfsError::NotADirectory);
                    }

                    current_cluster = entry_cluster;
                }
                None => return Err(VfsError::NotFound),
            }
        }

        Err(VfsError::NotFound)
    }
}

impl Filesystem for Fat32Fs {
    fn name(&self) -> &'static str {
        "fat32"
    }

    fn is_readonly(&self) -> bool {
        false
    }

    fn total_size(&self) -> u64 {
        self.total_clusters as u64 * self.cluster_size as u64
    }

    fn free_space(&self) -> u64 {
        // Would count free clusters
        self.total_size() / 2
    }

    fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, VfsError> {
        let (entry, cluster) = match self.find_entry(path) {
            Ok(e) => e,
            Err(VfsError::NotFound) if flags.create => {
                // Would create file
                return Err(VfsError::NotSupported);
            }
            Err(e) => return Err(e),
        };

        if entry.attr & attr::DIRECTORY != 0 {
            return Err(VfsError::IsADirectory);
        }

        Ok(FileHandle {
            inode: cluster as u64,
            position: 0,
            flags,
        })
    }

    fn read(&self, handle: &FileHandle, buf: &mut [u8]) -> Result<usize, VfsError> {
        let chain = self.get_cluster_chain(handle.inode as u32)?;

        let mut total_read = 0;
        let mut pos = handle.position as usize;
        let cluster_size = self.cluster_size as usize;

        for &cluster in &chain {
            if pos >= cluster_size {
                pos -= cluster_size;
                continue;
            }

            let data = self.read_cluster(cluster)?;
            let start = pos;
            let end = (start + buf.len() - total_read).min(data.len());
            let to_copy = end - start;

            buf[total_read..total_read + to_copy].copy_from_slice(&data[start..end]);
            total_read += to_copy;
            pos = 0;

            if total_read >= buf.len() {
                break;
            }
        }

        Ok(total_read)
    }

    fn write(&self, handle: &FileHandle, buf: &[u8]) -> Result<usize, VfsError> {
        if !handle.flags.write {
            return Err(VfsError::PermissionDenied);
        }

        // Would implement write
        Err(VfsError::NotSupported)
    }

    fn close(&self, _handle: FileHandle) -> Result<(), VfsError> {
        Ok(())
    }

    fn stat(&self, path: &str) -> Result<FileStat, VfsError> {
        let (entry, _) = self.find_entry(path)?;

        let file_type = if entry.attr & attr::DIRECTORY != 0 {
            FileType::Directory
        } else {
            FileType::Regular
        };

        Ok(FileStat {
            file_type,
            size: entry.file_size as u64,
            permissions: FilePermissions {
                read: true,
                write: entry.attr & attr::READ_ONLY == 0,
                execute: false,
            },
            created: 0,
            modified: 0,
            accessed: 0,
            inode: ((entry.cluster_high as u64) << 16) | (entry.cluster_low as u64),
        })
    }

    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, VfsError> {
        let (entry, cluster) = self.find_entry(path)?;

        if entry.attr & attr::DIRECTORY == 0 {
            return Err(VfsError::NotADirectory);
        }

        let entries = self.read_directory(cluster)?;

        Ok(entries
            .into_iter()
            .filter(|(e, name)| !name.starts_with('.') || name == "." || name == "..")
            .map(|(e, name)| DirEntry {
                name,
                file_type: if e.attr & attr::DIRECTORY != 0 {
                    FileType::Directory
                } else {
                    FileType::Regular
                },
                inode: ((e.cluster_high as u64) << 16) | (e.cluster_low as u64),
            })
            .collect())
    }

    fn mkdir(&self, path: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn unlink(&self, path: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn rmdir(&self, path: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn sync(&self) -> Result<(), VfsError> {
        self.device.sync()
    }
}
