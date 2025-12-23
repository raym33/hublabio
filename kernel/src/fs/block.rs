//! Block Device Layer
//!
//! Abstract block device interface for storage.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::{Mutex, RwLock};

use super::BlockDevice;
use crate::vfs::VfsError;

/// Block device registry
static DEVICES: RwLock<BTreeMap<String, Arc<dyn BlockDevice>>> = RwLock::new(BTreeMap::new());

/// Register a block device
pub fn register(name: &str, device: Arc<dyn BlockDevice>) {
    DEVICES.write().insert(String::from(name), device);
    crate::kinfo!("Block device registered: {}", name);
}

/// Unregister a block device
pub fn unregister(name: &str) {
    DEVICES.write().remove(name);
}

/// Get a block device by name
pub fn get(name: &str) -> Option<Arc<dyn BlockDevice>> {
    DEVICES.read().get(name).cloned()
}

/// List all block devices
pub fn list() -> Vec<String> {
    DEVICES.read().keys().cloned().collect()
}

/// Memory-backed block device (for testing)
pub struct RamDisk {
    data: Mutex<Vec<u8>>,
    block_size: usize,
    blocks: u64,
}

impl RamDisk {
    /// Create a new RAM disk
    pub fn new(size_mb: usize) -> Self {
        let block_size = 512;
        let blocks = (size_mb * 1024 * 1024 / block_size) as u64;

        Self {
            data: Mutex::new(alloc::vec![0u8; size_mb * 1024 * 1024]),
            block_size,
            blocks,
        }
    }

    /// Create with initial data
    pub fn with_data(data: Vec<u8>, block_size: usize) -> Self {
        let blocks = (data.len() / block_size) as u64;

        Self {
            data: Mutex::new(data),
            block_size,
            blocks,
        }
    }
}

impl BlockDevice for RamDisk {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn total_blocks(&self) -> u64 {
        self.blocks
    }

    fn read_block(&self, block: u64, buf: &mut [u8]) -> Result<(), VfsError> {
        if block >= self.blocks {
            return Err(VfsError::IoError);
        }

        let data = self.data.lock();
        let offset = (block as usize) * self.block_size;
        let end = offset + self.block_size.min(buf.len());

        if end > data.len() {
            return Err(VfsError::IoError);
        }

        buf[..end - offset].copy_from_slice(&data[offset..end]);
        Ok(())
    }

    fn write_block(&self, block: u64, buf: &[u8]) -> Result<(), VfsError> {
        if block >= self.blocks {
            return Err(VfsError::IoError);
        }

        let mut data = self.data.lock();
        let offset = (block as usize) * self.block_size;
        let end = offset + self.block_size.min(buf.len());

        if end > data.len() {
            return Err(VfsError::IoError);
        }

        data[offset..end].copy_from_slice(&buf[..end - offset]);
        Ok(())
    }

    fn sync(&self) -> Result<(), VfsError> {
        Ok(())
    }
}

/// SD/MMC card block device
pub struct SdCard {
    base: usize,
    block_size: usize,
    total_blocks: u64,
}

impl SdCard {
    /// SD card controller base address (BCM2711)
    pub const BASE_BCM2711: usize = 0xFE340000;

    /// Create SD card device
    pub fn new(base: usize) -> Result<Self, VfsError> {
        // Would initialize SD card controller
        // For now, return error since not implemented
        Err(VfsError::NotSupported)
    }
}

impl BlockDevice for SdCard {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    fn read_block(&self, block: u64, buf: &mut [u8]) -> Result<(), VfsError> {
        // Would send CMD17 (READ_SINGLE_BLOCK)
        Err(VfsError::NotSupported)
    }

    fn write_block(&self, block: u64, buf: &[u8]) -> Result<(), VfsError> {
        // Would send CMD24 (WRITE_BLOCK)
        Err(VfsError::NotSupported)
    }

    fn sync(&self) -> Result<(), VfsError> {
        Ok(())
    }
}

/// NVMe block device
pub struct NvmeDevice {
    base: usize,
    namespace: u32,
    block_size: usize,
    total_blocks: u64,
}

impl NvmeDevice {
    /// Create NVMe device
    pub fn new(base: usize, namespace: u32) -> Result<Self, VfsError> {
        // Would initialize NVMe controller
        Err(VfsError::NotSupported)
    }
}

impl BlockDevice for NvmeDevice {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    fn read_block(&self, block: u64, buf: &mut [u8]) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn write_block(&self, block: u64, buf: &[u8]) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn sync(&self) -> Result<(), VfsError> {
        Ok(())
    }
}

/// Partition information
#[derive(Clone, Debug)]
pub struct Partition {
    pub device: String,
    pub number: u8,
    pub start_lba: u64,
    pub size_sectors: u64,
    pub partition_type: PartitionType,
    pub bootable: bool,
}

/// Partition types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionType {
    Empty,
    Fat12,
    Fat16,
    Fat32,
    Fat32Lba,
    ExtendedChs,
    ExtendedLba,
    Linux,
    LinuxSwap,
    LinuxLvm,
    Efi,
    Unknown(u8),
}

impl From<u8> for PartitionType {
    fn from(byte: u8) -> Self {
        match byte {
            0x00 => Self::Empty,
            0x01 => Self::Fat12,
            0x04 | 0x06 | 0x0E => Self::Fat16,
            0x0B => Self::Fat32,
            0x0C => Self::Fat32Lba,
            0x05 => Self::ExtendedChs,
            0x0F => Self::ExtendedLba,
            0x83 => Self::Linux,
            0x82 => Self::LinuxSwap,
            0x8E => Self::LinuxLvm,
            0xEF => Self::Efi,
            _ => Self::Unknown(byte),
        }
    }
}

/// Parse MBR partition table
pub fn parse_mbr(device: &dyn BlockDevice) -> Result<Vec<Partition>, VfsError> {
    let mut mbr = alloc::vec![0u8; 512];
    device.read_block(0, &mut mbr)?;

    // Check MBR signature
    if mbr[510] != 0x55 || mbr[511] != 0xAA {
        return Err(VfsError::InvalidPath);
    }

    let mut partitions = Vec::new();

    // Parse 4 primary partitions
    for i in 0..4 {
        let offset = 446 + i * 16;
        let entry = &mbr[offset..offset + 16];

        let partition_type = entry[4];
        if partition_type == 0 {
            continue;
        }

        let start_lba = u32::from_le_bytes([entry[8], entry[9], entry[10], entry[11]]) as u64;
        let size_sectors = u32::from_le_bytes([entry[12], entry[13], entry[14], entry[15]]) as u64;

        partitions.push(Partition {
            device: String::new(),
            number: (i + 1) as u8,
            start_lba,
            size_sectors,
            partition_type: PartitionType::from(partition_type),
            bootable: entry[0] == 0x80,
        });
    }

    Ok(partitions)
}

/// Parse GPT partition table
pub fn parse_gpt(device: &dyn BlockDevice) -> Result<Vec<Partition>, VfsError> {
    // GPT header is at LBA 1
    let mut header = alloc::vec![0u8; 512];
    device.read_block(1, &mut header)?;

    // Check signature "EFI PART"
    if &header[0..8] != b"EFI PART" {
        return Err(VfsError::InvalidPath);
    }

    let partition_entry_lba = u64::from_le_bytes(
        header[72..80].try_into().unwrap()
    );
    let num_entries = u32::from_le_bytes(
        header[80..84].try_into().unwrap()
    );
    let entry_size = u32::from_le_bytes(
        header[84..88].try_into().unwrap()
    ) as usize;

    let mut partitions = Vec::new();

    // Read partition entries
    let entries_per_block = 512 / entry_size;
    let blocks_to_read = (num_entries as usize + entries_per_block - 1) / entries_per_block;

    for block in 0..blocks_to_read as u64 {
        let mut buf = alloc::vec![0u8; 512];
        device.read_block(partition_entry_lba + block, &mut buf)?;

        for i in 0..entries_per_block {
            let offset = i * entry_size;
            if offset + entry_size > buf.len() {
                break;
            }

            let entry = &buf[offset..offset + entry_size];

            // Check if partition type GUID is zero (unused)
            if entry[0..16].iter().all(|&b| b == 0) {
                continue;
            }

            let start_lba = u64::from_le_bytes(entry[32..40].try_into().unwrap());
            let end_lba = u64::from_le_bytes(entry[40..48].try_into().unwrap());

            let partition_num = (block as usize * entries_per_block + i + 1) as u8;

            partitions.push(Partition {
                device: String::new(),
                number: partition_num,
                start_lba,
                size_sectors: end_lba - start_lba + 1,
                partition_type: PartitionType::Efi, // Simplified
                bootable: false,
            });
        }
    }

    Ok(partitions)
}

/// Initialize block device subsystem
pub fn init() {
    // Create a small RAM disk for testing
    let ramdisk = Arc::new(RamDisk::new(16)); // 16 MB
    register("ram0", ramdisk);

    crate::kprintln!("  Block device layer initialized");
}
