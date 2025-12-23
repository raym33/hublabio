//! Block Device Layer
//!
//! Abstract block device interface for storage.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
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

/// SD/MMC card block device wrapper
/// Connects the sdmmc driver to the block device abstraction
pub struct SdCardBlock {
    controller_id: u32,
    block_size: usize,
    total_blocks: u64,
}

impl SdCardBlock {
    /// SD card controller base address (BCM2711 - Raspberry Pi 4)
    pub const BASE_BCM2711: usize = 0xFE340000;

    /// SD card controller base address (QEMU virt)
    pub const BASE_QEMU: usize = 0x0900_0000;

    /// Create SD card block device from existing controller
    pub fn from_controller(controller_id: u32) -> Result<Self, VfsError> {
        // Get card info from controller
        let (_, total_sectors, _) =
            crate::drivers::sdmmc::get_card_info(controller_id).ok_or(VfsError::IoError)?;

        Ok(Self {
            controller_id,
            block_size: 512,
            total_blocks: total_sectors,
        })
    }

    /// Initialize SD card at given base address
    pub fn new(base: usize) -> Result<Self, VfsError> {
        // Register controller
        let controller_id =
            crate::drivers::sdmmc::register_controller(base).map_err(|_| VfsError::IoError)?;

        // Detect card
        crate::drivers::sdmmc::detect_card(controller_id).map_err(|_| VfsError::IoError)?;

        Self::from_controller(controller_id)
    }

    /// Get the underlying controller ID
    pub fn controller_id(&self) -> u32 {
        self.controller_id
    }
}

impl BlockDevice for SdCardBlock {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    fn read_block(&self, block: u64, buf: &mut [u8]) -> Result<(), VfsError> {
        if block >= self.total_blocks {
            return Err(VfsError::IoError);
        }

        if buf.len() < self.block_size {
            return Err(VfsError::IoError);
        }

        crate::drivers::sdmmc::read_sectors(self.controller_id, block, 1, buf)
            .map_err(|_| VfsError::IoError)
    }

    fn write_block(&self, block: u64, buf: &[u8]) -> Result<(), VfsError> {
        if block >= self.total_blocks {
            return Err(VfsError::IoError);
        }

        if buf.len() < self.block_size {
            return Err(VfsError::IoError);
        }

        crate::drivers::sdmmc::write_sectors(self.controller_id, block, 1, buf)
            .map_err(|_| VfsError::IoError)
    }

    fn sync(&self) -> Result<(), VfsError> {
        // SD card writes are synchronous
        Ok(())
    }
}

/// Read multiple blocks efficiently
impl SdCardBlock {
    /// Read multiple blocks at once
    pub fn read_blocks(
        &self,
        start_block: u64,
        count: u32,
        buf: &mut [u8],
    ) -> Result<(), VfsError> {
        if start_block + count as u64 > self.total_blocks {
            return Err(VfsError::IoError);
        }

        if buf.len() < (count as usize * self.block_size) {
            return Err(VfsError::IoError);
        }

        crate::drivers::sdmmc::read_sectors(self.controller_id, start_block, count, buf)
            .map_err(|_| VfsError::IoError)
    }

    /// Write multiple blocks at once
    pub fn write_blocks(&self, start_block: u64, count: u32, buf: &[u8]) -> Result<(), VfsError> {
        if start_block + count as u64 > self.total_blocks {
            return Err(VfsError::IoError);
        }

        if buf.len() < (count as usize * self.block_size) {
            return Err(VfsError::IoError);
        }

        crate::drivers::sdmmc::write_sectors(self.controller_id, start_block, count, buf)
            .map_err(|_| VfsError::IoError)
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

    let partition_entry_lba = u64::from_le_bytes(header[72..80].try_into().unwrap());
    let num_entries = u32::from_le_bytes(header[80..84].try_into().unwrap());
    let entry_size = u32::from_le_bytes(header[84..88].try_into().unwrap()) as usize;

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

// ============================================================================
// Block Cache
// ============================================================================

/// Block cache entry
struct CacheEntry {
    /// Block number
    block: u64,
    /// Device name
    device: String,
    /// Data
    data: Vec<u8>,
    /// Is dirty (needs write-back)
    dirty: bool,
    /// Last access time (ticks)
    last_access: u64,
    /// Reference count
    ref_count: u32,
}

/// Block cache
pub struct BlockCache {
    /// Cache entries
    entries: Mutex<Vec<CacheEntry>>,
    /// Maximum entries
    max_entries: usize,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
    writebacks: AtomicU64,
}

use core::sync::atomic::AtomicU64;

impl BlockCache {
    /// Create a new block cache
    pub const fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            writebacks: AtomicU64::new(0),
        }
    }

    /// Read a block through the cache
    pub fn read(
        &self,
        device_name: &str,
        device: &dyn BlockDevice,
        block: u64,
        buf: &mut [u8],
    ) -> Result<(), VfsError> {
        let mut entries = self.entries.lock();

        // Check cache
        for entry in entries.iter_mut() {
            if entry.device == device_name && entry.block == block {
                // Cache hit
                self.hits.fetch_add(1, Ordering::Relaxed);
                buf[..entry.data.len()].copy_from_slice(&entry.data);
                entry.last_access = get_ticks();
                entry.ref_count += 1;
                return Ok(());
            }
        }

        // Cache miss
        self.misses.fetch_add(1, Ordering::Relaxed);

        // Read from device
        device.read_block(block, buf)?;

        // Add to cache
        if entries.len() >= self.max_entries {
            // Evict least recently used entry
            self.evict_lru(&mut entries, device_name)?;
        }

        entries.push(CacheEntry {
            block,
            device: String::from(device_name),
            data: buf.to_vec(),
            dirty: false,
            last_access: get_ticks(),
            ref_count: 1,
        });

        Ok(())
    }

    /// Write a block through the cache
    pub fn write(
        &self,
        device_name: &str,
        device: &dyn BlockDevice,
        block: u64,
        buf: &[u8],
        write_through: bool,
    ) -> Result<(), VfsError> {
        let mut entries = self.entries.lock();

        // Check if block is in cache
        for entry in entries.iter_mut() {
            if entry.device == device_name && entry.block == block {
                // Update cache entry
                entry.data.clear();
                entry.data.extend_from_slice(buf);
                entry.last_access = get_ticks();
                entry.dirty = !write_through;

                if write_through {
                    device.write_block(block, buf)?;
                }

                return Ok(());
            }
        }

        // Not in cache - add it
        if entries.len() >= self.max_entries {
            self.evict_lru(&mut entries, device_name)?;
        }

        if write_through {
            device.write_block(block, buf)?;
        }

        entries.push(CacheEntry {
            block,
            device: String::from(device_name),
            data: buf.to_vec(),
            dirty: !write_through,
            last_access: get_ticks(),
            ref_count: 1,
        });

        Ok(())
    }

    /// Sync all dirty blocks to disk
    pub fn sync(&self, device_name: &str, device: &dyn BlockDevice) -> Result<(), VfsError> {
        let mut entries = self.entries.lock();

        for entry in entries.iter_mut() {
            if entry.device == device_name && entry.dirty {
                device.write_block(entry.block, &entry.data)?;
                entry.dirty = false;
                self.writebacks.fetch_add(1, Ordering::Relaxed);
            }
        }

        device.sync()
    }

    /// Sync all devices
    pub fn sync_all(&self) -> Result<(), VfsError> {
        let devices_list: Vec<String> = list();

        for name in devices_list {
            if let Some(device) = get(&name) {
                self.sync(&name, device.as_ref())?;
            }
        }

        Ok(())
    }

    /// Evict least recently used entry
    fn evict_lru(&self, entries: &mut Vec<CacheEntry>, device_name: &str) -> Result<(), VfsError> {
        if entries.is_empty() {
            return Ok(());
        }

        // Find LRU entry
        let mut lru_idx = 0;
        let mut lru_time = u64::MAX;

        for (i, entry) in entries.iter().enumerate() {
            if entry.ref_count == 0 && entry.last_access < lru_time {
                lru_time = entry.last_access;
                lru_idx = i;
            }
        }

        // Write back if dirty
        let entry = &entries[lru_idx];
        if entry.dirty {
            if let Some(device) = get(&entry.device) {
                device.write_block(entry.block, &entry.data)?;
                self.writebacks.fetch_add(1, Ordering::Relaxed);
            }
        }

        entries.remove(lru_idx);
        Ok(())
    }

    /// Invalidate cache entries for a device
    pub fn invalidate(&self, device_name: &str) {
        let mut entries = self.entries.lock();
        entries.retain(|e| e.device != device_name);
    }

    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.writebacks.load(Ordering::Relaxed),
        )
    }
}

/// Global block cache
static BLOCK_CACHE: BlockCache = BlockCache::new(256); // 256 blocks = 128KB with 512-byte blocks

/// Get ticks for LRU tracking
fn get_ticks() -> u64 {
    // Use system tick counter
    crate::time::monotonic_ns() / 1_000_000 // Convert to ms
}

/// Cached block read
pub fn cached_read(device_name: &str, block: u64, buf: &mut [u8]) -> Result<(), VfsError> {
    let device = get(device_name).ok_or(VfsError::NotFound)?;
    BLOCK_CACHE.read(device_name, device.as_ref(), block, buf)
}

/// Cached block write
pub fn cached_write(device_name: &str, block: u64, buf: &[u8]) -> Result<(), VfsError> {
    let device = get(device_name).ok_or(VfsError::NotFound)?;
    BLOCK_CACHE.write(device_name, device.as_ref(), block, buf, false)
}

/// Sync cached blocks
pub fn sync_cache(device_name: &str) -> Result<(), VfsError> {
    let device = get(device_name).ok_or(VfsError::NotFound)?;
    BLOCK_CACHE.sync(device_name, device.as_ref())
}

/// Sync all cached blocks
pub fn sync_all() -> Result<(), VfsError> {
    BLOCK_CACHE.sync_all()
}

/// Get cache statistics
pub fn cache_stats() -> (u64, u64, u64) {
    BLOCK_CACHE.stats()
}

// ============================================================================
// Partition Block Device
// ============================================================================

/// A partition view of a block device
pub struct PartitionDevice {
    /// Parent device name
    parent_device: String,
    /// Start LBA
    start_lba: u64,
    /// Number of sectors
    sectors: u64,
    /// Block size (inherited from parent)
    block_size: usize,
}

impl PartitionDevice {
    /// Create a partition device
    pub fn new(parent_name: &str, start_lba: u64, sectors: u64) -> Result<Self, VfsError> {
        let parent = get(parent_name).ok_or(VfsError::NotFound)?;

        Ok(Self {
            parent_device: String::from(parent_name),
            start_lba,
            sectors,
            block_size: parent.block_size(),
        })
    }
}

impl BlockDevice for PartitionDevice {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn total_blocks(&self) -> u64 {
        self.sectors
    }

    fn read_block(&self, block: u64, buf: &mut [u8]) -> Result<(), VfsError> {
        if block >= self.sectors {
            return Err(VfsError::IoError);
        }

        let parent = get(&self.parent_device).ok_or(VfsError::IoError)?;
        parent.read_block(self.start_lba + block, buf)
    }

    fn write_block(&self, block: u64, buf: &[u8]) -> Result<(), VfsError> {
        if block >= self.sectors {
            return Err(VfsError::IoError);
        }

        let parent = get(&self.parent_device).ok_or(VfsError::IoError)?;
        parent.write_block(self.start_lba + block, buf)
    }

    fn sync(&self) -> Result<(), VfsError> {
        let parent = get(&self.parent_device).ok_or(VfsError::IoError)?;
        parent.sync()
    }
}

/// Probe and register partitions for a device
pub fn probe_partitions(device_name: &str) -> Result<usize, VfsError> {
    let device = get(device_name).ok_or(VfsError::NotFound)?;

    // Try GPT first
    let partitions = match parse_gpt(device.as_ref()) {
        Ok(parts) => parts,
        Err(_) => {
            // Fall back to MBR
            parse_mbr(device.as_ref())?
        }
    };

    let count = partitions.len();

    for part in partitions {
        let part_name = alloc::format!("{}p{}", device_name, part.number);
        let part_dev = Arc::new(PartitionDevice::new(
            device_name,
            part.start_lba,
            part.size_sectors,
        )?);

        register(&part_name, part_dev);
        crate::kinfo!(
            "  Partition {}: {} sectors starting at LBA {}",
            part_name,
            part.size_sectors,
            part.start_lba
        );
    }

    Ok(count)
}

/// Initialize block device subsystem
pub fn init() {
    // Create a small RAM disk for testing
    let ramdisk = Arc::new(RamDisk::new(16)); // 16 MB
    register("ram0", ramdisk);

    crate::kprintln!("  Block device layer initialized");
    crate::kprintln!("  Block cache: {} entries", 256);
}
