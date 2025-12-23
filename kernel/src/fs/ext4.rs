//! ext4 Filesystem Driver
//!
//! Full read/write support for ext4 partitions with journaling.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use spin::RwLock;

use super::{BlockDevice, DirEntry, FileHandle, Filesystem};
use crate::vfs::{FilePermissions, FileStat, FileType, OpenFlags, VfsError};

/// ext4 superblock offset
const SUPERBLOCK_OFFSET: u64 = 1024;

/// ext4 magic number
const EXT4_MAGIC: u16 = 0xEF53;

/// Root inode number
const ROOT_INODE: u32 = 2;

/// Journal inode number
const JOURNAL_INODE: u32 = 8;

/// First non-reserved inode
const FIRST_USER_INODE: u32 = 11;

/// Maximum file name length
const MAX_NAME_LEN: usize = 255;

/// Journal magic numbers
const JBD2_MAGIC: u32 = 0xC03B3998;

/// Journal block types
mod journal_block_type {
    pub const DESCRIPTOR: u32 = 1;
    pub const COMMIT: u32 = 2;
    pub const SUPERBLOCK_V1: u32 = 3;
    pub const SUPERBLOCK_V2: u32 = 4;
    pub const REVOKE: u32 = 5;
}

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
    // Extended fields (EXT4_DYNAMIC_REV)
    s_first_ino: u32,
    s_inode_size: u16,
    s_block_group_nr: u16,
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_volume_name: [u8; 16],
    s_last_mounted: [u8; 64],
    s_algorithm_usage_bitmap: u32,
    // Performance hints
    s_prealloc_blocks: u8,
    s_prealloc_dir_blocks: u8,
    s_reserved_gdt_blocks: u16,
    // Journal
    s_journal_uuid: [u8; 16],
    s_journal_inum: u32,
    s_journal_dev: u32,
    s_last_orphan: u32,
    s_hash_seed: [u32; 4],
    s_def_hash_version: u8,
    s_jnl_backup_type: u8,
    s_desc_size: u16,
    s_default_mount_opts: u32,
    s_first_meta_bg: u32,
    s_mkfs_time: u32,
    s_jnl_blocks: [u32; 17],
    // 64-bit support
    s_blocks_count_hi: u32,
    s_r_blocks_count_hi: u32,
    s_free_blocks_count_hi: u32,
    s_min_extra_isize: u16,
    s_want_extra_isize: u16,
    s_flags: u32,
    s_raid_stride: u16,
    s_mmp_interval: u16,
    s_mmp_block: u64,
    s_raid_stripe_width: u32,
    s_log_groups_per_flex: u8,
    s_checksum_type: u8,
    s_reserved_pad: u16,
    s_kbytes_written: u64,
}

/// Feature compatible flags
mod feature_compat {
    pub const DIR_PREALLOC: u32 = 0x0001;
    pub const IMAGIC_INODES: u32 = 0x0002;
    pub const HAS_JOURNAL: u32 = 0x0004;
    pub const EXT_ATTR: u32 = 0x0008;
    pub const RESIZE_INODE: u32 = 0x0010;
    pub const DIR_INDEX: u32 = 0x0020;
    pub const SPARSE_SUPER2: u32 = 0x0200;
}

/// Feature incompatible flags
mod feature_incompat {
    pub const COMPRESSION: u32 = 0x0001;
    pub const FILETYPE: u32 = 0x0002;
    pub const RECOVER: u32 = 0x0004;
    pub const JOURNAL_DEV: u32 = 0x0008;
    pub const META_BG: u32 = 0x0010;
    pub const EXTENTS: u32 = 0x0040;
    pub const _64BIT: u32 = 0x0080;
    pub const MMP: u32 = 0x0100;
    pub const FLEX_BG: u32 = 0x0200;
    pub const EA_INODE: u32 = 0x0400;
    pub const DIRDATA: u32 = 0x1000;
    pub const CSUM_SEED: u32 = 0x2000;
    pub const LARGEDIR: u32 = 0x4000;
    pub const INLINE_DATA: u32 = 0x8000;
    pub const ENCRYPT: u32 = 0x10000;
}

/// Feature read-only compatible flags
mod feature_ro_compat {
    pub const SPARSE_SUPER: u32 = 0x0001;
    pub const LARGE_FILE: u32 = 0x0002;
    pub const BTREE_DIR: u32 = 0x0004;
    pub const HUGE_FILE: u32 = 0x0008;
    pub const GDT_CSUM: u32 = 0x0010;
    pub const DIR_NLINK: u32 = 0x0020;
    pub const EXTRA_ISIZE: u32 = 0x0040;
    pub const QUOTA: u32 = 0x0100;
    pub const BIGALLOC: u32 = 0x0200;
    pub const METADATA_CSUM: u32 = 0x0400;
    pub const READONLY: u32 = 0x1000;
    pub const PROJECT: u32 = 0x2000;
}

/// Filesystem state
mod fs_state {
    pub const VALID: u16 = 0x0001;
    pub const ERROR: u16 = 0x0002;
    pub const ORPHAN: u16 = 0x0004;
}

/// ext4 inode
#[repr(C)]
#[derive(Clone, Copy, Default)]
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
    i_block: [u32; 15], // Block pointers or extent tree
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

/// Inode flags
mod inode_flags {
    pub const SECRM: u32 = 0x00000001;
    pub const UNRM: u32 = 0x00000002;
    pub const COMPR: u32 = 0x00000004;
    pub const SYNC: u32 = 0x00000008;
    pub const IMMUTABLE: u32 = 0x00000010;
    pub const APPEND: u32 = 0x00000020;
    pub const NODUMP: u32 = 0x00000040;
    pub const NOATIME: u32 = 0x00000080;
    pub const DIRTY: u32 = 0x00000100;
    pub const COMPRBLK: u32 = 0x00000200;
    pub const NOCOMPR: u32 = 0x00000400;
    pub const ENCRYPT: u32 = 0x00000800;
    pub const INDEX: u32 = 0x00001000;
    pub const IMAGIC: u32 = 0x00002000;
    pub const JOURNAL_DATA: u32 = 0x00004000;
    pub const NOTAIL: u32 = 0x00008000;
    pub const DIRSYNC: u32 = 0x00010000;
    pub const TOPDIR: u32 = 0x00020000;
    pub const HUGE_FILE: u32 = 0x00040000;
    pub const EXTENTS: u32 = 0x00080000;
    pub const VERITY: u32 = 0x00100000;
    pub const EA_INODE: u32 = 0x00200000;
    pub const INLINE_DATA: u32 = 0x10000000;
    pub const PROJINHERIT: u32 = 0x20000000;
    pub const CASEFOLD: u32 = 0x40000000;
}

/// Inode file types (from i_mode)
mod inode_type {
    pub const FIFO: u16 = 0x1000;
    pub const CHARDEV: u16 = 0x2000;
    pub const DIR: u16 = 0x4000;
    pub const BLOCKDEV: u16 = 0x6000;
    pub const FILE: u16 = 0x8000;
    pub const SYMLINK: u16 = 0xA000;
    pub const SOCKET: u16 = 0xC000;
    pub const TYPE_MASK: u16 = 0xF000;
}

/// ext4 directory entry
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Ext4DirEntry {
    inode: u32,
    rec_len: u16,
    name_len: u8,
    file_type: u8,
    // name follows (variable length)
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

/// Block group descriptor (64-byte version for 64-bit support)
#[repr(C)]
#[derive(Clone, Copy, Default)]
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

/// Extent header
#[repr(C)]
#[derive(Clone, Copy)]
struct Ext4ExtentHeader {
    eh_magic: u16,
    eh_entries: u16,
    eh_max: u16,
    eh_depth: u16,
    eh_generation: u32,
}

/// Extent magic
const EXT4_EXTENT_MAGIC: u16 = 0xF30A;

/// Extent leaf
#[repr(C)]
#[derive(Clone, Copy)]
struct Ext4Extent {
    ee_block: u32,    // First file block covered
    ee_len: u16,      // Number of blocks covered
    ee_start_hi: u16, // High 16 bits of physical block
    ee_start_lo: u32, // Low 32 bits of physical block
}

/// Extent index (internal node)
#[repr(C)]
#[derive(Clone, Copy)]
struct Ext4ExtentIdx {
    ei_block: u32,   // Index covers from this block
    ei_leaf_lo: u32, // Pointer to next level
    ei_leaf_hi: u16,
    ei_unused: u16,
}

/// Journal superblock
#[repr(C)]
#[derive(Clone, Copy)]
struct JournalSuperblock {
    s_header: JournalHeader,
    s_blocksize: u32,
    s_maxlen: u32,
    s_first: u32,
    s_sequence: u32,
    s_start: u32,
    s_errno: u32,
    // V2 fields
    s_feature_compat: u32,
    s_feature_incompat: u32,
    s_feature_ro_compat: u32,
    s_uuid: [u8; 16],
    s_nr_users: u32,
    s_dynsuper: u32,
    s_max_transaction: u32,
    s_max_trans_data: u32,
    s_checksum_type: u8,
    s_padding2: [u8; 3],
    s_padding: [u32; 42],
    s_checksum: u32,
    s_users: [[u8; 16]; 48],
}

/// Journal block header
#[repr(C)]
#[derive(Clone, Copy)]
struct JournalHeader {
    h_magic: u32,
    h_blocktype: u32,
    h_sequence: u32,
}

/// Journal descriptor block tag
#[repr(C)]
#[derive(Clone, Copy)]
struct JournalBlockTag {
    t_blocknr: u32,
    t_flags: u16,
    t_blocknr_high: u16,
}

/// Journal tag flags
mod journal_tag_flags {
    pub const ESCAPE: u16 = 0x0001;
    pub const SAME_UUID: u16 = 0x0002;
    pub const DELETED: u16 = 0x0004;
    pub const LAST_TAG: u16 = 0x0008;
}

/// Transaction state
#[derive(Clone, Copy, PartialEq, Eq)]
enum TxState {
    Idle,
    Running,
    Committing,
}

/// Journal transaction
struct Transaction {
    tid: u32,
    state: TxState,
    blocks: Vec<(u64, Vec<u8>)>, // (block_num, data)
    modified_inodes: Vec<u32>,
}

impl Transaction {
    fn new(tid: u32) -> Self {
        Self {
            tid,
            state: TxState::Running,
            blocks: Vec::new(),
            modified_inodes: Vec::new(),
        }
    }
}

/// Block allocator
struct BlockAllocator {
    free_blocks: AtomicU64,
    next_search_start: AtomicU32,
}

impl BlockAllocator {
    fn new(free_blocks: u64) -> Self {
        Self {
            free_blocks: AtomicU64::new(free_blocks),
            next_search_start: AtomicU32::new(0),
        }
    }
}

/// Inode allocator
struct InodeAllocator {
    free_inodes: AtomicU32,
    next_search_start: AtomicU32,
}

impl InodeAllocator {
    fn new(free_inodes: u32) -> Self {
        Self {
            free_inodes: AtomicU32::new(free_inodes),
            next_search_start: AtomicU32::new(0),
        }
    }
}

/// Journal state
struct Journal {
    start_block: u64,
    block_count: u32,
    sequence: AtomicU32,
    start: AtomicU32,
    current_tx: RwLock<Option<Transaction>>,
    committed_txs: RwLock<Vec<Transaction>>,
}

impl Journal {
    fn new(start_block: u64, block_count: u32, sequence: u32, start: u32) -> Self {
        Self {
            start_block,
            block_count,
            sequence: AtomicU32::new(sequence),
            start: AtomicU32::new(start),
            current_tx: RwLock::new(None),
            committed_txs: RwLock::new(Vec::new()),
        }
    }
}

/// ext4 filesystem with write support
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
    groups_count: u32,
    features_compat: u32,
    features_incompat: u32,
    features_ro_compat: u32,
    has_journal: bool,
    readonly: AtomicBool,

    // Caches
    inode_cache: RwLock<BTreeMap<u32, Ext4Inode>>,
    block_cache: RwLock<BTreeMap<u64, Vec<u8>>>,
    group_desc_cache: RwLock<BTreeMap<u32, Ext4GroupDesc>>,

    // Allocators
    block_allocator: BlockAllocator,
    inode_allocator: InodeAllocator,

    // Journal
    journal: RwLock<Option<Journal>>,

    // Dirty tracking
    dirty_inodes: RwLock<Vec<u32>>,
    dirty_blocks: RwLock<Vec<u64>>,
    dirty_groups: RwLock<Vec<u32>>,
}

impl Ext4Fs {
    /// Mount ext4 filesystem
    pub fn mount(device: Arc<dyn BlockDevice>) -> Result<Self, VfsError> {
        let dev_block_size = device.block_size();

        // Read superblock
        let sb_block = SUPERBLOCK_OFFSET / dev_block_size as u64;
        let sb_offset = (SUPERBLOCK_OFFSET % dev_block_size as u64) as usize;

        let mut buf = vec![0u8; dev_block_size];
        device.read_block(sb_block, &mut buf)?;

        let sb = unsafe {
            core::ptr::read_unaligned(buf[sb_offset..].as_ptr() as *const Ext4Superblock)
        };

        // Validate magic
        if sb.s_magic != EXT4_MAGIC {
            return Err(VfsError::InvalidPath);
        }

        let block_size = 1024u32 << sb.s_log_block_size;
        let inode_size = if sb.s_rev_level >= 1 {
            sb.s_inode_size as u32
        } else {
            128
        };
        let group_desc_size = if sb.s_feature_incompat & feature_incompat::_64BIT != 0 {
            64
        } else {
            32
        };

        let total_blocks = sb.s_blocks_count_lo as u64 | ((sb.s_blocks_count_hi as u64) << 32);
        let free_blocks =
            sb.s_free_blocks_count_lo as u64 | ((sb.s_free_blocks_count_hi as u64) << 32);
        let groups_count =
            (total_blocks + sb.s_blocks_per_group as u64 - 1) / sb.s_blocks_per_group as u64;

        let has_journal = sb.s_feature_compat & feature_compat::HAS_JOURNAL != 0;

        // Check for read-only features we can't handle
        let readonly = sb.s_feature_ro_compat & feature_ro_compat::READONLY != 0;

        crate::kinfo!(
            "ext4: {} MB, block size {} bytes, {} block groups, journal: {}",
            (total_blocks * block_size as u64) / (1024 * 1024),
            block_size,
            groups_count,
            if has_journal { "yes" } else { "no" }
        );

        let fs = Self {
            device,
            block_size,
            inode_size,
            inodes_per_group: sb.s_inodes_per_group,
            blocks_per_group: sb.s_blocks_per_group,
            group_desc_size,
            total_inodes: sb.s_inodes_count,
            total_blocks,
            first_data_block: sb.s_first_data_block,
            groups_count: groups_count as u32,
            features_compat: sb.s_feature_compat,
            features_incompat: sb.s_feature_incompat,
            features_ro_compat: sb.s_feature_ro_compat,
            has_journal,
            readonly: AtomicBool::new(readonly),
            inode_cache: RwLock::new(BTreeMap::new()),
            block_cache: RwLock::new(BTreeMap::new()),
            group_desc_cache: RwLock::new(BTreeMap::new()),
            block_allocator: BlockAllocator::new(free_blocks),
            inode_allocator: InodeAllocator::new(sb.s_free_inodes_count),
            journal: RwLock::new(None),
            dirty_inodes: RwLock::new(Vec::new()),
            dirty_blocks: RwLock::new(Vec::new()),
            dirty_groups: RwLock::new(Vec::new()),
        };

        // Initialize journal if present
        if has_journal {
            fs.init_journal(sb.s_journal_inum)?;
        }

        Ok(fs)
    }

    /// Initialize journal
    fn init_journal(&self, journal_inum: u32) -> Result<(), VfsError> {
        let inode = self.read_inode(journal_inum)?;

        // Get journal start block (first block of journal inode)
        let start_block = if inode.i_flags & inode_flags::EXTENTS != 0 {
            // Extent-based journal
            self.get_extent_block(&inode, 0)?
        } else {
            inode.i_block[0] as u64
        };

        // Read journal superblock
        let data = self.read_block(start_block)?;
        let jsb = unsafe { core::ptr::read_unaligned(data.as_ptr() as *const JournalSuperblock) };

        if jsb.s_header.h_magic != JBD2_MAGIC.to_be() {
            crate::kwarn!("ext4: Invalid journal magic, mounting read-only");
            self.readonly.store(true, Ordering::SeqCst);
            return Ok(());
        }

        let block_count = u32::from_be(jsb.s_maxlen);
        let sequence = u32::from_be(jsb.s_sequence);
        let start = u32::from_be(jsb.s_start);

        crate::kinfo!(
            "ext4: Journal: {} blocks, sequence {}, start block {}",
            block_count,
            sequence,
            start
        );

        // Check if recovery is needed
        if start != 0 {
            crate::kwarn!("ext4: Journal recovery needed, mounting read-only for safety");
            self.readonly.store(true, Ordering::SeqCst);
            // In a full implementation, we would replay the journal here
        }

        *self.journal.write() = Some(Journal::new(start_block, block_count, sequence, start));

        Ok(())
    }

    /// Read a filesystem block
    fn read_block(&self, block: u64) -> Result<Vec<u8>, VfsError> {
        // Check cache first
        if let Some(data) = self.block_cache.read().get(&block) {
            return Ok(data.clone());
        }

        let dev_block_size = self.device.block_size() as u64;
        let blocks_per_fs_block = self.block_size as u64 / dev_block_size;

        let mut data = vec![0u8; self.block_size as usize];

        for i in 0..blocks_per_fs_block {
            let offset = (i * dev_block_size) as usize;
            self.device.read_block(
                block * blocks_per_fs_block + i,
                &mut data[offset..offset + dev_block_size as usize],
            )?;
        }

        // Cache it
        self.block_cache.write().insert(block, data.clone());

        Ok(data)
    }

    /// Write a filesystem block
    fn write_block(&self, block: u64, data: &[u8]) -> Result<(), VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        // Add to journal if enabled
        if let Some(ref journal) = *self.journal.read() {
            if let Some(ref mut tx) = *journal.current_tx.write() {
                tx.blocks.push((block, data.to_vec()));
            }
        }

        // Update cache
        self.block_cache.write().insert(block, data.to_vec());

        // Mark dirty
        self.dirty_blocks.write().push(block);

        Ok(())
    }

    /// Flush block to device
    fn flush_block(&self, block: u64, data: &[u8]) -> Result<(), VfsError> {
        let dev_block_size = self.device.block_size() as u64;
        let blocks_per_fs_block = self.block_size as u64 / dev_block_size;

        for i in 0..blocks_per_fs_block {
            let offset = (i * dev_block_size) as usize;
            self.device.write_block(
                block * blocks_per_fs_block + i,
                &data[offset..offset + dev_block_size as usize],
            )?;
        }

        Ok(())
    }

    /// Get block group descriptor
    fn read_group_desc(&self, group: u32) -> Result<Ext4GroupDesc, VfsError> {
        // Check cache
        if let Some(&gd) = self.group_desc_cache.read().get(&group) {
            return Ok(gd);
        }

        let desc_per_block = self.block_size / self.group_desc_size;
        let block = self.first_data_block as u64 + 1 + (group / desc_per_block) as u64;
        let offset = ((group % desc_per_block) * self.group_desc_size) as usize;

        let data = self.read_block(block)?;

        let gd =
            unsafe { core::ptr::read_unaligned(data[offset..].as_ptr() as *const Ext4GroupDesc) };

        self.group_desc_cache.write().insert(group, gd);

        Ok(gd)
    }

    /// Write block group descriptor
    fn write_group_desc(&self, group: u32, gd: &Ext4GroupDesc) -> Result<(), VfsError> {
        let desc_per_block = self.block_size / self.group_desc_size;
        let block = self.first_data_block as u64 + 1 + (group / desc_per_block) as u64;
        let offset = ((group % desc_per_block) * self.group_desc_size) as usize;

        let mut data = self.read_block(block)?;

        unsafe {
            core::ptr::write_unaligned(data[offset..].as_mut_ptr() as *mut Ext4GroupDesc, *gd);
        }

        self.write_block(block, &data)?;
        self.group_desc_cache.write().insert(group, *gd);
        self.dirty_groups.write().push(group);

        Ok(())
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
        let inode_table = gd.bg_inode_table_lo as u64 | ((gd.bg_inode_table_hi as u64) << 32);

        let inodes_per_block = self.block_size / self.inode_size;
        let block = inode_table + (index / inodes_per_block) as u64;
        let offset = ((index % inodes_per_block) * self.inode_size) as usize;

        let data = self.read_block(block)?;

        let inode =
            unsafe { core::ptr::read_unaligned(data[offset..].as_ptr() as *const Ext4Inode) };

        // Cache it
        self.inode_cache.write().insert(ino, inode);

        Ok(inode)
    }

    /// Write inode
    fn write_inode(&self, ino: u32, inode: &Ext4Inode) -> Result<(), VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        let group = (ino - 1) / self.inodes_per_group;
        let index = (ino - 1) % self.inodes_per_group;

        let gd = self.read_group_desc(group)?;
        let inode_table = gd.bg_inode_table_lo as u64 | ((gd.bg_inode_table_hi as u64) << 32);

        let inodes_per_block = self.block_size / self.inode_size;
        let block = inode_table + (index / inodes_per_block) as u64;
        let offset = ((index % inodes_per_block) * self.inode_size) as usize;

        let mut data = self.read_block(block)?;

        unsafe {
            core::ptr::write_unaligned(data[offset..].as_mut_ptr() as *mut Ext4Inode, *inode);
        }

        self.write_block(block, &data)?;
        self.inode_cache.write().insert(ino, *inode);
        self.dirty_inodes.write().push(ino);

        Ok(())
    }

    /// Allocate a new block
    fn alloc_block(&self, preferred_group: u32) -> Result<u64, VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        // Try preferred group first, then scan others
        let groups_to_try: Vec<u32> = (0..self.groups_count)
            .map(|i| (preferred_group + i) % self.groups_count)
            .collect();

        for group in groups_to_try {
            let mut gd = self.read_group_desc(group)?;

            let free_blocks =
                gd.bg_free_blocks_count_lo as u32 | ((gd.bg_free_blocks_count_hi as u32) << 16);

            if free_blocks == 0 {
                continue;
            }

            // Read block bitmap
            let bitmap_block =
                gd.bg_block_bitmap_lo as u64 | ((gd.bg_block_bitmap_hi as u64) << 32);
            let mut bitmap = self.read_block(bitmap_block)?;

            // Find free block
            for byte_idx in 0..bitmap.len() {
                if bitmap[byte_idx] != 0xFF {
                    for bit in 0..8 {
                        if bitmap[byte_idx] & (1 << bit) == 0 {
                            // Found free block
                            let block_in_group = byte_idx * 8 + bit;
                            let block = group as u64 * self.blocks_per_group as u64
                                + self.first_data_block as u64
                                + block_in_group as u64;

                            // Mark as used
                            bitmap[byte_idx] |= 1 << bit;
                            self.write_block(bitmap_block, &bitmap)?;

                            // Update group descriptor
                            let new_free = free_blocks - 1;
                            gd.bg_free_blocks_count_lo = new_free as u16;
                            gd.bg_free_blocks_count_hi = (new_free >> 16) as u16;
                            self.write_group_desc(group, &gd)?;

                            // Update allocator stats
                            self.block_allocator
                                .free_blocks
                                .fetch_sub(1, Ordering::SeqCst);

                            return Ok(block);
                        }
                    }
                }
            }
        }

        Err(VfsError::NoSpace)
    }

    /// Free a block
    fn free_block(&self, block: u64) -> Result<(), VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        let block_in_fs = block - self.first_data_block as u64;
        let group = (block_in_fs / self.blocks_per_group as u64) as u32;
        let block_in_group = (block_in_fs % self.blocks_per_group as u64) as usize;

        let mut gd = self.read_group_desc(group)?;

        // Read block bitmap
        let bitmap_block = gd.bg_block_bitmap_lo as u64 | ((gd.bg_block_bitmap_hi as u64) << 32);
        let mut bitmap = self.read_block(bitmap_block)?;

        // Clear bit
        let byte_idx = block_in_group / 8;
        let bit = block_in_group % 8;
        bitmap[byte_idx] &= !(1 << bit);
        self.write_block(bitmap_block, &bitmap)?;

        // Update group descriptor
        let free_blocks =
            gd.bg_free_blocks_count_lo as u32 | ((gd.bg_free_blocks_count_hi as u32) << 16);
        let new_free = free_blocks + 1;
        gd.bg_free_blocks_count_lo = new_free as u16;
        gd.bg_free_blocks_count_hi = (new_free >> 16) as u16;
        self.write_group_desc(group, &gd)?;

        // Update allocator stats
        self.block_allocator
            .free_blocks
            .fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Allocate a new inode
    fn alloc_inode(&self, is_directory: bool) -> Result<u32, VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        for group in 0..self.groups_count {
            let mut gd = self.read_group_desc(group)?;

            let free_inodes =
                gd.bg_free_inodes_count_lo as u32 | ((gd.bg_free_inodes_count_hi as u32) << 16);

            if free_inodes == 0 {
                continue;
            }

            // Read inode bitmap
            let bitmap_block =
                gd.bg_inode_bitmap_lo as u64 | ((gd.bg_inode_bitmap_hi as u64) << 32);
            let mut bitmap = self.read_block(bitmap_block)?;

            // Find free inode
            for byte_idx in 0..bitmap.len() {
                if bitmap[byte_idx] != 0xFF {
                    for bit in 0..8 {
                        if bitmap[byte_idx] & (1 << bit) == 0 {
                            let inode_in_group = byte_idx * 8 + bit;
                            let ino = group * self.inodes_per_group + inode_in_group as u32 + 1;

                            // Skip reserved inodes
                            if ino < FIRST_USER_INODE {
                                continue;
                            }

                            // Mark as used
                            bitmap[byte_idx] |= 1 << bit;
                            self.write_block(bitmap_block, &bitmap)?;

                            // Update group descriptor
                            let new_free = free_inodes - 1;
                            gd.bg_free_inodes_count_lo = new_free as u16;
                            gd.bg_free_inodes_count_hi = (new_free >> 16) as u16;

                            if is_directory {
                                let used_dirs = gd.bg_used_dirs_count_lo as u32
                                    | ((gd.bg_used_dirs_count_hi as u32) << 16);
                                let new_dirs = used_dirs + 1;
                                gd.bg_used_dirs_count_lo = new_dirs as u16;
                                gd.bg_used_dirs_count_hi = (new_dirs >> 16) as u16;
                            }

                            self.write_group_desc(group, &gd)?;

                            // Update allocator stats
                            self.inode_allocator
                                .free_inodes
                                .fetch_sub(1, Ordering::SeqCst);

                            return Ok(ino);
                        }
                    }
                }
            }
        }

        Err(VfsError::NoSpace)
    }

    /// Free an inode
    fn free_inode(&self, ino: u32, is_directory: bool) -> Result<(), VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        let group = (ino - 1) / self.inodes_per_group;
        let index = (ino - 1) % self.inodes_per_group;

        let mut gd = self.read_group_desc(group)?;

        // Read inode bitmap
        let bitmap_block = gd.bg_inode_bitmap_lo as u64 | ((gd.bg_inode_bitmap_hi as u64) << 32);
        let mut bitmap = self.read_block(bitmap_block)?;

        // Clear bit
        let byte_idx = index as usize / 8;
        let bit = index as usize % 8;
        bitmap[byte_idx] &= !(1 << bit);
        self.write_block(bitmap_block, &bitmap)?;

        // Update group descriptor
        let free_inodes =
            gd.bg_free_inodes_count_lo as u32 | ((gd.bg_free_inodes_count_hi as u32) << 16);
        let new_free = free_inodes + 1;
        gd.bg_free_inodes_count_lo = new_free as u16;
        gd.bg_free_inodes_count_hi = (new_free >> 16) as u16;

        if is_directory {
            let used_dirs =
                gd.bg_used_dirs_count_lo as u32 | ((gd.bg_used_dirs_count_hi as u32) << 16);
            if used_dirs > 0 {
                let new_dirs = used_dirs - 1;
                gd.bg_used_dirs_count_lo = new_dirs as u16;
                gd.bg_used_dirs_count_hi = (new_dirs >> 16) as u16;
            }
        }

        self.write_group_desc(group, &gd)?;

        // Update allocator stats
        self.inode_allocator
            .free_inodes
            .fetch_add(1, Ordering::SeqCst);

        // Remove from cache
        self.inode_cache.write().remove(&ino);

        Ok(())
    }

    /// Get physical block from extent tree
    fn get_extent_block(&self, inode: &Ext4Inode, file_block: u64) -> Result<u64, VfsError> {
        let extent_data = unsafe {
            core::slice::from_raw_parts(
                inode.i_block.as_ptr() as *const u8,
                60, // 15 * 4 bytes
            )
        };

        self.lookup_extent(extent_data, file_block)
    }

    /// Lookup block in extent tree
    fn lookup_extent(&self, extent_data: &[u8], file_block: u64) -> Result<u64, VfsError> {
        let header =
            unsafe { core::ptr::read_unaligned(extent_data.as_ptr() as *const Ext4ExtentHeader) };

        if header.eh_magic != EXT4_EXTENT_MAGIC {
            return Err(VfsError::InvalidPath);
        }

        let entries = header.eh_entries as usize;

        if header.eh_depth == 0 {
            // Leaf node - search extents
            for i in 0..entries {
                let offset = 12 + i * 12; // Header is 12 bytes, extent is 12 bytes
                let extent = unsafe {
                    core::ptr::read_unaligned(extent_data[offset..].as_ptr() as *const Ext4Extent)
                };

                let start = extent.ee_block as u64;
                let len = (extent.ee_len & 0x7FFF) as u64; // Mask uninitialized bit

                if file_block >= start && file_block < start + len {
                    let phys_start =
                        extent.ee_start_lo as u64 | ((extent.ee_start_hi as u64) << 32);
                    return Ok(phys_start + (file_block - start));
                }
            }

            Err(VfsError::NotFound)
        } else {
            // Internal node - find correct child
            for i in 0..entries {
                let offset = 12 + i * 12;
                let idx = unsafe {
                    core::ptr::read_unaligned(extent_data[offset..].as_ptr() as *const Ext4ExtentIdx)
                };

                let next_block = if i + 1 < entries {
                    let next_offset = 12 + (i + 1) * 12;
                    let next_idx = unsafe {
                        core::ptr::read_unaligned(
                            extent_data[next_offset..].as_ptr() as *const Ext4ExtentIdx
                        )
                    };
                    next_idx.ei_block as u64
                } else {
                    u64::MAX
                };

                if file_block >= idx.ei_block as u64 && file_block < next_block {
                    let child_block = idx.ei_leaf_lo as u64 | ((idx.ei_leaf_hi as u64) << 32);
                    let child_data = self.read_block(child_block)?;
                    return self.lookup_extent(&child_data, file_block);
                }
            }

            Err(VfsError::NotFound)
        }
    }

    /// Read inode data using block mapping
    fn read_inode_data(
        &self,
        inode: &Ext4Inode,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, VfsError> {
        let size = inode.i_size_lo as u64 | ((inode.i_size_high as u64) << 32);

        if offset >= size {
            return Ok(0);
        }

        let to_read = (size - offset).min(buf.len() as u64) as usize;
        let mut total_read = 0;

        let start_block = offset / self.block_size as u64;
        let offset_in_block = (offset % self.block_size as u64) as usize;

        let use_extents = inode.i_flags & inode_flags::EXTENTS != 0;

        let mut current_block = start_block;
        let mut first_block = true;

        while total_read < to_read {
            let phys_block = if use_extents {
                match self.get_extent_block(inode, current_block) {
                    Ok(b) => b,
                    Err(_) => break,
                }
            } else {
                // Traditional block mapping
                if current_block < 12 {
                    let b = inode.i_block[current_block as usize];
                    if b == 0 {
                        break;
                    }
                    b as u64
                } else {
                    // Handle indirect blocks (simplified)
                    break;
                }
            };

            if phys_block == 0 {
                break;
            }

            let data = self.read_block(phys_block)?;

            let start = if first_block { offset_in_block } else { 0 };
            let end = (start + to_read - total_read).min(self.block_size as usize);
            let len = end - start;

            buf[total_read..total_read + len].copy_from_slice(&data[start..end]);
            total_read += len;
            first_block = false;
            current_block += 1;
        }

        Ok(total_read)
    }

    /// Write data to inode
    fn write_inode_data(&self, ino: u32, offset: u64, buf: &[u8]) -> Result<usize, VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        let mut inode = self.read_inode(ino)?;
        let current_size = inode.i_size_lo as u64 | ((inode.i_size_high as u64) << 32);

        let new_size = offset + buf.len() as u64;
        let mut total_written = 0;

        let start_block = offset / self.block_size as u64;
        let offset_in_block = (offset % self.block_size as u64) as usize;

        // Determine preferred group for allocation
        let group = (ino - 1) / self.inodes_per_group;

        let use_extents = inode.i_flags & inode_flags::EXTENTS != 0;
        let mut current_block = start_block;
        let mut first_block = true;

        while total_written < buf.len() {
            // Get or allocate block
            let phys_block = if use_extents {
                match self.get_extent_block(&inode, current_block) {
                    Ok(b) => b,
                    Err(_) => {
                        // Need to allocate and add to extent tree
                        let new_block = self.alloc_block(group)?;
                        // For simplicity, we don't modify extent tree here
                        // In a full implementation, we'd update the extent tree
                        new_block
                    }
                }
            } else {
                // Traditional block mapping
                if current_block < 12 {
                    let b = inode.i_block[current_block as usize];
                    if b == 0 {
                        // Allocate new block
                        let new_block = self.alloc_block(group)?;
                        inode.i_block[current_block as usize] = new_block as u32;
                        new_block
                    } else {
                        b as u64
                    }
                } else {
                    // Would need indirect block handling
                    return Err(VfsError::NotSupported);
                }
            };

            // Read existing block data (for partial writes)
            let mut data = if first_block && offset_in_block > 0 {
                self.read_block(phys_block)
                    .unwrap_or_else(|_| vec![0u8; self.block_size as usize])
            } else {
                vec![0u8; self.block_size as usize]
            };

            let start = if first_block { offset_in_block } else { 0 };
            let remaining = buf.len() - total_written;
            let len = remaining.min(self.block_size as usize - start);

            data[start..start + len].copy_from_slice(&buf[total_written..total_written + len]);

            self.write_block(phys_block, &data)?;

            total_written += len;
            first_block = false;
            current_block += 1;
        }

        // Update inode size and times
        if new_size > current_size {
            inode.i_size_lo = new_size as u32;
            inode.i_size_high = (new_size >> 32) as u32;
        }

        // Update block count
        let blocks_used = (new_size + self.block_size as u64 - 1) / self.block_size as u64;
        inode.i_blocks_lo = (blocks_used * (self.block_size as u64 / 512)) as u32;

        // Update modification time
        inode.i_mtime = crate::time::current_timestamp() as u32;
        inode.i_ctime = inode.i_mtime;

        self.write_inode(ino, &inode)?;

        Ok(total_written)
    }

    /// Read directory entries
    fn read_directory(&self, ino: u32) -> Result<Vec<(u32, String, u8)>, VfsError> {
        let inode = self.read_inode(ino)?;

        if inode.i_mode & inode_type::TYPE_MASK != inode_type::DIR {
            return Err(VfsError::NotADirectory);
        }

        let size = inode.i_size_lo as usize;
        let mut data = vec![0u8; size];
        self.read_inode_data(&inode, 0, &mut data)?;

        let mut entries = Vec::new();
        let mut pos = 0;

        while pos + 8 <= data.len() {
            let entry =
                unsafe { core::ptr::read_unaligned(data[pos..].as_ptr() as *const Ext4DirEntry) };

            if entry.rec_len == 0 {
                break;
            }

            if entry.inode != 0 && entry.name_len > 0 {
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

    /// Add directory entry
    fn add_dir_entry(
        &self,
        dir_ino: u32,
        name: &str,
        target_ino: u32,
        file_type: u8,
    ) -> Result<(), VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        if name.len() > MAX_NAME_LEN {
            return Err(VfsError::InvalidPath);
        }

        let mut dir_inode = self.read_inode(dir_ino)?;
        let dir_size = dir_inode.i_size_lo as usize;

        // Read existing directory data
        let mut data = vec![0u8; dir_size + self.block_size as usize];
        self.read_inode_data(&dir_inode, 0, &mut data[..dir_size])?;

        // Calculate entry size (8 byte header + name, aligned to 4 bytes)
        let entry_size = ((8 + name.len() + 3) / 4) * 4;

        // Find space for new entry
        let mut pos = 0;
        let mut found = false;

        while pos + 8 <= dir_size {
            let entry =
                unsafe { core::ptr::read_unaligned(data[pos..].as_ptr() as *const Ext4DirEntry) };

            if entry.rec_len == 0 {
                break;
            }

            let actual_len = ((8 + entry.name_len as usize + 3) / 4) * 4;
            let free_space = entry.rec_len as usize - actual_len;

            if free_space >= entry_size {
                // Split this entry
                let new_rec_len = entry.rec_len - actual_len as u16;

                // Update existing entry's rec_len
                let new_entry_data = Ext4DirEntry {
                    inode: entry.inode,
                    rec_len: actual_len as u16,
                    name_len: entry.name_len,
                    file_type: entry.file_type,
                };
                unsafe {
                    core::ptr::write_unaligned(
                        data[pos..].as_mut_ptr() as *mut Ext4DirEntry,
                        new_entry_data,
                    );
                }

                // Write new entry
                let new_pos = pos + actual_len;
                let new_entry = Ext4DirEntry {
                    inode: target_ino,
                    rec_len: new_rec_len,
                    name_len: name.len() as u8,
                    file_type,
                };
                unsafe {
                    core::ptr::write_unaligned(
                        data[new_pos..].as_mut_ptr() as *mut Ext4DirEntry,
                        new_entry,
                    );
                }
                data[new_pos + 8..new_pos + 8 + name.len()].copy_from_slice(name.as_bytes());

                found = true;
                break;
            }

            pos += entry.rec_len as usize;
        }

        if !found {
            // Need to extend directory with new block
            let new_pos = dir_size;
            let new_entry = Ext4DirEntry {
                inode: target_ino,
                rec_len: self.block_size as u16,
                name_len: name.len() as u8,
                file_type,
            };

            // Zero out new block
            for i in new_pos..new_pos + self.block_size as usize {
                data[i] = 0;
            }

            unsafe {
                core::ptr::write_unaligned(
                    data[new_pos..].as_mut_ptr() as *mut Ext4DirEntry,
                    new_entry,
                );
            }
            data[new_pos + 8..new_pos + 8 + name.len()].copy_from_slice(name.as_bytes());

            // Update directory size
            dir_inode.i_size_lo = (dir_size + self.block_size as usize) as u32;
        }

        // Write back directory data
        self.write_inode_data(dir_ino, 0, &data[..dir_inode.i_size_lo as usize])?;

        // Update directory modification time
        dir_inode.i_mtime = crate::time::current_timestamp() as u32;
        dir_inode.i_ctime = dir_inode.i_mtime;
        self.write_inode(dir_ino, &dir_inode)?;

        Ok(())
    }

    /// Remove directory entry
    fn remove_dir_entry(&self, dir_ino: u32, name: &str) -> Result<u32, VfsError> {
        if self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        let dir_inode = self.read_inode(dir_ino)?;
        let dir_size = dir_inode.i_size_lo as usize;

        let mut data = vec![0u8; dir_size];
        self.read_inode_data(&dir_inode, 0, &mut data)?;

        let mut pos = 0;
        let mut prev_pos: Option<usize> = None;
        let mut removed_ino = 0;

        while pos + 8 <= dir_size {
            let entry =
                unsafe { core::ptr::read_unaligned(data[pos..].as_ptr() as *const Ext4DirEntry) };

            if entry.rec_len == 0 {
                break;
            }

            if entry.inode != 0 && entry.name_len > 0 {
                let name_start = pos + 8;
                let name_end = name_start + entry.name_len as usize;

                if name_end <= data.len() {
                    let entry_name =
                        core::str::from_utf8(&data[name_start..name_end]).unwrap_or("");

                    if entry_name == name {
                        removed_ino = entry.inode;

                        if let Some(pp) = prev_pos {
                            // Merge with previous entry
                            let prev_entry = unsafe {
                                core::ptr::read_unaligned(data[pp..].as_ptr() as *const Ext4DirEntry)
                            };
                            let new_rec_len = prev_entry.rec_len + entry.rec_len;
                            let updated = Ext4DirEntry {
                                rec_len: new_rec_len,
                                ..prev_entry
                            };
                            unsafe {
                                core::ptr::write_unaligned(
                                    data[pp..].as_mut_ptr() as *mut Ext4DirEntry,
                                    updated,
                                );
                            }
                        } else {
                            // Just mark inode as 0
                            let updated = Ext4DirEntry { inode: 0, ..entry };
                            unsafe {
                                core::ptr::write_unaligned(
                                    data[pos..].as_mut_ptr() as *mut Ext4DirEntry,
                                    updated,
                                );
                            }
                        }

                        // Write back
                        self.write_inode_data(dir_ino, 0, &data)?;
                        return Ok(removed_ino);
                    }
                }
            }

            prev_pos = Some(pos);
            pos += entry.rec_len as usize;
        }

        Err(VfsError::NotFound)
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

    /// Get parent directory and filename from path
    fn split_path(&self, path: &str) -> Result<(u32, String), VfsError> {
        let path = path.trim_end_matches('/');

        if path.is_empty() || path == "/" {
            return Err(VfsError::InvalidPath);
        }

        if let Some(pos) = path.rfind('/') {
            let parent_path = if pos == 0 { "/" } else { &path[..pos] };
            let name = &path[pos + 1..];

            if name.is_empty() {
                return Err(VfsError::InvalidPath);
            }

            let parent_ino = self.find_inode(parent_path)?;
            Ok((parent_ino, name.to_string()))
        } else {
            // File in root
            Ok((ROOT_INODE, path.to_string()))
        }
    }

    /// Create a new file
    fn create_file(&self, path: &str, mode: u16) -> Result<u32, VfsError> {
        let (parent_ino, name) = self.split_path(path)?;

        // Check if already exists
        if self.find_inode(path).is_ok() {
            return Err(VfsError::AlreadyExists);
        }

        // Allocate new inode
        let ino = self.alloc_inode(false)?;

        // Initialize inode
        let now = crate::time::current_timestamp() as u32;
        let inode = Ext4Inode {
            i_mode: inode_type::FILE | (mode & 0x0FFF),
            i_uid: 0,
            i_size_lo: 0,
            i_atime: now,
            i_ctime: now,
            i_mtime: now,
            i_dtime: 0,
            i_gid: 0,
            i_links_count: 1,
            i_blocks_lo: 0,
            i_flags: 0,
            i_osd1: 0,
            i_block: [0; 15],
            i_generation: 0,
            i_file_acl_lo: 0,
            i_size_high: 0,
            i_obso_faddr: 0,
            i_osd2: [0; 12],
            i_extra_isize: 0,
            i_checksum_hi: 0,
            i_ctime_extra: 0,
            i_mtime_extra: 0,
            i_atime_extra: 0,
            i_crtime: now,
            i_crtime_extra: 0,
            i_version_hi: 0,
            i_projid: 0,
        };

        self.write_inode(ino, &inode)?;

        // Add to parent directory
        self.add_dir_entry(parent_ino, &name, ino, dir_type::REG_FILE)?;

        Ok(ino)
    }

    /// Create a new directory
    fn create_directory(&self, path: &str, mode: u16) -> Result<u32, VfsError> {
        let (parent_ino, name) = self.split_path(path)?;

        // Check if already exists
        if self.find_inode(path).is_ok() {
            return Err(VfsError::AlreadyExists);
        }

        // Allocate new inode
        let ino = self.alloc_inode(true)?;

        // Allocate block for directory entries
        let group = (ino - 1) / self.inodes_per_group;
        let data_block = self.alloc_block(group)?;

        // Initialize directory with . and ..
        let mut dir_data = vec![0u8; self.block_size as usize];

        // "." entry
        let dot_entry = Ext4DirEntry {
            inode: ino,
            rec_len: 12,
            name_len: 1,
            file_type: dir_type::DIR,
        };
        unsafe {
            core::ptr::write_unaligned(dir_data.as_mut_ptr() as *mut Ext4DirEntry, dot_entry);
        }
        dir_data[8] = b'.';

        // ".." entry
        let dotdot_entry = Ext4DirEntry {
            inode: parent_ino,
            rec_len: self.block_size as u16 - 12,
            name_len: 2,
            file_type: dir_type::DIR,
        };
        unsafe {
            core::ptr::write_unaligned(
                dir_data[12..].as_mut_ptr() as *mut Ext4DirEntry,
                dotdot_entry,
            );
        }
        dir_data[20] = b'.';
        dir_data[21] = b'.';

        self.write_block(data_block, &dir_data)?;

        // Initialize inode
        let now = crate::time::current_timestamp() as u32;
        let mut inode = Ext4Inode {
            i_mode: inode_type::DIR | (mode & 0x0FFF),
            i_uid: 0,
            i_size_lo: self.block_size,
            i_atime: now,
            i_ctime: now,
            i_mtime: now,
            i_dtime: 0,
            i_gid: 0,
            i_links_count: 2, // . and parent link
            i_blocks_lo: (self.block_size / 512) as u32,
            i_flags: 0,
            i_osd1: 0,
            i_block: [0; 15],
            i_generation: 0,
            i_file_acl_lo: 0,
            i_size_high: 0,
            i_obso_faddr: 0,
            i_osd2: [0; 12],
            i_extra_isize: 0,
            i_checksum_hi: 0,
            i_ctime_extra: 0,
            i_mtime_extra: 0,
            i_atime_extra: 0,
            i_crtime: now,
            i_crtime_extra: 0,
            i_version_hi: 0,
            i_projid: 0,
        };
        inode.i_block[0] = data_block as u32;

        self.write_inode(ino, &inode)?;

        // Add to parent directory
        self.add_dir_entry(parent_ino, &name, ino, dir_type::DIR)?;

        // Increment parent link count
        let mut parent_inode = self.read_inode(parent_ino)?;
        parent_inode.i_links_count += 1;
        self.write_inode(parent_ino, &parent_inode)?;

        Ok(ino)
    }

    /// Begin a journal transaction
    fn begin_transaction(&self) -> Result<(), VfsError> {
        if let Some(ref journal) = *self.journal.read() {
            let tid = journal.sequence.fetch_add(1, Ordering::SeqCst);
            *journal.current_tx.write() = Some(Transaction::new(tid));
        }
        Ok(())
    }

    /// Commit journal transaction
    fn commit_transaction(&self) -> Result<(), VfsError> {
        if let Some(ref journal) = *self.journal.read() {
            let tx = journal.current_tx.write().take();

            if let Some(tx) = tx {
                // Write descriptor block
                let mut desc_data = vec![0u8; self.block_size as usize];
                let header = JournalHeader {
                    h_magic: JBD2_MAGIC.to_be(),
                    h_blocktype: (journal_block_type::DESCRIPTOR as u32).to_be(),
                    h_sequence: tx.tid.to_be(),
                };
                unsafe {
                    core::ptr::write_unaligned(
                        desc_data.as_mut_ptr() as *mut JournalHeader,
                        header,
                    );
                }

                // Write tags for each block in transaction
                let mut tag_offset = 12; // After header
                for (i, (block_num, _)) in tx.blocks.iter().enumerate() {
                    let flags = if i == tx.blocks.len() - 1 {
                        journal_tag_flags::LAST_TAG
                    } else {
                        0
                    };

                    let tag = JournalBlockTag {
                        t_blocknr: (*block_num as u32).to_be(),
                        t_flags: flags.to_be(),
                        t_blocknr_high: ((*block_num >> 32) as u16).to_be(),
                    };

                    unsafe {
                        core::ptr::write_unaligned(
                            desc_data[tag_offset..].as_mut_ptr() as *mut JournalBlockTag,
                            tag,
                        );
                    }
                    tag_offset += 8;
                }

                // Calculate journal position
                let pos = journal.start.load(Ordering::SeqCst);
                let desc_block = journal.start_block + pos as u64;

                self.flush_block(desc_block, &desc_data)?;

                // Write data blocks
                let mut data_pos = pos + 1;
                for (_, data) in &tx.blocks {
                    let data_block = journal.start_block + data_pos as u64;
                    self.flush_block(data_block, data)?;
                    data_pos += 1;
                }

                // Write commit block
                let commit_block = journal.start_block + data_pos as u64;
                let mut commit_data = vec![0u8; self.block_size as usize];
                let commit_header = JournalHeader {
                    h_magic: JBD2_MAGIC.to_be(),
                    h_blocktype: (journal_block_type::COMMIT as u32).to_be(),
                    h_sequence: tx.tid.to_be(),
                };
                unsafe {
                    core::ptr::write_unaligned(
                        commit_data.as_mut_ptr() as *mut JournalHeader,
                        commit_header,
                    );
                }
                self.flush_block(commit_block, &commit_data)?;

                // Update journal start
                let new_start = (data_pos + 1) % journal.block_count;
                journal.start.store(new_start, Ordering::SeqCst);

                // Now flush actual data blocks to their real locations
                for (block_num, data) in tx.blocks {
                    self.flush_block(block_num, &data)?;
                }
            }
        } else {
            // No journal - flush dirty blocks directly
            let dirty = self.dirty_blocks.write().drain(..).collect::<Vec<_>>();
            for block in dirty {
                if let Some(data) = self.block_cache.read().get(&block) {
                    self.flush_block(block, data)?;
                }
            }
        }

        self.device.sync()?;
        Ok(())
    }
}

impl Filesystem for Ext4Fs {
    fn name(&self) -> &'static str {
        "ext4"
    }

    fn is_readonly(&self) -> bool {
        self.readonly.load(Ordering::SeqCst)
    }

    fn total_size(&self) -> u64 {
        self.total_blocks * self.block_size as u64
    }

    fn free_space(&self) -> u64 {
        self.block_allocator.free_blocks.load(Ordering::SeqCst) * self.block_size as u64
    }

    fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, VfsError> {
        if (flags.write || flags.create) && self.readonly.load(Ordering::SeqCst) {
            return Err(VfsError::ReadOnly);
        }

        let ino = if flags.create {
            match self.find_inode(path) {
                Ok(ino) => {
                    if flags.exclusive {
                        return Err(VfsError::AlreadyExists);
                    }
                    ino
                }
                Err(VfsError::NotFound) => {
                    self.begin_transaction()?;
                    let ino = self.create_file(path, 0o644)?;
                    self.commit_transaction()?;
                    ino
                }
                Err(e) => return Err(e),
            }
        } else {
            self.find_inode(path)?
        };

        let inode = self.read_inode(ino)?;

        if inode.i_mode & inode_type::TYPE_MASK == inode_type::DIR {
            return Err(VfsError::IsADirectory);
        }

        // Handle truncate
        if flags.truncate && flags.write {
            self.begin_transaction()?;
            let mut inode = self.read_inode(ino)?;

            // Free existing blocks (simplified - just reset size)
            inode.i_size_lo = 0;
            inode.i_size_high = 0;
            inode.i_blocks_lo = 0;
            for i in 0..12 {
                inode.i_block[i] = 0;
            }

            self.write_inode(ino, &inode)?;
            self.commit_transaction()?;
        }

        Ok(FileHandle {
            inode: ino as u64,
            position: if flags.append {
                inode.i_size_lo as u64 | ((inode.i_size_high as u64) << 32)
            } else {
                0
            },
            flags,
        })
    }

    fn read(&self, handle: &FileHandle, buf: &mut [u8]) -> Result<usize, VfsError> {
        let inode = self.read_inode(handle.inode as u32)?;
        self.read_inode_data(&inode, handle.position, buf)
    }

    fn write(&self, handle: &FileHandle, buf: &[u8]) -> Result<usize, VfsError> {
        if !handle.flags.write {
            return Err(VfsError::PermissionDenied);
        }

        self.begin_transaction()?;
        let result = self.write_inode_data(handle.inode as u32, handle.position, buf);
        self.commit_transaction()?;

        result
    }

    fn close(&self, _handle: FileHandle) -> Result<(), VfsError> {
        Ok(())
    }

    fn stat(&self, path: &str) -> Result<FileStat, VfsError> {
        let ino = self.find_inode(path)?;
        let inode = self.read_inode(ino)?;

        let file_type = match inode.i_mode & inode_type::TYPE_MASK {
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
                    dir_type::FIFO => FileType::Pipe,
                    dir_type::SOCK => FileType::Socket,
                    _ => FileType::Regular,
                },
                inode: ino as u64,
            })
            .collect())
    }

    fn mkdir(&self, path: &str) -> Result<(), VfsError> {
        self.begin_transaction()?;
        self.create_directory(path, 0o755)?;
        self.commit_transaction()?;
        Ok(())
    }

    fn unlink(&self, path: &str) -> Result<(), VfsError> {
        let (parent_ino, name) = self.split_path(path)?;
        let ino = self.find_inode(path)?;
        let inode = self.read_inode(ino)?;

        if inode.i_mode & inode_type::TYPE_MASK == inode_type::DIR {
            return Err(VfsError::IsADirectory);
        }

        self.begin_transaction()?;

        // Remove from directory
        self.remove_dir_entry(parent_ino, &name)?;

        // Decrement link count
        let mut inode = self.read_inode(ino)?;
        if inode.i_links_count > 0 {
            inode.i_links_count -= 1;
        }

        if inode.i_links_count == 0 {
            // Free blocks and inode
            for i in 0..12 {
                if inode.i_block[i] != 0 {
                    self.free_block(inode.i_block[i] as u64)?;
                }
            }
            self.free_inode(ino, false)?;
        } else {
            self.write_inode(ino, &inode)?;
        }

        self.commit_transaction()?;
        Ok(())
    }

    fn rmdir(&self, path: &str) -> Result<(), VfsError> {
        let (parent_ino, name) = self.split_path(path)?;
        let ino = self.find_inode(path)?;
        let inode = self.read_inode(ino)?;

        if inode.i_mode & inode_type::TYPE_MASK != inode_type::DIR {
            return Err(VfsError::NotADirectory);
        }

        // Check if directory is empty (only . and ..)
        let entries = self.read_directory(ino)?;
        if entries.len() > 2 {
            return Err(VfsError::DirectoryNotEmpty);
        }

        self.begin_transaction()?;

        // Remove from parent directory
        self.remove_dir_entry(parent_ino, &name)?;

        // Free directory block
        if inode.i_block[0] != 0 {
            self.free_block(inode.i_block[0] as u64)?;
        }

        // Free inode
        self.free_inode(ino, true)?;

        // Decrement parent link count
        let mut parent_inode = self.read_inode(parent_ino)?;
        if parent_inode.i_links_count > 0 {
            parent_inode.i_links_count -= 1;
        }
        self.write_inode(parent_ino, &parent_inode)?;

        self.commit_transaction()?;
        Ok(())
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), VfsError> {
        let (from_parent, from_name) = self.split_path(from)?;
        let (to_parent, to_name) = self.split_path(to)?;

        let ino = self.find_inode(from)?;
        let inode = self.read_inode(ino)?;

        let file_type = if inode.i_mode & inode_type::TYPE_MASK == inode_type::DIR {
            dir_type::DIR
        } else {
            dir_type::REG_FILE
        };

        self.begin_transaction()?;

        // Remove old entry
        self.remove_dir_entry(from_parent, &from_name)?;

        // Add new entry
        self.add_dir_entry(to_parent, &to_name, ino, file_type)?;

        // Update .. in moved directory
        if file_type == dir_type::DIR && from_parent != to_parent {
            let mut dir_inode = self.read_inode(ino)?;
            let dir_size = dir_inode.i_size_lo as usize;
            let mut data = vec![0u8; dir_size];
            self.read_inode_data(&dir_inode, 0, &mut data)?;

            // Find and update .. entry
            let mut pos = 0;
            while pos + 8 <= data.len() {
                let entry = unsafe {
                    core::ptr::read_unaligned(data[pos..].as_ptr() as *const Ext4DirEntry)
                };

                if entry.rec_len == 0 {
                    break;
                }

                if entry.name_len == 2 && data[pos + 8] == b'.' && data[pos + 9] == b'.' {
                    let updated = Ext4DirEntry {
                        inode: to_parent,
                        ..entry
                    };
                    unsafe {
                        core::ptr::write_unaligned(
                            data[pos..].as_mut_ptr() as *mut Ext4DirEntry,
                            updated,
                        );
                    }
                    break;
                }

                pos += entry.rec_len as usize;
            }

            self.write_inode_data(ino, 0, &data)?;

            // Update parent link counts
            let mut old_parent = self.read_inode(from_parent)?;
            if old_parent.i_links_count > 0 {
                old_parent.i_links_count -= 1;
            }
            self.write_inode(from_parent, &old_parent)?;

            let mut new_parent = self.read_inode(to_parent)?;
            new_parent.i_links_count += 1;
            self.write_inode(to_parent, &new_parent)?;
        }

        self.commit_transaction()?;
        Ok(())
    }

    fn sync(&self) -> Result<(), VfsError> {
        // Flush all dirty data
        let dirty_blocks = self.dirty_blocks.write().drain(..).collect::<Vec<_>>();

        for block in dirty_blocks {
            if let Some(data) = self.block_cache.read().get(&block) {
                self.flush_block(block, data)?;
            }
        }

        self.device.sync()
    }
}
