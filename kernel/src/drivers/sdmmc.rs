//! SD/MMC Storage Driver
//!
//! Driver for SD cards and eMMC storage devices.
//! Supports SDHC, SDXC, and eMMC specifications.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use spin::{Mutex, RwLock};

/// SD card type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardType {
    /// Unknown/not detected
    Unknown,
    /// SD v1.x (up to 2GB)
    SdV1,
    /// SDHC (2GB-32GB)
    SdHc,
    /// SDXC (32GB-2TB)
    SdXc,
    /// MMC card
    Mmc,
    /// eMMC (embedded)
    Emmc,
}

/// SD card state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardState {
    /// Not inserted
    NotPresent,
    /// Card detected, not initialized
    Detected,
    /// Initialized and ready
    Ready,
    /// Transferring data
    Busy,
    /// Error state
    Error,
}

/// SD/MMC controller registers (generic SDHCI)
#[repr(C)]
struct SdhciRegs {
    /// SDMA system address / argument 2
    sdma_addr: u32,
    /// Block size / block count
    block_size: u16,
    block_count: u16,
    /// Argument
    argument: u32,
    /// Transfer mode / command
    transfer_mode: u16,
    command: u16,
    /// Response registers
    response: [u32; 4],
    /// Buffer data port
    buffer_data: u32,
    /// Present state
    present_state: u32,
    /// Host control / power control / block gap control / wake-up control
    host_control: u8,
    power_control: u8,
    block_gap_control: u8,
    wakeup_control: u8,
    /// Clock control / timeout control / software reset
    clock_control: u16,
    timeout_control: u8,
    software_reset: u8,
    /// Normal interrupt status / error interrupt status
    normal_int_status: u16,
    error_int_status: u16,
    /// Normal interrupt enable / error interrupt enable
    normal_int_enable: u16,
    error_int_enable: u16,
    /// Normal interrupt signal enable / error interrupt signal enable
    normal_int_signal: u16,
    error_int_signal: u16,
    /// Auto CMD error status / host control 2
    auto_cmd_error: u16,
    host_control2: u16,
    /// Capabilities
    capabilities: u64,
    /// Maximum current capabilities
    max_current: u64,
    /// Force event
    force_event: u32,
    /// ADMA error status / ADMA system address
    adma_error: u32,
    adma_addr: u64,
    /// Preset values
    preset_values: [u16; 8],
    /// Reserved
    _reserved: [u8; 112],
    /// Slot interrupt status / host controller version
    slot_int_status: u16,
    host_version: u16,
}

/// SD command codes
mod cmd {
    pub const GO_IDLE_STATE: u16 = 0;
    pub const ALL_SEND_CID: u16 = 2;
    pub const SEND_RELATIVE_ADDR: u16 = 3;
    pub const SET_DSR: u16 = 4;
    pub const SWITCH_FUNC: u16 = 6;
    pub const SELECT_CARD: u16 = 7;
    pub const SEND_IF_COND: u16 = 8;
    pub const SEND_CSD: u16 = 9;
    pub const SEND_CID: u16 = 10;
    pub const STOP_TRANSMISSION: u16 = 12;
    pub const SEND_STATUS: u16 = 13;
    pub const SET_BLOCKLEN: u16 = 16;
    pub const READ_SINGLE_BLOCK: u16 = 17;
    pub const READ_MULTIPLE_BLOCK: u16 = 18;
    pub const WRITE_SINGLE_BLOCK: u16 = 24;
    pub const WRITE_MULTIPLE_BLOCK: u16 = 25;
    pub const ERASE_WR_BLK_START: u16 = 32;
    pub const ERASE_WR_BLK_END: u16 = 33;
    pub const ERASE: u16 = 38;
    pub const APP_CMD: u16 = 55;
    pub const READ_OCR: u16 = 58;

    // Application commands (after APP_CMD)
    pub const ACMD_SET_BUS_WIDTH: u16 = 6;
    pub const ACMD_SD_STATUS: u16 = 13;
    pub const ACMD_SEND_NUM_WR_BLKS: u16 = 22;
    pub const ACMD_SET_WR_BLK_ERASE: u16 = 23;
    pub const ACMD_SD_SEND_OP_COND: u16 = 41;
    pub const ACMD_SET_CLR_CARD_DETECT: u16 = 42;
    pub const ACMD_SEND_SCR: u16 = 51;
}

/// Response types
#[derive(Clone, Copy, Debug)]
pub enum ResponseType {
    None,
    R1,      // Normal response
    R1b,     // Normal response with busy
    R2,      // CID/CSD register
    R3,      // OCR register
    R6,      // Published RCA response
    R7,      // Card interface condition
}

/// Card identification register (CID)
#[derive(Clone, Copy, Debug, Default)]
pub struct CardCid {
    pub manufacturer_id: u8,
    pub oem_id: [u8; 2],
    pub product_name: [u8; 5],
    pub product_revision: u8,
    pub serial_number: u32,
    pub manufacturing_date: u16,
}

/// Card specific data (CSD)
#[derive(Clone, Copy, Debug, Default)]
pub struct CardCsd {
    pub csd_version: u8,
    pub taac: u8,
    pub nsac: u8,
    pub tran_speed: u8,
    pub ccc: u16,
    pub read_bl_len: u8,
    pub c_size: u32,
    pub sector_size: u8,
    pub erase_blk_en: bool,
    pub wp_grp_size: u8,
}

/// SD card info
pub struct SdCard {
    /// Controller ID
    pub controller: u32,
    /// Card type
    pub card_type: CardType,
    /// Card state
    pub state: CardState,
    /// Relative card address
    pub rca: u16,
    /// Card identification
    pub cid: CardCid,
    /// Card specific data
    pub csd: CardCsd,
    /// Total sectors (512 bytes each)
    pub total_sectors: u64,
    /// Current clock speed (Hz)
    pub clock_hz: u32,
    /// Bus width (1, 4, or 8 bits)
    pub bus_width: u8,
    /// High speed mode enabled
    pub high_speed: bool,
    /// Card locked
    pub locked: bool,
}

impl SdCard {
    fn new(controller: u32) -> Self {
        Self {
            controller,
            card_type: CardType::Unknown,
            state: CardState::NotPresent,
            rca: 0,
            cid: CardCid::default(),
            csd: CardCsd::default(),
            total_sectors: 0,
            clock_hz: 400_000, // Start at 400kHz
            bus_width: 1,
            high_speed: false,
            locked: false,
        }
    }

    /// Get capacity in bytes
    pub fn capacity(&self) -> u64 {
        self.total_sectors * 512
    }

    /// Get capacity string
    pub fn capacity_string(&self) -> String {
        let bytes = self.capacity();
        if bytes >= 1024 * 1024 * 1024 * 1024 {
            alloc::format!("{} TB", bytes / (1024 * 1024 * 1024 * 1024))
        } else if bytes >= 1024 * 1024 * 1024 {
            alloc::format!("{} GB", bytes / (1024 * 1024 * 1024))
        } else if bytes >= 1024 * 1024 {
            alloc::format!("{} MB", bytes / (1024 * 1024))
        } else {
            alloc::format!("{} KB", bytes / 1024)
        }
    }
}

/// SD/MMC controller
pub struct SdController {
    /// Controller ID
    pub id: u32,
    /// Base address of registers
    pub base_addr: usize,
    /// Current card
    pub card: Option<SdCard>,
    /// Capabilities
    pub capabilities: SdCapabilities,
    /// Statistics
    pub stats: SdStats,
}

/// Controller capabilities
#[derive(Clone, Copy, Debug, Default)]
pub struct SdCapabilities {
    pub max_clock: u32,
    pub base_clock: u32,
    pub timeout_clock: u32,
    pub max_block_len: u32,
    pub support_8bit: bool,
    pub support_adma2: bool,
    pub support_hs200: bool,
    pub support_hs400: bool,
    pub support_ddr50: bool,
    pub support_sdr104: bool,
    pub voltage_33: bool,
    pub voltage_30: bool,
    pub voltage_18: bool,
}

/// Statistics
#[derive(Default)]
pub struct SdStats {
    pub reads: AtomicU64,
    pub writes: AtomicU64,
    pub bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,
    pub errors: AtomicU32,
}

impl SdController {
    /// Create new controller
    pub fn new(id: u32, base_addr: usize) -> Self {
        Self {
            id,
            base_addr,
            card: None,
            capabilities: SdCapabilities::default(),
            stats: SdStats::default(),
        }
    }

    /// Initialize controller
    pub fn init(&mut self) -> Result<(), SdError> {
        // Reset controller
        self.reset()?;

        // Read capabilities
        self.read_capabilities();

        // Set initial clock (400kHz for initialization)
        self.set_clock(400_000)?;

        // Set bus width to 1 bit
        self.set_bus_width(1)?;

        // Enable interrupts
        self.enable_interrupts();

        crate::kinfo!("SD controller {} initialized", self.id);

        Ok(())
    }

    /// Reset controller
    fn reset(&mut self) -> Result<(), SdError> {
        // Would write to software reset register
        // Wait for reset complete
        Ok(())
    }

    /// Read controller capabilities
    fn read_capabilities(&mut self) {
        // Would read capabilities register
        self.capabilities = SdCapabilities {
            max_clock: 52_000_000,
            base_clock: 48_000_000,
            timeout_clock: 1_000_000,
            max_block_len: 512,
            support_8bit: false,
            support_adma2: true,
            support_hs200: false,
            support_hs400: false,
            support_ddr50: false,
            support_sdr104: false,
            voltage_33: true,
            voltage_30: false,
            voltage_18: false,
        };
    }

    /// Set clock speed
    fn set_clock(&mut self, hz: u32) -> Result<(), SdError> {
        // Calculate divisor
        let _divisor = if hz >= self.capabilities.base_clock {
            1
        } else {
            (self.capabilities.base_clock + hz - 1) / hz
        };

        // Would program clock register
        if let Some(ref mut card) = self.card {
            card.clock_hz = hz;
        }

        Ok(())
    }

    /// Set bus width
    fn set_bus_width(&mut self, width: u8) -> Result<(), SdError> {
        if width != 1 && width != 4 && width != 8 {
            return Err(SdError::InvalidParam);
        }

        if width == 8 && !self.capabilities.support_8bit {
            return Err(SdError::NotSupported);
        }

        // Would program host control register
        if let Some(ref mut card) = self.card {
            card.bus_width = width;
        }

        Ok(())
    }

    /// Enable interrupts
    fn enable_interrupts(&mut self) {
        // Would enable relevant interrupts
    }

    /// Send command
    fn send_cmd(&mut self, cmd: u16, arg: u32, resp_type: ResponseType) -> Result<[u32; 4], SdError> {
        // Would:
        // 1. Wait for command line ready
        // 2. Write argument
        // 3. Write command
        // 4. Wait for completion
        // 5. Read response

        // Placeholder response
        Ok([0; 4])
    }

    /// Send application command (preceded by CMD55)
    fn send_acmd(&mut self, cmd: u16, arg: u32, resp_type: ResponseType) -> Result<[u32; 4], SdError> {
        let rca = self.card.as_ref().map(|c| c.rca).unwrap_or(0);

        // Send CMD55 first
        self.send_cmd(cmd::APP_CMD, (rca as u32) << 16, ResponseType::R1)?;

        // Then send application command
        self.send_cmd(cmd, arg, resp_type)
    }

    /// Detect and initialize card
    pub fn detect_card(&mut self) -> Result<(), SdError> {
        // Send CMD0 (GO_IDLE_STATE)
        self.send_cmd(cmd::GO_IDLE_STATE, 0, ResponseType::None)?;

        // Create card structure
        let mut card = SdCard::new(self.id);

        // Send CMD8 (SEND_IF_COND) to check for SD v2+
        let check_pattern = 0x1AA;
        match self.send_cmd(cmd::SEND_IF_COND, check_pattern, ResponseType::R7) {
            Ok(resp) => {
                if (resp[0] & 0xFFF) == check_pattern {
                    // SD v2+ card
                    card.card_type = CardType::SdHc;
                } else {
                    return Err(SdError::UnsupportedCard);
                }
            }
            Err(_) => {
                // SD v1 or MMC
                card.card_type = CardType::SdV1;
            }
        }

        // Send ACMD41 to initialize SD card
        let hcs = if card.card_type == CardType::SdHc { 1 << 30 } else { 0 };
        let arg = 0x00FF8000 | hcs; // OCR with voltage window

        for _ in 0..100 {
            let resp = self.send_acmd(cmd::ACMD_SD_SEND_OP_COND, arg, ResponseType::R3)?;

            if resp[0] & (1 << 31) != 0 {
                // Card is ready
                if resp[0] & (1 << 30) != 0 {
                    // High capacity
                    card.card_type = CardType::SdHc;
                }
                break;
            }

            // Wait and retry
            // Would use proper delay
        }

        // Send CMD2 (ALL_SEND_CID)
        let resp = self.send_cmd(cmd::ALL_SEND_CID, 0, ResponseType::R2)?;
        card.cid = parse_cid(&resp);

        // Send CMD3 (SEND_RELATIVE_ADDR)
        let resp = self.send_cmd(cmd::SEND_RELATIVE_ADDR, 0, ResponseType::R6)?;
        card.rca = (resp[0] >> 16) as u16;

        // Send CMD9 (SEND_CSD)
        let resp = self.send_cmd(cmd::SEND_CSD, (card.rca as u32) << 16, ResponseType::R2)?;
        card.csd = parse_csd(&resp);

        // Calculate total sectors
        card.total_sectors = calculate_sectors(&card.csd);

        // Select card (CMD7)
        self.send_cmd(cmd::SELECT_CARD, (card.rca as u32) << 16, ResponseType::R1b)?;

        // Set block length to 512 bytes
        self.send_cmd(cmd::SET_BLOCKLEN, 512, ResponseType::R1)?;

        // Switch to 4-bit bus if supported
        if card.card_type != CardType::Mmc {
            self.send_acmd(cmd::ACMD_SET_BUS_WIDTH, 2, ResponseType::R1)?; // 2 = 4-bit
            self.set_bus_width(4)?;
        }

        // Increase clock speed
        let target_clock = if card.high_speed { 50_000_000 } else { 25_000_000 };
        self.set_clock(target_clock)?;

        card.state = CardState::Ready;
        self.card = Some(card);

        if let Some(ref card) = self.card {
            crate::kinfo!("SD card detected: {:?}, capacity: {}",
                         card.card_type, card.capacity_string());
        }

        Ok(())
    }

    /// Read sectors
    pub fn read_sectors(&mut self, start_sector: u64, count: u32, buf: &mut [u8]) -> Result<(), SdError> {
        if buf.len() < (count as usize * 512) {
            return Err(SdError::InvalidParam);
        }

        let card = self.card.as_ref().ok_or(SdError::NoCard)?;

        if card.state != CardState::Ready {
            return Err(SdError::NotReady);
        }

        if start_sector + count as u64 > card.total_sectors {
            return Err(SdError::OutOfRange);
        }

        // For SDHC/SDXC, address is in blocks; for SD v1, in bytes
        let addr = if card.card_type == CardType::SdHc {
            start_sector as u32
        } else {
            (start_sector * 512) as u32
        };

        if count == 1 {
            // Single block read
            self.send_cmd(cmd::READ_SINGLE_BLOCK, addr, ResponseType::R1)?;
        } else {
            // Multi-block read
            self.send_cmd(cmd::READ_MULTIPLE_BLOCK, addr, ResponseType::R1)?;
        }

        // Would read data from buffer port
        // For now, zero the buffer
        buf[..count as usize * 512].fill(0);

        if count > 1 {
            // Stop transmission
            self.send_cmd(cmd::STOP_TRANSMISSION, 0, ResponseType::R1b)?;
        }

        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_read.fetch_add((count as u64) * 512, Ordering::Relaxed);

        Ok(())
    }

    /// Write sectors
    pub fn write_sectors(&mut self, start_sector: u64, count: u32, buf: &[u8]) -> Result<(), SdError> {
        if buf.len() < (count as usize * 512) {
            return Err(SdError::InvalidParam);
        }

        let card = self.card.as_ref().ok_or(SdError::NoCard)?;

        if card.state != CardState::Ready {
            return Err(SdError::NotReady);
        }

        if start_sector + count as u64 > card.total_sectors {
            return Err(SdError::OutOfRange);
        }

        if card.locked {
            return Err(SdError::WriteProtected);
        }

        let addr = if card.card_type == CardType::SdHc {
            start_sector as u32
        } else {
            (start_sector * 512) as u32
        };

        if count == 1 {
            // Single block write
            self.send_cmd(cmd::WRITE_SINGLE_BLOCK, addr, ResponseType::R1)?;
        } else {
            // Multi-block write
            self.send_cmd(cmd::WRITE_MULTIPLE_BLOCK, addr, ResponseType::R1)?;
        }

        // Would write data to buffer port

        if count > 1 {
            self.send_cmd(cmd::STOP_TRANSMISSION, 0, ResponseType::R1b)?;
        }

        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_written.fetch_add((count as u64) * 512, Ordering::Relaxed);

        Ok(())
    }

    /// Erase sectors
    pub fn erase_sectors(&mut self, start_sector: u64, count: u64) -> Result<(), SdError> {
        let card = self.card.as_ref().ok_or(SdError::NoCard)?;

        if card.state != CardState::Ready {
            return Err(SdError::NotReady);
        }

        if start_sector + count > card.total_sectors {
            return Err(SdError::OutOfRange);
        }

        if card.locked {
            return Err(SdError::WriteProtected);
        }

        let start_addr = if card.card_type == CardType::SdHc {
            start_sector as u32
        } else {
            (start_sector * 512) as u32
        };

        let end_addr = if card.card_type == CardType::SdHc {
            (start_sector + count - 1) as u32
        } else {
            ((start_sector + count - 1) * 512) as u32
        };

        // Set erase start
        self.send_cmd(cmd::ERASE_WR_BLK_START, start_addr, ResponseType::R1)?;

        // Set erase end
        self.send_cmd(cmd::ERASE_WR_BLK_END, end_addr, ResponseType::R1)?;

        // Erase
        self.send_cmd(cmd::ERASE, 0, ResponseType::R1b)?;

        Ok(())
    }

    /// Get card status
    pub fn get_status(&mut self) -> Result<u32, SdError> {
        let rca = self.card.as_ref().ok_or(SdError::NoCard)?.rca;
        let resp = self.send_cmd(cmd::SEND_STATUS, (rca as u32) << 16, ResponseType::R1)?;
        Ok(resp[0])
    }
}

/// Parse CID register
fn parse_cid(resp: &[u32; 4]) -> CardCid {
    let mut cid = CardCid::default();

    // CID is 128 bits, big-endian
    cid.manufacturer_id = ((resp[3] >> 16) & 0xFF) as u8;
    cid.oem_id[0] = ((resp[3] >> 8) & 0xFF) as u8;
    cid.oem_id[1] = (resp[3] & 0xFF) as u8;

    cid.product_name[0] = ((resp[2] >> 24) & 0xFF) as u8;
    cid.product_name[1] = ((resp[2] >> 16) & 0xFF) as u8;
    cid.product_name[2] = ((resp[2] >> 8) & 0xFF) as u8;
    cid.product_name[3] = (resp[2] & 0xFF) as u8;
    cid.product_name[4] = ((resp[1] >> 24) & 0xFF) as u8;

    cid.product_revision = ((resp[1] >> 16) & 0xFF) as u8;
    cid.serial_number = ((resp[1] & 0xFFFF) << 16) | ((resp[0] >> 16) & 0xFFFF);
    cid.manufacturing_date = ((resp[0] >> 4) & 0xFFF) as u16;

    cid
}

/// Parse CSD register
fn parse_csd(resp: &[u32; 4]) -> CardCsd {
    let mut csd = CardCsd::default();

    csd.csd_version = ((resp[3] >> 22) & 0x3) as u8;

    if csd.csd_version == 1 {
        // CSD v2.0 (SDHC/SDXC)
        csd.c_size = ((resp[1] & 0x3F) << 16) | ((resp[0] >> 16) & 0xFFFF);
    } else {
        // CSD v1.0 (SD)
        let c_size = ((resp[2] & 0x3FF) << 2) | ((resp[1] >> 30) & 0x3);
        let c_size_mult = ((resp[1] >> 15) & 0x7) as u8;
        let read_bl_len = ((resp[2] >> 16) & 0xF) as u8;
        csd.c_size = (c_size + 1) * (1 << (c_size_mult + 2)) * (1 << read_bl_len) / 512;
    }

    csd.read_bl_len = ((resp[2] >> 16) & 0xF) as u8;
    csd.tran_speed = ((resp[3] >> 8) & 0xFF) as u8;

    csd
}

/// Calculate total sectors from CSD
fn calculate_sectors(csd: &CardCsd) -> u64 {
    if csd.csd_version == 1 {
        // SDHC/SDXC
        (csd.c_size as u64 + 1) * 1024
    } else {
        // SD
        csd.c_size as u64
    }
}

/// SD error
#[derive(Clone, Copy, Debug)]
pub enum SdError {
    /// Timeout
    Timeout,
    /// CRC error
    CrcError,
    /// No card present
    NoCard,
    /// Card not ready
    NotReady,
    /// Invalid parameter
    InvalidParam,
    /// Out of range
    OutOfRange,
    /// Write protected
    WriteProtected,
    /// Not supported
    NotSupported,
    /// Unsupported card
    UnsupportedCard,
    /// I/O error
    IoError,
}

impl SdError {
    pub fn to_errno(&self) -> i32 {
        match self {
            SdError::Timeout => -110,       // ETIMEDOUT
            SdError::CrcError => -74,       // EBADMSG
            SdError::NoCard => -19,         // ENODEV
            SdError::NotReady => -11,       // EAGAIN
            SdError::InvalidParam => -22,   // EINVAL
            SdError::OutOfRange => -34,     // ERANGE
            SdError::WriteProtected => -30, // EROFS
            SdError::NotSupported => -95,   // EOPNOTSUPP
            SdError::UnsupportedCard => -95,
            SdError::IoError => -5,         // EIO
        }
    }
}

// ============================================================================
// Block Device Interface
// ============================================================================

/// Block device for SD card
pub struct SdBlockDevice {
    controller_id: u32,
}

impl SdBlockDevice {
    pub fn new(controller_id: u32) -> Self {
        Self { controller_id }
    }
}

// ============================================================================
// Global State
// ============================================================================

/// SD controllers
static CONTROLLERS: RwLock<BTreeMap<u32, Mutex<SdController>>> = RwLock::new(BTreeMap::new());

/// Next controller ID
static NEXT_CONTROLLER_ID: AtomicU32 = AtomicU32::new(0);

/// Register a new SD controller
pub fn register_controller(base_addr: usize) -> Result<u32, SdError> {
    let id = NEXT_CONTROLLER_ID.fetch_add(1, Ordering::SeqCst);

    let mut controller = SdController::new(id, base_addr);
    controller.init()?;

    CONTROLLERS.write().insert(id, Mutex::new(controller));

    Ok(id)
}

/// Get controller by ID
pub fn get_controller(id: u32) -> Option<&'static Mutex<SdController>> {
    CONTROLLERS.read().get(&id).map(|c| unsafe {
        &*(c as *const Mutex<SdController>)
    })
}

/// Detect card on controller
pub fn detect_card(controller_id: u32) -> Result<(), SdError> {
    let controllers = CONTROLLERS.read();
    let controller = controllers.get(&controller_id).ok_or(SdError::NoCard)?;
    controller.lock().detect_card()
}

/// Read sectors from SD card
pub fn read_sectors(controller_id: u32, start: u64, count: u32, buf: &mut [u8]) -> Result<(), SdError> {
    let controllers = CONTROLLERS.read();
    let controller = controllers.get(&controller_id).ok_or(SdError::NoCard)?;
    controller.lock().read_sectors(start, count, buf)
}

/// Write sectors to SD card
pub fn write_sectors(controller_id: u32, start: u64, count: u32, buf: &[u8]) -> Result<(), SdError> {
    let controllers = CONTROLLERS.read();
    let controller = controllers.get(&controller_id).ok_or(SdError::NoCard)?;
    controller.lock().write_sectors(start, count, buf)
}

/// Get card info
pub fn get_card_info(controller_id: u32) -> Option<(CardType, u64, String)> {
    let controllers = CONTROLLERS.read();
    let controller = controllers.get(&controller_id)?;
    let ctrl = controller.lock();
    let card = ctrl.card.as_ref()?;
    Some((card.card_type, card.total_sectors, card.capacity_string()))
}

/// Initialize SD/MMC subsystem
pub fn init() {
    // Would probe for controllers in device tree
    // For now, just log
    crate::kprintln!("  SD/MMC subsystem initialized");
}
