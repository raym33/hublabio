//! Ethernet Driver
//!
//! Ethernet frame handling and driver interface.

use alloc::vec::Vec;
use spin::Mutex;

use super::{MacAddress, NetError, PacketBuffer};

/// Ethernet frame header size
pub const HEADER_SIZE: usize = 14;

/// Minimum Ethernet frame size (excluding FCS)
pub const MIN_FRAME_SIZE: usize = 60;

/// Maximum Ethernet frame size (excluding FCS)
pub const MAX_FRAME_SIZE: usize = 1514;

/// Maximum payload (MTU)
pub const MAX_PAYLOAD: usize = 1500;

/// EtherType values
pub mod ethertype {
    pub const IPV4: u16 = 0x0800;
    pub const ARP: u16 = 0x0806;
    pub const IPV6: u16 = 0x86DD;
    pub const VLAN: u16 = 0x8100;
}

/// Ethernet frame header
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct EthernetHeader {
    pub dst_mac: [u8; 6],
    pub src_mac: [u8; 6],
    pub ethertype: [u8; 2],
}

impl EthernetHeader {
    pub fn destination(&self) -> MacAddress {
        MacAddress(self.dst_mac)
    }

    pub fn source(&self) -> MacAddress {
        MacAddress(self.src_mac)
    }

    pub fn ethertype(&self) -> u16 {
        u16::from_be_bytes(self.ethertype)
    }

    pub fn set_destination(&mut self, mac: MacAddress) {
        self.dst_mac = mac.0;
    }

    pub fn set_source(&mut self, mac: MacAddress) {
        self.src_mac = mac.0;
    }

    pub fn set_ethertype(&mut self, etype: u16) {
        self.ethertype = etype.to_be_bytes();
    }
}

/// Parse Ethernet frame
pub fn parse(data: &[u8]) -> Result<(EthernetHeader, &[u8]), NetError> {
    if data.len() < HEADER_SIZE {
        return Err(NetError::InvalidPacket);
    }

    let header = unsafe { core::ptr::read_unaligned(data.as_ptr() as *const EthernetHeader) };

    let payload = &data[HEADER_SIZE..];

    Ok((header, payload))
}

/// Build Ethernet frame
pub fn build(
    dst: MacAddress,
    src: MacAddress,
    ethertype: u16,
    payload: &[u8],
) -> Result<Vec<u8>, NetError> {
    if payload.len() > MAX_PAYLOAD {
        return Err(NetError::BufferTooSmall);
    }

    let mut frame = Vec::with_capacity(HEADER_SIZE + payload.len().max(46));

    // Destination MAC
    frame.extend_from_slice(&dst.0);
    // Source MAC
    frame.extend_from_slice(&src.0);
    // EtherType
    frame.extend_from_slice(&ethertype.to_be_bytes());
    // Payload
    frame.extend_from_slice(payload);

    // Pad to minimum size
    while frame.len() < MIN_FRAME_SIZE {
        frame.push(0);
    }

    Ok(frame)
}

/// Ethernet driver trait
pub trait EthernetDriver: Send + Sync {
    /// Get MAC address
    fn mac_address(&self) -> MacAddress;

    /// Send a frame
    fn send(&mut self, frame: &[u8]) -> Result<(), NetError>;

    /// Receive a frame (non-blocking)
    fn try_receive(&mut self) -> Option<Vec<u8>>;

    /// Check if link is up
    fn link_up(&self) -> bool;

    /// Get link speed in Mbps
    fn link_speed(&self) -> u32;
}

/// BCM GENET Ethernet driver (Raspberry Pi 4)
pub struct BcmGenet {
    base: usize,
    mac: MacAddress,
    rx_ring: RxRing,
    tx_ring: TxRing,
    link_up: bool,
}

/// RX descriptor ring
struct RxRing {
    descriptors: Vec<DmaDescriptor>,
    buffers: Vec<Vec<u8>>,
    head: usize,
    tail: usize,
}

/// TX descriptor ring
struct TxRing {
    descriptors: Vec<DmaDescriptor>,
    buffers: Vec<Option<Vec<u8>>>,
    head: usize,
    tail: usize,
}

/// DMA descriptor
#[repr(C)]
struct DmaDescriptor {
    address: u32,
    length_status: u32,
}

/// BCM GENET register offsets
mod genet_regs {
    pub const SYS_REV_CTRL: usize = 0x00;
    pub const SYS_PORT_CTRL: usize = 0x04;
    pub const UMAC_CMD: usize = 0x808;
    pub const UMAC_MAC0: usize = 0x80C;
    pub const UMAC_MAC1: usize = 0x810;
    pub const MDIO_CMD: usize = 0xE14;
}

impl BcmGenet {
    /// BCM GENET base address for Pi 4
    pub const BASE_BCM2711: usize = 0xFD580000;

    /// Create a new GENET driver
    pub fn new(base: usize) -> Self {
        Self {
            base,
            mac: MacAddress::ZERO,
            rx_ring: RxRing {
                descriptors: Vec::new(),
                buffers: Vec::new(),
                head: 0,
                tail: 0,
            },
            tx_ring: TxRing {
                descriptors: Vec::new(),
                buffers: Vec::new(),
                head: 0,
                tail: 0,
            },
            link_up: false,
        }
    }

    /// Read register
    fn read_reg(&self, offset: usize) -> u32 {
        unsafe { core::ptr::read_volatile((self.base + offset) as *const u32) }
    }

    /// Write register
    fn write_reg(&self, offset: usize, value: u32) {
        unsafe {
            core::ptr::write_volatile((self.base + offset) as *mut u32, value);
        }
    }

    /// Initialize the driver
    pub fn init(&mut self) -> Result<(), NetError> {
        // Read hardware revision
        let rev = self.read_reg(regs::SYS_REV_CTRL);
        crate::kdebug!("GENET revision: 0x{:08X}", rev);

        // Read MAC address from hardware
        let mac0 = self.read_reg(regs::UMAC_MAC0);
        let mac1 = self.read_reg(regs::UMAC_MAC1);

        self.mac = MacAddress([
            ((mac0 >> 24) & 0xFF) as u8,
            ((mac0 >> 16) & 0xFF) as u8,
            ((mac0 >> 8) & 0xFF) as u8,
            (mac0 & 0xFF) as u8,
            ((mac1 >> 8) & 0xFF) as u8,
            (mac1 & 0xFF) as u8,
        ]);

        crate::kinfo!("GENET MAC: {}", self.mac);

        // Initialize DMA rings
        self.init_rings();

        // Enable UMAC
        self.write_reg(regs::UMAC_CMD, 0x25); // RX_EN | TX_EN | SPEED_100

        self.link_up = true;

        Ok(())
    }

    /// Initialize DMA rings
    fn init_rings(&mut self) {
        const RING_SIZE: usize = 256;
        const BUFFER_SIZE: usize = 2048;

        // Initialize RX ring
        self.rx_ring.descriptors = Vec::with_capacity(RING_SIZE);
        self.rx_ring.buffers = Vec::with_capacity(RING_SIZE);

        for _ in 0..RING_SIZE {
            let buffer = alloc::vec![0u8; BUFFER_SIZE];
            self.rx_ring.descriptors.push(DmaDescriptor {
                address: buffer.as_ptr() as u32,
                length_status: BUFFER_SIZE as u32,
            });
            self.rx_ring.buffers.push(buffer);
        }

        // Initialize TX ring
        self.tx_ring.descriptors = Vec::with_capacity(RING_SIZE);
        self.tx_ring.buffers = Vec::with_capacity(RING_SIZE);

        for _ in 0..RING_SIZE {
            self.tx_ring.descriptors.push(DmaDescriptor {
                address: 0,
                length_status: 0,
            });
            self.tx_ring.buffers.push(None);
        }
    }
}

impl EthernetDriver for BcmGenet {
    fn mac_address(&self) -> MacAddress {
        self.mac
    }

    fn send(&mut self, frame: &[u8]) -> Result<(), NetError> {
        if !self.link_up {
            return Err(NetError::InterfaceDown);
        }

        if frame.len() > MAX_FRAME_SIZE {
            return Err(NetError::BufferTooSmall);
        }

        // In a real implementation, we would:
        // 1. Get next TX descriptor
        // 2. Copy frame to DMA buffer
        // 3. Update descriptor
        // 4. Ring doorbell

        crate::kdebug!("GENET TX: {} bytes", frame.len());

        Ok(())
    }

    fn try_receive(&mut self) -> Option<Vec<u8>> {
        if !self.link_up {
            return None;
        }

        // In a real implementation, we would:
        // 1. Check RX ring for completed descriptors
        // 2. Extract frame from buffer
        // 3. Replenish buffer
        // 4. Return frame

        None
    }

    fn link_up(&self) -> bool {
        self.link_up
    }

    fn link_speed(&self) -> u32 {
        if self.link_up {
            1000
        } else {
            0
        }
    }
}

/// Global Ethernet driver
static ETHERNET: Mutex<Option<BcmGenet>> = Mutex::new(None);

/// Initialize Ethernet for Raspberry Pi 4
pub fn init_bcm2711() -> Result<(), NetError> {
    let mut driver = BcmGenet::new(BcmGenet::BASE_BCM2711);
    driver.init()?;

    // Register network interface
    let mac = driver.mac_address();
    let iface = super::NetworkInterface::new("eth0", mac, super::InterfaceDriver::Ethernet);
    super::register_interface(iface);

    *ETHERNET.lock() = Some(driver);

    crate::kprintln!("  Ethernet (GENET) initialized");
    Ok(())
}

/// Send an Ethernet frame
pub fn send(frame: &[u8]) -> Result<(), NetError> {
    ETHERNET
        .lock()
        .as_mut()
        .ok_or(NetError::NotInitialized)?
        .send(frame)
}

/// Try to receive an Ethernet frame
pub fn try_receive() -> Option<Vec<u8>> {
    ETHERNET.lock().as_mut()?.try_receive()
}

/// Check if Ethernet is available
pub fn is_available() -> bool {
    ETHERNET.lock().is_some()
}
