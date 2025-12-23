//! Network Stack
//!
//! Minimal network stack for HubLab IO.
//! Supports Ethernet, TCP/IP, and basic protocols.

pub mod ethernet;
pub mod ip;
pub mod tcp;
pub mod udp;
pub mod socket;
pub mod dhcp;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use alloc::string::String;
use core::sync::atomic::{AtomicU32, Ordering};
use spin::{Mutex, RwLock};

/// Network interface index counter
static INTERFACE_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Global network interfaces
static INTERFACES: RwLock<BTreeMap<u32, NetworkInterface>> = RwLock::new(BTreeMap::new());

/// MAC address (6 bytes)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MacAddress(pub [u8; 6]);

impl MacAddress {
    pub const BROADCAST: Self = Self([0xFF; 6]);
    pub const ZERO: Self = Self([0; 6]);

    pub fn is_broadcast(&self) -> bool {
        *self == Self::BROADCAST
    }

    pub fn is_multicast(&self) -> bool {
        self.0[0] & 0x01 != 0
    }
}

impl core::fmt::Display for MacAddress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}

/// IPv4 address
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ipv4Address(pub [u8; 4]);

impl Ipv4Address {
    pub const UNSPECIFIED: Self = Self([0, 0, 0, 0]);
    pub const BROADCAST: Self = Self([255, 255, 255, 255]);
    pub const LOCALHOST: Self = Self([127, 0, 0, 1]);

    pub fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self([a, b, c, d])
    }

    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    pub fn to_u32(&self) -> u32 {
        u32::from_be_bytes(self.0)
    }

    pub fn from_u32(val: u32) -> Self {
        Self(val.to_be_bytes())
    }

    pub fn is_unspecified(&self) -> bool {
        *self == Self::UNSPECIFIED
    }

    pub fn is_broadcast(&self) -> bool {
        *self == Self::BROADCAST
    }

    pub fn is_loopback(&self) -> bool {
        self.0[0] == 127
    }

    pub fn is_private(&self) -> bool {
        matches!(
            self.0,
            [10, ..] | [172, 16..=31, ..] | [192, 168, ..]
        )
    }
}

impl core::fmt::Display for Ipv4Address {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}.{}.{}", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// Network interface configuration
#[derive(Clone, Debug)]
pub struct InterfaceConfig {
    pub ip_address: Ipv4Address,
    pub netmask: Ipv4Address,
    pub gateway: Ipv4Address,
    pub dns_servers: Vec<Ipv4Address>,
    pub mtu: u16,
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            ip_address: Ipv4Address::UNSPECIFIED,
            netmask: Ipv4Address::new(255, 255, 255, 0),
            gateway: Ipv4Address::UNSPECIFIED,
            dns_servers: Vec::new(),
            mtu: 1500,
        }
    }
}

/// Network interface state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterfaceState {
    Down,
    Up,
    Configuring,
    Error,
}

/// Network interface
pub struct NetworkInterface {
    pub index: u32,
    pub name: String,
    pub mac_address: MacAddress,
    pub config: InterfaceConfig,
    pub state: InterfaceState,
    pub driver: InterfaceDriver,
    pub stats: InterfaceStats,
}

/// Interface driver type
#[derive(Clone, Copy, Debug)]
pub enum InterfaceDriver {
    Loopback,
    Ethernet,
    WiFi,
    Virtual,
}

/// Interface statistics
#[derive(Clone, Debug, Default)]
pub struct InterfaceStats {
    pub rx_packets: u64,
    pub tx_packets: u64,
    pub rx_bytes: u64,
    pub tx_bytes: u64,
    pub rx_errors: u64,
    pub tx_errors: u64,
    pub rx_dropped: u64,
    pub tx_dropped: u64,
}

impl NetworkInterface {
    /// Create a new interface
    pub fn new(name: &str, mac: MacAddress, driver: InterfaceDriver) -> Self {
        let index = INTERFACE_COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            index,
            name: String::from(name),
            mac_address: mac,
            config: InterfaceConfig::default(),
            state: InterfaceState::Down,
            driver,
            stats: InterfaceStats::default(),
        }
    }

    /// Bring interface up
    pub fn up(&mut self) -> Result<(), NetError> {
        if self.state == InterfaceState::Up {
            return Ok(());
        }

        self.state = InterfaceState::Up;
        crate::kinfo!("Interface {} ({}) is up", self.name, self.mac_address);
        Ok(())
    }

    /// Bring interface down
    pub fn down(&mut self) -> Result<(), NetError> {
        self.state = InterfaceState::Down;
        crate::kinfo!("Interface {} is down", self.name);
        Ok(())
    }

    /// Configure interface
    pub fn configure(&mut self, config: InterfaceConfig) {
        self.config = config;
        crate::kinfo!(
            "Interface {} configured: {} / {}",
            self.name,
            self.config.ip_address,
            self.config.netmask
        );
    }

    /// Check if address is on same network
    pub fn is_same_network(&self, addr: Ipv4Address) -> bool {
        let local = self.config.ip_address.to_u32();
        let remote = addr.to_u32();
        let mask = self.config.netmask.to_u32();

        (local & mask) == (remote & mask)
    }
}

/// Network errors
#[derive(Clone, Debug)]
pub enum NetError {
    NotInitialized,
    InterfaceNotFound,
    InterfaceDown,
    NoRoute,
    ConnectionRefused,
    ConnectionReset,
    TimedOut,
    AddrInUse,
    AddrNotAvailable,
    NetworkUnreachable,
    HostUnreachable,
    BufferTooSmall,
    InvalidPacket,
    Io,
}

/// Initialize network subsystem
pub fn init() {
    // Create loopback interface
    let lo = NetworkInterface::new(
        "lo",
        MacAddress::ZERO,
        InterfaceDriver::Loopback,
    );

    let mut interfaces = INTERFACES.write();
    interfaces.insert(lo.index, lo);

    // Configure loopback
    if let Some(lo) = interfaces.get_mut(&0) {
        lo.configure(InterfaceConfig {
            ip_address: Ipv4Address::LOCALHOST,
            netmask: Ipv4Address::new(255, 0, 0, 0),
            gateway: Ipv4Address::UNSPECIFIED,
            dns_servers: Vec::new(),
            mtu: 65535,
        });
        let _ = lo.up();
    }

    crate::kprintln!("  Network stack initialized");
}

/// Register a new network interface
pub fn register_interface(iface: NetworkInterface) -> u32 {
    let index = iface.index;
    INTERFACES.write().insert(index, iface);
    index
}

/// Get interface by index
pub fn get_interface(index: u32) -> Option<NetworkInterface> {
    INTERFACES.read().get(&index).cloned()
}

/// List all interfaces
pub fn list_interfaces() -> Vec<(u32, String, InterfaceState)> {
    INTERFACES
        .read()
        .iter()
        .map(|(idx, iface)| (*idx, iface.name.clone(), iface.state))
        .collect()
}

/// Configure interface by name
pub fn configure_interface(name: &str, config: InterfaceConfig) -> Result<(), NetError> {
    let mut interfaces = INTERFACES.write();

    for iface in interfaces.values_mut() {
        if iface.name == name {
            iface.configure(config);
            return Ok(());
        }
    }

    Err(NetError::InterfaceNotFound)
}

/// Find route to destination
pub fn find_route(dest: Ipv4Address) -> Option<(u32, Ipv4Address)> {
    let interfaces = INTERFACES.read();

    // Check if destination is on a directly connected network
    for iface in interfaces.values() {
        if iface.state != InterfaceState::Up {
            continue;
        }

        if iface.is_same_network(dest) {
            return Some((iface.index, dest));
        }
    }

    // Use default gateway
    for iface in interfaces.values() {
        if iface.state != InterfaceState::Up {
            continue;
        }

        if !iface.config.gateway.is_unspecified() {
            return Some((iface.index, iface.config.gateway));
        }
    }

    None
}

/// Packet buffer for network I/O
pub struct PacketBuffer {
    data: Vec<u8>,
    head: usize,
    tail: usize,
}

impl PacketBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: alloc::vec![0u8; capacity],
            head: 0,
            tail: 0,
        }
    }

    pub fn with_headroom(capacity: usize, headroom: usize) -> Self {
        let mut buf = Self::new(capacity);
        buf.head = headroom;
        buf.tail = headroom;
        buf
    }

    pub fn len(&self) -> usize {
        self.tail - self.head
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data[self.head..self.tail]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[self.head..self.tail]
    }

    pub fn push_front(&mut self, data: &[u8]) -> Result<(), NetError> {
        if self.head < data.len() {
            return Err(NetError::BufferTooSmall);
        }
        self.head -= data.len();
        self.data[self.head..self.head + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn push_back(&mut self, data: &[u8]) -> Result<(), NetError> {
        if self.tail + data.len() > self.data.len() {
            return Err(NetError::BufferTooSmall);
        }
        self.data[self.tail..self.tail + data.len()].copy_from_slice(data);
        self.tail += data.len();
        Ok(())
    }

    pub fn pop_front(&mut self, len: usize) -> Option<&[u8]> {
        if len > self.len() {
            return None;
        }
        let result = &self.data[self.head..self.head + len];
        self.head += len;
        Some(result)
    }
}
