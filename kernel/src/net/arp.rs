//! ARP Protocol
//!
//! Address Resolution Protocol for IPv4 to MAC address mapping.

use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use spin::Mutex;
use core::sync::atomic::{AtomicU64, Ordering};

use super::{MacAddress, Ipv4Address, NetError};

/// ARP hardware types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareType {
    Ethernet = 1,
    Ieee802 = 6,
    Arcnet = 7,
    FrameRelay = 15,
    Atm = 16,
    Hdlc = 17,
    FibreChannel = 18,
    Serial = 20,
}

/// ARP operation codes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArpOp {
    Request = 1,
    Reply = 2,
    ReverseRequest = 3,
    ReverseReply = 4,
}

/// ARP packet header
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ArpPacket {
    pub hardware_type: u16,
    pub protocol_type: u16,
    pub hardware_len: u8,
    pub protocol_len: u8,
    pub operation: u16,
    pub sender_mac: [u8; 6],
    pub sender_ip: [u8; 4],
    pub target_mac: [u8; 6],
    pub target_ip: [u8; 4],
}

impl ArpPacket {
    /// Create ARP request
    pub fn request(sender_mac: MacAddress, sender_ip: Ipv4Address, target_ip: Ipv4Address) -> Self {
        Self {
            hardware_type: (HardwareType::Ethernet as u16).to_be(),
            protocol_type: 0x0800_u16.to_be(), // IPv4
            hardware_len: 6,
            protocol_len: 4,
            operation: (ArpOp::Request as u16).to_be(),
            sender_mac: sender_mac.0,
            sender_ip: sender_ip.0,
            target_mac: [0; 6],
            target_ip: target_ip.0,
        }
    }

    /// Create ARP reply
    pub fn reply(sender_mac: MacAddress, sender_ip: Ipv4Address,
                 target_mac: MacAddress, target_ip: Ipv4Address) -> Self {
        Self {
            hardware_type: (HardwareType::Ethernet as u16).to_be(),
            protocol_type: 0x0800_u16.to_be(),
            hardware_len: 6,
            protocol_len: 4,
            operation: (ArpOp::Reply as u16).to_be(),
            sender_mac: sender_mac.0,
            sender_ip: sender_ip.0,
            target_mac: target_mac.0,
            target_ip: target_ip.0,
        }
    }

    /// Get operation type
    pub fn op(&self) -> Option<ArpOp> {
        match u16::from_be(self.operation) {
            1 => Some(ArpOp::Request),
            2 => Some(ArpOp::Reply),
            3 => Some(ArpOp::ReverseRequest),
            4 => Some(ArpOp::ReverseReply),
            _ => None,
        }
    }

    /// Get sender MAC
    pub fn get_sender_mac(&self) -> MacAddress {
        MacAddress(self.sender_mac)
    }

    /// Get sender IP
    pub fn get_sender_ip(&self) -> Ipv4Address {
        Ipv4Address(self.sender_ip)
    }

    /// Get target MAC
    pub fn get_target_mac(&self) -> MacAddress {
        MacAddress(self.target_mac)
    }

    /// Get target IP
    pub fn get_target_ip(&self) -> Ipv4Address {
        Ipv4Address(self.target_ip)
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(28);
        bytes.extend_from_slice(&self.hardware_type.to_ne_bytes());
        bytes.extend_from_slice(&self.protocol_type.to_ne_bytes());
        bytes.push(self.hardware_len);
        bytes.push(self.protocol_len);
        bytes.extend_from_slice(&self.operation.to_ne_bytes());
        bytes.extend_from_slice(&self.sender_mac);
        bytes.extend_from_slice(&self.sender_ip);
        bytes.extend_from_slice(&self.target_mac);
        bytes.extend_from_slice(&self.target_ip);
        bytes
    }

    /// Parse from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 28 {
            return None;
        }

        Some(Self {
            hardware_type: u16::from_ne_bytes([data[0], data[1]]),
            protocol_type: u16::from_ne_bytes([data[2], data[3]]),
            hardware_len: data[4],
            protocol_len: data[5],
            operation: u16::from_ne_bytes([data[6], data[7]]),
            sender_mac: [data[8], data[9], data[10], data[11], data[12], data[13]],
            sender_ip: [data[14], data[15], data[16], data[17]],
            target_mac: [data[18], data[19], data[20], data[21], data[22], data[23]],
            target_ip: [data[24], data[25], data[26], data[27]],
        })
    }
}

/// ARP cache entry state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArpState {
    /// Entry is incomplete (waiting for reply)
    Incomplete,
    /// Entry is complete and valid
    Reachable,
    /// Entry is stale and needs refresh
    Stale,
    /// Entry is being probed
    Probe,
    /// Entry failed to resolve
    Failed,
    /// Static entry (never expires)
    Permanent,
}

/// ARP cache entry
#[derive(Clone, Debug)]
pub struct ArpEntry {
    pub mac: MacAddress,
    pub ip: Ipv4Address,
    pub state: ArpState,
    pub created: u64,
    pub last_used: u64,
    pub retries: u8,
}

impl ArpEntry {
    /// Create new incomplete entry
    pub fn incomplete(ip: Ipv4Address, timestamp: u64) -> Self {
        Self {
            mac: MacAddress::ZERO,
            ip,
            state: ArpState::Incomplete,
            created: timestamp,
            last_used: timestamp,
            retries: 0,
        }
    }

    /// Create new reachable entry
    pub fn reachable(ip: Ipv4Address, mac: MacAddress, timestamp: u64) -> Self {
        Self {
            mac,
            ip,
            state: ArpState::Reachable,
            created: timestamp,
            last_used: timestamp,
            retries: 0,
        }
    }

    /// Create static entry
    pub fn permanent(ip: Ipv4Address, mac: MacAddress) -> Self {
        Self {
            mac,
            ip,
            state: ArpState::Permanent,
            created: 0,
            last_used: 0,
            retries: 0,
        }
    }

    /// Check if entry is valid
    pub fn is_valid(&self) -> bool {
        matches!(self.state, ArpState::Reachable | ArpState::Stale | ArpState::Permanent)
    }

    /// Check if entry needs refresh
    pub fn needs_refresh(&self, timestamp: u64, timeout: u64) -> bool {
        if self.state == ArpState::Permanent {
            return false;
        }
        timestamp.saturating_sub(self.last_used) > timeout
    }
}

/// ARP cache
pub struct ArpCache {
    entries: BTreeMap<Ipv4Address, ArpEntry>,
    max_entries: usize,
}

impl ArpCache {
    /// Create new ARP cache
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            max_entries: 512,
        }
    }

    /// Look up MAC for IP
    pub fn lookup(&self, ip: &Ipv4Address) -> Option<MacAddress> {
        self.entries.get(ip)
            .filter(|e| e.is_valid())
            .map(|e| e.mac)
    }

    /// Get entry
    pub fn get(&self, ip: &Ipv4Address) -> Option<&ArpEntry> {
        self.entries.get(ip)
    }

    /// Get mutable entry
    pub fn get_mut(&mut self, ip: &Ipv4Address) -> Option<&mut ArpEntry> {
        self.entries.get_mut(ip)
    }

    /// Insert or update entry
    pub fn insert(&mut self, entry: ArpEntry) {
        // Evict old entries if needed
        if self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }
        self.entries.insert(entry.ip, entry);
    }

    /// Remove entry
    pub fn remove(&mut self, ip: &Ipv4Address) -> Option<ArpEntry> {
        self.entries.remove(ip)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get all entries
    pub fn entries(&self) -> impl Iterator<Item = &ArpEntry> {
        self.entries.values()
    }

    /// Evict oldest non-permanent entry
    fn evict_oldest(&mut self) {
        let oldest = self.entries
            .iter()
            .filter(|(_, e)| e.state != ArpState::Permanent)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(ip, _)| *ip);

        if let Some(ip) = oldest {
            self.entries.remove(&ip);
        }
    }

    /// Clean up expired entries
    pub fn cleanup(&mut self, timestamp: u64, reachable_timeout: u64, incomplete_timeout: u64) {
        let expired: Vec<_> = self.entries
            .iter()
            .filter(|(_, e)| {
                match e.state {
                    ArpState::Permanent => false,
                    ArpState::Incomplete | ArpState::Failed => {
                        timestamp.saturating_sub(e.created) > incomplete_timeout
                    }
                    _ => {
                        timestamp.saturating_sub(e.last_used) > reachable_timeout
                    }
                }
            })
            .map(|(ip, _)| *ip)
            .collect();

        for ip in expired {
            self.entries.remove(&ip);
        }
    }
}

/// Global ARP cache
static ARP_CACHE: Mutex<ArpCache> = Mutex::new(ArpCache::new());

/// Timestamp counter (simplified - would use real time in production)
static TIMESTAMP: AtomicU64 = AtomicU64::new(0);

fn current_timestamp() -> u64 {
    TIMESTAMP.fetch_add(1, Ordering::Relaxed)
}

/// Pending ARP requests
static PENDING_REQUESTS: Mutex<Vec<(Ipv4Address, u64)>> = Mutex::new(Vec::new());

/// Our MAC address (set by network interface)
static OUR_MAC: Mutex<MacAddress> = Mutex::new(MacAddress::ZERO);

/// Our IP address
static OUR_IP: Mutex<Ipv4Address> = Mutex::new(Ipv4Address::UNSPECIFIED);

/// Set our addresses
pub fn set_addresses(mac: MacAddress, ip: Ipv4Address) {
    *OUR_MAC.lock() = mac;
    *OUR_IP.lock() = ip;
}

/// Look up MAC address for IP
pub fn lookup(ip: Ipv4Address) -> Option<MacAddress> {
    // Check for broadcast
    if ip == Ipv4Address::BROADCAST {
        return Some(MacAddress::BROADCAST);
    }

    // Check cache
    ARP_CACHE.lock().lookup(&ip)
}

/// Resolve IP to MAC (may send ARP request)
pub fn resolve(ip: Ipv4Address) -> Result<MacAddress, NetError> {
    // Check cache first
    if let Some(mac) = lookup(ip) {
        return Ok(mac);
    }

    // Send ARP request
    send_request(ip)?;

    // Add to pending
    {
        let mut pending = PENDING_REQUESTS.lock();
        let timestamp = current_timestamp();
        pending.push((ip, timestamp));
    }

    // In real implementation, would wait for reply with timeout
    Err(NetError::HostNotFound)
}

/// Send ARP request
pub fn send_request(target_ip: Ipv4Address) -> Result<(), NetError> {
    let sender_mac = *OUR_MAC.lock();
    let sender_ip = *OUR_IP.lock();

    if sender_mac == MacAddress::ZERO || sender_ip == Ipv4Address::UNSPECIFIED {
        return Err(NetError::NotConfigured);
    }

    let packet = ArpPacket::request(sender_mac, sender_ip, target_ip);

    // Add incomplete entry to cache
    {
        let mut cache = ARP_CACHE.lock();
        let timestamp = current_timestamp();
        cache.insert(ArpEntry::incomplete(target_ip, timestamp));
    }

    // Send via Ethernet layer (broadcast)
    // In real implementation, would call ethernet::send()
    let _bytes = packet.to_bytes();

    crate::kdebug!("ARP: Sent request for {}", target_ip);
    Ok(())
}

/// Send ARP reply
pub fn send_reply(target_mac: MacAddress, target_ip: Ipv4Address) -> Result<(), NetError> {
    let sender_mac = *OUR_MAC.lock();
    let sender_ip = *OUR_IP.lock();

    if sender_mac == MacAddress::ZERO || sender_ip == Ipv4Address::UNSPECIFIED {
        return Err(NetError::NotConfigured);
    }

    let packet = ArpPacket::reply(sender_mac, sender_ip, target_mac, target_ip);
    let _bytes = packet.to_bytes();

    crate::kdebug!("ARP: Sent reply to {}", target_ip);
    Ok(())
}

/// Process incoming ARP packet
pub fn process_packet(data: &[u8]) -> Result<(), NetError> {
    let packet = ArpPacket::from_bytes(data).ok_or(NetError::InvalidData)?;

    let op = packet.op().ok_or(NetError::InvalidData)?;
    let sender_mac = packet.get_sender_mac();
    let sender_ip = packet.get_sender_ip();
    let target_ip = packet.get_target_ip();

    // Update cache with sender info
    {
        let mut cache = ARP_CACHE.lock();
        let timestamp = current_timestamp();
        cache.insert(ArpEntry::reachable(sender_ip, sender_mac, timestamp));
    }

    match op {
        ArpOp::Request => {
            // Check if request is for us
            let our_ip = *OUR_IP.lock();
            if target_ip == our_ip {
                crate::kdebug!("ARP: Request for us from {}", sender_ip);
                send_reply(sender_mac, sender_ip)?;
            }
        }
        ArpOp::Reply => {
            crate::kdebug!("ARP: Reply from {} is at {}", sender_ip, sender_mac);
            // Already updated cache above
        }
        _ => {}
    }

    Ok(())
}

/// Add static ARP entry
pub fn add_static(ip: Ipv4Address, mac: MacAddress) {
    let mut cache = ARP_CACHE.lock();
    cache.insert(ArpEntry::permanent(ip, mac));
}

/// Remove ARP entry
pub fn remove_entry(ip: Ipv4Address) {
    ARP_CACHE.lock().remove(&ip);
}

/// Clear ARP cache
pub fn clear_cache() {
    ARP_CACHE.lock().clear();
}

/// Get all ARP entries
pub fn get_entries() -> Vec<ArpEntry> {
    ARP_CACHE.lock().entries().cloned().collect()
}

/// Send gratuitous ARP (announce our presence)
pub fn send_gratuitous() -> Result<(), NetError> {
    let our_mac = *OUR_MAC.lock();
    let our_ip = *OUR_IP.lock();

    if our_mac == MacAddress::ZERO || our_ip == Ipv4Address::UNSPECIFIED {
        return Err(NetError::NotConfigured);
    }

    // Gratuitous ARP: sender and target are both us
    let packet = ArpPacket::request(our_mac, our_ip, our_ip);
    let _bytes = packet.to_bytes();

    crate::kinfo!("ARP: Sent gratuitous ARP for {}", our_ip);
    Ok(())
}

/// ARP statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct ArpStats {
    pub requests_sent: u64,
    pub requests_received: u64,
    pub replies_sent: u64,
    pub replies_received: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

static ARP_STATS: Mutex<ArpStats> = Mutex::new(ArpStats {
    requests_sent: 0,
    requests_received: 0,
    replies_sent: 0,
    replies_received: 0,
    cache_hits: 0,
    cache_misses: 0,
});

/// Get ARP statistics
pub fn get_stats() -> ArpStats {
    *ARP_STATS.lock()
}

/// Initialize ARP subsystem
pub fn init() {
    crate::kprintln!("  ARP protocol initialized");
}
