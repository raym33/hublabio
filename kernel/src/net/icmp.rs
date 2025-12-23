//! Internet Control Message Protocol (ICMP)
//!
//! Handles ICMP messages including ping (echo request/reply),
//! destination unreachable, time exceeded, etc.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use spin::Mutex;

use super::ip::{self, calculate_checksum, Ipv4Header};
use super::{Ipv4Address, NetError};

/// ICMP message types
pub mod types {
    pub const ECHO_REPLY: u8 = 0;
    pub const DEST_UNREACHABLE: u8 = 3;
    pub const SOURCE_QUENCH: u8 = 4;
    pub const REDIRECT: u8 = 5;
    pub const ECHO_REQUEST: u8 = 8;
    pub const TIME_EXCEEDED: u8 = 11;
    pub const PARAMETER_PROBLEM: u8 = 12;
    pub const TIMESTAMP_REQUEST: u8 = 13;
    pub const TIMESTAMP_REPLY: u8 = 14;
    pub const INFO_REQUEST: u8 = 15;
    pub const INFO_REPLY: u8 = 16;
    pub const ADDRESS_MASK_REQUEST: u8 = 17;
    pub const ADDRESS_MASK_REPLY: u8 = 18;
}

/// Destination unreachable codes
pub mod unreachable {
    pub const NET_UNREACHABLE: u8 = 0;
    pub const HOST_UNREACHABLE: u8 = 1;
    pub const PROTOCOL_UNREACHABLE: u8 = 2;
    pub const PORT_UNREACHABLE: u8 = 3;
    pub const FRAGMENTATION_NEEDED: u8 = 4;
    pub const SOURCE_ROUTE_FAILED: u8 = 5;
    pub const DEST_NET_UNKNOWN: u8 = 6;
    pub const DEST_HOST_UNKNOWN: u8 = 7;
}

/// Time exceeded codes
pub mod time_exceeded {
    pub const TTL_EXCEEDED: u8 = 0;
    pub const FRAGMENT_REASSEMBLY: u8 = 1;
}

/// ICMP header
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct IcmpHeader {
    pub icmp_type: u8,
    pub code: u8,
    pub checksum: [u8; 2],
    pub rest: [u8; 4], // Varies by type
}

impl IcmpHeader {
    /// Create echo request header
    pub fn echo_request(identifier: u16, sequence: u16) -> Self {
        Self {
            icmp_type: types::ECHO_REQUEST,
            code: 0,
            checksum: [0, 0],
            rest: [
                (identifier >> 8) as u8,
                (identifier & 0xFF) as u8,
                (sequence >> 8) as u8,
                (sequence & 0xFF) as u8,
            ],
        }
    }

    /// Create echo reply header
    pub fn echo_reply(identifier: u16, sequence: u16) -> Self {
        Self {
            icmp_type: types::ECHO_REPLY,
            code: 0,
            checksum: [0, 0],
            rest: [
                (identifier >> 8) as u8,
                (identifier & 0xFF) as u8,
                (sequence >> 8) as u8,
                (sequence & 0xFF) as u8,
            ],
        }
    }

    /// Create destination unreachable header
    pub fn dest_unreachable(code: u8) -> Self {
        Self {
            icmp_type: types::DEST_UNREACHABLE,
            code,
            checksum: [0, 0],
            rest: [0; 4],
        }
    }

    /// Create time exceeded header
    pub fn time_exceeded(code: u8) -> Self {
        Self {
            icmp_type: types::TIME_EXCEEDED,
            code,
            checksum: [0, 0],
            rest: [0; 4],
        }
    }

    /// Get identifier (for echo)
    pub fn identifier(&self) -> u16 {
        u16::from_be_bytes([self.rest[0], self.rest[1]])
    }

    /// Get sequence number (for echo)
    pub fn sequence(&self) -> u16 {
        u16::from_be_bytes([self.rest[2], self.rest[3]])
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> [u8; 8] {
        unsafe { core::mem::transmute_copy(self) }
    }
}

/// Ping request tracking
struct PingRequest {
    target: Ipv4Address,
    identifier: u16,
    sequence: u16,
    sent_time: u64,
    callback: Option<fn(Ipv4Address, u16, u64)>,
}

/// Active ping requests
static PING_REQUESTS: Mutex<BTreeMap<(u16, u16), PingRequest>> = Mutex::new(BTreeMap::new());

/// Ping sequence counter
static PING_SEQUENCE: AtomicU16 = AtomicU16::new(0);

/// Ping identifier
static PING_ID: AtomicU16 = AtomicU16::new(0x1234);

/// Timestamp counter (in microseconds, simplified)
static TIMESTAMP: AtomicU64 = AtomicU64::new(0);

fn current_time() -> u64 {
    TIMESTAMP.fetch_add(1000, Ordering::Relaxed) // Increment by 1ms
}

/// ICMP statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct IcmpStats {
    pub echo_requests_sent: u64,
    pub echo_requests_received: u64,
    pub echo_replies_sent: u64,
    pub echo_replies_received: u64,
    pub dest_unreachable_sent: u64,
    pub dest_unreachable_received: u64,
    pub time_exceeded_sent: u64,
    pub time_exceeded_received: u64,
    pub errors: u64,
}

static ICMP_STATS: Mutex<IcmpStats> = Mutex::new(IcmpStats {
    echo_requests_sent: 0,
    echo_requests_received: 0,
    echo_replies_sent: 0,
    echo_replies_received: 0,
    dest_unreachable_sent: 0,
    dest_unreachable_received: 0,
    time_exceeded_sent: 0,
    time_exceeded_received: 0,
    errors: 0,
});

/// Send ICMP packet
fn send(dst: Ipv4Address, header: &IcmpHeader, payload: &[u8]) -> Result<(), NetError> {
    let mut packet = Vec::with_capacity(8 + payload.len());
    packet.extend_from_slice(&header.to_bytes());
    packet.extend_from_slice(payload);

    // Calculate checksum
    let checksum = calculate_checksum(&packet);
    packet[2] = (checksum >> 8) as u8;
    packet[3] = (checksum & 0xFF) as u8;

    // Send via IP layer
    ip::send(dst, ip::protocol::ICMP, &packet)
}

/// Send ping (echo request)
pub fn ping(target: Ipv4Address, payload: &[u8]) -> Result<u16, NetError> {
    let identifier = PING_ID.load(Ordering::Relaxed);
    let sequence = PING_SEQUENCE.fetch_add(1, Ordering::SeqCst);

    let header = IcmpHeader::echo_request(identifier, sequence);
    send(target, &header, payload)?;

    // Track request
    {
        let mut requests = PING_REQUESTS.lock();
        requests.insert(
            (identifier, sequence),
            PingRequest {
                target,
                identifier,
                sequence,
                sent_time: current_time(),
                callback: None,
            },
        );
    }

    ICMP_STATS.lock().echo_requests_sent += 1;

    crate::kdebug!("ICMP: Sent echo request to {} seq={}", target, sequence);
    Ok(sequence)
}

/// Send ping with callback
pub fn ping_async(
    target: Ipv4Address,
    payload: &[u8],
    callback: fn(Ipv4Address, u16, u64),
) -> Result<u16, NetError> {
    let identifier = PING_ID.load(Ordering::Relaxed);
    let sequence = PING_SEQUENCE.fetch_add(1, Ordering::SeqCst);

    let header = IcmpHeader::echo_request(identifier, sequence);
    send(target, &header, payload)?;

    // Track request with callback
    {
        let mut requests = PING_REQUESTS.lock();
        requests.insert(
            (identifier, sequence),
            PingRequest {
                target,
                identifier,
                sequence,
                sent_time: current_time(),
                callback: Some(callback),
            },
        );
    }

    ICMP_STATS.lock().echo_requests_sent += 1;

    Ok(sequence)
}

/// Send destination unreachable
pub fn send_dest_unreachable(
    dst: Ipv4Address,
    code: u8,
    original_packet: &[u8],
) -> Result<(), NetError> {
    let header = IcmpHeader::dest_unreachable(code);

    // Include IP header + 8 bytes of original datagram
    let payload_len = original_packet.len().min(28);
    let payload = &original_packet[..payload_len];

    send(dst, &header, payload)?;

    ICMP_STATS.lock().dest_unreachable_sent += 1;

    Ok(())
}

/// Send time exceeded
pub fn send_time_exceeded(
    dst: Ipv4Address,
    code: u8,
    original_packet: &[u8],
) -> Result<(), NetError> {
    let header = IcmpHeader::time_exceeded(code);

    // Include IP header + 8 bytes of original datagram
    let payload_len = original_packet.len().min(28);
    let payload = &original_packet[..payload_len];

    send(dst, &header, payload)?;

    ICMP_STATS.lock().time_exceeded_sent += 1;

    Ok(())
}

/// Process received ICMP packet
pub fn receive(ip_header: &Ipv4Header, data: &[u8]) -> Result<(), NetError> {
    if data.len() < 8 {
        ICMP_STATS.lock().errors += 1;
        return Err(NetError::InvalidPacket);
    }

    // Verify checksum
    let checksum = calculate_checksum(data);
    if checksum != 0 && checksum != 0xFFFF {
        crate::kwarn!("ICMP: Invalid checksum");
        ICMP_STATS.lock().errors += 1;
        return Err(NetError::InvalidPacket);
    }

    let icmp_type = data[0];
    let code = data[1];
    let payload = &data[8..];

    match icmp_type {
        types::ECHO_REQUEST => {
            ICMP_STATS.lock().echo_requests_received += 1;
            handle_echo_request(ip_header, data)?;
        }
        types::ECHO_REPLY => {
            ICMP_STATS.lock().echo_replies_received += 1;
            handle_echo_reply(ip_header, data)?;
        }
        types::DEST_UNREACHABLE => {
            ICMP_STATS.lock().dest_unreachable_received += 1;
            handle_dest_unreachable(ip_header, code, payload)?;
        }
        types::TIME_EXCEEDED => {
            ICMP_STATS.lock().time_exceeded_received += 1;
            handle_time_exceeded(ip_header, code, payload)?;
        }
        types::REDIRECT => {
            handle_redirect(ip_header, data)?;
        }
        _ => {
            crate::kdebug!("ICMP: Unknown type {} code {}", icmp_type, code);
        }
    }

    Ok(())
}

/// Handle echo request (respond with echo reply)
fn handle_echo_request(ip_header: &Ipv4Header, data: &[u8]) -> Result<(), NetError> {
    let identifier = u16::from_be_bytes([data[4], data[5]]);
    let sequence = u16::from_be_bytes([data[6], data[7]]);
    let payload = &data[8..];

    crate::kdebug!(
        "ICMP: Echo request from {} id={} seq={}",
        ip_header.source(),
        identifier,
        sequence
    );

    // Build reply with same identifier, sequence, and payload
    let header = IcmpHeader::echo_reply(identifier, sequence);
    send(ip_header.source(), &header, payload)?;

    ICMP_STATS.lock().echo_replies_sent += 1;

    Ok(())
}

/// Handle echo reply
fn handle_echo_reply(ip_header: &Ipv4Header, data: &[u8]) -> Result<(), NetError> {
    let identifier = u16::from_be_bytes([data[4], data[5]]);
    let sequence = u16::from_be_bytes([data[6], data[7]]);

    let rtt = {
        let mut requests = PING_REQUESTS.lock();
        if let Some(request) = requests.remove(&(identifier, sequence)) {
            let rtt = current_time().saturating_sub(request.sent_time);

            if let Some(callback) = request.callback {
                callback(ip_header.source(), sequence, rtt);
            }

            rtt
        } else {
            0
        }
    };

    crate::kinfo!(
        "ICMP: Echo reply from {} id={} seq={} time={}us",
        ip_header.source(),
        identifier,
        sequence,
        rtt
    );

    Ok(())
}

/// Handle destination unreachable
fn handle_dest_unreachable(
    ip_header: &Ipv4Header,
    code: u8,
    _payload: &[u8],
) -> Result<(), NetError> {
    let reason = match code {
        unreachable::NET_UNREACHABLE => "Network unreachable",
        unreachable::HOST_UNREACHABLE => "Host unreachable",
        unreachable::PROTOCOL_UNREACHABLE => "Protocol unreachable",
        unreachable::PORT_UNREACHABLE => "Port unreachable",
        unreachable::FRAGMENTATION_NEEDED => "Fragmentation needed",
        unreachable::SOURCE_ROUTE_FAILED => "Source route failed",
        _ => "Unknown",
    };

    crate::kwarn!(
        "ICMP: Destination unreachable from {} - {}",
        ip_header.source(),
        reason
    );

    // Could notify upper layers (TCP/UDP) about the error

    Ok(())
}

/// Handle time exceeded
fn handle_time_exceeded(ip_header: &Ipv4Header, code: u8, _payload: &[u8]) -> Result<(), NetError> {
    let reason = match code {
        time_exceeded::TTL_EXCEEDED => "TTL exceeded in transit",
        time_exceeded::FRAGMENT_REASSEMBLY => "Fragment reassembly time exceeded",
        _ => "Unknown",
    };

    crate::kwarn!(
        "ICMP: Time exceeded from {} - {}",
        ip_header.source(),
        reason
    );

    Ok(())
}

/// Handle redirect
fn handle_redirect(ip_header: &Ipv4Header, data: &[u8]) -> Result<(), NetError> {
    if data.len() < 8 {
        return Err(NetError::InvalidPacket);
    }

    let gateway = Ipv4Address([data[4], data[5], data[6], data[7]]);

    crate::kinfo!(
        "ICMP: Redirect from {} to gateway {}",
        ip_header.source(),
        gateway
    );

    // Would update routing table here

    Ok(())
}

/// Clean up expired ping requests
pub fn cleanup_requests(timeout_us: u64) {
    let current = current_time();
    let mut requests = PING_REQUESTS.lock();

    let expired: Vec<_> = requests
        .iter()
        .filter(|(_, req)| current.saturating_sub(req.sent_time) > timeout_us)
        .map(|(key, _)| *key)
        .collect();

    for key in expired {
        if let Some(req) = requests.remove(&key) {
            crate::kdebug!(
                "ICMP: Ping to {} seq={} timed out",
                req.target,
                req.sequence
            );
        }
    }
}

/// Get ICMP statistics
pub fn get_stats() -> IcmpStats {
    *ICMP_STATS.lock()
}

/// Initialize ICMP subsystem
pub fn init() {
    // Initialize ping ID from random source if available
    // For now, use a fixed value
    PING_ID.store(0x1234, Ordering::Relaxed);

    crate::kprintln!("  ICMP protocol initialized");
}
