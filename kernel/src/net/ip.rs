//! Internet Protocol (IPv4)
//!
//! IPv4 packet handling.

use alloc::vec::Vec;

use super::{Ipv4Address, MacAddress, NetError, PacketBuffer};
use super::ethernet;

/// IPv4 header size (without options)
pub const HEADER_SIZE: usize = 20;

/// IPv4 protocol numbers
pub mod protocol {
    pub const ICMP: u8 = 1;
    pub const TCP: u8 = 6;
    pub const UDP: u8 = 17;
}

/// IPv4 header
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct Ipv4Header {
    pub version_ihl: u8,
    pub dscp_ecn: u8,
    pub total_length: [u8; 2],
    pub identification: [u8; 2],
    pub flags_fragment: [u8; 2],
    pub ttl: u8,
    pub protocol: u8,
    pub checksum: [u8; 2],
    pub src_addr: [u8; 4],
    pub dst_addr: [u8; 4],
}

impl Ipv4Header {
    /// Create a new IPv4 header
    pub fn new(src: Ipv4Address, dst: Ipv4Address, protocol: u8, payload_len: usize) -> Self {
        let total_length = (HEADER_SIZE + payload_len) as u16;

        let mut header = Self {
            version_ihl: 0x45, // IPv4, 5 words (20 bytes)
            dscp_ecn: 0,
            total_length: total_length.to_be_bytes(),
            identification: [0, 0],
            flags_fragment: [0x40, 0x00], // Don't fragment
            ttl: 64,
            protocol,
            checksum: [0, 0],
            src_addr: src.0,
            dst_addr: dst.0,
        };

        header.update_checksum();
        header
    }

    /// Get version (should be 4)
    pub fn version(&self) -> u8 {
        self.version_ihl >> 4
    }

    /// Get header length in bytes
    pub fn header_length(&self) -> usize {
        ((self.version_ihl & 0x0F) as usize) * 4
    }

    /// Get total packet length
    pub fn total_length(&self) -> u16 {
        u16::from_be_bytes(self.total_length)
    }

    /// Get payload length
    pub fn payload_length(&self) -> usize {
        self.total_length() as usize - self.header_length()
    }

    /// Get source address
    pub fn source(&self) -> Ipv4Address {
        Ipv4Address(self.src_addr)
    }

    /// Get destination address
    pub fn destination(&self) -> Ipv4Address {
        Ipv4Address(self.dst_addr)
    }

    /// Calculate and update checksum
    pub fn update_checksum(&mut self) {
        self.checksum = [0, 0];

        let bytes = unsafe {
            core::slice::from_raw_parts(
                self as *const _ as *const u8,
                HEADER_SIZE,
            )
        };

        let sum = calculate_checksum(bytes);
        self.checksum = sum.to_be_bytes();
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        let bytes = unsafe {
            core::slice::from_raw_parts(
                self as *const _ as *const u8,
                HEADER_SIZE,
            )
        };

        let mut sum: u32 = 0;
        for i in (0..bytes.len()).step_by(2) {
            let word = if i + 1 < bytes.len() {
                u16::from_be_bytes([bytes[i], bytes[i + 1]])
            } else {
                u16::from_be_bytes([bytes[i], 0])
            };
            sum += word as u32;
        }

        while sum >> 16 != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        sum == 0xFFFF
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        unsafe {
            core::mem::transmute_copy(self)
        }
    }
}

/// Calculate IP checksum
pub fn calculate_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;

    for i in (0..data.len()).step_by(2) {
        let word = if i + 1 < data.len() {
            u16::from_be_bytes([data[i], data[i + 1]])
        } else {
            u16::from_be_bytes([data[i], 0])
        };
        sum += word as u32;
    }

    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    !(sum as u16)
}

/// Parse IPv4 packet
pub fn parse(data: &[u8]) -> Result<(Ipv4Header, &[u8]), NetError> {
    if data.len() < HEADER_SIZE {
        return Err(NetError::InvalidPacket);
    }

    let header = unsafe {
        core::ptr::read_unaligned(data.as_ptr() as *const Ipv4Header)
    };

    if header.version() != 4 {
        return Err(NetError::InvalidPacket);
    }

    if !header.verify_checksum() {
        return Err(NetError::InvalidPacket);
    }

    let header_len = header.header_length();
    if data.len() < header_len {
        return Err(NetError::InvalidPacket);
    }

    let payload = &data[header_len..];

    Ok((header, payload))
}

/// Build IPv4 packet
pub fn build(
    src: Ipv4Address,
    dst: Ipv4Address,
    protocol: u8,
    payload: &[u8],
) -> Vec<u8> {
    let header = Ipv4Header::new(src, dst, protocol, payload.len());

    let mut packet = Vec::with_capacity(HEADER_SIZE + payload.len());
    packet.extend_from_slice(&header.to_bytes());
    packet.extend_from_slice(payload);

    packet
}

/// Send an IP packet
pub fn send(
    dst: Ipv4Address,
    protocol: u8,
    payload: &[u8],
) -> Result<(), NetError> {
    // Find route
    let (iface_idx, next_hop) = super::find_route(dst)
        .ok_or(NetError::NoRoute)?;

    // Get interface
    let iface = super::get_interface(iface_idx)
        .ok_or(NetError::InterfaceNotFound)?;

    // Build IP packet
    let packet = build(iface.config.ip_address, dst, protocol, payload);

    // Resolve MAC address (would use ARP in real implementation)
    let dst_mac = if dst.is_broadcast() {
        MacAddress::BROADCAST
    } else {
        // Simplified: use broadcast for now
        MacAddress::BROADCAST
    };

    // Build Ethernet frame
    let frame = ethernet::build(
        dst_mac,
        iface.mac_address,
        ethernet::ethertype::IPV4,
        &packet,
    )?;

    // Send frame
    ethernet::send(&frame)
}

/// Process received IP packet
pub fn receive(data: &[u8]) -> Result<(), NetError> {
    let (header, payload) = parse(data)?;

    crate::kdebug!(
        "IP: {} -> {} proto={} len={}",
        header.source(),
        header.destination(),
        header.protocol,
        header.payload_length()
    );

    // Dispatch to upper layer
    match header.protocol {
        protocol::ICMP => {
            // Handle ICMP
            handle_icmp(&header, payload)?;
        }
        protocol::TCP => {
            super::tcp::receive(&header, payload)?;
        }
        protocol::UDP => {
            super::udp::receive(&header, payload)?;
        }
        _ => {
            crate::kdebug!("Unknown IP protocol: {}", header.protocol);
        }
    }

    Ok(())
}

/// Handle ICMP packet (ping)
fn handle_icmp(ip_header: &Ipv4Header, data: &[u8]) -> Result<(), NetError> {
    if data.len() < 8 {
        return Err(NetError::InvalidPacket);
    }

    let icmp_type = data[0];
    let icmp_code = data[1];

    match (icmp_type, icmp_code) {
        (8, 0) => {
            // Echo request (ping)
            crate::kdebug!("ICMP Echo Request from {}", ip_header.source());

            // Build echo reply
            let mut reply = data.to_vec();
            reply[0] = 0; // Echo Reply

            // Recalculate ICMP checksum
            reply[2] = 0;
            reply[3] = 0;
            let checksum = calculate_checksum(&reply);
            reply[2] = (checksum >> 8) as u8;
            reply[3] = (checksum & 0xFF) as u8;

            // Send reply
            send(ip_header.source(), protocol::ICMP, &reply)?;
        }
        (0, 0) => {
            // Echo reply
            crate::kdebug!("ICMP Echo Reply from {}", ip_header.source());
        }
        _ => {
            crate::kdebug!("ICMP type={} code={}", icmp_type, icmp_code);
        }
    }

    Ok(())
}
