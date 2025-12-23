//! User Datagram Protocol (UDP)
//!
//! Connectionless datagram handling.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use spin::Mutex;

use super::{Ipv4Address, NetError};
use super::ip::{self, Ipv4Header};
use super::tcp::SocketAddr;

/// UDP header size
pub const HEADER_SIZE: usize = 8;

/// UDP header
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct UdpHeader {
    pub src_port: [u8; 2],
    pub dst_port: [u8; 2],
    pub length: [u8; 2],
    pub checksum: [u8; 2],
}

impl UdpHeader {
    pub fn source_port(&self) -> u16 {
        u16::from_be_bytes(self.src_port)
    }

    pub fn dest_port(&self) -> u16 {
        u16::from_be_bytes(self.dst_port)
    }

    pub fn length(&self) -> u16 {
        u16::from_be_bytes(self.length)
    }
}

/// Received datagram
#[derive(Clone, Debug)]
pub struct Datagram {
    pub from: SocketAddr,
    pub data: Vec<u8>,
}

/// UDP socket
struct UdpSocket {
    local: SocketAddr,
    recv_queue: VecDeque<Datagram>,
}

/// Bound UDP sockets
static SOCKETS: Mutex<BTreeMap<SocketAddr, UdpSocket>> = Mutex::new(BTreeMap::new());

/// Calculate UDP checksum
fn calculate_checksum(
    src_ip: Ipv4Address,
    dst_ip: Ipv4Address,
    udp_data: &[u8],
) -> u16 {
    let mut sum: u32 = 0;

    // Pseudo-header
    sum += u16::from_be_bytes([src_ip.0[0], src_ip.0[1]]) as u32;
    sum += u16::from_be_bytes([src_ip.0[2], src_ip.0[3]]) as u32;
    sum += u16::from_be_bytes([dst_ip.0[0], dst_ip.0[1]]) as u32;
    sum += u16::from_be_bytes([dst_ip.0[2], dst_ip.0[3]]) as u32;
    sum += ip::protocol::UDP as u32;
    sum += udp_data.len() as u32;

    // UDP data
    for i in (0..udp_data.len()).step_by(2) {
        let word = if i + 1 < udp_data.len() {
            u16::from_be_bytes([udp_data[i], udp_data[i + 1]])
        } else {
            u16::from_be_bytes([udp_data[i], 0])
        };
        sum += word as u32;
    }

    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    let result = !(sum as u16);
    if result == 0 { 0xFFFF } else { result }
}

/// Bind a UDP socket
pub fn bind(addr: SocketAddr) -> Result<(), NetError> {
    let mut sockets = SOCKETS.lock();

    if sockets.contains_key(&addr) {
        return Err(NetError::AddrInUse);
    }

    sockets.insert(addr, UdpSocket {
        local: addr,
        recv_queue: VecDeque::new(),
    });

    crate::kdebug!("UDP bound to {}:{}", addr.ip, addr.port);
    Ok(())
}

/// Close UDP socket
pub fn close(addr: &SocketAddr) {
    SOCKETS.lock().remove(addr);
}

/// Send UDP datagram
pub fn send_to(
    from: SocketAddr,
    to: SocketAddr,
    data: &[u8],
) -> Result<usize, NetError> {
    let total_len = HEADER_SIZE + data.len();

    // Build header
    let mut segment = Vec::with_capacity(total_len);

    // Source port
    segment.extend_from_slice(&from.port.to_be_bytes());
    // Destination port
    segment.extend_from_slice(&to.port.to_be_bytes());
    // Length
    segment.extend_from_slice(&(total_len as u16).to_be_bytes());
    // Checksum placeholder
    segment.extend_from_slice(&[0, 0]);
    // Payload
    segment.extend_from_slice(data);

    // Calculate checksum
    let checksum = calculate_checksum(from.ip, to.ip, &segment);
    segment[6] = (checksum >> 8) as u8;
    segment[7] = (checksum & 0xFF) as u8;

    // Send via IP
    ip::send(to.ip, ip::protocol::UDP, &segment)?;

    crate::kdebug!("UDP: sent {} bytes to {}:{}", data.len(), to.ip, to.port);
    Ok(data.len())
}

/// Receive UDP datagram
pub fn recv_from(addr: &SocketAddr) -> Option<Datagram> {
    SOCKETS.lock().get_mut(addr)?.recv_queue.pop_front()
}

/// Process received UDP datagram
pub fn receive(ip_header: &Ipv4Header, data: &[u8]) -> Result<(), NetError> {
    if data.len() < HEADER_SIZE {
        return Err(NetError::InvalidPacket);
    }

    let header = unsafe {
        core::ptr::read_unaligned(data.as_ptr() as *const UdpHeader)
    };

    let payload = &data[HEADER_SIZE..];

    let from = SocketAddr::new(ip_header.source(), header.source_port());
    let to = SocketAddr::new(ip_header.destination(), header.dest_port());

    crate::kdebug!(
        "UDP: {}:{} -> {}:{} len={}",
        from.ip, from.port,
        to.ip, to.port,
        payload.len()
    );

    // Find matching socket
    let mut sockets = SOCKETS.lock();

    // Try exact match first
    if let Some(socket) = sockets.get_mut(&to) {
        socket.recv_queue.push_back(Datagram {
            from,
            data: payload.to_vec(),
        });
        return Ok(());
    }

    // Try wildcard match (UNSPECIFIED IP)
    let wildcard = SocketAddr::new(Ipv4Address::UNSPECIFIED, to.port);
    if let Some(socket) = sockets.get_mut(&wildcard) {
        socket.recv_queue.push_back(Datagram {
            from,
            data: payload.to_vec(),
        });
        return Ok(());
    }

    // No socket found, silently drop
    crate::kdebug!("UDP: no socket for port {}", to.port);
    Ok(())
}
