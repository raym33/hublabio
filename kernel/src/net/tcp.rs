//! Transmission Control Protocol (TCP)
//!
//! TCP connection handling.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use spin::Mutex;

use super::{Ipv4Address, NetError};
use super::ip::{self, Ipv4Header};

/// TCP header size (without options)
pub const HEADER_SIZE: usize = 20;

/// TCP flags
pub mod flags {
    pub const FIN: u8 = 0x01;
    pub const SYN: u8 = 0x02;
    pub const RST: u8 = 0x04;
    pub const PSH: u8 = 0x08;
    pub const ACK: u8 = 0x10;
    pub const URG: u8 = 0x20;
}

/// TCP connection state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcpState {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

/// TCP header
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct TcpHeader {
    pub src_port: [u8; 2],
    pub dst_port: [u8; 2],
    pub seq_num: [u8; 4],
    pub ack_num: [u8; 4],
    pub data_offset_flags: [u8; 2],
    pub window: [u8; 2],
    pub checksum: [u8; 2],
    pub urgent_ptr: [u8; 2],
}

impl TcpHeader {
    pub fn source_port(&self) -> u16 {
        u16::from_be_bytes(self.src_port)
    }

    pub fn dest_port(&self) -> u16 {
        u16::from_be_bytes(self.dst_port)
    }

    pub fn sequence(&self) -> u32 {
        u32::from_be_bytes(self.seq_num)
    }

    pub fn acknowledgment(&self) -> u32 {
        u32::from_be_bytes(self.ack_num)
    }

    pub fn data_offset(&self) -> usize {
        ((self.data_offset_flags[0] >> 4) as usize) * 4
    }

    pub fn flags(&self) -> u8 {
        self.data_offset_flags[1]
    }

    pub fn window_size(&self) -> u16 {
        u16::from_be_bytes(self.window)
    }

    pub fn set_source_port(&mut self, port: u16) {
        self.src_port = port.to_be_bytes();
    }

    pub fn set_dest_port(&mut self, port: u16) {
        self.dst_port = port.to_be_bytes();
    }

    pub fn set_sequence(&mut self, seq: u32) {
        self.seq_num = seq.to_be_bytes();
    }

    pub fn set_acknowledgment(&mut self, ack: u32) {
        self.ack_num = ack.to_be_bytes();
    }

    pub fn set_flags(&mut self, data_offset: u8, tcp_flags: u8) {
        self.data_offset_flags = [(data_offset << 4), tcp_flags];
    }

    pub fn set_window(&mut self, window: u16) {
        self.window = window.to_be_bytes();
    }
}

/// Socket address
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SocketAddr {
    pub ip: Ipv4Address,
    pub port: u16,
}

impl SocketAddr {
    pub fn new(ip: Ipv4Address, port: u16) -> Self {
        Self { ip, port }
    }
}

/// TCP connection key
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ConnectionKey {
    local: SocketAddr,
    remote: SocketAddr,
}

/// TCP connection
struct TcpConnection {
    state: TcpState,
    local: SocketAddr,
    remote: SocketAddr,
    send_next: u32,      // Next sequence number to send
    send_unack: u32,     // Oldest unacknowledged sequence
    recv_next: u32,      // Next expected sequence number
    recv_window: u16,    // Receive window size
    send_window: u16,    // Send window size
    send_buffer: VecDeque<u8>,
    recv_buffer: VecDeque<u8>,
    // Retransmission
    rto: u64,            // Retransmission timeout (ms)
    srtt: u64,           // Smoothed RTT
    rttvar: u64,         // RTT variance
    retries: u32,        // Retransmission count
    last_sent: u64,      // Time of last send
    // Congestion control
    cwnd: u32,           // Congestion window
    ssthresh: u32,       // Slow start threshold
    // Statistics
    bytes_sent: u64,
    bytes_received: u64,
    retransmits: u64,
}

/// Initial RTO (1 second)
const INITIAL_RTO: u64 = 1000;
/// Minimum RTO (200ms)
const MIN_RTO: u64 = 200;
/// Maximum RTO (120 seconds)
const MAX_RTO: u64 = 120_000;
/// Maximum retries before giving up
const MAX_RETRIES: u32 = 15;
/// Initial congestion window (MSS * 10)
const INITIAL_CWND: u32 = 14600;
/// Initial slow start threshold
const INITIAL_SSTHRESH: u32 = 65535;
/// Maximum segment size
const MSS: u32 = 1460;

impl TcpConnection {
    fn new(local: SocketAddr, remote: SocketAddr) -> Self {
        Self {
            state: TcpState::Closed,
            local,
            remote,
            send_next: generate_isn(),
            send_unack: 0,
            recv_next: 0,
            recv_window: 65535,
            send_window: 65535,
            send_buffer: VecDeque::new(),
            recv_buffer: VecDeque::new(),
            rto: INITIAL_RTO,
            srtt: 0,
            rttvar: 0,
            retries: 0,
            last_sent: 0,
            cwnd: INITIAL_CWND,
            ssthresh: INITIAL_SSTHRESH,
            bytes_sent: 0,
            bytes_received: 0,
            retransmits: 0,
        }
    }

    /// Update RTT measurements (RFC 6298)
    fn update_rtt(&mut self, measured_rtt: u64) {
        if self.srtt == 0 {
            // First measurement
            self.srtt = measured_rtt;
            self.rttvar = measured_rtt / 2;
        } else {
            // Subsequent measurements
            // RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - R'|
            // where beta = 1/4
            let diff = if self.srtt > measured_rtt {
                self.srtt - measured_rtt
            } else {
                measured_rtt - self.srtt
            };
            self.rttvar = (3 * self.rttvar + diff) / 4;

            // SRTT = (1 - alpha) * SRTT + alpha * R'
            // where alpha = 1/8
            self.srtt = (7 * self.srtt + measured_rtt) / 8;
        }

        // RTO = SRTT + max(G, K * RTTVAR)
        // where K = 4, G = clock granularity (1ms)
        self.rto = self.srtt + (4 * self.rttvar).max(1);
        self.rto = self.rto.clamp(MIN_RTO, MAX_RTO);
    }

    /// Handle retransmission timeout
    fn on_timeout(&mut self) {
        self.retries += 1;
        self.retransmits += 1;

        // Exponential backoff
        self.rto = (self.rto * 2).min(MAX_RTO);

        // Congestion control: reduce ssthresh and cwnd
        self.ssthresh = (self.cwnd / 2).max(2 * MSS);
        self.cwnd = MSS;
    }

    /// Handle successful ACK (congestion control)
    fn on_ack(&mut self, acked_bytes: u32) {
        self.retries = 0;

        if self.cwnd < self.ssthresh {
            // Slow start: increase cwnd by 1 MSS per ACK
            self.cwnd += MSS;
        } else {
            // Congestion avoidance: increase cwnd by MSS * MSS / cwnd per ACK
            self.cwnd += (MSS * MSS) / self.cwnd;
        }
    }

    /// Handle triple duplicate ACK (fast retransmit)
    fn on_triple_dup_ack(&mut self) {
        // Fast retransmit/fast recovery
        self.ssthresh = (self.cwnd / 2).max(2 * MSS);
        self.cwnd = self.ssthresh + 3 * MSS;
    }

    /// Get effective window size
    fn effective_window(&self) -> u32 {
        (self.send_window as u32).min(self.cwnd)
    }
}

/// Generate initial sequence number
fn generate_isn() -> u32 {
    static ISN_COUNTER: AtomicU32 = AtomicU32::new(0x12345678);
    ISN_COUNTER.fetch_add(64000, Ordering::SeqCst)
}

/// Active TCP connections
static CONNECTIONS: Mutex<BTreeMap<ConnectionKey, TcpConnection>> =
    Mutex::new(BTreeMap::new());

/// Listening sockets
static LISTENERS: Mutex<BTreeMap<SocketAddr, VecDeque<ConnectionKey>>> =
    Mutex::new(BTreeMap::new());

/// Ephemeral port counter
static NEXT_PORT: AtomicU32 = AtomicU32::new(49152);

/// Allocate ephemeral port
fn allocate_port() -> u16 {
    let port = NEXT_PORT.fetch_add(1, Ordering::SeqCst);
    if port > 65535 {
        NEXT_PORT.store(49152, Ordering::SeqCst);
    }
    port as u16
}

/// Calculate TCP checksum
fn calculate_checksum(
    src_ip: Ipv4Address,
    dst_ip: Ipv4Address,
    tcp_data: &[u8],
) -> u16 {
    let mut sum: u32 = 0;

    // Pseudo-header
    sum += u16::from_be_bytes([src_ip.0[0], src_ip.0[1]]) as u32;
    sum += u16::from_be_bytes([src_ip.0[2], src_ip.0[3]]) as u32;
    sum += u16::from_be_bytes([dst_ip.0[0], dst_ip.0[1]]) as u32;
    sum += u16::from_be_bytes([dst_ip.0[2], dst_ip.0[3]]) as u32;
    sum += ip::protocol::TCP as u32;
    sum += tcp_data.len() as u32;

    // TCP data
    for i in (0..tcp_data.len()).step_by(2) {
        let word = if i + 1 < tcp_data.len() {
            u16::from_be_bytes([tcp_data[i], tcp_data[i + 1]])
        } else {
            u16::from_be_bytes([tcp_data[i], 0])
        };
        sum += word as u32;
    }

    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    !(sum as u16)
}

/// Send TCP segment
fn send_segment(
    local: SocketAddr,
    remote: SocketAddr,
    seq: u32,
    ack: u32,
    tcp_flags: u8,
    payload: &[u8],
) -> Result<(), NetError> {
    let mut header = TcpHeader {
        src_port: [0; 2],
        dst_port: [0; 2],
        seq_num: [0; 4],
        ack_num: [0; 4],
        data_offset_flags: [0; 2],
        window: [0; 2],
        checksum: [0; 2],
        urgent_ptr: [0; 2],
    };

    header.set_source_port(local.port);
    header.set_dest_port(remote.port);
    header.set_sequence(seq);
    header.set_acknowledgment(ack);
    header.set_flags(5, tcp_flags); // 5 * 4 = 20 bytes header
    header.set_window(65535);

    // Build segment
    let header_bytes = unsafe {
        core::slice::from_raw_parts(
            &header as *const _ as *const u8,
            HEADER_SIZE,
        )
    };

    let mut segment = Vec::with_capacity(HEADER_SIZE + payload.len());
    segment.extend_from_slice(header_bytes);
    segment.extend_from_slice(payload);

    // Calculate checksum
    let checksum = calculate_checksum(local.ip, remote.ip, &segment);
    segment[16] = (checksum >> 8) as u8;
    segment[17] = (checksum & 0xFF) as u8;

    // Send via IP
    ip::send(remote.ip, ip::protocol::TCP, &segment)
}

/// Create listening socket
pub fn listen(addr: SocketAddr) -> Result<(), NetError> {
    let mut listeners = LISTENERS.lock();
    listeners.insert(addr, VecDeque::new());
    crate::kdebug!("TCP listening on {}", addr.port);
    Ok(())
}

/// Connect to remote host
pub fn connect(remote: SocketAddr) -> Result<ConnectionKey, NetError> {
    let local_port = allocate_port();

    // Get local IP from routing
    let (iface_idx, _) = super::find_route(remote.ip)
        .ok_or(NetError::NoRoute)?;

    let iface = super::get_interface(iface_idx)
        .ok_or(NetError::InterfaceNotFound)?;

    let local = SocketAddr::new(iface.config.ip_address, local_port);
    let key = ConnectionKey { local, remote };

    // Create connection
    let mut conn = TcpConnection::new(local, remote);
    conn.state = TcpState::SynSent;

    // Send SYN
    send_segment(local, remote, conn.send_next, 0, flags::SYN, &[])?;
    conn.send_next += 1;

    CONNECTIONS.lock().insert(key, conn);

    crate::kdebug!("TCP connecting to {}:{}", remote.ip, remote.port);
    Ok(key)
}

/// Send data on connection
pub fn send(key: &ConnectionKey, data: &[u8]) -> Result<usize, NetError> {
    let mut connections = CONNECTIONS.lock();
    let conn = connections.get_mut(key).ok_or(NetError::NotInitialized)?;

    if conn.state != TcpState::Established {
        return Err(NetError::NotInitialized);
    }

    // Add to send buffer
    for &byte in data {
        conn.send_buffer.push_back(byte);
    }

    // Send data (simplified: send all at once)
    let to_send: Vec<u8> = conn.send_buffer.drain(..).collect();

    send_segment(
        conn.local,
        conn.remote,
        conn.send_next,
        conn.recv_next,
        flags::ACK | flags::PSH,
        &to_send,
    )?;

    conn.send_next += to_send.len() as u32;

    Ok(data.len())
}

/// Receive data from connection
pub fn recv(key: &ConnectionKey, buf: &mut [u8]) -> Result<usize, NetError> {
    let mut connections = CONNECTIONS.lock();
    let conn = connections.get_mut(key).ok_or(NetError::NotInitialized)?;

    let available = conn.recv_buffer.len().min(buf.len());
    for i in 0..available {
        buf[i] = conn.recv_buffer.pop_front().unwrap();
    }

    Ok(available)
}

/// Close connection
pub fn close(key: &ConnectionKey) -> Result<(), NetError> {
    let mut connections = CONNECTIONS.lock();
    let conn = connections.get_mut(key).ok_or(NetError::NotInitialized)?;

    if conn.state == TcpState::Established {
        conn.state = TcpState::FinWait1;
        send_segment(
            conn.local,
            conn.remote,
            conn.send_next,
            conn.recv_next,
            flags::FIN | flags::ACK,
            &[],
        )?;
        conn.send_next += 1;
    }

    Ok(())
}

/// Process received TCP segment
pub fn receive(ip_header: &Ipv4Header, data: &[u8]) -> Result<(), NetError> {
    if data.len() < HEADER_SIZE {
        return Err(NetError::InvalidPacket);
    }

    let header = unsafe {
        core::ptr::read_unaligned(data.as_ptr() as *const TcpHeader)
    };

    let remote = SocketAddr::new(ip_header.source(), header.source_port());
    let local = SocketAddr::new(ip_header.destination(), header.dest_port());
    let key = ConnectionKey { local, remote };

    let payload_offset = header.data_offset();
    let payload = if data.len() > payload_offset {
        &data[payload_offset..]
    } else {
        &[]
    };

    crate::kdebug!(
        "TCP: {}:{} -> {}:{} flags={:02X} seq={} ack={} len={}",
        remote.ip, remote.port,
        local.ip, local.port,
        header.flags(),
        header.sequence(),
        header.acknowledgment(),
        payload.len()
    );

    let mut connections = CONNECTIONS.lock();

    // Check existing connection
    if let Some(conn) = connections.get_mut(&key) {
        process_segment(conn, &header, payload)?;
    } else {
        // Check for listening socket
        let mut listeners = LISTENERS.lock();
        let listen_addr = SocketAddr::new(Ipv4Address::UNSPECIFIED, local.port);

        if listeners.contains_key(&listen_addr) || listeners.contains_key(&local) {
            if header.flags() & flags::SYN != 0 {
                // New connection
                let mut conn = TcpConnection::new(local, remote);
                conn.recv_next = header.sequence() + 1;
                conn.state = TcpState::SynReceived;

                // Send SYN-ACK
                drop(listeners);
                send_segment(
                    local,
                    remote,
                    conn.send_next,
                    conn.recv_next,
                    flags::SYN | flags::ACK,
                    &[],
                )?;
                conn.send_next += 1;

                connections.insert(key, conn);
                crate::kdebug!("TCP: New connection from {}:{}", remote.ip, remote.port);
            }
        } else {
            // Send RST
            send_segment(local, remote, 0, header.sequence() + 1, flags::RST | flags::ACK, &[])?;
        }
    }

    Ok(())
}

/// Process TCP segment for existing connection
fn process_segment(
    conn: &mut TcpConnection,
    header: &TcpHeader,
    payload: &[u8],
) -> Result<(), NetError> {
    let flags = header.flags();

    match conn.state {
        TcpState::SynSent => {
            if flags & (flags::SYN | flags::ACK) == (flags::SYN | flags::ACK) {
                conn.recv_next = header.sequence() + 1;
                conn.send_unack = header.acknowledgment();
                conn.state = TcpState::Established;

                // Send ACK
                send_segment(conn.local, conn.remote, conn.send_next, conn.recv_next, flags::ACK, &[])?;
                crate::kinfo!("TCP: Connection established to {}:{}", conn.remote.ip, conn.remote.port);
            }
        }
        TcpState::SynReceived => {
            if flags & flags::ACK != 0 {
                conn.send_unack = header.acknowledgment();
                conn.state = TcpState::Established;
                crate::kinfo!("TCP: Connection established from {}:{}", conn.remote.ip, conn.remote.port);
            }
        }
        TcpState::Established => {
            // Handle incoming data
            if !payload.is_empty() {
                for &byte in payload {
                    conn.recv_buffer.push_back(byte);
                }
                conn.recv_next += payload.len() as u32;

                // Send ACK
                send_segment(conn.local, conn.remote, conn.send_next, conn.recv_next, flags::ACK, &[])?;
            }

            // Handle ACK
            if flags & flags::ACK != 0 {
                conn.send_unack = header.acknowledgment();
            }

            // Handle FIN
            if flags & flags::FIN != 0 {
                conn.recv_next += 1;
                conn.state = TcpState::CloseWait;
                send_segment(conn.local, conn.remote, conn.send_next, conn.recv_next, flags::ACK, &[])?;
            }
        }
        TcpState::FinWait1 => {
            if flags & flags::ACK != 0 {
                conn.state = TcpState::FinWait2;
            }
            if flags & flags::FIN != 0 {
                conn.recv_next += 1;
                conn.state = TcpState::TimeWait;
                send_segment(conn.local, conn.remote, conn.send_next, conn.recv_next, flags::ACK, &[])?;
            }
        }
        _ => {}
    }

    Ok(())
}

impl core::fmt::Display for SocketAddr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}", self.ip, self.port)
    }
}

/// TCP statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct TcpStats {
    pub connections_active: u64,
    pub connections_passive: u64,
    pub connections_established: u64,
    pub connections_reset: u64,
    pub segments_sent: u64,
    pub segments_received: u64,
    pub retransmits: u64,
    pub errors: u64,
}

static TCP_STATS: Mutex<TcpStats> = Mutex::new(TcpStats {
    connections_active: 0,
    connections_passive: 0,
    connections_established: 0,
    connections_reset: 0,
    segments_sent: 0,
    segments_received: 0,
    retransmits: 0,
    errors: 0,
});

/// Get TCP statistics
pub fn get_stats() -> TcpStats {
    *TCP_STATS.lock()
}

/// Get connection count
pub fn connection_count() -> usize {
    CONNECTIONS.lock().len()
}

/// Get connection state by key
pub fn get_state(key: &ConnectionKey) -> Option<TcpState> {
    CONNECTIONS.lock().get(key).map(|c| c.state)
}

/// Timer tick for TCP (called periodically)
pub fn timer_tick(current_time_ms: u64) {
    let mut connections = CONNECTIONS.lock();
    let mut to_remove = Vec::new();

    for (key, conn) in connections.iter_mut() {
        // Check for timeout in states that expect responses
        match conn.state {
            TcpState::SynSent | TcpState::SynReceived => {
                if conn.retries >= MAX_RETRIES {
                    // Connection timeout
                    to_remove.push(*key);
                    TCP_STATS.lock().errors += 1;
                } else if current_time_ms.saturating_sub(conn.last_sent) > conn.rto {
                    // Retransmit SYN
                    conn.on_timeout();
                    let _ = send_segment(
                        conn.local,
                        conn.remote,
                        conn.send_next - 1, // Resend with same seq
                        conn.recv_next,
                        if conn.state == TcpState::SynSent {
                            flags::SYN
                        } else {
                            flags::SYN | flags::ACK
                        },
                        &[],
                    );
                    conn.last_sent = current_time_ms;
                    TCP_STATS.lock().retransmits += 1;
                }
            }
            TcpState::Established => {
                // Check for data retransmission
                if conn.send_unack < conn.send_next {
                    if conn.retries >= MAX_RETRIES {
                        // Connection failed
                        conn.state = TcpState::Closed;
                        to_remove.push(*key);
                        TCP_STATS.lock().errors += 1;
                    } else if current_time_ms.saturating_sub(conn.last_sent) > conn.rto {
                        // Would retransmit unacknowledged data here
                        conn.on_timeout();
                        conn.last_sent = current_time_ms;
                        TCP_STATS.lock().retransmits += 1;
                    }
                }
            }
            TcpState::TimeWait => {
                // Clean up after 2*MSL (using 2 minutes)
                if current_time_ms.saturating_sub(conn.last_sent) > 120_000 {
                    to_remove.push(*key);
                }
            }
            TcpState::Closed => {
                to_remove.push(*key);
            }
            _ => {}
        }
    }

    // Remove closed/failed connections
    for key in to_remove {
        connections.remove(&key);
    }
}

/// Initialize TCP subsystem
pub fn init() {
    crate::kprintln!("  TCP protocol initialized");
}
