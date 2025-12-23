//! Socket API
//!
//! BSD-style socket interface for applications.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use spin::Mutex;

use super::tcp::{self, SocketAddr};
use super::udp;
use super::{Ipv4Address, NetError};

/// Socket descriptor counter
static SOCKET_COUNTER: AtomicU32 = AtomicU32::new(3); // 0,1,2 reserved

/// Socket types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocketType {
    Stream,   // TCP
    Datagram, // UDP
    Raw,      // Raw IP
}

/// Socket address family
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressFamily {
    Inet,  // IPv4
    Inet6, // IPv6
    Unix,  // Unix domain
}

/// Socket state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SocketState {
    Unbound,
    Bound,
    Listening,
    Connected,
    Closed,
}

/// Socket descriptor
pub type SocketFd = u32;

/// Socket
struct Socket {
    fd: SocketFd,
    family: AddressFamily,
    sock_type: SocketType,
    state: SocketState,
    local_addr: Option<SocketAddr>,
    remote_addr: Option<SocketAddr>,
    tcp_key: Option<tcp::ConnectionKey>,
    blocking: bool,
    recv_timeout: Option<u64>,
    send_timeout: Option<u64>,
}

/// Open sockets
static SOCKETS: Mutex<BTreeMap<SocketFd, Socket>> = Mutex::new(BTreeMap::new());

/// Create a socket
pub fn socket(family: AddressFamily, sock_type: SocketType) -> Result<SocketFd, NetError> {
    let fd = SOCKET_COUNTER.fetch_add(1, Ordering::SeqCst);

    let socket = Socket {
        fd,
        family,
        sock_type,
        state: SocketState::Unbound,
        local_addr: None,
        remote_addr: None,
        tcp_key: None,
        blocking: true,
        recv_timeout: None,
        send_timeout: None,
    };

    SOCKETS.lock().insert(fd, socket);

    crate::kdebug!("socket() = {} ({:?}, {:?})", fd, family, sock_type);
    Ok(fd)
}

/// Bind socket to address
pub fn bind(fd: SocketFd, addr: SocketAddr) -> Result<(), NetError> {
    let mut sockets = SOCKETS.lock();
    let socket = sockets.get_mut(&fd).ok_or(NetError::NotInitialized)?;

    if socket.state != SocketState::Unbound {
        return Err(NetError::InvalidPacket);
    }

    match socket.sock_type {
        SocketType::Datagram => {
            udp::bind(addr)?;
        }
        SocketType::Stream => {
            // TCP binding is implicit
        }
        _ => {}
    }

    socket.local_addr = Some(addr);
    socket.state = SocketState::Bound;

    crate::kdebug!("bind({}) = {}:{}", fd, addr.ip, addr.port);
    Ok(())
}

/// Listen for connections (TCP only)
pub fn listen(fd: SocketFd, backlog: u32) -> Result<(), NetError> {
    let mut sockets = SOCKETS.lock();
    let socket = sockets.get_mut(&fd).ok_or(NetError::NotInitialized)?;

    if socket.sock_type != SocketType::Stream {
        return Err(NetError::InvalidPacket);
    }

    let addr = socket.local_addr.ok_or(NetError::NotInitialized)?;
    tcp::listen(addr)?;

    socket.state = SocketState::Listening;

    crate::kdebug!("listen({}, {})", fd, backlog);
    Ok(())
}

/// Accept connection (TCP only)
pub fn accept(fd: SocketFd) -> Result<(SocketFd, SocketAddr), NetError> {
    let sockets = SOCKETS.lock();
    let socket = sockets.get(&fd).ok_or(NetError::NotInitialized)?;

    if socket.state != SocketState::Listening {
        return Err(NetError::InvalidPacket);
    }

    // In a real implementation, this would wait for an incoming connection
    // and return a new socket for that connection

    Err(NetError::NotInitialized) // No pending connections
}

/// Connect to remote address
pub fn connect(fd: SocketFd, addr: SocketAddr) -> Result<(), NetError> {
    let mut sockets = SOCKETS.lock();
    let socket = sockets.get_mut(&fd).ok_or(NetError::NotInitialized)?;

    match socket.sock_type {
        SocketType::Stream => {
            let key = tcp::connect(addr)?;
            socket.tcp_key = Some(key);
            socket.remote_addr = Some(addr);
            socket.state = SocketState::Connected;
        }
        SocketType::Datagram => {
            // UDP "connect" just sets default destination
            socket.remote_addr = Some(addr);
            socket.state = SocketState::Connected;
        }
        _ => return Err(NetError::InvalidPacket),
    }

    crate::kdebug!("connect({}) -> {}:{}", fd, addr.ip, addr.port);
    Ok(())
}

/// Send data on connected socket
pub fn send(fd: SocketFd, data: &[u8]) -> Result<usize, NetError> {
    let sockets = SOCKETS.lock();
    let socket = sockets.get(&fd).ok_or(NetError::NotInitialized)?;

    let remote = socket.remote_addr.ok_or(NetError::NotInitialized)?;

    match socket.sock_type {
        SocketType::Stream => {
            let key = socket.tcp_key.as_ref().ok_or(NetError::NotInitialized)?;
            tcp::send(key, data)
        }
        SocketType::Datagram => {
            let local = socket.local_addr.unwrap_or(SocketAddr::new(
                Ipv4Address::UNSPECIFIED,
                super::tcp::allocate_port(),
            ));
            udp::send_to(local, remote, data)
        }
        _ => Err(NetError::InvalidPacket),
    }
}

/// Receive data from connected socket
pub fn recv(fd: SocketFd, buf: &mut [u8]) -> Result<usize, NetError> {
    let sockets = SOCKETS.lock();
    let socket = sockets.get(&fd).ok_or(NetError::NotInitialized)?;

    match socket.sock_type {
        SocketType::Stream => {
            let key = socket.tcp_key.as_ref().ok_or(NetError::NotInitialized)?;
            tcp::recv(key, buf)
        }
        SocketType::Datagram => {
            let local = socket.local_addr.ok_or(NetError::NotInitialized)?;
            if let Some(datagram) = udp::recv_from(&local) {
                let len = datagram.data.len().min(buf.len());
                buf[..len].copy_from_slice(&datagram.data[..len]);
                Ok(len)
            } else {
                Ok(0)
            }
        }
        _ => Err(NetError::InvalidPacket),
    }
}

/// Send data to specific address (UDP)
pub fn sendto(fd: SocketFd, data: &[u8], addr: SocketAddr) -> Result<usize, NetError> {
    let sockets = SOCKETS.lock();
    let socket = sockets.get(&fd).ok_or(NetError::NotInitialized)?;

    if socket.sock_type != SocketType::Datagram {
        return Err(NetError::InvalidPacket);
    }

    let local = socket
        .local_addr
        .unwrap_or(SocketAddr::new(Ipv4Address::UNSPECIFIED, 0));

    udp::send_to(local, addr, data)
}

/// Receive data with sender address (UDP)
pub fn recvfrom(fd: SocketFd, buf: &mut [u8]) -> Result<(usize, SocketAddr), NetError> {
    let sockets = SOCKETS.lock();
    let socket = sockets.get(&fd).ok_or(NetError::NotInitialized)?;

    if socket.sock_type != SocketType::Datagram {
        return Err(NetError::InvalidPacket);
    }

    let local = socket.local_addr.ok_or(NetError::NotInitialized)?;

    if let Some(datagram) = udp::recv_from(&local) {
        let len = datagram.data.len().min(buf.len());
        buf[..len].copy_from_slice(&datagram.data[..len]);
        Ok((len, datagram.from))
    } else {
        Err(NetError::NotInitialized)
    }
}

/// Close socket
pub fn close(fd: SocketFd) -> Result<(), NetError> {
    let mut sockets = SOCKETS.lock();
    let socket = sockets.remove(&fd).ok_or(NetError::NotInitialized)?;

    match socket.sock_type {
        SocketType::Stream => {
            if let Some(key) = socket.tcp_key {
                tcp::close(&key)?;
            }
        }
        SocketType::Datagram => {
            if let Some(addr) = socket.local_addr {
                udp::close(&addr);
            }
        }
        _ => {}
    }

    crate::kdebug!("close({})", fd);
    Ok(())
}

/// Set socket option
pub fn setsockopt(fd: SocketFd, level: i32, name: i32, value: &[u8]) -> Result<(), NetError> {
    let mut sockets = SOCKETS.lock();
    let socket = sockets.get_mut(&fd).ok_or(NetError::NotInitialized)?;

    // Handle common options
    match (level, name) {
        (1, 1) => {
            // SO_REUSEADDR - allow reuse of local address
            Ok(())
        }
        (1, 20) => {
            // SO_RCVTIMEO - receive timeout
            if value.len() >= 8 {
                let secs = u32::from_ne_bytes([value[0], value[1], value[2], value[3]]);
                let usecs = u32::from_ne_bytes([value[4], value[5], value[6], value[7]]);
                socket.recv_timeout = Some((secs as u64) * 1_000_000 + (usecs as u64));
            }
            Ok(())
        }
        (1, 21) => {
            // SO_SNDTIMEO - send timeout
            if value.len() >= 8 {
                let secs = u32::from_ne_bytes([value[0], value[1], value[2], value[3]]);
                let usecs = u32::from_ne_bytes([value[4], value[5], value[6], value[7]]);
                socket.send_timeout = Some((secs as u64) * 1_000_000 + (usecs as u64));
            }
            Ok(())
        }
        _ => {
            crate::kdebug!("Unknown socket option: level={}, name={}", level, name);
            Ok(())
        }
    }
}

/// Get socket option
pub fn getsockopt(fd: SocketFd, level: i32, name: i32, buf: &mut [u8]) -> Result<usize, NetError> {
    let sockets = SOCKETS.lock();
    let _socket = sockets.get(&fd).ok_or(NetError::NotInitialized)?;

    // Return default values for most options
    match (level, name) {
        (1, 7) => {
            // SO_ERROR
            if buf.len() >= 4 {
                buf[0..4].copy_from_slice(&0u32.to_ne_bytes());
                Ok(4)
            } else {
                Err(NetError::BufferTooSmall)
            }
        }
        _ => {
            if buf.len() >= 4 {
                buf[0..4].copy_from_slice(&0u32.to_ne_bytes());
                Ok(4)
            } else {
                Err(NetError::BufferTooSmall)
            }
        }
    }
}

/// Shutdown socket
pub fn shutdown(fd: SocketFd, how: i32) -> Result<(), NetError> {
    let sockets = SOCKETS.lock();
    let socket = sockets.get(&fd).ok_or(NetError::NotInitialized)?;

    if socket.sock_type == SocketType::Stream {
        if let Some(ref key) = socket.tcp_key {
            if how == 1 || how == 2 {
                // SHUT_WR or SHUT_RDWR
                tcp::close(key)?;
            }
        }
    }

    Ok(())
}

// ============================================================================
// Syscall Interface Functions
// ============================================================================

/// Create socket from syscall (domain, type, protocol)
pub fn create_socket(domain: i32, sock_type: i32, protocol: i32) -> Result<SocketFd, NetError> {
    let _ = protocol; // Protocol is typically 0 for default

    // Translate domain
    let family = match domain {
        2 => AddressFamily::Inet,   // AF_INET
        10 => AddressFamily::Inet6, // AF_INET6
        1 => AddressFamily::Unix,   // AF_UNIX
        _ => return Err(NetError::InvalidPacket),
    };

    // Translate socket type
    let stype = match sock_type & 0xFF {
        1 => SocketType::Stream,   // SOCK_STREAM
        2 => SocketType::Datagram, // SOCK_DGRAM
        3 => SocketType::Raw,      // SOCK_RAW
        _ => return Err(NetError::InvalidPacket),
    };

    socket(family, stype)
}

/// Bind socket from syscall
pub fn bind_addr(fd: i32, addr_ptr: usize, addrlen: u32) -> Result<(), NetError> {
    if addrlen < 8 {
        return Err(NetError::InvalidPacket);
    }

    // Parse sockaddr_in structure
    let sockaddr = unsafe {
        let ptr = addr_ptr as *const u8;
        let family = u16::from_ne_bytes([*ptr, *ptr.add(1)]);
        let port = u16::from_be_bytes([*ptr.add(2), *ptr.add(3)]);
        let ip_bytes = [*ptr.add(4), *ptr.add(5), *ptr.add(6), *ptr.add(7)];

        (
            family,
            port,
            Ipv4Address::new(ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]),
        )
    };

    let addr = SocketAddr::new(sockaddr.2, sockaddr.1);
    bind(fd as SocketFd, addr)
}

/// Listen from syscall
pub fn listen_socket(fd: i32, backlog: i32) -> Result<(), NetError> {
    listen(fd as SocketFd, backlog as u32)
}

/// Accept from syscall
pub fn accept_socket(fd: i32, addr_ptr: usize, addrlen_ptr: usize) -> Result<SocketFd, NetError> {
    let (new_fd, remote_addr) = accept(fd as SocketFd)?;

    // Write remote address if buffer provided
    if addr_ptr != 0 && addrlen_ptr != 0 {
        unsafe {
            let addrlen = *(addrlen_ptr as *const u32);
            if addrlen >= 8 {
                let ptr = addr_ptr as *mut u8;
                // AF_INET
                *ptr = 2;
                *ptr.add(1) = 0;
                // Port (big endian)
                let port_bytes = remote_addr.port.to_be_bytes();
                *ptr.add(2) = port_bytes[0];
                *ptr.add(3) = port_bytes[1];
                // IP address
                *ptr.add(4) = remote_addr.ip.octets[0];
                *ptr.add(5) = remote_addr.ip.octets[1];
                *ptr.add(6) = remote_addr.ip.octets[2];
                *ptr.add(7) = remote_addr.ip.octets[3];

                *(addrlen_ptr as *mut u32) = 16; // sizeof(sockaddr_in)
            }
        }
    }

    Ok(new_fd)
}

/// Connect from syscall
pub fn connect_socket(fd: i32, addr_ptr: usize, addrlen: u32) -> Result<(), NetError> {
    if addrlen < 8 {
        return Err(NetError::InvalidPacket);
    }

    // Parse sockaddr_in structure
    let addr = unsafe {
        let ptr = addr_ptr as *const u8;
        let port = u16::from_be_bytes([*ptr.add(2), *ptr.add(3)]);
        let ip = Ipv4Address::new(*ptr.add(4), *ptr.add(5), *ptr.add(6), *ptr.add(7));
        SocketAddr::new(ip, port)
    };

    connect(fd as SocketFd, addr)
}

/// Send data from syscall
pub fn send_data(fd: i32, data: &[u8]) -> Result<usize, NetError> {
    send(fd as SocketFd, data)
}

/// Receive data from syscall
pub fn recv_data(fd: i32, buf: &mut [u8]) -> Result<usize, NetError> {
    recv(fd as SocketFd, buf)
}

/// Send to specific address from syscall
pub fn sendto_data(fd: i32, data: &[u8], addr_ptr: usize, addrlen: u32) -> Result<usize, NetError> {
    if addr_ptr == 0 {
        // No address - use connected address
        return send(fd as SocketFd, data);
    }

    if addrlen < 8 {
        return Err(NetError::InvalidPacket);
    }

    let addr = unsafe {
        let ptr = addr_ptr as *const u8;
        let port = u16::from_be_bytes([*ptr.add(2), *ptr.add(3)]);
        let ip = Ipv4Address::new(*ptr.add(4), *ptr.add(5), *ptr.add(6), *ptr.add(7));
        SocketAddr::new(ip, port)
    };

    sendto(fd as SocketFd, data, addr)
}

/// Receive from specific address from syscall
pub fn recvfrom_data(
    fd: i32,
    buf: &mut [u8],
    addr_ptr: usize,
    addrlen_ptr: usize,
) -> Result<usize, NetError> {
    if addr_ptr == 0 {
        // No address buffer - just receive
        return recv(fd as SocketFd, buf);
    }

    let (len, from_addr) = recvfrom(fd as SocketFd, buf)?;

    // Write source address
    if addrlen_ptr != 0 {
        unsafe {
            let addrlen = *(addrlen_ptr as *const u32);
            if addrlen >= 8 {
                let ptr = addr_ptr as *mut u8;
                *ptr = 2; // AF_INET
                *ptr.add(1) = 0;
                let port_bytes = from_addr.port.to_be_bytes();
                *ptr.add(2) = port_bytes[0];
                *ptr.add(3) = port_bytes[1];
                *ptr.add(4) = from_addr.ip.octets[0];
                *ptr.add(5) = from_addr.ip.octets[1];
                *ptr.add(6) = from_addr.ip.octets[2];
                *ptr.add(7) = from_addr.ip.octets[3];

                *(addrlen_ptr as *mut u32) = 16;
            }
        }
    }

    Ok(len)
}

/// Close socket from syscall
pub fn close_socket(fd: i32) -> Result<(), NetError> {
    close(fd as SocketFd)
}
