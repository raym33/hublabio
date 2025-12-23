//! Unix Domain Sockets
//!
//! Local IPC sockets for inter-process communication.
//! Supports stream (SOCK_STREAM) and datagram (SOCK_DGRAM) types.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use spin::{Mutex, RwLock};

use crate::process::Pid;
use crate::auth::{Uid, Gid};

/// Maximum socket path length
pub const UNIX_PATH_MAX: usize = 108;

/// Socket buffer size
pub const SOCKET_BUFFER_SIZE: usize = 65536;

/// Unix socket address
#[derive(Clone, Debug)]
pub struct UnixAddr {
    /// Socket path (empty for unnamed)
    pub path: String,
    /// Abstract namespace (starts with null byte)
    pub abstract_ns: bool,
}

impl UnixAddr {
    pub fn unnamed() -> Self {
        Self {
            path: String::new(),
            abstract_ns: false,
        }
    }

    pub fn pathname(path: &str) -> Self {
        Self {
            path: String::from(path),
            abstract_ns: false,
        }
    }

    pub fn abstract_name(name: &str) -> Self {
        Self {
            path: String::from(name),
            abstract_ns: true,
        }
    }

    pub fn is_unnamed(&self) -> bool {
        self.path.is_empty() && !self.abstract_ns
    }

    /// Convert to sockaddr_un bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; 2 + UNIX_PATH_MAX];
        buf[0] = 1; // AF_UNIX (low byte)
        buf[1] = 0; // AF_UNIX (high byte)

        if self.abstract_ns {
            // Abstract: starts with null byte
            let path_bytes = self.path.as_bytes();
            let len = path_bytes.len().min(UNIX_PATH_MAX - 1);
            buf[3..3 + len].copy_from_slice(&path_bytes[..len]);
        } else {
            let path_bytes = self.path.as_bytes();
            let len = path_bytes.len().min(UNIX_PATH_MAX - 1);
            buf[2..2 + len].copy_from_slice(&path_bytes[..len]);
        }

        buf
    }

    /// Parse from sockaddr_un bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 2 {
            return None;
        }

        // Check AF_UNIX
        let family = u16::from_ne_bytes([bytes[0], bytes[1]]);
        if family != 1 {
            return None;
        }

        if bytes.len() <= 2 {
            return Some(Self::unnamed());
        }

        // Check for abstract namespace
        if bytes[2] == 0 && bytes.len() > 3 {
            let end = bytes[3..].iter().position(|&b| b == 0).unwrap_or(bytes.len() - 3);
            let name = String::from_utf8_lossy(&bytes[3..3 + end]).into_owned();
            return Some(Self::abstract_name(&name));
        }

        // Regular pathname
        let end = bytes[2..].iter().position(|&b| b == 0).unwrap_or(bytes.len() - 2);
        let path = String::from_utf8_lossy(&bytes[2..2 + end]).into_owned();
        Some(Self::pathname(&path))
    }
}

/// Socket type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocketType {
    /// Connection-oriented stream
    Stream,
    /// Connectionless datagrams
    Dgram,
    /// Sequential packets
    SeqPacket,
}

/// Socket state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocketState {
    /// Initial state
    Unbound,
    /// Bound to address
    Bound,
    /// Listening for connections
    Listening,
    /// Connected
    Connected,
    /// Connecting (non-blocking)
    Connecting,
    /// Closed
    Closed,
}

/// Credentials passed with socket
#[derive(Clone, Copy, Debug)]
pub struct UCredentials {
    pub pid: i32,
    pub uid: Uid,
    pub gid: Gid,
}

/// Ancillary data (control messages)
#[derive(Clone, Debug)]
pub enum AncillaryData {
    /// File descriptors (SCM_RIGHTS)
    Rights(Vec<i32>),
    /// Credentials (SCM_CREDENTIALS)
    Credentials(UCredentials),
}

/// Message for datagram/seqpacket
#[derive(Clone)]
struct Message {
    data: Vec<u8>,
    from: Option<UnixAddr>,
    ancillary: Vec<AncillaryData>,
}

/// Unix socket
pub struct UnixSocket {
    /// Socket ID
    pub id: u64,
    /// Socket type
    pub socket_type: SocketType,
    /// Current state
    state: Mutex<SocketState>,
    /// Local address
    local_addr: RwLock<Option<UnixAddr>>,
    /// Peer address (connected)
    peer_addr: RwLock<Option<UnixAddr>>,
    /// Connected peer socket
    peer: RwLock<Option<Arc<UnixSocket>>>,
    /// Owning process
    pub owner: Pid,
    /// Owner credentials
    pub creds: UCredentials,

    // Stream socket buffers
    /// Receive buffer
    recv_buffer: Mutex<VecDeque<u8>>,
    /// Send buffer (peer's recv)
    send_buffer: Mutex<VecDeque<u8>>,

    // Datagram socket buffers
    /// Message queue
    messages: Mutex<VecDeque<Message>>,

    // Listening socket
    /// Pending connections
    pending: Mutex<VecDeque<Arc<UnixSocket>>>,
    /// Backlog size
    backlog: AtomicU32,

    /// Wait queue for blocking operations
    waiters: crate::waitqueue::WaitQueue,

    /// Socket options
    options: Mutex<SocketOptions>,

    /// Reference count
    refcount: AtomicU32,

    /// Shutdown flags
    shutdown_read: AtomicBool,
    shutdown_write: AtomicBool,
}

#[derive(Clone, Debug, Default)]
struct SocketOptions {
    /// Pass credentials (SO_PASSCRED)
    passcred: bool,
    /// Receive buffer size
    rcvbuf: usize,
    /// Send buffer size
    sndbuf: usize,
    /// Non-blocking
    nonblock: bool,
    /// Close on exec
    cloexec: bool,
}

impl UnixSocket {
    /// Create new socket
    pub fn new(socket_type: SocketType, owner: Pid) -> Arc<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        let creds = if let Some(proc) = crate::process::get(owner) {
            UCredentials {
                pid: owner.0 as i32,
                uid: proc.uid,
                gid: proc.gid,
            }
        } else {
            UCredentials { pid: 0, uid: 0, gid: 0 }
        };

        Arc::new(Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            socket_type,
            state: Mutex::new(SocketState::Unbound),
            local_addr: RwLock::new(None),
            peer_addr: RwLock::new(None),
            peer: RwLock::new(None),
            owner,
            creds,
            recv_buffer: Mutex::new(VecDeque::with_capacity(SOCKET_BUFFER_SIZE)),
            send_buffer: Mutex::new(VecDeque::with_capacity(SOCKET_BUFFER_SIZE)),
            messages: Mutex::new(VecDeque::with_capacity(64)),
            pending: Mutex::new(VecDeque::new()),
            backlog: AtomicU32::new(0),
            waiters: crate::waitqueue::WaitQueue::new(),
            options: Mutex::new(SocketOptions {
                rcvbuf: SOCKET_BUFFER_SIZE,
                sndbuf: SOCKET_BUFFER_SIZE,
                ..Default::default()
            }),
            refcount: AtomicU32::new(1),
            shutdown_read: AtomicBool::new(false),
            shutdown_write: AtomicBool::new(false),
        })
    }

    /// Bind socket to address
    pub fn bind(self: &Arc<Self>, addr: &UnixAddr) -> Result<(), SocketError> {
        let mut state = self.state.lock();

        if *state != SocketState::Unbound {
            return Err(SocketError::InvalidState);
        }

        // Check if address is already in use
        if !addr.is_unnamed() {
            if BOUND_SOCKETS.read().contains_key(&addr.path) {
                return Err(SocketError::AddrInUse);
            }

            // Register in namespace
            BOUND_SOCKETS.write().insert(addr.path.clone(), self.clone());
        }

        *self.local_addr.write() = Some(addr.clone());
        *state = SocketState::Bound;

        Ok(())
    }

    /// Listen for connections (stream only)
    pub fn listen(self: &Arc<Self>, backlog: i32) -> Result<(), SocketError> {
        if self.socket_type != SocketType::Stream {
            return Err(SocketError::OpNotSupported);
        }

        let mut state = self.state.lock();

        if *state != SocketState::Bound {
            return Err(SocketError::InvalidState);
        }

        self.backlog.store(backlog.max(1) as u32, Ordering::SeqCst);
        *state = SocketState::Listening;

        Ok(())
    }

    /// Accept connection
    pub fn accept(self: &Arc<Self>, nonblock: bool) -> Result<Arc<UnixSocket>, SocketError> {
        if self.socket_type != SocketType::Stream {
            return Err(SocketError::OpNotSupported);
        }

        loop {
            {
                let state = self.state.lock();
                if *state != SocketState::Listening {
                    return Err(SocketError::InvalidState);
                }
            }

            // Check pending connections
            if let Some(peer) = self.pending.lock().pop_front() {
                // Create connected socket pair
                let new_sock = UnixSocket::new(SocketType::Stream, self.owner);

                *new_sock.state.lock() = SocketState::Connected;
                *new_sock.peer.write() = Some(peer.clone());
                *new_sock.peer_addr.write() = peer.local_addr.read().clone();
                *new_sock.local_addr.write() = self.local_addr.read().clone();

                *peer.state.lock() = SocketState::Connected;
                *peer.peer.write() = Some(new_sock.clone());
                peer.waiters.wake_all();

                return Ok(new_sock);
            }

            if nonblock || self.options.lock().nonblock {
                return Err(SocketError::WouldBlock);
            }

            // Wait for connection
            self.waiters.wait();
        }
    }

    /// Connect to listening socket
    pub fn connect(self: &Arc<Self>, addr: &UnixAddr) -> Result<(), SocketError> {
        if self.socket_type == SocketType::Dgram {
            // Datagram: just set peer address
            *self.peer_addr.write() = Some(addr.clone());
            *self.state.lock() = SocketState::Connected;
            return Ok(());
        }

        // Stream: actual connection
        let mut state = self.state.lock();

        match *state {
            SocketState::Unbound | SocketState::Bound => {}
            SocketState::Connected => return Err(SocketError::IsConnected),
            _ => return Err(SocketError::InvalidState),
        }

        // Find target socket
        let target = BOUND_SOCKETS.read()
            .get(&addr.path)
            .cloned()
            .ok_or(SocketError::ConnRefused)?;

        {
            let target_state = target.state.lock();
            if *target_state != SocketState::Listening {
                return Err(SocketError::ConnRefused);
            }
        }

        // Check backlog
        let pending_count = target.pending.lock().len() as u32;
        if pending_count >= target.backlog.load(Ordering::SeqCst) {
            return Err(SocketError::ConnRefused);
        }

        // Add to pending
        *state = SocketState::Connecting;
        *self.peer_addr.write() = Some(addr.clone());
        target.pending.lock().push_back(self.clone());
        target.waiters.wake_one();

        drop(state);

        // Wait for accept
        let nonblock = self.options.lock().nonblock;
        if !nonblock {
            loop {
                let state = self.state.lock();
                if *state == SocketState::Connected {
                    return Ok(());
                }
                drop(state);
                self.waiters.wait();
            }
        }

        Ok(())
    }

    /// Send data (stream)
    pub fn send(&self, data: &[u8], flags: i32) -> Result<usize, SocketError> {
        if self.shutdown_write.load(Ordering::SeqCst) {
            return Err(SocketError::BrokenPipe);
        }

        let state = self.state.lock();
        if *state != SocketState::Connected {
            return Err(SocketError::NotConnected);
        }
        drop(state);

        let peer = self.peer.read().clone().ok_or(SocketError::NotConnected)?;

        // Add to peer's receive buffer
        let mut recv_buf = peer.recv_buffer.lock();
        let available = peer.options.lock().rcvbuf - recv_buf.len();

        if available == 0 {
            if flags & MSG_DONTWAIT != 0 || self.options.lock().nonblock {
                return Err(SocketError::WouldBlock);
            }
            // Would need to wait for space
        }

        let to_send = data.len().min(available);
        recv_buf.extend(&data[..to_send]);

        peer.waiters.wake_all();

        Ok(to_send)
    }

    /// Receive data (stream)
    pub fn recv(&self, buf: &mut [u8], flags: i32) -> Result<usize, SocketError> {
        if self.shutdown_read.load(Ordering::SeqCst) {
            return Ok(0); // EOF
        }

        loop {
            let state = self.state.lock();
            if *state != SocketState::Connected {
                // Check if peer closed
                if self.recv_buffer.lock().is_empty() {
                    return Ok(0); // EOF
                }
            }
            drop(state);

            let mut recv_buf = self.recv_buffer.lock();

            if !recv_buf.is_empty() {
                let to_read = buf.len().min(recv_buf.len());

                if flags & MSG_PEEK != 0 {
                    // Don't remove data
                    for (i, &byte) in recv_buf.iter().take(to_read).enumerate() {
                        buf[i] = byte;
                    }
                } else {
                    for i in 0..to_read {
                        buf[i] = recv_buf.pop_front().unwrap();
                    }
                }

                // Wake sender if they were blocked
                if let Some(peer) = self.peer.read().as_ref() {
                    peer.waiters.wake_all();
                }

                return Ok(to_read);
            }

            drop(recv_buf);

            if flags & MSG_DONTWAIT != 0 || self.options.lock().nonblock {
                return Err(SocketError::WouldBlock);
            }

            self.waiters.wait();
        }
    }

    /// Send datagram
    pub fn sendto(&self, data: &[u8], addr: Option<&UnixAddr>) -> Result<usize, SocketError> {
        if self.socket_type != SocketType::Dgram {
            return Err(SocketError::OpNotSupported);
        }

        let target_addr = addr
            .cloned()
            .or_else(|| self.peer_addr.read().clone())
            .ok_or(SocketError::DestAddrRequired)?;

        // Find target socket
        let target = BOUND_SOCKETS.read()
            .get(&target_addr.path)
            .cloned()
            .ok_or(SocketError::ConnRefused)?;

        // Create message
        let msg = Message {
            data: data.to_vec(),
            from: self.local_addr.read().clone(),
            ancillary: Vec::new(),
        };

        let mut messages = target.messages.lock();
        if messages.len() >= 64 {
            return Err(SocketError::WouldBlock);
        }

        messages.push_back(msg);
        target.waiters.wake_one();

        Ok(data.len())
    }

    /// Receive datagram
    pub fn recvfrom(&self, buf: &mut [u8], flags: i32) -> Result<(usize, Option<UnixAddr>), SocketError> {
        if self.socket_type != SocketType::Dgram {
            return Err(SocketError::OpNotSupported);
        }

        loop {
            let mut messages = self.messages.lock();

            if let Some(msg) = if flags & MSG_PEEK != 0 {
                messages.front().cloned()
            } else {
                messages.pop_front()
            } {
                let to_read = buf.len().min(msg.data.len());
                buf[..to_read].copy_from_slice(&msg.data[..to_read]);
                return Ok((to_read, msg.from));
            }

            drop(messages);

            if flags & MSG_DONTWAIT != 0 || self.options.lock().nonblock {
                return Err(SocketError::WouldBlock);
            }

            self.waiters.wait();
        }
    }

    /// Shutdown socket
    pub fn shutdown(&self, how: i32) -> Result<(), SocketError> {
        match how {
            SHUT_RD => self.shutdown_read.store(true, Ordering::SeqCst),
            SHUT_WR => {
                self.shutdown_write.store(true, Ordering::SeqCst);
                if let Some(peer) = self.peer.read().as_ref() {
                    peer.waiters.wake_all();
                }
            }
            SHUT_RDWR => {
                self.shutdown_read.store(true, Ordering::SeqCst);
                self.shutdown_write.store(true, Ordering::SeqCst);
                if let Some(peer) = self.peer.read().as_ref() {
                    peer.waiters.wake_all();
                }
            }
            _ => return Err(SocketError::Invalid),
        }
        Ok(())
    }

    /// Get socket option
    pub fn getsockopt(&self, level: i32, name: i32) -> Result<i32, SocketError> {
        let opts = self.options.lock();

        match (level, name) {
            (SOL_SOCKET, SO_RCVBUF) => Ok(opts.rcvbuf as i32),
            (SOL_SOCKET, SO_SNDBUF) => Ok(opts.sndbuf as i32),
            (SOL_SOCKET, SO_PASSCRED) => Ok(if opts.passcred { 1 } else { 0 }),
            (SOL_SOCKET, SO_TYPE) => Ok(self.socket_type as i32),
            (SOL_SOCKET, SO_ERROR) => Ok(0),
            _ => Err(SocketError::Invalid),
        }
    }

    /// Set socket option
    pub fn setsockopt(&self, level: i32, name: i32, value: i32) -> Result<(), SocketError> {
        let mut opts = self.options.lock();

        match (level, name) {
            (SOL_SOCKET, SO_RCVBUF) => {
                opts.rcvbuf = (value as usize).clamp(1024, 1024 * 1024);
            }
            (SOL_SOCKET, SO_SNDBUF) => {
                opts.sndbuf = (value as usize).clamp(1024, 1024 * 1024);
            }
            (SOL_SOCKET, SO_PASSCRED) => {
                opts.passcred = value != 0;
            }
            _ => return Err(SocketError::Invalid),
        }

        Ok(())
    }

    /// Get peer credentials
    pub fn peercred(&self) -> Result<UCredentials, SocketError> {
        let peer = self.peer.read().clone().ok_or(SocketError::NotConnected)?;
        Ok(peer.creds)
    }

    /// Check if readable
    pub fn is_readable(&self) -> bool {
        if self.shutdown_read.load(Ordering::SeqCst) {
            return true; // EOF is readable
        }

        match self.socket_type {
            SocketType::Stream => !self.recv_buffer.lock().is_empty(),
            SocketType::Dgram | SocketType::SeqPacket => !self.messages.lock().is_empty(),
        }
    }

    /// Check if writable
    pub fn is_writable(&self) -> bool {
        if self.shutdown_write.load(Ordering::SeqCst) {
            return false;
        }

        if let Some(peer) = self.peer.read().as_ref() {
            let recv_buf = peer.recv_buffer.lock();
            recv_buf.len() < peer.options.lock().rcvbuf
        } else {
            self.socket_type == SocketType::Dgram
        }
    }

    /// Close socket
    pub fn close(self: &Arc<Self>) {
        // Notify peer
        if let Some(peer) = self.peer.write().take() {
            *peer.peer.write() = None;
            peer.waiters.wake_all();
        }

        // Remove from bound sockets
        if let Some(addr) = self.local_addr.read().as_ref() {
            if !addr.is_unnamed() {
                BOUND_SOCKETS.write().remove(&addr.path);
            }
        }

        *self.state.lock() = SocketState::Closed;
        self.waiters.wake_all();
    }
}

// Message flags
pub const MSG_PEEK: i32 = 0x02;
pub const MSG_DONTWAIT: i32 = 0x40;
pub const MSG_NOSIGNAL: i32 = 0x4000;

// Shutdown flags
pub const SHUT_RD: i32 = 0;
pub const SHUT_WR: i32 = 1;
pub const SHUT_RDWR: i32 = 2;

// Socket options
pub const SOL_SOCKET: i32 = 1;
pub const SO_RCVBUF: i32 = 8;
pub const SO_SNDBUF: i32 = 7;
pub const SO_PASSCRED: i32 = 16;
pub const SO_TYPE: i32 = 3;
pub const SO_ERROR: i32 = 4;
pub const SO_PEERCRED: i32 = 17;

/// Socket error
#[derive(Clone, Copy, Debug)]
pub enum SocketError {
    /// Invalid argument
    Invalid,
    /// Address in use
    AddrInUse,
    /// Operation not supported
    OpNotSupported,
    /// Invalid state
    InvalidState,
    /// Would block
    WouldBlock,
    /// Connection refused
    ConnRefused,
    /// Not connected
    NotConnected,
    /// Already connected
    IsConnected,
    /// Destination address required
    DestAddrRequired,
    /// Broken pipe
    BrokenPipe,
    /// Permission denied
    Permission,
    /// Address not available
    AddrNotAvail,
}

impl SocketError {
    pub fn to_errno(&self) -> i32 {
        match self {
            SocketError::Invalid => -22,        // EINVAL
            SocketError::AddrInUse => -98,      // EADDRINUSE
            SocketError::OpNotSupported => -95, // EOPNOTSUPP
            SocketError::InvalidState => -22,   // EINVAL
            SocketError::WouldBlock => -11,     // EAGAIN
            SocketError::ConnRefused => -111,   // ECONNREFUSED
            SocketError::NotConnected => -107,  // ENOTCONN
            SocketError::IsConnected => -106,   // EISCONN
            SocketError::DestAddrRequired => -89, // EDESTADDRREQ
            SocketError::BrokenPipe => -32,     // EPIPE
            SocketError::Permission => -1,      // EPERM
            SocketError::AddrNotAvail => -99,   // EADDRNOTAVAIL
        }
    }
}

// ============================================================================
// Global State
// ============================================================================

/// Bound sockets by path
static BOUND_SOCKETS: RwLock<BTreeMap<String, Arc<UnixSocket>>> = RwLock::new(BTreeMap::new());

/// Create socket pair
pub fn socketpair(socket_type: SocketType) -> Result<(Arc<UnixSocket>, Arc<UnixSocket>), SocketError> {
    let pid = crate::process::current()
        .map(|p| p.pid)
        .ok_or(SocketError::Permission)?;

    let sock1 = UnixSocket::new(socket_type, pid);
    let sock2 = UnixSocket::new(socket_type, pid);

    *sock1.state.lock() = SocketState::Connected;
    *sock2.state.lock() = SocketState::Connected;

    *sock1.peer.write() = Some(sock2.clone());
    *sock2.peer.write() = Some(sock1.clone());

    Ok((sock1, sock2))
}

/// Initialize Unix socket subsystem
pub fn init() {
    crate::kprintln!("  Unix domain sockets initialized");
}
