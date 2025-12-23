//! Pipes and FIFOs
//!
//! Inter-process communication through unidirectional byte streams.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;
use core::sync::atomic::{AtomicU32, AtomicBool, Ordering};

/// Default pipe buffer size
pub const PIPE_BUF_SIZE: usize = 65536;

/// Atomic write size (POSIX PIPE_BUF)
pub const PIPE_BUF: usize = 4096;

/// Pipe error types
#[derive(Clone, Debug)]
pub enum PipeError {
    BrokenPipe,
    WouldBlock,
    InvalidFd,
    BufferFull,
    Closed,
}

/// Pipe buffer
struct PipeBuffer {
    data: VecDeque<u8>,
    capacity: usize,
    readers: u32,
    writers: u32,
    closed: bool,
}

impl PipeBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            readers: 1,
            writers: 1,
            closed: false,
        }
    }

    fn available_read(&self) -> usize {
        self.data.len()
    }

    fn available_write(&self) -> usize {
        self.capacity - self.data.len()
    }

    fn read(&mut self, buf: &mut [u8]) -> usize {
        let count = buf.len().min(self.data.len());
        for i in 0..count {
            buf[i] = self.data.pop_front().unwrap();
        }
        count
    }

    fn write(&mut self, buf: &[u8]) -> usize {
        let count = buf.len().min(self.available_write());
        for &byte in &buf[..count] {
            self.data.push_back(byte);
        }
        count
    }
}

/// Pipe endpoint
pub struct Pipe {
    buffer: Arc<Mutex<PipeBuffer>>,
    is_read_end: bool,
}

impl Pipe {
    /// Create a new pipe pair (read_end, write_end)
    pub fn new() -> (Self, Self) {
        Self::with_capacity(PIPE_BUF_SIZE)
    }

    /// Create pipe with custom capacity
    pub fn with_capacity(capacity: usize) -> (Self, Self) {
        let buffer = Arc::new(Mutex::new(PipeBuffer::new(capacity)));

        let read_end = Self {
            buffer: buffer.clone(),
            is_read_end: true,
        };

        let write_end = Self {
            buffer,
            is_read_end: false,
        };

        (read_end, write_end)
    }

    /// Read from pipe
    pub fn read(&self, buf: &mut [u8]) -> Result<usize, PipeError> {
        if !self.is_read_end {
            return Err(PipeError::InvalidFd);
        }

        let mut buffer = self.buffer.lock();

        if buffer.data.is_empty() {
            if buffer.writers == 0 || buffer.closed {
                return Ok(0); // EOF
            }
            return Err(PipeError::WouldBlock);
        }

        Ok(buffer.read(buf))
    }

    /// Write to pipe
    pub fn write(&self, buf: &[u8]) -> Result<usize, PipeError> {
        if self.is_read_end {
            return Err(PipeError::InvalidFd);
        }

        let mut buffer = self.buffer.lock();

        if buffer.readers == 0 || buffer.closed {
            return Err(PipeError::BrokenPipe);
        }

        if buffer.available_write() == 0 {
            return Err(PipeError::WouldBlock);
        }

        Ok(buffer.write(buf))
    }

    /// Check if data is available to read
    pub fn poll_read(&self) -> bool {
        let buffer = self.buffer.lock();
        !buffer.data.is_empty() || buffer.writers == 0 || buffer.closed
    }

    /// Check if space is available to write
    pub fn poll_write(&self) -> bool {
        let buffer = self.buffer.lock();
        buffer.available_write() > 0 && buffer.readers > 0 && !buffer.closed
    }

    /// Close this end of the pipe
    pub fn close(&self) {
        let mut buffer = self.buffer.lock();
        if self.is_read_end {
            buffer.readers = buffer.readers.saturating_sub(1);
        } else {
            buffer.writers = buffer.writers.saturating_sub(1);
        }
    }

    /// Get number of bytes available to read
    pub fn available(&self) -> usize {
        self.buffer.lock().available_read()
    }
}

impl Clone for Pipe {
    fn clone(&self) -> Self {
        let mut buffer = self.buffer.lock();
        if self.is_read_end {
            buffer.readers += 1;
        } else {
            buffer.writers += 1;
        }

        Self {
            buffer: self.buffer.clone(),
            is_read_end: self.is_read_end,
        }
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        self.close();
    }
}

/// Named pipe (FIFO)
pub struct Fifo {
    buffer: Mutex<PipeBuffer>,
    name: alloc::string::String,
    readers: AtomicU32,
    writers: AtomicU32,
}

impl Fifo {
    /// Create a new named FIFO
    pub fn new(name: &str) -> Self {
        Self {
            buffer: Mutex::new(PipeBuffer::new(PIPE_BUF_SIZE)),
            name: alloc::string::String::from(name),
            readers: AtomicU32::new(0),
            writers: AtomicU32::new(0),
        }
    }

    /// Open FIFO for reading
    pub fn open_read(&self) -> FifoHandle {
        self.readers.fetch_add(1, Ordering::SeqCst);
        FifoHandle {
            fifo: self as *const Fifo,
            is_read: true,
        }
    }

    /// Open FIFO for writing
    pub fn open_write(&self) -> FifoHandle {
        self.writers.fetch_add(1, Ordering::SeqCst);
        FifoHandle {
            fifo: self as *const Fifo,
            is_read: false,
        }
    }

    /// Get FIFO name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Handle to an open FIFO
pub struct FifoHandle {
    fifo: *const Fifo,
    is_read: bool,
}

impl FifoHandle {
    /// Read from FIFO
    pub fn read(&self, buf: &mut [u8]) -> Result<usize, PipeError> {
        if !self.is_read {
            return Err(PipeError::InvalidFd);
        }

        let fifo = unsafe { &*self.fifo };
        let mut buffer = fifo.buffer.lock();

        if buffer.data.is_empty() {
            if fifo.writers.load(Ordering::SeqCst) == 0 {
                return Ok(0);
            }
            return Err(PipeError::WouldBlock);
        }

        Ok(buffer.read(buf))
    }

    /// Write to FIFO
    pub fn write(&self, buf: &[u8]) -> Result<usize, PipeError> {
        if self.is_read {
            return Err(PipeError::InvalidFd);
        }

        let fifo = unsafe { &*self.fifo };

        if fifo.readers.load(Ordering::SeqCst) == 0 {
            return Err(PipeError::BrokenPipe);
        }

        let mut buffer = fifo.buffer.lock();
        if buffer.available_write() == 0 {
            return Err(PipeError::WouldBlock);
        }

        Ok(buffer.write(buf))
    }
}

impl Drop for FifoHandle {
    fn drop(&mut self) {
        let fifo = unsafe { &*self.fifo };
        if self.is_read {
            fifo.readers.fetch_sub(1, Ordering::SeqCst);
        } else {
            fifo.writers.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

// Note: FifoHandle is not Send/Sync due to raw pointer
// In a real implementation, would use Arc<Fifo>

/// Pipe registry for named pipes
use alloc::collections::BTreeMap;
use alloc::string::String;

static FIFOS: Mutex<BTreeMap<String, Arc<Fifo>>> = Mutex::new(BTreeMap::new());

/// Create a named FIFO
pub fn mkfifo(path: &str) -> Result<(), PipeError> {
    let mut fifos = FIFOS.lock();

    if fifos.contains_key(path) {
        return Err(PipeError::InvalidFd);
    }

    fifos.insert(String::from(path), Arc::new(Fifo::new(path)));
    Ok(())
}

/// Remove a named FIFO
pub fn unlink(path: &str) -> Result<(), PipeError> {
    let mut fifos = FIFOS.lock();
    fifos.remove(path).ok_or(PipeError::InvalidFd)?;
    Ok(())
}

/// Get a named FIFO
pub fn get_fifo(path: &str) -> Option<Arc<Fifo>> {
    FIFOS.lock().get(path).cloned()
}

/// Pipe statistics
static PIPES_CREATED: AtomicU32 = AtomicU32::new(0);
static PIPE_BYTES_WRITTEN: AtomicU32 = AtomicU32::new(0);

/// Create a pipe and return file descriptor pair
pub fn create_pipe() -> (Pipe, Pipe) {
    PIPES_CREATED.fetch_add(1, Ordering::Relaxed);
    Pipe::new()
}

/// Initialize pipe subsystem
pub fn init() {
    crate::kprintln!("  Pipe subsystem initialized");
}
