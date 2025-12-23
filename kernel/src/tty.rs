//! TTY/PTY Subsystem
//!
//! Terminal handling and pseudo-terminal support.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::Mutex;
use core::sync::atomic::{AtomicU32, AtomicBool, Ordering};

/// TTY buffer size
pub const TTY_BUFFER_SIZE: usize = 4096;

/// Special characters
pub mod chars {
    pub const EOF: u8 = 0x04;      // Ctrl+D
    pub const EOL: u8 = b'\n';     // Newline
    pub const ERASE: u8 = 0x7F;    // Backspace/Delete
    pub const KILL: u8 = 0x15;     // Ctrl+U (kill line)
    pub const INTR: u8 = 0x03;     // Ctrl+C
    pub const QUIT: u8 = 0x1C;     // Ctrl+\
    pub const SUSP: u8 = 0x1A;     // Ctrl+Z
    pub const START: u8 = 0x11;   // Ctrl+Q
    pub const STOP: u8 = 0x13;    // Ctrl+S
}

/// TTY input modes
#[derive(Clone, Copy, Debug)]
pub struct InputModes {
    pub icrnl: bool,   // Map CR to NL on input
    pub inlcr: bool,   // Map NL to CR on input
    pub igncr: bool,   // Ignore CR
    pub iuclc: bool,   // Map uppercase to lowercase
    pub ixon: bool,    // Enable XON/XOFF flow control
    pub ixoff: bool,   // Enable input flow control
    pub istrip: bool,  // Strip 8th bit
}

impl Default for InputModes {
    fn default() -> Self {
        Self {
            icrnl: true,
            inlcr: false,
            igncr: false,
            iuclc: false,
            ixon: true,
            ixoff: false,
            istrip: false,
        }
    }
}

/// TTY output modes
#[derive(Clone, Copy, Debug)]
pub struct OutputModes {
    pub opost: bool,   // Post-process output
    pub onlcr: bool,   // Map NL to CR-NL
    pub ocrnl: bool,   // Map CR to NL
    pub onocr: bool,   // No CR at column 0
    pub onlret: bool,  // NL performs CR function
    pub olcuc: bool,   // Map lowercase to uppercase
}

impl Default for OutputModes {
    fn default() -> Self {
        Self {
            opost: true,
            onlcr: true,
            ocrnl: false,
            onocr: false,
            onlret: false,
            olcuc: false,
        }
    }
}

/// TTY local modes
#[derive(Clone, Copy, Debug)]
pub struct LocalModes {
    pub echo: bool,    // Echo input
    pub echoe: bool,   // Echo erase as backspace
    pub echok: bool,   // Echo NL after kill
    pub echonl: bool,  // Echo NL
    pub icanon: bool,  // Canonical mode (line editing)
    pub isig: bool,    // Enable signals
    pub iexten: bool,  // Extended processing
    pub noflsh: bool,  // Disable flush after interrupt
}

impl Default for LocalModes {
    fn default() -> Self {
        Self {
            echo: true,
            echoe: true,
            echok: true,
            echonl: false,
            icanon: true,
            isig: true,
            iexten: true,
            noflsh: false,
        }
    }
}

/// TTY configuration (termios)
#[derive(Clone, Debug)]
pub struct Termios {
    pub input: InputModes,
    pub output: OutputModes,
    pub local: LocalModes,
    pub cc: [u8; 32],  // Control characters
}

impl Default for Termios {
    fn default() -> Self {
        let mut cc = [0u8; 32];
        cc[0] = chars::INTR;   // VINTR
        cc[1] = chars::QUIT;   // VQUIT
        cc[2] = chars::ERASE;  // VERASE
        cc[3] = chars::KILL;   // VKILL
        cc[4] = chars::EOF;    // VEOF
        cc[5] = 0;             // VTIME
        cc[6] = 1;             // VMIN
        cc[7] = chars::SUSP;   // VSUSP
        cc[8] = chars::START;  // VSTART
        cc[9] = chars::STOP;   // VSTOP

        Self {
            input: InputModes::default(),
            output: OutputModes::default(),
            local: LocalModes::default(),
            cc,
        }
    }
}

/// Window size
#[derive(Clone, Copy, Debug, Default)]
pub struct WinSize {
    pub rows: u16,
    pub cols: u16,
    pub xpixel: u16,
    pub ypixel: u16,
}

/// TTY device
pub struct Tty {
    pub name: String,
    pub termios: Mutex<Termios>,
    pub winsize: Mutex<WinSize>,
    pub input_buffer: Mutex<VecDeque<u8>>,
    pub output_buffer: Mutex<VecDeque<u8>>,
    pub line_buffer: Mutex<Vec<u8>>,  // For canonical mode
    pub foreground_pgrp: AtomicU32,
    pub session: AtomicU32,
    pub stopped: AtomicBool,
}

impl Tty {
    /// Create a new TTY
    pub fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
            termios: Mutex::new(Termios::default()),
            winsize: Mutex::new(WinSize { rows: 24, cols: 80, xpixel: 0, ypixel: 0 }),
            input_buffer: Mutex::new(VecDeque::with_capacity(TTY_BUFFER_SIZE)),
            output_buffer: Mutex::new(VecDeque::with_capacity(TTY_BUFFER_SIZE)),
            line_buffer: Mutex::new(Vec::with_capacity(256)),
            foreground_pgrp: AtomicU32::new(0),
            session: AtomicU32::new(0),
            stopped: AtomicBool::new(false),
        }
    }

    /// Process input character
    pub fn input(&self, c: u8) {
        let termios = self.termios.lock();

        // Handle special characters in isig mode
        if termios.local.isig {
            match c {
                c if c == termios.cc[0] => {
                    // VINTR - send SIGINT
                    crate::kdebug!("TTY: SIGINT");
                    return;
                }
                c if c == termios.cc[1] => {
                    // VQUIT - send SIGQUIT
                    crate::kdebug!("TTY: SIGQUIT");
                    return;
                }
                c if c == termios.cc[7] => {
                    // VSUSP - send SIGTSTP
                    crate::kdebug!("TTY: SIGTSTP");
                    return;
                }
                _ => {}
            }
        }

        // Input processing
        let mut c = c;
        if termios.input.istrip {
            c &= 0x7F;
        }
        if termios.input.icrnl && c == b'\r' {
            c = b'\n';
        } else if termios.input.inlcr && c == b'\n' {
            c = b'\r';
        } else if termios.input.igncr && c == b'\r' {
            return;
        }
        if termios.input.iuclc && c >= b'A' && c <= b'Z' {
            c = c - b'A' + b'a';
        }

        drop(termios);

        // Handle canonical mode
        let termios = self.termios.lock();
        if termios.local.icanon {
            drop(termios);
            self.canonical_input(c);
        } else {
            drop(termios);
            self.raw_input(c);
        }
    }

    /// Canonical mode input processing
    fn canonical_input(&self, c: u8) {
        let termios = self.termios.lock();
        let echo = termios.local.echo;

        if c == termios.cc[2] {
            // VERASE - backspace
            drop(termios);
            let mut line = self.line_buffer.lock();
            if !line.is_empty() {
                line.pop();
                if echo {
                    self.echo(b'\x08'); // Backspace
                    self.echo(b' ');
                    self.echo(b'\x08');
                }
            }
            return;
        }

        if c == termios.cc[3] {
            // VKILL - kill line
            drop(termios);
            let mut line = self.line_buffer.lock();
            if echo {
                for _ in 0..line.len() {
                    self.echo(b'\x08');
                    self.echo(b' ');
                    self.echo(b'\x08');
                }
            }
            line.clear();
            return;
        }

        if c == termios.cc[4] {
            // VEOF
            drop(termios);
            let mut line = self.line_buffer.lock();
            let mut input = self.input_buffer.lock();
            for &b in line.iter() {
                input.push_back(b);
            }
            line.clear();
            return;
        }

        drop(termios);

        // Add to line buffer
        let mut line = self.line_buffer.lock();
        line.push(c);

        if echo {
            self.echo(c);
        }

        // On newline, flush line to input buffer
        if c == b'\n' {
            let mut input = self.input_buffer.lock();
            for &b in line.iter() {
                input.push_back(b);
            }
            line.clear();
        }
    }

    /// Raw mode input
    fn raw_input(&self, c: u8) {
        let termios = self.termios.lock();
        let echo = termios.local.echo;
        drop(termios);

        self.input_buffer.lock().push_back(c);

        if echo {
            self.echo(c);
        }
    }

    /// Echo character to output
    fn echo(&self, c: u8) {
        self.output_buffer.lock().push_back(c);
    }

    /// Read from TTY
    pub fn read(&self, buf: &mut [u8]) -> usize {
        let mut input = self.input_buffer.lock();
        let mut count = 0;

        while count < buf.len() {
            if let Some(c) = input.pop_front() {
                buf[count] = c;
                count += 1;
            } else {
                break;
            }
        }

        count
    }

    /// Write to TTY
    pub fn write(&self, buf: &[u8]) -> usize {
        let termios = self.termios.lock();
        let mut output = self.output_buffer.lock();

        for &c in buf {
            let mut c = c;

            // Output processing
            if termios.output.opost {
                if termios.output.onlcr && c == b'\n' {
                    output.push_back(b'\r');
                }
                if termios.output.olcuc && c >= b'a' && c <= b'z' {
                    c = c - b'a' + b'A';
                }
            }

            output.push_back(c);
        }

        buf.len()
    }

    /// Flush output
    pub fn flush(&self) -> Vec<u8> {
        let mut output = self.output_buffer.lock();
        output.drain(..).collect()
    }

    /// Get/set termios
    pub fn get_termios(&self) -> Termios {
        self.termios.lock().clone()
    }

    pub fn set_termios(&self, termios: Termios) {
        *self.termios.lock() = termios;
    }

    /// Get/set window size
    pub fn get_winsize(&self) -> WinSize {
        *self.winsize.lock()
    }

    pub fn set_winsize(&self, size: WinSize) {
        *self.winsize.lock() = size;
    }

    /// Check if data available
    pub fn poll_read(&self) -> bool {
        !self.input_buffer.lock().is_empty()
    }
}

/// Pseudo-terminal pair
pub struct Pty {
    pub master: Arc<PtyMaster>,
    pub slave: Arc<PtySlave>,
}

/// PTY master side
pub struct PtyMaster {
    tty: Arc<Tty>,
    slave_name: String,
}

impl PtyMaster {
    /// Read from master (gets slave output)
    pub fn read(&self, buf: &mut [u8]) -> usize {
        let output = self.tty.flush();
        let count = buf.len().min(output.len());
        buf[..count].copy_from_slice(&output[..count]);
        count
    }

    /// Write to master (goes to slave input)
    pub fn write(&self, buf: &[u8]) -> usize {
        for &c in buf {
            self.tty.input(c);
        }
        buf.len()
    }

    /// Get slave device name
    pub fn slave_name(&self) -> &str {
        &self.slave_name
    }
}

/// PTY slave side (acts like a terminal)
pub struct PtySlave {
    tty: Arc<Tty>,
}

impl PtySlave {
    /// Read from slave
    pub fn read(&self, buf: &mut [u8]) -> usize {
        self.tty.read(buf)
    }

    /// Write to slave
    pub fn write(&self, buf: &[u8]) -> usize {
        self.tty.write(buf)
    }

    /// Get termios
    pub fn get_termios(&self) -> Termios {
        self.tty.get_termios()
    }

    /// Set termios
    pub fn set_termios(&self, termios: Termios) {
        self.tty.set_termios(termios);
    }
}

/// PTY counter
static PTY_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Create new PTY pair
pub fn openpty() -> Pty {
    let num = PTY_COUNTER.fetch_add(1, Ordering::SeqCst);
    let slave_name = alloc::format!("/dev/pts/{}", num);

    let tty = Arc::new(Tty::new(&slave_name));

    Pty {
        master: Arc::new(PtyMaster {
            tty: tty.clone(),
            slave_name: slave_name.clone(),
        }),
        slave: Arc::new(PtySlave { tty }),
    }
}

/// Console TTY (tty0)
static CONSOLE: Mutex<Option<Arc<Tty>>> = Mutex::new(None);

/// Get console TTY
pub fn console() -> Option<Arc<Tty>> {
    CONSOLE.lock().clone()
}

/// Initialize TTY subsystem
pub fn init() {
    // Create console TTY
    let tty = Arc::new(Tty::new("/dev/tty0"));
    *CONSOLE.lock() = Some(tty);

    crate::kprintln!("  TTY subsystem initialized");
}
