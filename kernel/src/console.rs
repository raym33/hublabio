//! Kernel Console
//!
//! Unified console output for kernel messages.
//! Routes output to UART, framebuffer, or both.

use core::fmt::{self, Write};
use spin::Mutex;

use crate::drivers::{uart, framebuffer};

/// Console output mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// No output (early boot)
    None,
    /// UART only (serial console)
    Uart,
    /// Framebuffer only (graphical console)
    Framebuffer,
    /// Both UART and framebuffer
    Both,
}

/// Global console state
pub static CONSOLE: Mutex<Console> = Mutex::new(Console::new());

/// Console driver
pub struct Console {
    mode: OutputMode,
    initialized: bool,
}

impl Console {
    /// Create a new console (uninitialized)
    pub const fn new() -> Self {
        Self {
            mode: OutputMode::None,
            initialized: false,
        }
    }

    /// Set output mode
    pub fn set_mode(&mut self, mode: OutputMode) {
        self.mode = mode;
    }

    /// Get current output mode
    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Mark as initialized
    pub fn mark_initialized(&mut self) {
        self.initialized = true;
    }

    /// Write a character
    pub fn putc(&self, c: char) {
        match self.mode {
            OutputMode::None => {
                // Early boot: try UART anyway
                if uart::is_ready() {
                    uart::putc(c as u8);
                }
            }
            OutputMode::Uart => {
                uart::putc(c as u8);
            }
            OutputMode::Framebuffer => {
                framebuffer::write_char(c);
            }
            OutputMode::Both => {
                uart::putc(c as u8);
                framebuffer::write_char(c);
            }
        }
    }

    /// Write a string
    pub fn puts(&self, s: &str) {
        for c in s.chars() {
            self.putc(c);
        }
    }
}

impl Write for Console {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.puts(s);
        Ok(())
    }
}

/// Console writer for format_args
pub struct ConsoleWriter;

impl Write for ConsoleWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let console = CONSOLE.lock();
        console.puts(s);
        Ok(())
    }
}

/// Initialize console
pub fn init() {
    // Initialize UART first for early output
    uart::init();

    let mut console = CONSOLE.lock();

    if uart::is_ready() {
        console.set_mode(OutputMode::Uart);
    }

    console.mark_initialized();
}

/// Initialize console with framebuffer
pub fn init_with_framebuffer() {
    let mut console = CONSOLE.lock();

    if uart::is_ready() && framebuffer::is_available() {
        console.set_mode(OutputMode::Both);
    } else if framebuffer::is_available() {
        console.set_mode(OutputMode::Framebuffer);
    } else if uart::is_ready() {
        console.set_mode(OutputMode::Uart);
    }
}

/// Print function for kprint! macro
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    let mut writer = ConsoleWriter;
    writer.write_fmt(args).unwrap();
}

/// Read a character from console (blocking)
pub fn getc() -> char {
    uart::getc() as char
}

/// Try to read a character from console (non-blocking)
pub fn try_getc() -> Option<char> {
    uart::try_getc().map(|c| c as char)
}

/// Read a line from console
pub fn read_line(buf: &mut [u8]) -> usize {
    let mut pos = 0;

    loop {
        let c = getc();

        match c {
            '\r' | '\n' => {
                _print(format_args!("\n"));
                break;
            }
            '\x08' | '\x7f' => {
                // Backspace
                if pos > 0 {
                    pos -= 1;
                    _print(format_args!("\x08 \x08"));
                }
            }
            _ if pos < buf.len() => {
                buf[pos] = c as u8;
                pos += 1;
                _print(format_args!("{}", c));
            }
            _ => {}
        }
    }

    pos
}

/// Clear the console
pub fn clear() {
    let console = CONSOLE.lock();

    match console.mode() {
        OutputMode::Uart | OutputMode::Both => {
            // ANSI escape sequence to clear screen
            uart::write("\x1b[2J\x1b[H");
        }
        OutputMode::Framebuffer => {
            framebuffer::clear();
        }
        OutputMode::None => {}
    }

    if console.mode() == OutputMode::Both {
        framebuffer::clear();
    }
}

/// Set console colors (framebuffer only)
pub fn set_color(fg: u32, bg: u32) {
    if let Some(ref mut fb) = *crate::drivers::framebuffer::FRAMEBUFFER.lock() {
        fb.set_fg_color(fg);
        fb.set_bg_color(bg);
    }
}

/// ANSI color codes for UART
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";
    pub const BOLD: &str = "\x1b[1m";
}

/// Log levels
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Fatal = 4,
}

/// Current log level (filters messages below this level)
static LOG_LEVEL: Mutex<LogLevel> = Mutex::new(LogLevel::Debug);

/// Set minimum log level
pub fn set_log_level(level: LogLevel) {
    *LOG_LEVEL.lock() = level;
}

/// Get current log level
pub fn log_level() -> LogLevel {
    *LOG_LEVEL.lock()
}

/// Log a message with level
pub fn log(level: LogLevel, args: fmt::Arguments) {
    if level < log_level() {
        return;
    }

    let (prefix, color) = match level {
        LogLevel::Debug => ("[DEBUG]", ansi::CYAN),
        LogLevel::Info => ("[INFO] ", ansi::GREEN),
        LogLevel::Warn => ("[WARN] ", ansi::YELLOW),
        LogLevel::Error => ("[ERROR]", ansi::RED),
        LogLevel::Fatal => ("[FATAL]", ansi::MAGENTA),
    };

    _print(format_args!("{}{}{} {}\n", color, prefix, ansi::RESET, args));
}

/// Debug log macro
#[macro_export]
macro_rules! kdebug {
    ($($arg:tt)*) => ($crate::console::log($crate::console::LogLevel::Debug, format_args!($($arg)*)));
}

/// Info log macro
#[macro_export]
macro_rules! kinfo {
    ($($arg:tt)*) => ($crate::console::log($crate::console::LogLevel::Info, format_args!($($arg)*)));
}

/// Warning log macro
#[macro_export]
macro_rules! kwarn {
    ($($arg:tt)*) => ($crate::console::log($crate::console::LogLevel::Warn, format_args!($($arg)*)));
}

/// Error log macro
#[macro_export]
macro_rules! kerror {
    ($($arg:tt)*) => ($crate::console::log($crate::console::LogLevel::Error, format_args!($($arg)*)));
}

/// Fatal log macro
#[macro_export]
macro_rules! kfatal {
    ($($arg:tt)*) => ($crate::console::log($crate::console::LogLevel::Fatal, format_args!($($arg)*)));
}
