//! Framebuffer Driver
//!
//! Simple framebuffer driver for console output.

use core::ptr::write_volatile;
use spin::Mutex;

use crate::FramebufferInfo;

/// Global framebuffer instance
pub static FRAMEBUFFER: Mutex<Option<Framebuffer>> = Mutex::new(None);

/// Framebuffer driver
pub struct Framebuffer {
    addr: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u8,
    cursor_x: u32,
    cursor_y: u32,
    fg_color: u32,
    bg_color: u32,
}

/// Basic 8x16 font (simplified)
const FONT_WIDTH: u32 = 8;
const FONT_HEIGHT: u32 = 16;

impl Framebuffer {
    /// Create a new framebuffer from boot info
    pub fn new(info: &FramebufferInfo) -> Self {
        Self {
            addr: info.address,
            width: info.width,
            height: info.height,
            pitch: info.pitch,
            bpp: info.bpp,
            cursor_x: 0,
            cursor_y: 0,
            fg_color: 0xFFFFFF, // White
            bg_color: 0x000000, // Black
        }
    }

    /// Clear the screen
    pub fn clear(&mut self) {
        let total_size = (self.pitch * self.height) as usize;

        unsafe {
            let ptr = self.addr as *mut u8;
            for i in 0..total_size {
                write_volatile(ptr.add(i), 0);
            }
        }

        self.cursor_x = 0;
        self.cursor_y = 0;
    }

    /// Set a pixel
    pub fn set_pixel(&self, x: u32, y: u32, color: u32) {
        if x >= self.width || y >= self.height {
            return;
        }

        let offset = (y * self.pitch + x * (self.bpp as u32 / 8)) as usize;

        unsafe {
            let ptr = (self.addr + offset) as *mut u32;
            write_volatile(ptr, color);
        }
    }

    /// Draw a character at position
    pub fn draw_char(&self, x: u32, y: u32, c: char) {
        // Simplified font rendering - just draw a box for now
        // A real implementation would use a proper font bitmap

        let char_code = c as u32;

        for dy in 0..FONT_HEIGHT {
            for dx in 0..FONT_WIDTH {
                // Simple pattern: draw character outline
                let on = if char_code >= 32 && char_code < 127 {
                    // Draw a simple pattern based on char code
                    let pattern = get_font_pattern(c, dx, dy);
                    pattern
                } else {
                    false
                };

                let color = if on { self.fg_color } else { self.bg_color };
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }

    /// Write a character at cursor position
    pub fn write_char(&mut self, c: char) {
        match c {
            '\n' => {
                self.cursor_x = 0;
                self.cursor_y += FONT_HEIGHT;
                if self.cursor_y + FONT_HEIGHT > self.height {
                    self.scroll();
                }
            }
            '\r' => {
                self.cursor_x = 0;
            }
            '\t' => {
                let spaces = 4 - (self.cursor_x / FONT_WIDTH) % 4;
                for _ in 0..spaces {
                    self.write_char(' ');
                }
            }
            _ => {
                self.draw_char(self.cursor_x, self.cursor_y, c);
                self.cursor_x += FONT_WIDTH;
                if self.cursor_x + FONT_WIDTH > self.width {
                    self.cursor_x = 0;
                    self.cursor_y += FONT_HEIGHT;
                    if self.cursor_y + FONT_HEIGHT > self.height {
                        self.scroll();
                    }
                }
            }
        }
    }

    /// Write a string
    pub fn write_str(&mut self, s: &str) {
        for c in s.chars() {
            self.write_char(c);
        }
    }

    /// Scroll the screen up by one line
    fn scroll(&mut self) {
        // Move all lines up by FONT_HEIGHT
        let line_bytes = (self.pitch * FONT_HEIGHT) as usize;
        let total_bytes = (self.pitch * self.height) as usize;

        unsafe {
            let ptr = self.addr as *mut u8;

            // Copy lines up
            for i in 0..(total_bytes - line_bytes) {
                let src = ptr.add(i + line_bytes);
                let dst = ptr.add(i);
                write_volatile(dst, *src);
            }

            // Clear bottom line
            for i in (total_bytes - line_bytes)..total_bytes {
                write_volatile(ptr.add(i), 0);
            }
        }

        self.cursor_y -= FONT_HEIGHT;
    }

    /// Set foreground color
    pub fn set_fg_color(&mut self, color: u32) {
        self.fg_color = color;
    }

    /// Set background color
    pub fn set_bg_color(&mut self, color: u32) {
        self.bg_color = color;
    }

    /// Get screen dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Get font pattern for character (simplified)
fn get_font_pattern(c: char, x: u32, y: u32) -> bool {
    // Very simplified font - just patterns for basic recognition
    // A real implementation would use a bitmap font

    match c {
        ' ' => false,
        '_' => y == FONT_HEIGHT - 2,
        '-' => y == FONT_HEIGHT / 2 && x > 1 && x < FONT_WIDTH - 2,
        '|' => x == FONT_WIDTH / 2,
        '+' => x == FONT_WIDTH / 2 || y == FONT_HEIGHT / 2,
        '=' => y == FONT_HEIGHT / 2 - 1 || y == FONT_HEIGHT / 2 + 1,
        '.' => y >= FONT_HEIGHT - 4 && x >= FONT_WIDTH / 2 - 1 && x <= FONT_WIDTH / 2 + 1,
        ':' => {
            let dot = x >= FONT_WIDTH / 2 - 1 && x <= FONT_WIDTH / 2 + 1;
            dot && (y >= 3 && y <= 5 || y >= FONT_HEIGHT - 5 && y <= FONT_HEIGHT - 3)
        }
        '0'..='9' => {
            // Simple digit patterns
            let border_x = x == 1 || x == FONT_WIDTH - 2;
            let border_y = y == 2 || y == FONT_HEIGHT - 3;
            border_x || border_y
        }
        'A'..='Z' | 'a'..='z' => {
            // Simple letter pattern - outline
            let border_x = x == 1 || x == FONT_WIDTH - 2;
            let border_y = y == 2 || y == FONT_HEIGHT - 3;
            let middle_y = y == FONT_HEIGHT / 2;
            border_x || border_y || (middle_y && x > 1 && x < FONT_WIDTH - 2)
        }
        _ => {
            // Default: draw a box
            let border_x = x == 1 || x == FONT_WIDTH - 2;
            let border_y = y == 2 || y == FONT_HEIGHT - 3;
            border_x || border_y
        }
    }
}

/// Initialize framebuffer from boot info
pub fn init(info: &FramebufferInfo) {
    let mut fb = Framebuffer::new(info);
    fb.clear();
    *FRAMEBUFFER.lock() = Some(fb);
}

/// Write to framebuffer console
pub fn write_str(s: &str) {
    if let Some(ref mut fb) = *FRAMEBUFFER.lock() {
        fb.write_str(s);
    }
}

/// Write a character
pub fn write_char(c: char) {
    if let Some(ref mut fb) = *FRAMEBUFFER.lock() {
        fb.write_char(c);
    }
}

/// Clear the screen
pub fn clear() {
    if let Some(ref mut fb) = *FRAMEBUFFER.lock() {
        fb.clear();
    }
}

/// Check if framebuffer is available
pub fn is_available() -> bool {
    FRAMEBUFFER.lock().is_some()
}
