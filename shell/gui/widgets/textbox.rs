//! TextBox Widget

use alloc::string::String;

use super::Widget;
use crate::gui::input::MouseButton;
use crate::gui::theme::Theme;
use crate::gui::{Color, Rect};

/// TextBox widget for text input
pub struct TextBox {
    bounds: Rect,
    text: String,
    placeholder: String,
    cursor: usize,
    focused: bool,
    enabled: bool,
    visible: bool,
    password: bool,
    max_length: usize,
}

impl TextBox {
    /// Create a new textbox
    pub fn new(x: i32, y: i32, width: u32) -> Self {
        Self {
            bounds: Rect::new(x, y, width, 24),
            text: String::new(),
            placeholder: String::new(),
            cursor: 0,
            focused: false,
            enabled: true,
            visible: true,
            password: false,
            max_length: 256,
        }
    }

    /// Set text
    pub fn set_text(&mut self, text: &str) {
        self.text = String::from(text);
        self.cursor = self.text.len();
    }

    /// Get text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Set placeholder
    pub fn set_placeholder(&mut self, placeholder: &str) {
        self.placeholder = String::from(placeholder);
    }

    /// Set password mode
    pub fn set_password(&mut self, password: bool) {
        self.password = password;
    }

    /// Set max length
    pub fn set_max_length(&mut self, max: usize) {
        self.max_length = max;
    }

    /// Is focused
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Set focus
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }
}

impl Widget for TextBox {
    fn bounds(&self) -> Rect {
        self.bounds
    }

    fn set_position(&mut self, x: i32, y: i32) {
        self.bounds.x = x;
        self.bounds.y = y;
    }

    fn set_size(&mut self, width: u32, height: u32) {
        self.bounds.width = width;
        self.bounds.height = height;
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn draw(&self, buffer: &mut [u32], screen_width: u32, parent_bounds: Rect, theme: &Theme) {
        if !self.visible {
            return;
        }

        let abs_x = parent_bounds.x + self.bounds.x;
        let abs_y = parent_bounds.y + self.bounds.y;

        // Background
        let bg_color = if self.enabled {
            theme.input_bg
        } else {
            theme.button_disabled
        };

        let bg = bg_color.to_argb();
        for y in 0..self.bounds.height {
            for x in 0..self.bounds.width {
                let px = abs_x + x as i32;
                let py = abs_y + y as i32;

                if px >= 0 && py >= 0 {
                    let idx = (py as u32 * screen_width + px as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = bg;
                    }
                }
            }
        }

        // Border
        let border_color = if self.focused {
            theme.accent_color
        } else {
            theme.border_color
        };

        let border = border_color.to_argb();

        // Top and bottom
        for x in 0..self.bounds.width {
            let px = abs_x + x as i32;
            if px >= 0 {
                if abs_y >= 0 {
                    let idx = (abs_y as u32 * screen_width + px as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
                let by = abs_y + self.bounds.height as i32 - 1;
                if by >= 0 {
                    let idx = (by as u32 * screen_width + px as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
            }
        }

        // Left and right
        for y in 0..self.bounds.height {
            let py = abs_y + y as i32;
            if py >= 0 {
                if abs_x >= 0 {
                    let idx = (py as u32 * screen_width + abs_x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
                let rx = abs_x + self.bounds.width as i32 - 1;
                if rx >= 0 {
                    let idx = (py as u32 * screen_width + rx as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
            }
        }

        // Text or placeholder
        let (display_text, text_color) = if self.text.is_empty() {
            (&self.placeholder, theme.text_placeholder)
        } else if self.password {
            // For password mode, we'd show asterisks but need owned string
            (&self.text, theme.text_color)
        } else {
            (&self.text, theme.text_color)
        };

        let text_color_val = text_color.to_argb();
        let char_width = 7;
        let char_height = 10;
        let text_x = abs_x + 4;
        let text_y = abs_y + (self.bounds.height as i32 - char_height) / 2;

        let mut cx = text_x;
        for (i, c) in display_text.chars().enumerate() {
            let display_char = if self.password && !self.text.is_empty() {
                '*'
            } else {
                c
            };

            for dy in 0..char_height {
                for dx in 0..6 {
                    let on = get_char_pixel(display_char, dx as u32, dy as u32);
                    if on {
                        let px = cx + dx;
                        let py = text_y + dy;

                        if px >= 0 && py >= 0 && px < abs_x + self.bounds.width as i32 - 4 {
                            let idx = (py as u32 * screen_width + px as u32) as usize;
                            if idx < buffer.len() {
                                buffer[idx] = text_color_val;
                            }
                        }
                    }
                }
            }
            cx += char_width;
        }

        // Cursor
        if self.focused && self.enabled {
            let cursor_x = text_x + (self.cursor as i32 * char_width);
            let cursor_color = theme.text_color.to_argb();

            for dy in 0..char_height {
                let py = text_y + dy;
                if cursor_x >= 0 && py >= 0 && cursor_x < abs_x + self.bounds.width as i32 - 4 {
                    let idx = (py as u32 * screen_width + cursor_x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = cursor_color;
                    }
                }
            }
        }
    }

    fn handle_mouse_button(&mut self, x: i32, y: i32, button: MouseButton, pressed: bool) {
        if button == MouseButton::Left && pressed {
            self.focused = self.bounds.contains(x, y);
        }
    }

    fn handle_key(&mut self, scancode: u8, pressed: bool) {
        if !self.focused || !pressed {
            return;
        }

        match scancode {
            14 => {
                // Backspace
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.text.remove(self.cursor);
                }
            }
            75 => {
                // Left arrow
                if self.cursor > 0 {
                    self.cursor -= 1;
                }
            }
            77 => {
                // Right arrow
                if self.cursor < self.text.len() {
                    self.cursor += 1;
                }
            }
            71 => {
                // Home
                self.cursor = 0;
            }
            79 => {
                // End
                self.cursor = self.text.len();
            }
            83 => {
                // Delete
                if self.cursor < self.text.len() {
                    self.text.remove(self.cursor);
                }
            }
            _ => {}
        }
    }

    fn handle_char(&mut self, c: char) {
        if !self.focused || !self.enabled {
            return;
        }

        // Only printable characters
        if c.is_control() {
            return;
        }

        if self.text.len() < self.max_length {
            self.text.insert(self.cursor, c);
            self.cursor += 1;
        }
    }
}

fn get_char_pixel(c: char, x: u32, y: u32) -> bool {
    match c {
        'A'..='Z' | 'a'..='z' => {
            let border_x = x == 0 || x == 5;
            let border_y = y == 0 || y == 9;
            let middle = y == 4 && x > 0 && x < 5;
            border_x || border_y || middle
        }
        '0'..='9' => {
            let border_x = x == 0 || x == 5;
            let border_y = y == 0 || y == 9;
            border_x || border_y
        }
        '*' => (y >= 2 && y <= 6) && (x >= 1 && x <= 4),
        ' ' => false,
        _ => {
            let border_x = x == 0 || x == 5;
            let border_y = y == 0 || y == 9;
            border_x || border_y
        }
    }
}
