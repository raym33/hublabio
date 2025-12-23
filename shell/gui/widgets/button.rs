//! Button Widget

use alloc::boxed::Box;
use alloc::string::String;

use super::Widget;
use crate::gui::input::MouseButton;
use crate::gui::theme::Theme;
use crate::gui::{Color, Rect};

/// Button widget
pub struct Button {
    bounds: Rect,
    text: String,
    enabled: bool,
    visible: bool,
    hovered: bool,
    pressed: bool,
    on_click: Option<Box<dyn Fn() + Send + Sync>>,
}

impl Button {
    /// Create a new button
    pub fn new(text: &str, x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            bounds: Rect::new(x, y, width, height),
            text: String::from(text),
            enabled: true,
            visible: true,
            hovered: false,
            pressed: false,
            on_click: None,
        }
    }

    /// Set button text
    pub fn set_text(&mut self, text: &str) {
        self.text = String::from(text);
    }

    /// Set click handler
    pub fn on_click<F: Fn() + Send + Sync + 'static>(&mut self, callback: F) {
        self.on_click = Some(Box::new(callback));
    }

    /// Set enabled state
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Widget for Button {
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

        // Determine colors based on state
        let (bg_color, border_color, text_color) = if !self.enabled {
            (
                theme.button_disabled,
                theme.border_color_inactive,
                theme.text_disabled,
            )
        } else if self.pressed {
            (theme.button_pressed, theme.accent_color, theme.button_text)
        } else if self.hovered {
            (theme.button_hover, theme.accent_color, theme.button_text)
        } else {
            (theme.button_bg, theme.border_color, theme.button_text)
        };

        // Draw button background
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

        // Draw border
        let border = border_color.to_argb();

        // Top and bottom
        for x in 0..self.bounds.width {
            let px = abs_x + x as i32;
            let top_y = abs_y;
            let bottom_y = abs_y + self.bounds.height as i32 - 1;

            if px >= 0 {
                if top_y >= 0 {
                    let idx = (top_y as u32 * screen_width + px as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
                if bottom_y >= 0 {
                    let idx = (bottom_y as u32 * screen_width + px as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
            }
        }

        // Left and right
        for y in 0..self.bounds.height {
            let py = abs_y + y as i32;
            let left_x = abs_x;
            let right_x = abs_x + self.bounds.width as i32 - 1;

            if py >= 0 {
                if left_x >= 0 {
                    let idx = (py as u32 * screen_width + left_x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
                if right_x >= 0 {
                    let idx = (py as u32 * screen_width + right_x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = border;
                    }
                }
            }
        }

        // Draw text (centered)
        let text_color_val = text_color.to_argb();
        let char_width = 7;
        let char_height = 10;
        let text_width = self.text.len() as i32 * char_width;
        let text_x = abs_x + (self.bounds.width as i32 - text_width) / 2;
        let text_y = abs_y + (self.bounds.height as i32 - char_height) / 2;

        let mut cx = text_x;
        for c in self.text.chars() {
            for dy in 0..char_height {
                for dx in 0..6 {
                    let on = get_char_pixel(c, dx as u32, dy as u32);
                    if on {
                        let px = cx + dx;
                        let py = text_y + dy;

                        if px >= 0 && py >= 0 {
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
    }

    fn handle_mouse_move(&mut self, x: i32, y: i32) {
        self.hovered = self.bounds.contains(x, y);
    }

    fn handle_mouse_button(&mut self, x: i32, y: i32, button: MouseButton, pressed: bool) {
        if !self.enabled {
            return;
        }

        if button == MouseButton::Left {
            if pressed && self.bounds.contains(x, y) {
                self.pressed = true;
            } else if !pressed && self.pressed {
                self.pressed = false;
                if self.bounds.contains(x, y) {
                    // Click!
                    if let Some(ref callback) = self.on_click {
                        callback();
                    }
                }
            }
        }
    }
}

/// Simple character pixel lookup
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
        ' ' => false,
        _ => {
            let border_x = x == 0 || x == 5;
            let border_y = y == 0 || y == 9;
            border_x || border_y
        }
    }
}
