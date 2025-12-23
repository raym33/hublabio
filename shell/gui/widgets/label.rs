//! Label Widget

use alloc::string::String;

use super::Widget;
use crate::gui::theme::Theme;
use crate::gui::{Color, Rect};

/// Text alignment
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

/// Label widget for displaying text
pub struct Label {
    bounds: Rect,
    text: String,
    color: Option<Color>,
    align: TextAlign,
    visible: bool,
}

impl Label {
    /// Create a new label
    pub fn new(text: &str, x: i32, y: i32) -> Self {
        let width = (text.len() * 7).max(10) as u32;

        Self {
            bounds: Rect::new(x, y, width, 12),
            text: String::from(text),
            color: None,
            align: TextAlign::Left,
            visible: true,
        }
    }

    /// Create with specific size
    pub fn with_size(text: &str, x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            bounds: Rect::new(x, y, width, height),
            text: String::from(text),
            color: None,
            align: TextAlign::Left,
            visible: true,
        }
    }

    /// Set text
    pub fn set_text(&mut self, text: &str) {
        self.text = String::from(text);
    }

    /// Get text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Set color
    pub fn set_color(&mut self, color: Color) {
        self.color = Some(color);
    }

    /// Set alignment
    pub fn set_align(&mut self, align: TextAlign) {
        self.align = align;
    }
}

impl Widget for Label {
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

    fn draw(&self, buffer: &mut [u32], screen_width: u32, parent_bounds: Rect, theme: &Theme) {
        if !self.visible {
            return;
        }

        let abs_x = parent_bounds.x + self.bounds.x;
        let abs_y = parent_bounds.y + self.bounds.y;

        let text_color = self.color.unwrap_or(theme.text_color);
        let text_color_val = text_color.to_argb();

        let char_width = 7;
        let char_height = 10;
        let text_width = self.text.len() as i32 * char_width;

        let start_x = match self.align {
            TextAlign::Left => abs_x,
            TextAlign::Center => abs_x + (self.bounds.width as i32 - text_width) / 2,
            TextAlign::Right => abs_x + self.bounds.width as i32 - text_width,
        };

        let start_y = abs_y + (self.bounds.height as i32 - char_height) / 2;

        let mut cx = start_x;
        for c in self.text.chars() {
            for dy in 0..char_height {
                for dx in 0..6 {
                    let on = get_char_pixel(c, dx as u32, dy as u32);
                    if on {
                        let px = cx + dx;
                        let py = start_y + dy;

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
        ' ' => false,
        '.' => y == 8 && x == 2,
        ':' => (y == 3 || y == 6) && x == 2,
        '-' => y == 4 && x > 0 && x < 5,
        _ => {
            let border_x = x == 0 || x == 5;
            let border_y = y == 0 || y == 9;
            border_x || border_y
        }
    }
}
