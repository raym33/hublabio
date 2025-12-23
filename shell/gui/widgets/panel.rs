//! Panel Widget

use crate::gui::{Rect, Color};
use crate::gui::theme::Theme;
use super::Widget;

/// Panel widget (container with background)
pub struct Panel {
    bounds: Rect,
    background: Option<Color>,
    border: bool,
    visible: bool,
}

impl Panel {
    /// Create a new panel
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            bounds: Rect::new(x, y, width, height),
            background: None,
            border: true,
            visible: true,
        }
    }

    /// Set background color
    pub fn set_background(&mut self, color: Color) {
        self.background = Some(color);
    }

    /// Set border visibility
    pub fn set_border(&mut self, border: bool) {
        self.border = border;
    }
}

impl Widget for Panel {
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

        // Background
        let bg_color = self.background.unwrap_or(theme.panel_bg);
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
        if self.border {
            let border = theme.border_color.to_argb();

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
        }
    }
}
