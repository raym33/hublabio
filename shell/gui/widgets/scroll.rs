//! ScrollView Widget

use alloc::boxed::Box;

use crate::gui::{Rect, Color};
use crate::gui::input::MouseButton;
use crate::gui::theme::Theme;
use super::Widget;

/// ScrollView widget
pub struct ScrollView {
    bounds: Rect,
    content_width: u32,
    content_height: u32,
    scroll_x: i32,
    scroll_y: i32,
    child: Option<Box<dyn Widget>>,
    visible: bool,
    show_scrollbars: bool,
}

impl ScrollView {
    /// Create a new scroll view
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            bounds: Rect::new(x, y, width, height),
            content_width: width,
            content_height: height,
            scroll_x: 0,
            scroll_y: 0,
            child: None,
            visible: true,
            show_scrollbars: true,
        }
    }

    /// Set content size
    pub fn set_content_size(&mut self, width: u32, height: u32) {
        self.content_width = width;
        self.content_height = height;
    }

    /// Set child widget
    pub fn set_child(&mut self, widget: Box<dyn Widget>) {
        self.child = Some(widget);
    }

    /// Scroll to position
    pub fn scroll_to(&mut self, x: i32, y: i32) {
        self.scroll_x = x.max(0).min((self.content_width as i32 - self.bounds.width as i32).max(0));
        self.scroll_y = y.max(0).min((self.content_height as i32 - self.bounds.height as i32).max(0));
    }

    /// Get scroll position
    pub fn scroll_position(&self) -> (i32, i32) {
        (self.scroll_x, self.scroll_y)
    }

    /// Show/hide scrollbars
    pub fn set_scrollbars_visible(&mut self, visible: bool) {
        self.show_scrollbars = visible;
    }

    /// Max scroll Y
    fn max_scroll_y(&self) -> i32 {
        (self.content_height as i32 - self.bounds.height as i32).max(0)
    }

    /// Max scroll X
    fn max_scroll_x(&self) -> i32 {
        (self.content_width as i32 - self.bounds.width as i32).max(0)
    }
}

impl Widget for ScrollView {
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
        let bg = theme.content_bg.to_argb();
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

        // Draw child with scroll offset
        if let Some(ref child) = self.child {
            let child_bounds = Rect::new(
                abs_x - self.scroll_x,
                abs_y - self.scroll_y,
                self.content_width,
                self.content_height,
            );

            // TODO: Implement proper clipping
            child.draw(buffer, screen_width, child_bounds, theme);
        }

        // Draw scrollbars
        if self.show_scrollbars {
            let scrollbar_width = 8u32;

            // Vertical scrollbar
            if self.content_height > self.bounds.height {
                let track_height = self.bounds.height - scrollbar_width;
                let thumb_height = ((self.bounds.height as f32 / self.content_height as f32) * track_height as f32) as u32;
                let thumb_height = thumb_height.max(20);

                let thumb_y = if self.max_scroll_y() > 0 {
                    ((self.scroll_y as f32 / self.max_scroll_y() as f32) * (track_height - thumb_height) as f32) as u32
                } else {
                    0
                };

                // Track
                let track_color = theme.scrollbar_track.to_argb();
                let thumb_color = theme.scrollbar_thumb.to_argb();

                for y in 0..track_height {
                    for x in 0..scrollbar_width {
                        let px = abs_x + self.bounds.width as i32 - scrollbar_width as i32 + x as i32;
                        let py = abs_y + y as i32;

                        if px >= 0 && py >= 0 {
                            let idx = (py as u32 * screen_width + px as u32) as usize;
                            if idx < buffer.len() {
                                let is_thumb = y >= thumb_y && y < thumb_y + thumb_height;
                                buffer[idx] = if is_thumb { thumb_color } else { track_color };
                            }
                        }
                    }
                }
            }

            // Horizontal scrollbar
            if self.content_width > self.bounds.width {
                let track_width = self.bounds.width - scrollbar_width;
                let thumb_width = ((self.bounds.width as f32 / self.content_width as f32) * track_width as f32) as u32;
                let thumb_width = thumb_width.max(20);

                let thumb_x = if self.max_scroll_x() > 0 {
                    ((self.scroll_x as f32 / self.max_scroll_x() as f32) * (track_width - thumb_width) as f32) as u32
                } else {
                    0
                };

                let track_color = theme.scrollbar_track.to_argb();
                let thumb_color = theme.scrollbar_thumb.to_argb();

                for y in 0..scrollbar_width {
                    for x in 0..track_width {
                        let px = abs_x + x as i32;
                        let py = abs_y + self.bounds.height as i32 - scrollbar_width as i32 + y as i32;

                        if px >= 0 && py >= 0 {
                            let idx = (py as u32 * screen_width + px as u32) as usize;
                            if idx < buffer.len() {
                                let is_thumb = x >= thumb_x && x < thumb_x + thumb_width;
                                buffer[idx] = if is_thumb { thumb_color } else { track_color };
                            }
                        }
                    }
                }
            }
        }

        // Border
        let border = theme.border_color.to_argb();

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

    fn handle_scroll(&mut self, dx: i32, dy: i32) {
        let scroll_speed = 20;

        self.scroll_x = (self.scroll_x + dx * scroll_speed)
            .max(0)
            .min(self.max_scroll_x());

        self.scroll_y = (self.scroll_y + dy * scroll_speed)
            .max(0)
            .min(self.max_scroll_y());
    }

    fn handle_mouse_move(&mut self, x: i32, y: i32) {
        if let Some(ref mut child) = self.child {
            child.handle_mouse_move(x + self.scroll_x, y + self.scroll_y);
        }
    }

    fn handle_mouse_button(&mut self, x: i32, y: i32, button: MouseButton, pressed: bool) {
        if let Some(ref mut child) = self.child {
            child.handle_mouse_button(x + self.scroll_x, y + self.scroll_y, button, pressed);
        }
    }

    fn handle_key(&mut self, scancode: u8, pressed: bool) {
        if pressed {
            let scroll_speed = 20;

            match scancode {
                72 => self.scroll_y = (self.scroll_y - scroll_speed).max(0), // Up
                80 => self.scroll_y = (self.scroll_y + scroll_speed).min(self.max_scroll_y()), // Down
                75 => self.scroll_x = (self.scroll_x - scroll_speed).max(0), // Left
                77 => self.scroll_x = (self.scroll_x + scroll_speed).min(self.max_scroll_x()), // Right
                73 => self.scroll_y = (self.scroll_y - self.bounds.height as i32).max(0), // Page Up
                81 => self.scroll_y = (self.scroll_y + self.bounds.height as i32).min(self.max_scroll_y()), // Page Down
                71 => { self.scroll_x = 0; self.scroll_y = 0; } // Home
                79 => { self.scroll_x = self.max_scroll_x(); self.scroll_y = self.max_scroll_y(); } // End
                _ => {
                    if let Some(ref mut child) = self.child {
                        child.handle_key(scancode, pressed);
                    }
                }
            }
        }
    }

    fn handle_char(&mut self, c: char) {
        if let Some(ref mut child) = self.child {
            child.handle_char(c);
        }
    }
}
