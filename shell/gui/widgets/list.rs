//! ListView Widget

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use super::Widget;
use crate::gui::input::MouseButton;
use crate::gui::theme::Theme;
use crate::gui::{Color, Rect};

/// List item
pub struct ListItem {
    pub text: String,
    pub data: Option<u64>,
}

/// ListView widget
pub struct ListView {
    bounds: Rect,
    items: Vec<ListItem>,
    selected: Option<usize>,
    scroll_offset: usize,
    item_height: u32,
    visible: bool,
    on_select: Option<Box<dyn Fn(usize) + Send + Sync>>,
}

impl ListView {
    /// Create a new list view
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            bounds: Rect::new(x, y, width, height),
            items: Vec::new(),
            selected: None,
            scroll_offset: 0,
            item_height: 20,
            visible: true,
            on_select: None,
        }
    }

    /// Add an item
    pub fn add_item(&mut self, text: &str) {
        self.items.push(ListItem {
            text: String::from(text),
            data: None,
        });
    }

    /// Add an item with data
    pub fn add_item_with_data(&mut self, text: &str, data: u64) {
        self.items.push(ListItem {
            text: String::from(text),
            data: Some(data),
        });
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
        self.selected = None;
        self.scroll_offset = 0;
    }

    /// Get selected index
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// Get selected item
    pub fn selected_item(&self) -> Option<&ListItem> {
        self.selected.and_then(|idx| self.items.get(idx))
    }

    /// Set selection
    pub fn set_selected(&mut self, index: Option<usize>) {
        self.selected = index;
    }

    /// Set selection callback
    pub fn on_select<F: Fn(usize) + Send + Sync + 'static>(&mut self, callback: F) {
        self.on_select = Some(Box::new(callback));
    }

    /// Set item height
    pub fn set_item_height(&mut self, height: u32) {
        self.item_height = height;
    }

    /// Number of items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Visible items count
    fn visible_items(&self) -> usize {
        (self.bounds.height / self.item_height) as usize
    }
}

impl Widget for ListView {
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
        let bg = theme.list_bg.to_argb();
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

        // Draw items
        let visible = self.visible_items();
        for i in 0..visible {
            let item_idx = self.scroll_offset + i;
            if item_idx >= self.items.len() {
                break;
            }

            let item = &self.items[item_idx];
            let item_y = abs_y + (i as u32 * self.item_height) as i32;

            // Selection highlight
            if self.selected == Some(item_idx) {
                let sel_bg = theme.list_selection.to_argb();
                for y in 0..self.item_height {
                    for x in 0..self.bounds.width {
                        let px = abs_x + x as i32;
                        let py = item_y + y as i32;

                        if px >= 0 && py >= 0 {
                            let idx = (py as u32 * screen_width + px as u32) as usize;
                            if idx < buffer.len() {
                                buffer[idx] = sel_bg;
                            }
                        }
                    }
                }
            }

            // Item text
            let text_color = if self.selected == Some(item_idx) {
                theme.list_selection_text
            } else {
                theme.text_color
            };

            let text_color_val = text_color.to_argb();
            let char_width = 7;
            let char_height = 10;
            let text_x = abs_x + 4;
            let text_y = item_y + (self.item_height as i32 - char_height) / 2;

            let mut cx = text_x;
            for c in item.text.chars() {
                if cx >= abs_x + self.bounds.width as i32 - 4 {
                    break;
                }

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

    fn handle_mouse_button(&mut self, x: i32, y: i32, button: MouseButton, pressed: bool) {
        if button == MouseButton::Left && pressed && self.bounds.contains(x, y) {
            let rel_y = y - self.bounds.y;
            let item_idx = self.scroll_offset + (rel_y as u32 / self.item_height) as usize;

            if item_idx < self.items.len() {
                self.selected = Some(item_idx);

                if let Some(ref callback) = self.on_select {
                    callback(item_idx);
                }
            }
        }
    }

    fn handle_scroll(&mut self, _dx: i32, dy: i32) {
        if dy < 0 && self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        } else if dy > 0 && self.scroll_offset + self.visible_items() < self.items.len() {
            self.scroll_offset += 1;
        }
    }

    fn handle_key(&mut self, scancode: u8, pressed: bool) {
        if !pressed {
            return;
        }

        match scancode {
            72 => {
                // Up arrow
                if let Some(sel) = self.selected {
                    if sel > 0 {
                        self.selected = Some(sel - 1);
                        if sel - 1 < self.scroll_offset {
                            self.scroll_offset = sel - 1;
                        }
                    }
                } else if !self.items.is_empty() {
                    self.selected = Some(0);
                }
            }
            80 => {
                // Down arrow
                if let Some(sel) = self.selected {
                    if sel + 1 < self.items.len() {
                        self.selected = Some(sel + 1);
                        if sel + 1 >= self.scroll_offset + self.visible_items() {
                            self.scroll_offset = sel + 2 - self.visible_items();
                        }
                    }
                } else if !self.items.is_empty() {
                    self.selected = Some(0);
                }
            }
            28 => {
                // Enter
                if let Some(sel) = self.selected {
                    if let Some(ref callback) = self.on_select {
                        callback(sel);
                    }
                }
            }
            _ => {}
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
        '/' => x == y / 2,
        _ => {
            let border_x = x == 0 || x == 5;
            let border_y = y == 0 || y == 9;
            border_x || border_y
        }
    }
}
