//! GUI Widgets
//!
//! Basic UI widgets for the HubLab IO GUI.

pub mod button;
pub mod label;
pub mod textbox;
pub mod panel;
pub mod list;
pub mod scroll;

pub use button::Button;
pub use label::Label;
pub use textbox::TextBox;
pub use panel::Panel;
pub use list::ListView;
pub use scroll::ScrollView;

use alloc::boxed::Box;

use crate::gui::{Rect, Color};
use crate::gui::input::MouseButton;
use crate::gui::theme::Theme;

/// Widget trait
pub trait Widget: Send + Sync {
    /// Get widget bounds relative to parent
    fn bounds(&self) -> Rect;

    /// Set widget position
    fn set_position(&mut self, x: i32, y: i32);

    /// Set widget size
    fn set_size(&mut self, width: u32, height: u32);

    /// Check if widget is visible
    fn is_visible(&self) -> bool {
        true
    }

    /// Check if widget is enabled
    fn is_enabled(&self) -> bool {
        true
    }

    /// Draw the widget
    fn draw(&self, buffer: &mut [u32], screen_width: u32, parent_bounds: Rect, theme: &Theme);

    /// Handle mouse move
    fn handle_mouse_move(&mut self, _x: i32, _y: i32) {}

    /// Handle mouse button
    fn handle_mouse_button(&mut self, _x: i32, _y: i32, _button: MouseButton, _pressed: bool) {}

    /// Handle key press
    fn handle_key(&mut self, _scancode: u8, _pressed: bool) {}

    /// Handle character input
    fn handle_char(&mut self, _c: char) {}

    /// Handle scroll
    fn handle_scroll(&mut self, _dx: i32, _dy: i32) {}

    /// Check if point is inside widget
    fn contains(&self, x: i32, y: i32) -> bool {
        self.bounds().contains(x, y)
    }
}

/// Container widget that holds child widgets
pub struct Container {
    bounds: Rect,
    children: alloc::vec::Vec<Box<dyn Widget>>,
    visible: bool,
    background: Option<Color>,
}

impl Container {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            bounds: Rect::new(x, y, width, height),
            children: alloc::vec::Vec::new(),
            visible: true,
            background: None,
        }
    }

    pub fn add_child(&mut self, widget: Box<dyn Widget>) {
        self.children.push(widget);
    }

    pub fn set_background(&mut self, color: Color) {
        self.background = Some(color);
    }

    pub fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut self.children
    }
}

impl Widget for Container {
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

        let abs_bounds = Rect::new(
            parent_bounds.x + self.bounds.x,
            parent_bounds.y + self.bounds.y,
            self.bounds.width,
            self.bounds.height,
        );

        // Draw background if set
        if let Some(bg) = self.background {
            let color = bg.to_argb();
            for y in 0..abs_bounds.height {
                for x in 0..abs_bounds.width {
                    let px = abs_bounds.x + x as i32;
                    let py = abs_bounds.y + y as i32;

                    if px >= 0 && py >= 0 {
                        let idx = (py as u32 * screen_width + px as u32) as usize;
                        if idx < buffer.len() {
                            buffer[idx] = color;
                        }
                    }
                }
            }
        }

        // Draw children
        for child in &self.children {
            child.draw(buffer, screen_width, abs_bounds, theme);
        }
    }

    fn handle_mouse_move(&mut self, x: i32, y: i32) {
        for child in &mut self.children {
            let child_bounds = child.bounds();
            if child_bounds.contains(x, y) {
                child.handle_mouse_move(x - child_bounds.x, y - child_bounds.y);
            }
        }
    }

    fn handle_mouse_button(&mut self, x: i32, y: i32, button: MouseButton, pressed: bool) {
        for child in &mut self.children {
            let child_bounds = child.bounds();
            if child_bounds.contains(x, y) {
                child.handle_mouse_button(x - child_bounds.x, y - child_bounds.y, button, pressed);
            }
        }
    }

    fn handle_key(&mut self, scancode: u8, pressed: bool) {
        // Forward to all children (focused child would handle)
        for child in &mut self.children {
            child.handle_key(scancode, pressed);
        }
    }

    fn handle_char(&mut self, c: char) {
        for child in &mut self.children {
            child.handle_char(c);
        }
    }

    fn handle_scroll(&mut self, dx: i32, dy: i32) {
        for child in &mut self.children {
            child.handle_scroll(dx, dy);
        }
    }
}
