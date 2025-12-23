//! Window Management
//!
//! Window creation, management, and rendering.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use super::input::MouseButton;
use super::theme::Theme;
use super::widgets::Widget;
use super::{Color, Point, Rect, Size};

/// Window identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WindowId(pub u32);

bitflags::bitflags! {
    /// Window flags
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct WindowFlags: u32 {
        /// Window has a title bar
        const TITLEBAR = 1 << 0;
        /// Window can be resized
        const RESIZABLE = 1 << 1;
        /// Window can be minimized
        const MINIMIZABLE = 1 << 2;
        /// Window can be maximized
        const MAXIMIZABLE = 1 << 3;
        /// Window has a close button
        const CLOSABLE = 1 << 4;
        /// Window is borderless
        const BORDERLESS = 1 << 5;
        /// Window is a popup (doesn't take focus)
        const POPUP = 1 << 6;
        /// Window is always on top
        const ALWAYS_ON_TOP = 1 << 7;
        /// Window is a modal dialog
        const MODAL = 1 << 8;
        /// Window has a drop shadow
        const SHADOW = 1 << 9;
        /// Default window flags
        const DEFAULT = Self::TITLEBAR.bits() | Self::CLOSABLE.bits() | Self::SHADOW.bits();
    }
}

impl Default for WindowFlags {
    fn default() -> Self {
        WindowFlags::DEFAULT
    }
}

/// Window state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowState {
    Normal,
    Minimized,
    Maximized,
    Fullscreen,
}

/// Window structure
pub struct Window {
    /// Window ID
    id: WindowId,

    /// Window title
    title: String,

    /// Window bounds
    bounds: Rect,

    /// Saved bounds (for restore from maximized)
    saved_bounds: Rect,

    /// Window flags
    flags: WindowFlags,

    /// Window state
    state: WindowState,

    /// Is window visible
    visible: bool,

    /// Content buffer (ARGB pixels)
    content: Vec<u32>,

    /// Content dirty flag
    dirty: AtomicBool,

    /// Root widget (optional)
    root_widget: Option<Box<dyn Widget>>,

    /// Event callbacks
    on_close: Option<Box<dyn Fn() + Send + Sync>>,
    on_resize: Option<Box<dyn Fn(u32, u32) + Send + Sync>>,

    /// Dragging state
    dragging: bool,
    drag_start: Point,
    drag_offset: Point,
}

impl Window {
    /// Create a new window
    pub fn new(
        id: WindowId,
        title: &str,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        flags: WindowFlags,
    ) -> Self {
        let content_size = (width * height) as usize;

        Self {
            id,
            title: String::from(title),
            bounds: Rect::new(x, y, width, height),
            saved_bounds: Rect::new(x, y, width, height),
            flags,
            state: WindowState::Normal,
            visible: true,
            content: alloc::vec![0u32; content_size],
            dirty: AtomicBool::new(true),
            root_widget: None,
            on_close: None,
            on_resize: None,
            dragging: false,
            drag_start: Point::new(0, 0),
            drag_offset: Point::new(0, 0),
        }
    }

    /// Get window ID
    pub fn id(&self) -> WindowId {
        self.id
    }

    /// Get window title
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Set window title
    pub fn set_title(&mut self, title: &str) {
        self.title = String::from(title);
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Get window bounds
    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Set window position
    pub fn set_position(&mut self, x: i32, y: i32) {
        self.bounds.x = x;
        self.bounds.y = y;
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Set window size
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.bounds.width = width;
        self.bounds.height = height;

        // Resize content buffer
        let content_size = (width * height) as usize;
        self.content.resize(content_size, 0);

        if let Some(ref callback) = self.on_resize {
            callback(width, height);
        }

        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Get content bounds (excluding title bar)
    pub fn content_bounds(&self) -> Rect {
        if self.has_titlebar() {
            Rect::new(
                self.bounds.x,
                self.bounds.y + 24, // Title bar height
                self.bounds.width,
                self.bounds.height.saturating_sub(24),
            )
        } else {
            self.bounds
        }
    }

    /// Check if window has titlebar
    pub fn has_titlebar(&self) -> bool {
        self.flags.contains(WindowFlags::TITLEBAR)
    }

    /// Check if window has close button
    pub fn has_close_button(&self) -> bool {
        self.flags.contains(WindowFlags::CLOSABLE)
    }

    /// Is window visible
    pub fn is_visible(&self) -> bool {
        self.visible && self.state != WindowState::Minimized
    }

    /// Show window
    pub fn show(&mut self) {
        self.visible = true;
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Hide window
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Minimize window
    pub fn minimize(&mut self) {
        if self.flags.contains(WindowFlags::MINIMIZABLE) {
            self.state = WindowState::Minimized;
        }
    }

    /// Maximize window
    pub fn maximize(&mut self, screen_width: u32, screen_height: u32) {
        if self.flags.contains(WindowFlags::MAXIMIZABLE) {
            match self.state {
                WindowState::Maximized => {
                    // Restore
                    self.bounds = self.saved_bounds;
                    self.state = WindowState::Normal;
                }
                _ => {
                    // Save current bounds and maximize
                    self.saved_bounds = self.bounds;
                    self.bounds = Rect::new(0, 0, screen_width, screen_height);
                    self.state = WindowState::Maximized;
                }
            }
            self.dirty.store(true, Ordering::SeqCst);
        }
    }

    /// Restore window to normal state
    pub fn restore(&mut self) {
        match self.state {
            WindowState::Minimized | WindowState::Maximized | WindowState::Fullscreen => {
                self.bounds = self.saved_bounds;
                self.state = WindowState::Normal;
                self.dirty.store(true, Ordering::SeqCst);
            }
            _ => {}
        }
    }

    /// Set root widget
    pub fn set_root_widget(&mut self, widget: Box<dyn Widget>) {
        self.root_widget = Some(widget);
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Set close callback
    pub fn on_close<F: Fn() + Send + Sync + 'static>(&mut self, callback: F) {
        self.on_close = Some(Box::new(callback));
    }

    /// Set resize callback
    pub fn on_resize<F: Fn(u32, u32) + Send + Sync + 'static>(&mut self, callback: F) {
        self.on_resize = Some(Box::new(callback));
    }

    /// Handle mouse move
    pub fn handle_mouse_move(&mut self, x: i32, y: i32) {
        if self.dragging {
            // Move window
            self.bounds.x = x - self.drag_offset.x;
            self.bounds.y = y - self.drag_offset.y;
            self.dirty.store(true, Ordering::SeqCst);
            return;
        }

        // Forward to widgets
        if let Some(ref mut widget) = self.root_widget {
            let content = self.content_bounds();
            widget.handle_mouse_move(x - content.x, y - content.y);
        }
    }

    /// Handle mouse button
    pub fn handle_mouse_button(&mut self, x: i32, y: i32, button: MouseButton, pressed: bool) {
        if pressed && button == MouseButton::Left {
            // Check title bar for dragging
            if self.has_titlebar() && y < 24 {
                // Check close button
                if self.has_close_button() && x >= self.bounds.width as i32 - 24 {
                    if let Some(ref callback) = self.on_close {
                        callback();
                    }
                    return;
                }

                // Start dragging
                self.dragging = true;
                self.drag_start = Point::new(x, y);
                self.drag_offset = Point::new(x, y);
                return;
            }
        }

        if !pressed && button == MouseButton::Left {
            self.dragging = false;
        }

        // Forward to widgets
        if let Some(ref mut widget) = self.root_widget {
            let content = self.content_bounds();
            widget.handle_mouse_button(x - content.x, y - content.y, button, pressed);
        }
    }

    /// Handle key press
    pub fn handle_key(&mut self, scancode: u8, pressed: bool) {
        if let Some(ref mut widget) = self.root_widget {
            widget.handle_key(scancode, pressed);
        }
    }

    /// Handle character input
    pub fn handle_char(&mut self, c: char) {
        if let Some(ref mut widget) = self.root_widget {
            widget.handle_char(c);
        }
    }

    /// Handle scroll
    pub fn handle_scroll(&mut self, dx: i32, dy: i32) {
        if let Some(ref mut widget) = self.root_widget {
            widget.handle_scroll(dx, dy);
        }
    }

    /// Draw window content
    pub fn draw(&self, buffer: &mut [u32], screen_width: u32, content_bounds: Rect, theme: &Theme) {
        // Draw content background
        let bg = theme.content_bg.to_argb();

        for y in 0..content_bounds.height {
            for x in 0..content_bounds.width {
                let screen_x = content_bounds.x + x as i32;
                let screen_y = content_bounds.y + y as i32;

                if screen_x >= 0 && screen_y >= 0 {
                    let idx = (screen_y as u32 * screen_width + screen_x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = bg;
                    }
                }
            }
        }

        // Draw widgets
        if let Some(ref widget) = self.root_widget {
            widget.draw(buffer, screen_width, content_bounds, theme);
        }
    }

    /// Clear content buffer
    pub fn clear(&mut self, color: Color) {
        let c = color.to_argb();
        for pixel in self.content.iter_mut() {
            *pixel = c;
        }
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Set pixel in content buffer
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Color) {
        if x < self.bounds.width && y < self.bounds.height {
            let idx = (y * self.bounds.width + x) as usize;
            if idx < self.content.len() {
                self.content[idx] = color.to_argb();
                self.dirty.store(true, Ordering::SeqCst);
            }
        }
    }

    /// Fill rectangle in content
    pub fn fill_rect(&mut self, rect: Rect, color: Color) {
        let c = color.to_argb();

        for y in 0..rect.height {
            for x in 0..rect.width {
                let px = rect.x + x as i32;
                let py = rect.y + y as i32;

                if px >= 0
                    && px < self.bounds.width as i32
                    && py >= 0
                    && py < self.bounds.height as i32
                {
                    let idx = (py as u32 * self.bounds.width + px as u32) as usize;
                    if idx < self.content.len() {
                        self.content[idx] = c;
                    }
                }
            }
        }

        self.dirty.store(true, Ordering::SeqCst);
    }
}

unsafe impl Send for Window {}
unsafe impl Sync for Window {}
