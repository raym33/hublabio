//! Window Compositor
//!
//! Manages windows, layering, and screen composition.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::{Mutex, RwLock};
use core::sync::atomic::{AtomicU32, AtomicBool, Ordering};

use super::{Rect, Point, Size, Color};
use super::window::{Window, WindowId, WindowFlags};
use super::input::{InputEvent, MouseButton, TouchPhase};
use super::theme::Theme;
use super::render::Renderer;

/// Compositor state
pub struct Compositor {
    /// Screen dimensions
    width: u32,
    height: u32,

    /// Framebuffer address
    framebuffer: usize,

    /// Back buffer for double buffering
    back_buffer: Mutex<Vec<u32>>,

    /// All windows
    windows: RwLock<BTreeMap<WindowId, Arc<RwLock<Window>>>>,

    /// Window z-order (front to back)
    z_order: RwLock<Vec<WindowId>>,

    /// Focused window
    focused: AtomicU32,

    /// Next window ID
    next_id: AtomicU32,

    /// Theme
    theme: RwLock<Theme>,

    /// Mouse position
    mouse_x: AtomicU32,
    mouse_y: AtomicU32,

    /// Mouse buttons pressed
    mouse_buttons: AtomicU32,

    /// Dirty flag (needs redraw)
    dirty: AtomicBool,

    /// Cursor visible
    cursor_visible: AtomicBool,

    /// Wallpaper color
    wallpaper: RwLock<Color>,
}

impl Compositor {
    /// Create a new compositor
    pub fn new(width: u32, height: u32, framebuffer: usize) -> Self {
        let buffer_size = (width * height) as usize;

        Self {
            width,
            height,
            framebuffer,
            back_buffer: Mutex::new(alloc::vec![0u32; buffer_size]),
            windows: RwLock::new(BTreeMap::new()),
            z_order: RwLock::new(Vec::new()),
            focused: AtomicU32::new(0),
            next_id: AtomicU32::new(1),
            theme: RwLock::new(Theme::default()),
            mouse_x: AtomicU32::new(width / 2),
            mouse_y: AtomicU32::new(height / 2),
            mouse_buttons: AtomicU32::new(0),
            dirty: AtomicBool::new(true),
            cursor_visible: AtomicBool::new(true),
            wallpaper: RwLock::new(Color::from_rgb(0x2D2D2D)),
        }
    }

    /// Get screen dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Create a new window
    pub fn create_window(
        &self,
        title: &str,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        flags: WindowFlags,
    ) -> WindowId {
        let id = WindowId(self.next_id.fetch_add(1, Ordering::SeqCst));

        let window = Window::new(id, title, x, y, width, height, flags);
        let window = Arc::new(RwLock::new(window));

        self.windows.write().insert(id, window);
        self.z_order.write().push(id);

        // Focus the new window if it's not a popup
        if !flags.contains(WindowFlags::POPUP) {
            self.focused.store(id.0, Ordering::SeqCst);
        }

        self.dirty.store(true, Ordering::SeqCst);

        id
    }

    /// Close a window
    pub fn close_window(&self, id: WindowId) {
        self.windows.write().remove(&id);
        self.z_order.write().retain(|&wid| wid != id);

        // Update focus if this was the focused window
        if self.focused.load(Ordering::SeqCst) == id.0 {
            let z_order = self.z_order.read();
            if let Some(&new_focus) = z_order.last() {
                self.focused.store(new_focus.0, Ordering::SeqCst);
            } else {
                self.focused.store(0, Ordering::SeqCst);
            }
        }

        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Get a window by ID
    pub fn get_window(&self, id: WindowId) -> Option<Arc<RwLock<Window>>> {
        self.windows.read().get(&id).cloned()
    }

    /// Bring window to front
    pub fn raise_window(&self, id: WindowId) {
        let mut z_order = self.z_order.write();
        if let Some(pos) = z_order.iter().position(|&wid| wid == id) {
            z_order.remove(pos);
            z_order.push(id);
        }

        self.focused.store(id.0, Ordering::SeqCst);
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Get focused window
    pub fn focused_window(&self) -> Option<WindowId> {
        let id = self.focused.load(Ordering::SeqCst);
        if id == 0 {
            None
        } else {
            Some(WindowId(id))
        }
    }

    /// Process input event
    pub fn handle_input(&self, event: InputEvent) {
        match event {
            InputEvent::MouseMove { x, y } => {
                self.mouse_x.store(x.max(0).min(self.width as i32 - 1) as u32, Ordering::SeqCst);
                self.mouse_y.store(y.max(0).min(self.height as i32 - 1) as u32, Ordering::SeqCst);
                self.dirty.store(true, Ordering::SeqCst);

                // Send to focused window
                if let Some(id) = self.focused_window() {
                    if let Some(window) = self.get_window(id) {
                        let mut win = window.write();
                        let bounds = win.bounds();
                        win.handle_mouse_move(x - bounds.x, y - bounds.y);
                    }
                }
            }

            InputEvent::MouseButton { button, pressed } => {
                let mut buttons = self.mouse_buttons.load(Ordering::SeqCst);
                if pressed {
                    buttons |= 1 << (button as u32);
                } else {
                    buttons &= !(1 << (button as u32));
                }
                self.mouse_buttons.store(buttons, Ordering::SeqCst);

                let mx = self.mouse_x.load(Ordering::SeqCst) as i32;
                let my = self.mouse_y.load(Ordering::SeqCst) as i32;

                if pressed && button == MouseButton::Left {
                    // Check which window was clicked
                    let z_order = self.z_order.read();
                    for &id in z_order.iter().rev() {
                        if let Some(window) = self.get_window(id) {
                            let win = window.read();
                            if win.bounds().contains(mx, my) {
                                drop(win);
                                drop(z_order);

                                // Raise and focus
                                self.raise_window(id);

                                // Send click to window
                                if let Some(window) = self.get_window(id) {
                                    let mut win = window.write();
                                    let bounds = win.bounds();
                                    win.handle_mouse_button(
                                        mx - bounds.x,
                                        my - bounds.y,
                                        button,
                                        pressed,
                                    );
                                }
                                return;
                            }
                        }
                    }
                }

                // Send to focused window
                if let Some(id) = self.focused_window() {
                    if let Some(window) = self.get_window(id) {
                        let mut win = window.write();
                        let bounds = win.bounds();
                        win.handle_mouse_button(mx - bounds.x, my - bounds.y, button, pressed);
                    }
                }
            }

            InputEvent::Touch { phase, x, y, .. } => {
                // Convert touch to mouse events
                match phase {
                    TouchPhase::Started => {
                        self.handle_input(InputEvent::MouseMove { x, y });
                        self.handle_input(InputEvent::MouseButton {
                            button: MouseButton::Left,
                            pressed: true,
                        });
                    }
                    TouchPhase::Moved => {
                        self.handle_input(InputEvent::MouseMove { x, y });
                    }
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        self.handle_input(InputEvent::MouseButton {
                            button: MouseButton::Left,
                            pressed: false,
                        });
                    }
                }
            }

            InputEvent::Key { scancode, pressed, .. } => {
                if let Some(id) = self.focused_window() {
                    if let Some(window) = self.get_window(id) {
                        let mut win = window.write();
                        win.handle_key(scancode, pressed);
                    }
                }
            }

            InputEvent::Character(c) => {
                if let Some(id) = self.focused_window() {
                    if let Some(window) = self.get_window(id) {
                        let mut win = window.write();
                        win.handle_char(c);
                    }
                }
            }

            InputEvent::Scroll { dx, dy } => {
                if let Some(id) = self.focused_window() {
                    if let Some(window) = self.get_window(id) {
                        let mut win = window.write();
                        win.handle_scroll(dx, dy);
                    }
                }
            }
        }
    }

    /// Mark compositor as needing redraw
    pub fn invalidate(&self) {
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Invalidate a specific region
    pub fn invalidate_rect(&self, _rect: Rect) {
        // For now, just invalidate everything
        // A more sophisticated compositor would track dirty regions
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Compose and render the screen
    pub fn compose(&self) {
        if !self.dirty.swap(false, Ordering::SeqCst) {
            return;
        }

        let mut buffer = self.back_buffer.lock();
        let theme = self.theme.read();
        let wallpaper = self.wallpaper.read();

        // Clear with wallpaper
        let bg = wallpaper.to_argb();
        for pixel in buffer.iter_mut() {
            *pixel = bg;
        }

        // Draw windows back to front
        let z_order = self.z_order.read();
        let windows = self.windows.read();

        for &id in z_order.iter() {
            if let Some(window) = windows.get(&id) {
                let win = window.read();
                if win.is_visible() {
                    self.draw_window(&mut buffer, &win, &theme, id == self.focused_window().unwrap_or(WindowId(0)));
                }
            }
        }

        // Draw cursor
        if self.cursor_visible.load(Ordering::SeqCst) {
            let mx = self.mouse_x.load(Ordering::SeqCst);
            let my = self.mouse_y.load(Ordering::SeqCst);
            self.draw_cursor(&mut buffer, mx, my);
        }

        // Copy to framebuffer
        self.flip(&buffer);
    }

    /// Draw a window
    fn draw_window(&self, buffer: &mut [u32], window: &Window, theme: &Theme, focused: bool) {
        let bounds = window.bounds();

        // Window shadow
        self.draw_shadow(buffer, bounds, 8);

        // Window background
        let bg_color = if focused {
            theme.window_bg
        } else {
            theme.window_bg_inactive
        };

        self.fill_rect(buffer, bounds, bg_color);

        // Title bar
        if window.has_titlebar() {
            let title_rect = Rect::new(
                bounds.x,
                bounds.y,
                bounds.width,
                theme.titlebar_height,
            );

            let title_color = if focused {
                theme.titlebar_bg
            } else {
                theme.titlebar_bg_inactive
            };

            self.fill_rect(buffer, title_rect, title_color);

            // Title text
            let title = window.title();
            let text_x = bounds.x + 8;
            let text_y = bounds.y + (theme.titlebar_height as i32 - 12) / 2;
            self.draw_text(buffer, text_x, text_y, title, theme.titlebar_text);

            // Close button
            if window.has_close_button() {
                let close_x = bounds.x + bounds.width as i32 - 24;
                let close_y = bounds.y + 4;
                self.draw_close_button(buffer, close_x, close_y, theme);
            }
        }

        // Window border
        let border_color = if focused {
            theme.border_color
        } else {
            theme.border_color_inactive
        };
        self.draw_rect(buffer, bounds, border_color);

        // Draw window content
        let content_bounds = window.content_bounds();
        window.draw(buffer, self.width, content_bounds, theme);
    }

    /// Draw window shadow
    fn draw_shadow(&self, buffer: &mut [u32], bounds: Rect, radius: i32) {
        let shadow_color = Color::with_alpha(0, 0, 0, 64);

        for i in 0..radius {
            let alpha = 64 - (i as u8 * 8);
            let color = Color::with_alpha(0, 0, 0, alpha).to_argb();

            let shadow_rect = Rect::new(
                bounds.x + i,
                bounds.y + i,
                bounds.width,
                bounds.height,
            );

            // Just draw the edges of the shadow
            for x in shadow_rect.x..(shadow_rect.x + shadow_rect.width as i32) {
                let y = shadow_rect.y + shadow_rect.height as i32 - 1 + i;
                if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                    let idx = (y as u32 * self.width + x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = self.blend_pixel(buffer[idx], color);
                    }
                }
            }

            for y in shadow_rect.y..(shadow_rect.y + shadow_rect.height as i32) {
                let x = shadow_rect.x + shadow_rect.width as i32 - 1 + i;
                if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                    let idx = (y as u32 * self.width + x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = self.blend_pixel(buffer[idx], color);
                    }
                }
            }
        }
    }

    /// Draw close button
    fn draw_close_button(&self, buffer: &mut [u32], x: i32, y: i32, theme: &Theme) {
        let size = 16;
        let bg_rect = Rect::new(x, y, size, size);
        self.fill_rect(buffer, bg_rect, Color::from_rgb(0xFF5555));

        // X mark
        let color = Color::WHITE.to_argb();
        for i in 3..13 {
            let x1 = x + i;
            let y1 = y + i;
            let x2 = x + i;
            let y2 = y + (size as i32 - i);

            if x1 >= 0 && x1 < self.width as i32 && y1 >= 0 && y1 < self.height as i32 {
                let idx = (y1 as u32 * self.width + x1 as u32) as usize;
                if idx < buffer.len() {
                    buffer[idx] = color;
                }
            }

            if x2 >= 0 && x2 < self.width as i32 && y2 >= 0 && y2 < self.height as i32 {
                let idx = (y2 as u32 * self.width + x2 as u32) as usize;
                if idx < buffer.len() {
                    buffer[idx] = color;
                }
            }
        }
    }

    /// Fill a rectangle
    fn fill_rect(&self, buffer: &mut [u32], rect: Rect, color: Color) {
        let color_val = color.to_argb();

        for y in rect.y..(rect.y + rect.height as i32) {
            if y < 0 || y >= self.height as i32 {
                continue;
            }

            for x in rect.x..(rect.x + rect.width as i32) {
                if x < 0 || x >= self.width as i32 {
                    continue;
                }

                let idx = (y as u32 * self.width + x as u32) as usize;
                if idx < buffer.len() {
                    if color.a == 255 {
                        buffer[idx] = color_val;
                    } else {
                        buffer[idx] = self.blend_pixel(buffer[idx], color_val);
                    }
                }
            }
        }
    }

    /// Draw rectangle outline
    fn draw_rect(&self, buffer: &mut [u32], rect: Rect, color: Color) {
        let color_val = color.to_argb();

        // Top and bottom
        for x in rect.x..(rect.x + rect.width as i32) {
            if x >= 0 && x < self.width as i32 {
                if rect.y >= 0 && rect.y < self.height as i32 {
                    let idx = (rect.y as u32 * self.width + x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = color_val;
                    }
                }

                let bottom_y = rect.y + rect.height as i32 - 1;
                if bottom_y >= 0 && bottom_y < self.height as i32 {
                    let idx = (bottom_y as u32 * self.width + x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = color_val;
                    }
                }
            }
        }

        // Left and right
        for y in rect.y..(rect.y + rect.height as i32) {
            if y >= 0 && y < self.height as i32 {
                if rect.x >= 0 && rect.x < self.width as i32 {
                    let idx = (y as u32 * self.width + rect.x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = color_val;
                    }
                }

                let right_x = rect.x + rect.width as i32 - 1;
                if right_x >= 0 && right_x < self.width as i32 {
                    let idx = (y as u32 * self.width + right_x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = color_val;
                    }
                }
            }
        }
    }

    /// Draw text (simplified)
    fn draw_text(&self, buffer: &mut [u32], x: i32, y: i32, text: &str, color: Color) {
        let color_val = color.to_argb();
        let mut cx = x;

        for c in text.chars() {
            // Very simple 6x10 character rendering
            for dy in 0..10 {
                for dx in 0..6 {
                    let on = self.get_char_pixel(c, dx, dy);
                    if on {
                        let px = cx + dx as i32;
                        let py = y + dy as i32;
                        if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                            let idx = (py as u32 * self.width + px as u32) as usize;
                            if idx < buffer.len() {
                                buffer[idx] = color_val;
                            }
                        }
                    }
                }
            }
            cx += 7;
        }
    }

    /// Get character pixel (simple font)
    fn get_char_pixel(&self, c: char, x: u32, y: u32) -> bool {
        // Simple patterns for basic characters
        match c {
            'A'..='Z' | 'a'..='z' => {
                // Box with middle line
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
            '-' => y == 4 && x > 0 && x < 5,
            '_' => y == 9,
            '.' => y == 8 && x == 2,
            _ => {
                // Default box
                let border_x = x == 0 || x == 5;
                let border_y = y == 0 || y == 9;
                border_x || border_y
            }
        }
    }

    /// Draw cursor
    fn draw_cursor(&self, buffer: &mut [u32], x: u32, y: u32) {
        let cursor_color = Color::WHITE.to_argb();
        let outline_color = Color::BLACK.to_argb();

        // Arrow cursor (12x18)
        let cursor_shape: &[(i32, i32)] = &[
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11),
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11),
            (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
            (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
            (5, 5), (5, 6), (5, 7),
            (6, 6),
        ];

        for &(dx, dy) in cursor_shape {
            let px = x as i32 + dx;
            let py = y as i32 + dy;

            if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                let idx = (py as u32 * self.width + px as u32) as usize;
                if idx < buffer.len() {
                    buffer[idx] = cursor_color;
                }
            }
        }

        // Outline
        let outline: &[(i32, i32)] = &[
            (1, 0), (0, 12), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7), (7, 6),
        ];

        for &(dx, dy) in outline {
            let px = x as i32 + dx;
            let py = y as i32 + dy;

            if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                let idx = (py as u32 * self.width + px as u32) as usize;
                if idx < buffer.len() {
                    buffer[idx] = outline_color;
                }
            }
        }
    }

    /// Blend two pixels
    fn blend_pixel(&self, bg: u32, fg: u32) -> u32 {
        let fg_a = (fg >> 24) & 0xFF;
        if fg_a == 255 {
            return fg;
        }
        if fg_a == 0 {
            return bg;
        }

        let inv_a = 255 - fg_a;

        let bg_r = (bg >> 16) & 0xFF;
        let bg_g = (bg >> 8) & 0xFF;
        let bg_b = bg & 0xFF;

        let fg_r = (fg >> 16) & 0xFF;
        let fg_g = (fg >> 8) & 0xFF;
        let fg_b = fg & 0xFF;

        let r = (bg_r * inv_a + fg_r * fg_a) / 255;
        let g = (bg_g * inv_a + fg_g * fg_a) / 255;
        let b = (bg_b * inv_a + fg_b * fg_a) / 255;

        0xFF000000 | (r << 16) | (g << 8) | b
    }

    /// Copy back buffer to framebuffer
    fn flip(&self, buffer: &[u32]) {
        let fb = self.framebuffer as *mut u32;
        let len = buffer.len();

        unsafe {
            for i in 0..len {
                core::ptr::write_volatile(fb.add(i), buffer[i]);
            }
        }
    }

    /// Set wallpaper color
    pub fn set_wallpaper(&self, color: Color) {
        *self.wallpaper.write() = color;
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Set theme
    pub fn set_theme(&self, theme: Theme) {
        *self.theme.write() = theme;
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Get current theme
    pub fn theme(&self) -> Theme {
        self.theme.read().clone()
    }

    /// Show/hide cursor
    pub fn set_cursor_visible(&self, visible: bool) {
        self.cursor_visible.store(visible, Ordering::SeqCst);
        self.dirty.store(true, Ordering::SeqCst);
    }

    /// Get list of window IDs
    pub fn window_list(&self) -> Vec<WindowId> {
        self.z_order.read().clone()
    }
}

unsafe impl Send for Compositor {}
unsafe impl Sync for Compositor {}
