//! GUI Compositor
//!
//! A lightweight window compositor for HubLab IO with support for
//! windows, widgets, and touch/mouse input.

pub mod compositor;
pub mod font;
pub mod input;
pub mod layout;
pub mod render;
pub mod theme;
pub mod widgets;
pub mod window;

pub use compositor::Compositor;
pub use font::{char_height, char_width, draw_text, text_width, TextAlign};
pub use input::{InputEvent, Modifiers, MouseButton, TouchPhase};
pub use layout::{Alignment, Anchor, Direction, GridLayout, LinearLayout};
pub use theme::Theme;
pub use widgets::Widget;
pub use window::{Window, WindowFlags, WindowId};

use alloc::sync::Arc;
use spin::RwLock;

/// Global compositor instance
static COMPOSITOR: RwLock<Option<Arc<Compositor>>> = RwLock::new(None);

/// Initialize the GUI compositor
pub fn init(width: u32, height: u32, framebuffer: usize) {
    let compositor = Compositor::new(width, height, framebuffer);
    *COMPOSITOR.write() = Some(Arc::new(compositor));
    crate::kprintln!("  GUI compositor initialized ({}x{})", width, height);
}

/// Get the global compositor
pub fn compositor() -> Option<Arc<Compositor>> {
    COMPOSITOR.read().clone()
}

/// Check if compositor is initialized
pub fn is_initialized() -> bool {
    COMPOSITOR.read().is_some()
}

/// Rectangle structure
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x
            && px < self.x + self.width as i32
            && py >= self.y
            && py < self.y + self.height as i32
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        !(self.x + self.width as i32 <= other.x
            || other.x + other.width as i32 <= self.x
            || self.y + self.height as i32 <= other.y
            || other.y + other.height as i32 <= self.y)
    }

    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        if !self.intersects(other) {
            return None;
        }

        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = (self.x + self.width as i32).min(other.x + other.width as i32);
        let bottom = (self.y + self.height as i32).min(other.y + other.height as i32);

        Some(Rect {
            x,
            y,
            width: (right - x) as u32,
            height: (bottom - y) as u32,
        })
    }
}

/// Point structure
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Size structure
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// Color structure (ARGB)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Color {
    pub a: u8,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn new(a: u8, r: u8, g: u8, b: u8) -> Self {
        Self { a, r, g, b }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { a: 255, r, g, b }
    }

    pub const fn from_rgb(rgb: u32) -> Self {
        Self {
            a: 255,
            r: ((rgb >> 16) & 0xFF) as u8,
            g: ((rgb >> 8) & 0xFF) as u8,
            b: (rgb & 0xFF) as u8,
        }
    }

    pub const fn from_argb(argb: u32) -> Self {
        Self {
            a: ((argb >> 24) & 0xFF) as u8,
            r: ((argb >> 16) & 0xFF) as u8,
            g: ((argb >> 8) & 0xFF) as u8,
            b: (argb & 0xFF) as u8,
        }
    }

    pub const fn to_argb(&self) -> u32 {
        ((self.a as u32) << 24) | ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
    }

    pub fn blend(&self, other: &Color) -> Color {
        if other.a == 255 {
            return *other;
        }
        if other.a == 0 {
            return *self;
        }

        let alpha = other.a as u32;
        let inv_alpha = 255 - alpha;

        Color {
            a: 255,
            r: ((self.r as u32 * inv_alpha + other.r as u32 * alpha) / 255) as u8,
            g: ((self.g as u32 * inv_alpha + other.g as u32 * alpha) / 255) as u8,
            b: ((self.b as u32 * inv_alpha + other.b as u32 * alpha) / 255) as u8,
        }
    }

    // Common colors
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    pub const WHITE: Color = Color::rgb(255, 255, 255);
    pub const RED: Color = Color::rgb(255, 0, 0);
    pub const GREEN: Color = Color::rgb(0, 255, 0);
    pub const BLUE: Color = Color::rgb(0, 0, 255);
    pub const GRAY: Color = Color::rgb(128, 128, 128);
    pub const DARK_GRAY: Color = Color::rgb(64, 64, 64);
    pub const LIGHT_GRAY: Color = Color::rgb(192, 192, 192);
    pub const TRANSPARENT: Color = Color::new(0, 0, 0, 0);
}
