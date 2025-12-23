//! UI Module
//!
//! Widget toolkit for building TUI applications.

#[cfg(feature = "no_std")]
use alloc::{boxed::Box, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{boxed::Box, string::String, vec::Vec};

/// Application trait
pub trait App {
    /// Initialize the application
    fn init(&mut self);

    /// Handle an event
    fn handle_event(&mut self, event: Event) -> bool;

    /// Render the application
    fn render(&self) -> View;

    /// Called when app is about to close
    fn on_close(&mut self) {}
}

/// UI Event
#[derive(Debug, Clone)]
pub enum Event {
    /// Key press
    Key(KeyEvent),
    /// Mouse event
    Mouse(MouseEvent),
    /// Touch event
    Touch(TouchEvent),
    /// Resize event
    Resize { width: u16, height: u16 },
    /// Focus gained
    FocusGained,
    /// Focus lost
    FocusLost,
    /// Tick (for animations)
    Tick,
}

/// Key event
#[derive(Debug, Clone)]
pub struct KeyEvent {
    pub code: KeyCode,
    pub modifiers: Modifiers,
}

/// Key codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    Char(char),
    Enter,
    Escape,
    Backspace,
    Tab,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    Delete,
    Insert,
    F(u8),
}

/// Key modifiers
#[derive(Debug, Clone, Copy, Default)]
pub struct Modifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
}

/// Mouse event
#[derive(Debug, Clone)]
pub struct MouseEvent {
    pub x: u16,
    pub y: u16,
    pub button: MouseButton,
    pub kind: MouseEventKind,
}

/// Mouse button
#[derive(Debug, Clone, Copy)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Mouse event kind
#[derive(Debug, Clone, Copy)]
pub enum MouseEventKind {
    Down,
    Up,
    Drag,
    Moved,
    ScrollUp,
    ScrollDown,
}

/// Touch event
#[derive(Debug, Clone)]
pub struct TouchEvent {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub kind: TouchKind,
}

/// Touch event kind
#[derive(Debug, Clone, Copy)]
pub enum TouchKind {
    Start,
    Move,
    End,
    Cancel,
}

/// View (render tree)
#[derive(Debug, Clone)]
pub struct View {
    pub widget: Widget,
}

impl View {
    /// Create a new view
    pub fn new(widget: Widget) -> Self {
        Self { widget }
    }
}

/// Widget types
#[derive(Debug, Clone)]
pub enum Widget {
    /// Text widget
    Text {
        content: String,
        style: TextStyle,
    },
    /// Container widget
    Container {
        children: Vec<Widget>,
        layout: Layout,
        style: ContainerStyle,
    },
    /// Input field
    Input {
        value: String,
        placeholder: String,
        focused: bool,
    },
    /// Button widget
    Button {
        label: String,
        disabled: bool,
    },
    /// List widget
    List {
        items: Vec<ListItem>,
        selected: usize,
    },
    /// Progress bar
    Progress {
        value: f32,
        max: f32,
    },
    /// Scrollable area
    Scroll {
        child: Box<Widget>,
        offset: u16,
    },
    /// Empty widget
    Empty,
}

/// Text style
#[derive(Debug, Clone, Default)]
pub struct TextStyle {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub color: Option<Color>,
}

/// Container style
#[derive(Debug, Clone, Default)]
pub struct ContainerStyle {
    pub padding: u16,
    pub margin: u16,
    pub border: bool,
    pub background: Option<Color>,
}

/// Layout direction
#[derive(Debug, Clone, Copy, Default)]
pub enum Layout {
    #[default]
    Vertical,
    Horizontal,
    Stack,
}

/// List item
#[derive(Debug, Clone)]
pub struct ListItem {
    pub label: String,
    pub value: String,
}

/// Color
#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// Widget builder helpers
impl Widget {
    pub fn text(content: &str) -> Self {
        Widget::Text {
            content: String::from(content),
            style: TextStyle::default(),
        }
    }

    pub fn button(label: &str) -> Self {
        Widget::Button {
            label: String::from(label),
            disabled: false,
        }
    }

    pub fn input(value: &str, placeholder: &str) -> Self {
        Widget::Input {
            value: String::from(value),
            placeholder: String::from(placeholder),
            focused: false,
        }
    }

    pub fn vstack(children: Vec<Widget>) -> Self {
        Widget::Container {
            children,
            layout: Layout::Vertical,
            style: ContainerStyle::default(),
        }
    }

    pub fn hstack(children: Vec<Widget>) -> Self {
        Widget::Container {
            children,
            layout: Layout::Horizontal,
            style: ContainerStyle::default(),
        }
    }
}
