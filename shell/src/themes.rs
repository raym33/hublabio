//! Shell Themes
//!
//! Visual themes for the TUI interface.

use alloc::string::String;

/// Color (RGB)
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as u8,
            g: ((hex >> 8) & 0xFF) as u8,
            b: (hex & 0xFF) as u8,
        }
    }

    /// Convert to ANSI escape sequence
    pub fn to_ansi_fg(&self) -> String {
        alloc::format!("\x1b[38;2;{};{};{}m", self.r, self.g, self.b)
    }

    pub fn to_ansi_bg(&self) -> String {
        alloc::format!("\x1b[48;2;{};{};{}m", self.r, self.g, self.b)
    }
}

/// Theme colors
#[derive(Clone, Debug)]
pub struct Theme {
    pub name: &'static str,
    pub background: Color,
    pub foreground: Color,
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
    pub error: Color,
    pub warning: Color,
    pub success: Color,
    pub muted: Color,
    pub border: Color,
}

impl Theme {
    /// Material Dark theme
    pub const MATERIAL_DARK: Theme = Theme {
        name: "material-dark",
        background: Color::from_hex(0x121212),
        foreground: Color::from_hex(0xE0E0E0),
        primary: Color::from_hex(0xBB86FC),
        secondary: Color::from_hex(0x03DAC6),
        accent: Color::from_hex(0xCF6679),
        error: Color::from_hex(0xCF6679),
        warning: Color::from_hex(0xFFB74D),
        success: Color::from_hex(0x81C784),
        muted: Color::from_hex(0x757575),
        border: Color::from_hex(0x333333),
    };

    /// AMOLED Black theme
    pub const AMOLED: Theme = Theme {
        name: "amoled",
        background: Color::from_hex(0x000000),
        foreground: Color::from_hex(0xFFFFFF),
        primary: Color::from_hex(0x00E5FF),
        secondary: Color::from_hex(0x69F0AE),
        accent: Color::from_hex(0xFF4081),
        error: Color::from_hex(0xFF5252),
        warning: Color::from_hex(0xFFD740),
        success: Color::from_hex(0x69F0AE),
        muted: Color::from_hex(0x616161),
        border: Color::from_hex(0x212121),
    };

    /// Light theme
    pub const LIGHT: Theme = Theme {
        name: "light",
        background: Color::from_hex(0xFAFAFA),
        foreground: Color::from_hex(0x212121),
        primary: Color::from_hex(0x6200EE),
        secondary: Color::from_hex(0x03DAC6),
        accent: Color::from_hex(0xB00020),
        error: Color::from_hex(0xB00020),
        warning: Color::from_hex(0xFF6D00),
        success: Color::from_hex(0x00C853),
        muted: Color::from_hex(0x9E9E9E),
        border: Color::from_hex(0xE0E0E0),
    };

    /// Nord theme
    pub const NORD: Theme = Theme {
        name: "nord",
        background: Color::from_hex(0x2E3440),
        foreground: Color::from_hex(0xECEFF4),
        primary: Color::from_hex(0x88C0D0),
        secondary: Color::from_hex(0x81A1C1),
        accent: Color::from_hex(0xB48EAD),
        error: Color::from_hex(0xBF616A),
        warning: Color::from_hex(0xEBCB8B),
        success: Color::from_hex(0xA3BE8C),
        muted: Color::from_hex(0x4C566A),
        border: Color::from_hex(0x3B4252),
    };

    /// Dracula theme
    pub const DRACULA: Theme = Theme {
        name: "dracula",
        background: Color::from_hex(0x282A36),
        foreground: Color::from_hex(0xF8F8F2),
        primary: Color::from_hex(0xBD93F9),
        secondary: Color::from_hex(0x8BE9FD),
        accent: Color::from_hex(0xFF79C6),
        error: Color::from_hex(0xFF5555),
        warning: Color::from_hex(0xF1FA8C),
        success: Color::from_hex(0x50FA7B),
        muted: Color::from_hex(0x6272A4),
        border: Color::from_hex(0x44475A),
    };

    /// Get theme by name
    pub fn by_name(name: &str) -> &'static Theme {
        match name.to_lowercase().as_str() {
            "amoled" | "amoled-black" => &Self::AMOLED,
            "light" => &Self::LIGHT,
            "nord" => &Self::NORD,
            "dracula" => &Self::DRACULA,
            _ => &Self::MATERIAL_DARK,
        }
    }

    /// List available themes
    pub fn available() -> &'static [&'static str] {
        &["material-dark", "amoled", "light", "nord", "dracula"]
    }
}

/// Icon set for TUI
pub struct Icons;

impl Icons {
    // Navigation
    pub const HOME: &'static str = "";
    pub const BACK: &'static str = "";
    pub const MENU: &'static str = "";

    // Apps
    pub const TERMINAL: &'static str = "";
    pub const FILE: &'static str = "";
    pub const FOLDER: &'static str = "";
    pub const SETTINGS: &'static str = "";
    pub const AI: &'static str = "";
    pub const CHAT: &'static str = "";
    pub const MUSIC: &'static str = "";
    pub const VIDEO: &'static str = "";
    pub const IMAGE: &'static str = "";
    pub const CAMERA: &'static str = "";
    pub const CALENDAR: &'static str = "";
    pub const CLOCK: &'static str = "";

    // System
    pub const WIFI: &'static str = "";
    pub const BLUETOOTH: &'static str = "";
    pub const BATTERY_FULL: &'static str = "";
    pub const BATTERY_HALF: &'static str = "";
    pub const BATTERY_LOW: &'static str = "";
    pub const BATTERY_CHARGING: &'static str = "";
    pub const CPU: &'static str = "";
    pub const MEMORY: &'static str = "";
    pub const STORAGE: &'static str = "";

    // Status
    pub const CHECK: &'static str = "";
    pub const CROSS: &'static str = "";
    pub const WARNING: &'static str = "";
    pub const INFO: &'static str = "";
    pub const ERROR: &'static str = "";
    pub const LOADING: &'static str = "";

    // Actions
    pub const PLUS: &'static str = "";
    pub const MINUS: &'static str = "";
    pub const SEARCH: &'static str = "";
    pub const EDIT: &'static str = "";
    pub const DELETE: &'static str = "";
    pub const REFRESH: &'static str = "";
    pub const DOWNLOAD: &'static str = "";
    pub const UPLOAD: &'static str = "";

    // Arrows
    pub const ARROW_UP: &'static str = "";
    pub const ARROW_DOWN: &'static str = "";
    pub const ARROW_LEFT: &'static str = "";
    pub const ARROW_RIGHT: &'static str = "";

    // ASCII fallbacks for terminals without Nerd Fonts
    pub mod ascii {
        pub const HOME: &'static str = "[H]";
        pub const BACK: &'static str = "<-";
        pub const MENU: &'static str = "=";
        pub const TERMINAL: &'static str = ">_";
        pub const FILE: &'static str = "F";
        pub const FOLDER: &'static str = "D";
        pub const SETTINGS: &'static str = "*";
        pub const AI: &'static str = "AI";
        pub const CHECK: &'static str = "[x]";
        pub const CROSS: &'static str = "[X]";
        pub const WARNING: &'static str = "!";
        pub const ERROR: &'static str = "E";
    }
}
