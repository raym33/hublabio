//! GUI Theme System
//!
//! Color schemes and styling for the GUI.

use super::Color;

/// Theme configuration
#[derive(Clone)]
pub struct Theme {
    /// Window title bar background
    pub titlebar_bg: Color,
    /// Window title bar text
    pub titlebar_text: Color,
    /// Window title bar (active)
    pub titlebar_active: Color,
    /// Window content background
    pub content_bg: Color,
    /// Primary text color
    pub text_primary: Color,
    /// Secondary text color
    pub text_secondary: Color,
    /// Accent color
    pub accent: Color,
    /// Button background
    pub button_bg: Color,
    /// Button hover
    pub button_hover: Color,
    /// Button pressed
    pub button_pressed: Color,
    /// Button text
    pub button_text: Color,
    /// Input background
    pub input_bg: Color,
    /// Input border
    pub input_border: Color,
    /// Input focused border
    pub input_focus: Color,
    /// Selection background
    pub selection_bg: Color,
    /// Border color
    pub border_color: Color,
    /// Shadow color
    pub shadow: Color,
    /// Scrollbar track
    pub scrollbar_track: Color,
    /// Scrollbar thumb
    pub scrollbar_thumb: Color,
    /// Desktop background
    pub desktop_bg: Color,
    /// Error color
    pub error: Color,
    /// Success color
    pub success: Color,
    /// Warning color
    pub warning: Color,
}

impl Theme {
    /// Create dark theme (default)
    pub fn dark() -> Self {
        Self {
            titlebar_bg: Color::new(255, 40, 40, 45),
            titlebar_text: Color::new(255, 220, 220, 225),
            titlebar_active: Color::new(255, 60, 60, 70),
            content_bg: Color::new(255, 30, 30, 35),
            text_primary: Color::new(255, 230, 230, 235),
            text_secondary: Color::new(255, 150, 150, 155),
            accent: Color::new(255, 100, 149, 237),  // Cornflower blue
            button_bg: Color::new(255, 55, 55, 60),
            button_hover: Color::new(255, 70, 70, 80),
            button_pressed: Color::new(255, 45, 45, 50),
            button_text: Color::new(255, 230, 230, 235),
            input_bg: Color::new(255, 25, 25, 30),
            input_border: Color::new(255, 70, 70, 75),
            input_focus: Color::new(255, 100, 149, 237),
            selection_bg: Color::new(255, 60, 100, 160),
            border_color: Color::new(255, 60, 60, 65),
            shadow: Color::new(100, 0, 0, 0),
            scrollbar_track: Color::new(255, 35, 35, 40),
            scrollbar_thumb: Color::new(255, 80, 80, 90),
            desktop_bg: Color::new(255, 20, 20, 25),
            error: Color::new(255, 220, 80, 80),
            success: Color::new(255, 80, 200, 120),
            warning: Color::new(255, 220, 180, 80),
        }
    }

    /// Create light theme
    pub fn light() -> Self {
        Self {
            titlebar_bg: Color::new(255, 230, 230, 235),
            titlebar_text: Color::new(255, 40, 40, 45),
            titlebar_active: Color::new(255, 210, 210, 220),
            content_bg: Color::new(255, 250, 250, 252),
            text_primary: Color::new(255, 30, 30, 35),
            text_secondary: Color::new(255, 100, 100, 105),
            accent: Color::new(255, 0, 122, 204),
            button_bg: Color::new(255, 225, 225, 230),
            button_hover: Color::new(255, 210, 210, 220),
            button_pressed: Color::new(255, 195, 195, 205),
            button_text: Color::new(255, 30, 30, 35),
            input_bg: Color::new(255, 255, 255, 255),
            input_border: Color::new(255, 180, 180, 185),
            input_focus: Color::new(255, 0, 122, 204),
            selection_bg: Color::new(255, 180, 210, 255),
            border_color: Color::new(255, 200, 200, 205),
            shadow: Color::new(50, 0, 0, 0),
            scrollbar_track: Color::new(255, 235, 235, 240),
            scrollbar_thumb: Color::new(255, 180, 180, 190),
            desktop_bg: Color::new(255, 240, 240, 245),
            error: Color::new(255, 200, 60, 60),
            success: Color::new(255, 60, 160, 90),
            warning: Color::new(255, 200, 150, 60),
        }
    }

    /// Create high contrast theme
    pub fn high_contrast() -> Self {
        Self {
            titlebar_bg: Color::new(255, 0, 0, 0),
            titlebar_text: Color::new(255, 255, 255, 255),
            titlebar_active: Color::new(255, 0, 80, 160),
            content_bg: Color::new(255, 0, 0, 0),
            text_primary: Color::new(255, 255, 255, 255),
            text_secondary: Color::new(255, 200, 200, 200),
            accent: Color::new(255, 0, 255, 255),
            button_bg: Color::new(255, 0, 0, 0),
            button_hover: Color::new(255, 0, 80, 160),
            button_pressed: Color::new(255, 0, 60, 120),
            button_text: Color::new(255, 255, 255, 255),
            input_bg: Color::new(255, 0, 0, 0),
            input_border: Color::new(255, 255, 255, 255),
            input_focus: Color::new(255, 0, 255, 255),
            selection_bg: Color::new(255, 0, 80, 160),
            border_color: Color::new(255, 255, 255, 255),
            shadow: Color::new(0, 0, 0, 0),
            scrollbar_track: Color::new(255, 0, 0, 0),
            scrollbar_thumb: Color::new(255, 255, 255, 255),
            desktop_bg: Color::new(255, 0, 0, 0),
            error: Color::new(255, 255, 0, 0),
            success: Color::new(255, 0, 255, 0),
            warning: Color::new(255, 255, 255, 0),
        }
    }

    /// AI-inspired purple theme
    pub fn ai_purple() -> Self {
        Self {
            titlebar_bg: Color::new(255, 35, 25, 50),
            titlebar_text: Color::new(255, 220, 210, 240),
            titlebar_active: Color::new(255, 60, 40, 90),
            content_bg: Color::new(255, 25, 18, 35),
            text_primary: Color::new(255, 230, 225, 245),
            text_secondary: Color::new(255, 150, 140, 170),
            accent: Color::new(255, 147, 112, 219),  // Medium purple
            button_bg: Color::new(255, 50, 35, 70),
            button_hover: Color::new(255, 70, 50, 100),
            button_pressed: Color::new(255, 40, 28, 55),
            button_text: Color::new(255, 230, 225, 245),
            input_bg: Color::new(255, 20, 15, 30),
            input_border: Color::new(255, 70, 55, 95),
            input_focus: Color::new(255, 147, 112, 219),
            selection_bg: Color::new(255, 90, 60, 140),
            border_color: Color::new(255, 55, 40, 75),
            shadow: Color::new(100, 10, 5, 20),
            scrollbar_track: Color::new(255, 30, 22, 42),
            scrollbar_thumb: Color::new(255, 80, 60, 110),
            desktop_bg: Color::new(255, 15, 10, 22),
            error: Color::new(255, 255, 100, 130),
            success: Color::new(255, 100, 220, 160),
            warning: Color::new(255, 255, 200, 100),
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

/// Theme manager for runtime theme switching
pub struct ThemeManager {
    current: Theme,
    name: &'static str,
}

impl ThemeManager {
    /// Create new theme manager with dark theme
    pub fn new() -> Self {
        Self {
            current: Theme::dark(),
            name: "dark",
        }
    }

    /// Get current theme
    pub fn theme(&self) -> &Theme {
        &self.current
    }

    /// Get theme name
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Set dark theme
    pub fn set_dark(&mut self) {
        self.current = Theme::dark();
        self.name = "dark";
    }

    /// Set light theme
    pub fn set_light(&mut self) {
        self.current = Theme::light();
        self.name = "light";
    }

    /// Set high contrast theme
    pub fn set_high_contrast(&mut self) {
        self.current = Theme::high_contrast();
        self.name = "high_contrast";
    }

    /// Set AI purple theme
    pub fn set_ai_purple(&mut self) {
        self.current = Theme::ai_purple();
        self.name = "ai_purple";
    }

    /// Set theme by name
    pub fn set_by_name(&mut self, name: &str) -> bool {
        match name {
            "dark" => { self.set_dark(); true }
            "light" => { self.set_light(); true }
            "high_contrast" => { self.set_high_contrast(); true }
            "ai_purple" => { self.set_ai_purple(); true }
            _ => false
        }
    }
}

impl Default for ThemeManager {
    fn default() -> Self {
        Self::new()
    }
}
