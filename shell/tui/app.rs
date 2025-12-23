//! =============================================================================
//! HUBLABIO TUI SHELL
//! =============================================================================
//! Terminal user interface for HubLab IO. Provides an Android-like experience
//! in the terminal with app launcher, status bar, and AI integration.
//! =============================================================================

use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;

/// ANSI escape codes for terminal control
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const ITALIC: &str = "\x1b[3m";
    pub const UNDERLINE: &str = "\x1b[4m";

    // Colors
    pub const FG_BLACK: &str = "\x1b[30m";
    pub const FG_RED: &str = "\x1b[31m";
    pub const FG_GREEN: &str = "\x1b[32m";
    pub const FG_YELLOW: &str = "\x1b[33m";
    pub const FG_BLUE: &str = "\x1b[34m";
    pub const FG_MAGENTA: &str = "\x1b[35m";
    pub const FG_CYAN: &str = "\x1b[36m";
    pub const FG_WHITE: &str = "\x1b[37m";

    pub const BG_BLACK: &str = "\x1b[40m";
    pub const BG_BLUE: &str = "\x1b[44m";
    pub const BG_MAGENTA: &str = "\x1b[45m";
    pub const BG_CYAN: &str = "\x1b[46m";

    // True color (24-bit)
    pub fn fg_rgb(r: u8, g: u8, b: u8) -> String {
        alloc::format!("\x1b[38;2;{};{};{}m", r, g, b)
    }

    pub fn bg_rgb(r: u8, g: u8, b: u8) -> String {
        alloc::format!("\x1b[48;2;{};{};{}m", r, g, b)
    }

    // Cursor control
    pub const CLEAR: &str = "\x1b[2J";
    pub const HOME: &str = "\x1b[H";
    pub const HIDE_CURSOR: &str = "\x1b[?25l";
    pub const SHOW_CURSOR: &str = "\x1b[?25h";

    pub fn move_to(row: u16, col: u16) -> String {
        alloc::format!("\x1b[{};{}H", row, col)
    }
}

/// UI Theme
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: &'static str,
    pub background: (u8, u8, u8),
    pub foreground: (u8, u8, u8),
    pub primary: (u8, u8, u8),
    pub secondary: (u8, u8, u8),
    pub accent: (u8, u8, u8),
    pub surface: (u8, u8, u8),
    pub error: (u8, u8, u8),
}

impl Theme {
    pub const MATERIAL_DARK: Theme = Theme {
        name: "Material Dark",
        background: (18, 18, 18),
        foreground: (255, 255, 255),
        primary: (187, 134, 252),    // Purple
        secondary: (3, 218, 198),    // Teal
        accent: (207, 102, 121),     // Pink
        surface: (30, 30, 30),
        error: (207, 102, 121),
    };

    pub const AMOLED: Theme = Theme {
        name: "AMOLED",
        background: (0, 0, 0),
        foreground: (255, 255, 255),
        primary: (0, 255, 136),      // Green
        secondary: (0, 212, 255),    // Blue
        accent: (255, 61, 127),      // Pink
        surface: (20, 20, 20),
        error: (255, 61, 127),
    };

    pub const LIGHT: Theme = Theme {
        name: "Light",
        background: (245, 245, 245),
        foreground: (0, 0, 0),
        primary: (98, 0, 238),       // Purple
        secondary: (3, 218, 198),
        accent: (255, 61, 127),
        surface: (255, 255, 255),
        error: (176, 0, 32),
    };

    pub fn bg(&self) -> String {
        ansi::bg_rgb(self.background.0, self.background.1, self.background.2)
    }

    pub fn fg(&self) -> String {
        ansi::fg_rgb(self.foreground.0, self.foreground.1, self.foreground.2)
    }

    pub fn primary(&self) -> String {
        ansi::fg_rgb(self.primary.0, self.primary.1, self.primary.2)
    }

    pub fn secondary(&self) -> String {
        ansi::fg_rgb(self.secondary.0, self.secondary.1, self.secondary.2)
    }
}

/// App icon (emoji/symbol)
pub type AppIcon = &'static str;

/// Application definition
#[derive(Debug, Clone)]
pub struct AppDef {
    pub id: &'static str,
    pub name: &'static str,
    pub icon: AppIcon,
    pub category: AppCategory,
}

/// App category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppCategory {
    System,
    AI,
    Productivity,
    Media,
    Network,
    Hardware,
}

/// Built-in apps
pub const APPS: &[AppDef] = &[
    // AI Apps
    AppDef { id: "chat", name: "R Chat", icon: "🤖", category: AppCategory::AI },
    AppDef { id: "voice", name: "Voice", icon: "🎤", category: AppCategory::AI },
    AppDef { id: "translate", name: "Translate", icon: "🌍", category: AppCategory::AI },
    AppDef { id: "agents", name: "Agents", icon: "🧠", category: AppCategory::AI },

    // System Apps
    AppDef { id: "files", name: "Files", icon: "📁", category: AppCategory::System },
    AppDef { id: "terminal", name: "Terminal", icon: "💻", category: AppCategory::System },
    AppDef { id: "settings", name: "Settings", icon: "⚙️", category: AppCategory::System },
    AppDef { id: "system", name: "System", icon: "📊", category: AppCategory::System },

    // Productivity
    AppDef { id: "notes", name: "Notes", icon: "📝", category: AppCategory::Productivity },
    AppDef { id: "calendar", name: "Calendar", icon: "📅", category: AppCategory::Productivity },
    AppDef { id: "clock", name: "Clock", icon: "⏰", category: AppCategory::Productivity },
    AppDef { id: "calc", name: "Calculator", icon: "🔢", category: AppCategory::Productivity },

    // Media
    AppDef { id: "camera", name: "Camera", icon: "📷", category: AppCategory::Media },
    AppDef { id: "gallery", name: "Gallery", icon: "🖼️", category: AppCategory::Media },
    AppDef { id: "music", name: "Music", icon: "🎵", category: AppCategory::Media },
    AppDef { id: "video", name: "Video", icon: "🎬", category: AppCategory::Media },

    // Network
    AppDef { id: "wifi", name: "WiFi", icon: "📶", category: AppCategory::Network },
    AppDef { id: "bluetooth", name: "Bluetooth", icon: "🔵", category: AppCategory::Network },
    AppDef { id: "browser", name: "Browser", icon: "🌐", category: AppCategory::Network },
    AppDef { id: "network", name: "Network", icon: "🔌", category: AppCategory::Network },

    // Hardware
    AppDef { id: "gpio", name: "GPIO", icon: "💡", category: AppCategory::Hardware },
    AppDef { id: "power", name: "Power", icon: "🔋", category: AppCategory::Hardware },
    AppDef { id: "sensors", name: "Sensors", icon: "🌡️", category: AppCategory::Hardware },
];

/// System status
#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub battery_percent: u8,
    pub battery_charging: bool,
    pub wifi_connected: bool,
    pub wifi_signal: u8,  // 0-4
    pub bluetooth_on: bool,
    pub time_hour: u8,
    pub time_minute: u8,
    pub notifications: u8,
}

impl Default for SystemStatus {
    fn default() -> Self {
        Self {
            battery_percent: 100,
            battery_charging: false,
            wifi_connected: true,
            wifi_signal: 4,
            bluetooth_on: false,
            time_hour: 12,
            time_minute: 0,
            notifications: 0,
        }
    }
}

/// TUI Application state
pub struct TuiApp {
    /// Terminal width
    pub width: u16,
    /// Terminal height
    pub height: u16,
    /// Current theme
    pub theme: Theme,
    /// System status
    pub status: SystemStatus,
    /// Selected app index
    pub selected_app: usize,
    /// Current screen
    pub current_screen: Screen,
    /// AI chat history
    pub chat_history: Vec<ChatMessage>,
    /// Is running
    running: bool,
}

/// Current screen
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Screen {
    Home,
    App(String),
    Settings,
    Notifications,
}

/// Chat message
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub timestamp: u64,
}

/// Chat role
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

impl TuiApp {
    /// Create a new TUI application
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            theme: Theme::MATERIAL_DARK,
            status: SystemStatus::default(),
            selected_app: 0,
            current_screen: Screen::Home,
            chat_history: Vec::new(),
            running: true,
        }
    }

    /// Set theme
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// Cycle through themes
    pub fn cycle_theme(&mut self) {
        self.theme = match self.theme.name {
            "Material Dark" => Theme::AMOLED,
            "AMOLED" => Theme::LIGHT,
            _ => Theme::MATERIAL_DARK,
        };
    }

    /// Update system status
    pub fn update_status(&mut self, status: SystemStatus) {
        self.status = status;
    }

    /// Render the entire screen
    pub fn render(&self) -> String {
        let mut output = String::new();

        // Clear screen and set background
        output.push_str(ansi::CLEAR);
        output.push_str(ansi::HOME);
        output.push_str(ansi::HIDE_CURSOR);
        output.push_str(&self.theme.bg());

        // Render status bar
        output.push_str(&self.render_status_bar());

        // Render content
        match &self.current_screen {
            Screen::Home => output.push_str(&self.render_home()),
            Screen::App(app_id) => output.push_str(&self.render_app(app_id)),
            Screen::Settings => output.push_str(&self.render_settings()),
            Screen::Notifications => output.push_str(&self.render_notifications()),
        }

        // Render navigation bar
        output.push_str(&self.render_nav_bar());

        output.push_str(ansi::RESET);
        output
    }

    /// Render status bar (top)
    fn render_status_bar(&self) -> String {
        let mut bar = String::new();

        // Background for status bar
        bar.push_str(&ansi::bg_rgb(
            self.theme.surface.0,
            self.theme.surface.1,
            self.theme.surface.2,
        ));

        // Position at top
        bar.push_str(&ansi::move_to(1, 1));

        // Signal strength
        let signal = match self.status.wifi_signal {
            4 => "▁▂▄█",
            3 => "▁▂▄░",
            2 => "▁▂░░",
            1 => "▁░░░",
            _ => "░░░░",
        };

        // WiFi icon
        let wifi = if self.status.wifi_connected {
            alloc::format!("{}📶 ", self.theme.fg())
        } else {
            alloc::format!("{}📵 ", ansi::fg_rgb(128, 128, 128))
        };

        // Time
        let time = alloc::format!(
            "{:02}:{:02}",
            self.status.time_hour,
            self.status.time_minute
        );

        // Battery
        let battery = if self.status.battery_charging {
            alloc::format!("🔋⚡{}%", self.status.battery_percent)
        } else {
            let icon = match self.status.battery_percent {
                90..=100 => "🔋",
                60..=89 => "🔋",
                30..=59 => "🪫",
                10..=29 => "🪫",
                _ => "🪫",
            };
            alloc::format!("{} {}%", icon, self.status.battery_percent)
        };

        // Compose status bar
        let left = alloc::format!("{} {} HubLab IO", signal, wifi);
        let right = alloc::format!("{} {}", time, battery);

        let padding = self.width as usize - left.len() - right.len();

        bar.push_str(&self.theme.fg());
        bar.push_str(&left);
        bar.push_str(&" ".repeat(padding.max(1)));
        bar.push_str(&right);

        bar
    }

    /// Render home screen with app grid
    fn render_home(&self) -> String {
        let mut home = String::new();

        // App grid (4 columns)
        let cols = 4;
        let app_width = self.width as usize / cols;

        home.push_str(&self.theme.bg());
        home.push_str(&self.theme.fg());

        // Start after status bar
        let start_row = 3;
        let apps_per_row = cols;

        for (i, app) in APPS.iter().enumerate() {
            let row = start_row + (i / apps_per_row) as u16 * 3;
            let col = ((i % apps_per_row) * app_width + app_width / 2) as u16;

            // Icon
            home.push_str(&ansi::move_to(row, col - 1));
            if i == self.selected_app {
                home.push_str(&self.theme.primary());
                home.push_str("▶");
            }
            home.push_str(app.icon);
            if i == self.selected_app {
                home.push_str("◀");
                home.push_str(&self.theme.fg());
            }

            // Name
            home.push_str(&ansi::move_to(row + 1, col - app.name.len() as u16 / 2));
            if i == self.selected_app {
                home.push_str(&self.theme.primary());
            } else {
                home.push_str(&self.theme.fg());
            }
            home.push_str(app.name);
        }

        home
    }

    /// Render app screen
    fn render_app(&self, app_id: &str) -> String {
        match app_id {
            "chat" => self.render_chat(),
            "terminal" => self.render_terminal(),
            "settings" => self.render_settings(),
            _ => self.render_placeholder(app_id),
        }
    }

    /// Render AI chat app
    fn render_chat(&self) -> String {
        let mut chat = String::new();

        chat.push_str(&ansi::move_to(3, 1));
        chat.push_str(&self.theme.primary());
        chat.push_str("╭───────────────────────────────────────────────────────────╮\n");
        chat.push_str("│                    🤖 R Chat - AI Assistant               │\n");
        chat.push_str("╰───────────────────────────────────────────────────────────╯\n");

        chat.push_str(&self.theme.fg());

        // Chat history
        let start_row = 6;
        for (i, msg) in self.chat_history.iter().rev().take(10).enumerate() {
            chat.push_str(&ansi::move_to(start_row + i as u16 * 2, 2));

            match msg.role {
                ChatRole::User => {
                    chat.push_str(&self.theme.secondary());
                    chat.push_str("You: ");
                }
                ChatRole::Assistant => {
                    chat.push_str(&self.theme.primary());
                    chat.push_str("AI: ");
                }
                ChatRole::System => {
                    chat.push_str(&ansi::fg_rgb(128, 128, 128));
                    chat.push_str("System: ");
                }
            }

            chat.push_str(&self.theme.fg());
            chat.push_str(&msg.content);
        }

        // Input prompt
        let input_row = self.height - 4;
        chat.push_str(&ansi::move_to(input_row, 1));
        chat.push_str(&self.theme.surface.0.to_string());
        chat.push_str("╭───────────────────────────────────────────────────────────╮\n");
        chat.push_str("│ > _                                                        │\n");
        chat.push_str("╰───────────────────────────────────────────────────────────╯");

        chat
    }

    /// Render terminal app
    fn render_terminal(&self) -> String {
        let mut term = String::new();

        term.push_str(&ansi::move_to(3, 1));
        term.push_str(&self.theme.fg());
        term.push_str("╭───────────────────────────────────────────────────────────╮\n");
        term.push_str("│                    💻 Terminal                            │\n");
        term.push_str("╰───────────────────────────────────────────────────────────╯\n");

        term.push_str(&ansi::move_to(6, 1));
        term.push_str(&self.theme.primary());
        term.push_str("hublabio");
        term.push_str(&ansi::fg_rgb(128, 128, 128));
        term.push_str("@");
        term.push_str(&self.theme.secondary());
        term.push_str("pi");
        term.push_str(&self.theme.fg());
        term.push_str(":~$ ");
        term.push_str(ansi::SHOW_CURSOR);

        term
    }

    /// Render settings screen
    fn render_settings(&self) -> String {
        let mut settings = String::new();

        settings.push_str(&ansi::move_to(3, 1));
        settings.push_str(&self.theme.primary());
        settings.push_str("╭───────────────────────────────────────────────────────────╮\n");
        settings.push_str("│                    ⚙️ Settings                            │\n");
        settings.push_str("╰───────────────────────────────────────────────────────────╯\n\n");

        settings.push_str(&self.theme.fg());

        let items = [
            ("🌙", "Theme", &*alloc::format!("{}", self.theme.name)),
            ("🔊", "Volume", "80%"),
            ("🔆", "Brightness", "70%"),
            ("📶", "WiFi", if self.status.wifi_connected { "Connected" } else { "Off" }),
            ("🔵", "Bluetooth", if self.status.bluetooth_on { "On" } else { "Off" }),
            ("🤖", "AI Model", "qwen2.5:0.5b"),
            ("🎤", "Voice", "Enabled"),
            ("💾", "Storage", "12.4 GB free"),
        ];

        for (i, (icon, name, value)) in items.iter().enumerate() {
            settings.push_str(&ansi::move_to(7 + i as u16, 3));
            settings.push_str(icon);
            settings.push_str(" ");
            settings.push_str(name);

            let padding = 40 - name.len();
            settings.push_str(&" ".repeat(padding));

            settings.push_str(&self.theme.secondary());
            settings.push_str(value);
            settings.push_str(&self.theme.fg());
        }

        settings
    }

    /// Render notifications
    fn render_notifications(&self) -> String {
        let mut notif = String::new();

        notif.push_str(&ansi::move_to(3, 1));
        notif.push_str(&self.theme.primary());
        notif.push_str("╭───────────────────────────────────────────────────────────╮\n");
        notif.push_str("│                    🔔 Notifications                       │\n");
        notif.push_str("╰───────────────────────────────────────────────────────────╯\n\n");

        notif.push_str(&self.theme.fg());
        notif.push_str("  No new notifications\n");

        notif
    }

    /// Render placeholder for apps
    fn render_placeholder(&self, app_id: &str) -> String {
        let app = APPS.iter().find(|a| a.id == app_id);
        let name = app.map(|a| a.name).unwrap_or("Unknown");
        let icon = app.map(|a| a.icon).unwrap_or("❓");

        let mut placeholder = String::new();

        placeholder.push_str(&ansi::move_to(3, 1));
        placeholder.push_str(&self.theme.primary());
        placeholder.push_str("╭───────────────────────────────────────────────────────────╮\n");
        placeholder.push_str(&alloc::format!(
            "│                    {} {}",
            icon, name
        ));
        placeholder.push_str(&" ".repeat(55 - name.len()));
        placeholder.push_str("│\n");
        placeholder.push_str("╰───────────────────────────────────────────────────────────╯\n\n");

        placeholder.push_str(&self.theme.fg());
        placeholder.push_str("  Coming soon...\n");

        placeholder
    }

    /// Render navigation bar (bottom)
    fn render_nav_bar(&self) -> String {
        let mut nav = String::new();

        // Position at bottom
        nav.push_str(&ansi::move_to(self.height - 1, 1));

        // Background
        nav.push_str(&ansi::bg_rgb(
            self.theme.surface.0,
            self.theme.surface.1,
            self.theme.surface.2,
        ));

        // Clear line
        nav.push_str(&" ".repeat(self.width as usize));

        // Navigation buttons
        let third = self.width / 3;

        nav.push_str(&ansi::move_to(self.height - 1, third / 2));
        nav.push_str(&self.theme.fg());
        nav.push_str("◀ Back");

        nav.push_str(&ansi::move_to(self.height - 1, third + third / 2 - 2));
        nav.push_str(&self.theme.primary());
        nav.push_str("● Home");

        nav.push_str(&ansi::move_to(self.height - 1, 2 * third + third / 2 - 3));
        nav.push_str(&self.theme.fg());
        nav.push_str("▢ Recent");

        nav
    }

    /// Handle input
    pub fn handle_input(&mut self, input: char) {
        match input {
            // Navigation
            'h' | 'H' => self.go_home(),
            'q' | 'Q' => self.running = false,
            '\x1b' => self.go_back(), // Escape

            // Theme
            't' | 'T' => self.cycle_theme(),

            // App selection
            'j' | 'J' => self.select_next(),
            'k' | 'K' => self.select_prev(),
            '\n' | '\r' => self.open_selected(),

            // Notifications
            'n' | 'N' => self.toggle_notifications(),

            _ => {}
        }
    }

    /// Go to home screen
    pub fn go_home(&mut self) {
        self.current_screen = Screen::Home;
    }

    /// Go back
    pub fn go_back(&mut self) {
        self.current_screen = Screen::Home;
    }

    /// Select next app
    fn select_next(&mut self) {
        if self.current_screen == Screen::Home {
            self.selected_app = (self.selected_app + 1) % APPS.len();
        }
    }

    /// Select previous app
    fn select_prev(&mut self) {
        if self.current_screen == Screen::Home {
            self.selected_app = if self.selected_app == 0 {
                APPS.len() - 1
            } else {
                self.selected_app - 1
            };
        }
    }

    /// Open selected app
    fn open_selected(&mut self) {
        if self.current_screen == Screen::Home {
            let app_id = APPS[self.selected_app].id.to_string();
            self.current_screen = Screen::App(app_id);
        }
    }

    /// Toggle notifications
    fn toggle_notifications(&mut self) {
        if self.current_screen == Screen::Notifications {
            self.current_screen = Screen::Home;
        } else {
            self.current_screen = Screen::Notifications;
        }
    }

    /// Add chat message
    pub fn add_chat_message(&mut self, role: ChatRole, content: String) {
        self.chat_history.push(ChatMessage {
            role,
            content,
            timestamp: 0, // TODO: Get actual time
        });
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Quit application
    pub fn quit(&mut self) {
        self.running = false;
    }
}

/// Entry point for TUI
pub fn run() -> ! {
    // Get terminal size (hardcoded for now)
    let width = 80;
    let height = 24;

    let mut app = TuiApp::new(width, height);

    // Add welcome message
    app.add_chat_message(
        ChatRole::System,
        String::from("Welcome to HubLab IO! Say 'Hey HubLab' to activate voice control."),
    );

    loop {
        // Render
        let output = app.render();

        // In real implementation, write to terminal
        // write_to_terminal(&output);

        if !app.is_running() {
            break;
        }

        // Read input (in real implementation)
        // let input = read_input();
        // app.handle_input(input);
    }

    // Cleanup
    // write_to_terminal(ansi::SHOW_CURSOR);
    // write_to_terminal(ansi::RESET);

    loop {}
}
