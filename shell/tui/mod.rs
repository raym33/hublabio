//! TUI (Terminal User Interface) Module
//!
//! Complete terminal interface for HubLab IO with:
//! - App launcher and home screen
//! - Line editing with cursor/selection/history
//! - Scrollable views and lists
//! - Themes and styling

pub mod app;
pub mod editor;
pub mod scroll;

// Re-export main types
pub use app::{
    TuiApp, Theme, Screen, AppDef, AppCategory, SystemStatus,
    ChatMessage, ChatRole, APPS, ansi,
};
pub use editor::LineEditor;
pub use scroll::{ScrollBuffer, ScrollList, Scrollbar};

use alloc::string::String;
use alloc::vec::Vec;

/// Input event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputEvent {
    /// Regular character
    Char(char),
    /// Enter/Return key
    Enter,
    /// Backspace
    Backspace,
    /// Delete
    Delete,
    /// Tab
    Tab,
    /// Escape
    Escape,
    /// Arrow keys
    Up,
    Down,
    Left,
    Right,
    /// Home/End
    Home,
    End,
    /// Page Up/Down
    PageUp,
    PageDown,
    /// Control + key
    Ctrl(char),
    /// Alt + key
    Alt(char),
    /// Function keys
    F(u8),
    /// Mouse click (x, y)
    Click(u16, u16),
    /// Mouse scroll
    ScrollUp,
    ScrollDown,
    /// Window resize
    Resize(u16, u16),
    /// Unknown/unhandled
    Unknown,
}

/// Parse ANSI escape sequence to InputEvent
pub fn parse_input(bytes: &[u8]) -> InputEvent {
    if bytes.is_empty() {
        return InputEvent::Unknown;
    }

    // Single byte
    if bytes.len() == 1 {
        return match bytes[0] {
            0x1b => InputEvent::Escape,
            0x0d | 0x0a => InputEvent::Enter,
            0x7f | 0x08 => InputEvent::Backspace,
            0x09 => InputEvent::Tab,
            // Ctrl + letter (1-26)
            c @ 1..=26 => InputEvent::Ctrl((c + b'a' - 1) as char),
            // Regular character
            c if c >= 0x20 && c < 0x7f => InputEvent::Char(c as char),
            _ => InputEvent::Unknown,
        };
    }

    // Escape sequences
    if bytes[0] == 0x1b {
        if bytes.len() >= 2 && bytes[1] == b'[' {
            // CSI sequences
            if bytes.len() >= 3 {
                match bytes[2] {
                    b'A' => return InputEvent::Up,
                    b'B' => return InputEvent::Down,
                    b'C' => return InputEvent::Right,
                    b'D' => return InputEvent::Left,
                    b'H' => return InputEvent::Home,
                    b'F' => return InputEvent::End,
                    b'3' if bytes.len() >= 4 && bytes[3] == b'~' => return InputEvent::Delete,
                    b'5' if bytes.len() >= 4 && bytes[3] == b'~' => return InputEvent::PageUp,
                    b'6' if bytes.len() >= 4 && bytes[3] == b'~' => return InputEvent::PageDown,
                    // Function keys
                    b'1' if bytes.len() >= 4 => {
                        match bytes[3] {
                            b'1' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(1),
                            b'2' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(2),
                            b'3' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(3),
                            b'4' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(4),
                            b'5' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(5),
                            b'7' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(6),
                            b'8' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(7),
                            b'9' if bytes.len() >= 5 && bytes[4] == b'~' => return InputEvent::F(8),
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        } else if bytes.len() >= 2 {
            // Alt + key
            let c = bytes[1] as char;
            if c.is_ascii() {
                return InputEvent::Alt(c);
            }
        }
    }

    InputEvent::Unknown
}

/// Enhanced TUI shell that combines app, editor, and scroll
pub struct TuiShell {
    /// Main TUI app
    pub app: TuiApp,
    /// Line editor for input
    pub editor: LineEditor,
    /// Output scroll buffer
    pub output: ScrollBuffer,
    /// Terminal width
    width: u16,
    /// Terminal height
    height: u16,
}

impl TuiShell {
    /// Create new TUI shell
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            app: TuiApp::new(width, height),
            editor: LineEditor::new(1000),
            output: ScrollBuffer::new(5000, height as usize - 4),
            width,
            height,
        }
    }

    /// Handle input event
    pub fn handle_event(&mut self, event: InputEvent) {
        match self.app.current_screen {
            Screen::Home => self.handle_home_event(event),
            Screen::App(ref app_id) => {
                let app_id = app_id.clone();
                self.handle_app_event(&app_id, event);
            }
            _ => self.handle_default_event(event),
        }
    }

    /// Handle home screen events
    fn handle_home_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::Up | InputEvent::Char('k') => {
                // Move up in app grid (4 columns)
                let current = self.app.selected_app;
                if current >= 4 {
                    self.app.selected_app = current - 4;
                }
            }
            InputEvent::Down | InputEvent::Char('j') => {
                let current = self.app.selected_app;
                if current + 4 < APPS.len() {
                    self.app.selected_app = current + 4;
                }
            }
            InputEvent::Left | InputEvent::Char('h') => {
                if self.app.selected_app > 0 {
                    self.app.selected_app -= 1;
                }
            }
            InputEvent::Right | InputEvent::Char('l') => {
                if self.app.selected_app < APPS.len() - 1 {
                    self.app.selected_app += 1;
                }
            }
            InputEvent::Enter => {
                let app_id = APPS[self.app.selected_app].id;
                self.app.current_screen = Screen::App(String::from(app_id));
            }
            InputEvent::Char('t') => self.app.cycle_theme(),
            InputEvent::Char('q') => self.app.quit(),
            InputEvent::Escape => self.app.quit(),
            _ => {}
        }
    }

    /// Handle app screen events
    fn handle_app_event(&mut self, app_id: &str, event: InputEvent) {
        match app_id {
            "terminal" => self.handle_terminal_event(event),
            "chat" => self.handle_chat_event(event),
            _ => self.handle_default_event(event),
        }
    }

    /// Handle terminal app events
    fn handle_terminal_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::Char(c) => {
                self.editor.insert(c);
            }
            InputEvent::Backspace => {
                self.editor.backspace();
            }
            InputEvent::Delete => {
                self.editor.delete();
            }
            InputEvent::Left => {
                self.editor.move_left();
            }
            InputEvent::Right => {
                self.editor.move_right();
            }
            InputEvent::Home => {
                self.editor.move_home();
            }
            InputEvent::End => {
                self.editor.move_end();
            }
            InputEvent::Up => {
                self.editor.history_prev();
            }
            InputEvent::Down => {
                self.editor.history_next();
            }
            InputEvent::Enter => {
                let command = self.editor.submit();
                self.output.push(alloc::format!("$ {}", command));
                // Execute command here
                self.output.push(String::from("Command executed."));
            }
            InputEvent::Tab => {
                // Tab completion
                let completions = self.get_completions();
                self.editor.set_completions(completions);
                self.editor.complete();
            }
            InputEvent::Ctrl('c') => {
                self.editor.clear();
                self.output.push(String::from("^C"));
            }
            InputEvent::Ctrl('l') => {
                self.output.clear();
            }
            InputEvent::Ctrl('u') => {
                self.editor.clear();
            }
            InputEvent::Ctrl('w') => {
                self.editor.delete_word_back();
            }
            InputEvent::PageUp => {
                self.output.page_up();
            }
            InputEvent::PageDown => {
                self.output.page_down();
            }
            InputEvent::Escape => {
                self.app.go_home();
            }
            _ => {}
        }
    }

    /// Handle chat app events
    fn handle_chat_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::Char(c) => {
                self.editor.insert(c);
            }
            InputEvent::Backspace => {
                self.editor.backspace();
            }
            InputEvent::Enter => {
                let message = self.editor.submit();
                if !message.is_empty() {
                    self.app.add_chat_message(ChatRole::User, message);
                    // AI response would be generated here
                    self.app.add_chat_message(
                        ChatRole::Assistant,
                        String::from("I'm processing your request..."),
                    );
                }
            }
            InputEvent::Escape => {
                self.app.go_home();
            }
            _ => self.handle_terminal_event(event),
        }
    }

    /// Handle default events
    fn handle_default_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::Escape => self.app.go_home(),
            InputEvent::Char('q') => self.app.go_home(),
            _ => {}
        }
    }

    /// Get command completions
    fn get_completions(&self) -> Vec<String> {
        let input = self.editor.content();
        let commands = [
            "help", "exit", "clear", "ls", "cd", "cat", "pwd",
            "ps", "top", "ai", "pkg", "gpio", "wifi", "bluetooth",
        ];

        commands
            .iter()
            .filter(|cmd| cmd.starts_with(&input))
            .map(|s| String::from(*s))
            .collect()
    }

    /// Render the shell
    pub fn render(&self) -> String {
        // The main app handles rendering, but we need to inject
        // editor content and output buffer for terminal/chat screens
        self.app.render()
    }

    /// Render terminal with editor and output
    pub fn render_terminal(&self) -> String {
        let mut output = String::new();
        let theme = &self.app.theme;

        // Header
        output.push_str(&ansi::move_to(3, 1));
        output.push_str(&theme.primary());
        output.push_str("╭───────────────────────────────────────────────────────────╮\n");
        output.push_str("│                    💻 Terminal                            │\n");
        output.push_str("╰───────────────────────────────────────────────────────────╯\n");

        // Output area with scroll
        let visible = self.output.visible_lines();
        for (i, line) in visible.iter().enumerate() {
            output.push_str(&ansi::move_to(6 + i as u16, 2));
            output.push_str(&theme.fg());
            output.push_str(line);
        }

        // Scroll indicator
        if self.output.can_scroll_up() || self.output.can_scroll_down() {
            output.push_str(&ansi::move_to(6, self.width - 5));
            output.push_str(&theme.secondary());
            output.push_str(&self.output.scroll_info());
        }

        // Input line
        let input_row = self.height - 3;
        output.push_str(&ansi::move_to(input_row, 1));
        output.push_str(&theme.primary());
        output.push_str("hublabio");
        output.push_str(&ansi::fg_rgb(128, 128, 128));
        output.push_str("@");
        output.push_str(&theme.secondary());
        output.push_str("pi");
        output.push_str(&theme.fg());
        output.push_str(":~$ ");

        // Editor content with cursor
        output.push_str(&self.editor.render_with_cursor('█'));

        output
    }

    /// Update terminal size
    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        self.output.set_viewport_height(height as usize - 4);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.app.is_running()
    }
}

/// Run the TUI shell
pub fn run_shell() -> ! {
    let mut shell = TuiShell::new(80, 24);

    // Welcome message
    shell.output.push(String::from("HubLab IO Shell v0.1.0"));
    shell.output.push(String::from("Type 'help' for available commands."));
    shell.output.push(String::new());

    loop {
        // Render
        let _output = shell.render();
        // In real implementation: write_to_terminal(&output);

        if !shell.is_running() {
            break;
        }

        // Read input and handle
        // In real implementation:
        // let bytes = read_input_bytes();
        // let event = parse_input(&bytes);
        // shell.handle_event(event);
    }

    loop {}
}
