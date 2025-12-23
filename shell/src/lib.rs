//! HubLab IO Shell
//!
//! User interface for the operating system.
//! Supports TUI, voice, and GUI modes.

#![no_std]

extern crate alloc;

#[cfg(feature = "tui")]
#[path = "../tui/mod.rs"]
pub mod tui;

#[cfg(feature = "gui")]
#[path = "../gui/mod.rs"]
pub mod gui;

#[cfg(feature = "voice")]
#[path = "../voice/mod.rs"]
pub mod voice;

pub mod commands;
pub mod ai_chat;
pub mod themes;

use alloc::string::String;
use alloc::vec::Vec;

/// Shell version
pub const VERSION: &str = "0.1.0";

/// Shell configuration
#[derive(Clone, Debug)]
pub struct ShellConfig {
    /// Shell prompt
    pub prompt: String,
    /// Theme name
    pub theme: String,
    /// Enable AI assistant
    pub ai_enabled: bool,
    /// History size
    pub history_size: usize,
    /// Auto-complete enabled
    pub autocomplete: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            prompt: String::from("hublab> "),
            theme: String::from("material-dark"),
            ai_enabled: true,
            history_size: 1000,
            autocomplete: true,
        }
    }
}

/// Shell state
pub struct Shell {
    config: ShellConfig,
    history: Vec<String>,
    history_index: usize,
    current_input: String,
    running: bool,
}

impl Shell {
    /// Create a new shell
    pub fn new(config: ShellConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            history_index: 0,
            current_input: String::new(),
            running: false,
        }
    }

    /// Start the shell
    pub fn run(&mut self) {
        self.running = true;
        log::info!("HubLab IO Shell v{} started", VERSION);
    }

    /// Process input line
    pub fn process_input(&mut self, input: &str) -> ShellResult {
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return ShellResult::Empty;
        }

        // Add to history
        self.history.push(String::from(trimmed));
        self.history_index = self.history.len();

        // Check for built-in commands
        if let Some(result) = self.try_builtin(trimmed) {
            return result;
        }

        // Check for AI query (starts with ?)
        if trimmed.starts_with('?') {
            return ShellResult::AiQuery(String::from(&trimmed[1..]));
        }

        // Parse as command
        ShellResult::Command(self.parse_command(trimmed))
    }

    /// Try built-in commands
    fn try_builtin(&mut self, input: &str) -> Option<ShellResult> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts.first()?;

        match *cmd {
            "exit" | "quit" => {
                self.running = false;
                Some(ShellResult::Exit)
            }
            "help" => Some(ShellResult::Output(self.help_text())),
            "history" => Some(ShellResult::Output(self.format_history())),
            "clear" => Some(ShellResult::Clear),
            "version" => Some(ShellResult::Output(
                alloc::format!("HubLab IO Shell v{}", VERSION)
            )),
            "theme" => {
                if let Some(name) = parts.get(1) {
                    self.config.theme = String::from(*name);
                    Some(ShellResult::Output(
                        alloc::format!("Theme set to: {}", name)
                    ))
                } else {
                    Some(ShellResult::Output(
                        alloc::format!("Current theme: {}", self.config.theme)
                    ))
                }
            }
            _ => None,
        }
    }

    /// Parse a command line
    fn parse_command(&self, input: &str) -> ParsedCommand {
        let mut parts = input.split_whitespace();
        let name = parts.next().unwrap_or("").to_string();
        let args: Vec<String> = parts.map(String::from).collect();

        ParsedCommand { name, args }
    }

    /// Generate help text
    fn help_text(&self) -> String {
        String::from(
            "HubLab IO Shell Commands:\n\
             \n\
             Built-in Commands:\n\
             help        Show this help message\n\
             clear       Clear the screen\n\
             history     Show command history\n\
             theme [n]   Get/set theme\n\
             version     Show version\n\
             exit        Exit the shell\n\
             \n\
             AI Assistant:\n\
             ?<query>    Ask AI a question\n\
             \n\
             System Commands:\n\
             ls          List files\n\
             cd          Change directory\n\
             cat         Display file contents\n\
             ps          List processes\n\
             top         System monitor\n\
             ai          AI model management\n\
             pkg         Package manager\n"
        )
    }

    /// Format history
    fn format_history(&self) -> String {
        self.history
            .iter()
            .enumerate()
            .map(|(i, cmd)| alloc::format!("{:4}  {}", i + 1, cmd))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Navigate history up
    pub fn history_up(&mut self) -> Option<&str> {
        if self.history_index > 0 {
            self.history_index -= 1;
            self.history.get(self.history_index).map(|s| s.as_str())
        } else {
            None
        }
    }

    /// Navigate history down
    pub fn history_down(&mut self) -> Option<&str> {
        if self.history_index < self.history.len() {
            self.history_index += 1;
            if self.history_index < self.history.len() {
                self.history.get(self.history_index).map(|s| s.as_str())
            } else {
                Some("")
            }
        } else {
            None
        }
    }

    /// Check if shell is running
    pub fn is_running(&self) -> bool {
        self.running
    }
}

/// Parsed command
#[derive(Clone, Debug)]
pub struct ParsedCommand {
    pub name: String,
    pub args: Vec<String>,
}

/// Shell result
#[derive(Clone, Debug)]
pub enum ShellResult {
    /// Empty input
    Empty,
    /// Command to execute
    Command(ParsedCommand),
    /// AI query
    AiQuery(String),
    /// Output to display
    Output(String),
    /// Clear screen
    Clear,
    /// Exit shell
    Exit,
    /// Error
    Error(String),
}
