//! Shell Commands
//!
//! Built-in command implementations.

use alloc::string::String;
use alloc::vec::Vec;

pub mod fs;
pub mod system;
pub mod ai;
pub mod pkg;

/// Command execution result
#[derive(Debug)]
pub enum CommandResult {
    /// Success with optional output
    Success(Option<String>),
    /// Error with message
    Error(String),
    /// Exit the shell
    Exit,
    /// Continue to next command
    Continue,
}

/// Command trait
pub trait Command {
    /// Command name
    fn name(&self) -> &'static str;

    /// Command description
    fn description(&self) -> &'static str;

    /// Usage string
    fn usage(&self) -> &'static str;

    /// Execute the command
    fn execute(&self, args: &[String]) -> CommandResult;

    /// Tab completion
    fn complete(&self, _partial: &str) -> Vec<String> {
        Vec::new()
    }
}

/// Command registry
pub struct CommandRegistry {
    commands: Vec<&'static dyn Command>,
}

impl CommandRegistry {
    /// Create a new registry with built-in commands
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Register a command
    pub fn register(&mut self, cmd: &'static dyn Command) {
        self.commands.push(cmd);
    }

    /// Find a command by name
    pub fn find(&self, name: &str) -> Option<&'static dyn Command> {
        self.commands.iter()
            .find(|c| c.name() == name)
            .copied()
    }

    /// Get all commands
    pub fn all(&self) -> &[&'static dyn Command] {
        &self.commands
    }

    /// Tab complete a partial command
    pub fn complete(&self, partial: &str) -> Vec<String> {
        self.commands.iter()
            .filter(|c| c.name().starts_with(partial))
            .map(|c| String::from(c.name()))
            .collect()
    }
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in commands

/// Help command
pub struct HelpCommand;

impl Command for HelpCommand {
    fn name(&self) -> &'static str { "help" }
    fn description(&self) -> &'static str { "Show help information" }
    fn usage(&self) -> &'static str { "help [command]" }

    fn execute(&self, _args: &[String]) -> CommandResult {
        CommandResult::Success(Some(String::from(
            "HubLab IO Shell - Type 'help <command>' for detailed help\n\
             \n\
             Commands:\n\
             help     - Show this help\n\
             ls       - List directory contents\n\
             cd       - Change directory\n\
             cat      - Display file contents\n\
             ps       - List processes\n\
             top      - System monitor\n\
             ai       - AI assistant\n\
             pkg      - Package manager\n\
             exit     - Exit shell"
        )))
    }
}

/// Clear command
pub struct ClearCommand;

impl Command for ClearCommand {
    fn name(&self) -> &'static str { "clear" }
    fn description(&self) -> &'static str { "Clear the screen" }
    fn usage(&self) -> &'static str { "clear" }

    fn execute(&self, _args: &[String]) -> CommandResult {
        // Send ANSI clear sequence
        CommandResult::Success(Some(String::from("\x1b[2J\x1b[H")))
    }
}

/// Echo command
pub struct EchoCommand;

impl Command for EchoCommand {
    fn name(&self) -> &'static str { "echo" }
    fn description(&self) -> &'static str { "Display a line of text" }
    fn usage(&self) -> &'static str { "echo [text...]" }

    fn execute(&self, args: &[String]) -> CommandResult {
        CommandResult::Success(Some(args.join(" ")))
    }
}

/// Exit command
pub struct ExitCommand;

impl Command for ExitCommand {
    fn name(&self) -> &'static str { "exit" }
    fn description(&self) -> &'static str { "Exit the shell" }
    fn usage(&self) -> &'static str { "exit" }

    fn execute(&self, _args: &[String]) -> CommandResult {
        CommandResult::Exit
    }
}

/// Pwd command
pub struct PwdCommand;

impl Command for PwdCommand {
    fn name(&self) -> &'static str { "pwd" }
    fn description(&self) -> &'static str { "Print working directory" }
    fn usage(&self) -> &'static str { "pwd" }

    fn execute(&self, _args: &[String]) -> CommandResult {
        // TODO: Get actual working directory
        CommandResult::Success(Some(String::from("/")))
    }
}

/// Env command
pub struct EnvCommand;

impl Command for EnvCommand {
    fn name(&self) -> &'static str { "env" }
    fn description(&self) -> &'static str { "Display environment variables" }
    fn usage(&self) -> &'static str { "env" }

    fn execute(&self, _args: &[String]) -> CommandResult {
        CommandResult::Success(Some(String::from(
            "PATH=/bin:/usr/bin\n\
             HOME=/home/user\n\
             SHELL=/bin/hush\n\
             TERM=hublab-tui\n\
             LANG=en_US.UTF-8"
        )))
    }
}
