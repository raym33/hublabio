//! System Commands
//!
//! Commands for system information and management.

use super::{Command, CommandResult};
use alloc::string::String;

/// Process list command
pub struct PsCommand;

impl Command for PsCommand {
    fn name(&self) -> &'static str {
        "ps"
    }
    fn description(&self) -> &'static str {
        "List processes"
    }
    fn usage(&self) -> &'static str {
        "ps"
    }

    fn execute(&self, _args: &[String]) -> CommandResult {
        // TODO: Get actual process list
        CommandResult::Success(Some(String::from(
            "  PID TTY          TIME CMD\n\
               1   ?        00:00:01 init\n\
               2   ?        00:00:00 ai-runtime\n\
               3   tty1     00:00:00 shell",
        )))
    }
}

/// Top/system monitor command
pub struct TopCommand;

impl Command for TopCommand {
    fn name(&self) -> &'static str {
        "top"
    }
    fn description(&self) -> &'static str {
        "System resource monitor"
    }
    fn usage(&self) -> &'static str {
        "top"
    }

    fn execute(&self, _args: &[String]) -> CommandResult {
        // TODO: Get actual system stats
        CommandResult::Success(Some(String::from(
            "HubLab IO System Monitor\n\
             \n\
             CPU:  [####------] 40%\n\
             RAM:  [######----] 60%  512MB/1GB\n\
             NPU:  [##--------] 20%  Hailo-8L\n\
             \n\
             Uptime: 1h 23m 45s",
        )))
    }
}

/// Uname command
pub struct UnameCommand;

impl Command for UnameCommand {
    fn name(&self) -> &'static str {
        "uname"
    }
    fn description(&self) -> &'static str {
        "Print system information"
    }
    fn usage(&self) -> &'static str {
        "uname [-a]"
    }

    fn execute(&self, args: &[String]) -> CommandResult {
        let show_all = args.iter().any(|a| a == "-a");
        if show_all {
            CommandResult::Success(Some(String::from(
                "HubLab 0.1.0 hublab-rpi5 aarch64 GNU/Linux",
            )))
        } else {
            CommandResult::Success(Some(String::from("HubLab")))
        }
    }
}

/// Reboot command
pub struct RebootCommand;

impl Command for RebootCommand {
    fn name(&self) -> &'static str {
        "reboot"
    }
    fn description(&self) -> &'static str {
        "Reboot the system"
    }
    fn usage(&self) -> &'static str {
        "reboot"
    }

    fn execute(&self, _args: &[String]) -> CommandResult {
        // TODO: Implement actual reboot
        CommandResult::Success(Some(String::from("System will reboot...")))
    }
}

/// Shutdown command
pub struct ShutdownCommand;

impl Command for ShutdownCommand {
    fn name(&self) -> &'static str {
        "shutdown"
    }
    fn description(&self) -> &'static str {
        "Shutdown the system"
    }
    fn usage(&self) -> &'static str {
        "shutdown [-h now]"
    }

    fn execute(&self, _args: &[String]) -> CommandResult {
        // TODO: Implement actual shutdown
        CommandResult::Success(Some(String::from("System will shutdown...")))
    }
}

/// Free memory command
pub struct FreeCommand;

impl Command for FreeCommand {
    fn name(&self) -> &'static str {
        "free"
    }
    fn description(&self) -> &'static str {
        "Display memory usage"
    }
    fn usage(&self) -> &'static str {
        "free [-h]"
    }

    fn execute(&self, _args: &[String]) -> CommandResult {
        // TODO: Get actual memory stats
        CommandResult::Success(Some(String::from(
            "              total        used        free      shared  buff/cache   available\n\
             Mem:          1024M        512M        256M         32M        256M        400M\n\
             Swap:            0M          0M          0M",
        )))
    }
}
