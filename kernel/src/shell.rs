//! Kernel Shell
//!
//! Interactive debug shell for kernel diagnostics and control.
//! Accessible via serial console or kernel debugger.

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

/// Shell enabled flag
static SHELL_ENABLED: AtomicBool = AtomicBool::new(false);

/// Command history
static HISTORY: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

/// Maximum history entries
const MAX_HISTORY: usize = 100;

/// Shell command handler
pub type CommandHandler = fn(&[&str]) -> Result<String, ShellError>;

/// Shell command
struct ShellCommand {
    name: &'static str,
    description: &'static str,
    usage: &'static str,
    handler: CommandHandler,
}

/// Built-in commands
static COMMANDS: &[ShellCommand] = &[
    ShellCommand {
        name: "help",
        description: "Show available commands",
        usage: "help [command]",
        handler: cmd_help,
    },
    ShellCommand {
        name: "info",
        description: "Show system information",
        usage: "info",
        handler: cmd_info,
    },
    ShellCommand {
        name: "ps",
        description: "List processes",
        usage: "ps [-a] [-l]",
        handler: cmd_ps,
    },
    ShellCommand {
        name: "mem",
        description: "Show memory statistics",
        usage: "mem",
        handler: cmd_mem,
    },
    ShellCommand {
        name: "cpu",
        description: "Show CPU information",
        usage: "cpu",
        handler: cmd_cpu,
    },
    ShellCommand {
        name: "uptime",
        description: "Show system uptime",
        usage: "uptime",
        handler: cmd_uptime,
    },
    ShellCommand {
        name: "dmesg",
        description: "Show kernel log",
        usage: "dmesg [-n lines]",
        handler: cmd_dmesg,
    },
    ShellCommand {
        name: "kill",
        description: "Send signal to process",
        usage: "kill [-signal] pid",
        handler: cmd_kill,
    },
    ShellCommand {
        name: "reboot",
        description: "Reboot the system",
        usage: "reboot",
        handler: cmd_reboot,
    },
    ShellCommand {
        name: "poweroff",
        description: "Power off the system",
        usage: "poweroff",
        handler: cmd_poweroff,
    },
    ShellCommand {
        name: "mount",
        description: "Show mounted filesystems",
        usage: "mount",
        handler: cmd_mount,
    },
    ShellCommand {
        name: "ls",
        description: "List directory contents",
        usage: "ls [path]",
        handler: cmd_ls,
    },
    ShellCommand {
        name: "cat",
        description: "Display file contents",
        usage: "cat <file>",
        handler: cmd_cat,
    },
    ShellCommand {
        name: "echo",
        description: "Print text",
        usage: "echo <text>",
        handler: cmd_echo,
    },
    ShellCommand {
        name: "clear",
        description: "Clear screen",
        usage: "clear",
        handler: cmd_clear,
    },
    ShellCommand {
        name: "lsmod",
        description: "List loaded modules",
        usage: "lsmod",
        handler: cmd_lsmod,
    },
    ShellCommand {
        name: "lspci",
        description: "List PCI devices",
        usage: "lspci",
        handler: cmd_lspci,
    },
    ShellCommand {
        name: "lsusb",
        description: "List USB devices",
        usage: "lsusb",
        handler: cmd_lsusb,
    },
    ShellCommand {
        name: "ifconfig",
        description: "Show network interfaces",
        usage: "ifconfig",
        handler: cmd_ifconfig,
    },
    ShellCommand {
        name: "sysctl",
        description: "Get/set kernel parameters",
        usage: "sysctl [name[=value]]",
        handler: cmd_sysctl,
    },
    ShellCommand {
        name: "trace",
        description: "Toggle syscall tracing",
        usage: "trace [pid]",
        handler: cmd_trace,
    },
    ShellCommand {
        name: "panic",
        description: "Trigger kernel panic (debug)",
        usage: "panic [message]",
        handler: cmd_panic,
    },
    ShellCommand {
        name: "dump",
        description: "Dump memory",
        usage: "dump <addr> [len]",
        handler: cmd_dump,
    },
    ShellCommand {
        name: "perf",
        description: "Performance statistics",
        usage: "perf",
        handler: cmd_perf,
    },
    ShellCommand {
        name: "history",
        description: "Show command history",
        usage: "history",
        handler: cmd_history,
    },
];

/// Shell error
#[derive(Clone, Debug)]
pub enum ShellError {
    UnknownCommand,
    InvalidArgs,
    NotFound,
    PermissionDenied,
    IoError,
    Other(String),
}

impl ShellError {
    pub fn message(&self) -> String {
        match self {
            ShellError::UnknownCommand => "Unknown command".to_string(),
            ShellError::InvalidArgs => "Invalid arguments".to_string(),
            ShellError::NotFound => "Not found".to_string(),
            ShellError::PermissionDenied => "Permission denied".to_string(),
            ShellError::IoError => "I/O error".to_string(),
            ShellError::Other(msg) => msg.clone(),
        }
    }
}

// ============================================================================
// Command Handlers
// ============================================================================

fn cmd_help(args: &[&str]) -> Result<String, ShellError> {
    if args.len() > 1 {
        // Help for specific command
        let cmd_name = args[1];
        for cmd in COMMANDS {
            if cmd.name == cmd_name {
                return Ok(format!(
                    "{} - {}\n\nUsage: {}\n",
                    cmd.name, cmd.description, cmd.usage
                ));
            }
        }
        return Err(ShellError::UnknownCommand);
    }

    let mut output = String::from("Available commands:\n\n");
    for cmd in COMMANDS {
        output.push_str(&format!("  {:12} - {}\n", cmd.name, cmd.description));
    }
    output.push_str("\nType 'help <command>' for more information.\n");
    Ok(output)
}

fn cmd_info(_args: &[&str]) -> Result<String, ShellError> {
    let uptime_ns = crate::time::monotonic_ns();
    let uptime_s = uptime_ns / 1_000_000_000;
    let hours = uptime_s / 3600;
    let mins = (uptime_s % 3600) / 60;
    let secs = uptime_s % 60;

    let output = format!(
        "{} Kernel v{}\n\
         Architecture: {}\n\
         Uptime: {}h {}m {}s\n\
         Processes: {}\n",
        crate::NAME,
        crate::VERSION,
        core::env!("CARGO_CFG_TARGET_ARCH"),
        hours,
        mins,
        secs,
        crate::process::count(),
    );

    Ok(output)
}

fn cmd_ps(args: &[&str]) -> Result<String, ShellError> {
    let _show_all = args.contains(&"-a");
    let _long_format = args.contains(&"-l");

    let mut output = String::from("  PID   PPID  STATE   NAME\n");
    output.push_str("------------------------------\n");

    // Would iterate over processes
    // For now, placeholder
    output.push_str("    1      0  S       init\n");
    output.push_str("    2      1  S       kthreadd\n");

    Ok(output)
}

fn cmd_mem(_args: &[&str]) -> Result<String, ShellError> {
    // Would get actual memory stats
    let output = format!(
        "Memory Statistics:\n\
         -----------------\n\
         Total:     {} MB\n\
         Used:      {} MB\n\
         Free:      {} MB\n\
         Cached:    {} MB\n\
         Buffers:   {} MB\n\
         Swap:      {} MB\n",
        512, 256, 256, 64, 32, 0
    );

    Ok(output)
}

fn cmd_cpu(_args: &[&str]) -> Result<String, ShellError> {
    let output = format!(
        "CPU Information:\n\
         ---------------\n\
         Architecture: {}\n\
         Cores: {}\n\
         Features: NEON, AES, SHA\n",
        core::env!("CARGO_CFG_TARGET_ARCH"),
        1
    );

    Ok(output)
}

fn cmd_uptime(_args: &[&str]) -> Result<String, ShellError> {
    let uptime_ns = crate::time::monotonic_ns();
    let uptime_s = uptime_ns / 1_000_000_000;
    let days = uptime_s / 86400;
    let hours = (uptime_s % 86400) / 3600;
    let mins = (uptime_s % 3600) / 60;
    let secs = uptime_s % 60;

    let output = format!("Uptime: {} days, {}:{}:{}\n", days, hours, mins, secs);

    Ok(output)
}

fn cmd_dmesg(args: &[&str]) -> Result<String, ShellError> {
    let mut lines = 20;

    // Parse -n argument
    let mut i = 1;
    while i < args.len() {
        if args[i] == "-n" && i + 1 < args.len() {
            lines = args[i + 1].parse().unwrap_or(20);
            i += 2;
        } else {
            i += 1;
        }
    }

    // Would read from kernel log buffer
    let output = format!(
        "[    0.000000] {} Kernel v{}\n\
         [    0.000001] Initializing subsystems...\n\
         [    0.000002] Memory manager initialized\n\
         [    0.000003] Scheduler initialized\n\
         (showing last {} lines)\n",
        crate::NAME,
        crate::VERSION,
        lines
    );

    Ok(output)
}

fn cmd_kill(args: &[&str]) -> Result<String, ShellError> {
    if args.len() < 2 {
        return Err(ShellError::InvalidArgs);
    }

    let mut signal = 15; // SIGTERM default
    let mut pid_str = args[1];

    if args.len() > 2 && args[1].starts_with('-') {
        signal = args[1][1..].parse().unwrap_or(15);
        pid_str = args[2];
    }

    let pid: u32 = pid_str.parse().map_err(|_| ShellError::InvalidArgs)?;

    // Would send signal
    crate::signal::send_signal(
        crate::process::Pid(pid),
        crate::signal::Signal::from_num(signal).unwrap_or(crate::signal::Signal::SIGTERM),
    );

    Ok(format!("Sent signal {} to PID {}\n", signal, pid))
}

fn cmd_reboot(_args: &[&str]) -> Result<String, ShellError> {
    crate::kprintln!("Rebooting...");
    crate::arch::reboot();
    Ok(String::new())
}

fn cmd_poweroff(_args: &[&str]) -> Result<String, ShellError> {
    crate::kprintln!("Powering off...");
    // Would trigger power off
    loop {
        crate::arch::halt();
    }
}

fn cmd_mount(_args: &[&str]) -> Result<String, ShellError> {
    let mut output = String::from("Mounted filesystems:\n");
    output.push_str("--------------------\n");

    // Would list from VFS
    output.push_str("devfs      on /dev      type devfs (rw)\n");
    output.push_str("procfs     on /proc     type procfs (rw)\n");
    output.push_str("sysfs      on /sys      type sysfs (rw)\n");

    Ok(output)
}

fn cmd_ls(args: &[&str]) -> Result<String, ShellError> {
    let path = if args.len() > 1 { args[1] } else { "/" };

    let mut output = format!("Contents of {}:\n", path);

    // Would list from VFS
    if path == "/" || path == "/dev" {
        output.push_str("  null    zero    random    urandom\n");
        output.push_str("  tty0    tty1    console\n");
    } else if path == "/proc" {
        output.push_str("  1/    2/    self/\n");
        output.push_str("  cpuinfo    meminfo    version\n");
    } else {
        return Err(ShellError::NotFound);
    }

    Ok(output)
}

fn cmd_cat(args: &[&str]) -> Result<String, ShellError> {
    if args.len() < 2 {
        return Err(ShellError::InvalidArgs);
    }

    let path = args[1];

    // Would read from VFS
    let content = match path {
        "/proc/version" => format!(
            "{} version {} ({})\n",
            crate::NAME,
            crate::VERSION,
            core::env!("CARGO_CFG_TARGET_ARCH")
        ),
        "/proc/uptime" => {
            let uptime_s = crate::time::monotonic_ns() / 1_000_000_000;
            format!("{}.00 {}.00\n", uptime_s, uptime_s / 2)
        }
        _ => return Err(ShellError::NotFound),
    };

    Ok(content)
}

fn cmd_echo(args: &[&str]) -> Result<String, ShellError> {
    let output = args[1..].join(" ") + "\n";
    Ok(output)
}

fn cmd_clear(_args: &[&str]) -> Result<String, ShellError> {
    // ANSI escape to clear screen
    Ok("\x1b[2J\x1b[H".to_string())
}

fn cmd_lsmod(_args: &[&str]) -> Result<String, ShellError> {
    let output = String::from(
        "Module                  Size  Used by\n\
         kernel (builtin)\n",
    );
    Ok(output)
}

fn cmd_lspci(_args: &[&str]) -> Result<String, ShellError> {
    let output = String::from("No PCI devices (embedded platform)\n");
    Ok(output)
}

fn cmd_lsusb(_args: &[&str]) -> Result<String, ShellError> {
    let mut output = String::from("USB devices:\n");

    // Would list from USB subsystem
    output.push_str("Bus 001 Device 001: ID 0000:0000 Root Hub\n");

    Ok(output)
}

fn cmd_ifconfig(_args: &[&str]) -> Result<String, ShellError> {
    let output = String::from(
        "lo: flags=73<UP,LOOPBACK,RUNNING>\n\
         \tinet 127.0.0.1  netmask 255.0.0.0\n\
         \tinet6 ::1  prefixlen 128\n\
         \tloop  txqueuelen 1000\n\n",
    );
    Ok(output)
}

fn cmd_sysctl(args: &[&str]) -> Result<String, ShellError> {
    if args.len() < 2 {
        // List all
        let output = String::from(
            "kernel.hostname = hublab\n\
             kernel.ostype = HubLabIO\n\
             kernel.osrelease = 0.1.0\n\
             kernel.version = #1\n\
             vm.swappiness = 60\n",
        );
        return Ok(output);
    }

    let param = args[1];
    if param.contains('=') {
        // Set value
        let parts: Vec<&str> = param.splitn(2, '=').collect();
        Ok(format!("{} = {}\n", parts[0], parts[1]))
    } else {
        // Get value
        match param {
            "kernel.hostname" => Ok("kernel.hostname = hublab\n".to_string()),
            "kernel.ostype" => Ok("kernel.ostype = HubLabIO\n".to_string()),
            _ => Err(ShellError::NotFound),
        }
    }
}

fn cmd_trace(args: &[&str]) -> Result<String, ShellError> {
    if args.len() < 2 {
        return Ok("Usage: trace <pid>\n".to_string());
    }

    let pid: u32 = args[1].parse().map_err(|_| ShellError::InvalidArgs)?;
    Ok(format!("Tracing PID {} (not implemented)\n", pid))
}

fn cmd_panic(args: &[&str]) -> Result<String, ShellError> {
    let msg = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "Manual panic from shell".to_string()
    };

    panic!("{}", msg);
}

fn cmd_dump(args: &[&str]) -> Result<String, ShellError> {
    if args.len() < 2 {
        return Err(ShellError::InvalidArgs);
    }

    let addr: usize = usize::from_str_radix(args[1].trim_start_matches("0x"), 16)
        .map_err(|_| ShellError::InvalidArgs)?;

    let len: usize = if args.len() > 2 {
        args[2].parse().unwrap_or(64)
    } else {
        64
    };

    let mut output = format!("Memory dump at 0x{:016x}:\n", addr);

    for offset in (0..len).step_by(16) {
        output.push_str(&format!("{:016x}: ", addr + offset));

        // Hex bytes
        for i in 0..16 {
            if offset + i < len {
                let byte = unsafe { *((addr + offset + i) as *const u8) };
                output.push_str(&format!("{:02x} ", byte));
            } else {
                output.push_str("   ");
            }
            if i == 7 {
                output.push(' ');
            }
        }

        output.push_str(" |");

        // ASCII
        for i in 0..16 {
            if offset + i < len {
                let byte = unsafe { *((addr + offset + i) as *const u8) };
                if byte >= 0x20 && byte < 0x7F {
                    output.push(byte as char);
                } else {
                    output.push('.');
                }
            }
        }

        output.push_str("|\n");
    }

    Ok(output)
}

fn cmd_perf(_args: &[&str]) -> Result<String, ShellError> {
    let output = format!(
        "Performance Statistics:\n\
         ----------------------\n\
         Syscalls:     {}\n\
         Context switches: {}\n\
         Interrupts:   {}\n\
         Page faults:  {}\n",
        0,
        0,
        0,
        crate::pagefault::get_stats().0
    );

    Ok(output)
}

fn cmd_history(_args: &[&str]) -> Result<String, ShellError> {
    let history = HISTORY.lock();
    let mut output = String::new();

    for (i, cmd) in history.iter().enumerate() {
        output.push_str(&format!("{:4}  {}\n", i + 1, cmd));
    }

    Ok(output)
}

// ============================================================================
// Shell Interface
// ============================================================================

/// Execute a shell command
pub fn execute(line: &str) -> Result<String, ShellError> {
    let line = line.trim();

    if line.is_empty() {
        return Ok(String::new());
    }

    // Add to history
    {
        let mut history = HISTORY.lock();
        history.push_back(line.to_string());
        if history.len() > MAX_HISTORY {
            history.pop_front();
        }
    }

    // Parse command and arguments
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(String::new());
    }

    let cmd_name = parts[0];

    // Find command handler
    for cmd in COMMANDS {
        if cmd.name == cmd_name {
            return (cmd.handler)(&parts);
        }
    }

    Err(ShellError::UnknownCommand)
}

/// Interactive shell loop
pub fn run() {
    if !SHELL_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    crate::kprintln!();
    crate::kprintln!("=== {} Kernel Shell ===", crate::NAME);
    crate::kprintln!("Type 'help' for available commands.");
    crate::kprintln!();

    let mut line_buf = String::new();

    loop {
        crate::kprint!("kernel> ");

        // Read line from input
        line_buf.clear();

        loop {
            // Would read from console/UART
            let ch = crate::drivers::input::read_char();

            match ch {
                '\n' | '\r' => {
                    crate::kprintln!();
                    break;
                }
                '\x08' | '\x7F' => {
                    // Backspace
                    if !line_buf.is_empty() {
                        line_buf.pop();
                        crate::kprint!("\x08 \x08");
                    }
                }
                '\x03' => {
                    // Ctrl+C
                    crate::kprintln!("^C");
                    line_buf.clear();
                    break;
                }
                '\x04' => {
                    // Ctrl+D - exit
                    crate::kprintln!();
                    crate::kprintln!("Exiting shell.");
                    return;
                }
                _ if ch >= ' ' => {
                    line_buf.push(ch);
                    crate::kprint!("{}", ch);
                }
                _ => {}
            }
        }

        // Execute command
        match execute(&line_buf) {
            Ok(output) => {
                if !output.is_empty() {
                    crate::kprint!("{}", output);
                }
            }
            Err(e) => {
                crate::kprintln!("Error: {}", e.message());
            }
        }
    }
}

/// Enable shell
pub fn enable() {
    SHELL_ENABLED.store(true, Ordering::SeqCst);
}

/// Disable shell
pub fn disable() {
    SHELL_ENABLED.store(false, Ordering::SeqCst);
}

/// Check if shell is enabled
pub fn is_enabled() -> bool {
    SHELL_ENABLED.load(Ordering::Relaxed)
}

/// Initialize shell
pub fn init() {
    // Shell is disabled by default - enable via boot param or magic sysrq
    crate::kprintln!("  Kernel shell initialized (disabled)");
}
