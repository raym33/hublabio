//! Command Line Parser
//!
//! Parses kernel command line arguments.

use alloc::collections::BTreeMap;
use alloc::string::String;

/// Parsed command line
#[derive(Clone, Debug)]
pub struct CommandLine {
    /// Key-value pairs
    params: BTreeMap<String, String>,
    /// Raw command line
    raw: String,
}

impl CommandLine {
    /// Parse a command line string
    pub fn parse(cmdline: &str) -> Self {
        let mut params = BTreeMap::new();

        for part in cmdline.split_whitespace() {
            if let Some((key, value)) = part.split_once('=') {
                params.insert(String::from(key), String::from(value));
            } else {
                // Boolean flag
                params.insert(String::from(part), String::from("1"));
            }
        }

        Self {
            params,
            raw: String::from(cmdline),
        }
    }

    /// Get a parameter value
    pub fn get(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }

    /// Check if a flag is present
    pub fn has(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    /// Get as integer
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|s| s.parse().ok())
    }

    /// Get as unsigned integer
    pub fn get_uint(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|s| {
            if s.starts_with("0x") || s.starts_with("0X") {
                u64::from_str_radix(&s[2..], 16).ok()
            } else {
                s.parse().ok()
            }
        })
    }

    /// Get raw command line
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Iterator over all parameters
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.params.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// Well-known parameters
pub mod params {
    /// Root device
    pub const ROOT: &str = "root";
    /// Init program
    pub const INIT: &str = "init";
    /// Console device
    pub const CONSOLE: &str = "console";
    /// Log level
    pub const LOGLEVEL: &str = "loglevel";
    /// AI model path
    pub const AI_MODEL: &str = "ai.model";
    /// AI inference mode
    pub const AI_MODE: &str = "ai.mode";
    /// Memory limit
    pub const MEM: &str = "mem";
    /// Number of CPUs
    pub const MAXCPUS: &str = "maxcpus";
    /// No graphics
    pub const NOFB: &str = "nofb";
    /// Debug mode
    pub const DEBUG: &str = "debug";
    /// Quiet boot
    pub const QUIET: &str = "quiet";
    /// Panic timeout
    pub const PANIC: &str = "panic";
}

/// Default command line
pub fn default() -> CommandLine {
    CommandLine::parse("console=ttyS0,115200 loglevel=4")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let cmdline = CommandLine::parse("root=/dev/mmcblk0p2 console=ttyS0,115200 debug");

        assert_eq!(cmdline.get("root"), Some("/dev/mmcblk0p2"));
        assert_eq!(cmdline.get("console"), Some("ttyS0,115200"));
        assert!(cmdline.has("debug"));
        assert!(!cmdline.has("quiet"));
    }

    #[test]
    fn test_hex() {
        let cmdline = CommandLine::parse("addr=0x1000 size=4096");

        assert_eq!(cmdline.get_uint("addr"), Some(0x1000));
        assert_eq!(cmdline.get_uint("size"), Some(4096));
    }
}
