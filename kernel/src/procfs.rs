//! Process Filesystem (procfs)
//!
//! Provides /proc filesystem for process and system information.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

/// Read procfs entry
pub fn read_entry(path: &str) -> Option<Vec<u8>> {
    let parts: Vec<&str> = path.trim_start_matches("/proc/").split('/').collect();

    if parts.is_empty() {
        return None;
    }

    match parts[0] {
        // System information
        "meminfo" => Some(generate_meminfo()),
        "cpuinfo" => Some(generate_cpuinfo()),
        "version" => Some(generate_version()),
        "uptime" => Some(generate_uptime()),
        "loadavg" => Some(generate_loadavg()),
        "stat" => Some(generate_stat()),
        "filesystems" => Some(generate_filesystems()),
        "mounts" => Some(generate_mounts()),
        "cmdline" => Some(generate_cmdline()),
        "interrupts" => Some(generate_interrupts()),

        // Self (current process)
        "self" => {
            if let Some(proc) = crate::process::current() {
                read_process_entry(proc.pid.0, &parts[1..])
            } else {
                None
            }
        }

        // Process by PID
        pid_str => {
            if let Ok(pid) = pid_str.parse::<u64>() {
                read_process_entry(pid, &parts[1..])
            } else {
                None
            }
        }
    }
}

/// Read process-specific entry
fn read_process_entry(pid: u64, parts: &[&str]) -> Option<Vec<u8>> {
    // Find process
    let processes = crate::process::list();
    let proc = processes.iter().find(|p| p.pid.0 == pid)?;

    if parts.is_empty() {
        // Directory listing
        return Some(b"cmdline\ncomm\nenviron\nexe\nfd\nmaps\nmem\nstat\nstatus\n".to_vec());
    }

    match parts[0] {
        "stat" => Some(generate_process_stat(proc)),
        "status" => Some(generate_process_status(proc)),
        "cmdline" => Some(generate_process_cmdline(proc)),
        "comm" => Some(generate_process_comm(proc)),
        "environ" => Some(generate_process_environ(proc)),
        "maps" => Some(generate_process_maps(proc)),
        "exe" => Some(generate_process_exe(proc)),
        "cwd" => Some(proc.cwd.as_bytes().to_vec()),
        "fd" => {
            if parts.len() == 1 {
                // List file descriptors
                Some(generate_process_fds(proc))
            } else {
                // Specific fd info
                None
            }
        }
        _ => None,
    }
}

/// Generate /proc/meminfo
fn generate_meminfo() -> Vec<u8> {
    let stats = crate::memory::stats();

    let mut s = String::new();
    let _ = writeln!(s, "MemTotal:       {} kB", stats.total / 1024);
    let _ = writeln!(s, "MemFree:        {} kB", stats.free / 1024);
    let _ = writeln!(s, "MemAvailable:   {} kB", stats.free / 1024);
    let _ = writeln!(s, "Buffers:        0 kB");
    let _ = writeln!(s, "Cached:         0 kB");
    let _ = writeln!(s, "SwapTotal:      0 kB");
    let _ = writeln!(s, "SwapFree:       0 kB");

    s.into_bytes()
}

/// Generate /proc/cpuinfo
fn generate_cpuinfo() -> Vec<u8> {
    let mut s = String::new();

    // ARM64 CPU info
    let _ = writeln!(s, "processor\t: 0");
    let _ = writeln!(s, "BogoMIPS\t: 100.00");
    let _ = writeln!(s, "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics");
    let _ = writeln!(s, "CPU implementer\t: 0x41");
    let _ = writeln!(s, "CPU architecture: 8");
    let _ = writeln!(s, "CPU variant\t: 0x0");
    let _ = writeln!(s, "CPU part\t: 0xd08");
    let _ = writeln!(s, "CPU revision\t: 3");
    let _ = writeln!(s, "");
    let _ = writeln!(s, "Hardware\t: HubLab IO");
    let _ = writeln!(s, "Revision\t: 0000");
    let _ = writeln!(s, "Serial\t\t: 0000000000000000");

    s.into_bytes()
}

/// Generate /proc/version
fn generate_version() -> Vec<u8> {
    format!(
        "HubLab IO version {} (rust@hublab) (rustc {}) #1 SMP\n",
        crate::VERSION,
        "1.75.0"
    ).into_bytes()
}

/// Generate /proc/uptime
fn generate_uptime() -> Vec<u8> {
    let uptime = crate::syscall::uptime_secs();
    format!("{}.00 0.00\n", uptime).into_bytes()
}

/// Generate /proc/loadavg
fn generate_loadavg() -> Vec<u8> {
    let procs = crate::process::list().len();
    format!("0.00 0.00 0.00 1/{} 1\n", procs).into_bytes()
}

/// Generate /proc/stat
fn generate_stat() -> Vec<u8> {
    let mut s = String::new();

    // CPU stats (all zeros for now)
    let _ = writeln!(s, "cpu  0 0 0 0 0 0 0 0 0 0");
    let _ = writeln!(s, "cpu0 0 0 0 0 0 0 0 0 0 0");

    // Process stats
    let _ = writeln!(s, "processes 1");
    let _ = writeln!(s, "procs_running 1");
    let _ = writeln!(s, "procs_blocked 0");

    // Boot time
    let _ = writeln!(s, "btime 0");

    s.into_bytes()
}

/// Generate /proc/filesystems
fn generate_filesystems() -> Vec<u8> {
    let mut s = String::new();

    let _ = writeln!(s, "nodev\tramfs");
    let _ = writeln!(s, "nodev\tprocfs");
    let _ = writeln!(s, "nodev\tsysfs");
    let _ = writeln!(s, "nodev\tdevfs");
    let _ = writeln!(s, "\tfat32");
    let _ = writeln!(s, "\text4");

    s.into_bytes()
}

/// Generate /proc/mounts
fn generate_mounts() -> Vec<u8> {
    let mut s = String::new();

    let _ = writeln!(s, "rootfs / ramfs rw 0 0");
    let _ = writeln!(s, "proc /proc procfs rw 0 0");
    let _ = writeln!(s, "sys /sys sysfs rw 0 0");
    let _ = writeln!(s, "dev /dev devfs rw 0 0");

    s.into_bytes()
}

/// Generate /proc/cmdline
fn generate_cmdline() -> Vec<u8> {
    b"console=ttyAMA0 root=/dev/ram0 rw init=/bin/init\n".to_vec()
}

/// Generate /proc/interrupts
fn generate_interrupts() -> Vec<u8> {
    let mut s = String::new();

    let _ = writeln!(s, "           CPU0");
    let _ = writeln!(s, "  0:          0   IO-APIC    timer");
    let _ = writeln!(s, "  1:          0   IO-APIC    keyboard");

    // Add IRQ counts
    for irq in 0..16 {
        let count = crate::arch::aarch64::interrupt::irq_count(irq);
        if count > 0 {
            let _ = writeln!(s, "{:3}:   {:8}   GIC    irq{}", irq, count, irq);
        }
    }

    s.into_bytes()
}

/// Generate /proc/[pid]/stat
fn generate_process_stat(proc: &crate::process::Process) -> Vec<u8> {
    // Format: pid (comm) state ppid pgrp session tty_nr tpgid flags
    //         minflt cminflt majflt cmajflt utime stime cutime cstime
    //         priority nice num_threads itrealvalue starttime vsize rss

    let state = match proc.state() {
        crate::process::ProcessState::Ready => 'R',
        crate::process::ProcessState::Running => 'R',
        crate::process::ProcessState::Blocked(_) => 'S',
        crate::process::ProcessState::Zombie => 'Z',
        crate::process::ProcessState::Sleeping(_) => 'S',
    };

    let memory = proc.memory.lock();

    format!(
        "{} ({}) {} {} {} {} 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 {} {}\n",
        proc.pid.0,
        proc.name,
        state,
        proc.ppid.0,
        proc.pid.0, // pgrp
        proc.pid.0, // session
        memory.heap_end - memory.heap_start,
        0 // rss
    ).into_bytes()
}

/// Generate /proc/[pid]/status
fn generate_process_status(proc: &crate::process::Process) -> Vec<u8> {
    let mut s = String::new();

    let state_str = match proc.state() {
        crate::process::ProcessState::Ready => "R (running)",
        crate::process::ProcessState::Running => "R (running)",
        crate::process::ProcessState::Blocked(_) => "S (sleeping)",
        crate::process::ProcessState::Zombie => "Z (zombie)",
        crate::process::ProcessState::Sleeping(_) => "S (sleeping)",
    };

    let memory = proc.memory.lock();

    let _ = writeln!(s, "Name:\t{}", proc.name);
    let _ = writeln!(s, "State:\t{}", state_str);
    let _ = writeln!(s, "Pid:\t{}", proc.pid.0);
    let _ = writeln!(s, "PPid:\t{}", proc.ppid.0);
    let _ = writeln!(s, "Uid:\t{}\t{}\t{}\t{}", proc.uid, proc.uid, proc.uid, proc.uid);
    let _ = writeln!(s, "Gid:\t{}\t{}\t{}\t{}", proc.gid, proc.gid, proc.gid, proc.gid);
    let _ = writeln!(s, "VmSize:\t{} kB", (memory.heap_end - memory.heap_start) / 1024);
    let _ = writeln!(s, "VmRSS:\t0 kB");
    let _ = writeln!(s, "Threads:\t1");

    s.into_bytes()
}

/// Generate /proc/[pid]/cmdline
fn generate_process_cmdline(proc: &crate::process::Process) -> Vec<u8> {
    // Command line arguments separated by NUL
    let mut cmdline = proc.name.as_bytes().to_vec();
    cmdline.push(0);
    cmdline
}

/// Generate /proc/[pid]/comm
fn generate_process_comm(proc: &crate::process::Process) -> Vec<u8> {
    format!("{}\n", proc.name).into_bytes()
}

/// Generate /proc/[pid]/environ
fn generate_process_environ(proc: &crate::process::Process) -> Vec<u8> {
    // Environment variables separated by NUL
    let mut environ = Vec::new();

    for (key, value) in &proc.env {
        environ.extend_from_slice(key.as_bytes());
        environ.push(b'=');
        environ.extend_from_slice(value.as_bytes());
        environ.push(0);
    }

    environ
}

/// Generate /proc/[pid]/maps
fn generate_process_maps(proc: &crate::process::Process) -> Vec<u8> {
    let memory = proc.memory.lock();

    let mut s = String::new();

    // Heap
    let _ = writeln!(
        s,
        "{:016x}-{:016x} rw-p 00000000 00:00 0          [heap]",
        memory.heap_start,
        memory.heap_end
    );

    // Stack
    let _ = writeln!(
        s,
        "{:016x}-{:016x} rw-p 00000000 00:00 0          [stack]",
        memory.stack_top - memory.stack_size,
        memory.stack_top
    );

    s.into_bytes()
}

/// Generate /proc/[pid]/exe
fn generate_process_exe(proc: &crate::process::Process) -> Vec<u8> {
    // Would be a symlink to the executable
    proc.name.as_bytes().to_vec()
}

/// Generate /proc/[pid]/fd listing
fn generate_process_fds(proc: &crate::process::Process) -> Vec<u8> {
    let mut s = String::new();

    // Standard fds
    let _ = writeln!(s, "0");
    let _ = writeln!(s, "1");
    let _ = writeln!(s, "2");

    // List open fds
    for fd in proc.list_fds() {
        let _ = writeln!(s, "{}", fd);
    }

    s.into_bytes()
}

/// List directory entries in procfs
pub fn list_dir(path: &str) -> Vec<String> {
    let path = path.trim_start_matches("/proc").trim_start_matches('/');

    if path.is_empty() {
        // Root of /proc
        let mut entries = vec![
            String::from("meminfo"),
            String::from("cpuinfo"),
            String::from("version"),
            String::from("uptime"),
            String::from("loadavg"),
            String::from("stat"),
            String::from("filesystems"),
            String::from("mounts"),
            String::from("cmdline"),
            String::from("interrupts"),
            String::from("self"),
        ];

        // Add process directories
        for proc in crate::process::list() {
            entries.push(format!("{}", proc.pid.0));
        }

        entries
    } else if path == "self" {
        // /proc/self contents
        vec![
            String::from("cmdline"),
            String::from("comm"),
            String::from("environ"),
            String::from("exe"),
            String::from("fd"),
            String::from("maps"),
            String::from("stat"),
            String::from("status"),
            String::from("cwd"),
        ]
    } else if let Ok(pid) = path.parse::<u64>() {
        // /proc/[pid] contents
        let processes = crate::process::list();
        if processes.iter().any(|p| p.pid.0 == pid) {
            vec![
                String::from("cmdline"),
                String::from("comm"),
                String::from("environ"),
                String::from("exe"),
                String::from("fd"),
                String::from("maps"),
                String::from("stat"),
                String::from("status"),
                String::from("cwd"),
            ]
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    }
}

/// Check if path exists in procfs
pub fn exists(path: &str) -> bool {
    read_entry(path).is_some() || !list_dir(path).is_empty()
}

/// Initialize procfs
pub fn init() {
    crate::kprintln!("  Procfs initialized");
}
