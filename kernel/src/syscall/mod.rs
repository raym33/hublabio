//! System Call Interface
//!
//! Provides the syscall interface for user-space programs.
//! ARM64 uses SVC instruction to trigger syscalls.

use crate::process::{self, Pid};
use crate::vfs;
use crate::ipc;

/// Syscall numbers
pub mod nr {
    // Process management
    pub const EXIT: usize = 0;
    pub const FORK: usize = 1;
    pub const EXEC: usize = 2;
    pub const WAIT: usize = 3;
    pub const GETPID: usize = 4;
    pub const GETPPID: usize = 5;
    pub const KILL: usize = 6;
    pub const YIELD: usize = 7;

    // File operations
    pub const OPEN: usize = 10;
    pub const CLOSE: usize = 11;
    pub const READ: usize = 12;
    pub const WRITE: usize = 13;
    pub const LSEEK: usize = 14;
    pub const STAT: usize = 15;
    pub const FSTAT: usize = 16;
    pub const MKDIR: usize = 17;
    pub const RMDIR: usize = 18;
    pub const UNLINK: usize = 19;
    pub const READDIR: usize = 20;
    pub const GETCWD: usize = 21;
    pub const CHDIR: usize = 22;

    // Memory management
    pub const BRK: usize = 30;
    pub const MMAP: usize = 31;
    pub const MUNMAP: usize = 32;
    pub const MPROTECT: usize = 33;

    // IPC
    pub const IPC_CREATE: usize = 40;
    pub const IPC_SEND: usize = 41;
    pub const IPC_RECV: usize = 42;
    pub const IPC_CLOSE: usize = 43;
    pub const IPC_LOOKUP: usize = 44;
    pub const IPC_REGISTER: usize = 45;

    // Time
    pub const TIME: usize = 50;
    pub const SLEEP: usize = 51;
    pub const NANOSLEEP: usize = 52;

    // System info
    pub const UNAME: usize = 60;
    pub const SYSINFO: usize = 61;

    // AI (HubLab IO specific)
    pub const AI_LOAD: usize = 100;
    pub const AI_GENERATE: usize = 101;
    pub const AI_TOKENIZE: usize = 102;
    pub const AI_UNLOAD: usize = 103;
}

/// Syscall error codes
pub mod errno {
    pub const SUCCESS: isize = 0;
    pub const EPERM: isize = -1;      // Operation not permitted
    pub const ENOENT: isize = -2;     // No such file or directory
    pub const ESRCH: isize = -3;      // No such process
    pub const EINTR: isize = -4;      // Interrupted system call
    pub const EIO: isize = -5;        // I/O error
    pub const ENOMEM: isize = -12;    // Out of memory
    pub const EACCES: isize = -13;    // Permission denied
    pub const EFAULT: isize = -14;    // Bad address
    pub const EBUSY: isize = -16;     // Device or resource busy
    pub const EEXIST: isize = -17;    // File exists
    pub const ENOTDIR: isize = -20;   // Not a directory
    pub const EISDIR: isize = -21;    // Is a directory
    pub const EINVAL: isize = -22;    // Invalid argument
    pub const EMFILE: isize = -24;    // Too many open files
    pub const ENOSPC: isize = -28;    // No space left on device
    pub const ENOSYS: isize = -38;    // Function not implemented
    pub const ECHILD: isize = -10;    // No child processes
}

/// Syscall result type
pub type SyscallResult = isize;

/// Initialize syscall interface
pub fn init() {
    // Set up exception vector for SVC
    // This is done in assembly in the exception handler
    crate::kprintln!("  Syscall interface ready");
}

/// Main syscall dispatcher
///
/// Called from the exception handler when SVC is executed.
/// Arguments are passed in x0-x5, syscall number in x8.
#[no_mangle]
pub extern "C" fn syscall_handler(
    nr: usize,
    arg0: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
    arg4: usize,
    arg5: usize,
) -> SyscallResult {
    match nr {
        // Process management
        nr::EXIT => sys_exit(arg0 as i32),
        nr::FORK => sys_fork(),
        nr::EXEC => sys_exec(arg0, arg1, arg2),
        nr::WAIT => sys_wait(arg0 as i32),
        nr::GETPID => sys_getpid(),
        nr::GETPPID => sys_getppid(),
        nr::KILL => sys_kill(arg0 as u64, arg1 as i32),
        nr::YIELD => sys_yield(),

        // File operations
        nr::OPEN => sys_open(arg0, arg1 as u32, arg2 as u32),
        nr::CLOSE => sys_close(arg0 as u32),
        nr::READ => sys_read(arg0 as u32, arg1, arg2),
        nr::WRITE => sys_write(arg0 as u32, arg1, arg2),
        nr::LSEEK => sys_lseek(arg0 as u32, arg1 as i64, arg2 as u32),
        nr::STAT => sys_stat(arg0, arg1),
        nr::GETCWD => sys_getcwd(arg0, arg1),
        nr::CHDIR => sys_chdir(arg0),

        // Memory management
        nr::BRK => sys_brk(arg0),
        nr::MMAP => sys_mmap(arg0, arg1, arg2 as u32, arg3 as u32, arg4 as u32, arg5),
        nr::MUNMAP => sys_munmap(arg0, arg1),

        // IPC
        nr::IPC_CREATE => sys_ipc_create(),
        nr::IPC_SEND => sys_ipc_send(arg0 as u64, arg1 as u32, arg2, arg3),
        nr::IPC_RECV => sys_ipc_recv(arg0 as u64, arg1, arg2),
        nr::IPC_CLOSE => sys_ipc_close(arg0 as u64),
        nr::IPC_LOOKUP => sys_ipc_lookup(arg0),
        nr::IPC_REGISTER => sys_ipc_register(arg0, arg1 as u64),

        // Time
        nr::TIME => sys_time(arg0),
        nr::SLEEP => sys_sleep(arg0 as u64),

        // System info
        nr::UNAME => sys_uname(arg0),
        nr::SYSINFO => sys_sysinfo(arg0),

        // AI
        nr::AI_LOAD => sys_ai_load(arg0, arg1),
        nr::AI_GENERATE => sys_ai_generate(arg0, arg1, arg2, arg3),

        _ => errno::ENOSYS,
    }
}

// ============================================================================
// Process Management Syscalls
// ============================================================================

fn sys_exit(code: i32) -> SyscallResult {
    if let Some(proc) = process::current() {
        proc.exit(code);
    }
    // Should not return
    errno::SUCCESS
}

fn sys_fork() -> SyscallResult {
    match process::fork() {
        Ok(pid) => pid.0 as isize,
        Err(_) => errno::ENOMEM,
    }
}

fn sys_exec(path_ptr: usize, argv_ptr: usize, envp_ptr: usize) -> SyscallResult {
    // TODO: Read path from user memory
    let _ = (path_ptr, argv_ptr, envp_ptr);
    errno::ENOSYS
}

fn sys_wait(pid: i32) -> SyscallResult {
    let target = if pid > 0 {
        Some(Pid(pid as u64))
    } else {
        None
    };

    match process::wait(target) {
        Ok((_, code)) => code as isize,
        Err(_) => errno::ECHILD,
    }
}

fn sys_getpid() -> SyscallResult {
    process::current()
        .map(|p| p.pid.0 as isize)
        .unwrap_or(errno::ESRCH)
}

fn sys_getppid() -> SyscallResult {
    process::current()
        .map(|p| p.ppid.0 as isize)
        .unwrap_or(errno::ESRCH)
}

fn sys_kill(pid: u64, signal: i32) -> SyscallResult {
    match process::kill(Pid(pid), signal) {
        Ok(()) => errno::SUCCESS,
        Err(_) => errno::ESRCH,
    }
}

fn sys_yield() -> SyscallResult {
    // TODO: Trigger reschedule
    errno::SUCCESS
}

// ============================================================================
// File Operation Syscalls
// ============================================================================

fn sys_open(path_ptr: usize, flags: u32, mode: u32) -> SyscallResult {
    let _ = (path_ptr, flags, mode);
    // TODO: Read path from user memory and open file
    errno::ENOSYS
}

fn sys_close(fd: u32) -> SyscallResult {
    if let Some(proc) = process::current() {
        if proc.close_fd(fd) {
            return errno::SUCCESS;
        }
    }
    errno::EINVAL
}

fn sys_read(fd: u32, buf_ptr: usize, count: usize) -> SyscallResult {
    let _ = (fd, buf_ptr, count);
    // TODO: Implement read
    errno::ENOSYS
}

fn sys_write(fd: u32, buf_ptr: usize, count: usize) -> SyscallResult {
    // Special case: stdout/stderr
    if fd == 1 || fd == 2 {
        // Read from user buffer and print
        // TODO: Proper user memory access
        let slice = unsafe {
            core::slice::from_raw_parts(buf_ptr as *const u8, count)
        };
        if let Ok(s) = core::str::from_utf8(slice) {
            crate::kprint!("{}", s);
            return count as isize;
        }
    }
    errno::EINVAL
}

fn sys_lseek(fd: u32, offset: i64, whence: u32) -> SyscallResult {
    let _ = (fd, offset, whence);
    errno::ENOSYS
}

fn sys_stat(path_ptr: usize, stat_ptr: usize) -> SyscallResult {
    let _ = (path_ptr, stat_ptr);
    errno::ENOSYS
}

fn sys_getcwd(buf_ptr: usize, size: usize) -> SyscallResult {
    if let Some(proc) = process::current() {
        let cwd = &proc.cwd;
        if cwd.len() + 1 > size {
            return errno::EINVAL;
        }
        // TODO: Proper user memory copy
        let buf = unsafe {
            core::slice::from_raw_parts_mut(buf_ptr as *mut u8, size)
        };
        buf[..cwd.len()].copy_from_slice(cwd.as_bytes());
        buf[cwd.len()] = 0;
        return buf_ptr as isize;
    }
    errno::ESRCH
}

fn sys_chdir(path_ptr: usize) -> SyscallResult {
    let _ = path_ptr;
    // TODO: Implement chdir
    errno::ENOSYS
}

// ============================================================================
// Memory Management Syscalls
// ============================================================================

fn sys_brk(new_brk: usize) -> SyscallResult {
    if let Some(proc) = process::current() {
        let mut memory = proc.memory.lock();

        if new_brk == 0 {
            // Query current brk
            return memory.heap_end as isize;
        }

        if new_brk >= memory.heap_start && new_brk < memory.stack_top - memory.stack_size {
            memory.heap_end = new_brk;
            return new_brk as isize;
        }

        return errno::ENOMEM;
    }
    errno::ESRCH
}

fn sys_mmap(
    addr: usize,
    length: usize,
    prot: u32,
    flags: u32,
    fd: u32,
    offset: usize,
) -> SyscallResult {
    let _ = (addr, length, prot, flags, fd, offset);
    // TODO: Implement mmap
    errno::ENOSYS
}

fn sys_munmap(addr: usize, length: usize) -> SyscallResult {
    let _ = (addr, length);
    // TODO: Implement munmap
    errno::ENOSYS
}

// ============================================================================
// IPC Syscalls
// ============================================================================

fn sys_ipc_create() -> SyscallResult {
    let (endpoint_a, _endpoint_b) = ipc::create_channel();
    endpoint_a.channel_id().0 as isize
}

fn sys_ipc_send(channel_id: u64, msg_type: u32, data_ptr: usize, len: usize) -> SyscallResult {
    let _ = (channel_id, msg_type, data_ptr, len);
    // TODO: Implement send
    errno::ENOSYS
}

fn sys_ipc_recv(channel_id: u64, buf_ptr: usize, buf_len: usize) -> SyscallResult {
    let _ = (channel_id, buf_ptr, buf_len);
    // TODO: Implement recv
    errno::ENOSYS
}

fn sys_ipc_close(channel_id: u64) -> SyscallResult {
    let _ = channel_id;
    errno::SUCCESS
}

fn sys_ipc_lookup(name_ptr: usize) -> SyscallResult {
    let _ = name_ptr;
    // TODO: Read name from user memory
    errno::ENOSYS
}

fn sys_ipc_register(name_ptr: usize, channel_id: u64) -> SyscallResult {
    let _ = (name_ptr, channel_id);
    // TODO: Read name from user memory
    errno::ENOSYS
}

// ============================================================================
// Time Syscalls
// ============================================================================

fn sys_time(time_ptr: usize) -> SyscallResult {
    // TODO: Get real time
    if time_ptr != 0 {
        unsafe {
            *(time_ptr as *mut u64) = 0;
        }
    }
    errno::SUCCESS
}

fn sys_sleep(seconds: u64) -> SyscallResult {
    let _ = seconds;
    // TODO: Implement sleep
    errno::SUCCESS
}

// ============================================================================
// System Info Syscalls
// ============================================================================

/// utsname structure
#[repr(C)]
pub struct Utsname {
    pub sysname: [u8; 65],
    pub nodename: [u8; 65],
    pub release: [u8; 65],
    pub version: [u8; 65],
    pub machine: [u8; 65],
}

fn sys_uname(buf_ptr: usize) -> SyscallResult {
    let buf = unsafe { &mut *(buf_ptr as *mut Utsname) };

    fn copy_str(dst: &mut [u8; 65], src: &str) {
        let bytes = src.as_bytes();
        let len = bytes.len().min(64);
        dst[..len].copy_from_slice(&bytes[..len]);
        dst[len] = 0;
    }

    copy_str(&mut buf.sysname, "HubLabIO");
    copy_str(&mut buf.nodename, "hublab");
    copy_str(&mut buf.release, crate::VERSION);
    copy_str(&mut buf.version, "HubLab IO AI-Native OS");
    copy_str(&mut buf.machine, "aarch64");

    errno::SUCCESS
}

/// sysinfo structure
#[repr(C)]
pub struct Sysinfo {
    pub uptime: u64,
    pub totalram: u64,
    pub freeram: u64,
    pub procs: u32,
}

fn sys_sysinfo(buf_ptr: usize) -> SyscallResult {
    let buf = unsafe { &mut *(buf_ptr as *mut Sysinfo) };

    let mem_stats = crate::memory::stats();

    buf.uptime = 0; // TODO: Track uptime
    buf.totalram = mem_stats.total as u64;
    buf.freeram = mem_stats.free as u64;
    buf.procs = process::list().len() as u32;

    errno::SUCCESS
}

// ============================================================================
// AI Syscalls (HubLab IO Specific)
// ============================================================================

fn sys_ai_load(path_ptr: usize, path_len: usize) -> SyscallResult {
    let _ = (path_ptr, path_len);
    // TODO: Load AI model
    errno::ENOSYS
}

fn sys_ai_generate(prompt_ptr: usize, prompt_len: usize, out_ptr: usize, out_len: usize) -> SyscallResult {
    let _ = (prompt_ptr, prompt_len, out_ptr, out_len);
    // TODO: Generate text with AI
    errno::ENOSYS
}
