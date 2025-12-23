//! System Call Interface
//!
//! Provides the syscall interface for user-space programs.
//! ARM64 uses SVC instruction to trigger syscalls.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;

use crate::process::{self, Pid};
use crate::vfs;
use crate::ipc;
use crate::signal::{Signal, SignalAction};

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
    pub const CLONE: usize = 8;
    pub const GETUID: usize = 9;

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
    pub const DUP: usize = 23;
    pub const DUP2: usize = 24;
    pub const PIPE: usize = 25;
    pub const FCNTL: usize = 26;
    pub const IOCTL: usize = 27;

    // Memory management
    pub const BRK: usize = 30;
    pub const MMAP: usize = 31;
    pub const MUNMAP: usize = 32;
    pub const MPROTECT: usize = 33;
    pub const SHMGET: usize = 34;
    pub const SHMAT: usize = 35;
    pub const SHMDT: usize = 36;
    pub const SHMCTL: usize = 37;

    // IPC
    pub const IPC_CREATE: usize = 40;
    pub const IPC_SEND: usize = 41;
    pub const IPC_RECV: usize = 42;
    pub const IPC_CLOSE: usize = 43;
    pub const IPC_LOOKUP: usize = 44;
    pub const IPC_REGISTER: usize = 45;

    // Signals
    pub const SIGACTION: usize = 46;
    pub const SIGPROCMASK: usize = 47;
    pub const SIGSUSPEND: usize = 48;
    pub const SIGRETURN: usize = 49;

    // Time
    pub const TIME: usize = 50;
    pub const SLEEP: usize = 51;
    pub const NANOSLEEP: usize = 52;
    pub const GETTIMEOFDAY: usize = 53;
    pub const CLOCK_GETTIME: usize = 54;

    // System info
    pub const UNAME: usize = 60;
    pub const SYSINFO: usize = 61;
    pub const GETRLIMIT: usize = 62;
    pub const SETRLIMIT: usize = 63;

    // Synchronization
    pub const FUTEX: usize = 70;
    pub const SEM_INIT: usize = 71;
    pub const SEM_WAIT: usize = 72;
    pub const SEM_POST: usize = 73;
    pub const SEM_DESTROY: usize = 74;

    // Networking
    pub const SOCKET: usize = 80;
    pub const BIND: usize = 81;
    pub const LISTEN: usize = 82;
    pub const ACCEPT: usize = 83;
    pub const CONNECT: usize = 84;
    pub const SEND: usize = 85;
    pub const RECV: usize = 86;
    pub const SENDTO: usize = 87;
    pub const RECVFROM: usize = 88;
    pub const SETSOCKOPT: usize = 89;
    pub const GETSOCKOPT: usize = 90;

    // Scheduling / CPU affinity
    pub const SCHED_SETAFFINITY: usize = 95;
    pub const SCHED_GETAFFINITY: usize = 96;
    pub const SCHED_YIELD: usize = 97;
    pub const SCHED_GET_PRIORITY_MAX: usize = 98;
    pub const SCHED_GET_PRIORITY_MIN: usize = 99;

    // AI (HubLab IO specific)
    pub const AI_LOAD: usize = 100;
    pub const AI_GENERATE: usize = 101;
    pub const AI_TOKENIZE: usize = 102;
    pub const AI_UNLOAD: usize = 103;
    pub const AI_EMBED: usize = 104;

    // SMP specific
    pub const GETCPU: usize = 110;
    pub const SMP_INFO: usize = 111;
}

/// Syscall error codes (POSIX-compatible)
pub mod errno {
    pub const SUCCESS: isize = 0;
    pub const EPERM: isize = -1;      // Operation not permitted
    pub const ENOENT: isize = -2;     // No such file or directory
    pub const ESRCH: isize = -3;      // No such process
    pub const EINTR: isize = -4;      // Interrupted system call
    pub const EIO: isize = -5;        // I/O error
    pub const ENXIO: isize = -6;      // No such device or address
    pub const E2BIG: isize = -7;      // Argument list too long
    pub const ENOEXEC: isize = -8;    // Exec format error
    pub const EBADF: isize = -9;      // Bad file descriptor
    pub const ECHILD: isize = -10;    // No child processes
    pub const EAGAIN: isize = -11;    // Try again (EWOULDBLOCK)
    pub const ENOMEM: isize = -12;    // Out of memory
    pub const EACCES: isize = -13;    // Permission denied
    pub const EFAULT: isize = -14;    // Bad address
    pub const EBUSY: isize = -16;     // Device or resource busy
    pub const EEXIST: isize = -17;    // File exists
    pub const EXDEV: isize = -18;     // Cross-device link
    pub const ENODEV: isize = -19;    // No such device
    pub const ENOTDIR: isize = -20;   // Not a directory
    pub const EISDIR: isize = -21;    // Is a directory
    pub const EINVAL: isize = -22;    // Invalid argument
    pub const ENFILE: isize = -23;    // File table overflow
    pub const EMFILE: isize = -24;    // Too many open files
    pub const ENOTTY: isize = -25;    // Not a typewriter
    pub const ETXTBSY: isize = -26;   // Text file busy
    pub const EFBIG: isize = -27;     // File too large
    pub const ENOSPC: isize = -28;    // No space left on device
    pub const ESPIPE: isize = -29;    // Illegal seek
    pub const EROFS: isize = -30;     // Read-only file system
    pub const EMLINK: isize = -31;    // Too many links
    pub const EPIPE: isize = -32;     // Broken pipe
    pub const EDOM: isize = -33;      // Math argument out of domain
    pub const ERANGE: isize = -34;    // Math result not representable
    pub const EDEADLK: isize = -35;   // Resource deadlock would occur
    pub const ENAMETOOLONG: isize = -36; // File name too long
    pub const ENOLCK: isize = -37;    // No record locks available
    pub const ENOSYS: isize = -38;    // Function not implemented
    pub const ENOTEMPTY: isize = -39; // Directory not empty
    pub const ELOOP: isize = -40;     // Too many symbolic links
    pub const ETIMEDOUT: isize = -110; // Connection timed out
    pub const ECONNREFUSED: isize = -111; // Connection refused
}

/// Open flags
pub mod open_flags {
    pub const O_RDONLY: u32 = 0;
    pub const O_WRONLY: u32 = 1;
    pub const O_RDWR: u32 = 2;
    pub const O_CREAT: u32 = 0o100;
    pub const O_EXCL: u32 = 0o200;
    pub const O_TRUNC: u32 = 0o1000;
    pub const O_APPEND: u32 = 0o2000;
    pub const O_NONBLOCK: u32 = 0o4000;
    pub const O_DIRECTORY: u32 = 0o200000;
    pub const O_CLOEXEC: u32 = 0o2000000;
}

/// Seek whence
pub mod seek {
    pub const SEEK_SET: u32 = 0;
    pub const SEEK_CUR: u32 = 1;
    pub const SEEK_END: u32 = 2;
}

/// mmap protection flags
pub mod prot {
    pub const PROT_NONE: u32 = 0;
    pub const PROT_READ: u32 = 1;
    pub const PROT_WRITE: u32 = 2;
    pub const PROT_EXEC: u32 = 4;
}

/// mmap flags
pub mod mmap_flags {
    pub const MAP_SHARED: u32 = 0x01;
    pub const MAP_PRIVATE: u32 = 0x02;
    pub const MAP_FIXED: u32 = 0x10;
    pub const MAP_ANONYMOUS: u32 = 0x20;
}

/// Syscall result type
pub type SyscallResult = isize;

/// System uptime in ticks
static UPTIME_TICKS: AtomicU64 = AtomicU64::new(0);

/// Initialize syscall interface
pub fn init() {
    crate::kprintln!("  Syscall interface ready (100+ syscalls)");
}

/// Increment uptime (called by timer)
pub fn tick() {
    UPTIME_TICKS.fetch_add(1, Ordering::Relaxed);
}

/// Get uptime in seconds
pub fn uptime_secs() -> u64 {
    // Assuming 100 ticks per second
    UPTIME_TICKS.load(Ordering::Relaxed) / 100
}

/// Read a null-terminated string from user memory
unsafe fn read_user_string(ptr: usize, max_len: usize) -> Result<String, isize> {
    if ptr == 0 {
        return Err(errno::EFAULT);
    }

    let mut s = String::new();
    let mut p = ptr as *const u8;

    for _ in 0..max_len {
        let c = *p;
        if c == 0 {
            break;
        }
        s.push(c as char);
        p = p.add(1);
    }

    Ok(s)
}

/// Copy data from user memory
unsafe fn copy_from_user(dst: &mut [u8], src: usize, len: usize) -> Result<(), isize> {
    if src == 0 {
        return Err(errno::EFAULT);
    }

    let src_slice = core::slice::from_raw_parts(src as *const u8, len);
    dst[..len].copy_from_slice(src_slice);
    Ok(())
}

/// Copy data to user memory
unsafe fn copy_to_user(dst: usize, src: &[u8]) -> Result<(), isize> {
    if dst == 0 {
        return Err(errno::EFAULT);
    }

    let dst_slice = core::slice::from_raw_parts_mut(dst as *mut u8, src.len());
    dst_slice.copy_from_slice(src);
    Ok(())
}

/// Main syscall dispatcher
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
        nr::WAIT => sys_wait(arg0 as i32, arg1),
        nr::GETPID => sys_getpid(),
        nr::GETPPID => sys_getppid(),
        nr::KILL => sys_kill(arg0 as u64, arg1 as i32),
        nr::YIELD => sys_yield(),
        nr::CLONE => sys_clone(arg0, arg1, arg2, arg3, arg4),
        nr::GETUID => sys_getuid(),

        // File operations
        nr::OPEN => sys_open(arg0, arg1 as u32, arg2 as u32),
        nr::CLOSE => sys_close(arg0 as u32),
        nr::READ => sys_read(arg0 as u32, arg1, arg2),
        nr::WRITE => sys_write(arg0 as u32, arg1, arg2),
        nr::LSEEK => sys_lseek(arg0 as u32, arg1 as i64, arg2 as u32),
        nr::STAT => sys_stat(arg0, arg1),
        nr::FSTAT => sys_fstat(arg0 as u32, arg1),
        nr::MKDIR => sys_mkdir(arg0, arg1 as u32),
        nr::RMDIR => sys_rmdir(arg0),
        nr::UNLINK => sys_unlink(arg0),
        nr::READDIR => sys_readdir(arg0 as u32, arg1, arg2),
        nr::GETCWD => sys_getcwd(arg0, arg1),
        nr::CHDIR => sys_chdir(arg0),
        nr::DUP => sys_dup(arg0 as u32),
        nr::DUP2 => sys_dup2(arg0 as u32, arg1 as u32),
        nr::PIPE => sys_pipe(arg0),
        nr::FCNTL => sys_fcntl(arg0 as u32, arg1 as u32, arg2),
        nr::IOCTL => sys_ioctl(arg0 as u32, arg1 as u32, arg2),

        // Memory management
        nr::BRK => sys_brk(arg0),
        nr::MMAP => sys_mmap(arg0, arg1, arg2 as u32, arg3 as u32, arg4 as i32, arg5),
        nr::MUNMAP => sys_munmap(arg0, arg1),
        nr::MPROTECT => sys_mprotect(arg0, arg1, arg2 as u32),
        nr::SHMGET => sys_shmget(arg0 as i32, arg1, arg2 as i32),
        nr::SHMAT => sys_shmat(arg0 as i32, arg1, arg2 as i32),
        nr::SHMDT => sys_shmdt(arg0),
        nr::SHMCTL => sys_shmctl(arg0 as i32, arg1 as i32, arg2),

        // IPC
        nr::IPC_CREATE => sys_ipc_create(),
        nr::IPC_SEND => sys_ipc_send(arg0 as u64, arg1 as u32, arg2, arg3),
        nr::IPC_RECV => sys_ipc_recv(arg0 as u64, arg1, arg2),
        nr::IPC_CLOSE => sys_ipc_close(arg0 as u64),
        nr::IPC_LOOKUP => sys_ipc_lookup(arg0),
        nr::IPC_REGISTER => sys_ipc_register(arg0, arg1 as u64),

        // Signals
        nr::SIGACTION => sys_sigaction(arg0 as i32, arg1, arg2),
        nr::SIGPROCMASK => sys_sigprocmask(arg0 as i32, arg1, arg2),
        nr::SIGSUSPEND => sys_sigsuspend(arg0),
        nr::SIGRETURN => sys_sigreturn(),

        // Time
        nr::TIME => sys_time(arg0),
        nr::SLEEP => sys_sleep(arg0 as u64),
        nr::NANOSLEEP => sys_nanosleep(arg0, arg1),
        nr::GETTIMEOFDAY => sys_gettimeofday(arg0, arg1),
        nr::CLOCK_GETTIME => sys_clock_gettime(arg0 as i32, arg1),

        // System info
        nr::UNAME => sys_uname(arg0),
        nr::SYSINFO => sys_sysinfo(arg0),
        nr::GETRLIMIT => sys_getrlimit(arg0 as i32, arg1),
        nr::SETRLIMIT => sys_setrlimit(arg0 as i32, arg1),

        // Synchronization
        nr::FUTEX => sys_futex(arg0, arg1 as i32, arg2 as u32, arg3, arg4, arg5 as u32),
        nr::SEM_INIT => sys_sem_init(arg0, arg1 as i32, arg2 as u32),
        nr::SEM_WAIT => sys_sem_wait(arg0),
        nr::SEM_POST => sys_sem_post(arg0),
        nr::SEM_DESTROY => sys_sem_destroy(arg0),

        // Networking
        nr::SOCKET => sys_socket(arg0 as i32, arg1 as i32, arg2 as i32),
        nr::BIND => sys_bind(arg0 as i32, arg1, arg2 as u32),
        nr::LISTEN => sys_listen(arg0 as i32, arg1 as i32),
        nr::ACCEPT => sys_accept(arg0 as i32, arg1, arg2),
        nr::CONNECT => sys_connect(arg0 as i32, arg1, arg2 as u32),
        nr::SEND => sys_send(arg0 as i32, arg1, arg2, arg3 as i32),
        nr::RECV => sys_recv(arg0 as i32, arg1, arg2, arg3 as i32),
        nr::SENDTO => sys_sendto(arg0 as i32, arg1, arg2, arg3 as i32, arg4, arg5 as u32),
        nr::RECVFROM => sys_recvfrom(arg0 as i32, arg1, arg2, arg3 as i32, arg4, arg5),

        // AI
        nr::AI_LOAD => sys_ai_load(arg0, arg1),
        nr::AI_GENERATE => sys_ai_generate(arg0, arg1, arg2, arg3, arg4),
        nr::AI_TOKENIZE => sys_ai_tokenize(arg0, arg1, arg2, arg3),
        nr::AI_UNLOAD => sys_ai_unload(arg0 as u32),
        nr::AI_EMBED => sys_ai_embed(arg0, arg1, arg2, arg3),

        // Scheduling / CPU affinity
        nr::SCHED_SETAFFINITY => sys_sched_setaffinity(arg0 as u32, arg1, arg2),
        nr::SCHED_GETAFFINITY => sys_sched_getaffinity(arg0 as u32, arg1, arg2),
        nr::SCHED_YIELD => sys_sched_yield(),
        nr::SCHED_GET_PRIORITY_MAX => sys_sched_get_priority_max(arg0 as i32),
        nr::SCHED_GET_PRIORITY_MIN => sys_sched_get_priority_min(arg0 as i32),

        // SMP specific
        nr::GETCPU => sys_getcpu(arg0, arg1, arg2),
        nr::SMP_INFO => sys_smp_info(arg0),

        _ => {
            crate::kwarn!("Unknown syscall: {}", nr);
            errno::ENOSYS
        }
    }
}

// ============================================================================
// Process Management Syscalls
// ============================================================================

fn sys_exit(code: i32) -> SyscallResult {
    if let Some(proc) = process::current() {
        crate::kinfo!("Process {} exiting with code {}", proc.pid.0, code);
        proc.exit(code);
    }
    errno::SUCCESS
}

fn sys_fork() -> SyscallResult {
    match process::fork() {
        Ok(pid) => pid.0 as isize,
        Err(_) => errno::ENOMEM,
    }
}

fn sys_exec(path_ptr: usize, argv_ptr: usize, envp_ptr: usize) -> SyscallResult {
    // Read path from user memory
    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    // Read arguments
    let mut args: Vec<String> = Vec::new();
    if argv_ptr != 0 {
        unsafe {
            let mut argv = argv_ptr as *const usize;
            while *argv != 0 {
                if let Ok(arg) = read_user_string(*argv, 256) {
                    args.push(arg);
                }
                argv = argv.add(1);
            }
        }
    }

    // Read environment
    let mut envs: Vec<String> = Vec::new();
    if envp_ptr != 0 {
        unsafe {
            let mut envp = envp_ptr as *const usize;
            while *envp != 0 {
                if let Ok(env) = read_user_string(*envp, 256) {
                    envs.push(env);
                }
                envp = envp.add(1);
            }
        }
    }

    // Execute the program
    match crate::exec::exec_program(&path, &args, &envs) {
        Ok(()) => errno::SUCCESS,
        Err(_) => errno::ENOEXEC,
    }
}

fn sys_wait(pid: i32, status_ptr: usize) -> SyscallResult {
    let target = if pid > 0 {
        Some(Pid(pid as u64))
    } else if pid == -1 {
        None // Wait for any child
    } else {
        return errno::EINVAL;
    };

    match process::wait(target) {
        Ok((child_pid, exit_code)) => {
            if status_ptr != 0 {
                // Store exit status: (exit_code << 8)
                let status = (exit_code as u32) << 8;
                unsafe {
                    *(status_ptr as *mut u32) = status;
                }
            }
            child_pid.0 as isize
        }
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
    if let Ok(sig) = Signal::try_from(signal as u8) {
        match crate::signal::send(Pid(pid), sig) {
            Ok(()) => errno::SUCCESS,
            Err(_) => errno::ESRCH,
        }
    } else {
        errno::EINVAL
    }
}

fn sys_yield() -> SyscallResult {
    crate::scheduler::schedule();
    errno::SUCCESS
}

fn sys_clone(flags: usize, stack: usize, ptid: usize, tls: usize, ctid: usize) -> SyscallResult {
    let _ = (flags, stack, ptid, tls, ctid);
    // Clone is like fork but with more control
    // For now, just do a regular fork
    sys_fork()
}

fn sys_getuid() -> SyscallResult {
    process::current()
        .map(|p| p.uid as isize)
        .unwrap_or(0)
}

// ============================================================================
// File Operation Syscalls
// ============================================================================

fn sys_open(path_ptr: usize, flags: u32, mode: u32) -> SyscallResult {
    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    // Open file through VFS
    match vfs::open(&path, flags, mode) {
        Ok(fd) => fd as isize,
        Err(_) => errno::ENOENT,
    }
}

fn sys_close(fd: u32) -> SyscallResult {
    if let Some(proc) = process::current() {
        if proc.close_fd(fd) {
            return errno::SUCCESS;
        }
    }
    errno::EBADF
}

fn sys_read(fd: u32, buf_ptr: usize, count: usize) -> SyscallResult {
    if buf_ptr == 0 || count == 0 {
        return errno::EINVAL;
    }

    // Handle stdin (fd 0)
    if fd == 0 {
        // Read from TTY
        let mut buf = vec![0u8; count];
        match crate::tty::read(0, &mut buf) {
            Ok(n) => {
                unsafe {
                    if copy_to_user(buf_ptr, &buf[..n]).is_err() {
                        return errno::EFAULT;
                    }
                }
                n as isize
            }
            Err(_) => errno::EIO,
        }
    } else {
        // Read from file
        let mut buf = vec![0u8; count];
        match vfs::read(fd, &mut buf) {
            Ok(n) => {
                unsafe {
                    if copy_to_user(buf_ptr, &buf[..n]).is_err() {
                        return errno::EFAULT;
                    }
                }
                n as isize
            }
            Err(_) => errno::EIO,
        }
    }
}

fn sys_write(fd: u32, buf_ptr: usize, count: usize) -> SyscallResult {
    if buf_ptr == 0 || count == 0 {
        return 0;
    }

    let mut buf = vec![0u8; count];
    unsafe {
        if copy_from_user(&mut buf, buf_ptr, count).is_err() {
            return errno::EFAULT;
        }
    }

    // Handle stdout/stderr (fd 1, 2)
    if fd == 1 || fd == 2 {
        if let Ok(s) = core::str::from_utf8(&buf) {
            crate::kprint!("{}", s);
            return count as isize;
        }
        // Write raw bytes
        for &b in &buf {
            crate::console::putchar(b);
        }
        return count as isize;
    }

    // Write to file
    match vfs::write(fd, &buf) {
        Ok(n) => n as isize,
        Err(_) => errno::EIO,
    }
}

fn sys_lseek(fd: u32, offset: i64, whence: u32) -> SyscallResult {
    match vfs::seek(fd, offset, whence) {
        Ok(pos) => pos as isize,
        Err(_) => errno::EINVAL,
    }
}

fn sys_stat(path_ptr: usize, stat_ptr: usize) -> SyscallResult {
    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    match vfs::stat(&path) {
        Ok(stat) => {
            unsafe {
                *(stat_ptr as *mut vfs::Stat) = stat;
            }
            errno::SUCCESS
        }
        Err(_) => errno::ENOENT,
    }
}

fn sys_fstat(fd: u32, stat_ptr: usize) -> SyscallResult {
    match vfs::fstat(fd) {
        Ok(stat) => {
            unsafe {
                *(stat_ptr as *mut vfs::Stat) = stat;
            }
            errno::SUCCESS
        }
        Err(_) => errno::EBADF,
    }
}

fn sys_mkdir(path_ptr: usize, mode: u32) -> SyscallResult {
    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    match vfs::mkdir(&path, mode) {
        Ok(()) => errno::SUCCESS,
        Err(_) => errno::EEXIST,
    }
}

fn sys_rmdir(path_ptr: usize) -> SyscallResult {
    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    match vfs::rmdir(&path) {
        Ok(()) => errno::SUCCESS,
        Err(_) => errno::ENOENT,
    }
}

fn sys_unlink(path_ptr: usize) -> SyscallResult {
    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    match vfs::unlink(&path) {
        Ok(()) => errno::SUCCESS,
        Err(_) => errno::ENOENT,
    }
}

fn sys_readdir(fd: u32, dirent_ptr: usize, count: usize) -> SyscallResult {
    match vfs::readdir(fd, dirent_ptr, count) {
        Ok(n) => n as isize,
        Err(_) => errno::EBADF,
    }
}

fn sys_getcwd(buf_ptr: usize, size: usize) -> SyscallResult {
    if let Some(proc) = process::current() {
        let cwd = &proc.cwd;
        if cwd.len() + 1 > size {
            return errno::ERANGE;
        }

        let mut buf = vec![0u8; cwd.len() + 1];
        buf[..cwd.len()].copy_from_slice(cwd.as_bytes());
        buf[cwd.len()] = 0;

        unsafe {
            if copy_to_user(buf_ptr, &buf).is_err() {
                return errno::EFAULT;
            }
        }
        return buf_ptr as isize;
    }
    errno::ESRCH
}

fn sys_chdir(path_ptr: usize) -> SyscallResult {
    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    // Verify path exists and is a directory
    match vfs::stat(&path) {
        Ok(stat) => {
            if stat.is_dir() {
                if let Some(proc) = process::current() {
                    proc.set_cwd(path);
                    return errno::SUCCESS;
                }
            }
            errno::ENOTDIR
        }
        Err(_) => errno::ENOENT,
    }
}

fn sys_dup(oldfd: u32) -> SyscallResult {
    if let Some(proc) = process::current() {
        match proc.dup_fd(oldfd) {
            Some(newfd) => newfd as isize,
            None => errno::EBADF,
        }
    } else {
        errno::ESRCH
    }
}

fn sys_dup2(oldfd: u32, newfd: u32) -> SyscallResult {
    if let Some(proc) = process::current() {
        match proc.dup2_fd(oldfd, newfd) {
            Ok(()) => newfd as isize,
            Err(_) => errno::EBADF,
        }
    } else {
        errno::ESRCH
    }
}

fn sys_pipe(pipefd_ptr: usize) -> SyscallResult {
    match crate::pipe::create_pipe() {
        Ok((read_fd, write_fd)) => {
            unsafe {
                let fds = pipefd_ptr as *mut [i32; 2];
                (*fds)[0] = read_fd as i32;
                (*fds)[1] = write_fd as i32;
            }
            errno::SUCCESS
        }
        Err(_) => errno::EMFILE,
    }
}

fn sys_fcntl(fd: u32, cmd: u32, arg: usize) -> SyscallResult {
    // F_DUPFD = 0, F_GETFD = 1, F_SETFD = 2, F_GETFL = 3, F_SETFL = 4
    match cmd {
        0 => sys_dup(fd), // F_DUPFD
        1 => errno::SUCCESS, // F_GETFD - return 0 (no close-on-exec)
        2 => errno::SUCCESS, // F_SETFD - ignore
        3 => errno::SUCCESS, // F_GETFL - return 0 (O_RDONLY)
        4 => errno::SUCCESS, // F_SETFL - ignore
        _ => {
            let _ = arg;
            errno::EINVAL
        }
    }
}

fn sys_ioctl(fd: u32, request: u32, arg: usize) -> SyscallResult {
    // Handle TTY ioctls
    if fd <= 2 {
        return crate::tty::ioctl(0, request, arg)
            .map(|v| v as isize)
            .unwrap_or(errno::ENOTTY);
    }

    errno::ENOTTY
}

// ============================================================================
// Memory Management Syscalls
// ============================================================================

fn sys_brk(new_brk: usize) -> SyscallResult {
    if let Some(proc) = process::current() {
        let mut memory = proc.memory.lock();

        if new_brk == 0 {
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

/// Memory mapping structure
#[derive(Clone)]
pub struct MmapRegion {
    pub start: usize,
    pub length: usize,
    pub prot: u32,
    pub flags: u32,
    pub fd: i32,
    pub offset: usize,
}

/// Global mmap regions (simplified implementation)
static MMAP_REGIONS: Mutex<Vec<MmapRegion>> = Mutex::new(Vec::new());
static MMAP_BASE: AtomicU64 = AtomicU64::new(0x4000_0000_0000); // 64TB

fn sys_mmap(
    addr: usize,
    length: usize,
    prot: u32,
    flags: u32,
    fd: i32,
    offset: usize,
) -> SyscallResult {
    if length == 0 {
        return errno::EINVAL;
    }

    // Round up to page size
    let page_size = crate::memory::PAGE_SIZE;
    let length = (length + page_size - 1) & !(page_size - 1);

    // Determine address
    let map_addr = if flags & mmap_flags::MAP_FIXED != 0 {
        if addr == 0 {
            return errno::EINVAL;
        }
        addr
    } else if addr != 0 {
        addr
    } else {
        // Allocate new address
        MMAP_BASE.fetch_add(length as u64, Ordering::SeqCst) as usize
    };

    // Get current process
    let proc = match process::current() {
        Some(p) => p,
        None => return errno::ESRCH,
    };

    // Calculate page flags
    let mut page_flags = crate::memory::PageFlags::PRESENT | crate::memory::PageFlags::USER;
    if prot & prot::PROT_WRITE != 0 {
        page_flags |= crate::memory::PageFlags::WRITABLE;
    }
    if prot & prot::PROT_EXEC == 0 {
        page_flags |= crate::memory::PageFlags::NO_EXECUTE;
    }

    // Get page table address
    let page_table = {
        let memory = proc.memory.lock();
        memory.page_table as usize
    };

    if page_table == 0 {
        return errno::ENOMEM;
    }

    // Allocate and map pages
    let num_pages = length / page_size;
    for i in 0..num_pages {
        let vaddr = map_addr + i * page_size;

        // Allocate physical frame
        let frame = match crate::memory::allocate_frame() {
            Some(f) => f,
            None => return errno::ENOMEM,
        };

        // Map the page
        mmap_page(page_table, vaddr, frame.start_address(), page_flags);

        // Zero the page for anonymous mappings
        if flags & mmap_flags::MAP_ANONYMOUS != 0 {
            unsafe {
                core::ptr::write_bytes(frame.start_address() as *mut u8, 0, page_size);
            }
        }
    }

    // For file-backed mapping, read content
    if fd >= 0 && flags & mmap_flags::MAP_ANONYMOUS == 0 {
        // Read file content into mapped pages
        let mut buf = vec![0u8; length];
        if vfs::seek(fd as u32, offset as i64, seek::SEEK_SET).is_ok() {
            if let Ok(n) = vfs::read(fd as u32, &mut buf) {
                // Copy to mapped memory
                // In a real implementation, we'd write to the physical pages
                // through their virtual addresses
                crate::kdebug!("mmap: Read {} bytes from fd {} at offset {}", n, fd, offset);
            }
        }
    }

    // Add to process memory regions
    {
        let mut memory = proc.memory.lock();
        let mut mem_flags = crate::process::MemoryFlags::empty();
        if prot & prot::PROT_READ != 0 {
            mem_flags |= crate::process::MemoryFlags::READ;
        }
        if prot & prot::PROT_WRITE != 0 {
            mem_flags |= crate::process::MemoryFlags::WRITE;
        }
        if prot & prot::PROT_EXEC != 0 {
            mem_flags |= crate::process::MemoryFlags::EXEC;
        }
        mem_flags |= crate::process::MemoryFlags::USER;

        memory.regions.push(crate::process::MemoryRegion {
            start: map_addr,
            end: map_addr + length,
            flags: mem_flags,
            name: if fd >= 0 {
                alloc::string::String::from("[file]")
            } else {
                alloc::string::String::from("[anon]")
            },
        });
    }

    // Store region info for munmap
    let region = MmapRegion {
        start: map_addr,
        length,
        prot,
        flags,
        fd,
        offset,
    };
    MMAP_REGIONS.lock().push(region);

    map_addr as isize
}

/// Map a single page in the address space
fn mmap_page(page_table: usize, vaddr: usize, paddr: usize, flags: crate::memory::PageFlags) {
    let l0 = page_table as *mut u64;

    let l0_idx = (vaddr >> 39) & 0x1FF;
    let l1_idx = (vaddr >> 30) & 0x1FF;
    let l2_idx = (vaddr >> 21) & 0x1FF;
    let l3_idx = (vaddr >> 12) & 0x1FF;

    unsafe {
        // Get or create L1 table
        let l1 = mmap_get_or_create_table(l0, l0_idx);
        if l1.is_null() { return; }

        // Get or create L2 table
        let l2 = mmap_get_or_create_table(l1, l1_idx);
        if l2.is_null() { return; }

        // Get or create L3 table
        let l3 = mmap_get_or_create_table(l2, l2_idx);
        if l3.is_null() { return; }

        // Set L3 entry (final mapping)
        let entry = (paddr as u64) | flags.bits() | 0x3;
        *l3.add(l3_idx) = entry;

        // Invalidate TLB for this page
        core::arch::asm!(
            "dsb ishst",
            "tlbi vaae1is, {0}",
            "dsb ish",
            "isb",
            in(reg) vaddr >> 12,
        );
    }
}

/// Get or create next level table for mmap
unsafe fn mmap_get_or_create_table(table: *mut u64, index: usize) -> *mut u64 {
    let entry = *table.add(index);

    if entry & 0x1 == 0 {
        // Not present, create new table
        if let Some(frame) = crate::memory::allocate_frame() {
            let new_table = frame.start_address() as *mut u64;

            // Zero the new table
            for i in 0..512 {
                *new_table.add(i) = 0;
            }

            // Set entry to point to new table
            *table.add(index) = (frame.start_address() as u64) | 0x3;

            return new_table;
        }
        return core::ptr::null_mut();
    }

    // Extract address from entry
    (entry & 0x0000_FFFF_FFFF_F000) as *mut u64
}

fn sys_munmap(addr: usize, length: usize) -> SyscallResult {
    if addr == 0 || length == 0 {
        return errno::EINVAL;
    }

    // Round up to page size
    let page_size = crate::memory::PAGE_SIZE;
    let length = (length + page_size - 1) & !(page_size - 1);

    // Get current process
    let proc = match process::current() {
        Some(p) => p,
        None => return errno::ESRCH,
    };

    // Get page table address
    let page_table = {
        let memory = proc.memory.lock();
        memory.page_table as usize
    };

    if page_table != 0 {
        // Unmap pages and free physical frames
        let num_pages = length / page_size;
        for i in 0..num_pages {
            let vaddr = addr + i * page_size;
            munmap_page(page_table, vaddr);
        }
    }

    // Remove from process memory regions
    {
        let mut memory = proc.memory.lock();
        memory.regions.retain(|r| !(r.start == addr && r.end == addr + length));
    }

    // Remove from mmap regions
    let mut regions = MMAP_REGIONS.lock();
    regions.retain(|r| !(r.start == addr && r.length == length));

    errno::SUCCESS
}

/// Unmap a single page and free the physical frame
fn munmap_page(page_table: usize, vaddr: usize) {
    let l0 = page_table as *mut u64;

    let l0_idx = (vaddr >> 39) & 0x1FF;
    let l1_idx = (vaddr >> 30) & 0x1FF;
    let l2_idx = (vaddr >> 21) & 0x1FF;
    let l3_idx = (vaddr >> 12) & 0x1FF;

    unsafe {
        // Walk page tables
        let l0_entry = *l0.add(l0_idx);
        if l0_entry & 0x1 == 0 { return; }

        let l1 = (l0_entry & 0x0000_FFFF_FFFF_F000) as *mut u64;
        let l1_entry = *l1.add(l1_idx);
        if l1_entry & 0x1 == 0 { return; }

        let l2 = (l1_entry & 0x0000_FFFF_FFFF_F000) as *mut u64;
        let l2_entry = *l2.add(l2_idx);
        if l2_entry & 0x1 == 0 { return; }

        let l3 = (l2_entry & 0x0000_FFFF_FFFF_F000) as *mut u64;
        let l3_entry = *l3.add(l3_idx);
        if l3_entry & 0x1 == 0 { return; }

        // Get physical address and free the frame
        let paddr = (l3_entry & 0x0000_FFFF_FFFF_F000) as usize;
        let frame = crate::memory::PhysFrame::containing_address(paddr);
        crate::memory::deallocate_frame(frame);

        // Clear the entry
        *l3.add(l3_idx) = 0;

        // Invalidate TLB
        core::arch::asm!(
            "dsb ishst",
            "tlbi vaae1is, {0}",
            "dsb ish",
            "isb",
            in(reg) vaddr >> 12,
        );
    }
}

fn sys_mprotect(addr: usize, length: usize, prot: u32) -> SyscallResult {
    let mut regions = MMAP_REGIONS.lock();

    for region in regions.iter_mut() {
        if region.start == addr && region.length == length {
            region.prot = prot;
            return errno::SUCCESS;
        }
    }

    errno::EINVAL
}

// ============================================================================
// Shared Memory Syscalls
// ============================================================================

/// Shared memory segment
struct ShmSegment {
    key: i32,
    id: i32,
    size: usize,
    data: Vec<u8>,
    attachments: usize,
}

static SHM_SEGMENTS: Mutex<Vec<ShmSegment>> = Mutex::new(Vec::new());
static SHM_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn sys_shmget(key: i32, size: usize, flags: i32) -> SyscallResult {
    let mut segments = SHM_SEGMENTS.lock();

    // Check if segment already exists
    for seg in segments.iter() {
        if seg.key == key && key != 0 {
            return seg.id as isize;
        }
    }

    // Create new segment
    let id = SHM_ID_COUNTER.fetch_add(1, Ordering::SeqCst) as i32;
    let _ = flags;

    segments.push(ShmSegment {
        key,
        id,
        size,
        data: vec![0u8; size],
        attachments: 0,
    });

    id as isize
}

fn sys_shmat(shmid: i32, addr: usize, flags: i32) -> SyscallResult {
    let mut segments = SHM_SEGMENTS.lock();
    let _ = (addr, flags);

    for seg in segments.iter_mut() {
        if seg.id == shmid {
            seg.attachments += 1;
            return seg.data.as_ptr() as isize;
        }
    }

    errno::EINVAL
}

fn sys_shmdt(addr: usize) -> SyscallResult {
    let mut segments = SHM_SEGMENTS.lock();

    for seg in segments.iter_mut() {
        if seg.data.as_ptr() as usize == addr {
            if seg.attachments > 0 {
                seg.attachments -= 1;
            }
            return errno::SUCCESS;
        }
    }

    errno::EINVAL
}

fn sys_shmctl(shmid: i32, cmd: i32, buf: usize) -> SyscallResult {
    let _ = buf;

    // IPC_RMID = 0
    if cmd == 0 {
        let mut segments = SHM_SEGMENTS.lock();
        segments.retain(|s| s.id != shmid);
        return errno::SUCCESS;
    }

    errno::EINVAL
}

// ============================================================================
// IPC Syscalls
// ============================================================================

fn sys_ipc_create() -> SyscallResult {
    let (endpoint_a, _endpoint_b) = ipc::create_channel();
    endpoint_a.channel_id().0 as isize
}

fn sys_ipc_send(channel_id: u64, msg_type: u32, data_ptr: usize, len: usize) -> SyscallResult {
    if len > 4096 {
        return errno::E2BIG;
    }

    let mut data = vec![0u8; len];
    unsafe {
        if copy_from_user(&mut data, data_ptr, len).is_err() {
            return errno::EFAULT;
        }
    }

    match ipc::send(ipc::ChannelId(channel_id), msg_type, &data) {
        Ok(()) => errno::SUCCESS,
        Err(_) => errno::EPIPE,
    }
}

fn sys_ipc_recv(channel_id: u64, buf_ptr: usize, buf_len: usize) -> SyscallResult {
    match ipc::recv(ipc::ChannelId(channel_id), buf_len) {
        Ok((msg_type, data)) => {
            let copy_len = data.len().min(buf_len);
            unsafe {
                if copy_to_user(buf_ptr, &data[..copy_len]).is_err() {
                    return errno::EFAULT;
                }
            }
            // Return message type in high bits, length in low bits
            ((msg_type as isize) << 32) | (copy_len as isize)
        }
        Err(_) => errno::EAGAIN,
    }
}

fn sys_ipc_close(channel_id: u64) -> SyscallResult {
    ipc::close(ipc::ChannelId(channel_id));
    errno::SUCCESS
}

fn sys_ipc_lookup(name_ptr: usize) -> SyscallResult {
    let name = match unsafe { read_user_string(name_ptr, 128) } {
        Ok(n) => n,
        Err(e) => return e,
    };

    match ipc::lookup(&name) {
        Some(id) => id.0 as isize,
        None => errno::ENOENT,
    }
}

fn sys_ipc_register(name_ptr: usize, channel_id: u64) -> SyscallResult {
    let name = match unsafe { read_user_string(name_ptr, 128) } {
        Ok(n) => n,
        Err(e) => return e,
    };

    match ipc::register_endpoint(&name, ipc::ChannelId(channel_id)) {
        Ok(()) => errno::SUCCESS,
        Err(_) => errno::EEXIST,
    }
}

// ============================================================================
// Signal Syscalls
// ============================================================================

fn sys_sigaction(signum: i32, act: usize, oldact: usize) -> SyscallResult {
    if let Ok(signal) = Signal::try_from(signum as u8) {
        // Get old action if requested
        if oldact != 0 {
            if let Some(old) = crate::signal::get_handler(signal) {
                unsafe {
                    *(oldact as *mut SignalAction) = old;
                }
            }
        }

        // Set new action if provided
        if act != 0 {
            let new_action = unsafe { *(act as *const SignalAction) };
            crate::signal::set_handler(signal, new_action);
        }

        errno::SUCCESS
    } else {
        errno::EINVAL
    }
}

fn sys_sigprocmask(how: i32, set: usize, oldset: usize) -> SyscallResult {
    let _ = (how, set, oldset);
    // SIG_BLOCK = 0, SIG_UNBLOCK = 1, SIG_SETMASK = 2
    errno::SUCCESS
}

fn sys_sigsuspend(mask: usize) -> SyscallResult {
    let _ = mask;
    // Wait for a signal
    crate::scheduler::schedule();
    errno::EINTR
}

fn sys_sigreturn() -> SyscallResult {
    // Return from signal handler
    errno::SUCCESS
}

// ============================================================================
// Time Syscalls
// ============================================================================

fn sys_time(time_ptr: usize) -> SyscallResult {
    let secs = crate::time::system_time_secs();

    if time_ptr != 0 {
        unsafe {
            *(time_ptr as *mut u64) = secs;
        }
    }

    secs as isize
}

fn sys_sleep(seconds: u64) -> SyscallResult {
    crate::time::sleep_ms(seconds * 1000);
    errno::SUCCESS
}

/// Timespec structure
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

fn sys_nanosleep(req: usize, rem: usize) -> SyscallResult {
    let timespec = unsafe { *(req as *const Timespec) };

    let ms = (timespec.tv_sec * 1000) + (timespec.tv_nsec / 1_000_000);
    crate::time::sleep_ms(ms as u64);

    if rem != 0 {
        unsafe {
            *(rem as *mut Timespec) = Timespec { tv_sec: 0, tv_nsec: 0 };
        }
    }

    errno::SUCCESS
}

/// Timeval structure
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

fn sys_gettimeofday(tv: usize, tz: usize) -> SyscallResult {
    let _ = tz;

    if tv != 0 {
        let secs = crate::time::system_time_secs();
        let usecs = crate::time::system_time_usecs() % 1_000_000;

        unsafe {
            *(tv as *mut Timeval) = Timeval {
                tv_sec: secs as i64,
                tv_usec: usecs as i64,
            };
        }
    }

    errno::SUCCESS
}

fn sys_clock_gettime(clock_id: i32, tp: usize) -> SyscallResult {
    let _ = clock_id; // CLOCK_REALTIME = 0, CLOCK_MONOTONIC = 1

    if tp != 0 {
        let secs = crate::time::system_time_secs();
        let nsecs = (crate::time::system_time_usecs() % 1_000_000) * 1000;

        unsafe {
            *(tp as *mut Timespec) = Timespec {
                tv_sec: secs as i64,
                tv_nsec: nsecs as i64,
            };
        }
    }

    errno::SUCCESS
}

// ============================================================================
// System Info Syscalls
// ============================================================================

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

#[repr(C)]
pub struct Sysinfo {
    pub uptime: i64,
    pub loads: [u64; 3],
    pub totalram: u64,
    pub freeram: u64,
    pub sharedram: u64,
    pub bufferram: u64,
    pub totalswap: u64,
    pub freeswap: u64,
    pub procs: u16,
    pub totalhigh: u64,
    pub freehigh: u64,
    pub mem_unit: u32,
}

fn sys_sysinfo(buf_ptr: usize) -> SyscallResult {
    let buf = unsafe { &mut *(buf_ptr as *mut Sysinfo) };
    let mem_stats = crate::memory::stats();

    buf.uptime = uptime_secs() as i64;
    buf.loads = [0, 0, 0]; // Load averages
    buf.totalram = mem_stats.total as u64;
    buf.freeram = mem_stats.free as u64;
    buf.sharedram = 0;
    buf.bufferram = 0;
    buf.totalswap = 0;
    buf.freeswap = 0;
    buf.procs = process::list().len() as u16;
    buf.totalhigh = 0;
    buf.freehigh = 0;
    buf.mem_unit = 1;

    errno::SUCCESS
}

#[repr(C)]
pub struct Rlimit {
    pub rlim_cur: u64,
    pub rlim_max: u64,
}

fn sys_getrlimit(resource: i32, rlim: usize) -> SyscallResult {
    let limit = unsafe { &mut *(rlim as *mut Rlimit) };

    // Default limits
    match resource {
        0 => { // RLIMIT_CPU
            limit.rlim_cur = u64::MAX;
            limit.rlim_max = u64::MAX;
        }
        1 => { // RLIMIT_FSIZE
            limit.rlim_cur = u64::MAX;
            limit.rlim_max = u64::MAX;
        }
        2 => { // RLIMIT_DATA
            limit.rlim_cur = 64 * 1024 * 1024;
            limit.rlim_max = 64 * 1024 * 1024;
        }
        3 => { // RLIMIT_STACK
            limit.rlim_cur = 8 * 1024 * 1024;
            limit.rlim_max = 64 * 1024 * 1024;
        }
        7 => { // RLIMIT_NOFILE
            limit.rlim_cur = 1024;
            limit.rlim_max = 4096;
        }
        _ => {
            limit.rlim_cur = u64::MAX;
            limit.rlim_max = u64::MAX;
        }
    }

    errno::SUCCESS
}

fn sys_setrlimit(resource: i32, rlim: usize) -> SyscallResult {
    let _ = (resource, rlim);
    errno::SUCCESS
}

// ============================================================================
// Synchronization Syscalls
// ============================================================================

/// Semaphore structure
struct Semaphore {
    value: AtomicU64,
    waiters: Mutex<Vec<Pid>>,
}

static SEMAPHORES: Mutex<Vec<(usize, Semaphore)>> = Mutex::new(Vec::new());

fn sys_futex(uaddr: usize, op: i32, val: u32, timeout: usize, uaddr2: usize, val3: u32) -> SyscallResult {
    let _ = (timeout, uaddr2, val3);

    // FUTEX_WAIT = 0, FUTEX_WAKE = 1
    match op & 0x7F {
        0 => { // FUTEX_WAIT
            let current = unsafe { *(uaddr as *const u32) };
            if current != val {
                return errno::EAGAIN;
            }
            // Block until woken
            crate::scheduler::schedule();
            errno::SUCCESS
        }
        1 => { // FUTEX_WAKE
            // Wake up to 'val' waiters
            errno::SUCCESS
        }
        _ => errno::EINVAL,
    }
}

fn sys_sem_init(sem: usize, pshared: i32, value: u32) -> SyscallResult {
    let _ = pshared;

    let new_sem = Semaphore {
        value: AtomicU64::new(value as u64),
        waiters: Mutex::new(Vec::new()),
    };

    SEMAPHORES.lock().push((sem, new_sem));
    errno::SUCCESS
}

fn sys_sem_wait(sem: usize) -> SyscallResult {
    let sems = SEMAPHORES.lock();

    for (addr, s) in sems.iter() {
        if *addr == sem {
            loop {
                let val = s.value.load(Ordering::SeqCst);
                if val > 0 {
                    if s.value.compare_exchange(val, val - 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                        return errno::SUCCESS;
                    }
                } else {
                    // Block
                    drop(sems);
                    crate::scheduler::schedule();
                    return sys_sem_wait(sem);
                }
            }
        }
    }

    errno::EINVAL
}

fn sys_sem_post(sem: usize) -> SyscallResult {
    let sems = SEMAPHORES.lock();

    for (addr, s) in sems.iter() {
        if *addr == sem {
            s.value.fetch_add(1, Ordering::SeqCst);
            return errno::SUCCESS;
        }
    }

    errno::EINVAL
}

fn sys_sem_destroy(sem: usize) -> SyscallResult {
    let mut sems = SEMAPHORES.lock();
    sems.retain(|(addr, _)| *addr != sem);
    errno::SUCCESS
}

// ============================================================================
// Networking Syscalls
// ============================================================================

fn sys_socket(domain: i32, sock_type: i32, protocol: i32) -> SyscallResult {
    match crate::net::socket::create_socket(domain, sock_type, protocol) {
        Ok(fd) => fd as isize,
        Err(_) => errno::EINVAL,
    }
}

fn sys_bind(sockfd: i32, addr: usize, addrlen: u32) -> SyscallResult {
    let _ = (sockfd, addr, addrlen);
    errno::SUCCESS
}

fn sys_listen(sockfd: i32, backlog: i32) -> SyscallResult {
    let _ = (sockfd, backlog);
    errno::SUCCESS
}

fn sys_accept(sockfd: i32, addr: usize, addrlen: usize) -> SyscallResult {
    let _ = (sockfd, addr, addrlen);
    errno::EAGAIN
}

fn sys_connect(sockfd: i32, addr: usize, addrlen: u32) -> SyscallResult {
    let _ = (sockfd, addr, addrlen);
    errno::SUCCESS
}

fn sys_send(sockfd: i32, buf: usize, len: usize, flags: i32) -> SyscallResult {
    let _ = flags;

    let mut data = vec![0u8; len];
    unsafe {
        if copy_from_user(&mut data, buf, len).is_err() {
            return errno::EFAULT;
        }
    }

    match crate::net::socket::send(sockfd, &data) {
        Ok(n) => n as isize,
        Err(_) => errno::EIO,
    }
}

fn sys_recv(sockfd: i32, buf: usize, len: usize, flags: i32) -> SyscallResult {
    let _ = flags;

    let mut data = vec![0u8; len];
    match crate::net::socket::recv(sockfd, &mut data) {
        Ok(n) => {
            unsafe {
                if copy_to_user(buf, &data[..n]).is_err() {
                    return errno::EFAULT;
                }
            }
            n as isize
        }
        Err(_) => errno::EIO,
    }
}

fn sys_sendto(sockfd: i32, buf: usize, len: usize, flags: i32, dest_addr: usize, addrlen: u32) -> SyscallResult {
    let _ = (dest_addr, addrlen);
    sys_send(sockfd, buf, len, flags)
}

fn sys_recvfrom(sockfd: i32, buf: usize, len: usize, flags: i32, src_addr: usize, addrlen: usize) -> SyscallResult {
    let _ = (src_addr, addrlen);
    sys_recv(sockfd, buf, len, flags)
}

// ============================================================================
// AI Syscalls (HubLab IO Specific)
// ============================================================================

/// Loaded AI model handle
static AI_MODEL_HANDLE: AtomicU64 = AtomicU64::new(0);

fn sys_ai_load(path_ptr: usize, path_len: usize) -> SyscallResult {
    let _ = path_len;

    let path = match unsafe { read_user_string(path_ptr, 256) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    crate::kinfo!("AI: Loading model from {}", path);

    // Return a model handle
    let handle = AI_MODEL_HANDLE.fetch_add(1, Ordering::SeqCst) + 1;
    handle as isize
}

fn sys_ai_generate(
    model_handle: usize,
    prompt_ptr: usize,
    prompt_len: usize,
    out_ptr: usize,
    out_len: usize,
) -> SyscallResult {
    let _ = (model_handle, prompt_len);

    let prompt = match unsafe { read_user_string(prompt_ptr, 1024) } {
        Ok(p) => p,
        Err(e) => return e,
    };

    crate::kinfo!("AI: Generate with prompt: {}", &prompt[..prompt.len().min(50)]);

    // Placeholder response
    let response = "AI response placeholder - model not loaded";
    let copy_len = response.len().min(out_len);

    unsafe {
        if copy_to_user(out_ptr, response.as_bytes()).is_err() {
            return errno::EFAULT;
        }
    }

    copy_len as isize
}

fn sys_ai_tokenize(
    model_handle: usize,
    text_ptr: usize,
    text_len: usize,
    tokens_ptr: usize,
) -> SyscallResult {
    let _ = (model_handle, text_len, tokens_ptr);

    let _text = match unsafe { read_user_string(text_ptr, 1024) } {
        Ok(t) => t,
        Err(e) => return e,
    };

    // Return number of tokens (placeholder)
    0
}

fn sys_ai_unload(model_handle: u32) -> SyscallResult {
    crate::kinfo!("AI: Unloading model {}", model_handle);
    errno::SUCCESS
}

fn sys_ai_embed(
    model_handle: usize,
    text_ptr: usize,
    text_len: usize,
    embed_ptr: usize,
) -> SyscallResult {
    let _ = (model_handle, text_len, embed_ptr);

    let _text = match unsafe { read_user_string(text_ptr, 1024) } {
        Ok(t) => t,
        Err(e) => return e,
    };

    // Return embedding dimension (placeholder)
    0
}

// ============================================================================
// SMP / Scheduling Syscalls
// ============================================================================

/// Set CPU affinity mask for a process
fn sys_sched_setaffinity(pid: u32, cpusetsize: usize, mask_ptr: usize) -> SyscallResult {
    // pid 0 means current process
    let target_pid = if pid == 0 {
        if let Some(proc) = process::current() {
            proc.pid
        } else {
            return errno::ESRCH;
        }
    } else {
        Pid(pid as u64)
    };

    // Read the CPU mask from user space
    if cpusetsize < 8 {
        return errno::EINVAL;
    }

    let mask: u64 = unsafe {
        if let Ok(m) = copy_u64_from_user(mask_ptr) {
            m
        } else {
            return errno::EFAULT;
        }
    };

    // Validate mask - at least one CPU must be set
    if mask == 0 {
        return errno::EINVAL;
    }

    // Check that set CPUs are actually online
    let online_mask = get_online_cpu_mask();
    if mask & online_mask == 0 {
        return errno::EINVAL;
    }

    // Set the affinity
    crate::scheduler::set_affinity(target_pid, mask);

    errno::SUCCESS
}

/// Get CPU affinity mask for a process
fn sys_sched_getaffinity(pid: u32, cpusetsize: usize, mask_ptr: usize) -> SyscallResult {
    // pid 0 means current process
    let target_pid = if pid == 0 {
        if let Some(proc) = process::current() {
            proc.pid
        } else {
            return errno::ESRCH;
        }
    } else {
        Pid(pid as u64)
    };

    if cpusetsize < 8 {
        return errno::EINVAL;
    }

    let mask = crate::scheduler::get_affinity(target_pid);

    unsafe {
        if copy_u64_to_user(mask_ptr, mask).is_err() {
            return errno::EFAULT;
        }
    }

    8 // Return size of mask
}

/// Yield the processor
fn sys_sched_yield() -> SyscallResult {
    crate::scheduler::yield_now();
    errno::SUCCESS
}

/// Get maximum priority for a scheduling policy
fn sys_sched_get_priority_max(policy: i32) -> SyscallResult {
    let _ = policy;
    // Our scheduler uses 0-63 priority range
    63
}

/// Get minimum priority for a scheduling policy
fn sys_sched_get_priority_min(policy: i32) -> SyscallResult {
    let _ = policy;
    0
}

/// Get the CPU the calling thread is running on
fn sys_getcpu(cpu_ptr: usize, node_ptr: usize, _unused: usize) -> SyscallResult {
    let cpu = crate::smp::cpu_id();

    if cpu_ptr != 0 {
        unsafe {
            if copy_u32_to_user(cpu_ptr, cpu).is_err() {
                return errno::EFAULT;
            }
        }
    }

    if node_ptr != 0 {
        // NUMA node - we only have one node for now
        unsafe {
            if copy_u32_to_user(node_ptr, 0).is_err() {
                return errno::EFAULT;
            }
        }
    }

    errno::SUCCESS
}

/// SMP info structure for user space
#[repr(C)]
struct SmpInfoUser {
    online_cpus: u32,
    total_cpus: u32,
    current_cpu: u32,
    _reserved: u32,
    per_cpu_ticks: [u64; crate::smp::MAX_CPUS],
    per_cpu_ctx_switches: [u64; crate::smp::MAX_CPUS],
}

/// Get SMP system information
fn sys_smp_info(info_ptr: usize) -> SyscallResult {
    if info_ptr == 0 {
        return errno::EFAULT;
    }

    let stats = crate::smp::stats();

    let mut info = SmpInfoUser {
        online_cpus: stats.online_cpus as u32,
        total_cpus: crate::smp::MAX_CPUS as u32,
        current_cpu: crate::smp::cpu_id(),
        _reserved: 0,
        per_cpu_ticks: [0; crate::smp::MAX_CPUS],
        per_cpu_ctx_switches: [0; crate::smp::MAX_CPUS],
    };

    // Fill in per-CPU stats
    for cpu in 0..crate::smp::MAX_CPUS {
        if let Some(pcpu) = crate::smp::get(cpu as u32) {
            info.per_cpu_ticks[cpu] = pcpu.ticks.load(core::sync::atomic::Ordering::Relaxed);
            info.per_cpu_ctx_switches[cpu] = pcpu.context_switches.load(core::sync::atomic::Ordering::Relaxed);
        }
    }

    unsafe {
        let src = &info as *const SmpInfoUser as *const u8;
        let size = core::mem::size_of::<SmpInfoUser>();
        if copy_to_user(info_ptr, core::slice::from_raw_parts(src, size)).is_err() {
            return errno::EFAULT;
        }
    }

    errno::SUCCESS
}

/// Get a mask of online CPUs
fn get_online_cpu_mask() -> u64 {
    let mut mask = 0u64;
    for cpu in 0..crate::smp::MAX_CPUS as u32 {
        if crate::smp::is_online(cpu) {
            mask |= 1 << cpu;
        }
    }
    mask
}

/// Copy a u64 from user space
unsafe fn copy_u64_from_user(ptr: usize) -> Result<u64, ()> {
    if ptr == 0 {
        return Err(());
    }
    // In a real implementation, validate the address
    Ok(*(ptr as *const u64))
}

/// Copy a u64 to user space
unsafe fn copy_u64_to_user(ptr: usize, value: u64) -> Result<(), ()> {
    if ptr == 0 {
        return Err(());
    }
    // In a real implementation, validate the address
    *(ptr as *mut u64) = value;
    Ok(())
}

/// Copy a u32 to user space
unsafe fn copy_u32_to_user(ptr: usize, value: u32) -> Result<(), ()> {
    if ptr == 0 {
        return Err(());
    }
    // In a real implementation, validate the address
    *(ptr as *mut u32) = value;
    Ok(())
}
