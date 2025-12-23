//! Process Management
//!
//! Provides process and thread abstractions for the kernel.
//! Implements task creation, scheduling, and lifecycle management.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use spin::{Mutex, RwLock};

pub mod elf;
pub mod context;

/// Process ID counter
static PID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Thread ID counter
static TID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Global process table
static PROCESSES: RwLock<BTreeMap<Pid, Arc<Process>>> = RwLock::new(BTreeMap::new());

/// Current process per CPU (simplified: single CPU for now)
static CURRENT_PID: AtomicU64 = AtomicU64::new(0);

/// Process identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pid(pub u64);

/// Thread identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tid(pub u64);

/// Process state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessState {
    /// Process is being created
    Creating,
    /// Process is ready to run
    Ready,
    /// Process is currently running
    Running,
    /// Process is blocked waiting for something
    Blocked(BlockReason),
    /// Process is stopped (e.g., by signal)
    Stopped,
    /// Process has exited but not yet reaped (zombie)
    Zombie(i32),
}

/// Reason for blocking
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockReason {
    /// Waiting for I/O
    Io,
    /// Waiting for IPC message
    Ipc,
    /// Sleeping
    Sleep,
    /// Waiting for child process
    WaitChild,
    /// Waiting for mutex/semaphore
    Sync,
}

/// Thread state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadState {
    Ready,
    Running,
    Blocked(BlockReason),
    Terminated,
}

/// Process priority (0-255, lower = higher priority)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    pub const IDLE: Self = Self(255);
    pub const LOW: Self = Self(200);
    pub const NORMAL: Self = Self(128);
    pub const HIGH: Self = Self(64);
    pub const REALTIME: Self = Self(0);
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// CPU context for context switching
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct CpuContext {
    // General purpose registers (x0-x30)
    pub x: [u64; 31],
    // Stack pointer
    pub sp: u64,
    // Program counter (saved ELR_EL1)
    pub pc: u64,
    // Process state (saved SPSR_EL1)
    pub pstate: u64,
    // Thread pointer (TPIDR_EL0)
    pub tpidr: u64,
}

/// Memory space for a process
pub struct MemorySpace {
    /// Page table root physical address
    pub page_table: u64,
    /// Heap start
    pub heap_start: usize,
    /// Heap end (brk)
    pub heap_end: usize,
    /// Stack top
    pub stack_top: usize,
    /// Stack size
    pub stack_size: usize,
    /// Mapped regions
    pub regions: Vec<MemoryRegion>,
}

/// A mapped memory region
#[derive(Clone, Debug)]
pub struct MemoryRegion {
    pub start: usize,
    pub end: usize,
    pub flags: MemoryFlags,
    pub name: String,
}

bitflags::bitflags! {
    /// Memory region flags
    #[derive(Clone, Copy, Debug)]
    pub struct MemoryFlags: u32 {
        const READ = 1 << 0;
        const WRITE = 1 << 1;
        const EXEC = 1 << 2;
        const USER = 1 << 3;
        const SHARED = 1 << 4;
    }
}

/// File descriptor table entry
#[derive(Clone, Debug)]
pub struct FileDescriptor {
    pub fd: u32,
    pub flags: u32,
    pub path: String,
    pub offset: u64,
}

/// A thread within a process
pub struct Thread {
    /// Thread ID
    pub tid: Tid,
    /// Owning process
    pub pid: Pid,
    /// Thread state
    pub state: ThreadState,
    /// CPU context
    pub context: CpuContext,
    /// Kernel stack
    pub kernel_stack: usize,
    /// Kernel stack size
    pub kernel_stack_size: usize,
    /// Thread-local storage pointer
    pub tls: usize,
    /// Thread name
    pub name: String,
}

/// A process
pub struct Process {
    /// Process ID
    pub pid: Pid,
    /// Parent process ID
    pub ppid: Pid,
    /// Process state
    pub state: Mutex<ProcessState>,
    /// Process name
    pub name: String,
    /// Command line arguments
    pub args: Vec<String>,
    /// Environment variables
    pub env: Vec<(String, String)>,
    /// Working directory
    pub cwd: String,
    /// Priority
    pub priority: Priority,
    /// Memory space
    pub memory: Mutex<MemorySpace>,
    /// Threads
    pub threads: RwLock<BTreeMap<Tid, Thread>>,
    /// Main thread ID
    pub main_tid: Tid,
    /// File descriptors
    pub files: Mutex<BTreeMap<u32, FileDescriptor>>,
    /// Next file descriptor number
    pub next_fd: AtomicU32,
    /// Child processes
    pub children: Mutex<Vec<Pid>>,
    /// Exit code (when zombie)
    pub exit_code: Mutex<Option<i32>>,
    /// User ID
    pub uid: u32,
    /// Group ID
    pub gid: u32,
    /// CPU time used (nanoseconds)
    pub cpu_time: AtomicU64,
    /// Creation time (nanoseconds since boot)
    pub created_at: u64,
}

impl Process {
    /// Create a new process
    pub fn new(name: &str, ppid: Pid) -> Arc<Self> {
        let pid = Pid(PID_COUNTER.fetch_add(1, Ordering::SeqCst));
        let main_tid = Tid(TID_COUNTER.fetch_add(1, Ordering::SeqCst));

        let process = Arc::new(Self {
            pid,
            ppid,
            state: Mutex::new(ProcessState::Creating),
            name: String::from(name),
            args: Vec::new(),
            env: Vec::new(),
            cwd: String::from("/"),
            priority: Priority::default(),
            memory: Mutex::new(MemorySpace {
                page_table: 0,
                heap_start: 0,
                heap_end: 0,
                stack_top: 0,
                stack_size: 0,
                regions: Vec::new(),
            }),
            threads: RwLock::new(BTreeMap::new()),
            main_tid,
            files: Mutex::new(BTreeMap::new()),
            next_fd: AtomicU32::new(3), // 0,1,2 reserved for stdin/out/err
            children: Mutex::new(Vec::new()),
            exit_code: Mutex::new(None),
            uid: 0,
            gid: 0,
            cpu_time: AtomicU64::new(0),
            created_at: 0, // TODO: Get current time
        });

        // Create main thread
        let main_thread = Thread {
            tid: main_tid,
            pid,
            state: ThreadState::Ready,
            context: CpuContext::default(),
            kernel_stack: 0,
            kernel_stack_size: 0,
            tls: 0,
            name: String::from("main"),
        };

        process.threads.write().insert(main_tid, main_thread);

        // Add to process table
        PROCESSES.write().insert(pid, process.clone());

        process
    }

    /// Set process state
    pub fn set_state(&self, state: ProcessState) {
        *self.state.lock() = state;
    }

    /// Get process state
    pub fn get_state(&self) -> ProcessState {
        *self.state.lock()
    }

    /// Add a file descriptor
    pub fn add_fd(&self, path: &str, flags: u32) -> u32 {
        let fd = self.next_fd.fetch_add(1, Ordering::SeqCst);
        self.files.lock().insert(fd, FileDescriptor {
            fd,
            flags,
            path: String::from(path),
            offset: 0,
        });
        fd
    }

    /// Close a file descriptor
    pub fn close_fd(&self, fd: u32) -> bool {
        self.files.lock().remove(&fd).is_some()
    }

    /// Get a file descriptor
    pub fn get_fd(&self, fd: u32) -> Option<FileDescriptor> {
        self.files.lock().get(&fd).cloned()
    }

    /// Create a new thread
    pub fn create_thread(&self, name: &str, entry: usize, stack: usize) -> Tid {
        let tid = Tid(TID_COUNTER.fetch_add(1, Ordering::SeqCst));

        let mut context = CpuContext::default();
        context.pc = entry as u64;
        context.sp = stack as u64;

        let thread = Thread {
            tid,
            pid: self.pid,
            state: ThreadState::Ready,
            context,
            kernel_stack: 0,
            kernel_stack_size: 0,
            tls: 0,
            name: String::from(name),
        };

        self.threads.write().insert(tid, thread);
        tid
    }

    /// Exit the process
    pub fn exit(&self, code: i32) {
        *self.exit_code.lock() = Some(code);
        self.set_state(ProcessState::Zombie(code));

        // Mark all threads as terminated
        for (_, thread) in self.threads.write().iter_mut() {
            thread.state = ThreadState::Terminated;
        }

        // TODO: Notify parent process
        // TODO: Reparent children to init
    }
}

/// Initialize process subsystem
pub fn init() {
    crate::kprintln!("  Process manager initialized");
}

/// Spawn the init process (PID 1)
pub fn spawn_init() {
    let init = Process::new("init", Pid(0));

    // Set up init's memory space
    {
        let mut memory = init.memory.lock();
        memory.stack_top = 0x7FFF_FFFF_0000;
        memory.stack_size = 1024 * 1024; // 1MB stack
        memory.heap_start = 0x1000_0000;
        memory.heap_end = 0x1000_0000;
    }

    // Set up standard file descriptors
    init.files.lock().insert(0, FileDescriptor {
        fd: 0,
        flags: 0, // O_RDONLY
        path: String::from("/dev/console"),
        offset: 0,
    });
    init.files.lock().insert(1, FileDescriptor {
        fd: 1,
        flags: 1, // O_WRONLY
        path: String::from("/dev/console"),
        offset: 0,
    });
    init.files.lock().insert(2, FileDescriptor {
        fd: 2,
        flags: 1, // O_WRONLY
        path: String::from("/dev/console"),
        offset: 0,
    });

    init.set_state(ProcessState::Ready);

    CURRENT_PID.store(init.pid.0, Ordering::Release);

    crate::kprintln!("[PROC] Init process created (PID {})", init.pid.0);
}

/// Get current process
pub fn current() -> Option<Arc<Process>> {
    let pid = Pid(CURRENT_PID.load(Ordering::Acquire));
    PROCESSES.read().get(&pid).cloned()
}

/// Get process by PID
pub fn get(pid: Pid) -> Option<Arc<Process>> {
    PROCESSES.read().get(&pid).cloned()
}

/// List all processes
pub fn list() -> Vec<Arc<Process>> {
    PROCESSES.read().values().cloned().collect()
}

/// Fork the current process
pub fn fork() -> Result<Pid, &'static str> {
    let parent = current().ok_or("No current process")?;

    let child = Process::new(&parent.name, parent.pid);

    // Copy memory space (COW would be better but simplified for now)
    {
        let parent_mem = parent.memory.lock();
        let mut child_mem = child.memory.lock();
        child_mem.heap_start = parent_mem.heap_start;
        child_mem.heap_end = parent_mem.heap_end;
        child_mem.stack_top = parent_mem.stack_top;
        child_mem.stack_size = parent_mem.stack_size;
        child_mem.regions = parent_mem.regions.clone();
        // TODO: Actually copy/COW page tables
    }

    // Copy file descriptors
    {
        let parent_files = parent.files.lock();
        let mut child_files = child.files.lock();
        for (fd, desc) in parent_files.iter() {
            child_files.insert(*fd, desc.clone());
        }
    }

    // Copy environment
    // child.env = parent.env.clone(); // Would need interior mutability

    // Add to parent's children
    parent.children.lock().push(child.pid);

    child.set_state(ProcessState::Ready);

    Ok(child.pid)
}

/// Execute a new program in the current process
pub fn exec(path: &str, args: &[&str]) -> Result<(), &'static str> {
    let process = current().ok_or("No current process")?;

    // TODO: Load ELF file from path
    // TODO: Set up new memory space
    // TODO: Set entry point
    // TODO: Set up stack with args

    crate::kprintln!("[PROC] exec({}, {:?}) - not fully implemented", path, args);

    Ok(())
}

/// Wait for a child process to exit
pub fn wait(pid: Option<Pid>) -> Result<(Pid, i32), &'static str> {
    let parent = current().ok_or("No current process")?;

    // Find a zombie child
    let children = parent.children.lock();
    for &child_pid in children.iter() {
        if let Some(target) = pid {
            if child_pid != target {
                continue;
            }
        }

        if let Some(child) = get(child_pid) {
            if let ProcessState::Zombie(code) = child.get_state() {
                // Remove child from process table
                drop(children);
                PROCESSES.write().remove(&child_pid);
                parent.children.lock().retain(|&p| p != child_pid);
                return Ok((child_pid, code));
            }
        }
    }

    // No zombie child found
    // TODO: Block and wait
    Err("No child process to wait for")
}

/// Kill a process
pub fn kill(pid: Pid, signal: i32) -> Result<(), &'static str> {
    let process = get(pid).ok_or("Process not found")?;

    match signal {
        9 => {
            // SIGKILL - force exit
            process.exit(-9);
            Ok(())
        }
        15 => {
            // SIGTERM - request exit
            // TODO: Send signal to process
            process.exit(-15);
            Ok(())
        }
        _ => {
            // TODO: Handle other signals
            Err("Signal not supported")
        }
    }
}

/// Process information for display
#[derive(Clone, Debug)]
pub struct ProcessInfo {
    pub pid: u64,
    pub ppid: u64,
    pub name: String,
    pub state: &'static str,
    pub priority: u8,
    pub threads: usize,
    pub cpu_time: u64,
}

/// Get process info for all processes
pub fn info() -> Vec<ProcessInfo> {
    PROCESSES.read().values().map(|p| {
        let state_str = match p.get_state() {
            ProcessState::Creating => "creating",
            ProcessState::Ready => "ready",
            ProcessState::Running => "running",
            ProcessState::Blocked(_) => "blocked",
            ProcessState::Stopped => "stopped",
            ProcessState::Zombie(_) => "zombie",
        };

        ProcessInfo {
            pid: p.pid.0,
            ppid: p.ppid.0,
            name: p.name.clone(),
            state: state_str,
            priority: p.priority.0,
            threads: p.threads.read().len(),
            cpu_time: p.cpu_time.load(Ordering::Relaxed),
        }
    }).collect()
}
