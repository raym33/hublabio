//! Task Management
//!
//! Core task (thread) management including context switching,
//! process lifecycle, and wait queue integration.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::string::String;
use core::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use spin::{Mutex, RwLock};

use crate::process::{Pid, ProcessState, BlockReason, CpuContext, Process};
use crate::memory::{PAGE_SIZE, PageFlags, PhysFrame, VirtPage, allocate_frame, deallocate_frame};
use crate::memory::paging::AddressSpace;
use crate::signal::{Signal, SignalState, SignalInfo, SignalAction};

/// Task identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub u64);

/// Task ID counter
static TASK_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Current task ID per CPU
static CURRENT_TASK: AtomicU64 = AtomicU64::new(0);

/// Global task table
static TASKS: RwLock<BTreeMap<TaskId, Arc<Task>>> = RwLock::new(BTreeMap::new());

/// Wait queues for child exit notification
static WAIT_QUEUES: Mutex<BTreeMap<Pid, VecDeque<WaitEntry>>> = Mutex::new(BTreeMap::new());

/// Task state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskState {
    /// Task is ready to run
    Ready,
    /// Task is currently running
    Running,
    /// Task is blocked waiting for something
    Blocked(WaitReason),
    /// Task is stopped (by signal)
    Stopped,
    /// Task has exited
    Exited(i32),
}

/// Reason for blocking
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitReason {
    /// Waiting for child process
    Child,
    /// Waiting for I/O
    Io,
    /// Sleeping
    Sleep { until: u64 },
    /// Waiting on futex
    Futex { addr: usize },
    /// Waiting for signal
    Signal,
    /// Waiting for pipe
    Pipe,
    /// Waiting for socket
    Socket,
}

/// Entry in wait queue
#[derive(Clone, Debug)]
pub struct WaitEntry {
    pub task_id: TaskId,
    pub waiting_for: WaitTarget,
}

/// What a task is waiting for
#[derive(Clone, Debug)]
pub enum WaitTarget {
    /// Any child
    AnyChild,
    /// Specific child PID
    ChildPid(Pid),
    /// Process group
    ProcessGroup(u32),
}

/// Kernel stack size
pub const KERNEL_STACK_SIZE: usize = 16 * 1024; // 16 KB

/// User stack size
pub const USER_STACK_SIZE: usize = 1024 * 1024; // 1 MB

/// User stack top address
pub const USER_STACK_TOP: usize = 0x7FFF_FFFF_0000;

/// Task structure
pub struct Task {
    /// Task ID
    pub id: TaskId,
    /// Process this task belongs to
    pub process: Arc<Process>,
    /// Task state
    pub state: Mutex<TaskState>,
    /// CPU context
    pub context: Mutex<CpuContext>,
    /// Kernel stack base
    pub kernel_stack: usize,
    /// Kernel stack size
    pub kernel_stack_size: usize,
    /// User stack top
    pub user_stack_top: usize,
    /// Signal state
    pub signals: Mutex<SignalState>,
    /// Task name
    pub name: String,
    /// Time slice remaining (microseconds)
    pub time_slice: AtomicU64,
    /// Total CPU time used (nanoseconds)
    pub cpu_time: AtomicU64,
    /// In kernel mode
    pub in_kernel: AtomicBool,
}

impl Task {
    /// Create a new task for a process
    pub fn new(process: Arc<Process>, name: &str) -> Option<Arc<Self>> {
        let id = TaskId(TASK_ID_COUNTER.fetch_add(1, Ordering::SeqCst));

        // Allocate kernel stack
        let kernel_stack = allocate_kernel_stack()?;

        let task = Arc::new(Self {
            id,
            process,
            state: Mutex::new(TaskState::Ready),
            context: Mutex::new(CpuContext::default()),
            kernel_stack,
            kernel_stack_size: KERNEL_STACK_SIZE,
            user_stack_top: USER_STACK_TOP,
            signals: Mutex::new(SignalState::default()),
            name: String::from(name),
            time_slice: AtomicU64::new(10_000), // 10ms
            cpu_time: AtomicU64::new(0),
            in_kernel: AtomicBool::new(true),
        });

        // Add to task table
        TASKS.write().insert(id, task.clone());

        Some(task)
    }

    /// Set up task context for user mode execution
    pub fn setup_user_context(
        &self,
        entry_point: usize,
        stack_pointer: usize,
        arg0: usize,
    ) {
        let mut ctx = self.context.lock();
        *ctx = CpuContext::default();
        ctx.pc = entry_point as u64;
        ctx.sp = stack_pointer as u64;
        ctx.x[0] = arg0 as u64;
        // SPSR_EL1: EL0t (user mode), interrupts enabled
        ctx.pstate = 0;
    }

    /// Set up task context for kernel mode execution
    pub fn setup_kernel_context(
        &self,
        entry_point: usize,
        arg0: usize,
    ) {
        let mut ctx = self.context.lock();
        *ctx = CpuContext::default();
        ctx.pc = entry_point as u64;
        ctx.sp = (self.kernel_stack + self.kernel_stack_size) as u64;
        ctx.x[0] = arg0 as u64;
        // SPSR_EL1: EL1t (kernel mode)
        ctx.pstate = 0b0100;
    }

    /// Get task state
    pub fn get_state(&self) -> TaskState {
        *self.state.lock()
    }

    /// Set task state
    pub fn set_state(&self, state: TaskState) {
        *self.state.lock() = state;
    }

    /// Block task with reason
    pub fn block(&self, reason: WaitReason) {
        self.set_state(TaskState::Blocked(reason));
    }

    /// Unblock task
    pub fn unblock(&self) {
        let mut state = self.state.lock();
        if let TaskState::Blocked(_) = *state {
            *state = TaskState::Ready;
        }
    }

    /// Send signal to this task
    pub fn send_signal(&self, signal: Signal, sender_pid: u32) {
        let mut signals = self.signals.lock();
        signals.pending.add(SignalInfo {
            signal,
            sender_pid,
            code: 0,
            value: 0,
        });

        // Wake up if blocked waiting for signal
        let mut state = self.state.lock();
        if let TaskState::Blocked(WaitReason::Signal) = *state {
            *state = TaskState::Ready;
        }
    }

    /// Check for pending signals
    pub fn has_pending_signals(&self) -> bool {
        let signals = self.signals.lock();
        signals.pending.any_pending()
    }

    /// Deliver pending signals
    pub fn deliver_signals(&self) -> Option<(Signal, SignalAction)> {
        let mut signals = self.signals.lock();

        if let Some(info) = signals.pending.next(&signals.blocked) {
            let action = signals.handlers.get(info.signal);
            return Some((info.signal, action));
        }

        None
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        // Free kernel stack
        free_kernel_stack(self.kernel_stack);
    }
}

/// Allocate a kernel stack
fn allocate_kernel_stack() -> Option<usize> {
    // Allocate pages for kernel stack
    let pages_needed = (KERNEL_STACK_SIZE + PAGE_SIZE - 1) / PAGE_SIZE;

    // Allocate contiguous frames
    let mut frames = Vec::new();
    for _ in 0..pages_needed {
        frames.push(allocate_frame()?);
    }

    // Return base address of first frame
    Some(frames[0].start_address())
}

/// Free a kernel stack
fn free_kernel_stack(addr: usize) {
    let pages_needed = (KERNEL_STACK_SIZE + PAGE_SIZE - 1) / PAGE_SIZE;

    for i in 0..pages_needed {
        let frame = PhysFrame::containing_address(addr + i * PAGE_SIZE);
        deallocate_frame(frame);
    }
}

/// Get current task
pub fn current() -> Option<Arc<Task>> {
    let id = TaskId(CURRENT_TASK.load(Ordering::Acquire));
    if id.0 == 0 {
        return None;
    }
    TASKS.read().get(&id).cloned()
}

/// Get task by ID
pub fn get(id: TaskId) -> Option<Arc<Task>> {
    TASKS.read().get(&id).cloned()
}

/// Set current task
pub fn set_current(task: &Task) {
    CURRENT_TASK.store(task.id.0, Ordering::Release);
}

/// Context switch from current task to next task
///
/// # Safety
/// Must be called with interrupts disabled
#[no_mangle]
pub unsafe fn context_switch(from: &Task, to: &Task) {
    // Save current context
    let from_ctx = from.context.lock();
    let to_ctx = to.context.lock();

    // Update task states
    from.set_state(TaskState::Ready);
    to.set_state(TaskState::Running);

    // Update current task
    CURRENT_TASK.store(to.id.0, Ordering::Release);

    // Switch address spaces if different processes
    if from.process.pid != to.process.pid {
        // Get target process page table
        let memory = to.process.memory.lock();
        if memory.page_table != 0 {
            switch_address_space(memory.page_table);
        }
    }

    // Perform actual context switch (assembly)
    context_switch_asm(
        &*from_ctx as *const CpuContext as *mut CpuContext,
        &*to_ctx as *const CpuContext,
    );
}

/// Switch address space (change TTBR0_EL1)
unsafe fn switch_address_space(page_table: u64) {
    core::arch::asm!(
        "msr ttbr0_el1, {0}",
        "isb",
        "tlbi vmalle1",
        "dsb sy",
        "isb",
        in(reg) page_table,
    );
}

/// Assembly context switch
///
/// # Safety
/// Pointers must be valid
#[naked]
unsafe extern "C" fn context_switch_asm(
    _from: *mut CpuContext,
    _to: *const CpuContext,
) {
    core::arch::asm!(
        // Save callee-saved registers to from context
        "stp x19, x20, [x0, #152]",   // x[19], x[20]
        "stp x21, x22, [x0, #168]",   // x[21], x[22]
        "stp x23, x24, [x0, #184]",   // x[23], x[24]
        "stp x25, x26, [x0, #200]",   // x[25], x[26]
        "stp x27, x28, [x0, #216]",   // x[27], x[28]
        "stp x29, x30, [x0, #232]",   // x[29] (fp), x[30] (lr)
        "mov x2, sp",
        "str x2, [x0, #248]",          // sp
        "adr x2, 1f",
        "str x2, [x0, #256]",          // pc (return address)

        // Restore callee-saved registers from to context
        "ldp x19, x20, [x1, #152]",
        "ldp x21, x22, [x1, #168]",
        "ldp x23, x24, [x1, #184]",
        "ldp x25, x26, [x1, #200]",
        "ldp x27, x28, [x1, #216]",
        "ldp x29, x30, [x1, #232]",
        "ldr x2, [x1, #248]",
        "mov sp, x2",
        "ldr x2, [x1, #256]",
        "br x2",

        "1:",
        "ret",
        options(noreturn)
    );
}

/// Register a task waiting for a child
pub fn wait_for_child(task_id: TaskId, target: WaitTarget) {
    let parent_pid = if let Some(task) = get(task_id) {
        task.process.pid
    } else {
        return;
    };

    let mut queues = WAIT_QUEUES.lock();
    let queue = queues.entry(parent_pid).or_insert_with(VecDeque::new);
    queue.push_back(WaitEntry { task_id, waiting_for: target });
}

/// Notify parent that child has exited
pub fn notify_child_exit(child_pid: Pid, parent_pid: Pid, exit_code: i32) {
    let mut queues = WAIT_QUEUES.lock();

    if let Some(queue) = queues.get_mut(&parent_pid) {
        // Find and wake any waiting tasks
        let mut i = 0;
        while i < queue.len() {
            let entry = &queue[i];
            let should_wake = match &entry.waiting_for {
                WaitTarget::AnyChild => true,
                WaitTarget::ChildPid(pid) => *pid == child_pid,
                WaitTarget::ProcessGroup(_pgid) => false, // TODO: implement
            };

            if should_wake {
                if let Some(task) = get(entry.task_id) {
                    task.unblock();
                }
                queue.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

/// Exit the current task
pub fn exit(code: i32) {
    if let Some(task) = current() {
        // Set task state to exited
        task.set_state(TaskState::Exited(code));

        // Update process state
        let process = &task.process;
        process.exit(code);

        // Notify parent
        notify_child_exit(process.pid, process.ppid, code);

        // Remove from task table
        TASKS.write().remove(&task.id);

        // Reschedule (this task can no longer run)
        crate::scheduler::schedule();
    }
}

/// Wait for a child process
pub fn wait_pid(pid: Option<Pid>, options: u32) -> Result<(Pid, i32), i32> {
    let task = current().ok_or(-1)?;
    let process = &task.process;

    // Non-blocking check first
    let children = process.children.lock();
    for &child_pid in children.iter() {
        if let Some(target_pid) = pid {
            if child_pid != target_pid {
                continue;
            }
        }

        if let Some(child) = crate::process::get(child_pid) {
            if let ProcessState::Zombie(code) = child.get_state() {
                drop(children);

                // Reap the child
                crate::process::PROCESSES.write().remove(&child_pid);
                process.children.lock().retain(|&p| p != child_pid);

                return Ok((child_pid, code));
            }
        }
    }
    drop(children);

    // WNOHANG - don't block
    const WNOHANG: u32 = 1;
    if options & WNOHANG != 0 {
        return Err(-11); // EAGAIN
    }

    // Block and wait
    let target = match pid {
        Some(p) => WaitTarget::ChildPid(p),
        None => WaitTarget::AnyChild,
    };

    wait_for_child(task.id, target);
    task.block(WaitReason::Child);

    // Yield to scheduler
    crate::scheduler::schedule();

    // When we wake up, try again
    wait_pid(pid, WNOHANG)
}

/// Fork the current process
pub fn fork() -> Result<Pid, i32> {
    let current_task = current().ok_or(-1)?;
    let parent = &current_task.process;

    // Create new process
    let child_process = Process::new(&parent.name, parent.pid);
    let child_pid = child_process.pid;

    // Copy memory space with COW
    {
        let parent_mem = parent.memory.lock();
        let mut child_mem = child_process.memory.lock();

        // Copy memory layout
        child_mem.heap_start = parent_mem.heap_start;
        child_mem.heap_end = parent_mem.heap_end;
        child_mem.stack_top = parent_mem.stack_top;
        child_mem.stack_size = parent_mem.stack_size;
        child_mem.regions = parent_mem.regions.clone();

        // Create new address space with COW pages
        if parent_mem.page_table != 0 {
            if let Some(space) = AddressSpace::new() {
                child_mem.page_table = space.root_addr() as u64;

                // Copy page tables with COW flag
                copy_page_tables_cow(
                    parent_mem.page_table as usize,
                    child_mem.page_table as usize,
                );
            }
        }
    }

    // Copy file descriptors
    {
        let parent_files = parent.files.lock();
        let mut child_files = child_process.files.lock();
        for (fd, desc) in parent_files.iter() {
            child_files.insert(*fd, desc.clone());
        }
    }

    // Add to parent's children
    parent.children.lock().push(child_pid);

    // Create task for child
    if let Some(child_task) = Task::new(child_process.clone(), &current_task.name) {
        // Copy context from parent
        {
            let parent_ctx = current_task.context.lock();
            let mut child_ctx = child_task.context.lock();
            *child_ctx = parent_ctx.clone();

            // Child returns 0 from fork
            child_ctx.x[0] = 0;
        }

        // Add child to scheduler
        crate::scheduler::add(child_pid, crate::scheduler::Priority::NORMAL);

        child_process.set_state(ProcessState::Ready);

        return Ok(child_pid);
    }

    Err(-12) // ENOMEM
}

/// Copy page tables with Copy-on-Write
fn copy_page_tables_cow(parent_root: usize, child_root: usize) {
    // This would walk the parent page tables and create COW mappings
    // For each present page:
    // 1. Mark parent page as read-only
    // 2. Create same mapping in child (read-only, COW flag)
    // 3. Increment reference count for the physical page

    crate::cow::copy_address_space(parent_root, child_root);
}

/// Execute a new program in the current process
pub fn exec(
    path: &str,
    argv: &[&str],
    envp: &[&str],
) -> Result<(), i32> {
    let task = current().ok_or(-1)?;
    let process = &task.process;

    // Read ELF file
    let data = crate::vfs::read_file(path)
        .map_err(|_| -2)?; // ENOENT

    if data.is_empty() {
        return Err(-8); // ENOEXEC
    }

    // Load ELF
    let program = crate::exec::load_elf(&data)
        .map_err(|_| -8)?; // ENOEXEC

    // Clear existing memory mappings
    {
        let mut memory = process.memory.lock();
        memory.regions.clear();

        // Create new address space
        if let Some(space) = AddressSpace::new() {
            // Free old address space
            if memory.page_table != 0 {
                // TODO: Free old page tables
            }
            memory.page_table = space.root_addr() as u64;
        }
    }

    // Load segments into memory
    load_elf_segments(&data, process)?;

    // Set up user stack with arguments
    let (sp, _argc) = setup_user_stack(
        process,
        program.stack_top,
        argv,
        envp,
    )?;

    // Update process memory info
    {
        let mut memory = process.memory.lock();
        memory.heap_start = program.brk;
        memory.heap_end = program.brk;
        memory.stack_top = program.stack_top;
    }

    // Set up task context for new program
    task.setup_user_context(program.entry_point, sp, argv.len());

    // Reset signal handlers
    {
        let mut signals = task.signals.lock();
        signals.handlers.reset_all();
    }

    crate::kinfo!(
        "exec: Process {} executing {} at 0x{:x}",
        process.pid.0, path, program.entry_point
    );

    // Don't return - switch directly to user mode
    switch_to_user_mode(&task);

    Ok(())
}

/// Load ELF segments into process memory
fn load_elf_segments(data: &[u8], process: &Process) -> Result<(), i32> {
    let header = crate::exec::parse_elf_header(data)
        .map_err(|_| -8)?;
    let phdrs = crate::exec::parse_program_headers(data, &header);

    let memory = process.memory.lock();
    let page_table_addr = memory.page_table as usize;
    drop(memory);

    for ph in &phdrs {
        if ph.seg_type != crate::exec::SegmentType::Load as u32 {
            continue;
        }

        let vaddr = ph.vaddr as usize;
        let offset = ph.offset as usize;
        let filesz = ph.filesz as usize;
        let memsz = ph.memsz as usize;
        let flags = ph.flags;

        // Calculate page flags
        let mut page_flags = PageFlags::PRESENT | PageFlags::USER;
        if flags & crate::exec::segment_flags::WRITE != 0 {
            page_flags |= PageFlags::WRITABLE;
        }
        if flags & crate::exec::segment_flags::EXECUTE == 0 {
            page_flags |= PageFlags::NO_EXECUTE;
        }

        // Allocate and map pages
        let start_page = vaddr / PAGE_SIZE;
        let end_page = (vaddr + memsz + PAGE_SIZE - 1) / PAGE_SIZE;

        for page_num in start_page..end_page {
            let page_vaddr = page_num * PAGE_SIZE;

            // Allocate physical frame
            let frame = allocate_frame().ok_or(-12)?; // ENOMEM

            // Map the page
            map_page(page_table_addr, page_vaddr, frame.start_address(), page_flags);

            // Copy data from ELF file
            let page_offset_in_segment = page_vaddr.saturating_sub(vaddr);
            let file_offset = offset + page_offset_in_segment;

            if file_offset < offset + filesz {
                let copy_size = core::cmp::min(
                    PAGE_SIZE,
                    offset + filesz - file_offset,
                );

                unsafe {
                    let dest = frame.start_address() as *mut u8;
                    let src = &data[file_offset..file_offset + copy_size];
                    core::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        dest,
                        copy_size,
                    );

                    // Zero remaining part of page
                    if copy_size < PAGE_SIZE {
                        core::ptr::write_bytes(
                            dest.add(copy_size),
                            0,
                            PAGE_SIZE - copy_size,
                        );
                    }
                }
            } else {
                // BSS - zero the page
                unsafe {
                    let dest = frame.start_address() as *mut u8;
                    core::ptr::write_bytes(dest, 0, PAGE_SIZE);
                }
            }
        }

        // Add to memory regions
        process.memory.lock().regions.push(crate::process::MemoryRegion {
            start: vaddr,
            end: vaddr + memsz,
            flags: crate::process::MemoryFlags::from_bits_truncate(page_flags.bits() as u32),
            name: String::from("[elf]"),
        });
    }

    Ok(())
}

/// Map a page in the address space
fn map_page(page_table: usize, vaddr: usize, paddr: usize, flags: PageFlags) {
    // Walk page table and create mapping
    let l0 = page_table as *mut u64;

    let l0_idx = (vaddr >> 39) & 0x1FF;
    let l1_idx = (vaddr >> 30) & 0x1FF;
    let l2_idx = (vaddr >> 21) & 0x1FF;
    let l3_idx = (vaddr >> 12) & 0x1FF;

    unsafe {
        // Get or create L1 table
        let l1 = get_or_create_table(l0, l0_idx);
        // Get or create L2 table
        let l2 = get_or_create_table(l1, l1_idx);
        // Get or create L3 table
        let l3 = get_or_create_table(l2, l2_idx);

        // Set L3 entry (final mapping)
        let entry = (paddr as u64) | flags.bits() | 0x3;
        *l3.add(l3_idx) = entry;
    }
}

/// Get or create next level table
unsafe fn get_or_create_table(table: *mut u64, index: usize) -> *mut u64 {
    let entry = *table.add(index);

    if entry & 0x1 == 0 {
        // Not present, create new table
        if let Some(frame) = allocate_frame() {
            let new_table = frame.start_address() as *mut u64;

            // Zero the new table
            for i in 0..512 {
                *new_table.add(i) = 0;
            }

            // Set entry to point to new table
            *table.add(index) = (frame.start_address() as u64) | 0x3;

            return new_table;
        }
        // Allocation failed, return null
        return core::ptr::null_mut();
    }

    // Extract address from entry
    (entry & 0x0000_FFFF_FFFF_F000) as *mut u64
}

/// Set up user stack with arguments and environment
fn setup_user_stack(
    process: &Process,
    stack_top: usize,
    argv: &[&str],
    envp: &[&str],
) -> Result<(usize, usize), i32> {
    // Allocate stack pages
    let stack_pages = USER_STACK_SIZE / PAGE_SIZE;
    let stack_base = stack_top - USER_STACK_SIZE;

    let memory = process.memory.lock();
    let page_table = memory.page_table as usize;
    drop(memory);

    for i in 0..stack_pages {
        let vaddr = stack_base + i * PAGE_SIZE;
        let frame = allocate_frame().ok_or(-12)?;

        map_page(
            page_table,
            vaddr,
            frame.start_address(),
            PageFlags::PRESENT | PageFlags::WRITABLE | PageFlags::USER | PageFlags::NO_EXECUTE,
        );

        // Zero the page
        unsafe {
            core::ptr::write_bytes(frame.start_address() as *mut u8, 0, PAGE_SIZE);
        }
    }

    // Add stack region
    process.memory.lock().regions.push(crate::process::MemoryRegion {
        start: stack_base,
        end: stack_top,
        flags: crate::process::MemoryFlags::READ | crate::process::MemoryFlags::WRITE | crate::process::MemoryFlags::USER,
        name: String::from("[stack]"),
    });

    // Calculate total size needed for strings
    let mut total_str_size = 0;
    for arg in argv {
        total_str_size += arg.len() + 1; // +1 for null terminator
    }
    for env in envp {
        total_str_size += env.len() + 1;
    }

    // Align to 16 bytes
    total_str_size = (total_str_size + 15) & !15;

    // Pointer array size (argv, NULL, envp, NULL)
    let ptr_array_size = (argv.len() + 1 + envp.len() + 1 + 1) * 8; // +1 for argc

    // Calculate final stack pointer
    let mut sp = stack_top - total_str_size - ptr_array_size;
    sp &= !15; // 16-byte alignment

    // Get physical address for stack top area (for writing)
    // Note: In a real implementation, we'd use the page table to translate
    // For now, we write directly to the physical pages

    let argc = argv.len();

    Ok((sp, argc))
}

/// Switch to user mode (never returns)
fn switch_to_user_mode(task: &Task) -> ! {
    let ctx = task.context.lock();

    unsafe {
        core::arch::asm!(
            // Set up for ERET to EL0
            "msr elr_el1, {entry}",
            "msr sp_el0, {sp}",
            "msr spsr_el1, {pstate}",

            // Set x0 (first argument)
            "mov x0, {arg0}",

            // Clear other registers for security
            "mov x1, xzr",
            "mov x2, xzr",
            "mov x3, xzr",
            "mov x4, xzr",
            "mov x5, xzr",
            "mov x6, xzr",
            "mov x7, xzr",
            "mov x8, xzr",
            "mov x9, xzr",
            "mov x10, xzr",
            "mov x11, xzr",
            "mov x12, xzr",
            "mov x13, xzr",
            "mov x14, xzr",
            "mov x15, xzr",
            "mov x16, xzr",
            "mov x17, xzr",
            "mov x18, xzr",
            "mov x19, xzr",
            "mov x20, xzr",
            "mov x21, xzr",
            "mov x22, xzr",
            "mov x23, xzr",
            "mov x24, xzr",
            "mov x25, xzr",
            "mov x26, xzr",
            "mov x27, xzr",
            "mov x28, xzr",
            "mov x29, xzr",
            "mov x30, xzr",

            // Return to user mode
            "eret",
            entry = in(reg) ctx.pc,
            sp = in(reg) ctx.sp,
            pstate = in(reg) ctx.pstate,
            arg0 = in(reg) ctx.x[0],
            options(noreturn)
        );
    }
}

/// Initialize task subsystem
pub fn init() {
    crate::kprintln!("  Task subsystem initialized");
}
