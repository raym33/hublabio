//! Context Switching
//!
//! Low-level CPU context save/restore for task switching.

use super::CpuContext;

/// Save the current CPU context
///
/// # Safety
/// This function must be called with interrupts disabled
#[naked]
#[no_mangle]
pub unsafe extern "C" fn context_save(ctx: *mut CpuContext) {
    core::arch::asm!(
        // Save general purpose registers x0-x30
        "stp x0, x1, [x0, #0]",
        "stp x2, x3, [x0, #16]",
        "stp x4, x5, [x0, #32]",
        "stp x6, x7, [x0, #48]",
        "stp x8, x9, [x0, #64]",
        "stp x10, x11, [x0, #80]",
        "stp x12, x13, [x0, #96]",
        "stp x14, x15, [x0, #112]",
        "stp x16, x17, [x0, #128]",
        "stp x18, x19, [x0, #144]",
        "stp x20, x21, [x0, #160]",
        "stp x22, x23, [x0, #176]",
        "stp x24, x25, [x0, #192]",
        "stp x26, x27, [x0, #208]",
        "stp x28, x29, [x0, #224]",
        "str x30, [x0, #240]",
        // Save SP
        "mov x1, sp",
        "str x1, [x0, #248]",
        // Save PC (return address)
        "str x30, [x0, #256]",
        // Save PSTATE (SPSR_EL1)
        "mrs x1, spsr_el1",
        "str x1, [x0, #264]",
        // Save TPIDR_EL0
        "mrs x1, tpidr_el0",
        "str x1, [x0, #272]",
        "ret",
        options(noreturn)
    );
}

/// Restore CPU context and switch to it
///
/// # Safety
/// This function never returns - it jumps to the saved context
#[naked]
#[no_mangle]
pub unsafe extern "C" fn context_restore(ctx: *const CpuContext) -> ! {
    core::arch::asm!(
        // Restore TPIDR_EL0
        "ldr x1, [x0, #272]",
        "msr tpidr_el0, x1",
        // Restore PSTATE
        "ldr x1, [x0, #264]",
        "msr spsr_el1, x1",
        // Restore PC to ELR_EL1
        "ldr x1, [x0, #256]",
        "msr elr_el1, x1",
        // Restore SP
        "ldr x1, [x0, #248]",
        "mov sp, x1",
        // Restore general purpose registers
        "ldp x28, x29, [x0, #224]",
        "ldr x30, [x0, #240]",
        "ldp x26, x27, [x0, #208]",
        "ldp x24, x25, [x0, #192]",
        "ldp x22, x23, [x0, #176]",
        "ldp x20, x21, [x0, #160]",
        "ldp x18, x19, [x0, #144]",
        "ldp x16, x17, [x0, #128]",
        "ldp x14, x15, [x0, #112]",
        "ldp x12, x13, [x0, #96]",
        "ldp x10, x11, [x0, #80]",
        "ldp x8, x9, [x0, #64]",
        "ldp x6, x7, [x0, #48]",
        "ldp x4, x5, [x0, #32]",
        "ldp x2, x3, [x0, #16]",
        "ldp x0, x1, [x0, #0]",
        // Return from exception
        "eret",
        options(noreturn)
    );
}

/// Switch from one context to another
///
/// # Safety
/// Both contexts must be valid
#[no_mangle]
pub unsafe extern "C" fn context_switch(from: *mut CpuContext, to: *const CpuContext) {
    // Save current context
    context_save_inline(from);

    // Restore new context
    context_restore(to);
}

/// Inline version of context save for switch
#[inline(always)]
unsafe fn context_save_inline(ctx: *mut CpuContext) {
    core::arch::asm!(
        "stp x0, x1, [{ctx}, #0]",
        "stp x2, x3, [{ctx}, #16]",
        "stp x4, x5, [{ctx}, #32]",
        "stp x6, x7, [{ctx}, #48]",
        "stp x8, x9, [{ctx}, #64]",
        "stp x10, x11, [{ctx}, #80]",
        "stp x12, x13, [{ctx}, #96]",
        "stp x14, x15, [{ctx}, #112]",
        "stp x16, x17, [{ctx}, #128]",
        "stp x18, x19, [{ctx}, #144]",
        "stp x20, x21, [{ctx}, #160]",
        "stp x22, x23, [{ctx}, #176]",
        "stp x24, x25, [{ctx}, #192]",
        "stp x26, x27, [{ctx}, #208]",
        "stp x28, x29, [{ctx}, #224]",
        "str x30, [{ctx}, #240]",
        "mov x1, sp",
        "str x1, [{ctx}, #248]",
        "adr x1, 1f",
        "str x1, [{ctx}, #256]",
        "mrs x1, spsr_el1",
        "str x1, [{ctx}, #264]",
        "mrs x1, tpidr_el0",
        "str x1, [{ctx}, #272]",
        "1:",
        ctx = in(reg) ctx,
        out("x1") _,
        options(nostack)
    );
}

/// Initialize a new context for a thread
pub fn init_context(ctx: &mut CpuContext, entry: usize, stack: usize, arg: usize) {
    // Clear all registers
    *ctx = CpuContext::default();

    // Set entry point
    ctx.pc = entry as u64;

    // Set stack pointer
    ctx.sp = stack as u64;

    // Set first argument (x0)
    ctx.x[0] = arg as u64;

    // Set PSTATE for EL0 (user mode)
    // SPSR_EL1: M[3:0] = 0 (EL0t), DAIF cleared
    ctx.pstate = 0;
}

/// Initialize a kernel thread context
pub fn init_kernel_context(ctx: &mut CpuContext, entry: usize, stack: usize, arg: usize) {
    // Clear all registers
    *ctx = CpuContext::default();

    // Set entry point
    ctx.pc = entry as u64;

    // Set stack pointer
    ctx.sp = stack as u64;

    // Set first argument (x0)
    ctx.x[0] = arg as u64;

    // Set PSTATE for EL1 (kernel mode)
    // SPSR_EL1: M[3:0] = 4 (EL1t), interrupts enabled
    ctx.pstate = 0b0100;
}
