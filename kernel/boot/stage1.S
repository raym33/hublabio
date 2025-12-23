// =============================================================================
// HUBLABIO BOOTLOADER - STAGE 1
// =============================================================================
// Assembly bootstrap for ARM64 (AArch64)
// Initializes the CPU, sets up initial stack, and jumps to Rust stage 2
// =============================================================================

.section .text.boot
.global _start

// =============================================================================
// Entry Point
// =============================================================================

_start:
    // Check processor ID, only proceed on core 0
    mrs     x0, mpidr_el1
    and     x0, x0, #3
    cbnz    x0, halt

    // Set stack pointer (4KB below kernel base)
    ldr     x0, =_stack_top
    mov     sp, x0

    // Clear BSS section
    ldr     x0, =__bss_start
    ldr     x1, =__bss_end
    sub     x1, x1, x0
    bl      memzero

    // Initialize MMU
    bl      mmu_init

    // Enable FPU/SIMD
    bl      fpu_enable

    // Jump to Rust stage 2
    bl      stage2_main

    // Should never return, but halt if it does
    b       halt

// =============================================================================
// Memory Zero
// =============================================================================
// x0 = start address
// x1 = size in bytes

memzero:
    cbz     x1, memzero_done
memzero_loop:
    str     xzr, [x0], #8
    subs    x1, x1, #8
    b.gt    memzero_loop
memzero_done:
    ret

// =============================================================================
// MMU Initialization
// =============================================================================

mmu_init:
    // Disable MMU first
    mrs     x0, sctlr_el1
    bic     x0, x0, #1          // Clear M bit
    msr     sctlr_el1, x0
    isb

    // Set up translation table base
    ldr     x0, =_page_tables
    msr     ttbr0_el1, x0
    msr     ttbr1_el1, x0

    // Configure translation control
    ldr     x0, =0x35183518     // 48-bit VA, 4KB granule
    msr     tcr_el1, x0

    // Configure memory attributes
    ldr     x0, =0xff440c0400   // Normal memory, device memory
    msr     mair_el1, x0

    // Enable MMU
    mrs     x0, sctlr_el1
    orr     x0, x0, #1          // Set M bit
    orr     x0, x0, #(1 << 2)   // Enable data cache
    orr     x0, x0, #(1 << 12)  // Enable instruction cache
    msr     sctlr_el1, x0
    isb

    ret

// =============================================================================
// FPU/SIMD Enable
// =============================================================================

fpu_enable:
    // Enable FPU/SIMD in CPACR_EL1
    mrs     x0, cpacr_el1
    orr     x0, x0, #(3 << 20)  // FPEN = 11
    msr     cpacr_el1, x0
    isb
    ret

// =============================================================================
// Halt (park secondary cores)
// =============================================================================

halt:
    wfe
    b       halt

// =============================================================================
// Exception Vector Table
// =============================================================================

.align 11
.global exception_vector_table
exception_vector_table:
    // Current EL with SP0
    .align 7
    b       sync_handler
    .align 7
    b       irq_handler
    .align 7
    b       fiq_handler
    .align 7
    b       serror_handler

    // Current EL with SPx
    .align 7
    b       sync_handler
    .align 7
    b       irq_handler
    .align 7
    b       fiq_handler
    .align 7
    b       serror_handler

    // Lower EL using AArch64
    .align 7
    b       sync_handler
    .align 7
    b       irq_handler
    .align 7
    b       fiq_handler
    .align 7
    b       serror_handler

    // Lower EL using AArch32
    .align 7
    b       sync_handler
    .align 7
    b       irq_handler
    .align 7
    b       fiq_handler
    .align 7
    b       serror_handler

sync_handler:
    b       exception_entry
irq_handler:
    b       exception_entry
fiq_handler:
    b       exception_entry
serror_handler:
    b       exception_entry

exception_entry:
    // Save context
    stp     x0, x1, [sp, #-16]!
    stp     x2, x3, [sp, #-16]!
    stp     x4, x5, [sp, #-16]!
    stp     x6, x7, [sp, #-16]!
    stp     x8, x9, [sp, #-16]!
    stp     x10, x11, [sp, #-16]!
    stp     x12, x13, [sp, #-16]!
    stp     x14, x15, [sp, #-16]!
    stp     x16, x17, [sp, #-16]!
    stp     x18, x19, [sp, #-16]!
    stp     x20, x21, [sp, #-16]!
    stp     x22, x23, [sp, #-16]!
    stp     x24, x25, [sp, #-16]!
    stp     x26, x27, [sp, #-16]!
    stp     x28, x29, [sp, #-16]!
    str     x30, [sp, #-8]!

    // Call Rust exception handler
    mov     x0, sp
    bl      handle_exception

    // Restore context
    ldr     x30, [sp], #8
    ldp     x28, x29, [sp], #16
    ldp     x26, x27, [sp], #16
    ldp     x24, x25, [sp], #16
    ldp     x22, x23, [sp], #16
    ldp     x20, x21, [sp], #16
    ldp     x18, x19, [sp], #16
    ldp     x16, x17, [sp], #16
    ldp     x14, x15, [sp], #16
    ldp     x12, x13, [sp], #16
    ldp     x10, x11, [sp], #16
    ldp     x8, x9, [sp], #16
    ldp     x6, x7, [sp], #16
    ldp     x4, x5, [sp], #16
    ldp     x2, x3, [sp], #16
    ldp     x0, x1, [sp], #16

    eret

// =============================================================================
// Data Section
// =============================================================================

.section .data
.align 12
_page_tables:
    .space 4096 * 4     // 4 pages for initial page tables

.section .bss
.align 16
_stack_bottom:
    .space 16384        // 16KB stack
_stack_top:
