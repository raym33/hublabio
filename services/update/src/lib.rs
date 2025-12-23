//! OTA Update Service
//!
//! Provides over-the-air update functionality.

#![no_std]

extern crate alloc;

mod update;
pub use update::*;
