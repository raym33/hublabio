//! Device Tree Blob (DTB) Parser
//!
//! Parses Flattened Device Tree (FDT) to discover hardware.

use alloc::string::String;
use alloc::vec::Vec;

/// DTB header magic
const DTB_MAGIC: u32 = 0xD00DFEED;

/// DTB header
#[repr(C)]
pub struct DtbHeader {
    pub magic: u32,
    pub totalsize: u32,
    pub off_dt_struct: u32,
    pub off_dt_strings: u32,
    pub off_mem_rsvmap: u32,
    pub version: u32,
    pub last_comp_version: u32,
    pub boot_cpuid_phys: u32,
    pub size_dt_strings: u32,
    pub size_dt_struct: u32,
}

/// DTB token types
const FDT_BEGIN_NODE: u32 = 0x00000001;
const FDT_END_NODE: u32 = 0x00000002;
const FDT_PROP: u32 = 0x00000003;
const FDT_NOP: u32 = 0x00000004;
const FDT_END: u32 = 0x00000009;

/// Property descriptor
#[repr(C)]
pub struct FdtProp {
    pub len: u32,
    pub nameoff: u32,
}

/// Parsed device node
#[derive(Clone, Debug)]
pub struct DeviceNode {
    pub name: String,
    pub unit_address: Option<u64>,
    pub compatible: Option<String>,
    pub reg: Option<Vec<(u64, u64)>>, // (address, size) pairs
    pub interrupts: Option<Vec<u32>>,
    pub children: Vec<DeviceNode>,
}

impl DeviceNode {
    pub fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
            unit_address: None,
            compatible: None,
            reg: None,
            interrupts: None,
            children: Vec::new(),
        }
    }
}

/// Read big-endian u32
fn read_be_u32(data: &[u8], offset: usize) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Read big-endian u64
fn read_be_u64(data: &[u8], offset: usize) -> u64 {
    if offset + 8 > data.len() {
        return 0;
    }
    u64::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

/// Read null-terminated string
fn read_string(data: &[u8], offset: usize) -> &str {
    let mut end = offset;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }
    core::str::from_utf8(&data[offset..end]).unwrap_or("")
}

/// Align offset to 4 bytes
fn align4(offset: usize) -> usize {
    (offset + 3) & !3
}

/// Parse DTB at address
pub fn parse(dtb_addr: usize) {
    let header = unsafe { &*(dtb_addr as *const DtbHeader) };

    // Validate magic (need to swap endianness)
    let magic = u32::from_be(header.magic);
    if magic != DTB_MAGIC {
        crate::kprintln!("  DTB: Invalid magic 0x{:08X}", magic);
        return;
    }

    let total_size = u32::from_be(header.totalsize) as usize;
    let struct_offset = u32::from_be(header.off_dt_struct) as usize;
    let strings_offset = u32::from_be(header.off_dt_strings) as usize;
    let version = u32::from_be(header.version);

    crate::kprintln!("  DTB: version {}, {} bytes", version, total_size);

    // Create slice for entire DTB
    let data = unsafe { core::slice::from_raw_parts(dtb_addr as *const u8, total_size) };

    // Parse structure block
    let mut offset = struct_offset;
    let mut depth = 0;

    while offset < struct_offset + u32::from_be(header.size_dt_struct) as usize {
        let token = read_be_u32(data, offset);
        offset += 4;

        match token {
            FDT_BEGIN_NODE => {
                let name = read_string(data, offset);
                let name_len = name.len() + 1; // Include null terminator
                offset += align4(name_len);

                if depth <= 2 && !name.is_empty() {
                    let indent = "  ".repeat(depth + 2);
                    crate::kprintln!("{}Node: {}", indent, name);
                }
                depth += 1;
            }
            FDT_END_NODE => {
                depth -= 1;
            }
            FDT_PROP => {
                let len = read_be_u32(data, offset) as usize;
                let nameoff = read_be_u32(data, offset + 4) as usize;
                offset += 8;

                let prop_name = read_string(data, strings_offset + nameoff);
                let prop_data = &data[offset..offset + len];
                offset += align4(len);

                // Log interesting properties
                if depth <= 2 {
                    match prop_name {
                        "compatible" => {
                            let value = read_string(prop_data, 0);
                            crate::kprintln!("    compatible: {}", value);
                        }
                        "model" => {
                            let value = read_string(prop_data, 0);
                            crate::kprintln!("    model: {}", value);
                        }
                        _ => {}
                    }
                }
            }
            FDT_NOP => {
                // Skip
            }
            FDT_END => {
                break;
            }
            _ => {
                crate::kprintln!("  DTB: Unknown token 0x{:08X}", token);
                break;
            }
        }
    }
}

/// Find a node by compatible string
pub fn find_compatible(dtb_addr: usize, compatible: &str) -> Option<(u64, u64)> {
    let header = unsafe { &*(dtb_addr as *const DtbHeader) };

    let magic = u32::from_be(header.magic);
    if magic != DTB_MAGIC {
        return None;
    }

    let total_size = u32::from_be(header.totalsize) as usize;
    let struct_offset = u32::from_be(header.off_dt_struct) as usize;
    let strings_offset = u32::from_be(header.off_dt_strings) as usize;

    let data = unsafe { core::slice::from_raw_parts(dtb_addr as *const u8, total_size) };

    let mut offset = struct_offset;
    let mut found_compatible = false;
    let mut reg_addr: u64 = 0;
    let mut reg_size: u64 = 0;

    while offset < struct_offset + u32::from_be(header.size_dt_struct) as usize {
        let token = read_be_u32(data, offset);
        offset += 4;

        match token {
            FDT_BEGIN_NODE => {
                let name = read_string(data, offset);
                let name_len = name.len() + 1;
                offset += align4(name_len);
                found_compatible = false;
            }
            FDT_END_NODE => {
                if found_compatible && reg_addr != 0 {
                    return Some((reg_addr, reg_size));
                }
                found_compatible = false;
            }
            FDT_PROP => {
                let len = read_be_u32(data, offset) as usize;
                let nameoff = read_be_u32(data, offset + 4) as usize;
                offset += 8;

                let prop_name = read_string(data, strings_offset + nameoff);
                let prop_data = &data[offset..offset + len];
                offset += align4(len);

                if prop_name == "compatible" {
                    let value = read_string(prop_data, 0);
                    if value.contains(compatible) {
                        found_compatible = true;
                    }
                } else if prop_name == "reg" && found_compatible && len >= 16 {
                    reg_addr = read_be_u64(prop_data, 0);
                    reg_size = read_be_u64(prop_data, 8);
                }
            }
            FDT_END => break,
            _ => {}
        }
    }

    None
}
