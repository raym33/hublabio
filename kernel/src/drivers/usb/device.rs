//! USB Device Enumeration
//!
//! Device detection, configuration, and driver binding.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::hcd::{get_controller, PortStatus};
use super::{
    allocate_address, register_device, Configuration, DeviceDescriptor, DeviceState, Direction,
    Endpoint, EndpointType, Interface, SetupPacket, UsbClass, UsbDevice, UsbError, UsbSpeed,
};

/// Descriptor types
mod descriptor_type {
    pub const DEVICE: u8 = 1;
    pub const CONFIGURATION: u8 = 2;
    pub const STRING: u8 = 3;
    pub const INTERFACE: u8 = 4;
    pub const ENDPOINT: u8 = 5;
    pub const DEVICE_QUALIFIER: u8 = 6;
    pub const OTHER_SPEED: u8 = 7;
    pub const INTERFACE_POWER: u8 = 8;
    pub const OTG: u8 = 9;
    pub const DEBUG: u8 = 10;
    pub const INTERFACE_ASSOCIATION: u8 = 11;
    pub const HID: u8 = 33;
    pub const HID_REPORT: u8 = 34;
}

/// Parse device descriptor
fn parse_device_descriptor(data: &[u8]) -> Result<DeviceDescriptor, UsbError> {
    if data.len() < 18 {
        return Err(UsbError::InvalidDescriptor);
    }

    if data[0] != 18 || data[1] != descriptor_type::DEVICE {
        return Err(UsbError::InvalidDescriptor);
    }

    Ok(DeviceDescriptor {
        usb_version: u16::from_le_bytes([data[2], data[3]]),
        class: UsbClass::from(data[4]),
        subclass: data[5],
        protocol: data[6],
        max_packet_size: data[7],
        vendor_id: u16::from_le_bytes([data[8], data[9]]),
        product_id: u16::from_le_bytes([data[10], data[11]]),
        device_version: u16::from_le_bytes([data[12], data[13]]),
        manufacturer_string: None,
        product_string: None,
        serial_string: None,
    })
}

/// Parse configuration descriptor
fn parse_configuration(data: &[u8]) -> Result<Configuration, UsbError> {
    if data.len() < 9 {
        return Err(UsbError::InvalidDescriptor);
    }

    if data[1] != descriptor_type::CONFIGURATION {
        return Err(UsbError::InvalidDescriptor);
    }

    let total_length = u16::from_le_bytes([data[2], data[3]]) as usize;
    let num_interfaces = data[4];
    let config_value = data[5];
    let attributes = data[7];
    let max_power = data[8];

    let mut interfaces = Vec::new();
    let mut pos = 9;

    // Parse interfaces and endpoints
    while pos < total_length && pos < data.len() {
        let desc_len = data[pos] as usize;
        if desc_len == 0 || pos + desc_len > data.len() {
            break;
        }

        let desc_type = data[pos + 1];

        match desc_type {
            descriptor_type::INTERFACE => {
                if desc_len >= 9 {
                    interfaces.push(Interface {
                        number: data[pos + 2],
                        alternate: data[pos + 3],
                        class: UsbClass::from(data[pos + 5]),
                        subclass: data[pos + 6],
                        protocol: data[pos + 7],
                        endpoints: Vec::new(),
                    });
                }
            }
            descriptor_type::ENDPOINT => {
                if desc_len >= 7 {
                    let ep_addr = data[pos + 2];
                    let ep_attrs = data[pos + 3];

                    let endpoint = Endpoint {
                        address: ep_addr,
                        direction: if ep_addr & 0x80 != 0 {
                            Direction::In
                        } else {
                            Direction::Out
                        },
                        ep_type: match ep_attrs & 0x03 {
                            0 => EndpointType::Control,
                            1 => EndpointType::Isochronous,
                            2 => EndpointType::Bulk,
                            3 => EndpointType::Interrupt,
                            _ => EndpointType::Bulk,
                        },
                        max_packet_size: u16::from_le_bytes([data[pos + 4], data[pos + 5]]),
                        interval: data[pos + 6],
                    };

                    if let Some(iface) = interfaces.last_mut() {
                        iface.endpoints.push(endpoint);
                    }
                }
            }
            _ => {}
        }

        pos += desc_len;
    }

    Ok(Configuration {
        value: config_value,
        max_power,
        self_powered: (attributes & 0x40) != 0,
        remote_wakeup: (attributes & 0x20) != 0,
        interfaces,
    })
}

/// Get string descriptor
fn get_string(device_addr: u8, index: u8) -> Result<String, UsbError> {
    let controller = get_controller().ok_or(UsbError::NoDevice)?;
    let mut controller = controller.lock();

    // First get the string descriptor length
    let mut buf = [0u8; 4];
    let setup = SetupPacket::get_string_descriptor(index, 0x0409, 4);
    controller.control_transfer(device_addr, setup, &mut buf)?;

    let length = buf[0] as usize;
    if length < 2 {
        return Ok(String::new());
    }

    // Get full string descriptor
    let mut buf = alloc::vec![0u8; length];
    let setup = SetupPacket::get_string_descriptor(index, 0x0409, length as u16);
    controller.control_transfer(device_addr, setup, &mut buf)?;

    // Convert UTF-16LE to String
    let mut s = String::new();
    for i in (2..buf.len()).step_by(2) {
        if i + 1 < buf.len() {
            let c = u16::from_le_bytes([buf[i], buf[i + 1]]);
            if let Some(ch) = char::from_u32(c as u32) {
                s.push(ch);
            }
        }
    }

    Ok(s)
}

/// Enumerate new device on port
pub fn enumerate_device(port: u8) -> Result<Arc<UsbDevice>, UsbError> {
    let controller = get_controller().ok_or(UsbError::NoDevice)?;

    // Reset port and get speed
    let speed = controller.lock().reset_port(port)?;

    crate::kinfo!("USB: Device detected on port {} at {:?} speed", port, speed);

    // Allocate address
    let address = allocate_address();

    // Get device descriptor (first 8 bytes for max packet size)
    let mut buf = [0u8; 18];
    let setup = SetupPacket::get_device_descriptor();
    controller.lock().control_transfer(0, setup, &mut buf)?;

    // Parse device descriptor
    let descriptor = parse_device_descriptor(&buf)?;

    crate::kinfo!(
        "USB: Device {:04x}:{:04x} class={:?}",
        descriptor.vendor_id,
        descriptor.product_id,
        descriptor.class
    );

    // Set device address
    let setup = SetupPacket::set_address(address);
    controller.lock().control_transfer(0, setup, &mut [])?;

    // Wait for address change
    for _ in 0..10000 {
        core::hint::spin_loop();
    }

    // Get full device descriptor
    let setup = SetupPacket::get_device_descriptor();
    controller
        .lock()
        .control_transfer(address, setup, &mut buf)?;

    let descriptor = parse_device_descriptor(&buf)?;

    // Get configuration descriptor
    let mut config_buf = [0u8; 9];
    let setup = SetupPacket::get_config_descriptor(0, 9);
    controller
        .lock()
        .control_transfer(address, setup, &mut config_buf)?;

    let total_length = u16::from_le_bytes([config_buf[2], config_buf[3]]) as usize;
    let mut config_full = alloc::vec![0u8; total_length];
    let setup = SetupPacket::get_config_descriptor(0, total_length as u16);
    controller
        .lock()
        .control_transfer(address, setup, &mut config_full)?;

    let configuration = parse_configuration(&config_full)?;

    // Create device
    let device = Arc::new(UsbDevice {
        address,
        speed,
        state: spin::Mutex::new(DeviceState::Address),
        descriptor,
        configurations: alloc::vec![configuration],
        active_config: spin::Mutex::new(None),
        port,
        parent_hub: None,
    });

    // Register device
    register_device(device.clone());

    Ok(device)
}

/// Configure device with first configuration
pub fn configure_device(device: &UsbDevice) -> Result<(), UsbError> {
    if device.configurations.is_empty() {
        return Err(UsbError::InvalidDescriptor);
    }

    let config_value = device.configurations[0].value;

    let controller = get_controller().ok_or(UsbError::NoDevice)?;
    let setup = SetupPacket::set_configuration(config_value);
    controller
        .lock()
        .control_transfer(device.address, setup, &mut [])?;

    device.set_configuration(config_value)?;

    crate::kinfo!(
        "USB: Device {} configured with config {}",
        device.address,
        config_value
    );

    Ok(())
}

/// Find and bind driver for device
pub fn bind_driver(device: &UsbDevice) -> Result<(), UsbError> {
    match device.class() {
        UsbClass::Hub => {
            crate::kinfo!("USB: Hub device detected");
            // Would initialize hub driver
        }
        UsbClass::MassStorage => {
            crate::kinfo!("USB: Mass storage device detected");
            // Would initialize MSC driver
        }
        UsbClass::Hid => {
            crate::kinfo!("USB: HID device detected");
            // Would initialize HID driver
        }
        UsbClass::Cdc => {
            crate::kinfo!("USB: CDC device detected (serial)");
            // Would initialize CDC-ACM driver
        }
        UsbClass::Audio => {
            crate::kinfo!("USB: Audio device detected");
            // Would initialize audio driver
        }
        UsbClass::Video => {
            crate::kinfo!("USB: Video device detected");
            // Would initialize video driver
        }
        UsbClass::Wireless => {
            crate::kinfo!("USB: Wireless device detected");
            // Would initialize wireless driver (Bluetooth, etc.)
        }
        _ => {
            crate::kwarn!("USB: No driver for class {:?}", device.class());
        }
    }

    Ok(())
}

/// Process port change event
pub fn process_port_event(port: u8) -> Result<(), UsbError> {
    let controller = get_controller().ok_or(UsbError::NoDevice)?;
    let status = controller.lock().port_status(port);

    if status.connected && !status.enabled {
        // New device connected
        match enumerate_device(port) {
            Ok(device) => {
                configure_device(&device)?;
                bind_driver(&device)?;
            }
            Err(e) => {
                crate::kerror!("USB: Failed to enumerate device on port {}: {:?}", port, e);
            }
        }
    } else if !status.connected {
        // Device disconnected
        crate::kinfo!("USB: Device disconnected from port {}", port);
        // Would clean up device
    }

    Ok(())
}

/// Scan for USB devices
pub fn scan() -> Vec<Arc<UsbDevice>> {
    let controller = match get_controller() {
        Some(c) => c,
        None => return Vec::new(),
    };

    let controller = controller.lock();
    let num_ports = controller.root_hub_ports();
    drop(controller);

    let mut devices = Vec::new();

    for port in 0..num_ports {
        let controller = match get_controller() {
            Some(c) => c,
            None => continue,
        };

        let status = controller.lock().port_status(port);

        if status.connected {
            match enumerate_device(port) {
                Ok(device) => {
                    let _ = configure_device(&device);
                    let _ = bind_driver(&device);
                    devices.push(device);
                }
                Err(e) => {
                    crate::kerror!("USB: Scan failed on port {}: {:?}", port, e);
                }
            }
        }
    }

    devices
}
