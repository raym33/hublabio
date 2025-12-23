//! USB Hub Driver
//!
//! Support for USB hubs and port management.

use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::Mutex;

use super::{UsbDevice, UsbError, UsbClass, SetupPacket, RequestType, RequestTypeType, Direction, Recipient};
use super::hcd::{get_controller, PortStatus};
use super::device::{enumerate_device, configure_device, bind_driver};

/// Hub descriptor type
const HUB_DESCRIPTOR_TYPE: u8 = 0x29;

/// Hub class requests
mod hub_request {
    pub const GET_STATUS: u8 = 0;
    pub const CLEAR_FEATURE: u8 = 1;
    pub const SET_FEATURE: u8 = 3;
    pub const GET_DESCRIPTOR: u8 = 6;
    pub const SET_DESCRIPTOR: u8 = 7;
    pub const CLEAR_TT_BUFFER: u8 = 8;
    pub const RESET_TT: u8 = 9;
    pub const GET_TT_STATE: u8 = 10;
    pub const STOP_TT: u8 = 11;
}

/// Hub features
mod hub_feature {
    pub const C_HUB_LOCAL_POWER: u16 = 0;
    pub const C_HUB_OVER_CURRENT: u16 = 1;
}

/// Port features
mod port_feature {
    pub const CONNECTION: u16 = 0;
    pub const ENABLE: u16 = 1;
    pub const SUSPEND: u16 = 2;
    pub const OVER_CURRENT: u16 = 3;
    pub const RESET: u16 = 4;
    pub const POWER: u16 = 8;
    pub const LOW_SPEED: u16 = 9;
    pub const C_CONNECTION: u16 = 16;
    pub const C_ENABLE: u16 = 17;
    pub const C_SUSPEND: u16 = 18;
    pub const C_OVER_CURRENT: u16 = 19;
    pub const C_RESET: u16 = 20;
}

/// Hub descriptor
#[derive(Clone, Debug)]
pub struct HubDescriptor {
    pub num_ports: u8,
    pub characteristics: u16,
    pub power_on_time: u8,  // In 2ms units
    pub controller_current: u8,
    pub removable_ports: u64,
}

/// Hub port status
#[derive(Clone, Copy, Debug)]
pub struct HubPortStatus {
    pub status: u16,
    pub change: u16,
}

impl HubPortStatus {
    pub fn connected(&self) -> bool {
        self.status & (1 << 0) != 0
    }

    pub fn enabled(&self) -> bool {
        self.status & (1 << 1) != 0
    }

    pub fn suspended(&self) -> bool {
        self.status & (1 << 2) != 0
    }

    pub fn over_current(&self) -> bool {
        self.status & (1 << 3) != 0
    }

    pub fn reset(&self) -> bool {
        self.status & (1 << 4) != 0
    }

    pub fn powered(&self) -> bool {
        self.status & (1 << 8) != 0
    }

    pub fn low_speed(&self) -> bool {
        self.status & (1 << 9) != 0
    }

    pub fn high_speed(&self) -> bool {
        self.status & (1 << 10) != 0
    }

    pub fn connection_changed(&self) -> bool {
        self.change & (1 << 0) != 0
    }

    pub fn enable_changed(&self) -> bool {
        self.change & (1 << 1) != 0
    }
}

/// USB Hub instance
pub struct UsbHub {
    device: Arc<UsbDevice>,
    descriptor: HubDescriptor,
    port_status: Vec<HubPortStatus>,
}

impl UsbHub {
    /// Create new hub from device
    pub fn new(device: Arc<UsbDevice>) -> Result<Self, UsbError> {
        if device.class() != UsbClass::Hub {
            return Err(UsbError::NotSupported);
        }

        // Get hub descriptor
        let descriptor = Self::get_hub_descriptor(&device)?;

        let port_status = alloc::vec![HubPortStatus { status: 0, change: 0 }; descriptor.num_ports as usize];

        Ok(Self {
            device,
            descriptor,
            port_status,
        })
    }

    /// Get hub descriptor
    fn get_hub_descriptor(device: &UsbDevice) -> Result<HubDescriptor, UsbError> {
        let controller = get_controller().ok_or(UsbError::NoDevice)?;

        let setup = SetupPacket::new(
            RequestType {
                direction: Direction::In,
                req_type: RequestTypeType::Class,
                recipient: Recipient::Device,
            },
            hub_request::GET_DESCRIPTOR,
            (HUB_DESCRIPTOR_TYPE as u16) << 8,
            0,
            9,
        );

        let mut buf = [0u8; 9];
        controller.lock().control_transfer(device.address(), setup, &mut buf)?;

        if buf[0] < 7 || buf[1] != HUB_DESCRIPTOR_TYPE {
            return Err(UsbError::InvalidDescriptor);
        }

        Ok(HubDescriptor {
            num_ports: buf[2],
            characteristics: u16::from_le_bytes([buf[3], buf[4]]),
            power_on_time: buf[5],
            controller_current: buf[6],
            removable_ports: 0,  // Would parse from additional bytes
        })
    }

    /// Get port status
    pub fn get_port_status(&mut self, port: u8) -> Result<HubPortStatus, UsbError> {
        if port == 0 || port > self.descriptor.num_ports {
            return Err(UsbError::InvalidEndpoint);
        }

        let controller = get_controller().ok_or(UsbError::NoDevice)?;

        let setup = SetupPacket::new(
            RequestType {
                direction: Direction::In,
                req_type: RequestTypeType::Class,
                recipient: Recipient::Other,
            },
            hub_request::GET_STATUS,
            0,
            port as u16,
            4,
        );

        let mut buf = [0u8; 4];
        controller.lock().control_transfer(self.device.address(), setup, &mut buf)?;

        let status = HubPortStatus {
            status: u16::from_le_bytes([buf[0], buf[1]]),
            change: u16::from_le_bytes([buf[2], buf[3]]),
        };

        self.port_status[(port - 1) as usize] = status;

        Ok(status)
    }

    /// Set port feature
    pub fn set_port_feature(&mut self, port: u8, feature: u16) -> Result<(), UsbError> {
        if port == 0 || port > self.descriptor.num_ports {
            return Err(UsbError::InvalidEndpoint);
        }

        let controller = get_controller().ok_or(UsbError::NoDevice)?;

        let setup = SetupPacket::new(
            RequestType {
                direction: Direction::Out,
                req_type: RequestTypeType::Class,
                recipient: Recipient::Other,
            },
            hub_request::SET_FEATURE,
            feature,
            port as u16,
            0,
        );

        controller.lock().control_transfer(self.device.address(), setup, &mut [])?;

        Ok(())
    }

    /// Clear port feature
    pub fn clear_port_feature(&mut self, port: u8, feature: u16) -> Result<(), UsbError> {
        if port == 0 || port > self.descriptor.num_ports {
            return Err(UsbError::InvalidEndpoint);
        }

        let controller = get_controller().ok_or(UsbError::NoDevice)?;

        let setup = SetupPacket::new(
            RequestType {
                direction: Direction::Out,
                req_type: RequestTypeType::Class,
                recipient: Recipient::Other,
            },
            hub_request::CLEAR_FEATURE,
            feature,
            port as u16,
            0,
        );

        controller.lock().control_transfer(self.device.address(), setup, &mut [])?;

        Ok(())
    }

    /// Power on port
    pub fn power_on_port(&mut self, port: u8) -> Result<(), UsbError> {
        self.set_port_feature(port, port_feature::POWER)?;

        // Wait for power stable (in 2ms units)
        let delay = (self.descriptor.power_on_time as u32) * 2 * 1000;
        for _ in 0..delay {
            core::hint::spin_loop();
        }

        Ok(())
    }

    /// Reset port
    pub fn reset_port(&mut self, port: u8) -> Result<(), UsbError> {
        self.set_port_feature(port, port_feature::RESET)?;

        // Wait for reset complete (10-20ms)
        for _ in 0..50000 {
            core::hint::spin_loop();
        }

        // Clear reset change
        self.clear_port_feature(port, port_feature::C_RESET)?;

        Ok(())
    }

    /// Number of ports
    pub fn num_ports(&self) -> u8 {
        self.descriptor.num_ports
    }

    /// Initialize hub
    pub fn init(&mut self) -> Result<(), UsbError> {
        crate::kinfo!("USB Hub: {} ports", self.descriptor.num_ports);

        // Power on all ports
        for port in 1..=self.descriptor.num_ports {
            self.power_on_port(port)?;
        }

        // Check each port for devices
        for port in 1..=self.descriptor.num_ports {
            let status = self.get_port_status(port)?;

            if status.connected() {
                crate::kinfo!("USB Hub: Device on port {}", port);
                self.reset_port(port)?;

                // Would enumerate downstream device here
            }
        }

        Ok(())
    }

    /// Poll for port changes
    pub fn poll(&mut self) -> Result<Vec<u8>, UsbError> {
        let mut changed_ports = Vec::new();

        for port in 1..=self.descriptor.num_ports {
            let status = self.get_port_status(port)?;

            if status.connection_changed() {
                changed_ports.push(port);
                self.clear_port_feature(port, port_feature::C_CONNECTION)?;
            }
        }

        Ok(changed_ports)
    }

    /// Handle port change
    pub fn handle_port_change(&mut self, port: u8) -> Result<(), UsbError> {
        let status = self.get_port_status(port)?;

        if status.connected() && !status.enabled() {
            // New device connected
            crate::kinfo!("USB Hub: New device on port {}", port);
            self.reset_port(port)?;
            // Would enumerate device
        } else if !status.connected() {
            // Device disconnected
            crate::kinfo!("USB Hub: Device removed from port {}", port);
            // Would clean up device
        }

        Ok(())
    }
}

/// Global hub registry
static HUBS: Mutex<Vec<Arc<Mutex<UsbHub>>>> = Mutex::new(Vec::new());

/// Register a hub
pub fn register_hub(hub: Arc<Mutex<UsbHub>>) {
    HUBS.lock().push(hub);
}

/// Get all registered hubs
pub fn list_hubs() -> Vec<Arc<Mutex<UsbHub>>> {
    HUBS.lock().clone()
}

/// Poll all hubs for changes
pub fn poll_all() {
    let hubs = list_hubs();

    for hub in hubs {
        if let Ok(changed) = hub.lock().poll() {
            for port in changed {
                let _ = hub.lock().handle_port_change(port);
            }
        }
    }
}

/// Initialize hub driver
pub fn init() {
    crate::kprintln!("  USB Hub driver initialized");
}
