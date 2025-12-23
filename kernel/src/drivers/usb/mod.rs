//! USB Driver
//!
//! Universal Serial Bus support for HubLab IO.

pub mod hcd;
pub mod device;
pub mod hub;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::collections::BTreeMap;
use spin::{Mutex, RwLock};

/// USB device address counter
static NEXT_ADDRESS: Mutex<u8> = Mutex::new(1);

/// Registered USB devices
static DEVICES: RwLock<BTreeMap<u8, Arc<UsbDevice>>> = RwLock::new(BTreeMap::new());

/// USB speed
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsbSpeed {
    Low,      // 1.5 Mbps
    Full,     // 12 Mbps
    High,     // 480 Mbps
    Super,    // 5 Gbps
    SuperPlus, // 10+ Gbps
}

/// USB device class
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsbClass {
    Interface = 0x00,
    Audio = 0x01,
    Cdc = 0x02,
    Hid = 0x03,
    Physical = 0x05,
    Image = 0x06,
    Printer = 0x07,
    MassStorage = 0x08,
    Hub = 0x09,
    CdcData = 0x0A,
    SmartCard = 0x0B,
    ContentSecurity = 0x0D,
    Video = 0x0E,
    PersonalHealthcare = 0x0F,
    AudioVideo = 0x10,
    Billboard = 0x11,
    TypeCBridge = 0x12,
    Diagnostic = 0xDC,
    Wireless = 0xE0,
    Miscellaneous = 0xEF,
    Application = 0xFE,
    VendorSpecific = 0xFF,
}

impl From<u8> for UsbClass {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Interface,
            0x01 => Self::Audio,
            0x02 => Self::Cdc,
            0x03 => Self::Hid,
            0x05 => Self::Physical,
            0x06 => Self::Image,
            0x07 => Self::Printer,
            0x08 => Self::MassStorage,
            0x09 => Self::Hub,
            0x0A => Self::CdcData,
            0x0B => Self::SmartCard,
            0x0D => Self::ContentSecurity,
            0x0E => Self::Video,
            0x0F => Self::PersonalHealthcare,
            0x10 => Self::AudioVideo,
            0x11 => Self::Billboard,
            0x12 => Self::TypeCBridge,
            0xDC => Self::Diagnostic,
            0xE0 => Self::Wireless,
            0xEF => Self::Miscellaneous,
            0xFE => Self::Application,
            _ => Self::VendorSpecific,
        }
    }
}

/// USB endpoint type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointType {
    Control,
    Isochronous,
    Bulk,
    Interrupt,
}

/// USB transfer direction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Out,
    In,
}

/// USB endpoint descriptor
#[derive(Clone, Debug)]
pub struct Endpoint {
    pub address: u8,
    pub direction: Direction,
    pub ep_type: EndpointType,
    pub max_packet_size: u16,
    pub interval: u8,
}

impl Endpoint {
    pub fn number(&self) -> u8 {
        self.address & 0x0F
    }
}

/// USB interface descriptor
#[derive(Clone, Debug)]
pub struct Interface {
    pub number: u8,
    pub alternate: u8,
    pub class: UsbClass,
    pub subclass: u8,
    pub protocol: u8,
    pub endpoints: Vec<Endpoint>,
}

/// USB configuration descriptor
#[derive(Clone, Debug)]
pub struct Configuration {
    pub value: u8,
    pub max_power: u8,  // In 2mA units
    pub self_powered: bool,
    pub remote_wakeup: bool,
    pub interfaces: Vec<Interface>,
}

/// USB device descriptor
#[derive(Clone, Debug)]
pub struct DeviceDescriptor {
    pub usb_version: u16,
    pub class: UsbClass,
    pub subclass: u8,
    pub protocol: u8,
    pub max_packet_size: u8,
    pub vendor_id: u16,
    pub product_id: u16,
    pub device_version: u16,
    pub manufacturer_string: Option<String>,
    pub product_string: Option<String>,
    pub serial_string: Option<String>,
}

/// USB device state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceState {
    Attached,
    Powered,
    Default,
    Address,
    Configured,
    Suspended,
}

/// USB device
pub struct UsbDevice {
    pub address: u8,
    pub speed: UsbSpeed,
    pub state: Mutex<DeviceState>,
    pub descriptor: DeviceDescriptor,
    pub configurations: Vec<Configuration>,
    pub active_config: Mutex<Option<u8>>,
    pub port: u8,
    pub parent_hub: Option<u8>,
}

impl UsbDevice {
    /// Get device address
    pub fn address(&self) -> u8 {
        self.address
    }

    /// Get current state
    pub fn state(&self) -> DeviceState {
        *self.state.lock()
    }

    /// Get vendor ID
    pub fn vendor_id(&self) -> u16 {
        self.descriptor.vendor_id
    }

    /// Get product ID
    pub fn product_id(&self) -> u16 {
        self.descriptor.product_id
    }

    /// Get device class
    pub fn class(&self) -> UsbClass {
        self.descriptor.class
    }

    /// Set configuration
    pub fn set_configuration(&self, config: u8) -> Result<(), UsbError> {
        // Would send SET_CONFIGURATION request
        *self.active_config.lock() = Some(config);
        *self.state.lock() = DeviceState::Configured;
        Ok(())
    }
}

/// USB error types
#[derive(Clone, Debug)]
pub enum UsbError {
    NoDevice,
    DeviceNotConfigured,
    InvalidEndpoint,
    InvalidDescriptor,
    TransferError,
    Stall,
    Timeout,
    BufferOverflow,
    NotSupported,
    NoMemory,
    Busy,
}

/// USB request type
#[derive(Clone, Copy, Debug)]
pub struct RequestType {
    pub direction: Direction,
    pub req_type: RequestTypeType,
    pub recipient: Recipient,
}

#[derive(Clone, Copy, Debug)]
pub enum RequestTypeType {
    Standard,
    Class,
    Vendor,
}

#[derive(Clone, Copy, Debug)]
pub enum Recipient {
    Device,
    Interface,
    Endpoint,
    Other,
}

impl RequestType {
    pub fn to_byte(&self) -> u8 {
        let dir = match self.direction {
            Direction::Out => 0,
            Direction::In => 0x80,
        };
        let typ = match self.req_type {
            RequestTypeType::Standard => 0,
            RequestTypeType::Class => 0x20,
            RequestTypeType::Vendor => 0x40,
        };
        let rec = match self.recipient {
            Recipient::Device => 0,
            Recipient::Interface => 1,
            Recipient::Endpoint => 2,
            Recipient::Other => 3,
        };
        dir | typ | rec
    }
}

/// Standard USB requests
#[derive(Clone, Copy, Debug)]
pub enum StandardRequest {
    GetStatus = 0,
    ClearFeature = 1,
    SetFeature = 3,
    SetAddress = 5,
    GetDescriptor = 6,
    SetDescriptor = 7,
    GetConfiguration = 8,
    SetConfiguration = 9,
    GetInterface = 10,
    SetInterface = 11,
    SynchFrame = 12,
}

/// USB setup packet
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct SetupPacket {
    pub request_type: u8,
    pub request: u8,
    pub value: u16,
    pub index: u16,
    pub length: u16,
}

impl SetupPacket {
    pub fn new(request_type: RequestType, request: u8, value: u16, index: u16, length: u16) -> Self {
        Self {
            request_type: request_type.to_byte(),
            request,
            value,
            index,
            length,
        }
    }

    /// Get device descriptor
    pub fn get_device_descriptor() -> Self {
        Self::new(
            RequestType {
                direction: Direction::In,
                req_type: RequestTypeType::Standard,
                recipient: Recipient::Device,
            },
            StandardRequest::GetDescriptor as u8,
            0x0100, // Device descriptor
            0,
            18,
        )
    }

    /// Get configuration descriptor
    pub fn get_config_descriptor(index: u8, length: u16) -> Self {
        Self::new(
            RequestType {
                direction: Direction::In,
                req_type: RequestTypeType::Standard,
                recipient: Recipient::Device,
            },
            StandardRequest::GetDescriptor as u8,
            0x0200 | index as u16,
            0,
            length,
        )
    }

    /// Set address
    pub fn set_address(address: u8) -> Self {
        Self::new(
            RequestType {
                direction: Direction::Out,
                req_type: RequestTypeType::Standard,
                recipient: Recipient::Device,
            },
            StandardRequest::SetAddress as u8,
            address as u16,
            0,
            0,
        )
    }

    /// Set configuration
    pub fn set_configuration(config: u8) -> Self {
        Self::new(
            RequestType {
                direction: Direction::Out,
                req_type: RequestTypeType::Standard,
                recipient: Recipient::Device,
            },
            StandardRequest::SetConfiguration as u8,
            config as u16,
            0,
            0,
        )
    }

    /// Get string descriptor
    pub fn get_string_descriptor(index: u8, lang_id: u16, length: u16) -> Self {
        Self::new(
            RequestType {
                direction: Direction::In,
                req_type: RequestTypeType::Standard,
                recipient: Recipient::Device,
            },
            StandardRequest::GetDescriptor as u8,
            0x0300 | index as u16,
            lang_id,
            length,
        )
    }
}

/// Allocate next device address
pub fn allocate_address() -> u8 {
    let mut addr = NEXT_ADDRESS.lock();
    let current = *addr;
    *addr = addr.wrapping_add(1);
    if *addr == 0 {
        *addr = 1;
    }
    current
}

/// Register USB device
pub fn register_device(device: Arc<UsbDevice>) {
    let addr = device.address;
    DEVICES.write().insert(addr, device);
    crate::kinfo!("USB: Device {} registered", addr);
}

/// Unregister USB device
pub fn unregister_device(address: u8) {
    DEVICES.write().remove(&address);
    crate::kinfo!("USB: Device {} unregistered", address);
}

/// Get USB device
pub fn get_device(address: u8) -> Option<Arc<UsbDevice>> {
    DEVICES.read().get(&address).cloned()
}

/// List all USB devices
pub fn list_devices() -> Vec<Arc<UsbDevice>> {
    DEVICES.read().values().cloned().collect()
}

/// Initialize USB subsystem
pub fn init() {
    hcd::init();
    crate::kprintln!("  USB subsystem initialized");
}
