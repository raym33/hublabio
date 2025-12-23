//! USB Host Controller Driver
//!
//! Supports XHCI (USB 3.0), DWCI (Raspberry Pi), and generic controllers.

use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::Mutex;

use super::{UsbSpeed, UsbError, SetupPacket, UsbDevice};

/// Host controller types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerType {
    /// USB 1.1 Open Host Controller
    Ohci,
    /// USB 2.0 Enhanced Host Controller
    Ehci,
    /// USB 3.0 Extensible Host Controller
    Xhci,
    /// DesignWare Core USB Controller (Raspberry Pi)
    Dwc2,
}

/// Transfer status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferStatus {
    Pending,
    InProgress,
    Completed,
    Error,
    Stalled,
    Cancelled,
}

/// USB transfer
pub struct Transfer {
    pub device_address: u8,
    pub endpoint: u8,
    pub data: Vec<u8>,
    pub status: TransferStatus,
    pub actual_length: usize,
}

impl Transfer {
    pub fn new(device_address: u8, endpoint: u8, length: usize) -> Self {
        Self {
            device_address,
            endpoint,
            data: alloc::vec![0u8; length],
            status: TransferStatus::Pending,
            actual_length: 0,
        }
    }
}

/// USB host controller interface
pub trait HostController: Send + Sync {
    /// Get controller type
    fn controller_type(&self) -> ControllerType;

    /// Initialize the controller
    fn init(&mut self) -> Result<(), UsbError>;

    /// Reset the controller
    fn reset(&mut self) -> Result<(), UsbError>;

    /// Start the controller
    fn start(&mut self) -> Result<(), UsbError>;

    /// Stop the controller
    fn stop(&mut self) -> Result<(), UsbError>;

    /// Get number of root hub ports
    fn root_hub_ports(&self) -> u8;

    /// Get port status
    fn port_status(&self, port: u8) -> PortStatus;

    /// Reset port
    fn reset_port(&mut self, port: u8) -> Result<UsbSpeed, UsbError>;

    /// Enable port
    fn enable_port(&mut self, port: u8) -> Result<(), UsbError>;

    /// Disable port
    fn disable_port(&mut self, port: u8) -> Result<(), UsbError>;

    /// Control transfer
    fn control_transfer(
        &mut self,
        device: u8,
        setup: SetupPacket,
        data: &mut [u8],
    ) -> Result<usize, UsbError>;

    /// Bulk transfer
    fn bulk_transfer(
        &mut self,
        device: u8,
        endpoint: u8,
        data: &mut [u8],
        is_in: bool,
    ) -> Result<usize, UsbError>;

    /// Interrupt transfer
    fn interrupt_transfer(
        &mut self,
        device: u8,
        endpoint: u8,
        data: &mut [u8],
        is_in: bool,
    ) -> Result<usize, UsbError>;
}

/// Port status
#[derive(Clone, Copy, Debug)]
pub struct PortStatus {
    pub connected: bool,
    pub enabled: bool,
    pub suspended: bool,
    pub overcurrent: bool,
    pub reset: bool,
    pub power: bool,
    pub speed: UsbSpeed,
    pub changed: bool,
}

impl Default for PortStatus {
    fn default() -> Self {
        Self {
            connected: false,
            enabled: false,
            suspended: false,
            overcurrent: false,
            reset: false,
            power: false,
            speed: UsbSpeed::Full,
            changed: false,
        }
    }
}

/// DesignWare USB 2.0 Controller (Raspberry Pi)
pub struct Dwc2Controller {
    base: usize,
    ports: u8,
    initialized: bool,
}

impl Dwc2Controller {
    /// DWC2 base addresses for different platforms
    pub const BCM2837_BASE: usize = 0x3F980000;  // Pi 3
    pub const BCM2711_BASE: usize = 0xFE980000;  // Pi 4
    pub const BCM2712_BASE: usize = 0x1000480000; // Pi 5

    /// DWC2 registers
    const GOTGCTL: usize = 0x000;
    const GOTGINT: usize = 0x004;
    const GAHBCFG: usize = 0x008;
    const GUSBCFG: usize = 0x00C;
    const GRSTCTL: usize = 0x010;
    const GINTSTS: usize = 0x014;
    const GINTMSK: usize = 0x018;
    const HCFG: usize = 0x400;
    const HPRT: usize = 0x440;

    /// Create new DWC2 controller
    pub fn new(base: usize) -> Self {
        Self {
            base,
            ports: 1,
            initialized: false,
        }
    }

    fn read_reg(&self, offset: usize) -> u32 {
        unsafe { core::ptr::read_volatile((self.base + offset) as *const u32) }
    }

    fn write_reg(&mut self, offset: usize, value: u32) {
        unsafe { core::ptr::write_volatile((self.base + offset) as *mut u32, value) }
    }

    fn core_reset(&mut self) -> Result<(), UsbError> {
        // Wait for AHB idle
        let mut timeout = 100000;
        while self.read_reg(Self::GRSTCTL) & (1 << 31) == 0 {
            timeout -= 1;
            if timeout == 0 {
                return Err(UsbError::Timeout);
            }
        }

        // Core soft reset
        self.write_reg(Self::GRSTCTL, 1);

        timeout = 100000;
        while self.read_reg(Self::GRSTCTL) & 1 != 0 {
            timeout -= 1;
            if timeout == 0 {
                return Err(UsbError::Timeout);
            }
        }

        Ok(())
    }
}

impl HostController for Dwc2Controller {
    fn controller_type(&self) -> ControllerType {
        ControllerType::Dwc2
    }

    fn init(&mut self) -> Result<(), UsbError> {
        crate::kdebug!("DWC2: Initializing controller at 0x{:x}", self.base);

        // Perform core reset
        self.core_reset()?;

        // Configure PHY
        let mut gusbcfg = self.read_reg(Self::GUSBCFG);
        gusbcfg |= 1 << 6;  // PHY interface (internal)
        self.write_reg(Self::GUSBCFG, gusbcfg);

        // Configure AHB
        let gahbcfg = 1 | (1 << 5);  // Enable DMA, burst length 4
        self.write_reg(Self::GAHBCFG, gahbcfg);

        // Host mode
        gusbcfg = self.read_reg(Self::GUSBCFG);
        gusbcfg |= 1 << 29;  // Force host mode
        self.write_reg(Self::GUSBCFG, gusbcfg);

        // Wait for host mode
        for _ in 0..10000 {
            core::hint::spin_loop();
        }

        // Configure host
        self.write_reg(Self::HCFG, 1);  // FS/LS PHY clock

        self.initialized = true;
        crate::kinfo!("DWC2: Controller initialized");

        Ok(())
    }

    fn reset(&mut self) -> Result<(), UsbError> {
        self.core_reset()
    }

    fn start(&mut self) -> Result<(), UsbError> {
        // Enable port power
        let hprt = self.read_reg(Self::HPRT);
        self.write_reg(Self::HPRT, hprt | (1 << 12));

        // Enable interrupts
        self.write_reg(Self::GINTMSK, 0xFFFFFFFF);

        Ok(())
    }

    fn stop(&mut self) -> Result<(), UsbError> {
        self.write_reg(Self::GINTMSK, 0);
        Ok(())
    }

    fn root_hub_ports(&self) -> u8 {
        self.ports
    }

    fn port_status(&self, port: u8) -> PortStatus {
        if port != 0 {
            return PortStatus::default();
        }

        let hprt = self.read_reg(Self::HPRT);

        let speed = match (hprt >> 17) & 0x3 {
            0 => UsbSpeed::High,
            1 => UsbSpeed::Full,
            2 => UsbSpeed::Low,
            _ => UsbSpeed::Full,
        };

        PortStatus {
            connected: (hprt & 1) != 0,
            enabled: (hprt & (1 << 2)) != 0,
            suspended: (hprt & (1 << 7)) != 0,
            overcurrent: (hprt & (1 << 4)) != 0,
            reset: (hprt & (1 << 8)) != 0,
            power: (hprt & (1 << 12)) != 0,
            speed,
            changed: (hprt & (1 << 1)) != 0,
        }
    }

    fn reset_port(&mut self, port: u8) -> Result<UsbSpeed, UsbError> {
        if port != 0 {
            return Err(UsbError::InvalidEndpoint);
        }

        let mut hprt = self.read_reg(Self::HPRT);

        // Start reset
        hprt |= 1 << 8;  // Port reset
        hprt &= !(1 << 2);  // Clear enable
        self.write_reg(Self::HPRT, hprt);

        // Wait 50ms
        for _ in 0..500000 {
            core::hint::spin_loop();
        }

        // Clear reset
        hprt &= !(1 << 8);
        self.write_reg(Self::HPRT, hprt);

        // Wait for reset complete
        for _ in 0..100000 {
            core::hint::spin_loop();
        }

        let status = self.port_status(0);
        Ok(status.speed)
    }

    fn enable_port(&mut self, port: u8) -> Result<(), UsbError> {
        if port != 0 {
            return Err(UsbError::InvalidEndpoint);
        }

        let mut hprt = self.read_reg(Self::HPRT);
        hprt |= 1 << 2;  // Port enable
        self.write_reg(Self::HPRT, hprt);

        Ok(())
    }

    fn disable_port(&mut self, port: u8) -> Result<(), UsbError> {
        if port != 0 {
            return Err(UsbError::InvalidEndpoint);
        }

        let mut hprt = self.read_reg(Self::HPRT);
        hprt &= !(1 << 2);  // Port disable
        self.write_reg(Self::HPRT, hprt);

        Ok(())
    }

    fn control_transfer(
        &mut self,
        _device: u8,
        _setup: SetupPacket,
        _data: &mut [u8],
    ) -> Result<usize, UsbError> {
        if !self.initialized {
            return Err(UsbError::NotSupported);
        }

        // Would implement actual control transfer using DWC2 channels
        // For now, return placeholder
        Ok(0)
    }

    fn bulk_transfer(
        &mut self,
        _device: u8,
        _endpoint: u8,
        _data: &mut [u8],
        _is_in: bool,
    ) -> Result<usize, UsbError> {
        if !self.initialized {
            return Err(UsbError::NotSupported);
        }

        Ok(0)
    }

    fn interrupt_transfer(
        &mut self,
        _device: u8,
        _endpoint: u8,
        _data: &mut [u8],
        _is_in: bool,
    ) -> Result<usize, UsbError> {
        if !self.initialized {
            return Err(UsbError::NotSupported);
        }

        Ok(0)
    }
}

/// XHCI (USB 3.0) Controller
pub struct XhciController {
    base: usize,
    capability_regs: usize,
    operational_regs: usize,
    runtime_regs: usize,
    doorbell_regs: usize,
    ports: u8,
    initialized: bool,
}

impl XhciController {
    /// Create new XHCI controller
    pub fn new(base: usize) -> Self {
        Self {
            base,
            capability_regs: base,
            operational_regs: 0,
            runtime_regs: 0,
            doorbell_regs: 0,
            ports: 0,
            initialized: false,
        }
    }

    fn read_cap_reg(&self, offset: usize) -> u32 {
        unsafe { core::ptr::read_volatile((self.capability_regs + offset) as *const u32) }
    }

    fn read_op_reg(&self, offset: usize) -> u32 {
        unsafe { core::ptr::read_volatile((self.operational_regs + offset) as *const u32) }
    }

    fn write_op_reg(&mut self, offset: usize, value: u32) {
        unsafe { core::ptr::write_volatile((self.operational_regs + offset) as *mut u32, value) }
    }
}

impl HostController for XhciController {
    fn controller_type(&self) -> ControllerType {
        ControllerType::Xhci
    }

    fn init(&mut self) -> Result<(), UsbError> {
        crate::kdebug!("XHCI: Initializing controller at 0x{:x}", self.base);

        // Read capability registers
        let caplength = (self.read_cap_reg(0) & 0xFF) as usize;
        let hcsparams1 = self.read_cap_reg(4);

        self.operational_regs = self.capability_regs + caplength;
        self.runtime_regs = self.capability_regs + (self.read_cap_reg(0x18) as usize);
        self.doorbell_regs = self.capability_regs + (self.read_cap_reg(0x14) as usize);
        self.ports = (hcsparams1 >> 24) as u8;

        // Stop controller
        let mut usbcmd = self.read_op_reg(0);
        usbcmd &= !1;  // Clear Run/Stop
        self.write_op_reg(0, usbcmd);

        // Wait for halt
        while self.read_op_reg(4) & 1 == 0 {
            core::hint::spin_loop();
        }

        // Reset controller
        usbcmd = self.read_op_reg(0);
        usbcmd |= 1 << 1;  // HC Reset
        self.write_op_reg(0, usbcmd);

        while self.read_op_reg(0) & (1 << 1) != 0 {
            core::hint::spin_loop();
        }

        self.initialized = true;
        crate::kinfo!("XHCI: Controller initialized with {} ports", self.ports);

        Ok(())
    }

    fn reset(&mut self) -> Result<(), UsbError> {
        let mut usbcmd = self.read_op_reg(0);
        usbcmd |= 1 << 1;
        self.write_op_reg(0, usbcmd);

        while self.read_op_reg(0) & (1 << 1) != 0 {
            core::hint::spin_loop();
        }

        Ok(())
    }

    fn start(&mut self) -> Result<(), UsbError> {
        let mut usbcmd = self.read_op_reg(0);
        usbcmd |= 1;  // Run
        self.write_op_reg(0, usbcmd);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), UsbError> {
        let mut usbcmd = self.read_op_reg(0);
        usbcmd &= !1;  // Stop
        self.write_op_reg(0, usbcmd);
        Ok(())
    }

    fn root_hub_ports(&self) -> u8 {
        self.ports
    }

    fn port_status(&self, port: u8) -> PortStatus {
        if port >= self.ports {
            return PortStatus::default();
        }

        let portsc = self.read_op_reg(0x400 + (port as usize * 0x10));

        let speed = match (portsc >> 10) & 0xF {
            1 => UsbSpeed::Full,
            2 => UsbSpeed::Low,
            3 => UsbSpeed::High,
            4 => UsbSpeed::Super,
            5 => UsbSpeed::SuperPlus,
            _ => UsbSpeed::Full,
        };

        PortStatus {
            connected: (portsc & 1) != 0,
            enabled: (portsc & (1 << 1)) != 0,
            suspended: false,
            overcurrent: (portsc & (1 << 3)) != 0,
            reset: (portsc & (1 << 4)) != 0,
            power: (portsc & (1 << 9)) != 0,
            speed,
            changed: (portsc & (1 << 17)) != 0,
        }
    }

    fn reset_port(&mut self, port: u8) -> Result<UsbSpeed, UsbError> {
        if port >= self.ports {
            return Err(UsbError::InvalidEndpoint);
        }

        let reg_offset = 0x400 + (port as usize * 0x10);

        // Issue port reset
        let mut portsc = self.read_op_reg(reg_offset);
        portsc |= 1 << 4;  // Port Reset
        self.write_op_reg(reg_offset, portsc);

        // Wait for reset complete
        for _ in 0..100000 {
            portsc = self.read_op_reg(reg_offset);
            if portsc & (1 << 4) == 0 {
                break;
            }
            core::hint::spin_loop();
        }

        let status = self.port_status(port);
        Ok(status.speed)
    }

    fn enable_port(&mut self, _port: u8) -> Result<(), UsbError> {
        Ok(())
    }

    fn disable_port(&mut self, port: u8) -> Result<(), UsbError> {
        if port >= self.ports {
            return Err(UsbError::InvalidEndpoint);
        }

        let reg_offset = 0x400 + (port as usize * 0x10);
        let mut portsc = self.read_op_reg(reg_offset);
        portsc |= 1 << 1;  // Disable
        self.write_op_reg(reg_offset, portsc);

        Ok(())
    }

    fn control_transfer(
        &mut self,
        _device: u8,
        _setup: SetupPacket,
        _data: &mut [u8],
    ) -> Result<usize, UsbError> {
        if !self.initialized {
            return Err(UsbError::NotSupported);
        }
        Ok(0)
    }

    fn bulk_transfer(
        &mut self,
        _device: u8,
        _endpoint: u8,
        _data: &mut [u8],
        _is_in: bool,
    ) -> Result<usize, UsbError> {
        if !self.initialized {
            return Err(UsbError::NotSupported);
        }
        Ok(0)
    }

    fn interrupt_transfer(
        &mut self,
        _device: u8,
        _endpoint: u8,
        _data: &mut [u8],
        _is_in: bool,
    ) -> Result<usize, UsbError> {
        if !self.initialized {
            return Err(UsbError::NotSupported);
        }
        Ok(0)
    }
}

/// Global host controller
static HOST_CONTROLLER: Mutex<Option<Arc<Mutex<dyn HostController>>>> = Mutex::new(None);

/// Register host controller
pub fn register_controller(controller: Arc<Mutex<dyn HostController>>) {
    *HOST_CONTROLLER.lock() = Some(controller);
}

/// Get host controller
pub fn get_controller() -> Option<Arc<Mutex<dyn HostController>>> {
    HOST_CONTROLLER.lock().clone()
}

/// Initialize HCD
pub fn init() {
    crate::kprintln!("  USB Host Controller initialized");
}
