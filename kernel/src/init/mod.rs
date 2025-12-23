//! Init System and Service Manager
//!
//! The init system is PID 1, responsible for:
//! - Starting system services
//! - Managing service lifecycle
//! - Handling orphaned processes
//! - System shutdown/reboot

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::RwLock;

pub mod service;

pub use service::{Service, ServiceState, ServiceConfig};

/// Global service manager
pub static SERVICE_MANAGER: RwLock<ServiceManager> = RwLock::new(ServiceManager::new());

/// Service manager
pub struct ServiceManager {
    services: BTreeMap<String, Service>,
    boot_complete: bool,
}

impl ServiceManager {
    /// Create a new service manager
    pub const fn new() -> Self {
        Self {
            services: BTreeMap::new(),
            boot_complete: false,
        }
    }

    /// Register a service
    pub fn register(&mut self, config: ServiceConfig) -> Result<(), &'static str> {
        if self.services.contains_key(&config.name) {
            return Err("Service already registered");
        }

        let service = Service::new(config);
        let name = service.name().to_string();
        self.services.insert(name, service);
        Ok(())
    }

    /// Start a service by name
    pub fn start(&mut self, name: &str) -> Result<(), &'static str> {
        let service = self.services.get_mut(name)
            .ok_or("Service not found")?;
        service.start()
    }

    /// Stop a service by name
    pub fn stop(&mut self, name: &str) -> Result<(), &'static str> {
        let service = self.services.get_mut(name)
            .ok_or("Service not found")?;
        service.stop()
    }

    /// Restart a service by name
    pub fn restart(&mut self, name: &str) -> Result<(), &'static str> {
        self.stop(name)?;
        self.start(name)
    }

    /// Get service status
    pub fn status(&self, name: &str) -> Option<ServiceState> {
        self.services.get(name).map(|s| s.state())
    }

    /// List all services
    pub fn list(&self) -> Vec<(&str, ServiceState)> {
        self.services.iter()
            .map(|(name, svc)| (name.as_str(), svc.state()))
            .collect()
    }

    /// Start all enabled services
    pub fn start_all(&mut self) {
        // Collect names first to avoid borrow issues
        let names: Vec<String> = self.services.iter()
            .filter(|(_, svc)| svc.config().enabled)
            .map(|(name, _)| name.clone())
            .collect();

        for name in names {
            if let Err(e) = self.start(&name) {
                crate::kerror!("Failed to start service '{}': {}", name, e);
            }
        }
    }

    /// Stop all services
    pub fn stop_all(&mut self) {
        let names: Vec<String> = self.services.keys().cloned().collect();

        // Stop in reverse order
        for name in names.into_iter().rev() {
            if let Err(e) = self.stop(&name) {
                crate::kwarn!("Failed to stop service '{}': {}", name, e);
            }
        }
    }

    /// Mark boot as complete
    pub fn boot_complete(&mut self) {
        self.boot_complete = true;
        crate::kinfo!("Boot complete, all services started");
    }

    /// Check if boot is complete
    pub fn is_boot_complete(&self) -> bool {
        self.boot_complete
    }
}

/// Initialize the init system
pub fn init() {
    crate::kprintln!("  Initializing init system...");

    // Register core system services
    register_core_services();

    // Start all enabled services
    SERVICE_MANAGER.write().start_all();

    // Mark boot as complete
    SERVICE_MANAGER.write().boot_complete();
}

/// Register core system services
fn register_core_services() {
    let mut manager = SERVICE_MANAGER.write();

    // Console service (handles terminal I/O)
    let _ = manager.register(ServiceConfig {
        name: String::from("console"),
        description: String::from("Console and terminal service"),
        exec: String::from("/sbin/console"),
        enabled: true,
        restart_on_failure: true,
        priority: 100,
        dependencies: Vec::new(),
    });

    // AI Runtime service
    let _ = manager.register(ServiceConfig {
        name: String::from("ai-runtime"),
        description: String::from("AI inference runtime"),
        exec: String::from("/sbin/ai-runtime"),
        enabled: true,
        restart_on_failure: true,
        priority: 90,
        dependencies: Vec::new(),
    });

    // Network service
    let _ = manager.register(ServiceConfig {
        name: String::from("network"),
        description: String::from("Network manager"),
        exec: String::from("/sbin/networkd"),
        enabled: false, // Disabled until we have network drivers
        restart_on_failure: true,
        priority: 80,
        dependencies: Vec::new(),
    });

    // Device manager
    let _ = manager.register(ServiceConfig {
        name: String::from("devmgr"),
        description: String::from("Device manager"),
        exec: String::from("/sbin/devmgr"),
        enabled: true,
        restart_on_failure: true,
        priority: 95,
        dependencies: Vec::new(),
    });

    // Shell service
    let _ = manager.register(ServiceConfig {
        name: String::from("shell"),
        description: String::from("Interactive shell"),
        exec: String::from("/bin/shell"),
        enabled: true,
        restart_on_failure: true,
        priority: 50,
        dependencies: alloc::vec![String::from("console")],
    });

    crate::kinfo!("Registered {} core services", manager.services.len());
}

/// Init process entry point (runs as PID 1)
pub fn init_main() -> ! {
    crate::kprintln!();
    crate::kprintln!("  Init process started (PID 1)");

    // Initialize the service manager
    init();

    // Main init loop - reap orphaned processes
    loop {
        // Wait for child processes
        match crate::process::wait(None) {
            Ok((pid, status)) => {
                crate::kdebug!("Reaped orphan process PID {} with status {}", pid, status);

                // Check if it was a service and restart if needed
                check_service_exit(pid, status);
            }
            Err(_) => {
                // No children to reap, idle
                crate::arch::halt();
            }
        }
    }
}

/// Check if an exited process was a service and handle restart
fn check_service_exit(pid: crate::process::Pid, status: i32) {
    let mut manager = SERVICE_MANAGER.write();

    for (name, service) in manager.services.iter_mut() {
        if service.pid() == Some(pid) {
            crate::kwarn!("Service '{}' exited with status {}", name, status);

            if service.config().restart_on_failure && status != 0 {
                crate::kinfo!("Restarting service '{}'...", name);
                let _ = service.start();
            } else {
                service.mark_stopped();
            }
            break;
        }
    }
}

/// Shutdown the system
pub fn shutdown() -> ! {
    crate::kprintln!();
    crate::kprintln!("System shutdown initiated...");

    // Stop all services
    SERVICE_MANAGER.write().stop_all();

    crate::kprintln!("All services stopped.");
    crate::kprintln!("System halted. It is safe to power off.");

    loop {
        crate::arch::halt();
    }
}

/// Reboot the system
pub fn reboot() -> ! {
    crate::kprintln!();
    crate::kprintln!("System reboot initiated...");

    // Stop all services
    SERVICE_MANAGER.write().stop_all();

    crate::kprintln!("All services stopped.");
    crate::kprintln!("Rebooting...");

    // Platform-specific reboot
    crate::arch::reboot();
}
