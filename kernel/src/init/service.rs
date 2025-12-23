//! Service Management
//!
//! Individual service lifecycle and configuration.

use alloc::string::String;
use alloc::vec::Vec;

use crate::process::Pid;

/// Service state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceState {
    /// Not started
    Stopped,
    /// Starting up
    Starting,
    /// Running normally
    Running,
    /// Stopping
    Stopping,
    /// Failed to start or crashed
    Failed,
}

/// Service configuration
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Service name (unique identifier)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Executable path
    pub exec: String,
    /// Whether to start automatically
    pub enabled: bool,
    /// Restart on failure
    pub restart_on_failure: bool,
    /// Priority (higher = starts earlier)
    pub priority: u8,
    /// Services that must start before this one
    pub dependencies: Vec<String>,
}

/// Service instance
#[derive(Clone, Debug)]
pub struct Service {
    config: ServiceConfig,
    state: ServiceState,
    pid: Option<Pid>,
    restart_count: u32,
    last_exit_code: Option<i32>,
}

impl Service {
    /// Create a new service from configuration
    pub fn new(config: ServiceConfig) -> Self {
        Self {
            config,
            state: ServiceState::Stopped,
            pid: None,
            restart_count: 0,
            last_exit_code: None,
        }
    }

    /// Get service name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get service configuration
    pub fn config(&self) -> &ServiceConfig {
        &self.config
    }

    /// Get current state
    pub fn state(&self) -> ServiceState {
        self.state
    }

    /// Get process ID if running
    pub fn pid(&self) -> Option<Pid> {
        self.pid
    }

    /// Get restart count
    pub fn restart_count(&self) -> u32 {
        self.restart_count
    }

    /// Start the service
    pub fn start(&mut self) -> Result<(), &'static str> {
        match self.state {
            ServiceState::Running | ServiceState::Starting => {
                return Err("Service already running");
            }
            _ => {}
        }

        self.state = ServiceState::Starting;

        crate::kinfo!("Starting service: {}", self.config.name);

        // For now, simulate starting the service
        // In a real implementation, this would spawn the process
        match spawn_service_process(&self.config.exec) {
            Ok(pid) => {
                self.pid = Some(pid);
                self.state = ServiceState::Running;
                crate::kinfo!("Service '{}' started with PID {}", self.config.name, pid);
                Ok(())
            }
            Err(e) => {
                self.state = ServiceState::Failed;
                crate::kerror!("Failed to start service '{}': {}", self.config.name, e);
                Err(e)
            }
        }
    }

    /// Stop the service
    pub fn stop(&mut self) -> Result<(), &'static str> {
        match self.state {
            ServiceState::Stopped => {
                return Ok(()); // Already stopped
            }
            ServiceState::Running => {}
            _ => {
                return Err("Service not in stoppable state");
            }
        }

        self.state = ServiceState::Stopping;

        crate::kinfo!("Stopping service: {}", self.config.name);

        if let Some(pid) = self.pid {
            // Send SIGTERM
            if let Err(e) = crate::process::kill(pid, 15) {
                crate::kwarn!("Failed to send SIGTERM to {}: {}", pid, e);
            }

            // TODO: Wait with timeout, then SIGKILL
        }

        self.pid = None;
        self.state = ServiceState::Stopped;

        crate::kinfo!("Service '{}' stopped", self.config.name);
        Ok(())
    }

    /// Mark service as stopped (called when process exits)
    pub fn mark_stopped(&mut self) {
        self.state = ServiceState::Stopped;
        self.pid = None;
    }

    /// Mark service as failed
    pub fn mark_failed(&mut self, exit_code: i32) {
        self.state = ServiceState::Failed;
        self.last_exit_code = Some(exit_code);
        self.pid = None;
    }

    /// Increment restart count
    pub fn increment_restart(&mut self) {
        self.restart_count += 1;
    }
}

/// Spawn a service process
fn spawn_service_process(exec_path: &str) -> Result<Pid, &'static str> {
    // For now, we'll create a placeholder process
    // In a real implementation, this would:
    // 1. Load the ELF binary from exec_path
    // 2. Create a new process with fork()
    // 3. Execute the binary with exec()

    // Simulate with a dummy PID for now
    static mut NEXT_SERVICE_PID: Pid = 100;

    let pid = unsafe {
        let pid = NEXT_SERVICE_PID;
        NEXT_SERVICE_PID += 1;
        pid
    };

    crate::kdebug!("Would spawn '{}' as PID {}", exec_path, pid);

    // In the future, this would be:
    // let pid = crate::process::fork()?;
    // if pid == 0 {
    //     // Child process
    //     crate::process::exec(exec_path, &[])?;
    // }
    // Ok(pid)

    Ok(pid)
}

/// Service dependency resolver
pub struct DependencyResolver<'a> {
    services: &'a [ServiceConfig],
}

impl<'a> DependencyResolver<'a> {
    /// Create a new dependency resolver
    pub fn new(services: &'a [ServiceConfig]) -> Self {
        Self { services }
    }

    /// Get services in start order (respecting dependencies)
    pub fn resolve(&self) -> Result<Vec<&ServiceConfig>, &'static str> {
        let mut resolved: Vec<&ServiceConfig> = Vec::new();
        let mut unresolved: Vec<&ServiceConfig> = self.services.iter().collect();

        // Simple topological sort
        let max_iterations = self.services.len() * 2;
        let mut iterations = 0;

        while !unresolved.is_empty() {
            iterations += 1;
            if iterations > max_iterations {
                return Err("Circular dependency detected");
            }

            let mut made_progress = false;

            unresolved.retain(|svc| {
                let deps_satisfied = svc
                    .dependencies
                    .iter()
                    .all(|dep| resolved.iter().any(|r| r.name == *dep));

                if deps_satisfied {
                    resolved.push(svc);
                    made_progress = true;
                    false // Remove from unresolved
                } else {
                    true // Keep in unresolved
                }
            });

            if !made_progress && !unresolved.is_empty() {
                return Err("Unsatisfied dependencies");
            }
        }

        // Sort by priority within dependency order
        resolved.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_service(name: &str, deps: Vec<String>) -> ServiceConfig {
        ServiceConfig {
            name: String::from(name),
            description: String::new(),
            exec: String::new(),
            enabled: true,
            restart_on_failure: false,
            priority: 50,
            dependencies: deps,
        }
    }

    #[test]
    fn test_dependency_resolution() {
        let services = [
            test_service("shell", alloc::vec![String::from("console")]),
            test_service("console", Vec::new()),
            test_service("network", alloc::vec![String::from("devmgr")]),
            test_service("devmgr", Vec::new()),
        ];

        let resolver = DependencyResolver::new(&services);
        let order = resolver.resolve().unwrap();

        // Console and devmgr should come before their dependents
        let console_pos = order.iter().position(|s| s.name == "console").unwrap();
        let shell_pos = order.iter().position(|s| s.name == "shell").unwrap();
        let devmgr_pos = order.iter().position(|s| s.name == "devmgr").unwrap();
        let network_pos = order.iter().position(|s| s.name == "network").unwrap();

        assert!(console_pos < shell_pos);
        assert!(devmgr_pos < network_pos);
    }
}
