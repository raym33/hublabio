//! OTA Update Service
//!
//! Over-the-Air update system for HubLab IO.
//! Supports A/B partitioning, delta updates, and rollback.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Update server URL
pub const DEFAULT_UPDATE_SERVER: &str = "https://updates.hublabio.dev";

/// Update check interval (24 hours in seconds)
pub const UPDATE_CHECK_INTERVAL: u64 = 86400;

/// Update channel
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdateChannel {
    /// Stable releases
    Stable,
    /// Beta releases
    Beta,
    /// Development/nightly builds
    Dev,
    /// Custom channel
    Custom(String),
}

impl UpdateChannel {
    /// Get channel name string
    pub fn name(&self) -> &str {
        match self {
            UpdateChannel::Stable => "stable",
            UpdateChannel::Beta => "beta",
            UpdateChannel::Dev => "dev",
            UpdateChannel::Custom(name) => name,
        }
    }
}

/// Version information
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Build number
    pub build: Option<u32>,
    /// Pre-release tag
    pub prerelease: Option<String>,
}

impl Version {
    /// Create new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            build: None,
            prerelease: None,
        }
    }

    /// Parse version string (e.g., "1.2.3-beta.1+456")
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split(&['-', '+'][..]).collect();
        let version_str = parts.first()?;

        let numbers: Vec<&str> = version_str.split('.').collect();
        if numbers.len() < 3 {
            return None;
        }

        Some(Self {
            major: numbers[0].parse().ok()?,
            minor: numbers[1].parse().ok()?,
            patch: numbers[2].parse().ok()?,
            build: parts
                .iter()
                .find(|p| p.chars().all(|c| c.is_ascii_digit()))
                .and_then(|s| s.parse().ok()),
            prerelease: parts
                .get(1)
                .filter(|p| !p.chars().all(|c| c.is_ascii_digit()))
                .map(|s| s.to_string()),
        })
    }

    /// Format as string
    pub fn to_string(&self) -> String {
        let mut s = format!("{}.{}.{}", self.major, self.minor, self.patch);
        if let Some(ref pre) = self.prerelease {
            s.push('-');
            s.push_str(pre);
        }
        if let Some(build) = self.build {
            s.push('+');
            s.push_str(&build.to_string());
        }
        s
    }

    /// Check if this version is newer than other
    pub fn is_newer_than(&self, other: &Version) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        if self.patch != other.patch {
            return self.patch > other.patch;
        }
        // Pre-release versions are older than release versions
        match (&self.prerelease, &other.prerelease) {
            (None, Some(_)) => true,
            (Some(_), None) => false,
            _ => false,
        }
    }
}

/// Update manifest from server
#[derive(Clone, Debug)]
pub struct UpdateManifest {
    /// Version
    pub version: Version,
    /// Channel
    pub channel: UpdateChannel,
    /// Release date
    pub release_date: String,
    /// Changelog
    pub changelog: Vec<String>,
    /// Components to update
    pub components: Vec<ComponentUpdate>,
    /// Total download size
    pub download_size: u64,
    /// Installed size
    pub installed_size: u64,
    /// Required disk space
    pub required_space: u64,
    /// Signature
    pub signature: String,
    /// Minimum required version to update from
    pub min_version: Option<Version>,
}

/// Component update info
#[derive(Clone, Debug)]
pub struct ComponentUpdate {
    /// Component name
    pub name: String,
    /// Component version
    pub version: String,
    /// Download URL
    pub url: String,
    /// File hash (SHA-256)
    pub hash: String,
    /// Size in bytes
    pub size: u64,
    /// Is delta update
    pub is_delta: bool,
    /// Base version for delta (if is_delta)
    pub delta_base: Option<String>,
}

/// Update state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateState {
    /// No update in progress
    Idle,
    /// Checking for updates
    Checking,
    /// Update available
    Available,
    /// Downloading update
    Downloading,
    /// Verifying download
    Verifying,
    /// Installing update
    Installing,
    /// Waiting for reboot
    PendingReboot,
    /// Update failed
    Failed,
}

/// A/B partition slot
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// Slot A
    A,
    /// Slot B
    B,
}

impl Slot {
    /// Get the other slot
    pub fn other(&self) -> Slot {
        match self {
            Slot::A => Slot::B,
            Slot::B => Slot::A,
        }
    }
}

/// Partition information
#[derive(Clone, Debug)]
pub struct PartitionInfo {
    /// Slot
    pub slot: Slot,
    /// Bootable
    pub bootable: bool,
    /// Successful boot
    pub successful: bool,
    /// Priority (higher = preferred)
    pub priority: u8,
    /// Retry count
    pub tries_remaining: u8,
    /// Installed version
    pub version: Option<Version>,
}

/// Download progress
#[derive(Clone, Debug)]
pub struct DownloadProgress {
    /// Current component
    pub component: String,
    /// Bytes downloaded
    pub downloaded: u64,
    /// Total bytes
    pub total: u64,
    /// Download speed (bytes/sec)
    pub speed: u64,
    /// ETA in seconds
    pub eta: u64,
}

impl DownloadProgress {
    /// Get progress percentage
    pub fn percent(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        (self.downloaded as f32 / self.total as f32) * 100.0
    }
}

/// Update error
#[derive(Clone, Debug)]
pub enum UpdateError {
    /// Network error
    NetworkError(String),
    /// Download failed
    DownloadFailed,
    /// Verification failed
    VerificationFailed,
    /// Not enough space
    InsufficientSpace,
    /// Installation failed
    InstallFailed(String),
    /// Invalid manifest
    InvalidManifest,
    /// Signature invalid
    InvalidSignature,
    /// Version too old to update
    VersionTooOld,
    /// Already up to date
    AlreadyUpToDate,
    /// Update in progress
    UpdateInProgress,
    /// Rollback failed
    RollbackFailed,
}

/// Update configuration
#[derive(Clone, Debug)]
pub struct UpdateConfig {
    /// Update server URL
    pub server_url: String,
    /// Channel
    pub channel: UpdateChannel,
    /// Auto-check for updates
    pub auto_check: bool,
    /// Auto-download updates
    pub auto_download: bool,
    /// Auto-install updates
    pub auto_install: bool,
    /// Metered connection handling
    pub allow_metered: bool,
    /// Maximum bandwidth (bytes/sec, 0 = unlimited)
    pub max_bandwidth: u64,
    /// Verify signatures
    pub verify_signature: bool,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            server_url: String::from(DEFAULT_UPDATE_SERVER),
            channel: UpdateChannel::Stable,
            auto_check: true,
            auto_download: false,
            auto_install: false,
            allow_metered: false,
            max_bandwidth: 0,
            verify_signature: true,
        }
    }
}

/// OTA Update Manager
pub struct UpdateManager {
    /// Configuration
    config: UpdateConfig,
    /// Current state
    state: UpdateState,
    /// Current version
    current_version: Version,
    /// Active slot
    active_slot: Slot,
    /// Slot information
    slots: [PartitionInfo; 2],
    /// Available update manifest
    available_update: Option<UpdateManifest>,
    /// Download progress
    download_progress: Option<DownloadProgress>,
    /// Last check timestamp
    last_check: u64,
    /// Error message
    last_error: Option<UpdateError>,
}

impl UpdateManager {
    /// Create new update manager
    pub fn new(current_version: Version, config: UpdateConfig) -> Self {
        Self {
            config,
            state: UpdateState::Idle,
            current_version,
            active_slot: Slot::A,
            slots: [
                PartitionInfo {
                    slot: Slot::A,
                    bootable: true,
                    successful: true,
                    priority: 1,
                    tries_remaining: 3,
                    version: None,
                },
                PartitionInfo {
                    slot: Slot::B,
                    bootable: false,
                    successful: false,
                    priority: 0,
                    tries_remaining: 3,
                    version: None,
                },
            ],
            available_update: None,
            download_progress: None,
            last_check: 0,
            last_error: None,
        }
    }

    /// Check for available updates
    pub fn check_for_updates(&mut self) -> Result<Option<&UpdateManifest>, UpdateError> {
        if self.state != UpdateState::Idle {
            return Err(UpdateError::UpdateInProgress);
        }

        self.state = UpdateState::Checking;
        self.last_error = None;

        // TODO: Actually fetch manifest from server
        // let url = format!("{}/manifest/{}.json", self.config.server_url, self.config.channel.name());
        // let response = http_get(&url)?;
        // let manifest = parse_manifest(&response)?;

        // Placeholder: no update available
        self.state = UpdateState::Idle;
        self.last_check = 0; // TODO: get current time
        self.available_update = None;

        Ok(self.available_update.as_ref())
    }

    /// Start downloading an update
    pub fn download(&mut self) -> Result<(), UpdateError> {
        let manifest = self
            .available_update
            .as_ref()
            .ok_or(UpdateError::InvalidManifest)?;

        // Check disk space
        if !self.check_disk_space(manifest.required_space) {
            return Err(UpdateError::InsufficientSpace);
        }

        self.state = UpdateState::Downloading;
        self.download_progress = Some(DownloadProgress {
            component: String::new(),
            downloaded: 0,
            total: manifest.download_size,
            speed: 0,
            eta: 0,
        });

        // TODO: Actually download components
        // for component in &manifest.components {
        //     self.download_component(component)?;
        // }

        self.state = UpdateState::Verifying;
        // TODO: Verify hashes

        Ok(())
    }

    /// Install downloaded update
    pub fn install(&mut self) -> Result<(), UpdateError> {
        if self.state != UpdateState::Verifying {
            return Err(UpdateError::InstallFailed(String::from(
                "Not ready to install",
            )));
        }

        let manifest = self
            .available_update
            .as_ref()
            .ok_or(UpdateError::InvalidManifest)?;

        // Verify signature
        if self.config.verify_signature {
            self.verify_signature(manifest)?;
        }

        self.state = UpdateState::Installing;

        // Get target slot
        let target_slot = self.active_slot.other();
        let target_idx = if target_slot == Slot::A { 0 } else { 1 };

        // TODO: Actually install to target slot
        // for component in &manifest.components {
        //     self.install_component(component, target_slot)?;
        // }

        // Mark target slot as bootable
        self.slots[target_idx].bootable = true;
        self.slots[target_idx].successful = false;
        self.slots[target_idx].priority = 2;
        self.slots[target_idx].tries_remaining = 3;
        self.slots[target_idx].version = Some(manifest.version.clone());

        // Lower priority of current slot
        let current_idx = if self.active_slot == Slot::A { 0 } else { 1 };
        self.slots[current_idx].priority = 1;

        self.state = UpdateState::PendingReboot;

        Ok(())
    }

    /// Reboot to apply update
    pub fn reboot(&self) -> Result<(), UpdateError> {
        if self.state != UpdateState::PendingReboot {
            return Err(UpdateError::InstallFailed(String::from(
                "No update pending",
            )));
        }

        // TODO: Actually reboot
        // syscall(SYS_REBOOT, LINUX_REBOOT_CMD_RESTART);

        Ok(())
    }

    /// Mark current boot as successful
    pub fn mark_successful(&mut self) {
        let idx = if self.active_slot == Slot::A { 0 } else { 1 };
        self.slots[idx].successful = true;
        self.slots[idx].tries_remaining = 3;

        // TODO: Write to persistent storage
    }

    /// Rollback to previous version
    pub fn rollback(&mut self) -> Result<(), UpdateError> {
        let current_idx = if self.active_slot == Slot::A { 0 } else { 1 };
        let other_idx = if self.active_slot == Slot::A { 1 } else { 0 };

        if !self.slots[other_idx].bootable || !self.slots[other_idx].successful {
            return Err(UpdateError::RollbackFailed);
        }

        // Swap priorities
        self.slots[current_idx].priority = 0;
        self.slots[other_idx].priority = 2;

        self.state = UpdateState::PendingReboot;

        Ok(())
    }

    /// Cancel ongoing update
    pub fn cancel(&mut self) {
        match self.state {
            UpdateState::Downloading | UpdateState::Verifying => {
                // TODO: Abort download, clean up partial files
                self.state = UpdateState::Idle;
                self.download_progress = None;
            }
            _ => {}
        }
    }

    /// Get current state
    pub fn state(&self) -> UpdateState {
        self.state
    }

    /// Get current version
    pub fn current_version(&self) -> &Version {
        &self.current_version
    }

    /// Get available update info
    pub fn available_update(&self) -> Option<&UpdateManifest> {
        self.available_update.as_ref()
    }

    /// Get download progress
    pub fn progress(&self) -> Option<&DownloadProgress> {
        self.download_progress.as_ref()
    }

    /// Get active slot
    pub fn active_slot(&self) -> Slot {
        self.active_slot
    }

    /// Get slot info
    pub fn slot_info(&self, slot: Slot) -> &PartitionInfo {
        let idx = if slot == Slot::A { 0 } else { 1 };
        &self.slots[idx]
    }

    /// Get last error
    pub fn last_error(&self) -> Option<&UpdateError> {
        self.last_error.as_ref()
    }

    /// Set update channel
    pub fn set_channel(&mut self, channel: UpdateChannel) {
        self.config.channel = channel;
    }

    /// Check available disk space
    fn check_disk_space(&self, required: u64) -> bool {
        // TODO: Actually check filesystem
        true
    }

    /// Verify update signature
    fn verify_signature(&self, manifest: &UpdateManifest) -> Result<(), UpdateError> {
        // TODO: Verify Ed25519 or similar signature
        if manifest.signature.is_empty() {
            return Err(UpdateError::InvalidSignature);
        }
        Ok(())
    }

    /// Get update status summary
    pub fn status_summary(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Current version: {}\n",
            self.current_version.to_string()
        ));
        output.push_str(&format!("Active slot: {:?}\n", self.active_slot));
        output.push_str(&format!("Channel: {}\n", self.config.channel.name()));
        output.push_str(&format!("State: {:?}\n", self.state));

        if let Some(update) = &self.available_update {
            output.push_str(&format!(
                "\nAvailable update: {}\n",
                update.version.to_string()
            ));
            output.push_str(&format!(
                "Download size: {} MB\n",
                update.download_size / (1024 * 1024)
            ));
        }

        if let Some(progress) = &self.download_progress {
            output.push_str(&format!(
                "\nDownload progress: {:.1}%\n",
                progress.percent()
            ));
            output.push_str(&format!("Speed: {} KB/s\n", progress.speed / 1024));
            output.push_str(&format!("ETA: {} seconds\n", progress.eta));
        }

        output.push_str("\nSlot A: ");
        output.push_str(&self.format_slot(&self.slots[0]));
        output.push_str("\nSlot B: ");
        output.push_str(&self.format_slot(&self.slots[1]));

        output
    }

    fn format_slot(&self, slot: &PartitionInfo) -> String {
        let version = slot
            .version
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| String::from("(empty)"));

        format!(
            "{} bootable={} successful={} priority={} version={}",
            if slot.slot == Slot::A { "A" } else { "B" },
            slot.bootable,
            slot.successful,
            slot.priority,
            version
        )
    }
}

impl Default for UpdateManager {
    fn default() -> Self {
        Self::new(Version::new(0, 1, 0), UpdateConfig::default())
    }
}

/// Recovery mode update (for failed boots)
pub struct RecoveryUpdate {
    /// Recovery partition path
    pub recovery_path: String,
    /// USB update path
    pub usb_path: Option<String>,
}

impl RecoveryUpdate {
    /// Create recovery update handler
    pub fn new(recovery_path: &str) -> Self {
        Self {
            recovery_path: String::from(recovery_path),
            usb_path: None,
        }
    }

    /// Check for USB update
    pub fn check_usb(&mut self) -> bool {
        // TODO: Check for update file on USB
        // Look for /media/usb/hublabio-update.img
        false
    }

    /// Apply recovery update
    pub fn apply(&self) -> Result<(), UpdateError> {
        // TODO: Flash recovery/USB image to active slot
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);

        let v2 = Version::parse("1.2.3-beta.1+456").unwrap();
        assert_eq!(v2.prerelease, Some(String::from("beta.1")));
    }

    #[test]
    fn test_version_comparison() {
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 0, 1);
        let v3 = Version::new(2, 0, 0);

        assert!(v2.is_newer_than(&v1));
        assert!(v3.is_newer_than(&v2));
        assert!(!v1.is_newer_than(&v2));
    }

    #[test]
    fn test_slot_other() {
        assert_eq!(Slot::A.other(), Slot::B);
        assert_eq!(Slot::B.other(), Slot::A);
    }

    #[test]
    fn test_update_manager_creation() {
        let manager = UpdateManager::default();
        assert_eq!(manager.state(), UpdateState::Idle);
        assert_eq!(manager.active_slot(), Slot::A);
    }
}
