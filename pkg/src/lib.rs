//! HubLab IO Package Manager
//!
//! Complete package management system for HubLab IO.
//! Handles package installation, removal, updates, and repository management.

#![no_std]

extern crate alloc;

#[cfg(feature = "manager")]
#[path = "../manager/mod.rs"]
pub mod manager;

#[cfg(feature = "repo")]
#[path = "../repo/mod.rs"]
pub mod repo;

// Re-export main types
#[cfg(feature = "manager")]
pub use manager::{
    PackageManager, PackageInfo, Version, PackageCategory,
    Architecture, Dependency, VersionConstraint, InstallState,
    InstalledPackage, PackageError, PackageStatistics,
};

#[cfg(feature = "repo")]
pub use repo::{
    RepoConfig, RepoType, RepoIndex, RepoManager,
    IndexEntry, ReleaseInfo, DownloadProgress, DownloadState,
};

/// Package manager version
pub const VERSION: &str = "0.1.0";

/// Default repository URL
pub const DEFAULT_REPO_URL: &str = "https://pkg.hublabio.dev";

use alloc::string::String;

/// Initialize package manager with defaults
#[cfg(feature = "manager")]
pub fn init(root_dir: &str) -> PackageManager {
    let mut pm = PackageManager::new(root_dir);

    // Add default repository
    pm.add_repository(manager::Repository {
        name: String::from("hublabio"),
        url: String::from(DEFAULT_REPO_URL),
        enabled: true,
        priority: 100,
        last_updated: 0,
    });

    pm
}

/// Quick package search
#[cfg(feature = "manager")]
pub fn search(pm: &PackageManager, query: &str) -> alloc::vec::Vec<&PackageInfo> {
    pm.search(query)
}

/// Quick package install
#[cfg(feature = "manager")]
pub fn install(pm: &mut PackageManager, name: &str) -> Result<(), PackageError> {
    pm.install(name)
}

/// Quick package remove
#[cfg(feature = "manager")]
pub fn remove(pm: &mut PackageManager, name: &str) -> Result<(), PackageError> {
    pm.remove(name)
}
