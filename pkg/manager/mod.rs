//! Package Manager Module
//!
//! Manages software packages for HubLab IO, similar to apt/pkg.
//! Supports installation, removal, updates, and dependency resolution.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

/// Package identifier
pub type PackageId = String;

/// Package version (semver-like)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub prerelease: Option<String>,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: None,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() < 3 {
            return None;
        }

        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;

        // Handle prerelease suffix (e.g., "0-beta")
        let (patch_str, prerelease) = if let Some(idx) = parts[2].find('-') {
            let (p, pre) = parts[2].split_at(idx);
            (p, Some(String::from(&pre[1..])))
        } else {
            (parts[2], None)
        };

        let patch = patch_str.parse().ok()?;

        Some(Self {
            major,
            minor,
            patch,
            prerelease,
        })
    }

    pub fn to_string(&self) -> String {
        if let Some(ref pre) = self.prerelease {
            alloc::format!("{}.{}.{}-{}", self.major, self.minor, self.patch, pre)
        } else {
            alloc::format!("{}.{}.{}", self.major, self.minor, self.patch)
        }
    }

    pub fn is_compatible(&self, other: &Version) -> bool {
        self.major == other.major
    }
}

/// Package category
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageCategory {
    System,
    AI,
    Runtime,
    Driver,
    App,
    Library,
    Tool,
    Documentation,
}

impl PackageCategory {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "system" => Some(Self::System),
            "ai" => Some(Self::AI),
            "runtime" => Some(Self::Runtime),
            "driver" => Some(Self::Driver),
            "app" => Some(Self::App),
            "library" | "lib" => Some(Self::Library),
            "tool" => Some(Self::Tool),
            "doc" | "documentation" => Some(Self::Documentation),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::AI => "ai",
            Self::Runtime => "runtime",
            Self::Driver => "driver",
            Self::App => "app",
            Self::Library => "library",
            Self::Tool => "tool",
            Self::Documentation => "documentation",
        }
    }
}

/// Package architecture
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Arm64,
    RiscV64,
    X86_64,
    Any,
}

impl Architecture {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "arm64" | "aarch64" => Some(Self::Arm64),
            "riscv64" | "riscv" => Some(Self::RiscV64),
            "x86_64" | "amd64" => Some(Self::X86_64),
            "any" | "all" | "noarch" => Some(Self::Any),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Arm64 => "arm64",
            Self::RiscV64 => "riscv64",
            Self::X86_64 => "x86_64",
            Self::Any => "any",
        }
    }
}

/// Dependency specification
#[derive(Clone, Debug)]
pub struct Dependency {
    /// Package name
    pub name: String,
    /// Version constraint
    pub version_constraint: VersionConstraint,
    /// Is optional
    pub optional: bool,
}

/// Version constraint types
#[derive(Clone, Debug)]
pub enum VersionConstraint {
    /// Any version
    Any,
    /// Exact version
    Exact(Version),
    /// Minimum version (>=)
    MinVersion(Version),
    /// Maximum version (<=)
    MaxVersion(Version),
    /// Version range
    Range(Version, Version),
    /// Compatible (~)
    Compatible(Version),
}

impl VersionConstraint {
    pub fn satisfies(&self, version: &Version) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(v) => version == v,
            Self::MinVersion(v) => version >= v,
            Self::MaxVersion(v) => version <= v,
            Self::Range(min, max) => version >= min && version <= max,
            Self::Compatible(v) => version.is_compatible(v) && version >= v,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();

        if s == "*" || s.is_empty() {
            return Some(Self::Any);
        }

        if let Some(rest) = s.strip_prefix(">=") {
            return Version::parse(rest.trim()).map(Self::MinVersion);
        }

        if let Some(rest) = s.strip_prefix("<=") {
            return Version::parse(rest.trim()).map(Self::MaxVersion);
        }

        if let Some(rest) = s.strip_prefix("~") {
            return Version::parse(rest.trim()).map(Self::Compatible);
        }

        if let Some(rest) = s.strip_prefix("=") {
            return Version::parse(rest.trim()).map(Self::Exact);
        }

        // Default to exact version
        Version::parse(s).map(Self::Exact)
    }
}

/// Package metadata
#[derive(Clone, Debug)]
pub struct PackageInfo {
    /// Package name
    pub name: String,
    /// Version
    pub version: Version,
    /// Description
    pub description: String,
    /// Category
    pub category: PackageCategory,
    /// Architecture
    pub architecture: Architecture,
    /// Dependencies
    pub dependencies: Vec<Dependency>,
    /// Package size (bytes)
    pub size: u64,
    /// Installed size (bytes)
    pub installed_size: u64,
    /// Maintainer
    pub maintainer: String,
    /// Homepage URL
    pub homepage: Option<String>,
    /// License
    pub license: String,
    /// Files included
    pub files: Vec<String>,
}

impl PackageInfo {
    pub fn new(name: &str, version: Version) -> Self {
        Self {
            name: String::from(name),
            version,
            description: String::new(),
            category: PackageCategory::App,
            architecture: Architecture::Any,
            dependencies: Vec::new(),
            size: 0,
            installed_size: 0,
            maintainer: String::new(),
            homepage: None,
            license: String::from("MIT"),
            files: Vec::new(),
        }
    }

    pub fn full_name(&self) -> String {
        alloc::format!("{}-{}", self.name, self.version.to_string())
    }
}

/// Package installation state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallState {
    /// Not installed
    Available,
    /// Currently installed
    Installed,
    /// Update available
    Upgradable,
    /// Being installed
    Installing,
    /// Being removed
    Removing,
    /// Installation failed
    Failed,
    /// Held at current version
    Held,
}

/// Installed package record
#[derive(Clone, Debug)]
pub struct InstalledPackage {
    pub info: PackageInfo,
    pub state: InstallState,
    pub installed_at: u64,
    pub auto_installed: bool,
    pub config_files: Vec<String>,
}

/// Package manager errors
#[derive(Clone, Debug)]
pub enum PackageError {
    NotFound(String),
    AlreadyInstalled(String),
    NotInstalled(String),
    DependencyError(String),
    VersionConflict(String, String),
    DownloadFailed(String),
    VerificationFailed(String),
    InstallFailed(String),
    RemoveFailed(String),
    InsufficientSpace,
    PermissionDenied,
    DatabaseError(String),
    NetworkError(String),
}

/// Package operation
#[derive(Clone, Debug)]
pub enum Operation {
    Install(PackageInfo),
    Remove(String),
    Upgrade(String, Version),
    Downgrade(String, Version),
    Configure(String),
}

/// Package manager
pub struct PackageManager {
    /// Installed packages database
    installed: BTreeMap<String, InstalledPackage>,
    /// Available packages (from repositories)
    available: BTreeMap<String, Vec<PackageInfo>>,
    /// Package cache directory
    cache_dir: String,
    /// Installation root
    root_dir: String,
    /// Repository URLs
    repositories: Vec<Repository>,
    /// Current architecture
    architecture: Architecture,
    /// Pending operations
    pending: Vec<Operation>,
}

/// Repository definition
#[derive(Clone, Debug)]
pub struct Repository {
    pub name: String,
    pub url: String,
    pub enabled: bool,
    pub priority: u8,
    pub last_updated: u64,
}

impl PackageManager {
    /// Create new package manager
    pub fn new(root_dir: &str) -> Self {
        Self {
            installed: BTreeMap::new(),
            available: BTreeMap::new(),
            cache_dir: alloc::format!("{}/var/cache/pkg", root_dir),
            root_dir: String::from(root_dir),
            repositories: Vec::new(),
            architecture: Architecture::Arm64, // Default
            pending: Vec::new(),
        }
    }

    /// Set architecture
    pub fn set_architecture(&mut self, arch: Architecture) {
        self.architecture = arch;
    }

    /// Add repository
    pub fn add_repository(&mut self, repo: Repository) {
        self.repositories.push(repo);
    }

    /// Remove repository
    pub fn remove_repository(&mut self, name: &str) {
        self.repositories.retain(|r| r.name != name);
    }

    /// List repositories
    pub fn repositories(&self) -> &[Repository] {
        &self.repositories
    }

    /// Update package lists from repositories
    pub fn update(&mut self) -> Result<usize, PackageError> {
        // In real implementation, would fetch package lists from repos
        // For now, populate with mock data
        self.populate_mock_packages();
        Ok(self.available.len())
    }

    /// Populate mock packages for testing
    fn populate_mock_packages(&mut self) {
        let packages = [
            ("hublabio-kernel", "0.1.0", "HubLab IO Kernel", PackageCategory::System),
            ("hublabio-shell", "0.1.0", "HubLab IO Shell", PackageCategory::System),
            ("hublabio-ai-runtime", "0.1.0", "AI Runtime Engine", PackageCategory::AI),
            ("qwen2-0.5b", "1.0.0", "Qwen2 0.5B Language Model", PackageCategory::AI),
            ("whisper-tiny", "1.0.0", "Whisper Tiny Speech Recognition", PackageCategory::AI),
            ("piper-tts", "1.0.0", "Piper Text-to-Speech", PackageCategory::AI),
            ("llama-runtime", "0.1.0", "LLaMA Inference Runtime", PackageCategory::Runtime),
            ("gpio-driver", "0.1.0", "GPIO Driver", PackageCategory::Driver),
            ("wifi-driver", "0.1.0", "WiFi Driver", PackageCategory::Driver),
            ("bluetooth-driver", "0.1.0", "Bluetooth Driver", PackageCategory::Driver),
            ("file-manager", "0.1.0", "File Manager App", PackageCategory::App),
            ("system-monitor", "0.1.0", "System Monitor App", PackageCategory::App),
            ("settings", "0.1.0", "Settings App", PackageCategory::App),
            ("notes", "0.1.0", "Notes App", PackageCategory::App),
            ("calculator", "0.1.0", "Calculator App", PackageCategory::App),
        ];

        for (name, ver, desc, cat) in packages {
            let version = Version::parse(ver).unwrap_or(Version::new(0, 1, 0));
            let mut info = PackageInfo::new(name, version);
            info.description = String::from(desc);
            info.category = cat;
            info.architecture = self.architecture;
            info.size = 1024 * 1024; // 1MB placeholder
            info.installed_size = 2 * 1024 * 1024;

            self.available
                .entry(String::from(name))
                .or_insert_with(Vec::new)
                .push(info);
        }
    }

    /// Search for packages
    pub fn search(&self, query: &str) -> Vec<&PackageInfo> {
        let query_lower = query.to_lowercase();

        self.available
            .values()
            .flatten()
            .filter(|pkg| {
                pkg.name.to_lowercase().contains(&query_lower) ||
                pkg.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Get package info
    pub fn get_package(&self, name: &str) -> Option<&PackageInfo> {
        self.available
            .get(name)
            .and_then(|versions| versions.last())
    }

    /// Get all versions of a package
    pub fn get_versions(&self, name: &str) -> Option<&Vec<PackageInfo>> {
        self.available.get(name)
    }

    /// Check if package is installed
    pub fn is_installed(&self, name: &str) -> bool {
        self.installed.contains_key(name)
    }

    /// Get installed package
    pub fn get_installed(&self, name: &str) -> Option<&InstalledPackage> {
        self.installed.get(name)
    }

    /// List installed packages
    pub fn list_installed(&self) -> Vec<&InstalledPackage> {
        self.installed.values().collect()
    }

    /// List available packages
    pub fn list_available(&self) -> Vec<&PackageInfo> {
        self.available.values().flatten().collect()
    }

    /// List upgradable packages
    pub fn list_upgradable(&self) -> Vec<(&InstalledPackage, &PackageInfo)> {
        self.installed
            .values()
            .filter_map(|installed| {
                self.available
                    .get(&installed.info.name)
                    .and_then(|versions| versions.last())
                    .filter(|available| available.version > installed.info.version)
                    .map(|available| (installed, available))
            })
            .collect()
    }

    /// Resolve dependencies for a package
    pub fn resolve_dependencies(&self, name: &str) -> Result<Vec<String>, PackageError> {
        let package = self.get_package(name)
            .ok_or_else(|| PackageError::NotFound(String::from(name)))?;

        let mut resolved = Vec::new();
        let mut to_resolve = vec![package.name.clone()];
        let mut visited = alloc::collections::BTreeSet::new();

        while let Some(pkg_name) = to_resolve.pop() {
            if visited.contains(&pkg_name) {
                continue;
            }
            visited.insert(pkg_name.clone());

            if let Some(pkg) = self.get_package(&pkg_name) {
                for dep in &pkg.dependencies {
                    if !self.is_installed(&dep.name) && !dep.optional {
                        to_resolve.push(dep.name.clone());
                    }
                }
                resolved.push(pkg_name);
            }
        }

        Ok(resolved)
    }

    /// Install a package
    pub fn install(&mut self, name: &str) -> Result<(), PackageError> {
        if self.is_installed(name) {
            return Err(PackageError::AlreadyInstalled(String::from(name)));
        }

        let package = self.get_package(name)
            .ok_or_else(|| PackageError::NotFound(String::from(name)))?
            .clone();

        // Resolve dependencies
        let deps = self.resolve_dependencies(name)?;

        // Install dependencies first
        for dep in deps {
            if dep != name && !self.is_installed(&dep) {
                self.do_install(&dep)?;
            }
        }

        // Install the package
        self.do_install(name)?;

        Ok(())
    }

    /// Perform actual installation
    fn do_install(&mut self, name: &str) -> Result<(), PackageError> {
        let package = self.get_package(name)
            .ok_or_else(|| PackageError::NotFound(String::from(name)))?
            .clone();

        // In real implementation:
        // 1. Download package
        // 2. Verify checksum
        // 3. Extract files
        // 4. Run post-install scripts

        let installed = InstalledPackage {
            info: package,
            state: InstallState::Installed,
            installed_at: 0, // Would use real timestamp
            auto_installed: false,
            config_files: Vec::new(),
        };

        self.installed.insert(String::from(name), installed);
        Ok(())
    }

    /// Remove a package
    pub fn remove(&mut self, name: &str) -> Result<(), PackageError> {
        if !self.is_installed(name) {
            return Err(PackageError::NotInstalled(String::from(name)));
        }

        // Check if any installed package depends on this one
        for installed in self.installed.values() {
            for dep in &installed.info.dependencies {
                if dep.name == name && !dep.optional {
                    return Err(PackageError::DependencyError(
                        alloc::format!("{} depends on {}", installed.info.name, name)
                    ));
                }
            }
        }

        // In real implementation:
        // 1. Run pre-remove scripts
        // 2. Remove files
        // 3. Remove from database

        self.installed.remove(name);
        Ok(())
    }

    /// Upgrade a package
    pub fn upgrade(&mut self, name: &str) -> Result<(), PackageError> {
        let installed = self.get_installed(name)
            .ok_or_else(|| PackageError::NotInstalled(String::from(name)))?;

        let available = self.get_package(name)
            .ok_or_else(|| PackageError::NotFound(String::from(name)))?;

        if available.version <= installed.info.version {
            return Ok(()); // Already at latest version
        }

        // Remove old version and install new
        self.installed.remove(name);
        self.do_install(name)?;

        Ok(())
    }

    /// Upgrade all packages
    pub fn upgrade_all(&mut self) -> Result<usize, PackageError> {
        let upgradable: Vec<String> = self.list_upgradable()
            .iter()
            .map(|(pkg, _)| pkg.info.name.clone())
            .collect();

        let count = upgradable.len();

        for name in upgradable {
            self.upgrade(&name)?;
        }

        Ok(count)
    }

    /// Clean package cache
    pub fn clean_cache(&mut self) -> Result<u64, PackageError> {
        // Would remove cached package files
        Ok(0)
    }

    /// Auto-remove unused dependencies
    pub fn autoremove(&mut self) -> Result<Vec<String>, PackageError> {
        let mut removed = Vec::new();

        // Find auto-installed packages that are no longer needed
        let to_remove: Vec<String> = self.installed
            .values()
            .filter(|pkg| pkg.auto_installed)
            .filter(|pkg| {
                // Check if any package still depends on this
                !self.installed.values().any(|other| {
                    other.info.dependencies.iter().any(|dep| dep.name == pkg.info.name)
                })
            })
            .map(|pkg| pkg.info.name.clone())
            .collect();

        for name in to_remove {
            self.installed.remove(&name);
            removed.push(name);
        }

        Ok(removed)
    }

    /// Get package statistics
    pub fn statistics(&self) -> PackageStatistics {
        PackageStatistics {
            installed_count: self.installed.len(),
            available_count: self.available.values().map(|v| v.len()).sum(),
            upgradable_count: self.list_upgradable().len(),
            total_installed_size: self.installed.values()
                .map(|p| p.info.installed_size)
                .sum(),
        }
    }
}

/// Package statistics
#[derive(Clone, Debug)]
pub struct PackageStatistics {
    pub installed_count: usize,
    pub available_count: usize,
    pub upgradable_count: usize,
    pub total_installed_size: u64,
}

/// Package file format (simplified TOML-like)
pub mod manifest {
    use super::*;

    /// Parse package manifest
    pub fn parse(content: &str) -> Option<PackageInfo> {
        let mut name = String::new();
        let mut version = Version::new(0, 1, 0);
        let mut description = String::new();
        let mut category = PackageCategory::App;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');

                match key {
                    "name" => name = String::from(value),
                    "version" => {
                        if let Some(v) = Version::parse(value) {
                            version = v;
                        }
                    }
                    "description" => description = String::from(value),
                    "category" => {
                        if let Some(c) = PackageCategory::from_str(value) {
                            category = c;
                        }
                    }
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            return None;
        }

        let mut info = PackageInfo::new(&name, version);
        info.description = description;
        info.category = category;
        Some(info)
    }

    /// Generate package manifest
    pub fn generate(info: &PackageInfo) -> String {
        let mut manifest = String::new();

        manifest.push_str(&alloc::format!("name = \"{}\"\n", info.name));
        manifest.push_str(&alloc::format!("version = \"{}\"\n", info.version.to_string()));
        manifest.push_str(&alloc::format!("description = \"{}\"\n", info.description));
        manifest.push_str(&alloc::format!("category = \"{}\"\n", info.category.as_str()));
        manifest.push_str(&alloc::format!("architecture = \"{}\"\n", info.architecture.as_str()));
        manifest.push_str(&alloc::format!("license = \"{}\"\n", info.license));

        if !info.dependencies.is_empty() {
            manifest.push_str("\n[dependencies]\n");
            for dep in &info.dependencies {
                manifest.push_str(&alloc::format!("{} = \"*\"\n", dep.name));
            }
        }

        manifest
    }
}

/// CLI interface for package manager
pub mod cli {
    use super::*;

    /// Package manager command
    #[derive(Clone, Debug)]
    pub enum Command {
        Update,
        Upgrade,
        Install(Vec<String>),
        Remove(Vec<String>),
        Search(String),
        Show(String),
        List,
        ListInstalled,
        ListUpgradable,
        Clean,
        Autoremove,
        AddRepo(String, String),
        RemoveRepo(String),
        Help,
    }

    /// Parse command line
    pub fn parse_args(args: &[&str]) -> Option<Command> {
        if args.is_empty() {
            return Some(Command::Help);
        }

        match args[0] {
            "update" => Some(Command::Update),
            "upgrade" => Some(Command::Upgrade),
            "install" => {
                if args.len() > 1 {
                    Some(Command::Install(args[1..].iter().map(|s| String::from(*s)).collect()))
                } else {
                    None
                }
            }
            "remove" | "uninstall" => {
                if args.len() > 1 {
                    Some(Command::Remove(args[1..].iter().map(|s| String::from(*s)).collect()))
                } else {
                    None
                }
            }
            "search" => {
                if args.len() > 1 {
                    Some(Command::Search(String::from(args[1])))
                } else {
                    None
                }
            }
            "show" | "info" => {
                if args.len() > 1 {
                    Some(Command::Show(String::from(args[1])))
                } else {
                    None
                }
            }
            "list" => Some(Command::List),
            "list-installed" => Some(Command::ListInstalled),
            "list-upgradable" => Some(Command::ListUpgradable),
            "clean" => Some(Command::Clean),
            "autoremove" => Some(Command::Autoremove),
            "add-repo" => {
                if args.len() > 2 {
                    Some(Command::AddRepo(String::from(args[1]), String::from(args[2])))
                } else {
                    None
                }
            }
            "remove-repo" => {
                if args.len() > 1 {
                    Some(Command::RemoveRepo(String::from(args[1])))
                } else {
                    None
                }
            }
            "help" | "--help" | "-h" => Some(Command::Help),
            _ => None,
        }
    }

    /// Format help text
    pub fn help_text() -> String {
        String::from(
            "HubLab IO Package Manager (pkg)\n\n\
             Usage: pkg <command> [options]\n\n\
             Commands:\n\
               update            Update package lists\n\
               upgrade           Upgrade all packages\n\
               install <pkg>     Install package(s)\n\
               remove <pkg>      Remove package(s)\n\
               search <query>    Search for packages\n\
               show <pkg>        Show package details\n\
               list              List available packages\n\
               list-installed    List installed packages\n\
               list-upgradable   List upgradable packages\n\
               clean             Clean package cache\n\
               autoremove        Remove unused dependencies\n\
               add-repo <n> <u>  Add repository\n\
               remove-repo <n>   Remove repository\n\
               help              Show this help\n"
        )
    }

    /// Execute command
    pub fn execute(pm: &mut PackageManager, cmd: Command) -> String {
        match cmd {
            Command::Update => {
                match pm.update() {
                    Ok(count) => alloc::format!("Updated. {} packages available.", count),
                    Err(e) => alloc::format!("Update failed: {:?}", e),
                }
            }
            Command::Upgrade => {
                match pm.upgrade_all() {
                    Ok(count) => alloc::format!("Upgraded {} packages.", count),
                    Err(e) => alloc::format!("Upgrade failed: {:?}", e),
                }
            }
            Command::Install(packages) => {
                let mut results = Vec::new();
                for pkg in packages {
                    match pm.install(&pkg) {
                        Ok(_) => results.push(alloc::format!("Installed: {}", pkg)),
                        Err(e) => results.push(alloc::format!("Failed to install {}: {:?}", pkg, e)),
                    }
                }
                results.join("\n")
            }
            Command::Remove(packages) => {
                let mut results = Vec::new();
                for pkg in packages {
                    match pm.remove(&pkg) {
                        Ok(_) => results.push(alloc::format!("Removed: {}", pkg)),
                        Err(e) => results.push(alloc::format!("Failed to remove {}: {:?}", pkg, e)),
                    }
                }
                results.join("\n")
            }
            Command::Search(query) => {
                let results = pm.search(&query);
                if results.is_empty() {
                    String::from("No packages found.")
                } else {
                    results.iter()
                        .map(|p| alloc::format!("{} - {}", p.full_name(), p.description))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            Command::Show(name) => {
                if let Some(pkg) = pm.get_package(&name) {
                    alloc::format!(
                        "Package: {}\n\
                         Version: {}\n\
                         Category: {}\n\
                         Architecture: {}\n\
                         Description: {}\n\
                         Size: {} KB\n\
                         Installed: {}",
                        pkg.name,
                        pkg.version.to_string(),
                        pkg.category.as_str(),
                        pkg.architecture.as_str(),
                        pkg.description,
                        pkg.size / 1024,
                        if pm.is_installed(&name) { "Yes" } else { "No" }
                    )
                } else {
                    alloc::format!("Package not found: {}", name)
                }
            }
            Command::List => {
                let packages = pm.list_available();
                packages.iter()
                    .map(|p| alloc::format!("{:<30} {}", p.full_name(), p.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            Command::ListInstalled => {
                let packages = pm.list_installed();
                if packages.is_empty() {
                    String::from("No packages installed.")
                } else {
                    packages.iter()
                        .map(|p| alloc::format!("{:<30} {}", p.info.full_name(), p.info.description))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            Command::ListUpgradable => {
                let upgrades = pm.list_upgradable();
                if upgrades.is_empty() {
                    String::from("All packages are up to date.")
                } else {
                    upgrades.iter()
                        .map(|(old, new)| {
                            alloc::format!("{}: {} -> {}",
                                old.info.name,
                                old.info.version.to_string(),
                                new.version.to_string()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            Command::Clean => {
                match pm.clean_cache() {
                    Ok(freed) => alloc::format!("Cache cleaned. {} KB freed.", freed / 1024),
                    Err(e) => alloc::format!("Clean failed: {:?}", e),
                }
            }
            Command::Autoremove => {
                match pm.autoremove() {
                    Ok(removed) => {
                        if removed.is_empty() {
                            String::from("No packages to remove.")
                        } else {
                            alloc::format!("Removed: {}", removed.join(", "))
                        }
                    }
                    Err(e) => alloc::format!("Autoremove failed: {:?}", e),
                }
            }
            Command::AddRepo(name, url) => {
                pm.add_repository(Repository {
                    name: name.clone(),
                    url,
                    enabled: true,
                    priority: 100,
                    last_updated: 0,
                });
                alloc::format!("Repository '{}' added.", name)
            }
            Command::RemoveRepo(name) => {
                pm.remove_repository(&name);
                alloc::format!("Repository '{}' removed.", name)
            }
            Command::Help => help_text(),
        }
    }
}
