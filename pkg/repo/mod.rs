//! Package Repository Module
//!
//! Handles package repository management, indexing, and downloads.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;

/// Repository configuration
#[derive(Clone, Debug)]
pub struct RepoConfig {
    /// Repository name
    pub name: String,
    /// Base URL
    pub url: String,
    /// Repository type
    pub repo_type: RepoType,
    /// Enabled status
    pub enabled: bool,
    /// Priority (higher = preferred)
    pub priority: u8,
    /// GPG key ID for verification
    pub gpg_key: Option<String>,
    /// Distribution/release name
    pub distribution: String,
    /// Components to use
    pub components: Vec<String>,
}

/// Repository type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepoType {
    /// HubLab IO native format
    Native,
    /// Flat repository (single directory)
    Flat,
}

impl RepoConfig {
    /// Create new repository config
    pub fn new(name: &str, url: &str) -> Self {
        Self {
            name: String::from(name),
            url: String::from(url),
            repo_type: RepoType::Native,
            enabled: true,
            priority: 100,
            gpg_key: None,
            distribution: String::from("stable"),
            components: alloc::vec![String::from("main")],
        }
    }

    /// Get package index URL
    pub fn index_url(&self) -> String {
        match self.repo_type {
            RepoType::Native => {
                alloc::format!("{}/dists/{}/Packages.gz", self.url, self.distribution)
            }
            RepoType::Flat => {
                alloc::format!("{}/Packages.gz", self.url)
            }
        }
    }

    /// Get release file URL
    pub fn release_url(&self) -> String {
        match self.repo_type {
            RepoType::Native => {
                alloc::format!("{}/dists/{}/Release", self.url, self.distribution)
            }
            RepoType::Flat => {
                alloc::format!("{}/Release", self.url)
            }
        }
    }

    /// Get package download URL
    pub fn package_url(&self, filename: &str) -> String {
        alloc::format!("{}/{}", self.url, filename)
    }
}

impl Default for RepoConfig {
    fn default() -> Self {
        Self::new("default", "https://pkg.hublabio.dev")
    }
}

/// Package index entry
#[derive(Clone, Debug)]
pub struct IndexEntry {
    /// Package name
    pub name: String,
    /// Version string
    pub version: String,
    /// Architecture
    pub architecture: String,
    /// File path in repository
    pub filename: String,
    /// File size
    pub size: u64,
    /// SHA256 checksum
    pub sha256: String,
    /// Description
    pub description: String,
    /// Dependencies
    pub depends: Vec<String>,
    /// Section/category
    pub section: String,
    /// Maintainer
    pub maintainer: String,
}

impl IndexEntry {
    /// Parse from package index format
    pub fn parse(block: &str) -> Option<Self> {
        let mut entry = Self {
            name: String::new(),
            version: String::new(),
            architecture: String::new(),
            filename: String::new(),
            size: 0,
            sha256: String::new(),
            description: String::new(),
            depends: Vec::new(),
            section: String::new(),
            maintainer: String::new(),
        };

        for line in block.lines() {
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "Package" => entry.name = String::from(value),
                    "Version" => entry.version = String::from(value),
                    "Architecture" => entry.architecture = String::from(value),
                    "Filename" => entry.filename = String::from(value),
                    "Size" => entry.size = value.parse().unwrap_or(0),
                    "SHA256" => entry.sha256 = String::from(value),
                    "Description" => entry.description = String::from(value),
                    "Depends" => {
                        entry.depends = value
                            .split(',')
                            .map(|s| String::from(s.trim()))
                            .collect();
                    }
                    "Section" => entry.section = String::from(value),
                    "Maintainer" => entry.maintainer = String::from(value),
                    _ => {}
                }
            }
        }

        if entry.name.is_empty() || entry.version.is_empty() {
            return None;
        }

        Some(entry)
    }

    /// Generate package index entry
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        result.push_str(&alloc::format!("Package: {}\n", self.name));
        result.push_str(&alloc::format!("Version: {}\n", self.version));
        result.push_str(&alloc::format!("Architecture: {}\n", self.architecture));
        result.push_str(&alloc::format!("Filename: {}\n", self.filename));
        result.push_str(&alloc::format!("Size: {}\n", self.size));
        result.push_str(&alloc::format!("SHA256: {}\n", self.sha256));
        result.push_str(&alloc::format!("Description: {}\n", self.description));

        if !self.depends.is_empty() {
            result.push_str(&alloc::format!("Depends: {}\n", self.depends.join(", ")));
        }

        if !self.section.is_empty() {
            result.push_str(&alloc::format!("Section: {}\n", self.section));
        }

        if !self.maintainer.is_empty() {
            result.push_str(&alloc::format!("Maintainer: {}\n", self.maintainer));
        }

        result
    }
}

/// Repository index
pub struct RepoIndex {
    /// Repository configuration
    pub config: RepoConfig,
    /// Package entries indexed by name
    pub packages: BTreeMap<String, Vec<IndexEntry>>,
    /// Last update timestamp
    pub last_updated: u64,
    /// Release information
    pub release_info: ReleaseInfo,
}

/// Release file information
#[derive(Clone, Debug, Default)]
pub struct ReleaseInfo {
    pub origin: String,
    pub label: String,
    pub suite: String,
    pub codename: String,
    pub version: String,
    pub date: String,
    pub architectures: Vec<String>,
    pub components: Vec<String>,
    pub description: String,
}

impl ReleaseInfo {
    /// Parse release file
    pub fn parse(content: &str) -> Self {
        let mut info = Self::default();

        for line in content.lines() {
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "Origin" => info.origin = String::from(value),
                    "Label" => info.label = String::from(value),
                    "Suite" => info.suite = String::from(value),
                    "Codename" => info.codename = String::from(value),
                    "Version" => info.version = String::from(value),
                    "Date" => info.date = String::from(value),
                    "Architectures" => {
                        info.architectures = value
                            .split_whitespace()
                            .map(String::from)
                            .collect();
                    }
                    "Components" => {
                        info.components = value
                            .split_whitespace()
                            .map(String::from)
                            .collect();
                    }
                    "Description" => info.description = String::from(value),
                    _ => {}
                }
            }
        }

        info
    }

    /// Generate release file content
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        result.push_str(&alloc::format!("Origin: {}\n", self.origin));
        result.push_str(&alloc::format!("Label: {}\n", self.label));
        result.push_str(&alloc::format!("Suite: {}\n", self.suite));
        result.push_str(&alloc::format!("Codename: {}\n", self.codename));
        result.push_str(&alloc::format!("Version: {}\n", self.version));
        result.push_str(&alloc::format!("Date: {}\n", self.date));
        result.push_str(&alloc::format!("Architectures: {}\n", self.architectures.join(" ")));
        result.push_str(&alloc::format!("Components: {}\n", self.components.join(" ")));
        result.push_str(&alloc::format!("Description: {}\n", self.description));

        result
    }
}

impl RepoIndex {
    /// Create new repository index
    pub fn new(config: RepoConfig) -> Self {
        Self {
            config,
            packages: BTreeMap::new(),
            last_updated: 0,
            release_info: ReleaseInfo::default(),
        }
    }

    /// Parse package index
    pub fn parse_index(&mut self, content: &str) {
        // Split by double newline to get package blocks
        for block in content.split("\n\n") {
            if let Some(entry) = IndexEntry::parse(block) {
                self.packages
                    .entry(entry.name.clone())
                    .or_insert_with(Vec::new)
                    .push(entry);
            }
        }
    }

    /// Get package by name
    pub fn get_package(&self, name: &str) -> Option<&IndexEntry> {
        self.packages
            .get(name)
            .and_then(|versions| versions.last())
    }

    /// Get all versions of a package
    pub fn get_versions(&self, name: &str) -> Option<&Vec<IndexEntry>> {
        self.packages.get(name)
    }

    /// Search packages
    pub fn search(&self, query: &str) -> Vec<&IndexEntry> {
        let query_lower = query.to_lowercase();

        self.packages
            .values()
            .filter_map(|versions| versions.last())
            .filter(|entry| {
                entry.name.to_lowercase().contains(&query_lower) ||
                entry.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// List all packages
    pub fn list_packages(&self) -> Vec<&IndexEntry> {
        self.packages
            .values()
            .filter_map(|versions| versions.last())
            .collect()
    }

    /// Package count
    pub fn package_count(&self) -> usize {
        self.packages.len()
    }

    /// Generate package index content
    pub fn generate_index(&self) -> String {
        let mut result = String::new();

        for versions in self.packages.values() {
            for entry in versions {
                result.push_str(&entry.to_string());
                result.push_str("\n");
            }
        }

        result
    }
}

/// Repository manager
pub struct RepoManager {
    /// Configured repositories
    repos: Vec<RepoIndex>,
    /// Cache directory
    cache_dir: String,
}

impl RepoManager {
    /// Create new repository manager
    pub fn new(cache_dir: &str) -> Self {
        Self {
            repos: Vec::new(),
            cache_dir: String::from(cache_dir),
        }
    }

    /// Add repository
    pub fn add_repo(&mut self, config: RepoConfig) {
        self.repos.push(RepoIndex::new(config));
    }

    /// Remove repository
    pub fn remove_repo(&mut self, name: &str) {
        self.repos.retain(|r| r.config.name != name);
    }

    /// Get repository by name
    pub fn get_repo(&self, name: &str) -> Option<&RepoIndex> {
        self.repos.iter().find(|r| r.config.name == name)
    }

    /// Get repository by name (mutable)
    pub fn get_repo_mut(&mut self, name: &str) -> Option<&mut RepoIndex> {
        self.repos.iter_mut().find(|r| r.config.name == name)
    }

    /// List repositories
    pub fn list_repos(&self) -> &[RepoIndex] {
        &self.repos
    }

    /// Search all repositories
    pub fn search_all(&self, query: &str) -> Vec<(&RepoIndex, &IndexEntry)> {
        self.repos
            .iter()
            .filter(|r| r.config.enabled)
            .flat_map(|repo| {
                repo.search(query)
                    .into_iter()
                    .map(move |entry| (repo, entry))
            })
            .collect()
    }

    /// Get package from any repository
    pub fn get_package(&self, name: &str) -> Option<(&RepoIndex, &IndexEntry)> {
        self.repos
            .iter()
            .filter(|r| r.config.enabled)
            .sorted_by(|a, b| b.config.priority.cmp(&a.config.priority))
            .find_map(|repo| {
                repo.get_package(name).map(|entry| (repo, entry))
            })
    }

    /// Setup default repositories
    pub fn setup_defaults(&mut self) {
        // Main HubLab IO repository
        let mut main = RepoConfig::new("hublabio", "https://pkg.hublabio.dev");
        main.distribution = String::from("stable");
        main.components = alloc::vec![
            String::from("main"),
            String::from("ai"),
            String::from("drivers"),
        ];
        self.add_repo(main);

        // Community repository
        let mut community = RepoConfig::new("community", "https://pkg.hublabio.dev/community");
        community.priority = 50;
        self.add_repo(community);
    }
}

/// Sort helper trait
trait SortedBy<T> {
    fn sorted_by<F>(self, compare: F) -> alloc::vec::IntoIter<T>
    where
        F: FnMut(&T, &T) -> core::cmp::Ordering;
}

impl<T, I: Iterator<Item = T>> SortedBy<T> for I {
    fn sorted_by<F>(self, mut compare: F) -> alloc::vec::IntoIter<T>
    where
        F: FnMut(&T, &T) -> core::cmp::Ordering,
    {
        let mut vec: Vec<T> = self.collect();
        vec.sort_by(|a, b| compare(a, b));
        vec.into_iter()
    }
}

/// Download state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadState {
    Pending,
    Downloading,
    Verifying,
    Complete,
    Failed,
}

/// Download progress
#[derive(Clone, Debug)]
pub struct DownloadProgress {
    pub url: String,
    pub state: DownloadState,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub error: Option<String>,
}

impl DownloadProgress {
    pub fn new(url: &str, total: u64) -> Self {
        Self {
            url: String::from(url),
            state: DownloadState::Pending,
            bytes_downloaded: 0,
            total_bytes: total,
            error: None,
        }
    }

    pub fn percent(&self) -> u8 {
        if self.total_bytes == 0 {
            0
        } else {
            ((self.bytes_downloaded * 100) / self.total_bytes) as u8
        }
    }
}

/// Checksum verification
pub mod checksum {
    use alloc::string::String;

    /// Verify SHA256 checksum
    pub fn verify_sha256(data: &[u8], expected: &str) -> bool {
        let computed = sha256(data);
        computed == expected.to_lowercase()
    }

    /// Compute SHA256 (simplified - real implementation would use proper crypto)
    pub fn sha256(data: &[u8]) -> String {
        // Placeholder - real implementation would compute actual SHA256
        // Using a simple hash for demonstration
        let mut hash = 0u64;
        for &byte in data {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        alloc::format!("{:016x}{:016x}{:016x}{:016x}",
            hash, hash.rotate_left(16), hash.rotate_left(32), hash.rotate_left(48))
    }
}

/// Compression utilities
pub mod compression {
    use alloc::vec::Vec;

    /// Decompress gzip data
    pub fn decompress_gzip(data: &[u8]) -> Option<Vec<u8>> {
        // Placeholder - real implementation would decompress gzip
        // Check gzip magic number
        if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
            return None;
        }

        // For now, return data as-is (mock)
        Some(data.to_vec())
    }

    /// Compress to gzip
    pub fn compress_gzip(data: &[u8]) -> Vec<u8> {
        // Placeholder - real implementation would compress
        let mut result = vec![0x1f, 0x8b]; // gzip magic
        result.extend_from_slice(data);
        result
    }
}
