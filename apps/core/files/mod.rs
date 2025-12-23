//! File Manager Application
//!
//! A graphical file manager for HubLab IO.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::format;

/// File entry information
#[derive(Clone, Debug)]
pub struct FileEntry {
    /// File name
    pub name: String,
    /// Full path
    pub path: String,
    /// Is directory
    pub is_dir: bool,
    /// File size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub modified: u64,
    /// File permissions
    pub permissions: u16,
    /// Is hidden file
    pub hidden: bool,
    /// Is selected
    pub selected: bool,
}

impl FileEntry {
    /// Format file size for display
    pub fn format_size(&self) -> String {
        if self.is_dir {
            String::from("<DIR>")
        } else if self.size < 1024 {
            format!("{} B", self.size)
        } else if self.size < 1024 * 1024 {
            format!("{:.1} KB", self.size as f64 / 1024.0)
        } else if self.size < 1024 * 1024 * 1024 {
            format!("{:.1} MB", self.size as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", self.size as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Get file extension
    pub fn extension(&self) -> Option<&str> {
        if self.is_dir {
            return None;
        }
        self.name.rsplit('.').next()
    }

    /// Get icon for file type
    pub fn icon(&self) -> &'static str {
        if self.is_dir {
            return "folder";
        }

        match self.extension() {
            Some("rs") | Some("py") | Some("js") | Some("c") | Some("cpp") => "code",
            Some("txt") | Some("md") | Some("log") => "text",
            Some("jpg") | Some("png") | Some("gif") | Some("bmp") => "image",
            Some("mp3") | Some("wav") | Some("ogg") | Some("flac") => "audio",
            Some("mp4") | Some("avi") | Some("mkv") | Some("webm") => "video",
            Some("zip") | Some("tar") | Some("gz") | Some("xz") => "archive",
            Some("pdf") => "pdf",
            Some("gguf") | Some("bin") => "ai-model",
            Some("conf") | Some("toml") | Some("yaml") | Some("json") => "config",
            _ => "file",
        }
    }
}

/// Sort order for files
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortOrder {
    NameAsc,
    NameDesc,
    SizeAsc,
    SizeDesc,
    DateAsc,
    DateDesc,
    TypeAsc,
    TypeDesc,
}

/// View mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    List,
    Grid,
    Details,
}

/// File manager configuration
#[derive(Clone, Debug)]
pub struct FileManagerConfig {
    /// Show hidden files
    pub show_hidden: bool,
    /// Sort order
    pub sort_order: SortOrder,
    /// View mode
    pub view_mode: ViewMode,
    /// Show file extensions
    pub show_extensions: bool,
    /// Confirm before delete
    pub confirm_delete: bool,
    /// Show preview panel
    pub show_preview: bool,
    /// Show path bar
    pub show_path_bar: bool,
    /// Show status bar
    pub show_status_bar: bool,
}

impl Default for FileManagerConfig {
    fn default() -> Self {
        Self {
            show_hidden: false,
            sort_order: SortOrder::NameAsc,
            view_mode: ViewMode::Details,
            show_extensions: true,
            confirm_delete: true,
            show_preview: true,
            show_path_bar: true,
            show_status_bar: true,
        }
    }
}

/// Clipboard operation
#[derive(Clone, Debug)]
pub enum ClipboardOp {
    Copy(Vec<String>),
    Cut(Vec<String>),
}

/// File operation result
#[derive(Clone, Debug)]
pub enum FileOpResult {
    Success,
    Error(String),
    Cancelled,
    InProgress(u8), // Progress percentage
}

/// File manager state
pub struct FileManager {
    /// Current directory path
    current_path: String,
    /// Directory history (for back/forward)
    history: Vec<String>,
    /// History index
    history_index: usize,
    /// Current directory entries
    entries: Vec<FileEntry>,
    /// Selected entry index
    selected_index: usize,
    /// Configuration
    config: FileManagerConfig,
    /// Clipboard
    clipboard: Option<ClipboardOp>,
    /// Search query
    search_query: Option<String>,
    /// Filtered entries (when searching)
    filtered_indices: Option<Vec<usize>>,
    /// Bookmarks
    bookmarks: Vec<(String, String)>, // (name, path)
    /// Current operation
    current_op: Option<FileOpResult>,
}

impl FileManager {
    /// Create new file manager
    pub fn new() -> Self {
        Self {
            current_path: String::from("/"),
            history: alloc::vec![String::from("/")],
            history_index: 0,
            entries: Vec::new(),
            selected_index: 0,
            config: FileManagerConfig::default(),
            clipboard: None,
            search_query: None,
            filtered_indices: None,
            bookmarks: Self::default_bookmarks(),
            current_op: None,
        }
    }

    /// Default bookmarks
    fn default_bookmarks() -> Vec<(String, String)> {
        alloc::vec![
            (String::from("Home"), String::from("/home")),
            (String::from("Documents"), String::from("/home/documents")),
            (String::from("Downloads"), String::from("/home/downloads")),
            (String::from("Models"), String::from("/models")),
            (String::from("System"), String::from("/")),
        ]
    }

    /// Get current path
    pub fn current_path(&self) -> &str {
        &self.current_path
    }

    /// Get entries
    pub fn entries(&self) -> &[FileEntry] {
        &self.entries
    }

    /// Get visible entries (filtered if searching)
    pub fn visible_entries(&self) -> Vec<&FileEntry> {
        if let Some(ref indices) = self.filtered_indices {
            indices.iter()
                .filter_map(|&i| self.entries.get(i))
                .collect()
        } else {
            self.entries.iter().collect()
        }
    }

    /// Get selected index
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    /// Get selected entry
    pub fn selected_entry(&self) -> Option<&FileEntry> {
        let visible = self.visible_entries();
        visible.get(self.selected_index).copied()
    }

    /// Get selected entries (multi-select)
    pub fn selected_entries(&self) -> Vec<&FileEntry> {
        self.entries.iter()
            .filter(|e| e.selected)
            .collect()
    }

    /// Navigate to path
    pub fn navigate(&mut self, path: &str) -> Result<(), FileError> {
        // Validate path exists (would call VFS in real implementation)
        let normalized = self.normalize_path(path);

        // Load directory contents
        self.load_directory(&normalized)?;

        // Update history
        if self.history_index < self.history.len() - 1 {
            self.history.truncate(self.history_index + 1);
        }
        self.history.push(normalized.clone());
        self.history_index = self.history.len() - 1;

        self.current_path = normalized;
        self.selected_index = 0;
        self.clear_search();

        Ok(())
    }

    /// Navigate to parent directory
    pub fn go_up(&mut self) -> Result<(), FileError> {
        if self.current_path == "/" {
            return Ok(());
        }

        let parent = self.parent_path(&self.current_path);
        self.navigate(&parent)
    }

    /// Go back in history
    pub fn go_back(&mut self) -> Result<(), FileError> {
        if self.history_index > 0 {
            self.history_index -= 1;
            let path = self.history[self.history_index].clone();
            self.load_directory(&path)?;
            self.current_path = path;
            self.selected_index = 0;
        }
        Ok(())
    }

    /// Go forward in history
    pub fn go_forward(&mut self) -> Result<(), FileError> {
        if self.history_index < self.history.len() - 1 {
            self.history_index += 1;
            let path = self.history[self.history_index].clone();
            self.load_directory(&path)?;
            self.current_path = path;
            self.selected_index = 0;
        }
        Ok(())
    }

    /// Refresh current directory
    pub fn refresh(&mut self) -> Result<(), FileError> {
        self.load_directory(&self.current_path.clone())
    }

    /// Load directory contents
    fn load_directory(&mut self, path: &str) -> Result<(), FileError> {
        // In real implementation, this would call VFS readdir
        // For now, create mock entries
        self.entries.clear();

        // Add parent directory entry if not root
        if path != "/" {
            self.entries.push(FileEntry {
                name: String::from(".."),
                path: self.parent_path(path),
                is_dir: true,
                size: 0,
                modified: 0,
                permissions: 0o755,
                hidden: false,
                selected: false,
            });
        }

        // Mock entries (would be from VFS)
        let mock_entries = self.mock_directory_entries(path);
        self.entries.extend(mock_entries);

        // Sort entries
        self.sort_entries();

        Ok(())
    }

    /// Mock directory entries for testing
    fn mock_directory_entries(&self, path: &str) -> Vec<FileEntry> {
        match path {
            "/" => alloc::vec![
                FileEntry { name: String::from("home"), path: String::from("/home"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: false, selected: false },
                FileEntry { name: String::from("etc"), path: String::from("/etc"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: false, selected: false },
                FileEntry { name: String::from("var"), path: String::from("/var"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: false, selected: false },
                FileEntry { name: String::from("tmp"), path: String::from("/tmp"), is_dir: true, size: 0, modified: 0, permissions: 0o777, hidden: false, selected: false },
                FileEntry { name: String::from("dev"), path: String::from("/dev"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: false, selected: false },
                FileEntry { name: String::from("proc"), path: String::from("/proc"), is_dir: true, size: 0, modified: 0, permissions: 0o555, hidden: false, selected: false },
                FileEntry { name: String::from("sys"), path: String::from("/sys"), is_dir: true, size: 0, modified: 0, permissions: 0o555, hidden: false, selected: false },
                FileEntry { name: String::from("models"), path: String::from("/models"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: false, selected: false },
            ],
            "/home" => alloc::vec![
                FileEntry { name: String::from("documents"), path: String::from("/home/documents"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: false, selected: false },
                FileEntry { name: String::from("downloads"), path: String::from("/home/downloads"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: false, selected: false },
                FileEntry { name: String::from(".config"), path: String::from("/home/.config"), is_dir: true, size: 0, modified: 0, permissions: 0o755, hidden: true, selected: false },
                FileEntry { name: String::from("readme.txt"), path: String::from("/home/readme.txt"), is_dir: false, size: 1234, modified: 0, permissions: 0o644, hidden: false, selected: false },
            ],
            "/models" => alloc::vec![
                FileEntry { name: String::from("tinyllama-1.1b-q4.gguf"), path: String::from("/models/tinyllama-1.1b-q4.gguf"), is_dir: false, size: 668_000_000, modified: 0, permissions: 0o644, hidden: false, selected: false },
                FileEntry { name: String::from("qwen2-0.5b-q4.gguf"), path: String::from("/models/qwen2-0.5b-q4.gguf"), is_dir: false, size: 394_000_000, modified: 0, permissions: 0o644, hidden: false, selected: false },
                FileEntry { name: String::from("whisper-tiny.gguf"), path: String::from("/models/whisper-tiny.gguf"), is_dir: false, size: 75_000_000, modified: 0, permissions: 0o644, hidden: false, selected: false },
            ],
            _ => Vec::new(),
        }
    }

    /// Sort entries
    fn sort_entries(&mut self) {
        // Directories always first
        self.entries.sort_by(|a, b| {
            // ".." always first
            if a.name == ".." { return core::cmp::Ordering::Less; }
            if b.name == ".." { return core::cmp::Ordering::Greater; }

            // Directories before files
            match (a.is_dir, b.is_dir) {
                (true, false) => return core::cmp::Ordering::Less,
                (false, true) => return core::cmp::Ordering::Greater,
                _ => {}
            }

            // Apply sort order
            match self.config.sort_order {
                SortOrder::NameAsc => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                SortOrder::NameDesc => b.name.to_lowercase().cmp(&a.name.to_lowercase()),
                SortOrder::SizeAsc => a.size.cmp(&b.size),
                SortOrder::SizeDesc => b.size.cmp(&a.size),
                SortOrder::DateAsc => a.modified.cmp(&b.modified),
                SortOrder::DateDesc => b.modified.cmp(&a.modified),
                SortOrder::TypeAsc => a.extension().cmp(&b.extension()),
                SortOrder::TypeDesc => b.extension().cmp(&a.extension()),
            }
        });

        // Filter hidden files if needed
        if !self.config.show_hidden {
            self.entries.retain(|e| !e.hidden || e.name == "..");
        }
    }

    /// Normalize path
    fn normalize_path(&self, path: &str) -> String {
        let mut parts: Vec<&str> = Vec::new();

        for part in path.split('/') {
            match part {
                "" | "." => continue,
                ".." => { parts.pop(); }
                p => parts.push(p),
            }
        }

        if parts.is_empty() {
            String::from("/")
        } else {
            format!("/{}", parts.join("/"))
        }
    }

    /// Get parent path
    fn parent_path(&self, path: &str) -> String {
        if path == "/" {
            return String::from("/");
        }

        let trimmed = path.trim_end_matches('/');
        if let Some(pos) = trimmed.rfind('/') {
            if pos == 0 {
                String::from("/")
            } else {
                String::from(&trimmed[..pos])
            }
        } else {
            String::from("/")
        }
    }

    /// Move selection up
    pub fn select_up(&mut self) {
        let count = self.visible_entries().len();
        if count > 0 && self.selected_index > 0 {
            self.selected_index -= 1;
        }
    }

    /// Move selection down
    pub fn select_down(&mut self) {
        let count = self.visible_entries().len();
        if count > 0 && self.selected_index < count - 1 {
            self.selected_index += 1;
        }
    }

    /// Toggle selection on current entry
    pub fn toggle_selection(&mut self) {
        if let Some(ref indices) = self.filtered_indices {
            if let Some(&real_idx) = indices.get(self.selected_index) {
                if let Some(entry) = self.entries.get_mut(real_idx) {
                    entry.selected = !entry.selected;
                }
            }
        } else if let Some(entry) = self.entries.get_mut(self.selected_index) {
            entry.selected = !entry.selected;
        }
    }

    /// Select all entries
    pub fn select_all(&mut self) {
        for entry in &mut self.entries {
            if entry.name != ".." {
                entry.selected = true;
            }
        }
    }

    /// Deselect all entries
    pub fn deselect_all(&mut self) {
        for entry in &mut self.entries {
            entry.selected = false;
        }
    }

    /// Activate selected entry (open file or navigate to directory)
    pub fn activate(&mut self) -> Result<FileAction, FileError> {
        let entry = self.selected_entry().cloned();

        if let Some(entry) = entry {
            if entry.is_dir {
                self.navigate(&entry.path)?;
                Ok(FileAction::Navigated)
            } else {
                Ok(FileAction::Open(entry.path))
            }
        } else {
            Err(FileError::NoSelection)
        }
    }

    /// Start search
    pub fn search(&mut self, query: &str) {
        if query.is_empty() {
            self.clear_search();
            return;
        }

        self.search_query = Some(String::from(query));
        let query_lower = query.to_lowercase();

        self.filtered_indices = Some(
            self.entries.iter()
                .enumerate()
                .filter(|(_, e)| e.name.to_lowercase().contains(&query_lower))
                .map(|(i, _)| i)
                .collect()
        );

        self.selected_index = 0;
    }

    /// Clear search
    pub fn clear_search(&mut self) {
        self.search_query = None;
        self.filtered_indices = None;
    }

    /// Copy selected files to clipboard
    pub fn copy(&mut self) {
        let paths: Vec<String> = self.selected_entries()
            .iter()
            .map(|e| e.path.clone())
            .collect();

        if !paths.is_empty() {
            self.clipboard = Some(ClipboardOp::Copy(paths));
        }
    }

    /// Cut selected files to clipboard
    pub fn cut(&mut self) {
        let paths: Vec<String> = self.selected_entries()
            .iter()
            .map(|e| e.path.clone())
            .collect();

        if !paths.is_empty() {
            self.clipboard = Some(ClipboardOp::Cut(paths));
        }
    }

    /// Paste from clipboard
    pub fn paste(&mut self) -> Result<(), FileError> {
        let op = self.clipboard.take();

        match op {
            Some(ClipboardOp::Copy(paths)) => {
                for path in &paths {
                    let filename = path.rsplit('/').next().unwrap_or("file");
                    let dest = format!("{}/{}", self.current_path, filename);
                    // Would call VFS copy here
                    log::info!("Copy {} -> {}", path, dest);
                }
                self.clipboard = Some(ClipboardOp::Copy(paths)); // Keep for more pastes
            }
            Some(ClipboardOp::Cut(paths)) => {
                for path in &paths {
                    let filename = path.rsplit('/').next().unwrap_or("file");
                    let dest = format!("{}/{}", self.current_path, filename);
                    // Would call VFS rename here
                    log::info!("Move {} -> {}", path, dest);
                }
                // Don't keep cut items
            }
            None => return Err(FileError::ClipboardEmpty),
        }

        self.refresh()
    }

    /// Delete selected files
    pub fn delete(&mut self) -> Result<(), FileError> {
        let to_delete: Vec<String> = self.selected_entries()
            .iter()
            .map(|e| e.path.clone())
            .collect();

        if to_delete.is_empty() {
            return Err(FileError::NoSelection);
        }

        for path in &to_delete {
            // Would call VFS unlink/rmdir here
            log::info!("Delete {}", path);
        }

        self.refresh()
    }

    /// Create new directory
    pub fn create_directory(&mut self, name: &str) -> Result<(), FileError> {
        if name.is_empty() || name.contains('/') {
            return Err(FileError::InvalidName);
        }

        let path = format!("{}/{}", self.current_path, name);
        // Would call VFS mkdir here
        log::info!("Create directory {}", path);

        self.refresh()
    }

    /// Rename selected file
    pub fn rename(&mut self, new_name: &str) -> Result<(), FileError> {
        let entry = self.selected_entry().cloned();

        if let Some(entry) = entry {
            if new_name.is_empty() || new_name.contains('/') {
                return Err(FileError::InvalidName);
            }

            let new_path = format!("{}/{}", self.parent_path(&entry.path), new_name);
            // Would call VFS rename here
            log::info!("Rename {} -> {}", entry.path, new_path);

            self.refresh()
        } else {
            Err(FileError::NoSelection)
        }
    }

    /// Get config
    pub fn config(&self) -> &FileManagerConfig {
        &self.config
    }

    /// Get mutable config
    pub fn config_mut(&mut self) -> &mut FileManagerConfig {
        &mut self.config
    }

    /// Get bookmarks
    pub fn bookmarks(&self) -> &[(String, String)] {
        &self.bookmarks
    }

    /// Add bookmark
    pub fn add_bookmark(&mut self, name: &str, path: &str) {
        self.bookmarks.push((String::from(name), String::from(path)));
    }

    /// Remove bookmark
    pub fn remove_bookmark(&mut self, index: usize) {
        if index < self.bookmarks.len() {
            self.bookmarks.remove(index);
        }
    }

    /// Get status text
    pub fn status_text(&self) -> String {
        let total = self.entries.len();
        let selected = self.entries.iter().filter(|e| e.selected).count();

        if selected > 0 {
            format!("{} items selected", selected)
        } else {
            format!("{} items", total)
        }
    }
}

impl Default for FileManager {
    fn default() -> Self {
        Self::new()
    }
}

/// File action result
#[derive(Clone, Debug)]
pub enum FileAction {
    Navigated,
    Open(String),
    Preview(String),
}

/// File manager errors
#[derive(Clone, Debug)]
pub enum FileError {
    NotFound,
    PermissionDenied,
    InvalidPath,
    InvalidName,
    NoSelection,
    ClipboardEmpty,
    AlreadyExists,
    NotEmpty,
    IoError(String),
}
