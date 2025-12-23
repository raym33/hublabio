//! Settings Application
//!
//! System configuration for HubLab IO.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

/// Settings category
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsCategory {
    General,
    Display,
    Sound,
    Network,
    Bluetooth,
    AI,
    Storage,
    Users,
    Security,
    Updates,
    About,
}

impl SettingsCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Display => "Display",
            Self::Sound => "Sound",
            Self::Network => "Network",
            Self::Bluetooth => "Bluetooth",
            Self::AI => "AI Models",
            Self::Storage => "Storage",
            Self::Users => "Users",
            Self::Security => "Security",
            Self::Updates => "Updates",
            Self::About => "About",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::General => "settings",
            Self::Display => "display",
            Self::Sound => "sound",
            Self::Network => "network",
            Self::Bluetooth => "bluetooth",
            Self::AI => "ai",
            Self::Storage => "storage",
            Self::Users => "users",
            Self::Security => "security",
            Self::Updates => "updates",
            Self::About => "info",
        }
    }

    pub fn all() -> &'static [SettingsCategory] {
        &[
            Self::General,
            Self::Display,
            Self::Sound,
            Self::Network,
            Self::Bluetooth,
            Self::AI,
            Self::Storage,
            Self::Users,
            Self::Security,
            Self::Updates,
            Self::About,
        ]
    }
}

/// Setting value types
#[derive(Clone, Debug)]
pub enum SettingValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Choice(usize, Vec<String>),
    Range { value: i64, min: i64, max: i64 },
    Color { r: u8, g: u8, b: u8 },
}

/// Individual setting
#[derive(Clone, Debug)]
pub struct Setting {
    /// Setting key
    pub key: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Current value
    pub value: SettingValue,
    /// Is read-only
    pub readonly: bool,
    /// Requires restart
    pub requires_restart: bool,
}

impl Setting {
    pub fn bool(key: &str, name: &str, desc: &str, value: bool) -> Self {
        Self {
            key: String::from(key),
            name: String::from(name),
            description: String::from(desc),
            value: SettingValue::Bool(value),
            readonly: false,
            requires_restart: false,
        }
    }

    pub fn choice(key: &str, name: &str, desc: &str, selected: usize, options: Vec<&str>) -> Self {
        Self {
            key: String::from(key),
            name: String::from(name),
            description: String::from(desc),
            value: SettingValue::Choice(selected, options.into_iter().map(String::from).collect()),
            readonly: false,
            requires_restart: false,
        }
    }

    pub fn range(key: &str, name: &str, desc: &str, value: i64, min: i64, max: i64) -> Self {
        Self {
            key: String::from(key),
            name: String::from(name),
            description: String::from(desc),
            value: SettingValue::Range { value, min, max },
            readonly: false,
            requires_restart: false,
        }
    }

    pub fn string(key: &str, name: &str, desc: &str, value: &str) -> Self {
        Self {
            key: String::from(key),
            name: String::from(name),
            description: String::from(desc),
            value: SettingValue::String(String::from(value)),
            readonly: false,
            requires_restart: false,
        }
    }

    pub fn readonly(mut self) -> Self {
        self.readonly = true;
        self
    }

    pub fn restart_required(mut self) -> Self {
        self.requires_restart = true;
        self
    }
}

/// General settings
#[derive(Clone, Debug)]
pub struct GeneralSettings {
    /// Hostname
    pub hostname: String,
    /// Timezone
    pub timezone: String,
    /// Language
    pub language: String,
    /// Date format
    pub date_format: String,
    /// Time format (12/24)
    pub time_24h: bool,
    /// Auto-login
    pub auto_login: bool,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            hostname: String::from("hublab"),
            timezone: String::from("UTC"),
            language: String::from("en_US"),
            date_format: String::from("YYYY-MM-DD"),
            time_24h: true,
            auto_login: false,
        }
    }
}

/// Display settings
#[derive(Clone, Debug)]
pub struct DisplaySettings {
    /// Screen brightness (0-100)
    pub brightness: u8,
    /// Theme
    pub theme: String,
    /// Font size
    pub font_size: u8,
    /// Screen timeout (seconds, 0 = never)
    pub screen_timeout: u32,
    /// Night mode enabled
    pub night_mode: bool,
    /// Night mode schedule
    pub night_mode_auto: bool,
    /// Night mode start hour
    pub night_start: u8,
    /// Night mode end hour
    pub night_end: u8,
    /// Resolution
    pub resolution: String,
    /// Refresh rate
    pub refresh_rate: u8,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            brightness: 80,
            theme: String::from("dark"),
            font_size: 14,
            screen_timeout: 300,
            night_mode: false,
            night_mode_auto: true,
            night_start: 22,
            night_end: 6,
            resolution: String::from("1920x1080"),
            refresh_rate: 60,
        }
    }
}

/// Sound settings
#[derive(Clone, Debug)]
pub struct SoundSettings {
    /// Master volume (0-100)
    pub volume: u8,
    /// Muted
    pub muted: bool,
    /// System sounds enabled
    pub system_sounds: bool,
    /// Voice feedback enabled
    pub voice_feedback: bool,
    /// Voice feedback volume
    pub voice_volume: u8,
    /// Default voice
    pub default_voice: String,
    /// Speaking rate
    pub speaking_rate: f32,
    /// Microphone volume
    pub mic_volume: u8,
    /// Noise cancellation
    pub noise_cancel: bool,
}

impl Default for SoundSettings {
    fn default() -> Self {
        Self {
            volume: 70,
            muted: false,
            system_sounds: true,
            voice_feedback: true,
            voice_volume: 80,
            default_voice: String::from("default"),
            speaking_rate: 1.0,
            mic_volume: 80,
            noise_cancel: true,
        }
    }
}

/// Network settings
#[derive(Clone, Debug)]
pub struct NetworkSettings {
    /// WiFi enabled
    pub wifi_enabled: bool,
    /// Ethernet enabled
    pub ethernet_enabled: bool,
    /// Current WiFi network
    pub wifi_ssid: Option<String>,
    /// Auto-connect
    pub wifi_auto_connect: bool,
    /// DNS servers
    pub dns_servers: Vec<String>,
    /// DHCP enabled
    pub dhcp: bool,
    /// Static IP (if not DHCP)
    pub static_ip: Option<String>,
    /// Gateway
    pub gateway: Option<String>,
    /// Proxy enabled
    pub proxy_enabled: bool,
    /// Proxy address
    pub proxy_address: Option<String>,
}

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            wifi_enabled: true,
            ethernet_enabled: true,
            wifi_ssid: None,
            wifi_auto_connect: true,
            dns_servers: alloc::vec![
                String::from("1.1.1.1"),
                String::from("8.8.8.8"),
            ],
            dhcp: true,
            static_ip: None,
            gateway: None,
            proxy_enabled: false,
            proxy_address: None,
        }
    }
}

/// AI settings
#[derive(Clone, Debug)]
pub struct AiSettings {
    /// Default model
    pub default_model: String,
    /// Max tokens
    pub max_tokens: u32,
    /// Temperature
    pub temperature: f32,
    /// Top-P
    pub top_p: f32,
    /// Enable AI assistant
    pub assistant_enabled: bool,
    /// Wake word enabled
    pub wake_word_enabled: bool,
    /// Wake word
    pub wake_word: String,
    /// Voice input enabled
    pub voice_input: bool,
    /// Auto-download models
    pub auto_download: bool,
    /// Max model cache size (MB)
    pub cache_size_mb: u32,
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            default_model: String::from("tinyllama-1.1b-q4"),
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            assistant_enabled: true,
            wake_word_enabled: true,
            wake_word: String::from("hey hublab"),
            voice_input: true,
            auto_download: false,
            cache_size_mb: 2048,
        }
    }
}

/// Security settings
#[derive(Clone, Debug)]
pub struct SecuritySettings {
    /// Screen lock enabled
    pub screen_lock: bool,
    /// Lock timeout (seconds)
    pub lock_timeout: u32,
    /// PIN enabled
    pub pin_enabled: bool,
    /// Fingerprint enabled (if hardware available)
    pub fingerprint_enabled: bool,
    /// Auto-updates security patches
    pub auto_security_updates: bool,
    /// Firewall enabled
    pub firewall_enabled: bool,
    /// Allow root login
    pub allow_root: bool,
    /// SSH enabled
    pub ssh_enabled: bool,
    /// SSH port
    pub ssh_port: u16,
}

impl Default for SecuritySettings {
    fn default() -> Self {
        Self {
            screen_lock: true,
            lock_timeout: 300,
            pin_enabled: false,
            fingerprint_enabled: false,
            auto_security_updates: true,
            firewall_enabled: true,
            allow_root: false,
            ssh_enabled: true,
            ssh_port: 22,
        }
    }
}

/// About information
#[derive(Clone, Debug)]
pub struct AboutInfo {
    /// OS name
    pub os_name: String,
    /// OS version
    pub os_version: String,
    /// Build number
    pub build_number: String,
    /// Build date
    pub build_date: String,
    /// Kernel version
    pub kernel_version: String,
    /// Device model
    pub device_model: String,
    /// CPU info
    pub cpu: String,
    /// RAM
    pub ram: String,
    /// Storage
    pub storage: String,
}

impl Default for AboutInfo {
    fn default() -> Self {
        Self {
            os_name: String::from("HubLab IO"),
            os_version: String::from("0.1.0"),
            build_number: String::from("prototype-001"),
            build_date: String::from("December 2024"),
            kernel_version: String::from("0.1.0"),
            device_model: String::from("Raspberry Pi 5"),
            cpu: String::from("ARM Cortex-A76 (4 cores)"),
            ram: String::from("4 GB"),
            storage: String::from("32 GB"),
        }
    }
}

/// WiFi network entry
#[derive(Clone, Debug)]
pub struct WifiNetwork {
    pub ssid: String,
    pub signal_strength: i32, // dBm
    pub secured: bool,
    pub connected: bool,
    pub saved: bool,
}

/// AI Model entry
#[derive(Clone, Debug)]
pub struct AiModel {
    pub name: String,
    pub path: String,
    pub size_mb: u32,
    pub loaded: bool,
    pub default: bool,
    pub description: String,
}

/// Settings app state
pub struct SettingsApp {
    /// Current category
    current_category: SettingsCategory,
    /// Category index (for navigation)
    category_index: usize,
    /// Setting index within category
    setting_index: usize,
    /// General settings
    general: GeneralSettings,
    /// Display settings
    display: DisplaySettings,
    /// Sound settings
    sound: SoundSettings,
    /// Network settings
    network: NetworkSettings,
    /// AI settings
    ai: AiSettings,
    /// Security settings
    security: SecuritySettings,
    /// About info
    about: AboutInfo,
    /// Available WiFi networks
    wifi_networks: Vec<WifiNetwork>,
    /// Installed AI models
    ai_models: Vec<AiModel>,
    /// Has unsaved changes
    has_changes: bool,
    /// Is scanning WiFi
    wifi_scanning: bool,
}

impl SettingsApp {
    /// Create new settings app
    pub fn new() -> Self {
        Self {
            current_category: SettingsCategory::General,
            category_index: 0,
            setting_index: 0,
            general: GeneralSettings::default(),
            display: DisplaySettings::default(),
            sound: SoundSettings::default(),
            network: NetworkSettings::default(),
            ai: AiSettings::default(),
            security: SecuritySettings::default(),
            about: AboutInfo::default(),
            wifi_networks: Vec::new(),
            ai_models: Self::mock_ai_models(),
            has_changes: false,
            wifi_scanning: false,
        }
    }

    /// Mock AI models
    fn mock_ai_models() -> Vec<AiModel> {
        alloc::vec![
            AiModel {
                name: String::from("TinyLlama 1.1B (Q4)"),
                path: String::from("/models/tinyllama-1.1b-q4.gguf"),
                size_mb: 668,
                loaded: true,
                default: true,
                description: String::from("Small, fast model for general chat"),
            },
            AiModel {
                name: String::from("Qwen2 0.5B (Q4)"),
                path: String::from("/models/qwen2-0.5b-q4.gguf"),
                size_mb: 394,
                loaded: false,
                default: false,
                description: String::from("Compact model, good for simple tasks"),
            },
            AiModel {
                name: String::from("Whisper Tiny"),
                path: String::from("/models/whisper-tiny.gguf"),
                size_mb: 75,
                loaded: false,
                default: false,
                description: String::from("Speech-to-text model"),
            },
        ]
    }

    /// Get current category
    pub fn current_category(&self) -> SettingsCategory {
        self.current_category
    }

    /// Set current category
    pub fn set_category(&mut self, category: SettingsCategory) {
        self.current_category = category;
        self.category_index = SettingsCategory::all()
            .iter()
            .position(|&c| c == category)
            .unwrap_or(0);
        self.setting_index = 0;
    }

    /// Navigate to next category
    pub fn next_category(&mut self) {
        let categories = SettingsCategory::all();
        if self.category_index < categories.len() - 1 {
            self.category_index += 1;
            self.current_category = categories[self.category_index];
            self.setting_index = 0;
        }
    }

    /// Navigate to previous category
    pub fn prev_category(&mut self) {
        if self.category_index > 0 {
            self.category_index -= 1;
            self.current_category = SettingsCategory::all()[self.category_index];
            self.setting_index = 0;
        }
    }

    /// Get settings for current category
    pub fn current_settings(&self) -> Vec<Setting> {
        match self.current_category {
            SettingsCategory::General => self.general_settings(),
            SettingsCategory::Display => self.display_settings(),
            SettingsCategory::Sound => self.sound_settings(),
            SettingsCategory::Network => self.network_settings(),
            SettingsCategory::AI => self.ai_settings(),
            SettingsCategory::Security => self.security_settings(),
            SettingsCategory::About => self.about_settings(),
            _ => Vec::new(),
        }
    }

    fn general_settings(&self) -> Vec<Setting> {
        alloc::vec![
            Setting::string("hostname", "Hostname", "Device network name", &self.general.hostname)
                .restart_required(),
            Setting::choice("language", "Language", "System language", 0, alloc::vec!["English", "Spanish", "French", "German", "Japanese", "Chinese"]),
            Setting::choice("timezone", "Timezone", "System timezone", 0, alloc::vec!["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]),
            Setting::bool("time_24h", "24-hour time", "Use 24-hour time format", self.general.time_24h),
            Setting::bool("auto_login", "Auto-login", "Skip login screen on boot", self.general.auto_login),
        ]
    }

    fn display_settings(&self) -> Vec<Setting> {
        alloc::vec![
            Setting::range("brightness", "Brightness", "Screen brightness", self.display.brightness as i64, 0, 100),
            Setting::choice("theme", "Theme", "Color theme", 0, alloc::vec!["Dark", "Light", "AMOLED", "AI Purple"]),
            Setting::range("font_size", "Font Size", "UI font size", self.display.font_size as i64, 10, 24),
            Setting::range("screen_timeout", "Screen Timeout", "Turn off screen after inactivity (seconds)", self.display.screen_timeout as i64, 0, 3600),
            Setting::bool("night_mode", "Night Mode", "Reduce blue light", self.display.night_mode),
            Setting::bool("night_auto", "Auto Night Mode", "Enable night mode on schedule", self.display.night_mode_auto),
        ]
    }

    fn sound_settings(&self) -> Vec<Setting> {
        alloc::vec![
            Setting::range("volume", "Volume", "Master volume", self.sound.volume as i64, 0, 100),
            Setting::bool("muted", "Mute", "Mute all sounds", self.sound.muted),
            Setting::bool("system_sounds", "System Sounds", "Play sounds for system events", self.sound.system_sounds),
            Setting::bool("voice_feedback", "Voice Feedback", "Speak responses", self.sound.voice_feedback),
            Setting::range("voice_volume", "Voice Volume", "Text-to-speech volume", self.sound.voice_volume as i64, 0, 100),
            Setting::range("mic_volume", "Microphone", "Microphone input volume", self.sound.mic_volume as i64, 0, 100),
            Setting::bool("noise_cancel", "Noise Cancellation", "Reduce background noise", self.sound.noise_cancel),
        ]
    }

    fn network_settings(&self) -> Vec<Setting> {
        alloc::vec![
            Setting::bool("wifi_enabled", "WiFi", "Enable wireless networking", self.network.wifi_enabled),
            Setting::bool("ethernet", "Ethernet", "Enable wired networking", self.network.ethernet_enabled),
            Setting::bool("dhcp", "DHCP", "Get IP address automatically", self.network.dhcp),
            Setting::bool("proxy", "Proxy", "Use network proxy", self.network.proxy_enabled),
        ]
    }

    fn ai_settings(&self) -> Vec<Setting> {
        alloc::vec![
            Setting::bool("assistant", "AI Assistant", "Enable AI assistant", self.ai.assistant_enabled),
            Setting::bool("wake_word", "Wake Word", "Listen for wake word", self.ai.wake_word_enabled),
            Setting::string("wake_phrase", "Wake Phrase", "Phrase to activate assistant", &self.ai.wake_word),
            Setting::bool("voice_input", "Voice Input", "Enable voice commands", self.ai.voice_input),
            Setting::range("max_tokens", "Max Tokens", "Maximum response length", self.ai.max_tokens as i64, 32, 2048),
            Setting::range("cache_size", "Model Cache", "Maximum cache size (MB)", self.ai.cache_size_mb as i64, 256, 8192),
        ]
    }

    fn security_settings(&self) -> Vec<Setting> {
        alloc::vec![
            Setting::bool("screen_lock", "Screen Lock", "Lock screen after timeout", self.security.screen_lock),
            Setting::range("lock_timeout", "Lock Timeout", "Seconds before lock", self.security.lock_timeout as i64, 30, 3600),
            Setting::bool("pin_enabled", "PIN", "Require PIN to unlock", self.security.pin_enabled),
            Setting::bool("firewall", "Firewall", "Enable network firewall", self.security.firewall_enabled),
            Setting::bool("ssh", "SSH", "Enable SSH server", self.security.ssh_enabled),
            Setting::bool("auto_updates", "Auto Updates", "Automatically install security updates", self.security.auto_security_updates),
        ]
    }

    fn about_settings(&self) -> Vec<Setting> {
        alloc::vec![
            Setting::string("os_name", "OS Name", "", &self.about.os_name).readonly(),
            Setting::string("os_version", "Version", "", &self.about.os_version).readonly(),
            Setting::string("build", "Build", "", &self.about.build_number).readonly(),
            Setting::string("kernel", "Kernel", "", &self.about.kernel_version).readonly(),
            Setting::string("device", "Device", "", &self.about.device_model).readonly(),
            Setting::string("cpu", "CPU", "", &self.about.cpu).readonly(),
            Setting::string("ram", "RAM", "", &self.about.ram).readonly(),
            Setting::string("storage", "Storage", "", &self.about.storage).readonly(),
        ]
    }

    /// Get WiFi networks
    pub fn wifi_networks(&self) -> &[WifiNetwork] {
        &self.wifi_networks
    }

    /// Get AI models
    pub fn ai_models(&self) -> &[AiModel] {
        &self.ai_models
    }

    /// Scan for WiFi networks
    pub fn scan_wifi(&mut self) {
        self.wifi_scanning = true;
        // Mock networks
        self.wifi_networks = alloc::vec![
            WifiNetwork { ssid: String::from("HomeNetwork"), signal_strength: -45, secured: true, connected: true, saved: true },
            WifiNetwork { ssid: String::from("Neighbor_5G"), signal_strength: -65, secured: true, connected: false, saved: false },
            WifiNetwork { ssid: String::from("Guest"), signal_strength: -70, secured: false, connected: false, saved: false },
        ];
        self.wifi_scanning = false;
    }

    /// Connect to WiFi
    pub fn connect_wifi(&mut self, ssid: &str, password: Option<&str>) -> Result<(), SettingsError> {
        log::info!("Connecting to WiFi: {}", ssid);
        self.network.wifi_ssid = Some(String::from(ssid));
        Ok(())
    }

    /// Save settings
    pub fn save(&mut self) -> Result<(), SettingsError> {
        // Would persist to config file
        log::info!("Saving settings");
        self.has_changes = false;
        Ok(())
    }

    /// Reset to defaults
    pub fn reset_defaults(&mut self) {
        self.general = GeneralSettings::default();
        self.display = DisplaySettings::default();
        self.sound = SoundSettings::default();
        self.network = NetworkSettings::default();
        self.ai = AiSettings::default();
        self.security = SecuritySettings::default();
        self.has_changes = true;
    }

    /// Has unsaved changes
    pub fn has_changes(&self) -> bool {
        self.has_changes
    }

    /// Mark as changed
    pub fn mark_changed(&mut self) {
        self.has_changes = true;
    }
}

impl Default for SettingsApp {
    fn default() -> Self {
        Self::new()
    }
}

/// Settings errors
#[derive(Clone, Debug)]
pub enum SettingsError {
    InvalidValue,
    PermissionDenied,
    IoError(String),
    NetworkError(String),
}
