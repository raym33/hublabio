//! Input Device Drivers
//!
//! Keyboard, mouse, and touchscreen input handling.
//! Supports USB HID devices and PS/2 legacy devices.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use spin::{Mutex, RwLock};

/// Input event types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum EventType {
    /// Synchronization event
    Syn = 0x00,
    /// Key press/release
    Key = 0x01,
    /// Relative movement (mouse)
    Rel = 0x02,
    /// Absolute position (touchscreen)
    Abs = 0x03,
    /// Miscellaneous
    Msc = 0x04,
    /// Switch event
    Sw = 0x05,
    /// LED control
    Led = 0x11,
    /// Sound effect
    Snd = 0x12,
    /// Repeat settings
    Rep = 0x14,
    /// Force feedback
    Ff = 0x15,
}

/// Key codes (subset of Linux input.h)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum KeyCode {
    Reserved = 0,
    Esc = 1,
    Key1 = 2,
    Key2 = 3,
    Key3 = 4,
    Key4 = 5,
    Key5 = 6,
    Key6 = 7,
    Key7 = 8,
    Key8 = 9,
    Key9 = 10,
    Key0 = 11,
    Minus = 12,
    Equal = 13,
    Backspace = 14,
    Tab = 15,
    Q = 16,
    W = 17,
    E = 18,
    R = 19,
    T = 20,
    Y = 21,
    U = 22,
    I = 23,
    O = 24,
    P = 25,
    LeftBrace = 26,
    RightBrace = 27,
    Enter = 28,
    LeftCtrl = 29,
    A = 30,
    S = 31,
    D = 32,
    F = 33,
    G = 34,
    H = 35,
    J = 36,
    K = 37,
    L = 38,
    Semicolon = 39,
    Apostrophe = 40,
    Grave = 41,
    LeftShift = 42,
    Backslash = 43,
    Z = 44,
    X = 45,
    C = 46,
    V = 47,
    B = 48,
    N = 49,
    M = 50,
    Comma = 51,
    Dot = 52,
    Slash = 53,
    RightShift = 54,
    KpAsterisk = 55,
    LeftAlt = 56,
    Space = 57,
    CapsLock = 58,
    F1 = 59,
    F2 = 60,
    F3 = 61,
    F4 = 62,
    F5 = 63,
    F6 = 64,
    F7 = 65,
    F8 = 66,
    F9 = 67,
    F10 = 68,
    NumLock = 69,
    ScrollLock = 70,
    Kp7 = 71,
    Kp8 = 72,
    Kp9 = 73,
    KpMinus = 74,
    Kp4 = 75,
    Kp5 = 76,
    Kp6 = 77,
    KpPlus = 78,
    Kp1 = 79,
    Kp2 = 80,
    Kp3 = 81,
    Kp0 = 82,
    KpDot = 83,
    F11 = 87,
    F12 = 88,
    KpEnter = 96,
    RightCtrl = 97,
    KpSlash = 98,
    SysRq = 99,
    RightAlt = 100,
    Home = 102,
    Up = 103,
    PageUp = 104,
    Left = 105,
    Right = 106,
    End = 107,
    Down = 108,
    PageDown = 109,
    Insert = 110,
    Delete = 111,
    Pause = 119,
    LeftMeta = 125,
    RightMeta = 126,
    Compose = 127,
    // Mouse buttons
    BtnLeft = 0x110,
    BtnRight = 0x111,
    BtnMiddle = 0x112,
    BtnSide = 0x113,
    BtnExtra = 0x114,
}

impl KeyCode {
    pub fn from_u16(code: u16) -> Option<Self> {
        // Safe transmute for valid codes
        if code <= 127 || (code >= 0x110 && code <= 0x114) {
            Some(unsafe { core::mem::transmute(code) })
        } else {
            None
        }
    }
}

/// Relative axis codes (mouse movement)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum RelCode {
    X = 0x00,
    Y = 0x01,
    Z = 0x02,
    Rx = 0x03,
    Ry = 0x04,
    Rz = 0x05,
    HWheel = 0x06,
    Dial = 0x07,
    Wheel = 0x08,
    Misc = 0x09,
}

/// Absolute axis codes (touchscreen)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum AbsCode {
    X = 0x00,
    Y = 0x01,
    Z = 0x02,
    Rx = 0x03,
    Ry = 0x04,
    Rz = 0x05,
    Throttle = 0x06,
    Rudder = 0x07,
    Wheel = 0x08,
    Gas = 0x09,
    Brake = 0x0a,
    Pressure = 0x18,
    MtSlot = 0x2f,
    MtTouchMajor = 0x30,
    MtTouchMinor = 0x31,
    MtPositionX = 0x35,
    MtPositionY = 0x36,
    MtTrackingId = 0x39,
}

/// Input event (matches Linux struct input_event)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct InputEvent {
    /// Timestamp seconds
    pub time_sec: u64,
    /// Timestamp microseconds
    pub time_usec: u64,
    /// Event type
    pub event_type: u16,
    /// Event code
    pub code: u16,
    /// Event value
    pub value: i32,
}

impl InputEvent {
    pub fn new(event_type: EventType, code: u16, value: i32) -> Self {
        let now = crate::time::monotonic_ns();
        Self {
            time_sec: now / 1_000_000_000,
            time_usec: (now % 1_000_000_000) / 1000,
            event_type: event_type as u16,
            code,
            value,
        }
    }

    pub fn key(code: KeyCode, pressed: bool) -> Self {
        Self::new(EventType::Key, code as u16, if pressed { 1 } else { 0 })
    }

    pub fn rel(code: RelCode, value: i32) -> Self {
        Self::new(EventType::Rel, code as u16, value)
    }

    pub fn abs(code: AbsCode, value: i32) -> Self {
        Self::new(EventType::Abs, code as u16, value)
    }

    pub fn syn() -> Self {
        Self::new(EventType::Syn, 0, 0)
    }
}

/// Keyboard state
pub struct KeyboardState {
    /// Currently pressed keys
    pressed: [u8; 32], // Bitmap for 256 keys
    /// Modifier state
    modifiers: Modifiers,
    /// LED state
    leds: LedState,
    /// Repeat settings
    repeat_delay: u32,  // ms
    repeat_rate: u32,   // chars/sec
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
    pub caps_lock: bool,
    pub num_lock: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LedState {
    pub caps_lock: bool,
    pub num_lock: bool,
    pub scroll_lock: bool,
}

impl KeyboardState {
    pub fn new() -> Self {
        Self {
            pressed: [0; 32],
            modifiers: Modifiers::default(),
            leds: LedState::default(),
            repeat_delay: 500,
            repeat_rate: 30,
        }
    }

    /// Check if key is pressed
    pub fn is_pressed(&self, code: u16) -> bool {
        if code >= 256 {
            return false;
        }
        let byte = (code / 8) as usize;
        let bit = code % 8;
        (self.pressed[byte] & (1 << bit)) != 0
    }

    /// Set key state
    pub fn set_pressed(&mut self, code: u16, pressed: bool) {
        if code >= 256 {
            return;
        }
        let byte = (code / 8) as usize;
        let bit = code % 8;
        if pressed {
            self.pressed[byte] |= 1 << bit;
        } else {
            self.pressed[byte] &= !(1 << bit);
        }

        // Update modifiers
        match code {
            29 | 97 => self.modifiers.ctrl = pressed || self.is_pressed(29) || self.is_pressed(97),
            42 | 54 => self.modifiers.shift = pressed || self.is_pressed(42) || self.is_pressed(54),
            56 | 100 => self.modifiers.alt = pressed || self.is_pressed(56) || self.is_pressed(100),
            125 | 126 => self.modifiers.meta = pressed || self.is_pressed(125) || self.is_pressed(126),
            58 => if pressed { self.modifiers.caps_lock = !self.modifiers.caps_lock; self.leds.caps_lock = self.modifiers.caps_lock; },
            69 => if pressed { self.modifiers.num_lock = !self.modifiers.num_lock; self.leds.num_lock = self.modifiers.num_lock; },
            70 => if pressed { self.leds.scroll_lock = !self.leds.scroll_lock; },
            _ => {}
        }
    }

    /// Convert keycode to ASCII character
    pub fn to_char(&self, code: u16) -> Option<char> {
        let shifted = self.modifiers.shift ^ self.modifiers.caps_lock;

        let ch = match code {
            // Numbers
            2..=11 => {
                let digit = if code == 11 { 0 } else { code - 1 };
                if self.modifiers.shift {
                    match digit {
                        1 => '!', 2 => '@', 3 => '#', 4 => '$', 5 => '%',
                        6 => '^', 7 => '&', 8 => '*', 9 => '(', 0 => ')',
                        _ => return None,
                    }
                } else {
                    char::from_digit(digit as u32, 10)?
                }
            }
            // Letters
            16 => if shifted { 'Q' } else { 'q' },
            17 => if shifted { 'W' } else { 'w' },
            18 => if shifted { 'E' } else { 'e' },
            19 => if shifted { 'R' } else { 'r' },
            20 => if shifted { 'T' } else { 't' },
            21 => if shifted { 'Y' } else { 'y' },
            22 => if shifted { 'U' } else { 'u' },
            23 => if shifted { 'I' } else { 'i' },
            24 => if shifted { 'O' } else { 'o' },
            25 => if shifted { 'P' } else { 'p' },
            30 => if shifted { 'A' } else { 'a' },
            31 => if shifted { 'S' } else { 's' },
            32 => if shifted { 'D' } else { 'd' },
            33 => if shifted { 'F' } else { 'f' },
            34 => if shifted { 'G' } else { 'g' },
            35 => if shifted { 'H' } else { 'h' },
            36 => if shifted { 'J' } else { 'j' },
            37 => if shifted { 'K' } else { 'k' },
            38 => if shifted { 'L' } else { 'l' },
            44 => if shifted { 'Z' } else { 'z' },
            45 => if shifted { 'X' } else { 'x' },
            46 => if shifted { 'C' } else { 'c' },
            47 => if shifted { 'V' } else { 'v' },
            48 => if shifted { 'B' } else { 'b' },
            49 => if shifted { 'N' } else { 'n' },
            50 => if shifted { 'M' } else { 'm' },
            // Special keys
            12 => if self.modifiers.shift { '_' } else { '-' },
            13 => if self.modifiers.shift { '+' } else { '=' },
            14 => '\x08', // Backspace
            15 => '\t',   // Tab
            26 => if self.modifiers.shift { '{' } else { '[' },
            27 => if self.modifiers.shift { '}' } else { ']' },
            28 => '\n',   // Enter
            39 => if self.modifiers.shift { ':' } else { ';' },
            40 => if self.modifiers.shift { '"' } else { '\'' },
            41 => if self.modifiers.shift { '~' } else { '`' },
            43 => if self.modifiers.shift { '|' } else { '\\' },
            51 => if self.modifiers.shift { '<' } else { ',' },
            52 => if self.modifiers.shift { '>' } else { '.' },
            53 => if self.modifiers.shift { '?' } else { '/' },
            57 => ' ',    // Space
            // Control characters
            _ if self.modifiers.ctrl => {
                match code {
                    46 => '\x03', // Ctrl+C
                    32 => '\x04', // Ctrl+D
                    38 => '\x0C', // Ctrl+L
                    44 => '\x1A', // Ctrl+Z
                    _ => return None,
                }
            }
            _ => return None,
        };

        Some(ch)
    }
}

/// Mouse state
pub struct MouseState {
    /// Button state bitmap
    buttons: u32,
    /// X position (absolute mode)
    x: i32,
    /// Y position (absolute mode)
    y: i32,
    /// Scroll wheel position
    wheel: i32,
}

impl MouseState {
    pub fn new() -> Self {
        Self {
            buttons: 0,
            x: 0,
            y: 0,
            wheel: 0,
        }
    }

    pub fn is_button_pressed(&self, button: u32) -> bool {
        (self.buttons & (1 << button)) != 0
    }

    pub fn set_button(&mut self, button: u32, pressed: bool) {
        if pressed {
            self.buttons |= 1 << button;
        } else {
            self.buttons &= !(1 << button);
        }
    }

    pub fn move_relative(&mut self, dx: i32, dy: i32) {
        self.x = self.x.saturating_add(dx);
        self.y = self.y.saturating_add(dy);
    }

    pub fn set_absolute(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }
}

/// Input device
pub struct InputDevice {
    /// Device ID
    pub id: u32,
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: InputDeviceType,
    /// Event queue
    events: Mutex<VecDeque<InputEvent>>,
    /// Wait queue for blocking reads
    waiters: crate::waitqueue::WaitQueue,
    /// Device-specific state
    state: Mutex<InputDeviceState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputDeviceType {
    Keyboard,
    Mouse,
    Touchscreen,
    Touchpad,
    Joystick,
    Tablet,
}

enum InputDeviceState {
    Keyboard(KeyboardState),
    Mouse(MouseState),
    Touchscreen { x: i32, y: i32, touching: bool },
}

impl InputDevice {
    pub fn new_keyboard(id: u32, name: &str) -> Self {
        Self {
            id,
            name: String::from(name),
            device_type: InputDeviceType::Keyboard,
            events: Mutex::new(VecDeque::with_capacity(64)),
            waiters: crate::waitqueue::WaitQueue::new(),
            state: Mutex::new(InputDeviceState::Keyboard(KeyboardState::new())),
        }
    }

    pub fn new_mouse(id: u32, name: &str) -> Self {
        Self {
            id,
            name: String::from(name),
            device_type: InputDeviceType::Mouse,
            events: Mutex::new(VecDeque::with_capacity(64)),
            waiters: crate::waitqueue::WaitQueue::new(),
            state: Mutex::new(InputDeviceState::Mouse(MouseState::new())),
        }
    }

    pub fn new_touchscreen(id: u32, name: &str) -> Self {
        Self {
            id,
            name: String::from(name),
            device_type: InputDeviceType::Touchscreen,
            events: Mutex::new(VecDeque::with_capacity(64)),
            waiters: crate::waitqueue::WaitQueue::new(),
            state: Mutex::new(InputDeviceState::Touchscreen { x: 0, y: 0, touching: false }),
        }
    }

    /// Push event to queue
    pub fn push_event(&self, event: InputEvent) {
        let mut events = self.events.lock();
        if events.len() < 256 {
            events.push_back(event);
            self.waiters.wake_one();
        }
    }

    /// Read events (non-blocking)
    pub fn read_events(&self, buf: &mut [InputEvent]) -> usize {
        let mut events = self.events.lock();
        let count = buf.len().min(events.len());
        for i in 0..count {
            buf[i] = events.pop_front().unwrap();
        }
        count
    }

    /// Read events (blocking)
    pub fn read_events_blocking(&self, buf: &mut [InputEvent]) -> usize {
        loop {
            let count = self.read_events(buf);
            if count > 0 {
                return count;
            }
            self.waiters.wait();
        }
    }

    /// Check if events are available
    pub fn has_events(&self) -> bool {
        !self.events.lock().is_empty()
    }

    /// Handle keyboard event
    pub fn handle_key(&self, code: u16, pressed: bool) {
        let mut state = self.state.lock();
        if let InputDeviceState::Keyboard(ref mut kbd) = *state {
            kbd.set_pressed(code, pressed);
        }
        drop(state);

        // Push key event
        let value = if pressed { 1 } else { 0 };
        self.push_event(InputEvent::new(EventType::Key, code, value));
        self.push_event(InputEvent::syn());
    }

    /// Handle mouse movement
    pub fn handle_mouse_rel(&self, dx: i32, dy: i32) {
        let mut state = self.state.lock();
        if let InputDeviceState::Mouse(ref mut mouse) = *state {
            mouse.move_relative(dx, dy);
        }
        drop(state);

        if dx != 0 {
            self.push_event(InputEvent::rel(RelCode::X, dx));
        }
        if dy != 0 {
            self.push_event(InputEvent::rel(RelCode::Y, dy));
        }
        self.push_event(InputEvent::syn());
    }

    /// Handle mouse button
    pub fn handle_mouse_button(&self, button: u32, pressed: bool) {
        let mut state = self.state.lock();
        if let InputDeviceState::Mouse(ref mut mouse) = *state {
            mouse.set_button(button, pressed);
        }
        drop(state);

        let code = match button {
            0 => KeyCode::BtnLeft as u16,
            1 => KeyCode::BtnRight as u16,
            2 => KeyCode::BtnMiddle as u16,
            _ => return,
        };

        self.push_event(InputEvent::new(EventType::Key, code, if pressed { 1 } else { 0 }));
        self.push_event(InputEvent::syn());
    }

    /// Handle mouse wheel
    pub fn handle_mouse_wheel(&self, delta: i32) {
        self.push_event(InputEvent::rel(RelCode::Wheel, delta));
        self.push_event(InputEvent::syn());
    }

    /// Handle touchscreen event
    pub fn handle_touch(&self, x: i32, y: i32, touching: bool) {
        let mut state = self.state.lock();
        if let InputDeviceState::Touchscreen { x: ref mut tx, y: ref mut ty, touching: ref mut t } = *state {
            *tx = x;
            *ty = y;
            *t = touching;
        }
        drop(state);

        self.push_event(InputEvent::abs(AbsCode::X, x));
        self.push_event(InputEvent::abs(AbsCode::Y, y));
        self.push_event(InputEvent::new(EventType::Key, KeyCode::BtnLeft as u16, if touching { 1 } else { 0 }));
        self.push_event(InputEvent::syn());
    }

    /// Get keyboard character
    pub fn get_char(&self) -> Option<char> {
        let state = self.state.lock();
        if let InputDeviceState::Keyboard(ref kbd) = *state {
            // Find pressed key and convert
            for code in 0..256u16 {
                if kbd.is_pressed(code) {
                    return kbd.to_char(code);
                }
            }
        }
        None
    }
}

// ============================================================================
// USB HID Driver
// ============================================================================

/// USB HID boot protocol keyboard report
#[repr(C, packed)]
pub struct HidKeyboardReport {
    pub modifiers: u8,
    pub reserved: u8,
    pub keys: [u8; 6],
}

/// USB HID boot protocol mouse report
#[repr(C, packed)]
pub struct HidMouseReport {
    pub buttons: u8,
    pub x: i8,
    pub y: i8,
    pub wheel: i8,
}

/// Parse USB HID keyboard report
pub fn parse_hid_keyboard(device: &InputDevice, report: &HidKeyboardReport, prev: &HidKeyboardReport) {
    // Handle modifiers
    let mod_diff = report.modifiers ^ prev.modifiers;
    if mod_diff & 0x01 != 0 { device.handle_key(KeyCode::LeftCtrl as u16, report.modifiers & 0x01 != 0); }
    if mod_diff & 0x02 != 0 { device.handle_key(KeyCode::LeftShift as u16, report.modifiers & 0x02 != 0); }
    if mod_diff & 0x04 != 0 { device.handle_key(KeyCode::LeftAlt as u16, report.modifiers & 0x04 != 0); }
    if mod_diff & 0x08 != 0 { device.handle_key(KeyCode::LeftMeta as u16, report.modifiers & 0x08 != 0); }
    if mod_diff & 0x10 != 0 { device.handle_key(KeyCode::RightCtrl as u16, report.modifiers & 0x10 != 0); }
    if mod_diff & 0x20 != 0 { device.handle_key(KeyCode::RightShift as u16, report.modifiers & 0x20 != 0); }
    if mod_diff & 0x40 != 0 { device.handle_key(KeyCode::RightAlt as u16, report.modifiers & 0x40 != 0); }
    if mod_diff & 0x80 != 0 { device.handle_key(KeyCode::RightMeta as u16, report.modifiers & 0x80 != 0); }

    // Handle key releases (keys in prev but not in report)
    for &key in &prev.keys {
        if key != 0 && !report.keys.contains(&key) {
            if let Some(code) = hid_to_keycode(key) {
                device.handle_key(code, false);
            }
        }
    }

    // Handle key presses (keys in report but not in prev)
    for &key in &report.keys {
        if key != 0 && !prev.keys.contains(&key) {
            if let Some(code) = hid_to_keycode(key) {
                device.handle_key(code, true);
            }
        }
    }
}

/// Parse USB HID mouse report
pub fn parse_hid_mouse(device: &InputDevice, report: &HidMouseReport, prev: &HidMouseReport) {
    // Handle buttons
    let btn_diff = report.buttons ^ prev.buttons;
    if btn_diff & 0x01 != 0 { device.handle_mouse_button(0, report.buttons & 0x01 != 0); }
    if btn_diff & 0x02 != 0 { device.handle_mouse_button(1, report.buttons & 0x02 != 0); }
    if btn_diff & 0x04 != 0 { device.handle_mouse_button(2, report.buttons & 0x04 != 0); }

    // Handle movement
    if report.x != 0 || report.y != 0 {
        device.handle_mouse_rel(report.x as i32, report.y as i32);
    }

    // Handle wheel
    if report.wheel != 0 {
        device.handle_mouse_wheel(report.wheel as i32);
    }
}

/// Convert HID usage to keycode
fn hid_to_keycode(hid: u8) -> Option<u16> {
    // HID usage page 0x07 (Keyboard/Keypad) to Linux keycode
    match hid {
        0x04..=0x1D => Some((hid - 0x04 + 30) as u16), // A-Z -> 30-55
        0x1E..=0x27 => Some((hid - 0x1E + 2) as u16),  // 1-0 -> 2-11
        0x28 => Some(28),  // Enter
        0x29 => Some(1),   // Escape
        0x2A => Some(14),  // Backspace
        0x2B => Some(15),  // Tab
        0x2C => Some(57),  // Space
        0x2D => Some(12),  // -
        0x2E => Some(13),  // =
        0x2F => Some(26),  // [
        0x30 => Some(27),  // ]
        0x31 => Some(43),  // backslash
        0x33 => Some(39),  // ;
        0x34 => Some(40),  // '
        0x35 => Some(41),  // `
        0x36 => Some(51),  // ,
        0x37 => Some(52),  // .
        0x38 => Some(53),  // /
        0x39 => Some(58),  // Caps Lock
        0x3A..=0x45 => Some((hid - 0x3A + 59) as u16), // F1-F12
        0x49 => Some(110), // Insert
        0x4A => Some(102), // Home
        0x4B => Some(104), // Page Up
        0x4C => Some(111), // Delete
        0x4D => Some(107), // End
        0x4E => Some(109), // Page Down
        0x4F => Some(106), // Right
        0x50 => Some(105), // Left
        0x51 => Some(108), // Down
        0x52 => Some(103), // Up
        _ => None,
    }
}

// ============================================================================
// Global State
// ============================================================================

/// Registered input devices
static INPUT_DEVICES: RwLock<Vec<InputDevice>> = RwLock::new(Vec::new());

/// Next device ID
static NEXT_DEVICE_ID: AtomicU32 = AtomicU32::new(0);

/// Primary keyboard device
static PRIMARY_KEYBOARD: AtomicU32 = AtomicU32::new(u32::MAX);

/// Primary mouse device
static PRIMARY_MOUSE: AtomicU32 = AtomicU32::new(u32::MAX);

/// Register a new input device
pub fn register_device(device: InputDevice) -> u32 {
    let id = device.id;
    let device_type = device.device_type;

    INPUT_DEVICES.write().push(device);

    // Set as primary if first of its type
    match device_type {
        InputDeviceType::Keyboard => {
            PRIMARY_KEYBOARD.compare_exchange(u32::MAX, id, Ordering::SeqCst, Ordering::SeqCst).ok();
        }
        InputDeviceType::Mouse | InputDeviceType::Touchpad => {
            PRIMARY_MOUSE.compare_exchange(u32::MAX, id, Ordering::SeqCst, Ordering::SeqCst).ok();
        }
        _ => {}
    }

    crate::kinfo!("Registered input device {} (id={})", "", id);
    id
}

/// Create and register keyboard
pub fn create_keyboard(name: &str) -> u32 {
    let id = NEXT_DEVICE_ID.fetch_add(1, Ordering::SeqCst);
    let device = InputDevice::new_keyboard(id, name);
    register_device(device)
}

/// Create and register mouse
pub fn create_mouse(name: &str) -> u32 {
    let id = NEXT_DEVICE_ID.fetch_add(1, Ordering::SeqCst);
    let device = InputDevice::new_mouse(id, name);
    register_device(device)
}

/// Create and register touchscreen
pub fn create_touchscreen(name: &str) -> u32 {
    let id = NEXT_DEVICE_ID.fetch_add(1, Ordering::SeqCst);
    let device = InputDevice::new_touchscreen(id, name);
    register_device(device)
}

/// Get device by ID
pub fn get_device(id: u32) -> Option<&'static InputDevice> {
    let devices = INPUT_DEVICES.read();
    // This is safe because devices are never removed
    devices.iter().find(|d| d.id == id).map(|d| unsafe {
        &*(d as *const InputDevice)
    })
}

/// Get primary keyboard
pub fn keyboard() -> Option<&'static InputDevice> {
    let id = PRIMARY_KEYBOARD.load(Ordering::SeqCst);
    if id == u32::MAX { None } else { get_device(id) }
}

/// Get primary mouse
pub fn mouse() -> Option<&'static InputDevice> {
    let id = PRIMARY_MOUSE.load(Ordering::SeqCst);
    if id == u32::MAX { None } else { get_device(id) }
}

/// Read character from keyboard (blocking)
pub fn read_char() -> char {
    loop {
        if let Some(kbd) = keyboard() {
            let mut events = [InputEvent::syn(); 8];
            let count = kbd.read_events_blocking(&mut events);

            for i in 0..count {
                let event = &events[i];
                if event.event_type == EventType::Key as u16 && event.value == 1 {
                    let state = kbd.state.lock();
                    if let InputDeviceState::Keyboard(ref ks) = *state {
                        if let Some(ch) = ks.to_char(event.code) {
                            return ch;
                        }
                    }
                }
            }
        } else {
            // No keyboard - wait
            crate::scheduler::schedule();
        }
    }
}

/// Check if keyboard input available
pub fn keyboard_available() -> bool {
    keyboard().map(|k| k.has_events()).unwrap_or(false)
}

/// Initialize input subsystem
pub fn init() {
    // Create virtual keyboard and mouse for QEMU/testing
    create_keyboard("Virtual Keyboard");
    create_mouse("Virtual Mouse");

    crate::kprintln!("  Input subsystem initialized");
}
