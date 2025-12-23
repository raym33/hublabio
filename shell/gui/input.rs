//! GUI Input Handling
//!
//! Input event processing for mouse, keyboard, and touch.

use super::Point;

/// Mouse button identifiers
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

/// Keyboard modifiers
#[derive(Clone, Copy, Debug, Default)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,  // Super/Windows/Command key
}

impl Modifiers {
    pub const fn new() -> Self {
        Self {
            shift: false,
            ctrl: false,
            alt: false,
            meta: false,
        }
    }

    pub fn any(&self) -> bool {
        self.shift || self.ctrl || self.alt || self.meta
    }
}

/// Touch event phase
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}

/// Single touch point
#[derive(Clone, Copy, Debug)]
pub struct TouchPoint {
    pub id: u32,
    pub x: i32,
    pub y: i32,
    pub phase: TouchPhase,
    pub pressure: f32,
}

/// Input event types
#[derive(Clone, Debug)]
pub enum InputEvent {
    /// Mouse moved
    MouseMove {
        x: i32,
        y: i32,
    },
    /// Mouse button state changed
    MouseButton {
        x: i32,
        y: i32,
        button: MouseButton,
        pressed: bool,
    },
    /// Mouse wheel scrolled
    Scroll {
        x: i32,
        y: i32,
        dx: i32,
        dy: i32,
    },
    /// Key state changed
    Key {
        scancode: u8,
        pressed: bool,
        modifiers: Modifiers,
    },
    /// Character input (after keyboard processing)
    Char {
        c: char,
    },
    /// Touch event
    Touch {
        points: [Option<TouchPoint>; 10],
        count: usize,
    },
    /// Single touch (simplified)
    SingleTouch {
        x: i32,
        y: i32,
        phase: TouchPhase,
    },
}

/// Input state tracker
pub struct InputState {
    /// Current mouse position
    pub mouse_x: i32,
    pub mouse_y: i32,
    /// Mouse button states
    pub left_button: bool,
    pub right_button: bool,
    pub middle_button: bool,
    /// Keyboard modifiers
    pub modifiers: Modifiers,
    /// Active touch points
    pub touches: [Option<TouchPoint>; 10],
    /// Key states (256 keys)
    pub keys: [bool; 256],
}

impl InputState {
    pub const fn new() -> Self {
        Self {
            mouse_x: 0,
            mouse_y: 0,
            left_button: false,
            right_button: false,
            middle_button: false,
            modifiers: Modifiers::new(),
            touches: [None; 10],
            keys: [false; 256],
        }
    }

    /// Update state from input event
    pub fn update(&mut self, event: &InputEvent) {
        match event {
            InputEvent::MouseMove { x, y } => {
                self.mouse_x = *x;
                self.mouse_y = *y;
            }
            InputEvent::MouseButton { x, y, button, pressed } => {
                self.mouse_x = *x;
                self.mouse_y = *y;
                match button {
                    MouseButton::Left => self.left_button = *pressed,
                    MouseButton::Right => self.right_button = *pressed,
                    MouseButton::Middle => self.middle_button = *pressed,
                    _ => {}
                }
            }
            InputEvent::Key { scancode, pressed, modifiers } => {
                if (*scancode as usize) < 256 {
                    self.keys[*scancode as usize] = *pressed;
                }
                self.modifiers = *modifiers;
            }
            InputEvent::Touch { points, count: _ } => {
                self.touches = *points;
            }
            InputEvent::SingleTouch { x, y, phase } => {
                match phase {
                    TouchPhase::Started => {
                        self.touches[0] = Some(TouchPoint {
                            id: 0,
                            x: *x,
                            y: *y,
                            phase: *phase,
                            pressure: 1.0,
                        });
                    }
                    TouchPhase::Moved => {
                        if let Some(ref mut touch) = self.touches[0] {
                            touch.x = *x;
                            touch.y = *y;
                            touch.phase = *phase;
                        }
                    }
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        self.touches[0] = None;
                    }
                }
            }
            _ => {}
        }
    }

    /// Get mouse position
    pub fn mouse_pos(&self) -> Point {
        Point::new(self.mouse_x, self.mouse_y)
    }

    /// Check if key is pressed
    pub fn is_key_pressed(&self, scancode: u8) -> bool {
        self.keys[scancode as usize]
    }

    /// Get active touch count
    pub fn touch_count(&self) -> usize {
        self.touches.iter().filter(|t| t.is_some()).count()
    }

    /// Get first touch position
    pub fn first_touch(&self) -> Option<Point> {
        self.touches.iter()
            .find_map(|t| t.as_ref())
            .map(|t| Point::new(t.x, t.y))
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

/// Gesture recognizer
pub struct GestureRecognizer {
    /// Tap detection
    tap_start: Option<(i32, i32, u64)>,
    tap_threshold: i32,
    tap_timeout_ms: u64,

    /// Double tap detection
    last_tap: Option<(i32, i32, u64)>,
    double_tap_threshold: i32,
    double_tap_timeout_ms: u64,

    /// Swipe detection
    swipe_start: Option<(i32, i32)>,
    swipe_threshold: i32,

    /// Pinch detection
    pinch_start_distance: Option<f32>,

    /// Long press detection
    long_press_start: Option<(i32, i32, u64)>,
    long_press_threshold_ms: u64,
}

/// Detected gesture
#[derive(Clone, Debug)]
pub enum Gesture {
    Tap { x: i32, y: i32 },
    DoubleTap { x: i32, y: i32 },
    LongPress { x: i32, y: i32 },
    SwipeUp { start_x: i32, start_y: i32, end_x: i32, end_y: i32 },
    SwipeDown { start_x: i32, start_y: i32, end_x: i32, end_y: i32 },
    SwipeLeft { start_x: i32, start_y: i32, end_x: i32, end_y: i32 },
    SwipeRight { start_x: i32, start_y: i32, end_x: i32, end_y: i32 },
    PinchIn { center_x: i32, center_y: i32, scale: f32 },
    PinchOut { center_x: i32, center_y: i32, scale: f32 },
    Pan { dx: i32, dy: i32 },
}

impl GestureRecognizer {
    pub const fn new() -> Self {
        Self {
            tap_start: None,
            tap_threshold: 20,
            tap_timeout_ms: 300,
            last_tap: None,
            double_tap_threshold: 40,
            double_tap_timeout_ms: 400,
            swipe_start: None,
            swipe_threshold: 50,
            pinch_start_distance: None,
            long_press_start: None,
            long_press_threshold_ms: 500,
        }
    }

    /// Process touch event and return detected gesture
    pub fn process(&mut self, event: &InputEvent, current_time_ms: u64) -> Option<Gesture> {
        match event {
            InputEvent::SingleTouch { x, y, phase } => {
                match phase {
                    TouchPhase::Started => {
                        self.tap_start = Some((*x, *y, current_time_ms));
                        self.swipe_start = Some((*x, *y));
                        self.long_press_start = Some((*x, *y, current_time_ms));
                        None
                    }
                    TouchPhase::Moved => {
                        // Cancel tap/long press if moved too far
                        if let Some((sx, sy, _)) = self.tap_start {
                            let dx = (*x - sx).abs();
                            let dy = (*y - sy).abs();
                            if dx > self.tap_threshold || dy > self.tap_threshold {
                                self.tap_start = None;
                                self.long_press_start = None;
                            }
                        }
                        None
                    }
                    TouchPhase::Ended => {
                        let mut gesture = None;

                        // Check for tap
                        if let Some((sx, sy, start_time)) = self.tap_start.take() {
                            let dx = (*x - sx).abs();
                            let dy = (*y - sy).abs();
                            let duration = current_time_ms - start_time;

                            if dx <= self.tap_threshold && dy <= self.tap_threshold {
                                if duration < self.tap_timeout_ms {
                                    // Check for double tap
                                    if let Some((lx, ly, last_time)) = self.last_tap {
                                        let ldx = (*x - lx).abs();
                                        let ldy = (*y - ly).abs();
                                        if ldx <= self.double_tap_threshold &&
                                           ldy <= self.double_tap_threshold &&
                                           current_time_ms - last_time < self.double_tap_timeout_ms {
                                            gesture = Some(Gesture::DoubleTap { x: *x, y: *y });
                                            self.last_tap = None;
                                        } else {
                                            gesture = Some(Gesture::Tap { x: *x, y: *y });
                                            self.last_tap = Some((*x, *y, current_time_ms));
                                        }
                                    } else {
                                        gesture = Some(Gesture::Tap { x: *x, y: *y });
                                        self.last_tap = Some((*x, *y, current_time_ms));
                                    }
                                }
                            }
                        }

                        // Check for swipe
                        if gesture.is_none() {
                            if let Some((sx, sy)) = self.swipe_start.take() {
                                let dx = *x - sx;
                                let dy = *y - sy;
                                let adx = dx.abs();
                                let ady = dy.abs();

                                if adx > self.swipe_threshold || ady > self.swipe_threshold {
                                    if adx > ady {
                                        if dx > 0 {
                                            gesture = Some(Gesture::SwipeRight {
                                                start_x: sx, start_y: sy,
                                                end_x: *x, end_y: *y
                                            });
                                        } else {
                                            gesture = Some(Gesture::SwipeLeft {
                                                start_x: sx, start_y: sy,
                                                end_x: *x, end_y: *y
                                            });
                                        }
                                    } else {
                                        if dy > 0 {
                                            gesture = Some(Gesture::SwipeDown {
                                                start_x: sx, start_y: sy,
                                                end_x: *x, end_y: *y
                                            });
                                        } else {
                                            gesture = Some(Gesture::SwipeUp {
                                                start_x: sx, start_y: sy,
                                                end_x: *x, end_y: *y
                                            });
                                        }
                                    }
                                }
                            }
                        }

                        self.long_press_start = None;
                        gesture
                    }
                    TouchPhase::Cancelled => {
                        self.tap_start = None;
                        self.swipe_start = None;
                        self.long_press_start = None;
                        None
                    }
                }
            }
            _ => None
        }
    }

    /// Check for long press (call periodically)
    pub fn check_long_press(&mut self, current_time_ms: u64) -> Option<Gesture> {
        if let Some((x, y, start_time)) = self.long_press_start {
            if current_time_ms - start_time >= self.long_press_threshold_ms {
                self.long_press_start = None;
                self.tap_start = None;
                return Some(Gesture::LongPress { x, y });
            }
        }
        None
    }

    /// Reset recognizer state
    pub fn reset(&mut self) {
        self.tap_start = None;
        self.last_tap = None;
        self.swipe_start = None;
        self.pinch_start_distance = None;
        self.long_press_start = None;
    }
}

impl Default for GestureRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Common scancodes
pub mod scancodes {
    pub const ESCAPE: u8 = 1;
    pub const KEY_1: u8 = 2;
    pub const KEY_2: u8 = 3;
    pub const KEY_3: u8 = 4;
    pub const KEY_4: u8 = 5;
    pub const KEY_5: u8 = 6;
    pub const KEY_6: u8 = 7;
    pub const KEY_7: u8 = 8;
    pub const KEY_8: u8 = 9;
    pub const KEY_9: u8 = 10;
    pub const KEY_0: u8 = 11;
    pub const MINUS: u8 = 12;
    pub const EQUALS: u8 = 13;
    pub const BACKSPACE: u8 = 14;
    pub const TAB: u8 = 15;
    pub const KEY_Q: u8 = 16;
    pub const KEY_W: u8 = 17;
    pub const KEY_E: u8 = 18;
    pub const KEY_R: u8 = 19;
    pub const KEY_T: u8 = 20;
    pub const KEY_Y: u8 = 21;
    pub const KEY_U: u8 = 22;
    pub const KEY_I: u8 = 23;
    pub const KEY_O: u8 = 24;
    pub const KEY_P: u8 = 25;
    pub const LEFT_BRACKET: u8 = 26;
    pub const RIGHT_BRACKET: u8 = 27;
    pub const ENTER: u8 = 28;
    pub const LEFT_CTRL: u8 = 29;
    pub const KEY_A: u8 = 30;
    pub const KEY_S: u8 = 31;
    pub const KEY_D: u8 = 32;
    pub const KEY_F: u8 = 33;
    pub const KEY_G: u8 = 34;
    pub const KEY_H: u8 = 35;
    pub const KEY_J: u8 = 36;
    pub const KEY_K: u8 = 37;
    pub const KEY_L: u8 = 38;
    pub const SEMICOLON: u8 = 39;
    pub const APOSTROPHE: u8 = 40;
    pub const GRAVE: u8 = 41;
    pub const LEFT_SHIFT: u8 = 42;
    pub const BACKSLASH: u8 = 43;
    pub const KEY_Z: u8 = 44;
    pub const KEY_X: u8 = 45;
    pub const KEY_C: u8 = 46;
    pub const KEY_V: u8 = 47;
    pub const KEY_B: u8 = 48;
    pub const KEY_N: u8 = 49;
    pub const KEY_M: u8 = 50;
    pub const COMMA: u8 = 51;
    pub const PERIOD: u8 = 52;
    pub const SLASH: u8 = 53;
    pub const RIGHT_SHIFT: u8 = 54;
    pub const KEYPAD_MULTIPLY: u8 = 55;
    pub const LEFT_ALT: u8 = 56;
    pub const SPACE: u8 = 57;
    pub const CAPS_LOCK: u8 = 58;
    pub const F1: u8 = 59;
    pub const F2: u8 = 60;
    pub const F3: u8 = 61;
    pub const F4: u8 = 62;
    pub const F5: u8 = 63;
    pub const F6: u8 = 64;
    pub const F7: u8 = 65;
    pub const F8: u8 = 66;
    pub const F9: u8 = 67;
    pub const F10: u8 = 68;
    pub const NUM_LOCK: u8 = 69;
    pub const SCROLL_LOCK: u8 = 70;
    pub const HOME: u8 = 71;
    pub const UP: u8 = 72;
    pub const PAGE_UP: u8 = 73;
    pub const KEYPAD_MINUS: u8 = 74;
    pub const LEFT: u8 = 75;
    pub const KEYPAD_5: u8 = 76;
    pub const RIGHT: u8 = 77;
    pub const KEYPAD_PLUS: u8 = 78;
    pub const END: u8 = 79;
    pub const DOWN: u8 = 80;
    pub const PAGE_DOWN: u8 = 81;
    pub const INSERT: u8 = 82;
    pub const DELETE: u8 = 83;
    pub const F11: u8 = 87;
    pub const F12: u8 = 88;
}
