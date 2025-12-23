//! Line Editor Module
//!
//! Advanced text input with cursor movement, history, and editing.

use alloc::string::String;
use alloc::vec::Vec;

/// Line editor with cursor support
pub struct LineEditor {
    /// Current line content
    buffer: Vec<char>,
    /// Cursor position (character index)
    cursor: usize,
    /// History of previous inputs
    history: Vec<String>,
    /// Current history position
    history_index: usize,
    /// Maximum history size
    max_history: usize,
    /// Selection start (if any)
    selection_start: Option<usize>,
    /// Clipboard content
    clipboard: Option<String>,
    /// Undo stack
    undo_stack: Vec<(Vec<char>, usize)>,
    /// Prompt string
    prompt: String,
    /// Completion candidates
    completions: Vec<String>,
    /// Current completion index
    completion_index: usize,
    /// Is in completion mode
    completing: bool,
    /// Temporary buffer for history navigation
    temp_buffer: Option<String>,
}

impl LineEditor {
    /// Create new line editor
    pub fn new(max_history: usize) -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: 0,
            max_history,
            selection_start: None,
            clipboard: None,
            undo_stack: Vec::new(),
            prompt: String::from("> "),
            completions: Vec::new(),
            completion_index: 0,
            completing: false,
            temp_buffer: None,
        }
    }

    /// Create with default history size
    pub fn with_defaults() -> Self {
        Self::new(1000)
    }

    /// Set prompt
    pub fn set_prompt(&mut self, prompt: &str) {
        self.prompt = String::from(prompt);
    }

    /// Get prompt
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Get current buffer content
    pub fn content(&self) -> String {
        self.buffer.iter().collect()
    }

    /// Get cursor position
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Is buffer empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.save_undo();
        self.buffer.clear();
        self.cursor = 0;
        self.selection_start = None;
        self.completing = false;
    }

    /// Set content
    pub fn set_content(&mut self, content: &str) {
        self.save_undo();
        self.buffer = content.chars().collect();
        self.cursor = self.buffer.len();
        self.selection_start = None;
    }

    /// Insert character at cursor
    pub fn insert(&mut self, c: char) {
        self.save_undo();
        self.delete_selection();
        self.buffer.insert(self.cursor, c);
        self.cursor += 1;
        self.completing = false;
    }

    /// Insert string at cursor
    pub fn insert_str(&mut self, s: &str) {
        self.save_undo();
        self.delete_selection();
        for c in s.chars() {
            self.buffer.insert(self.cursor, c);
            self.cursor += 1;
        }
        self.completing = false;
    }

    /// Delete character before cursor (backspace)
    pub fn backspace(&mut self) {
        if self.selection_start.is_some() {
            self.delete_selection();
            return;
        }

        if self.cursor > 0 {
            self.save_undo();
            self.cursor -= 1;
            self.buffer.remove(self.cursor);
        }
        self.completing = false;
    }

    /// Delete character at cursor (delete)
    pub fn delete(&mut self) {
        if self.selection_start.is_some() {
            self.delete_selection();
            return;
        }

        if self.cursor < self.buffer.len() {
            self.save_undo();
            self.buffer.remove(self.cursor);
        }
        self.completing = false;
    }

    /// Delete word before cursor
    pub fn delete_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }

        self.save_undo();
        let start = self.find_word_start();
        let removed: String = self.buffer.drain(start..self.cursor).collect();
        self.cursor = start;
        self.clipboard = Some(removed);
        self.completing = false;
    }

    /// Delete word after cursor
    pub fn delete_word_forward(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }

        self.save_undo();
        let end = self.find_word_end();
        let removed: String = self.buffer.drain(self.cursor..end).collect();
        self.clipboard = Some(removed);
        self.completing = false;
    }

    /// Delete from cursor to end of line
    pub fn delete_to_end(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }

        self.save_undo();
        let removed: String = self.buffer.drain(self.cursor..).collect();
        self.clipboard = Some(removed);
        self.completing = false;
    }

    /// Delete from cursor to start of line
    pub fn delete_to_start(&mut self) {
        if self.cursor == 0 {
            return;
        }

        self.save_undo();
        let removed: String = self.buffer.drain(..self.cursor).collect();
        self.cursor = 0;
        self.clipboard = Some(removed);
        self.completing = false;
    }

    /// Move cursor left
    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
        self.selection_start = None;
        self.completing = false;
    }

    /// Move cursor right
    pub fn move_right(&mut self) {
        if self.cursor < self.buffer.len() {
            self.cursor += 1;
        }
        self.selection_start = None;
        self.completing = false;
    }

    /// Move cursor to start
    pub fn move_to_start(&mut self) {
        self.cursor = 0;
        self.selection_start = None;
        self.completing = false;
    }

    /// Move cursor to end
    pub fn move_to_end(&mut self) {
        self.cursor = self.buffer.len();
        self.selection_start = None;
        self.completing = false;
    }

    /// Move cursor left by word
    pub fn move_word_left(&mut self) {
        self.cursor = self.find_word_start();
        self.selection_start = None;
        self.completing = false;
    }

    /// Move cursor right by word
    pub fn move_word_right(&mut self) {
        self.cursor = self.find_word_end();
        self.selection_start = None;
        self.completing = false;
    }

    /// Find start of current word
    fn find_word_start(&self) -> usize {
        if self.cursor == 0 {
            return 0;
        }

        let mut pos = self.cursor - 1;

        // Skip whitespace
        while pos > 0 && self.buffer[pos].is_whitespace() {
            pos -= 1;
        }

        // Find word boundary
        while pos > 0 && !self.buffer[pos - 1].is_whitespace() {
            pos -= 1;
        }

        pos
    }

    /// Find end of current word
    fn find_word_end(&self) -> usize {
        if self.cursor >= self.buffer.len() {
            return self.buffer.len();
        }

        let mut pos = self.cursor;

        // Skip current word
        while pos < self.buffer.len() && !self.buffer[pos].is_whitespace() {
            pos += 1;
        }

        // Skip whitespace
        while pos < self.buffer.len() && self.buffer[pos].is_whitespace() {
            pos += 1;
        }

        pos
    }

    /// Start selection
    pub fn start_selection(&mut self) {
        self.selection_start = Some(self.cursor);
    }

    /// Get selection range
    pub fn selection(&self) -> Option<(usize, usize)> {
        self.selection_start.map(|start| {
            if start < self.cursor {
                (start, self.cursor)
            } else {
                (self.cursor, start)
            }
        })
    }

    /// Get selected text
    pub fn selected_text(&self) -> Option<String> {
        self.selection()
            .map(|(start, end)| self.buffer[start..end].iter().collect())
    }

    /// Delete selection
    fn delete_selection(&mut self) {
        if let Some((start, end)) = self.selection() {
            self.save_undo();
            self.buffer.drain(start..end);
            self.cursor = start;
            self.selection_start = None;
        }
    }

    /// Copy selection to clipboard
    pub fn copy(&mut self) {
        if let Some(text) = self.selected_text() {
            self.clipboard = Some(text);
        }
    }

    /// Cut selection to clipboard
    pub fn cut(&mut self) {
        self.copy();
        self.delete_selection();
    }

    /// Paste from clipboard
    pub fn paste(&mut self) {
        if let Some(ref text) = self.clipboard.clone() {
            self.insert_str(text);
        }
    }

    /// Save state for undo
    fn save_undo(&mut self) {
        self.undo_stack.push((self.buffer.clone(), self.cursor));

        // Limit undo stack size
        if self.undo_stack.len() > 100 {
            self.undo_stack.remove(0);
        }
    }

    /// Undo last operation
    pub fn undo(&mut self) {
        if let Some((buffer, cursor)) = self.undo_stack.pop() {
            self.buffer = buffer;
            self.cursor = cursor;
            self.selection_start = None;
        }
    }

    /// Swap characters around cursor (transpose)
    pub fn transpose(&mut self) {
        if self.cursor >= 2 {
            self.save_undo();
            self.buffer.swap(self.cursor - 1, self.cursor - 2);
        } else if self.cursor == 1 && self.buffer.len() >= 2 {
            self.save_undo();
            self.buffer.swap(0, 1);
            self.cursor = 2.min(self.buffer.len());
        }
    }

    /// Submit current line and add to history
    pub fn submit(&mut self) -> String {
        let content = self.content();

        // Add to history if not empty and not duplicate
        if !content.is_empty() {
            if self.history.last().map(|s| s != &content).unwrap_or(true) {
                self.history.push(content.clone());

                // Limit history size
                if self.history.len() > self.max_history {
                    self.history.remove(0);
                }
            }
        }

        // Reset state
        self.buffer.clear();
        self.cursor = 0;
        self.history_index = self.history.len();
        self.selection_start = None;
        self.temp_buffer = None;
        self.completing = false;

        content
    }

    /// Navigate to previous history entry
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }

        // Save current buffer if at end
        if self.history_index == self.history.len() {
            self.temp_buffer = Some(self.content());
        }

        if self.history_index > 0 {
            self.history_index -= 1;
            let content = self.history[self.history_index].clone();
            self.buffer = content.chars().collect();
            self.cursor = self.buffer.len();
        }
    }

    /// Navigate to next history entry
    pub fn history_next(&mut self) {
        if self.history_index < self.history.len() {
            self.history_index += 1;

            if self.history_index == self.history.len() {
                // Restore temp buffer
                if let Some(ref temp) = self.temp_buffer {
                    self.buffer = temp.chars().collect();
                } else {
                    self.buffer.clear();
                }
            } else {
                let content = self.history[self.history_index].clone();
                self.buffer = content.chars().collect();
            }

            self.cursor = self.buffer.len();
        }
    }

    /// Search history for pattern
    pub fn history_search(&self, pattern: &str) -> Vec<&String> {
        self.history
            .iter()
            .filter(|s| s.contains(pattern))
            .collect()
    }

    /// Set completion candidates
    pub fn set_completions(&mut self, candidates: Vec<String>) {
        self.completions = candidates;
        self.completion_index = 0;
        self.completing = !self.completions.is_empty();
    }

    /// Get current completion
    pub fn current_completion(&self) -> Option<&String> {
        if self.completing && !self.completions.is_empty() {
            Some(&self.completions[self.completion_index])
        } else {
            None
        }
    }

    /// Cycle to next completion
    pub fn next_completion(&mut self) {
        if !self.completions.is_empty() {
            self.completion_index = (self.completion_index + 1) % self.completions.len();
        }
    }

    /// Cycle to previous completion
    pub fn prev_completion(&mut self) {
        if !self.completions.is_empty() {
            self.completion_index = if self.completion_index == 0 {
                self.completions.len() - 1
            } else {
                self.completion_index - 1
            };
        }
    }

    /// Apply current completion
    pub fn apply_completion(&mut self) {
        if let Some(completion) = self.current_completion().cloned() {
            // Find word start to replace
            let word_start = self.find_word_start();
            self.save_undo();

            // Remove current partial word
            self.buffer.drain(word_start..self.cursor);
            self.cursor = word_start;

            // Insert completion
            for c in completion.chars() {
                self.buffer.insert(self.cursor, c);
                self.cursor += 1;
            }

            self.completing = false;
        }
    }

    /// Cancel completion
    pub fn cancel_completion(&mut self) {
        self.completing = false;
        self.completions.clear();
    }

    /// Get word at cursor (for completion)
    pub fn word_at_cursor(&self) -> String {
        let start = self.find_word_start();
        self.buffer[start..self.cursor].iter().collect()
    }

    /// Get history
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history_index = 0;
    }

    /// Alias for move_to_start
    pub fn move_home(&mut self) {
        self.move_to_start();
    }

    /// Alias for move_to_end
    pub fn move_end(&mut self) {
        self.move_to_end();
    }

    /// Tab completion (alias for apply_completion or cycle)
    pub fn complete(&mut self) {
        if self.completing {
            self.next_completion();
            self.apply_completion();
        } else if !self.completions.is_empty() {
            self.completing = true;
            self.apply_completion();
        }
    }

    /// Render content with visible cursor character
    pub fn render_with_cursor(&self, cursor_char: char) -> String {
        let mut result = String::new();

        for (i, c) in self.buffer.iter().enumerate() {
            if i == self.cursor {
                result.push(cursor_char);
            }
            result.push(*c);
        }

        // Cursor at end
        if self.cursor >= self.buffer.len() {
            result.push(cursor_char);
        }

        result
    }

    /// Render with ANSI cursor positioning
    pub fn render_ansi(&self, start_col: u16) -> String {
        let content = self.content();
        let cursor_col = start_col + self.cursor as u16;
        alloc::format!("{}\x1b[{}G", content, cursor_col)
    }
}

impl Default for LineEditor {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Input actions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorAction {
    // Character input
    Insert(char),

    // Deletion
    Backspace,
    Delete,
    DeleteWord,
    DeleteWordForward,
    DeleteToEnd,
    DeleteToStart,

    // Movement
    Left,
    Right,
    WordLeft,
    WordRight,
    Home,
    End,

    // History
    HistoryPrev,
    HistoryNext,

    // Editing
    Undo,
    Transpose,
    Clear,

    // Clipboard
    Copy,
    Cut,
    Paste,

    // Completion
    Complete,
    CompleteNext,
    CompletePrev,
    CancelComplete,

    // Submit
    Submit,
    Cancel,
}

/// Parse key sequence into action
pub fn parse_key(key: u8, ctrl: bool, alt: bool) -> Option<EditorAction> {
    match (key, ctrl, alt) {
        // Control keys
        (b'a', true, false) => Some(EditorAction::Home),
        (b'e', true, false) => Some(EditorAction::End),
        (b'b', true, false) => Some(EditorAction::Left),
        (b'f', true, false) => Some(EditorAction::Right),
        (b'w', true, false) => Some(EditorAction::DeleteWord),
        (b'd', true, false) => Some(EditorAction::Delete),
        (b'k', true, false) => Some(EditorAction::DeleteToEnd),
        (b'u', true, false) => Some(EditorAction::DeleteToStart),
        (b't', true, false) => Some(EditorAction::Transpose),
        (b'c', true, false) => Some(EditorAction::Cancel),
        (b'z', true, false) => Some(EditorAction::Undo),
        (b'y', true, false) => Some(EditorAction::Paste),
        (b'l', true, false) => Some(EditorAction::Clear),

        // Alt keys
        (b'b', false, true) => Some(EditorAction::WordLeft),
        (b'f', false, true) => Some(EditorAction::WordRight),
        (b'd', false, true) => Some(EditorAction::DeleteWordForward),

        // Special keys
        (0x7F, false, false) => Some(EditorAction::Backspace), // DEL
        (0x08, false, false) => Some(EditorAction::Backspace), // BS
        (b'\r', false, false) | (b'\n', false, false) => Some(EditorAction::Submit),
        (b'\t', false, false) => Some(EditorAction::Complete),
        (0x1B, false, false) => Some(EditorAction::Cancel), // ESC

        // Printable characters
        (c, false, false) if c >= 0x20 && c < 0x7F => Some(EditorAction::Insert(c as char)),

        _ => None,
    }
}
