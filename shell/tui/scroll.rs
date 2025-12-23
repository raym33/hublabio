//! Scrollable Views Module
//!
//! Scrollable text views and lists for the TUI.

use alloc::string::String;
use alloc::vec::Vec;

/// Scrollable text buffer
pub struct ScrollBuffer {
    /// Lines of text
    lines: Vec<String>,
    /// Maximum lines to keep
    max_lines: usize,
    /// Scroll offset (line index of top visible line)
    scroll_offset: usize,
    /// Viewport height
    viewport_height: usize,
    /// Auto-scroll to bottom
    auto_scroll: bool,
}

impl ScrollBuffer {
    /// Create new scroll buffer
    pub fn new(max_lines: usize, viewport_height: usize) -> Self {
        Self {
            lines: Vec::new(),
            max_lines,
            scroll_offset: 0,
            viewport_height,
            auto_scroll: true,
        }
    }

    /// Add a line
    pub fn push(&mut self, line: String) {
        self.lines.push(line);

        // Trim old lines
        while self.lines.len() > self.max_lines {
            self.lines.remove(0);
            if self.scroll_offset > 0 {
                self.scroll_offset -= 1;
            }
        }

        // Auto-scroll to bottom
        if self.auto_scroll {
            self.scroll_to_bottom();
        }
    }

    /// Add multiple lines
    pub fn push_str(&mut self, text: &str) {
        for line in text.lines() {
            self.push(String::from(line));
        }
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.lines.clear();
        self.scroll_offset = 0;
    }

    /// Get visible lines
    pub fn visible_lines(&self) -> &[String] {
        let start = self.scroll_offset;
        let end = (start + self.viewport_height).min(self.lines.len());
        &self.lines[start..end]
    }

    /// Get all lines
    pub fn all_lines(&self) -> &[String] {
        &self.lines
    }

    /// Total line count
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Scroll up by n lines
    pub fn scroll_up(&mut self, n: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(n);
        self.auto_scroll = false;
    }

    /// Scroll down by n lines
    pub fn scroll_down(&mut self, n: usize) {
        let max_offset = self.lines.len().saturating_sub(self.viewport_height);
        self.scroll_offset = (self.scroll_offset + n).min(max_offset);

        // Re-enable auto-scroll if at bottom
        if self.scroll_offset >= max_offset {
            self.auto_scroll = true;
        }
    }

    /// Scroll to top
    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = 0;
        self.auto_scroll = false;
    }

    /// Scroll to bottom
    pub fn scroll_to_bottom(&mut self) {
        let max_offset = self.lines.len().saturating_sub(self.viewport_height);
        self.scroll_offset = max_offset;
        self.auto_scroll = true;
    }

    /// Page up
    pub fn page_up(&mut self) {
        self.scroll_up(self.viewport_height.saturating_sub(1));
    }

    /// Page down
    pub fn page_down(&mut self) {
        self.scroll_down(self.viewport_height.saturating_sub(1));
    }

    /// Set viewport height
    pub fn set_viewport_height(&mut self, height: usize) {
        self.viewport_height = height;
    }

    /// Get scroll position (0.0 - 1.0)
    pub fn scroll_position(&self) -> f32 {
        if self.lines.len() <= self.viewport_height {
            return 0.0;
        }

        let max_offset = self.lines.len() - self.viewport_height;
        self.scroll_offset as f32 / max_offset as f32
    }

    /// Set scroll position (0.0 - 1.0)
    pub fn set_scroll_position(&mut self, pos: f32) {
        let max_offset = self.lines.len().saturating_sub(self.viewport_height);
        self.scroll_offset = (pos * max_offset as f32) as usize;
        self.auto_scroll = self.scroll_offset >= max_offset;
    }

    /// Can scroll up
    pub fn can_scroll_up(&self) -> bool {
        self.scroll_offset > 0
    }

    /// Can scroll down
    pub fn can_scroll_down(&self) -> bool {
        self.scroll_offset + self.viewport_height < self.lines.len()
    }

    /// Get scroll info text
    pub fn scroll_info(&self) -> String {
        if self.lines.is_empty() {
            return String::new();
        }

        let start = self.scroll_offset + 1;
        let end = (self.scroll_offset + self.viewport_height).min(self.lines.len());
        alloc::format!("Lines {}-{} of {}", start, end, self.lines.len())
    }

    /// Toggle auto-scroll
    pub fn toggle_auto_scroll(&mut self) {
        self.auto_scroll = !self.auto_scroll;
        if self.auto_scroll {
            self.scroll_to_bottom();
        }
    }

    /// Is auto-scroll enabled
    pub fn is_auto_scroll(&self) -> bool {
        self.auto_scroll
    }
}

/// Scrollable list
pub struct ScrollList<T> {
    /// List items
    items: Vec<T>,
    /// Selected index
    selected: usize,
    /// Scroll offset
    scroll_offset: usize,
    /// Viewport height
    viewport_height: usize,
    /// Wrap selection
    wrap_selection: bool,
}

impl<T> ScrollList<T> {
    /// Create new scroll list
    pub fn new(viewport_height: usize) -> Self {
        Self {
            items: Vec::new(),
            selected: 0,
            scroll_offset: 0,
            viewport_height,
            wrap_selection: true,
        }
    }

    /// Set items
    pub fn set_items(&mut self, items: Vec<T>) {
        self.items = items;
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Add item
    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    /// Clear items
    pub fn clear(&mut self) {
        self.items.clear();
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Get items
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Get mutable items
    pub fn items_mut(&mut self) -> &mut [T] {
        &mut self.items
    }

    /// Get selected index
    pub fn selected_index(&self) -> usize {
        self.selected
    }

    /// Get selected item
    pub fn selected_item(&self) -> Option<&T> {
        self.items.get(self.selected)
    }

    /// Get selected item mutable
    pub fn selected_item_mut(&mut self) -> Option<&mut T> {
        self.items.get_mut(self.selected)
    }

    /// Item count
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Select next item
    pub fn select_next(&mut self) {
        if self.items.is_empty() {
            return;
        }

        if self.selected < self.items.len() - 1 {
            self.selected += 1;
        } else if self.wrap_selection {
            self.selected = 0;
        }

        self.ensure_visible();
    }

    /// Select previous item
    pub fn select_prev(&mut self) {
        if self.items.is_empty() {
            return;
        }

        if self.selected > 0 {
            self.selected -= 1;
        } else if self.wrap_selection {
            self.selected = self.items.len() - 1;
        }

        self.ensure_visible();
    }

    /// Select first item
    pub fn select_first(&mut self) {
        self.selected = 0;
        self.scroll_offset = 0;
    }

    /// Select last item
    pub fn select_last(&mut self) {
        if !self.items.is_empty() {
            self.selected = self.items.len() - 1;
            self.ensure_visible();
        }
    }

    /// Select by index
    pub fn select(&mut self, index: usize) {
        if index < self.items.len() {
            self.selected = index;
            self.ensure_visible();
        }
    }

    /// Ensure selected item is visible
    fn ensure_visible(&mut self) {
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + self.viewport_height {
            self.scroll_offset = self.selected - self.viewport_height + 1;
        }
    }

    /// Get visible items with indices
    pub fn visible_items(&self) -> impl Iterator<Item = (usize, &T)> {
        let start = self.scroll_offset;
        let end = (start + self.viewport_height).min(self.items.len());
        self.items[start..end]
            .iter()
            .enumerate()
            .map(move |(i, item)| (start + i, item))
    }

    /// Page up
    pub fn page_up(&mut self) {
        if self.items.is_empty() {
            return;
        }

        let jump = self.viewport_height.saturating_sub(1);
        self.selected = self.selected.saturating_sub(jump);
        self.ensure_visible();
    }

    /// Page down
    pub fn page_down(&mut self) {
        if self.items.is_empty() {
            return;
        }

        let jump = self.viewport_height.saturating_sub(1);
        self.selected = (self.selected + jump).min(self.items.len() - 1);
        self.ensure_visible();
    }

    /// Set viewport height
    pub fn set_viewport_height(&mut self, height: usize) {
        self.viewport_height = height;
        self.ensure_visible();
    }

    /// Set wrap selection
    pub fn set_wrap_selection(&mut self, wrap: bool) {
        self.wrap_selection = wrap;
    }

    /// Get scroll info
    pub fn scroll_info(&self) -> String {
        if self.items.is_empty() {
            return String::from("0/0");
        }
        alloc::format!("{}/{}", self.selected + 1, self.items.len())
    }

    /// Remove selected item
    pub fn remove_selected(&mut self) -> Option<T> {
        if self.items.is_empty() {
            return None;
        }

        let item = self.items.remove(self.selected);

        if self.selected >= self.items.len() && self.selected > 0 {
            self.selected -= 1;
        }

        self.ensure_visible();
        Some(item)
    }

    /// Find first matching item
    pub fn find<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&T) -> bool,
    {
        self.items.iter().position(predicate)
    }

    /// Find and select first matching item
    pub fn find_and_select<F>(&mut self, predicate: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        if let Some(idx) = self.find(predicate) {
            self.select(idx);
            true
        } else {
            false
        }
    }
}

impl<T: Clone> ScrollList<T> {
    /// Get selected item clone
    pub fn selected_clone(&self) -> Option<T> {
        self.selected_item().cloned()
    }
}

/// Scrollbar renderer
pub struct Scrollbar {
    /// Character for track
    pub track_char: char,
    /// Character for thumb
    pub thumb_char: char,
    /// Height
    pub height: usize,
}

impl Scrollbar {
    pub fn new(height: usize) -> Self {
        Self {
            track_char: '│',
            thumb_char: '█',
            height,
        }
    }

    /// Render scrollbar
    pub fn render(&self, position: f32, thumb_size: f32) -> Vec<char> {
        let mut result = vec![self.track_char; self.height];

        let thumb_len = ((thumb_size * self.height as f32).max(1.0)) as usize;
        let thumb_pos =
            ((position * (self.height - thumb_len) as f32) as usize).min(self.height - thumb_len);

        for i in 0..thumb_len {
            if thumb_pos + i < self.height {
                result[thumb_pos + i] = self.thumb_char;
            }
        }

        result
    }

    /// Render as string
    pub fn render_string(&self, position: f32, thumb_size: f32) -> String {
        self.render(position, thumb_size).into_iter().collect()
    }
}

impl Default for Scrollbar {
    fn default() -> Self {
        Self::new(10)
    }
}
