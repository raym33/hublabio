//! GUI Layout System
//!
//! Layout management for arranging widgets.

use super::Rect;

/// Layout direction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Horizontal,
    Vertical,
}

/// Alignment options
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    Start,
    Center,
    End,
    Stretch,
}

/// Size constraint
#[derive(Clone, Copy, Debug)]
pub enum SizeConstraint {
    /// Fixed size in pixels
    Fixed(u32),
    /// Percentage of available space
    Percent(f32),
    /// Fill remaining space (with weight)
    Fill(f32),
    /// Fit to content
    FitContent,
}

impl Default for SizeConstraint {
    fn default() -> Self {
        Self::FitContent
    }
}

/// Layout item configuration
#[derive(Clone, Copy, Debug)]
pub struct LayoutItem {
    /// Width constraint
    pub width: SizeConstraint,
    /// Height constraint
    pub height: SizeConstraint,
    /// Horizontal alignment
    pub h_align: Alignment,
    /// Vertical alignment
    pub v_align: Alignment,
    /// Margin (top, right, bottom, left)
    pub margin: (u32, u32, u32, u32),
    /// Minimum size
    pub min_size: (u32, u32),
    /// Maximum size
    pub max_size: (u32, u32),
}

impl Default for LayoutItem {
    fn default() -> Self {
        Self {
            width: SizeConstraint::FitContent,
            height: SizeConstraint::FitContent,
            h_align: Alignment::Start,
            v_align: Alignment::Start,
            margin: (0, 0, 0, 0),
            min_size: (0, 0),
            max_size: (u32::MAX, u32::MAX),
        }
    }
}

impl LayoutItem {
    pub fn fixed(width: u32, height: u32) -> Self {
        Self {
            width: SizeConstraint::Fixed(width),
            height: SizeConstraint::Fixed(height),
            ..Default::default()
        }
    }

    pub fn fill() -> Self {
        Self {
            width: SizeConstraint::Fill(1.0),
            height: SizeConstraint::Fill(1.0),
            ..Default::default()
        }
    }

    pub fn with_margin(mut self, margin: u32) -> Self {
        self.margin = (margin, margin, margin, margin);
        self
    }

    pub fn with_margins(mut self, top: u32, right: u32, bottom: u32, left: u32) -> Self {
        self.margin = (top, right, bottom, left);
        self
    }

    pub fn centered(mut self) -> Self {
        self.h_align = Alignment::Center;
        self.v_align = Alignment::Center;
        self
    }
}

/// Linear layout (horizontal or vertical)
pub struct LinearLayout {
    /// Direction
    pub direction: Direction,
    /// Spacing between items
    pub spacing: u32,
    /// Padding (top, right, bottom, left)
    pub padding: (u32, u32, u32, u32),
    /// Main axis alignment
    pub main_align: Alignment,
    /// Cross axis alignment
    pub cross_align: Alignment,
}

impl LinearLayout {
    pub fn horizontal() -> Self {
        Self {
            direction: Direction::Horizontal,
            spacing: 0,
            padding: (0, 0, 0, 0),
            main_align: Alignment::Start,
            cross_align: Alignment::Start,
        }
    }

    pub fn vertical() -> Self {
        Self {
            direction: Direction::Vertical,
            spacing: 0,
            padding: (0, 0, 0, 0),
            main_align: Alignment::Start,
            cross_align: Alignment::Start,
        }
    }

    pub fn with_spacing(mut self, spacing: u32) -> Self {
        self.spacing = spacing;
        self
    }

    pub fn with_padding(mut self, padding: u32) -> Self {
        self.padding = (padding, padding, padding, padding);
        self
    }

    /// Calculate bounds for items
    pub fn layout(
        &self,
        container: Rect,
        items: &[LayoutItem],
        content_sizes: &[(u32, u32)],
    ) -> Vec<Rect> {
        let mut results = Vec::new();

        // Calculate available space
        let available_width = container
            .width
            .saturating_sub(self.padding.1 + self.padding.3);
        let available_height = container
            .height
            .saturating_sub(self.padding.0 + self.padding.2);

        let content_x = container.x + self.padding.3 as i32;
        let content_y = container.y + self.padding.0 as i32;

        match self.direction {
            Direction::Horizontal => {
                let total_spacing = self.spacing * items.len().saturating_sub(1) as u32;
                let mut total_fixed = 0u32;
                let mut total_fill_weight = 0.0f32;

                // First pass: calculate fixed sizes and total fill weight
                for (i, item) in items.iter().enumerate() {
                    let content_w = content_sizes.get(i).map(|(w, _)| *w).unwrap_or(0);

                    match item.width {
                        SizeConstraint::Fixed(w) => {
                            total_fixed += w + item.margin.1 + item.margin.3
                        }
                        SizeConstraint::Percent(p) => {
                            total_fixed += ((available_width as f32 * p) as u32)
                                + item.margin.1
                                + item.margin.3
                        }
                        SizeConstraint::Fill(w) => total_fill_weight += w,
                        SizeConstraint::FitContent => {
                            total_fixed += content_w + item.margin.1 + item.margin.3
                        }
                    }
                }

                let fill_space = available_width.saturating_sub(total_fixed + total_spacing);

                // Second pass: calculate positions
                let mut x = content_x;

                for (i, item) in items.iter().enumerate() {
                    let (content_w, content_h) = content_sizes.get(i).copied().unwrap_or((0, 0));

                    let item_width = match item.width {
                        SizeConstraint::Fixed(w) => w,
                        SizeConstraint::Percent(p) => (available_width as f32 * p) as u32,
                        SizeConstraint::Fill(w) => {
                            if total_fill_weight > 0.0 {
                                (fill_space as f32 * w / total_fill_weight) as u32
                            } else {
                                0
                            }
                        }
                        SizeConstraint::FitContent => content_w,
                    }
                    .clamp(item.min_size.0, item.max_size.0);

                    let item_height = match item.height {
                        SizeConstraint::Fixed(h) => h,
                        SizeConstraint::Percent(p) => (available_height as f32 * p) as u32,
                        SizeConstraint::Fill(_) => {
                            available_height.saturating_sub(item.margin.0 + item.margin.2)
                        }
                        SizeConstraint::FitContent => content_h,
                    }
                    .clamp(item.min_size.1, item.max_size.1);

                    let item_x = x + item.margin.3 as i32;
                    let item_y = match item.v_align {
                        Alignment::Start => content_y + item.margin.0 as i32,
                        Alignment::Center => {
                            content_y + ((available_height - item_height) / 2) as i32
                        }
                        Alignment::End => {
                            content_y + (available_height - item_height - item.margin.2) as i32
                        }
                        Alignment::Stretch => content_y + item.margin.0 as i32,
                    };

                    results.push(Rect::new(item_x, item_y, item_width, item_height));

                    x += item.margin.3 as i32
                        + item_width as i32
                        + item.margin.1 as i32
                        + self.spacing as i32;
                }
            }

            Direction::Vertical => {
                let total_spacing = self.spacing * items.len().saturating_sub(1) as u32;
                let mut total_fixed = 0u32;
                let mut total_fill_weight = 0.0f32;

                // First pass
                for (i, item) in items.iter().enumerate() {
                    let content_h = content_sizes.get(i).map(|(_, h)| *h).unwrap_or(0);

                    match item.height {
                        SizeConstraint::Fixed(h) => {
                            total_fixed += h + item.margin.0 + item.margin.2
                        }
                        SizeConstraint::Percent(p) => {
                            total_fixed += ((available_height as f32 * p) as u32)
                                + item.margin.0
                                + item.margin.2
                        }
                        SizeConstraint::Fill(w) => total_fill_weight += w,
                        SizeConstraint::FitContent => {
                            total_fixed += content_h + item.margin.0 + item.margin.2
                        }
                    }
                }

                let fill_space = available_height.saturating_sub(total_fixed + total_spacing);

                // Second pass
                let mut y = content_y;

                for (i, item) in items.iter().enumerate() {
                    let (content_w, content_h) = content_sizes.get(i).copied().unwrap_or((0, 0));

                    let item_width = match item.width {
                        SizeConstraint::Fixed(w) => w,
                        SizeConstraint::Percent(p) => (available_width as f32 * p) as u32,
                        SizeConstraint::Fill(_) => {
                            available_width.saturating_sub(item.margin.1 + item.margin.3)
                        }
                        SizeConstraint::FitContent => content_w,
                    }
                    .clamp(item.min_size.0, item.max_size.0);

                    let item_height = match item.height {
                        SizeConstraint::Fixed(h) => h,
                        SizeConstraint::Percent(p) => (available_height as f32 * p) as u32,
                        SizeConstraint::Fill(w) => {
                            if total_fill_weight > 0.0 {
                                (fill_space as f32 * w / total_fill_weight) as u32
                            } else {
                                0
                            }
                        }
                        SizeConstraint::FitContent => content_h,
                    }
                    .clamp(item.min_size.1, item.max_size.1);

                    let item_x = match item.h_align {
                        Alignment::Start => content_x + item.margin.3 as i32,
                        Alignment::Center => {
                            content_x + ((available_width - item_width) / 2) as i32
                        }
                        Alignment::End => {
                            content_x + (available_width - item_width - item.margin.1) as i32
                        }
                        Alignment::Stretch => content_x + item.margin.3 as i32,
                    };
                    let item_y = y + item.margin.0 as i32;

                    results.push(Rect::new(item_x, item_y, item_width, item_height));

                    y += item.margin.0 as i32
                        + item_height as i32
                        + item.margin.2 as i32
                        + self.spacing as i32;
                }
            }
        }

        results
    }
}

/// Grid layout
pub struct GridLayout {
    /// Number of columns
    pub columns: u32,
    /// Row height
    pub row_height: SizeConstraint,
    /// Column width
    pub column_width: SizeConstraint,
    /// Horizontal spacing
    pub h_spacing: u32,
    /// Vertical spacing
    pub v_spacing: u32,
    /// Padding
    pub padding: (u32, u32, u32, u32),
}

impl GridLayout {
    pub fn new(columns: u32) -> Self {
        Self {
            columns,
            row_height: SizeConstraint::FitContent,
            column_width: SizeConstraint::Fill(1.0),
            h_spacing: 0,
            v_spacing: 0,
            padding: (0, 0, 0, 0),
        }
    }

    pub fn with_spacing(mut self, h: u32, v: u32) -> Self {
        self.h_spacing = h;
        self.v_spacing = v;
        self
    }

    /// Calculate bounds for items
    pub fn layout(
        &self,
        container: Rect,
        item_count: usize,
        content_sizes: &[(u32, u32)],
    ) -> Vec<Rect> {
        let mut results = Vec::new();

        let available_width = container
            .width
            .saturating_sub(self.padding.1 + self.padding.3);
        let available_height = container
            .height
            .saturating_sub(self.padding.0 + self.padding.2);

        let content_x = container.x + self.padding.3 as i32;
        let content_y = container.y + self.padding.0 as i32;

        let total_h_spacing = self.h_spacing * (self.columns - 1);
        let col_width = match self.column_width {
            SizeConstraint::Fixed(w) => w,
            SizeConstraint::Fill(_) => {
                (available_width.saturating_sub(total_h_spacing)) / self.columns
            }
            _ => (available_width.saturating_sub(total_h_spacing)) / self.columns,
        };

        let rows = (item_count as u32 + self.columns - 1) / self.columns;
        let total_v_spacing = self.v_spacing * rows.saturating_sub(1);
        let row_height = match self.row_height {
            SizeConstraint::Fixed(h) => h,
            SizeConstraint::Fill(_) => (available_height.saturating_sub(total_v_spacing)) / rows,
            _ => content_sizes.iter().map(|(_, h)| *h).max().unwrap_or(30),
        };

        for i in 0..item_count {
            let col = (i as u32) % self.columns;
            let row = (i as u32) / self.columns;

            let x = content_x + (col * (col_width + self.h_spacing)) as i32;
            let y = content_y + (row * (row_height + self.v_spacing)) as i32;

            results.push(Rect::new(x, y, col_width, row_height));
        }

        results
    }
}

/// Anchor-based layout (for absolute positioning relative to edges)
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    /// Distance from left (None = not anchored)
    pub left: Option<i32>,
    /// Distance from top
    pub top: Option<i32>,
    /// Distance from right
    pub right: Option<i32>,
    /// Distance from bottom
    pub bottom: Option<i32>,
}

impl Anchor {
    pub const fn new() -> Self {
        Self {
            left: None,
            top: None,
            right: None,
            bottom: None,
        }
    }

    pub fn top_left(left: i32, top: i32) -> Self {
        Self {
            left: Some(left),
            top: Some(top),
            right: None,
            bottom: None,
        }
    }

    pub fn top_right(right: i32, top: i32) -> Self {
        Self {
            left: None,
            top: Some(top),
            right: Some(right),
            bottom: None,
        }
    }

    pub fn bottom_left(left: i32, bottom: i32) -> Self {
        Self {
            left: Some(left),
            top: None,
            right: None,
            bottom: Some(bottom),
        }
    }

    pub fn bottom_right(right: i32, bottom: i32) -> Self {
        Self {
            left: None,
            top: None,
            right: Some(right),
            bottom: Some(bottom),
        }
    }

    pub fn fill() -> Self {
        Self {
            left: Some(0),
            top: Some(0),
            right: Some(0),
            bottom: Some(0),
        }
    }

    /// Calculate bounds based on anchor and container
    pub fn calculate(&self, container: Rect, default_width: u32, default_height: u32) -> Rect {
        let x = match (self.left, self.right) {
            (Some(l), Some(r)) => container.x + l,
            (Some(l), None) => container.x + l,
            (None, Some(r)) => container.x + container.width as i32 - r - default_width as i32,
            (None, None) => container.x,
        };

        let y = match (self.top, self.bottom) {
            (Some(t), Some(b)) => container.y + t,
            (Some(t), None) => container.y + t,
            (None, Some(b)) => container.y + container.height as i32 - b - default_height as i32,
            (None, None) => container.y,
        };

        let width = match (self.left, self.right) {
            (Some(l), Some(r)) => (container.width as i32 - l - r).max(0) as u32,
            _ => default_width,
        };

        let height = match (self.top, self.bottom) {
            (Some(t), Some(b)) => (container.height as i32 - t - b).max(0) as u32,
            _ => default_height,
        };

        Rect::new(x, y, width, height)
    }
}

impl Default for Anchor {
    fn default() -> Self {
        Self::new()
    }
}
