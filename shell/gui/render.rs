//! GUI Rendering Primitives
//!
//! Basic drawing operations for the GUI.

use super::{Rect, Color, Point};

/// Draw a filled rectangle
pub fn fill_rect(
    buffer: &mut [u32],
    screen_width: u32,
    rect: Rect,
    color: Color,
) {
    let argb = color.to_argb();

    for y in 0..rect.height {
        for x in 0..rect.width {
            let px = rect.x + x as i32;
            let py = rect.y + y as i32;

            if px >= 0 && py >= 0 {
                let idx = (py as u32 * screen_width + px as u32) as usize;
                if idx < buffer.len() {
                    buffer[idx] = argb;
                }
            }
        }
    }
}

/// Draw a rectangle border
pub fn draw_rect(
    buffer: &mut [u32],
    screen_width: u32,
    rect: Rect,
    color: Color,
    thickness: u32,
) {
    let argb = color.to_argb();

    // Top edge
    for t in 0..thickness {
        let y = rect.y + t as i32;
        if y >= 0 {
            for x in 0..rect.width {
                let px = rect.x + x as i32;
                if px >= 0 {
                    let idx = (y as u32 * screen_width + px as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = argb;
                    }
                }
            }
        }
    }

    // Bottom edge
    for t in 0..thickness {
        let y = rect.y + rect.height as i32 - 1 - t as i32;
        if y >= 0 {
            for x in 0..rect.width {
                let px = rect.x + x as i32;
                if px >= 0 {
                    let idx = (y as u32 * screen_width + px as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = argb;
                    }
                }
            }
        }
    }

    // Left edge
    for t in 0..thickness {
        let x = rect.x + t as i32;
        if x >= 0 {
            for y in 0..rect.height {
                let py = rect.y + y as i32;
                if py >= 0 {
                    let idx = (py as u32 * screen_width + x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = argb;
                    }
                }
            }
        }
    }

    // Right edge
    for t in 0..thickness {
        let x = rect.x + rect.width as i32 - 1 - t as i32;
        if x >= 0 {
            for y in 0..rect.height {
                let py = rect.y + y as i32;
                if py >= 0 {
                    let idx = (py as u32 * screen_width + x as u32) as usize;
                    if idx < buffer.len() {
                        buffer[idx] = argb;
                    }
                }
            }
        }
    }
}

/// Draw a horizontal line
pub fn draw_hline(
    buffer: &mut [u32],
    screen_width: u32,
    x: i32,
    y: i32,
    width: u32,
    color: Color,
) {
    if y < 0 {
        return;
    }

    let argb = color.to_argb();

    for dx in 0..width {
        let px = x + dx as i32;
        if px >= 0 {
            let idx = (y as u32 * screen_width + px as u32) as usize;
            if idx < buffer.len() {
                buffer[idx] = argb;
            }
        }
    }
}

/// Draw a vertical line
pub fn draw_vline(
    buffer: &mut [u32],
    screen_width: u32,
    x: i32,
    y: i32,
    height: u32,
    color: Color,
) {
    if x < 0 {
        return;
    }

    let argb = color.to_argb();

    for dy in 0..height {
        let py = y + dy as i32;
        if py >= 0 {
            let idx = (py as u32 * screen_width + x as u32) as usize;
            if idx < buffer.len() {
                buffer[idx] = argb;
            }
        }
    }
}

/// Draw a line using Bresenham's algorithm
pub fn draw_line(
    buffer: &mut [u32],
    screen_width: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Color,
) {
    let argb = color.to_argb();

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let mut x = x0;
    let mut y = y0;

    loop {
        if x >= 0 && y >= 0 {
            let idx = (y as u32 * screen_width + x as u32) as usize;
            if idx < buffer.len() {
                buffer[idx] = argb;
            }
        }

        if x == x1 && y == y1 {
            break;
        }

        let e2 = 2 * err;

        if e2 >= dy {
            err += dy;
            x += sx;
        }

        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a circle outline using midpoint algorithm
pub fn draw_circle(
    buffer: &mut [u32],
    screen_width: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    color: Color,
) {
    let argb = color.to_argb();

    let mut x = radius;
    let mut y = 0;
    let mut err = 0;

    while x >= y {
        set_pixel_safe(buffer, screen_width, cx + x, cy + y, argb);
        set_pixel_safe(buffer, screen_width, cx + y, cy + x, argb);
        set_pixel_safe(buffer, screen_width, cx - y, cy + x, argb);
        set_pixel_safe(buffer, screen_width, cx - x, cy + y, argb);
        set_pixel_safe(buffer, screen_width, cx - x, cy - y, argb);
        set_pixel_safe(buffer, screen_width, cx - y, cy - x, argb);
        set_pixel_safe(buffer, screen_width, cx + y, cy - x, argb);
        set_pixel_safe(buffer, screen_width, cx + x, cy - y, argb);

        y += 1;
        err += 1 + 2 * y;

        if 2 * (err - x) + 1 > 0 {
            x -= 1;
            err += 1 - 2 * x;
        }
    }
}

/// Draw a filled circle
pub fn fill_circle(
    buffer: &mut [u32],
    screen_width: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    color: Color,
) {
    let argb = color.to_argb();

    let mut x = radius;
    let mut y = 0;
    let mut err = 0;

    while x >= y {
        draw_hline_safe(buffer, screen_width, cx - x, cy + y, (2 * x) as u32, argb);
        draw_hline_safe(buffer, screen_width, cx - y, cy + x, (2 * y) as u32, argb);
        draw_hline_safe(buffer, screen_width, cx - x, cy - y, (2 * x) as u32, argb);
        draw_hline_safe(buffer, screen_width, cx - y, cy - x, (2 * y) as u32, argb);

        y += 1;
        err += 1 + 2 * y;

        if 2 * (err - x) + 1 > 0 {
            x -= 1;
            err += 1 - 2 * x;
        }
    }
}

/// Draw a rounded rectangle
pub fn draw_rounded_rect(
    buffer: &mut [u32],
    screen_width: u32,
    rect: Rect,
    radius: u32,
    color: Color,
) {
    let argb = color.to_argb();
    let r = radius as i32;

    // Top edge (excluding corners)
    draw_hline(buffer, screen_width, rect.x + r, rect.y, rect.width - 2 * radius, color);

    // Bottom edge (excluding corners)
    draw_hline(buffer, screen_width, rect.x + r, rect.y + rect.height as i32 - 1, rect.width - 2 * radius, color);

    // Left edge (excluding corners)
    draw_vline(buffer, screen_width, rect.x, rect.y + r, rect.height - 2 * radius, color);

    // Right edge (excluding corners)
    draw_vline(buffer, screen_width, rect.x + rect.width as i32 - 1, rect.y + r, rect.height - 2 * radius, color);

    // Corners using arcs
    draw_corner_arc(buffer, screen_width, rect.x + r, rect.y + r, r, 2, argb);  // Top-left
    draw_corner_arc(buffer, screen_width, rect.x + rect.width as i32 - r - 1, rect.y + r, r, 1, argb);  // Top-right
    draw_corner_arc(buffer, screen_width, rect.x + r, rect.y + rect.height as i32 - r - 1, r, 3, argb);  // Bottom-left
    draw_corner_arc(buffer, screen_width, rect.x + rect.width as i32 - r - 1, rect.y + rect.height as i32 - r - 1, r, 0, argb);  // Bottom-right
}

/// Draw a filled rounded rectangle
pub fn fill_rounded_rect(
    buffer: &mut [u32],
    screen_width: u32,
    rect: Rect,
    radius: u32,
    color: Color,
) {
    let argb = color.to_argb();
    let r = radius as i32;

    // Fill main body (excluding corners)
    fill_rect(buffer, screen_width,
        Rect::new(rect.x + r, rect.y, rect.width - 2 * radius, rect.height),
        color);

    // Fill left side
    fill_rect(buffer, screen_width,
        Rect::new(rect.x, rect.y + r, radius, rect.height - 2 * radius),
        color);

    // Fill right side
    fill_rect(buffer, screen_width,
        Rect::new(rect.x + rect.width as i32 - r, rect.y + r, radius, rect.height - 2 * radius),
        color);

    // Fill corners with arcs
    fill_corner_arc(buffer, screen_width, rect.x + r, rect.y + r, r, 2, argb);
    fill_corner_arc(buffer, screen_width, rect.x + rect.width as i32 - r - 1, rect.y + r, r, 1, argb);
    fill_corner_arc(buffer, screen_width, rect.x + r, rect.y + rect.height as i32 - r - 1, r, 3, argb);
    fill_corner_arc(buffer, screen_width, rect.x + rect.width as i32 - r - 1, rect.y + rect.height as i32 - r - 1, r, 0, argb);
}

// Helper to draw a corner arc (quadrant: 0=BR, 1=TR, 2=TL, 3=BL)
fn draw_corner_arc(
    buffer: &mut [u32],
    screen_width: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    quadrant: u8,
    argb: u32,
) {
    let mut x = radius;
    let mut y = 0;
    let mut err = 0;

    while x >= y {
        match quadrant {
            0 => {  // Bottom-right
                set_pixel_safe(buffer, screen_width, cx + x, cy + y, argb);
                set_pixel_safe(buffer, screen_width, cx + y, cy + x, argb);
            }
            1 => {  // Top-right
                set_pixel_safe(buffer, screen_width, cx + x, cy - y, argb);
                set_pixel_safe(buffer, screen_width, cx + y, cy - x, argb);
            }
            2 => {  // Top-left
                set_pixel_safe(buffer, screen_width, cx - x, cy - y, argb);
                set_pixel_safe(buffer, screen_width, cx - y, cy - x, argb);
            }
            3 => {  // Bottom-left
                set_pixel_safe(buffer, screen_width, cx - x, cy + y, argb);
                set_pixel_safe(buffer, screen_width, cx - y, cy + x, argb);
            }
            _ => {}
        }

        y += 1;
        err += 1 + 2 * y;

        if 2 * (err - x) + 1 > 0 {
            x -= 1;
            err += 1 - 2 * x;
        }
    }
}

// Helper to fill a corner arc
fn fill_corner_arc(
    buffer: &mut [u32],
    screen_width: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    quadrant: u8,
    argb: u32,
) {
    let mut x = radius;
    let mut y = 0;
    let mut err = 0;

    while x >= y {
        match quadrant {
            0 => {  // Bottom-right
                draw_hline_safe(buffer, screen_width, cx, cy + y, x as u32 + 1, argb);
                draw_hline_safe(buffer, screen_width, cx, cy + x, y as u32 + 1, argb);
            }
            1 => {  // Top-right
                draw_hline_safe(buffer, screen_width, cx, cy - y, x as u32 + 1, argb);
                draw_hline_safe(buffer, screen_width, cx, cy - x, y as u32 + 1, argb);
            }
            2 => {  // Top-left
                draw_hline_safe(buffer, screen_width, cx - x, cy - y, x as u32 + 1, argb);
                draw_hline_safe(buffer, screen_width, cx - y, cy - x, y as u32 + 1, argb);
            }
            3 => {  // Bottom-left
                draw_hline_safe(buffer, screen_width, cx - x, cy + y, x as u32 + 1, argb);
                draw_hline_safe(buffer, screen_width, cx - y, cy + x, y as u32 + 1, argb);
            }
            _ => {}
        }

        y += 1;
        err += 1 + 2 * y;

        if 2 * (err - x) + 1 > 0 {
            x -= 1;
            err += 1 - 2 * x;
        }
    }
}

// Safe pixel setting helper
fn set_pixel_safe(buffer: &mut [u32], screen_width: u32, x: i32, y: i32, argb: u32) {
    if x >= 0 && y >= 0 {
        let idx = (y as u32 * screen_width + x as u32) as usize;
        if idx < buffer.len() {
            buffer[idx] = argb;
        }
    }
}

// Safe horizontal line helper
fn draw_hline_safe(buffer: &mut [u32], screen_width: u32, x: i32, y: i32, width: u32, argb: u32) {
    if y < 0 {
        return;
    }

    for dx in 0..width {
        let px = x + dx as i32;
        if px >= 0 {
            let idx = (y as u32 * screen_width + px as u32) as usize;
            if idx < buffer.len() {
                buffer[idx] = argb;
            }
        }
    }
}

/// Draw a triangle outline
pub fn draw_triangle(
    buffer: &mut [u32],
    screen_width: u32,
    p1: Point,
    p2: Point,
    p3: Point,
    color: Color,
) {
    draw_line(buffer, screen_width, p1.x, p1.y, p2.x, p2.y, color);
    draw_line(buffer, screen_width, p2.x, p2.y, p3.x, p3.y, color);
    draw_line(buffer, screen_width, p3.x, p3.y, p1.x, p1.y, color);
}

/// Draw a filled triangle using scanline algorithm
pub fn fill_triangle(
    buffer: &mut [u32],
    screen_width: u32,
    mut p1: Point,
    mut p2: Point,
    mut p3: Point,
    color: Color,
) {
    // Sort points by y-coordinate
    if p1.y > p2.y { core::mem::swap(&mut p1, &mut p2); }
    if p1.y > p3.y { core::mem::swap(&mut p1, &mut p3); }
    if p2.y > p3.y { core::mem::swap(&mut p2, &mut p3); }

    let argb = color.to_argb();

    // Fill bottom flat triangle and top flat triangle
    if p2.y == p3.y {
        fill_bottom_flat_triangle(buffer, screen_width, p1, p2, p3, argb);
    } else if p1.y == p2.y {
        fill_top_flat_triangle(buffer, screen_width, p1, p2, p3, argb);
    } else {
        // Split into two triangles
        let p4 = Point::new(
            p1.x + ((p2.y - p1.y) * (p3.x - p1.x)) / (p3.y - p1.y),
            p2.y,
        );
        fill_bottom_flat_triangle(buffer, screen_width, p1, p2, p4, argb);
        fill_top_flat_triangle(buffer, screen_width, p2, p4, p3, argb);
    }
}

fn fill_bottom_flat_triangle(
    buffer: &mut [u32],
    screen_width: u32,
    p1: Point,
    p2: Point,
    p3: Point,
    argb: u32,
) {
    let invslope1 = (p2.x - p1.x) as f32 / (p2.y - p1.y) as f32;
    let invslope2 = (p3.x - p1.x) as f32 / (p3.y - p1.y) as f32;

    let mut curx1 = p1.x as f32;
    let mut curx2 = p1.x as f32;

    for y in p1.y..=p2.y {
        let x1 = curx1 as i32;
        let x2 = curx2 as i32;
        let (start, end) = if x1 < x2 { (x1, x2) } else { (x2, x1) };

        draw_hline_safe(buffer, screen_width, start, y, (end - start + 1) as u32, argb);

        curx1 += invslope1;
        curx2 += invslope2;
    }
}

fn fill_top_flat_triangle(
    buffer: &mut [u32],
    screen_width: u32,
    p1: Point,
    p2: Point,
    p3: Point,
    argb: u32,
) {
    let invslope1 = (p3.x - p1.x) as f32 / (p3.y - p1.y) as f32;
    let invslope2 = (p3.x - p2.x) as f32 / (p3.y - p2.y) as f32;

    let mut curx1 = p3.x as f32;
    let mut curx2 = p3.x as f32;

    for y in (p1.y..=p3.y).rev() {
        let x1 = curx1 as i32;
        let x2 = curx2 as i32;
        let (start, end) = if x1 < x2 { (x1, x2) } else { (x2, x1) };

        draw_hline_safe(buffer, screen_width, start, y, (end - start + 1) as u32, argb);

        curx1 -= invslope1;
        curx2 -= invslope2;
    }
}

/// Blend source color onto destination with alpha
pub fn blend_pixel(dest: u32, src: u32) -> u32 {
    let sa = ((src >> 24) & 0xFF) as u32;

    if sa == 0 {
        return dest;
    }

    if sa == 255 {
        return src;
    }

    let sr = (src >> 16) & 0xFF;
    let sg = (src >> 8) & 0xFF;
    let sb = src & 0xFF;

    let dr = (dest >> 16) & 0xFF;
    let dg = (dest >> 8) & 0xFF;
    let db = dest & 0xFF;

    let inv_sa = 255 - sa;

    let r = (sr * sa + dr * inv_sa) / 255;
    let g = (sg * sa + dg * inv_sa) / 255;
    let b = (sb * sa + db * inv_sa) / 255;

    0xFF000000 | (r << 16) | (g << 8) | b
}

/// Copy a rectangular region
pub fn blit(
    dest: &mut [u32],
    dest_width: u32,
    dest_x: i32,
    dest_y: i32,
    src: &[u32],
    src_width: u32,
    src_rect: Rect,
) {
    for y in 0..src_rect.height {
        for x in 0..src_rect.width {
            let sx = src_rect.x + x as i32;
            let sy = src_rect.y + y as i32;
            let dx = dest_x + x as i32;
            let dy = dest_y + y as i32;

            if sx >= 0 && sy >= 0 && dx >= 0 && dy >= 0 {
                let src_idx = (sy as u32 * src_width + sx as u32) as usize;
                let dest_idx = (dy as u32 * dest_width + dx as u32) as usize;

                if src_idx < src.len() && dest_idx < dest.len() {
                    dest[dest_idx] = blend_pixel(dest[dest_idx], src[src_idx]);
                }
            }
        }
    }
}
