//! Quantization support for efficient model inference
//!
//! Supports various quantization formats used by GGUF models.

use alloc::vec::Vec;

/// Quantization types supported
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum QuantType {
    /// 32-bit float
    F32 = 0,
    /// 16-bit float
    F16 = 1,
    /// 4-bit quantization (32 weights per block)
    Q4_0 = 2,
    /// 4-bit with 16 weights per block
    Q4_1 = 3,
    /// 5-bit quantization
    Q5_0 = 6,
    /// 5-bit with additional bits
    Q5_1 = 7,
    /// 8-bit quantization
    Q8_0 = 8,
    /// 8-bit with 256 weights per block
    Q8_1 = 9,
    /// 2-bit quantization (aggressive)
    Q2_K = 10,
    /// 3-bit quantization
    Q3_K = 11,
    /// 4-bit k-quant
    Q4_K = 12,
    /// 5-bit k-quant
    Q5_K = 13,
    /// 6-bit k-quant
    Q6_K = 14,
    /// 8-bit k-quant
    Q8_K = 15,
    /// IQ2 (2-bit importance quantization)
    IQ2_XXS = 16,
    /// IQ2 extra small
    IQ2_XS = 17,
    /// IQ3 extra small
    IQ3_XXS = 18,
    /// IQ1 small
    IQ1_S = 19,
    /// IQ4 non-linear
    IQ4_NL = 20,
    /// IQ3 small
    IQ3_S = 21,
    /// IQ2 small
    IQ2_S = 22,
    /// IQ4 extra small
    IQ4_XS = 23,
}

impl QuantType {
    /// Get bytes per element (for non-blocked types)
    pub fn element_size(&self) -> usize {
        match self {
            QuantType::F32 => 4,
            QuantType::F16 => 2,
            _ => 0, // Block-based types
        }
    }

    /// Get block size for quantized types
    pub fn block_size(&self) -> usize {
        match self {
            QuantType::Q4_0 | QuantType::Q4_1 => 32,
            QuantType::Q5_0 | QuantType::Q5_1 => 32,
            QuantType::Q8_0 => 32,
            QuantType::Q8_1 => 32,
            QuantType::Q2_K => 256,
            QuantType::Q3_K => 256,
            QuantType::Q4_K => 256,
            QuantType::Q5_K => 256,
            QuantType::Q6_K => 256,
            QuantType::Q8_K => 256,
            _ => 1,
        }
    }

    /// Get bytes per block
    pub fn bytes_per_block(&self) -> usize {
        match self {
            QuantType::F32 => 4,
            QuantType::F16 => 2,
            QuantType::Q4_0 => 18, // 2 (scale) + 16 (32 * 4-bit / 8)
            QuantType::Q4_1 => 20, // 2 (scale) + 2 (min) + 16
            QuantType::Q5_0 => 22, // 2 + 4 (high bits) + 16
            QuantType::Q5_1 => 24,
            QuantType::Q8_0 => 34, // 2 + 32
            QuantType::Q8_1 => 36, // 2 + 2 + 32
            QuantType::Q2_K => 84,
            QuantType::Q3_K => 110,
            QuantType::Q4_K => 144,
            QuantType::Q5_K => 176,
            QuantType::Q6_K => 210,
            QuantType::Q8_K => 292,
            _ => 1,
        }
    }

    /// Get bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantType::F32 => 32.0,
            QuantType::F16 => 16.0,
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q4_K => 4.5,
            QuantType::Q5_0 | QuantType::Q5_1 | QuantType::Q5_K => 5.5,
            QuantType::Q8_0 | QuantType::Q8_1 | QuantType::Q8_K => 8.5,
            QuantType::Q2_K => 2.6,
            QuantType::Q3_K => 3.4,
            QuantType::Q6_K => 6.6,
            QuantType::IQ2_XXS => 2.1,
            QuantType::IQ2_XS => 2.3,
            QuantType::IQ2_S => 2.5,
            QuantType::IQ3_XXS => 3.1,
            QuantType::IQ3_S => 3.4,
            QuantType::IQ4_NL | QuantType::IQ4_XS => 4.3,
            QuantType::IQ1_S => 1.6,
        }
    }

    /// Check if this is a K-quant type
    pub fn is_k_quant(&self) -> bool {
        matches!(
            self,
            QuantType::Q2_K
                | QuantType::Q3_K
                | QuantType::Q4_K
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::Q8_K
        )
    }

    /// Check if this is an importance quantization type
    pub fn is_iq(&self) -> bool {
        matches!(
            self,
            QuantType::IQ2_XXS
                | QuantType::IQ2_XS
                | QuantType::IQ2_S
                | QuantType::IQ3_XXS
                | QuantType::IQ3_S
                | QuantType::IQ4_NL
                | QuantType::IQ4_XS
                | QuantType::IQ1_S
        )
    }
}

/// Q4_0 block structure
#[repr(C)]
pub struct BlockQ4_0 {
    /// Scale factor (f16)
    pub scale: u16,
    /// Quantized values (32 x 4-bit = 16 bytes)
    pub quants: [u8; 16],
}

/// Q8_0 block structure
#[repr(C)]
pub struct BlockQ8_0 {
    /// Scale factor (f16)
    pub scale: u16,
    /// Quantized values (32 x 8-bit)
    pub quants: [i8; 32],
}

/// Dequantize Q4_0 block to f32
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32; 32]) {
    let scale = f16_to_f32(block.scale);

    for (i, &byte) in block.quants.iter().enumerate() {
        let low = (byte & 0x0F) as i8 - 8;
        let high = (byte >> 4) as i8 - 8;

        output[i * 2] = (low as f32) * scale;
        output[i * 2 + 1] = (high as f32) * scale;
    }
}

/// Dequantize Q8_0 block to f32
pub fn dequantize_q8_0(block: &BlockQ8_0, output: &mut [f32; 32]) {
    let scale = f16_to_f32(block.scale);

    for (i, &q) in block.quants.iter().enumerate() {
        output[i] = (q as f32) * scale;
    }
}

/// Convert f16 (as u16) to f32
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = 0i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let new_exp = ((127 - 15 + 1 + e) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (new_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normal
        let new_exp = exp + (127 - 15);
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

/// Convert f32 to f16 (as u16)
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or subnormal -> zero
        sign << 15
    } else if exp == 255 {
        // Inf or NaN
        (sign << 15) | (0x1F << 10) | ((mant >> 13) as u16)
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            // Underflow to zero
            sign << 15
        } else if new_exp >= 31 {
            // Overflow to inf
            (sign << 15) | (0x1F << 10)
        } else {
            (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
        }
    }
}

/// Quantize f32 vector to Q4_0
pub fn quantize_to_q4_0(data: &[f32]) -> Vec<BlockQ4_0> {
    let num_blocks = (data.len() + 31) / 32;
    let mut blocks = Vec::with_capacity(num_blocks);

    for chunk in data.chunks(32) {
        let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 7.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut quants = [0u8; 16];

        for (i, pair) in chunk.chunks(2).enumerate() {
            let v0 = pair.get(0).copied().unwrap_or(0.0);
            let v1 = pair.get(1).copied().unwrap_or(0.0);

            let q0 = ((v0 * inv_scale).round() as i8).clamp(-8, 7) + 8;
            let q1 = ((v1 * inv_scale).round() as i8).clamp(-8, 7) + 8;

            quants[i] = (q0 as u8) | ((q1 as u8) << 4);
        }

        blocks.push(BlockQ4_0 {
            scale: f32_to_f16(scale),
            quants,
        });
    }

    blocks
}
