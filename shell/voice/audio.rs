//! Audio Processing Module
//!
//! Audio buffer management, format conversion, and device abstraction.

use alloc::string::String;
use alloc::vec::Vec;

/// Audio sample format
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    /// 8-bit unsigned
    U8,
    /// 16-bit signed little-endian
    S16Le,
    /// 16-bit signed big-endian
    S16Be,
    /// 32-bit float
    F32,
}

/// Audio format specification
#[derive(Clone, Copy, Debug)]
pub struct AudioFormat {
    /// Sample rate (e.g., 16000, 44100, 48000)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Sample format
    pub format: SampleFormat,
}

impl AudioFormat {
    pub const fn mono_16k() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            format: SampleFormat::S16Le,
        }
    }

    pub const fn stereo_48k() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            format: SampleFormat::S16Le,
        }
    }

    /// Bytes per sample
    pub fn bytes_per_sample(&self) -> usize {
        match self.format {
            SampleFormat::U8 => 1,
            SampleFormat::S16Le | SampleFormat::S16Be => 2,
            SampleFormat::F32 => 4,
        }
    }

    /// Bytes per frame (all channels)
    pub fn bytes_per_frame(&self) -> usize {
        self.bytes_per_sample() * self.channels as usize
    }

    /// Duration of samples in milliseconds
    pub fn duration_ms(&self, sample_count: usize) -> u32 {
        ((sample_count as u64 * 1000) / (self.sample_rate as u64 * self.channels as u64)) as u32
    }

    /// Sample count for duration in milliseconds
    pub fn samples_for_duration(&self, duration_ms: u32) -> usize {
        ((duration_ms as u64 * self.sample_rate as u64 * self.channels as u64) / 1000) as usize
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::mono_16k()
    }
}

/// Audio buffer with format information
#[derive(Clone)]
pub struct AudioBuffer {
    /// Audio samples
    pub samples: Vec<i16>,
    /// Audio format
    pub format: AudioFormat,
}

impl AudioBuffer {
    /// Create empty buffer
    pub fn new(format: AudioFormat) -> Self {
        Self {
            samples: Vec::new(),
            format,
        }
    }

    /// Create buffer with capacity
    pub fn with_capacity(format: AudioFormat, capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            format,
        }
    }

    /// Create from samples
    pub fn from_samples(samples: Vec<i16>, format: AudioFormat) -> Self {
        Self { samples, format }
    }

    /// Duration in milliseconds
    pub fn duration_ms(&self) -> u32 {
        self.format.duration_ms(self.samples.len())
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Is buffer empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Append samples
    pub fn append(&mut self, samples: &[i16]) {
        self.samples.extend_from_slice(samples);
    }

    /// Convert to mono
    pub fn to_mono(&self) -> AudioBuffer {
        if self.format.channels == 1 {
            return self.clone();
        }

        let channels = self.format.channels as usize;
        let mono_samples: Vec<i16> = self
            .samples
            .chunks(channels)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / channels as i32) as i16
            })
            .collect();

        AudioBuffer {
            samples: mono_samples,
            format: AudioFormat {
                sample_rate: self.format.sample_rate,
                channels: 1,
                format: self.format.format,
            },
        }
    }

    /// Resample to target rate (simple linear interpolation)
    pub fn resample(&self, target_rate: u32) -> AudioBuffer {
        if self.format.sample_rate == target_rate {
            return self.clone();
        }

        let ratio = self.format.sample_rate as f32 / target_rate as f32;
        let new_len = (self.samples.len() as f32 / ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_pos = i as f32 * ratio;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f32;

            let sample = if src_idx + 1 < self.samples.len() {
                let s0 = self.samples[src_idx] as f32;
                let s1 = self.samples[src_idx + 1] as f32;
                (s0 + (s1 - s0) * frac) as i16
            } else if src_idx < self.samples.len() {
                self.samples[src_idx]
            } else {
                0
            };

            resampled.push(sample);
        }

        AudioBuffer {
            samples: resampled,
            format: AudioFormat {
                sample_rate: target_rate,
                channels: self.format.channels,
                format: self.format.format,
            },
        }
    }

    /// Normalize audio (peak normalization)
    pub fn normalize(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let max_abs = self
            .samples
            .iter()
            .map(|&s| (s as i32).abs())
            .max()
            .unwrap_or(0);

        if max_abs == 0 {
            return;
        }

        let scale = 32767.0 / max_abs as f32;

        for sample in &mut self.samples {
            *sample = ((*sample as f32) * scale) as i16;
        }
    }

    /// Apply gain
    pub fn apply_gain(&mut self, gain: f32) {
        for sample in &mut self.samples {
            let new_val = (*sample as f32 * gain).clamp(-32768.0, 32767.0);
            *sample = new_val as i16;
        }
    }

    /// Get RMS level
    pub fn rms_level(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = self.samples.iter().map(|&s| (s as f64) * (s as f64)).sum();

        (sum_squares / self.samples.len() as f64).sqrt() as f32
    }

    /// Get peak level
    pub fn peak_level(&self) -> i16 {
        self.samples.iter().map(|&s| s.abs()).max().unwrap_or(0)
    }

    /// Convert to f32 samples (normalized -1.0 to 1.0)
    pub fn to_f32(&self) -> Vec<f32> {
        self.samples.iter().map(|&s| s as f32 / 32768.0).collect()
    }

    /// Create from f32 samples
    pub fn from_f32(samples: &[f32], format: AudioFormat) -> Self {
        let i16_samples: Vec<i16> = samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();

        Self {
            samples: i16_samples,
            format,
        }
    }
}

/// Audio device trait
pub trait AudioDevice: Send + Sync {
    /// Get device name
    fn name(&self) -> &str;

    /// Get supported formats
    fn supported_formats(&self) -> &[AudioFormat];

    /// Get current format
    fn format(&self) -> AudioFormat;

    /// Set format
    fn set_format(&mut self, format: AudioFormat) -> Result<(), AudioError>;

    /// Read samples (for input devices)
    fn read(&self, buffer: &mut [i16]) -> Result<usize, AudioError>;

    /// Write samples (for output devices)
    fn write(&self, buffer: &[i16]) -> Result<usize, AudioError>;

    /// Start streaming
    fn start(&mut self) -> Result<(), AudioError>;

    /// Stop streaming
    fn stop(&mut self) -> Result<(), AudioError>;

    /// Is streaming
    fn is_streaming(&self) -> bool;

    /// Get available samples (for input)
    fn available(&self) -> usize;

    /// Get buffer space (for output)
    fn buffer_space(&self) -> usize;
}

/// Audio error types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioError {
    DeviceNotFound,
    FormatNotSupported,
    DeviceBusy,
    BufferOverrun,
    BufferUnderrun,
    IoError,
    NotStreaming,
    AlreadyStreaming,
}

/// Simple ring buffer for audio
pub struct AudioRingBuffer {
    buffer: Vec<i16>,
    read_pos: usize,
    write_pos: usize,
    capacity: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: alloc::vec![0i16; capacity],
            read_pos: 0,
            write_pos: 0,
            capacity,
        }
    }

    pub fn write(&mut self, samples: &[i16]) -> usize {
        let available = self.capacity - self.len();
        let to_write = samples.len().min(available);

        for &sample in &samples[..to_write] {
            self.buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }

        to_write
    }

    pub fn read(&mut self, buffer: &mut [i16]) -> usize {
        let available = self.len();
        let to_read = buffer.len().min(available);

        for sample in &mut buffer[..to_read] {
            *sample = self.buffer[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }

        to_read
    }

    pub fn len(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    pub fn is_empty(&self) -> bool {
        self.read_pos == self.write_pos
    }

    pub fn is_full(&self) -> bool {
        self.len() == self.capacity - 1
    }

    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Null audio device (for testing)
pub struct NullAudioDevice {
    name: String,
    format: AudioFormat,
    streaming: bool,
}

impl NullAudioDevice {
    pub fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
            format: AudioFormat::default(),
            streaming: false,
        }
    }
}

impl AudioDevice for NullAudioDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn supported_formats(&self) -> &[AudioFormat] {
        &[]
    }

    fn format(&self) -> AudioFormat {
        self.format
    }

    fn set_format(&mut self, format: AudioFormat) -> Result<(), AudioError> {
        self.format = format;
        Ok(())
    }

    fn read(&self, buffer: &mut [i16]) -> Result<usize, AudioError> {
        // Return silence
        for sample in buffer.iter_mut() {
            *sample = 0;
        }
        Ok(buffer.len())
    }

    fn write(&self, buffer: &[i16]) -> Result<usize, AudioError> {
        // Discard audio
        Ok(buffer.len())
    }

    fn start(&mut self) -> Result<(), AudioError> {
        self.streaming = true;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), AudioError> {
        self.streaming = false;
        Ok(())
    }

    fn is_streaming(&self) -> bool {
        self.streaming
    }

    fn available(&self) -> usize {
        0
    }

    fn buffer_space(&self) -> usize {
        usize::MAX
    }
}
