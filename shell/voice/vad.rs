//! Voice Activity Detection (VAD)
//!
//! Detects speech presence in audio streams.

use alloc::vec::Vec;

/// Voice Activity Detector
pub struct VoiceActivityDetector {
    /// Energy threshold
    threshold: f32,
    /// Frame size in samples
    frame_size: usize,
    /// Smoothing factor
    smoothing: f32,
    /// Current smoothed energy
    smoothed_energy: f32,
    /// Speech start delay (frames)
    speech_start_delay: usize,
    /// Speech end delay (frames)
    speech_end_delay: usize,
    /// Current speech frame count
    speech_frames: usize,
    /// Current silence frame count
    silence_frames: usize,
    /// Is currently in speech
    in_speech: bool,
    /// History for adaptive threshold
    energy_history: Vec<f32>,
    /// History size
    history_size: usize,
}

impl VoiceActivityDetector {
    /// Create new VAD with sensitivity (0.0 - 1.0)
    pub fn new(sensitivity: f32) -> Self {
        // Lower sensitivity = higher threshold = less sensitive
        let threshold = 100.0 * (1.0 - sensitivity.clamp(0.0, 1.0));

        Self {
            threshold,
            frame_size: 160, // 10ms at 16kHz
            smoothing: 0.95,
            smoothed_energy: 0.0,
            speech_start_delay: 3,
            speech_end_delay: 20,
            speech_frames: 0,
            silence_frames: 0,
            in_speech: false,
            energy_history: Vec::with_capacity(100),
            history_size: 100,
        }
    }

    /// Set sensitivity (0.0 - 1.0)
    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        self.threshold = 100.0 * (1.0 - sensitivity.clamp(0.0, 1.0));
    }

    /// Process audio and return true if speech detected
    pub fn is_speech(&self, samples: &[i16]) -> bool {
        let energy = calculate_energy(samples);
        energy > self.threshold
    }

    /// Process frame and return VAD state
    pub fn process(&mut self, samples: &[i16]) -> VadState {
        let energy = calculate_energy(samples);

        // Update smoothed energy
        self.smoothed_energy =
            self.smoothing * self.smoothed_energy + (1.0 - self.smoothing) * energy;

        // Update history for adaptive threshold
        if self.energy_history.len() >= self.history_size {
            self.energy_history.remove(0);
        }
        self.energy_history.push(energy);

        // Calculate adaptive threshold
        let adaptive_threshold = self.calculate_adaptive_threshold();
        let is_active = self.smoothed_energy > adaptive_threshold;

        // State machine
        let previous_in_speech = self.in_speech;

        if is_active {
            self.speech_frames += 1;
            self.silence_frames = 0;

            if !self.in_speech && self.speech_frames >= self.speech_start_delay {
                self.in_speech = true;
            }
        } else {
            self.silence_frames += 1;
            self.speech_frames = 0;

            if self.in_speech && self.silence_frames >= self.speech_end_delay {
                self.in_speech = false;
            }
        }

        // Return state
        if self.in_speech && !previous_in_speech {
            VadState::SpeechStart
        } else if !self.in_speech && previous_in_speech {
            VadState::SpeechEnd
        } else if self.in_speech {
            VadState::Speech
        } else {
            VadState::Silence
        }
    }

    /// Calculate adaptive threshold based on history
    fn calculate_adaptive_threshold(&self) -> f32 {
        if self.energy_history.is_empty() {
            return self.threshold;
        }

        // Use percentile of energy history as noise floor
        let mut sorted = self.energy_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let noise_floor_idx = (sorted.len() as f32 * 0.1) as usize;
        let noise_floor = sorted.get(noise_floor_idx).copied().unwrap_or(0.0);

        // Threshold is noise floor plus margin
        (noise_floor * 3.0).max(self.threshold)
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.smoothed_energy = 0.0;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.in_speech = false;
        self.energy_history.clear();
    }

    /// Is currently in speech
    pub fn is_in_speech(&self) -> bool {
        self.in_speech
    }

    /// Get current energy level
    pub fn current_energy(&self) -> f32 {
        self.smoothed_energy
    }
}

/// VAD state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadState {
    /// No speech detected
    Silence,
    /// Speech just started
    SpeechStart,
    /// Speech ongoing
    Speech,
    /// Speech just ended
    SpeechEnd,
}

/// Calculate RMS energy of samples
fn calculate_energy(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_squares: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();

    (sum_squares / samples.len() as f64).sqrt() as f32
}

/// Calculate zero-crossing rate
pub fn zero_crossing_rate(samples: &[i16]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0) != (w[1] >= 0))
        .count();

    crossings as f32 / (samples.len() - 1) as f32
}

/// Simple noise gate
pub struct NoiseGate {
    threshold: f32,
    attack_time: f32,
    release_time: f32,
    current_gain: f32,
    sample_rate: u32,
}

impl NoiseGate {
    pub fn new(threshold: f32, sample_rate: u32) -> Self {
        Self {
            threshold,
            attack_time: 0.001,  // 1ms attack
            release_time: 0.050, // 50ms release
            current_gain: 0.0,
            sample_rate,
        }
    }

    pub fn process(&mut self, samples: &mut [i16]) {
        let attack_coeff = 1.0 - (-1.0 / (self.attack_time * self.sample_rate as f32)).exp();
        let release_coeff = 1.0 - (-1.0 / (self.release_time * self.sample_rate as f32)).exp();

        for sample in samples.iter_mut() {
            let level = (*sample as f32).abs() / 32768.0;
            let target_gain = if level > self.threshold { 1.0 } else { 0.0 };

            // Smooth gain changes
            let coeff = if target_gain > self.current_gain {
                attack_coeff
            } else {
                release_coeff
            };

            self.current_gain += coeff * (target_gain - self.current_gain);

            *sample = ((*sample as f32) * self.current_gain) as i16;
        }
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
}

/// Voice activity buffer - collects speech segments
pub struct VoiceActivityBuffer {
    vad: VoiceActivityDetector,
    buffer: Vec<i16>,
    pre_speech_buffer: Vec<i16>,
    pre_speech_frames: usize,
    max_duration_samples: usize,
    min_duration_samples: usize,
    is_collecting: bool,
}

impl VoiceActivityBuffer {
    pub fn new(sensitivity: f32, sample_rate: u32) -> Self {
        Self {
            vad: VoiceActivityDetector::new(sensitivity),
            buffer: Vec::new(),
            pre_speech_buffer: Vec::new(),
            pre_speech_frames: 3, // Keep 3 frames before speech
            max_duration_samples: (sample_rate * 30) as usize, // 30 seconds max
            min_duration_samples: (sample_rate / 4) as usize, // 250ms min
            is_collecting: false,
        }
    }

    /// Process audio frame
    pub fn process(&mut self, samples: &[i16]) -> Option<Vec<i16>> {
        let state = self.vad.process(samples);

        // Maintain pre-speech buffer
        if !self.is_collecting {
            self.pre_speech_buffer.extend_from_slice(samples);
            let max_pre = self.pre_speech_frames * samples.len();
            if self.pre_speech_buffer.len() > max_pre {
                let excess = self.pre_speech_buffer.len() - max_pre;
                self.pre_speech_buffer.drain(0..excess);
            }
        }

        match state {
            VadState::SpeechStart => {
                self.is_collecting = true;
                // Include pre-speech buffer
                self.buffer = self.pre_speech_buffer.clone();
                self.buffer.extend_from_slice(samples);
                None
            }
            VadState::Speech => {
                if self.is_collecting {
                    self.buffer.extend_from_slice(samples);

                    // Check max duration
                    if self.buffer.len() >= self.max_duration_samples {
                        self.is_collecting = false;
                        let result = core::mem::take(&mut self.buffer);
                        return Some(result);
                    }
                }
                None
            }
            VadState::SpeechEnd => {
                if self.is_collecting {
                    self.buffer.extend_from_slice(samples);
                    self.is_collecting = false;

                    // Check min duration
                    if self.buffer.len() >= self.min_duration_samples {
                        let result = core::mem::take(&mut self.buffer);
                        return Some(result);
                    } else {
                        self.buffer.clear();
                    }
                }
                None
            }
            VadState::Silence => None,
        }
    }

    /// Force end of utterance
    pub fn flush(&mut self) -> Option<Vec<i16>> {
        if self.is_collecting && self.buffer.len() >= self.min_duration_samples {
            self.is_collecting = false;
            let result = core::mem::take(&mut self.buffer);
            Some(result)
        } else {
            self.is_collecting = false;
            self.buffer.clear();
            None
        }
    }

    /// Reset buffer
    pub fn reset(&mut self) {
        self.vad.reset();
        self.buffer.clear();
        self.pre_speech_buffer.clear();
        self.is_collecting = false;
    }

    /// Is currently collecting speech
    pub fn is_collecting(&self) -> bool {
        self.is_collecting
    }
}
