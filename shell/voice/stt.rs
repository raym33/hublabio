//! Speech-to-Text (STT) Module
//!
//! Whisper-style speech recognition for HubLab IO.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use super::audio::AudioBuffer;

/// Transcription result
#[derive(Clone, Debug)]
pub struct TranscriptionResult {
    /// Transcribed text
    pub text: String,
    /// Detected language
    pub language: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Word-level timestamps (if available)
    pub words: Vec<WordTiming>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Word timing information
#[derive(Clone, Debug)]
pub struct WordTiming {
    /// Word text
    pub word: String,
    /// Start time in milliseconds
    pub start_ms: u32,
    /// End time in milliseconds
    pub end_ms: u32,
    /// Confidence for this word
    pub confidence: f32,
}

/// STT configuration
#[derive(Clone)]
pub struct SttConfig {
    /// Language hint (ISO 639-1 code)
    pub language: Option<String>,
    /// Enable word timestamps
    pub word_timestamps: bool,
    /// Enable translation to English
    pub translate: bool,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Enable beam search
    pub beam_size: u32,
    /// Suppress non-speech tokens
    pub suppress_non_speech: bool,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            language: None,
            word_timestamps: false,
            translate: false,
            temperature: 0.0,
            beam_size: 1,
            suppress_non_speech: true,
        }
    }
}

/// Speech-to-Text trait
pub trait SpeechToText: Send + Sync {
    /// Get model name
    fn model_name(&self) -> &str;

    /// Get supported languages
    fn supported_languages(&self) -> &[&str];

    /// Transcribe audio
    fn transcribe(&self, audio: &[i16], sample_rate: u32) -> Result<TranscriptionResult, SttError>;

    /// Transcribe with configuration
    fn transcribe_with_config(
        &self,
        audio: &[i16],
        sample_rate: u32,
        config: &SttConfig,
    ) -> Result<TranscriptionResult, SttError>;

    /// Detect language
    fn detect_language(&self, audio: &[i16], sample_rate: u32) -> Result<String, SttError>;

    /// Is model loaded
    fn is_loaded(&self) -> bool;
}

/// STT error types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SttError {
    ModelNotLoaded,
    InvalidAudio,
    UnsupportedFormat,
    UnsupportedLanguage,
    TranscriptionFailed,
    OutOfMemory,
    Timeout,
}

/// Whisper model sizes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperModel {
    Tiny,      // ~39M params, ~75MB
    Base,      // ~74M params, ~142MB
    Small,     // ~244M params, ~466MB
    Medium,    // ~769M params, ~1.5GB
    Large,     // ~1.5B params, ~2.9GB
}

impl WhisperModel {
    pub fn memory_requirement(&self) -> usize {
        match self {
            Self::Tiny => 75 * 1024 * 1024,
            Self::Base => 142 * 1024 * 1024,
            Self::Small => 466 * 1024 * 1024,
            Self::Medium => 1500 * 1024 * 1024,
            Self::Large => 2900 * 1024 * 1024,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Base => "base",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }
}

/// Whisper-style STT implementation (placeholder)
pub struct WhisperStt {
    model_size: WhisperModel,
    loaded: bool,
    model_data: Option<Vec<u8>>,
}

impl WhisperStt {
    /// Create new Whisper STT
    pub fn new(model_size: WhisperModel) -> Self {
        Self {
            model_size,
            loaded: false,
            model_data: None,
        }
    }

    /// Load model from memory
    pub fn load(&mut self, model_data: Vec<u8>) -> Result<(), SttError> {
        // In real implementation, parse and validate model
        self.model_data = Some(model_data);
        self.loaded = true;
        Ok(())
    }

    /// Unload model
    pub fn unload(&mut self) {
        self.model_data = None;
        self.loaded = false;
    }

    /// Get model size
    pub fn model_size(&self) -> WhisperModel {
        self.model_size
    }

    /// Preprocess audio for Whisper
    fn preprocess_audio(&self, audio: &[i16], sample_rate: u32) -> Vec<f32> {
        // Convert to f32
        let mut samples: Vec<f32> = audio.iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        // Resample to 16kHz if needed
        if sample_rate != 16000 {
            samples = resample(&samples, sample_rate, 16000);
        }

        // Pad or truncate to 30 seconds (480000 samples at 16kHz)
        let target_len = 16000 * 30;
        if samples.len() < target_len {
            samples.resize(target_len, 0.0);
        } else if samples.len() > target_len {
            samples.truncate(target_len);
        }

        // Compute log-mel spectrogram (placeholder)
        // In real implementation, this would compute 80-channel mel spectrogram
        samples
    }
}

impl SpeechToText for WhisperStt {
    fn model_name(&self) -> &str {
        self.model_size.name()
    }

    fn supported_languages(&self) -> &[&str] {
        &[
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
        ]
    }

    fn transcribe(&self, audio: &[i16], sample_rate: u32) -> Result<TranscriptionResult, SttError> {
        self.transcribe_with_config(audio, sample_rate, &SttConfig::default())
    }

    fn transcribe_with_config(
        &self,
        audio: &[i16],
        sample_rate: u32,
        config: &SttConfig,
    ) -> Result<TranscriptionResult, SttError> {
        if !self.loaded {
            return Err(SttError::ModelNotLoaded);
        }

        if audio.is_empty() {
            return Err(SttError::InvalidAudio);
        }

        // Preprocess audio
        let _processed = self.preprocess_audio(audio, sample_rate);

        // In real implementation, run inference here
        // For now, return placeholder result

        Ok(TranscriptionResult {
            text: String::from("[Transcription would appear here]"),
            language: config.language.clone().unwrap_or_else(|| String::from("en")),
            confidence: 0.95,
            words: Vec::new(),
            processing_time_ms: 100,
        })
    }

    fn detect_language(&self, audio: &[i16], sample_rate: u32) -> Result<String, SttError> {
        if !self.loaded {
            return Err(SttError::ModelNotLoaded);
        }

        // Preprocess and run language detection
        let _processed = self.preprocess_audio(audio, sample_rate);

        // Placeholder - return English
        Ok(String::from("en"))
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }
}

/// Simple resampling (linear interpolation)
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = from_rate as f32 / to_rate as f32;
    let new_len = (samples.len() as f32 / ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_pos = i as f32 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f32;

        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx] + (samples[src_idx + 1] - samples[src_idx]) * frac
        } else if src_idx < samples.len() {
            samples[src_idx]
        } else {
            0.0
        };

        resampled.push(sample);
    }

    resampled
}

/// Streaming STT for real-time transcription
pub struct StreamingStt {
    stt: Arc<dyn SpeechToText>,
    buffer: Vec<i16>,
    buffer_duration_ms: u32,
    chunk_duration_ms: u32,
    sample_rate: u32,
}

impl StreamingStt {
    pub fn new(stt: Arc<dyn SpeechToText>, sample_rate: u32) -> Self {
        Self {
            stt,
            buffer: Vec::new(),
            buffer_duration_ms: 0,
            chunk_duration_ms: 5000,  // 5 second chunks
            sample_rate,
        }
    }

    /// Add audio samples
    pub fn add_audio(&mut self, samples: &[i16]) {
        self.buffer.extend_from_slice(samples);
        self.buffer_duration_ms = (self.buffer.len() as u64 * 1000 / self.sample_rate as u64) as u32;
    }

    /// Process if enough audio
    pub fn process(&mut self) -> Option<TranscriptionResult> {
        if self.buffer_duration_ms >= self.chunk_duration_ms {
            let result = self.stt.transcribe(&self.buffer, self.sample_rate).ok();

            // Keep last second for context overlap
            let keep_samples = self.sample_rate as usize;
            if self.buffer.len() > keep_samples {
                self.buffer = self.buffer[self.buffer.len() - keep_samples..].to_vec();
            }
            self.buffer_duration_ms = (self.buffer.len() as u64 * 1000 / self.sample_rate as u64) as u32;

            result
        } else {
            None
        }
    }

    /// Flush remaining audio
    pub fn flush(&mut self) -> Option<TranscriptionResult> {
        if self.buffer.is_empty() {
            return None;
        }

        let result = self.stt.transcribe(&self.buffer, self.sample_rate).ok();
        self.buffer.clear();
        self.buffer_duration_ms = 0;
        result
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.buffer_duration_ms = 0;
    }
}
