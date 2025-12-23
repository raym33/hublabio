//! Voice Interface Module
//!
//! Speech-to-text (STT) and text-to-speech (TTS) for HubLab IO.
//! Supports Whisper-style transcription and Piper-style synthesis.

pub mod stt;
pub mod tts;
pub mod audio;
pub mod vad;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::RwLock;
use core::sync::atomic::{AtomicBool, Ordering};

pub use stt::{SpeechToText, TranscriptionResult};
pub use tts::{TextToSpeech, Voice, SpeechConfig};
pub use audio::{AudioBuffer, AudioFormat, AudioDevice};
pub use vad::VoiceActivityDetector;

/// Voice interface state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceState {
    Idle,
    Listening,
    Processing,
    Speaking,
    Error,
}

/// Voice command result
#[derive(Clone, Debug)]
pub struct VoiceCommand {
    /// Transcribed text
    pub text: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Language detected
    pub language: String,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Voice interface configuration
#[derive(Clone)]
pub struct VoiceConfig {
    /// Enable wake word detection
    pub wake_word_enabled: bool,
    /// Wake word phrase
    pub wake_word: String,
    /// Sample rate for audio
    pub sample_rate: u32,
    /// Enable noise reduction
    pub noise_reduction: bool,
    /// VAD sensitivity (0.0 - 1.0)
    pub vad_sensitivity: f32,
    /// Maximum recording duration (ms)
    pub max_record_duration_ms: u32,
    /// Enable continuous listening
    pub continuous_listening: bool,
    /// Default TTS voice
    pub default_voice: String,
    /// TTS speaking rate (0.5 - 2.0)
    pub speaking_rate: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            wake_word_enabled: true,
            wake_word: String::from("hey hublab"),
            sample_rate: 16000,
            noise_reduction: true,
            vad_sensitivity: 0.5,
            max_record_duration_ms: 30000,
            continuous_listening: false,
            default_voice: String::from("default"),
            speaking_rate: 1.0,
        }
    }
}

/// Voice interface manager
pub struct VoiceInterface {
    /// Current state
    state: RwLock<VoiceState>,
    /// Configuration
    config: RwLock<VoiceConfig>,
    /// Speech-to-text engine
    stt: Option<Arc<dyn SpeechToText>>,
    /// Text-to-speech engine
    tts: Option<Arc<dyn TextToSpeech>>,
    /// Voice activity detector
    vad: Option<VoiceActivityDetector>,
    /// Audio input device
    audio_input: Option<Arc<dyn AudioDevice>>,
    /// Audio output device
    audio_output: Option<Arc<dyn AudioDevice>>,
    /// Recording buffer
    recording_buffer: RwLock<Vec<i16>>,
    /// Is recording
    is_recording: AtomicBool,
    /// Is speaking
    is_speaking: AtomicBool,
    /// Command callbacks
    on_command: RwLock<Option<Arc<dyn Fn(VoiceCommand) + Send + Sync>>>,
    /// State change callbacks
    on_state_change: RwLock<Option<Arc<dyn Fn(VoiceState) + Send + Sync>>>,
}

impl VoiceInterface {
    /// Create a new voice interface
    pub fn new(config: VoiceConfig) -> Self {
        Self {
            state: RwLock::new(VoiceState::Idle),
            config: RwLock::new(config),
            stt: None,
            tts: None,
            vad: Some(VoiceActivityDetector::new(0.5)),
            audio_input: None,
            audio_output: None,
            recording_buffer: RwLock::new(Vec::new()),
            is_recording: AtomicBool::new(false),
            is_speaking: AtomicBool::new(false),
            on_command: RwLock::new(None),
            on_state_change: RwLock::new(None),
        }
    }

    /// Get current state
    pub fn state(&self) -> VoiceState {
        *self.state.read()
    }

    /// Set state and notify listeners
    fn set_state(&self, new_state: VoiceState) {
        let mut state = self.state.write();
        if *state != new_state {
            *state = new_state;
            if let Some(ref callback) = *self.on_state_change.read() {
                callback(new_state);
            }
        }
    }

    /// Set STT engine
    pub fn set_stt(&mut self, stt: Arc<dyn SpeechToText>) {
        self.stt = Some(stt);
    }

    /// Set TTS engine
    pub fn set_tts(&mut self, tts: Arc<dyn TextToSpeech>) {
        self.tts = Some(tts);
    }

    /// Set audio input device
    pub fn set_audio_input(&mut self, device: Arc<dyn AudioDevice>) {
        self.audio_input = Some(device);
    }

    /// Set audio output device
    pub fn set_audio_output(&mut self, device: Arc<dyn AudioDevice>) {
        self.audio_output = Some(device);
    }

    /// Set command callback
    pub fn on_command<F: Fn(VoiceCommand) + Send + Sync + 'static>(&self, callback: F) {
        *self.on_command.write() = Some(Arc::new(callback));
    }

    /// Set state change callback
    pub fn on_state_change<F: Fn(VoiceState) + Send + Sync + 'static>(&self, callback: F) {
        *self.on_state_change.write() = Some(Arc::new(callback));
    }

    /// Start listening for voice input
    pub fn start_listening(&self) -> Result<(), VoiceError> {
        if self.state() != VoiceState::Idle {
            return Err(VoiceError::InvalidState);
        }

        self.is_recording.store(true, Ordering::SeqCst);
        self.recording_buffer.write().clear();
        self.set_state(VoiceState::Listening);

        Ok(())
    }

    /// Stop listening
    pub fn stop_listening(&self) -> Result<Vec<i16>, VoiceError> {
        if self.state() != VoiceState::Listening {
            return Err(VoiceError::InvalidState);
        }

        self.is_recording.store(false, Ordering::SeqCst);
        let audio_data = self.recording_buffer.read().clone();
        self.set_state(VoiceState::Processing);

        Ok(audio_data)
    }

    /// Process audio and get transcription
    pub fn transcribe(&self, audio: &[i16]) -> Result<TranscriptionResult, VoiceError> {
        let stt = self.stt.as_ref().ok_or(VoiceError::NoEngine)?;

        self.set_state(VoiceState::Processing);

        let result = stt.transcribe(audio, self.config.read().sample_rate)
            .map_err(|_| VoiceError::TranscriptionFailed)?;

        self.set_state(VoiceState::Idle);

        Ok(result)
    }

    /// Speak text using TTS
    pub fn speak(&self, text: &str) -> Result<(), VoiceError> {
        let tts = self.tts.as_ref().ok_or(VoiceError::NoEngine)?;
        let config = self.config.read();

        self.set_state(VoiceState::Speaking);
        self.is_speaking.store(true, Ordering::SeqCst);

        let speech_config = SpeechConfig {
            voice: config.default_voice.clone(),
            rate: config.speaking_rate,
            pitch: 1.0,
            volume: 1.0,
        };

        let audio = tts.synthesize(text, &speech_config)
            .map_err(|_| VoiceError::SynthesisFailed)?;

        // Play audio through output device
        if let Some(ref output) = self.audio_output {
            output.write(&audio.samples)
                .map_err(|_| VoiceError::AudioError)?;
        }

        self.is_speaking.store(false, Ordering::SeqCst);
        self.set_state(VoiceState::Idle);

        Ok(())
    }

    /// Process incoming audio samples
    pub fn process_audio(&self, samples: &[i16]) {
        if !self.is_recording.load(Ordering::SeqCst) {
            return;
        }

        // Check VAD if available
        if let Some(ref vad) = self.vad {
            if vad.is_speech(samples) {
                self.recording_buffer.write().extend_from_slice(samples);
            }
        } else {
            self.recording_buffer.write().extend_from_slice(samples);
        }
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    /// Check if currently speaking
    pub fn is_speaking(&self) -> bool {
        self.is_speaking.load(Ordering::SeqCst)
    }

    /// Get configuration
    pub fn config(&self) -> VoiceConfig {
        self.config.read().clone()
    }

    /// Update configuration
    pub fn set_config(&self, config: VoiceConfig) {
        *self.config.write() = config;
    }
}

/// Voice interface errors
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceError {
    InvalidState,
    NoEngine,
    TranscriptionFailed,
    SynthesisFailed,
    AudioError,
    WakeWordNotDetected,
    Timeout,
    ModelNotLoaded,
}

/// Global voice interface
static VOICE: RwLock<Option<Arc<VoiceInterface>>> = RwLock::new(None);

/// Initialize voice interface
pub fn init() {
    let config = VoiceConfig::default();
    let interface = VoiceInterface::new(config);
    *VOICE.write() = Some(Arc::new(interface));
    crate::kprintln!("  Voice interface initialized");
}

/// Get global voice interface
pub fn voice() -> Option<Arc<VoiceInterface>> {
    VOICE.read().clone()
}

/// Check if voice is initialized
pub fn is_initialized() -> bool {
    VOICE.read().is_some()
}
