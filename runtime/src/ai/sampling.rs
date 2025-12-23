//! Sampling strategies for text generation

use alloc::vec::Vec;

/// Sampling configuration
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Temperature for softmax
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Random seed (None = random)
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Greedy decoding (always pick highest probability)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            ..Default::default()
        }
    }

    /// Creative sampling for story generation
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Precise sampling for code generation
    pub fn precise() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.8,
            top_k: 20,
            ..Default::default()
        }
    }
}

/// Simple PRNG for sampling
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next random u64
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate random f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

/// Sampler for selecting next token
pub struct Sampler {
    config: SamplingConfig,
    rng: Rng,
    token_counts: Vec<u32>,
}

impl Sampler {
    pub fn new(config: SamplingConfig, vocab_size: usize) -> Self {
        let seed = config.seed.unwrap_or(0x12345678);
        Self {
            config,
            rng: Rng::new(seed),
            token_counts: vec![0; vocab_size],
        }
    }

    /// Sample next token from logits
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let n = logits.len();

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            for (i, count) in self.token_counts.iter().enumerate() {
                if *count > 0 {
                    if logits[i] > 0.0 {
                        logits[i] /= self.config.repetition_penalty;
                    } else {
                        logits[i] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // Apply frequency penalty
        if self.config.frequency_penalty != 0.0 {
            for (i, count) in self.token_counts.iter().enumerate() {
                logits[i] -= self.config.frequency_penalty * (*count as f32);
            }
        }

        // Apply presence penalty
        if self.config.presence_penalty != 0.0 {
            for (i, count) in self.token_counts.iter().enumerate() {
                if *count > 0 {
                    logits[i] -= self.config.presence_penalty;
                }
            }
        }

        // Temperature = 0 means greedy
        if self.config.temperature == 0.0 {
            return self.argmax(logits);
        }

        // Apply temperature
        for logit in logits.iter_mut() {
            *logit /= self.config.temperature;
        }

        // Apply top-k filtering
        let mut indices: Vec<usize> = (0..n).collect();
        if self.config.top_k > 0 && self.config.top_k < n {
            indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            indices.truncate(self.config.top_k);
        }

        // Softmax over filtered indices
        let max_logit = indices.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = indices.iter().map(|&i| (logits[i] - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Apply top-p filtering
        if self.config.top_p < 1.0 {
            // Sort by probability
            let mut sorted_indices: Vec<usize> = (0..indices.len()).collect();
            sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

            let mut cumsum = 0.0;
            let mut cutoff = sorted_indices.len();
            for (i, &idx) in sorted_indices.iter().enumerate() {
                cumsum += probs[idx];
                if cumsum > self.config.top_p {
                    cutoff = i + 1;
                    break;
                }
            }

            // Zero out low probability tokens
            for &idx in &sorted_indices[cutoff..] {
                probs[idx] = 0.0;
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        // Sample from distribution
        let r = self.rng.next_f32();
        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                let token = indices[i] as u32;
                self.token_counts[token as usize] += 1;
                return token;
            }
        }

        // Fallback to first valid token
        let token = indices[0] as u32;
        self.token_counts[token as usize] += 1;
        token
    }

    /// Argmax for greedy decoding
    fn argmax(&self, logits: &[f32]) -> u32 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (i, &logit) in logits.iter().enumerate() {
            if logit > max_val {
                max_val = logit;
                max_idx = i;
            }
        }

        max_idx as u32
    }

    /// Reset token counts
    pub fn reset(&mut self) {
        for count in self.token_counts.iter_mut() {
            *count = 0;
        }
    }

    /// Update configuration
    pub fn set_config(&mut self, config: SamplingConfig) {
        self.config = config;
    }
}
