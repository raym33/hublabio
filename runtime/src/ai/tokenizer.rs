//! Tokenizer
//!
//! BPE tokenizer implementation for GGUF models.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// Token ID type
pub type TokenId = u32;

/// Special tokens
pub mod special {
    pub const BOS: u32 = 1; // Beginning of sequence
    pub const EOS: u32 = 2; // End of sequence
    pub const PAD: u32 = 0; // Padding
    pub const UNK: u32 = 3; // Unknown
}

/// BPE Tokenizer
pub struct Tokenizer {
    /// Token to ID mapping
    vocab: BTreeMap<String, TokenId>,
    /// ID to token mapping
    id_to_token: BTreeMap<TokenId, String>,
    /// Merge rules (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Special tokens
    special_tokens: BTreeMap<String, TokenId>,
    /// Vocabulary size
    vocab_size: usize,
}

impl Tokenizer {
    /// Create a new tokenizer from GGUF metadata
    pub fn from_gguf(tokens: &[String], scores: &[f32], merges: &[String]) -> Self {
        let mut vocab = BTreeMap::new();
        let mut id_to_token = BTreeMap::new();

        for (id, token) in tokens.iter().enumerate() {
            vocab.insert(token.clone(), id as TokenId);
            id_to_token.insert(id as TokenId, token.clone());
        }

        let merge_rules: Vec<(String, String)> = merges
            .iter()
            .filter_map(|m| {
                let parts: Vec<&str> = m.split(' ').collect();
                if parts.len() == 2 {
                    Some((String::from(parts[0]), String::from(parts[1])))
                } else {
                    None
                }
            })
            .collect();

        let mut special_tokens = BTreeMap::new();
        special_tokens.insert(String::from("<s>"), special::BOS);
        special_tokens.insert(String::from("</s>"), special::EOS);
        special_tokens.insert(String::from("<pad>"), special::PAD);
        special_tokens.insert(String::from("<unk>"), special::UNK);

        Self {
            vocab_size: tokens.len(),
            vocab,
            id_to_token,
            merges: merge_rules,
            special_tokens,
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = Vec::new();

        // Add BOS token
        tokens.push(special::BOS);

        // Simple character-level tokenization as fallback
        // Real implementation would use BPE merges
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Try to find longest matching token
            let mut best_len = 0;
            let mut best_id = special::UNK;

            for len in 1..=chars.len() - i {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.vocab.get(&substr) {
                    best_len = len;
                    best_id = id;
                }
            }

            if best_len > 0 {
                tokens.push(best_id);
                i += best_len;
            } else {
                // Single character fallback
                let c: String = chars[i].to_string();
                if let Some(&id) = self.vocab.get(&c) {
                    tokens.push(id);
                } else {
                    tokens.push(special::UNK);
                }
                i += 1;
            }
        }

        tokens
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[TokenId]) -> String {
        let mut result = String::new();

        for &id in tokens {
            // Skip special tokens
            if id == special::BOS || id == special::EOS || id == special::PAD {
                continue;
            }

            if let Some(token) = self.id_to_token.get(&id) {
                // Handle space tokens (Llama style: "▁" prefix)
                let text = token.replace('▁', " ");
                result.push_str(&text);
            }
        }

        result.trim_start().to_string()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Check if token is special
    pub fn is_special(&self, id: TokenId) -> bool {
        id <= 3 // BOS, EOS, PAD, UNK
    }

    /// Get token string
    pub fn get_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Get token ID
    pub fn get_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

/// Simple whitespace tokenizer for basic models
pub struct SimpleTokenizer {
    vocab: BTreeMap<String, TokenId>,
    id_to_token: BTreeMap<TokenId, String>,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut vocab = BTreeMap::new();
        let mut id_to_token = BTreeMap::new();

        // Add basic ASCII characters
        for (i, c) in (0u8..128).enumerate() {
            let s = String::from(c as char);
            vocab.insert(s.clone(), i as TokenId + 256);
            id_to_token.insert(i as TokenId + 256, s);
        }

        Self { vocab, id_to_token }
    }

    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = vec![special::BOS];
        for c in text.chars() {
            if let Some(&id) = self.vocab.get(&c.to_string()) {
                tokens.push(id);
            } else {
                tokens.push(special::UNK);
            }
        }
        tokens
    }

    pub fn decode(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .filter_map(|&id| {
                if id <= 3 {
                    None
                } else {
                    self.id_to_token.get(&id).cloned()
                }
            })
            .collect()
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}
