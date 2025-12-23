//! Code Signing and Verification
//!
//! Provides cryptographic verification for binaries and packages.
//! Uses Ed25519 signatures for code signing.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;

/// Public key for signature verification
#[derive(Clone, Debug)]
pub struct PublicKey {
    /// Key ID (fingerprint)
    pub id: String,
    /// Key data (32 bytes for Ed25519)
    pub data: [u8; 32],
    /// Key owner/description
    pub owner: String,
    /// Is revoked
    pub revoked: bool,
    /// Expiration timestamp (0 = no expiration)
    pub expires: u64,
}

impl PublicKey {
    /// Create new public key
    pub fn new(id: &str, data: [u8; 32], owner: &str) -> Self {
        Self {
            id: id.to_string(),
            data,
            owner: owner.to_string(),
            revoked: false,
            expires: 0,
        }
    }

    /// Check if key is valid (not revoked, not expired)
    pub fn is_valid(&self, current_time: u64) -> bool {
        !self.revoked && (self.expires == 0 || current_time < self.expires)
    }

    /// Get key fingerprint
    pub fn fingerprint(&self) -> String {
        // Simple fingerprint: first 8 bytes as hex
        self.data[..8].iter()
            .map(|b| alloc::format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Signature data
#[derive(Clone, Debug)]
pub struct Signature {
    /// Signature bytes (64 bytes for Ed25519)
    pub data: [u8; 64],
    /// Key ID used for signing
    pub key_id: String,
    /// Timestamp of signing
    pub timestamp: u64,
}

impl Signature {
    /// Create from raw bytes
    pub fn from_bytes(data: [u8; 64], key_id: &str, timestamp: u64) -> Self {
        Self {
            data,
            key_id: key_id.to_string(),
            timestamp,
        }
    }
}

/// Signed content wrapper
#[derive(Clone, Debug)]
pub struct SignedContent {
    /// Original content
    pub content: Vec<u8>,
    /// Signature
    pub signature: Signature,
    /// Content type
    pub content_type: ContentType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentType {
    /// Executable binary
    Binary,
    /// Package archive
    Package,
    /// Configuration file
    Config,
    /// AI model
    Model,
    /// Update payload
    Update,
}

/// Signature verification result
#[derive(Clone, Debug)]
pub enum VerifyResult {
    /// Signature valid
    Valid {
        key_id: String,
        timestamp: u64,
    },
    /// Signature invalid
    Invalid,
    /// Key not found
    KeyNotFound(String),
    /// Key revoked
    KeyRevoked(String),
    /// Key expired
    KeyExpired(String),
    /// Content modified
    ContentModified,
}

/// Code signer (for signing content)
pub struct CodeSigner {
    /// Private key (32 bytes for Ed25519)
    private_key: [u8; 32],
    /// Public key
    public_key: PublicKey,
}

impl CodeSigner {
    /// Create signer with existing key
    pub fn new(private_key: [u8; 32], public_key: PublicKey) -> Self {
        Self {
            private_key,
            public_key,
        }
    }

    /// Sign content
    pub fn sign(&self, content: &[u8], timestamp: u64) -> SignedContent {
        // In real implementation, this would use actual Ed25519
        // For now, create placeholder signature
        let mut signature_data = [0u8; 64];

        // Simple "signature" for development (NOT CRYPTOGRAPHICALLY SECURE)
        // Real implementation would use ed25519-dalek or similar
        let hash = simple_hash(content);
        signature_data[..32].copy_from_slice(&hash);
        signature_data[32..].copy_from_slice(&self.private_key);

        SignedContent {
            content: content.to_vec(),
            signature: Signature {
                data: signature_data,
                key_id: self.public_key.id.clone(),
                timestamp,
            },
            content_type: ContentType::Binary,
        }
    }

    /// Get public key
    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }
}

/// Code verifier (for verifying signatures)
pub struct CodeVerifier {
    /// Trusted public keys
    trusted_keys: Vec<PublicKey>,
    /// Allow unsigned content
    allow_unsigned: bool,
}

impl CodeVerifier {
    /// Create new verifier
    pub fn new() -> Self {
        Self {
            trusted_keys: Vec::new(),
            allow_unsigned: true, // Development default
        }
    }

    /// Create strict verifier (requires signatures)
    pub fn strict() -> Self {
        Self {
            trusted_keys: Vec::new(),
            allow_unsigned: false,
        }
    }

    /// Add trusted key
    pub fn add_trusted_key(&mut self, key: PublicKey) {
        self.trusted_keys.push(key);
    }

    /// Remove trusted key
    pub fn remove_trusted_key(&mut self, key_id: &str) {
        self.trusted_keys.retain(|k| k.id != key_id);
    }

    /// Revoke a key
    pub fn revoke_key(&mut self, key_id: &str) {
        for key in &mut self.trusted_keys {
            if key.id == key_id {
                key.revoked = true;
            }
        }
    }

    /// Verify signed content
    pub fn verify(&self, signed: &SignedContent, current_time: u64) -> VerifyResult {
        // Find the key
        let key = match self.trusted_keys.iter().find(|k| k.id == signed.signature.key_id) {
            Some(k) => k,
            None => return VerifyResult::KeyNotFound(signed.signature.key_id.clone()),
        };

        // Check key validity
        if key.revoked {
            return VerifyResult::KeyRevoked(key.id.clone());
        }

        if key.expires != 0 && current_time >= key.expires {
            return VerifyResult::KeyExpired(key.id.clone());
        }

        // Verify signature
        // In real implementation, use actual Ed25519 verification
        let expected_hash = simple_hash(&signed.content);
        if signed.signature.data[..32] != expected_hash {
            return VerifyResult::ContentModified;
        }

        // Simple key check (NOT CRYPTOGRAPHICALLY SECURE)
        // Real implementation would verify Ed25519 signature
        VerifyResult::Valid {
            key_id: key.id.clone(),
            timestamp: signed.signature.timestamp,
        }
    }

    /// Check if content should be allowed to execute
    pub fn should_allow(&self, signed: Option<&SignedContent>, current_time: u64) -> bool {
        match signed {
            Some(content) => {
                matches!(self.verify(content, current_time), VerifyResult::Valid { .. })
            }
            None => self.allow_unsigned,
        }
    }

    /// Get trusted keys
    pub fn trusted_keys(&self) -> &[PublicKey] {
        &self.trusted_keys
    }
}

impl Default for CodeVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple hash function (for development only)
/// Real implementation should use SHA-256 or BLAKE3
fn simple_hash(data: &[u8]) -> [u8; 32] {
    let mut hash = [0u8; 32];

    // Simple FNV-1a inspired hash (NOT CRYPTOGRAPHICALLY SECURE)
    let mut h: u64 = 0xcbf29ce484222325;
    for &byte in data {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }

    // Expand to 32 bytes
    for i in 0..4 {
        let offset = i * 8;
        let val = h.wrapping_add(i as u64);
        hash[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    }

    hash
}

/// Binary signature header
#[derive(Clone, Debug)]
pub struct BinarySignatureHeader {
    /// Magic bytes
    pub magic: [u8; 4],
    /// Version
    pub version: u32,
    /// Signature offset in file
    pub sig_offset: u64,
    /// Signature size
    pub sig_size: u32,
    /// Key ID
    pub key_id: [u8; 32],
}

impl BinarySignatureHeader {
    pub const MAGIC: [u8; 4] = [0x48, 0x4C, 0x53, 0x47]; // "HLSG"

    /// Parse header from bytes
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 52 {
            return None;
        }

        let magic = [data[0], data[1], data[2], data[3]];
        if magic != Self::MAGIC {
            return None;
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let sig_offset = u64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]);
        let sig_size = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);

        let mut key_id = [0u8; 32];
        key_id.copy_from_slice(&data[20..52]);

        Some(Self {
            magic,
            version,
            sig_offset,
            sig_size,
            key_id,
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(52);
        bytes.extend_from_slice(&self.magic);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.sig_offset.to_le_bytes());
        bytes.extend_from_slice(&self.sig_size.to_le_bytes());
        bytes.extend_from_slice(&self.key_id);
        bytes
    }
}

/// Trust store for managing certificates/keys
pub struct TrustStore {
    /// Root keys (highest trust)
    root_keys: Vec<PublicKey>,
    /// Developer keys
    developer_keys: Vec<PublicKey>,
    /// User-added keys
    user_keys: Vec<PublicKey>,
    /// Revocation list
    revoked: Vec<String>,
}

impl TrustStore {
    pub fn new() -> Self {
        Self {
            root_keys: Vec::new(),
            developer_keys: Vec::new(),
            user_keys: Vec::new(),
            revoked: Vec::new(),
        }
    }

    /// Add root key (system level)
    pub fn add_root_key(&mut self, key: PublicKey) {
        self.root_keys.push(key);
    }

    /// Add developer key
    pub fn add_developer_key(&mut self, key: PublicKey) {
        self.developer_keys.push(key);
    }

    /// Add user key
    pub fn add_user_key(&mut self, key: PublicKey) {
        self.user_keys.push(key);
    }

    /// Revoke a key
    pub fn revoke(&mut self, key_id: &str) {
        self.revoked.push(key_id.to_string());
    }

    /// Check if key is revoked
    pub fn is_revoked(&self, key_id: &str) -> bool {
        self.revoked.iter().any(|id| id == key_id)
    }

    /// Find key by ID
    pub fn find_key(&self, key_id: &str) -> Option<(&PublicKey, TrustLevel)> {
        if self.is_revoked(key_id) {
            return None;
        }

        if let Some(key) = self.root_keys.iter().find(|k| k.id == key_id) {
            return Some((key, TrustLevel::Root));
        }

        if let Some(key) = self.developer_keys.iter().find(|k| k.id == key_id) {
            return Some((key, TrustLevel::Developer));
        }

        if let Some(key) = self.user_keys.iter().find(|k| k.id == key_id) {
            return Some((key, TrustLevel::User));
        }

        None
    }

    /// Get all keys
    pub fn all_keys(&self) -> Vec<(&PublicKey, TrustLevel)> {
        let mut keys = Vec::new();

        for key in &self.root_keys {
            if !self.is_revoked(&key.id) {
                keys.push((key, TrustLevel::Root));
            }
        }

        for key in &self.developer_keys {
            if !self.is_revoked(&key.id) {
                keys.push((key, TrustLevel::Developer));
            }
        }

        for key in &self.user_keys {
            if !self.is_revoked(&key.id) {
                keys.push((key, TrustLevel::User));
            }
        }

        keys
    }
}

impl Default for TrustStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Trust level for keys
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// User-added key
    User = 1,
    /// Developer key
    Developer = 2,
    /// Root system key
    Root = 3,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_hash() {
        let hash1 = simple_hash(b"hello");
        let hash2 = simple_hash(b"hello");
        let hash3 = simple_hash(b"world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_public_key() {
        let key = PublicKey::new("test-key", [0u8; 32], "Test Owner");
        assert!(key.is_valid(0));
        assert!(!key.fingerprint().is_empty());
    }

    #[test]
    fn test_code_verifier() {
        let mut verifier = CodeVerifier::new();
        let key = PublicKey::new("test-key", [0u8; 32], "Test");
        verifier.add_trusted_key(key);

        assert!(verifier.should_allow(None, 0)); // Allow unsigned in dev mode
    }

    #[test]
    fn test_trust_store() {
        let mut store = TrustStore::new();
        let key = PublicKey::new("root-1", [0u8; 32], "Root CA");
        store.add_root_key(key);

        let found = store.find_key("root-1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().1, TrustLevel::Root);
    }
}
