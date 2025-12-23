//! AI Chat Interface
//!
//! Conversational AI interface for the shell.

use alloc::string::String;
use alloc::vec::Vec;

/// Chat message
#[derive(Clone, Debug)]
pub struct ChatMessage {
    /// Role (user, assistant, system)
    pub role: MessageRole,
    /// Message content
    pub content: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Message role
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Chat session
pub struct ChatSession {
    /// Message history
    messages: Vec<ChatMessage>,
    /// System prompt
    system_prompt: String,
    /// Model name
    model: String,
    /// Max context length
    max_context: usize,
}

impl ChatSession {
    /// Create new chat session
    pub fn new(model: &str) -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: String::from(
                "You are a helpful AI assistant running on HubLab IO. \
                 You help users with their questions and tasks.",
            ),
            model: String::from(model),
            max_context: 4096,
        }
    }

    /// Set system prompt
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = String::from(prompt);
    }

    /// Add user message
    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: MessageRole::User,
            content: String::from(content),
            timestamp: 0, // TODO: Get actual timestamp
        });
    }

    /// Add assistant message
    pub fn add_assistant_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: MessageRole::Assistant,
            content: String::from(content),
            timestamp: 0,
        });
    }

    /// Get conversation context for model
    pub fn get_context(&self) -> String {
        let mut context = alloc::format!("<|system|>\n{}\n", self.system_prompt);

        for msg in &self.messages {
            let role_tag = match msg.role {
                MessageRole::User => "<|user|>",
                MessageRole::Assistant => "<|assistant|>",
                MessageRole::System => "<|system|>",
            };
            context.push_str(&alloc::format!("{}\n{}\n", role_tag, msg.content));
        }

        context.push_str("<|assistant|>\n");
        context
    }

    /// Clear conversation history
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get message count
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Get model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get all messages
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }
}

impl Default for ChatSession {
    fn default() -> Self {
        Self::new("smollm2-1.7b")
    }
}

/// Chat completion request
#[derive(Clone, Debug)]
pub struct CompletionRequest {
    /// Input text/context
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Stop sequences
    pub stop: Vec<String>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            stop: Vec::new(),
        }
    }
}

/// Chat completion response
#[derive(Clone, Debug)]
pub struct CompletionResponse {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Time taken in ms
    pub time_ms: u64,
    /// Finish reason
    pub finish_reason: FinishReason,
}

/// Finish reason
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached max tokens
    MaxTokens,
    /// Hit stop sequence
    Stop,
    /// Model ended naturally
    EndOfText,
}
