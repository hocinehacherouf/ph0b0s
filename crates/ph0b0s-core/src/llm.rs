//! The LLM seam.
//!
//! Detection-pack code talks to a model only through `LlmAgent`. The trait is
//! deliberately small: chat with optional tools (the implementation owns the
//! tool-loop), structured output with schema validation, and stateful
//! sessions for multi-turn work. No streaming in v1 — addable later via a
//! defaulted method without breaking impls.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;

/// Newtype around a string so detection packs can choose any role they
/// need (`reasoner`, `triager`, `summarizer`, `pii-redactor`, ...).
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct AgentRoleKey(pub String);

impl AgentRoleKey {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl From<&str> for AgentRoleKey {
    fn from(s: &str) -> Self {
        AgentRoleKey(s.to_owned())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System { content: String },
    User { content: String },
    Assistant { content: String, tool_calls: Vec<ToolCall> },
    Tool { tool_call_id: String, name: String, content: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: String,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self { content: content.into() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub content: serde_json::Value,
    pub is_error: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub schema: serde_json::Value, // JSON Schema (Draft 2020-12)
    pub source: ToolSource,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolSource {
    Native,
    Mcp { server: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    /// Implementation-defined hints (temperature, max_tokens, ...). Avoid
    /// over-using this — anything with cross-vendor meaning belongs here.
    #[serde(default)]
    pub hints: BTreeMap<String, serde_json::Value>,
}

impl ChatRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::System { content: content.into() });
        self
    }

    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::User { content: content.into() });
        self
    }
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
            hints: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredRequest {
    pub messages: Vec<ChatMessage>,
    /// JSON Schema (Draft 2020-12) for the expected response.
    pub schema: serde_json::Value,
    /// A short human-readable name for the schema; used by some providers.
    pub schema_name: String,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub hints: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub finish_reason: FinishReason,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    #[default]
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_estimate: f64,
    pub cost_source: CostSource,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostSource {
    /// Provider returned token usage natively.
    #[default]
    Native,
    /// We estimated tokens (byte heuristic).
    Estimate,
    /// Unknown.
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionOptions {
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub hints: BTreeMap<String, serde_json::Value>,
}

#[async_trait]
pub trait LlmAgent: Send + Sync {
    /// Single-shot chat. The implementation owns the tool-call loop and
    /// returns the final assistant turn.
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Structured output validated against `req.schema`. Implementations
    /// should prefer the provider's native structured-output mode when
    /// available and fall back to a single repair retry otherwise.
    ///
    /// The raw JSON value is returned and deserialized by the caller — keeps
    /// the trait object-safe (no generic methods).
    async fn structured(&self, req: StructuredRequest) -> Result<serde_json::Value, LlmError>;

    /// Open a multi-turn session. Implementations decide token/budget caps;
    /// callers must treat dropped sessions as fatal.
    async fn session(
        &self,
        opts: SessionOptions,
    ) -> Result<Box<dyn LlmSession>, LlmError>;

    fn model_id(&self) -> &str;
    fn role(&self) -> &AgentRoleKey;
}

#[async_trait]
pub trait LlmSession: Send + Sync {
    async fn send(&mut self, msg: UserMessage) -> Result<ChatResponse, LlmError>;
    fn usage(&self) -> Usage;
}
