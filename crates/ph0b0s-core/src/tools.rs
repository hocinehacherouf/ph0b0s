//! Tool host abstraction. Detection-pack code registers Rust-native tools
//! via `register_native` and asks the host to mount tools discovered from
//! MCP servers via `mount_mcp`. The implementation behind the seam wires
//! these into the underlying agent runtime (today: adk-rust).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;
use crate::llm::ToolSpec;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    Stdio,
    Sse,
    StreamableHttp,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpServerSpec {
    pub name: String,
    pub transport: McpTransport,
    /// For stdio: argv of the launched server; for HTTP/SSE: URL.
    pub command_or_url: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[async_trait]
pub trait NativeTool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolError>;
}

#[async_trait]
pub trait ToolHost: Send + Sync {
    fn list(&self) -> Vec<ToolSpec>;
    async fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError>;
    fn register_native(&self, tool: Arc<dyn NativeTool>);
    async fn mount_mcp(&self, server: McpServerSpec) -> Result<(), ToolError>;
}
