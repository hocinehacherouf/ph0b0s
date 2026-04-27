//! MCP integration: spawn stdio MCP servers via `rmcp`, wrap each discovered
//! tool as a `NativeTool`, and return a handle for lifecycle management.
//!
//! This module is the only place outside `adk_rust::*` calls that imports
//! `rmcp::*`. Higher-level wiring (registering tools into `AdkToolHost`,
//! tracking handles for shutdown) lives in `tools.rs` and `agent.rs`.
//!
//! v1 scope: stdio transport only. SSE / streamable-HTTP transports return
//! `ToolError::McpTransport`. Adding them is a follow-up.

use std::sync::Arc;

use async_trait::async_trait;
use ph0b0s_core::error::ToolError;
use ph0b0s_core::llm::{ToolSource, ToolSpec};
use ph0b0s_core::tools::{McpServerSpec, McpTransport, NativeTool};
use rmcp::{ServiceExt, transport::TokioChildProcess};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// Lifecycle handle returned by [`mount`]. Holding this lets the controller
/// (typically `AdkToolHost`) cancel the MCP server's running task on shutdown.
///
/// Cloning the handle clones the underlying `CancellationToken`, so any clone
/// can trigger cancellation. Cancellation is idempotent.
#[derive(Clone)]
pub struct McpHandle {
    /// Server name as configured in the `McpServerSpec`. Used for log
    /// correlation and tool-source attribution.
    pub server_name: String,
    /// Token whose `cancel()` shuts down the underlying `RunningService`.
    /// We pass this to `serve_with_ct` so we keep a cloneable handle (the
    /// rmcp-owned `RunningServiceCancellationToken` consumes `self` on cancel
    /// and is not `Clone`).
    pub cancel: CancellationToken,
}

/// Outcome of a successful [`mount`]: the per-tool `NativeTool` wrappers and
/// the lifecycle handle.
pub struct MountResult {
    pub tools: Vec<Arc<dyn NativeTool>>,
    pub handle: McpHandle,
}

/// Spawn the configured stdio MCP server, list its tools, and return them
/// wrapped as `NativeTool` instances.
///
/// Errors map to `ToolError::McpTransport` for transport / protocol failures.
/// The caller is responsible for storing the returned [`McpHandle`] and
/// invoking `handle.cancel.cancel()` at shutdown to avoid orphaned children.
pub async fn mount(spec: McpServerSpec) -> Result<MountResult, ToolError> {
    if !matches!(spec.transport, McpTransport::Stdio) {
        return Err(ToolError::McpTransport(format!(
            "non-stdio MCP transports not yet supported: {:?}",
            spec.transport
        )));
    }
    if spec.command_or_url.is_empty() {
        return Err(ToolError::McpTransport(
            "stdio MCP server has no command".into(),
        ));
    }

    let mut cmd = Command::new(&spec.command_or_url[0]);
    cmd.args(&spec.command_or_url[1..]);
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }
    // Reap the child when the Command/transport drops. Without this, cancelling
    // the `RunningService` only stops the rmcp service task — the spawned MCP
    // server process can outlive us. With it, the tokio runtime sends SIGKILL
    // on Drop so transitive shutdown (toolset drop -> RunningService drop ->
    // transport drop -> Command drop) reaches the child.
    cmd.kill_on_drop(true);

    let transport = TokioChildProcess::new(cmd)
        .map_err(|e| ToolError::McpTransport(format!("spawn {}: {e}", spec.name)))?;

    // Use our own cancellation token so the returned `McpHandle` can be
    // cloned and shared. `RunningServiceCancellationToken` consumes `self`
    // on `cancel()` and is not `Clone`, so it's the wrong shape for a
    // handle held by the host registry.
    let cancel = CancellationToken::new();
    let running = ()
        .serve_with_ct(transport, cancel.clone())
        .await
        .map_err(|e| ToolError::McpTransport(format!("connect {}: {e}", spec.name)))?;

    let toolset = adk_rust::tool::McpToolset::new(running).with_name(spec.name.clone());

    // `Toolset::tools` takes `Arc<dyn ReadonlyContext>`. `SimpleToolContext`
    // satisfies this trait with sensible defaults; the MCP toolset only uses
    // it as an opaque parameter (it doesn't read fields off it for `tools()`).
    let ctx: Arc<dyn adk_rust::ReadonlyContext> = Arc::new(adk_rust::tool::SimpleToolContext::new(
        format!("ph0b0s-mcp:{}", spec.name),
    ));
    let inner_tools = adk_rust::Toolset::tools(&toolset, ctx)
        .await
        .map_err(|e| ToolError::McpTransport(format!("list tools {}: {e}", spec.name)))?;

    let server_name = spec.name.clone();
    let tools: Vec<Arc<dyn NativeTool>> = inner_tools
        .into_iter()
        .map(|t| {
            let schema = t
                .parameters_schema()
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
            Arc::new(McpToolWrapper {
                server_name: server_name.clone(),
                inner: t,
                schema,
            }) as Arc<dyn NativeTool>
        })
        .collect();

    Ok(MountResult {
        tools,
        handle: McpHandle {
            server_name,
            cancel,
        },
    })
}

/// `NativeTool` adapter for an `adk_rust::Tool` discovered via MCP.
///
/// Stores the tool's parameters schema eagerly (the trait method clones it
/// on every call) and the originating server name for `ToolSource::Mcp`.
struct McpToolWrapper {
    server_name: String,
    inner: Arc<dyn adk_rust::Tool>,
    schema: serde_json::Value,
}

#[async_trait]
impl NativeTool for McpToolWrapper {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.inner.name().to_owned(),
            description: Some(self.inner.description().to_owned()),
            schema: self.schema.clone(),
            source: ToolSource::Mcp {
                server: self.server_name.clone(),
            },
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        // `Tool::execute` requires a `ToolContext`; MCP tools don't read
        // session/agent state from it, so a lightweight default suffices.
        let ctx: Arc<dyn adk_rust::ToolContext> = Arc::new(adk_rust::tool::SimpleToolContext::new(
            format!("ph0b0s-mcp:{}", self.server_name),
        ));
        self.inner
            .execute(ctx, args)
            .await
            .map_err(|e| ToolError::Execution(format!("{}: {e}", self.server_name)))
    }
}
