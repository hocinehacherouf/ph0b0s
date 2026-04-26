//! Mock implementation of `ToolHost` plus a `CannedTool` helper that
//! satisfies `NativeTool` for tests that need a pre-canned tool to register.
//!
//! Dispatch policy in `invoke(name, args)`:
//!   1. Record `(name, args)`.
//!   2. If a canned-response queue exists for `name` and is non-empty, pop and return.
//!   3. Else, if a `NativeTool` is registered for `name`, delegate to it.
//!   4. Else, `ToolError::Unknown(name)`.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ph0b0s_core::error::ToolError;
use ph0b0s_core::llm::{ToolSource, ToolSpec};
use ph0b0s_core::tools::{McpServerSpec, NativeTool, ToolHost};

#[derive(Default)]
struct MockToolState {
    native_tools: Mutex<HashMap<String, Arc<dyn NativeTool>>>,
    canned_responses: Mutex<HashMap<String, VecDeque<Result<serde_json::Value, ToolError>>>>,
    mounted_mcp: Mutex<Vec<McpServerSpec>>,
    invocations: Mutex<Vec<(String, serde_json::Value)>>,
}

/// Mock `ToolHost`. Cloneable; clones share state via `Arc`.
#[derive(Clone, Default)]
pub struct MockToolHost {
    state: Arc<MockToolState>,
}

impl MockToolHost {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue_response(&self, name: &str, value: serde_json::Value) -> &Self {
        self.state
            .canned_responses
            .lock()
            .expect("canned_responses mutex poisoned")
            .entry(name.to_owned())
            .or_default()
            .push_back(Ok(value));
        self
    }

    pub fn enqueue_error(&self, name: &str, err: ToolError) -> &Self {
        self.state
            .canned_responses
            .lock()
            .expect("canned_responses mutex poisoned")
            .entry(name.to_owned())
            .or_default()
            .push_back(Err(err));
        self
    }

    pub fn invocations(&self) -> Vec<(String, serde_json::Value)> {
        self.state
            .invocations
            .lock()
            .expect("invocations mutex poisoned")
            .clone()
    }

    pub fn mounted_mcp(&self) -> Vec<McpServerSpec> {
        self.state
            .mounted_mcp
            .lock()
            .expect("mounted_mcp mutex poisoned")
            .clone()
    }
}

#[async_trait]
impl ToolHost for MockToolHost {
    fn list(&self) -> Vec<ToolSpec> {
        self.state
            .native_tools
            .lock()
            .expect("native_tools mutex poisoned")
            .values()
            .map(|t| t.spec())
            .collect()
    }

    async fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        // 1. Record.
        self.state
            .invocations
            .lock()
            .expect("invocations mutex poisoned")
            .push((name.to_owned(), args.clone()));

        // 2. Try the canned-response queue.
        let canned = {
            let mut map = self
                .state
                .canned_responses
                .lock()
                .expect("canned_responses mutex poisoned");
            map.get_mut(name).and_then(|q| q.pop_front())
        };
        if let Some(result) = canned {
            return result;
        }

        // 3. Delegate to a registered native tool, if any.
        let tool = {
            self.state
                .native_tools
                .lock()
                .expect("native_tools mutex poisoned")
                .get(name)
                .cloned()
        };
        match tool {
            Some(t) => t.call(args).await,
            None => Err(ToolError::Unknown(name.to_owned())),
        }
    }

    fn register_native(&self, tool: Arc<dyn NativeTool>) {
        let spec = tool.spec();
        self.state
            .native_tools
            .lock()
            .expect("native_tools mutex poisoned")
            .insert(spec.name, tool);
    }

    async fn mount_mcp(&self, server: McpServerSpec) -> Result<(), ToolError> {
        self.state
            .mounted_mcp
            .lock()
            .expect("mounted_mcp mutex poisoned")
            .push(server);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CannedTool — a NativeTool helper that returns a fixed response.
// ---------------------------------------------------------------------------

pub struct CannedTool {
    spec: ToolSpec,
    response: serde_json::Value,
}

impl CannedTool {
    pub fn new(name: impl Into<String>, response: serde_json::Value) -> Arc<Self> {
        let name = name.into();
        Arc::new(Self {
            spec: ToolSpec {
                name,
                description: Some("canned test tool".into()),
                schema: serde_json::json!({
                    "type": "object",
                    "additionalProperties": true
                }),
                source: ToolSource::Native,
            },
            response,
        })
    }
}

#[async_trait]
impl NativeTool for CannedTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, _args: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        Ok(self.response.clone())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::tools::McpTransport;

    #[tokio::test]
    async fn register_native_then_invoke_dispatches_to_tool() {
        let host = MockToolHost::new();
        host.register_native(CannedTool::new("echo", serde_json::json!({"echoed": true})));
        let out = host
            .invoke("echo", serde_json::json!({"x": 1}))
            .await
            .unwrap();
        assert_eq!(out, serde_json::json!({"echoed": true}));
    }

    #[tokio::test]
    async fn canned_response_overrides_registered_tool() {
        let host = MockToolHost::new();
        host.register_native(CannedTool::new("echo", serde_json::json!({"from": "tool"})));
        host.enqueue_response("echo", serde_json::json!({"from": "canned"}));

        let first = host.invoke("echo", serde_json::json!({})).await.unwrap();
        assert_eq!(first, serde_json::json!({"from": "canned"}));

        // After the canned queue drains, falls back to the registered tool.
        let second = host.invoke("echo", serde_json::json!({})).await.unwrap();
        assert_eq!(second, serde_json::json!({"from": "tool"}));
    }

    #[tokio::test]
    async fn unknown_tool_returns_unknown_error() {
        let host = MockToolHost::new();
        let err = host
            .invoke("does-not-exist", serde_json::json!({}))
            .await
            .unwrap_err();
        match err {
            ToolError::Unknown(name) => assert_eq!(name, "does-not-exist"),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn enqueue_error_propagates_for_named_tool() {
        let host = MockToolHost::new();
        host.enqueue_error("broken", ToolError::Execution("stub failure".into()));
        let err = host
            .invoke("broken", serde_json::json!({}))
            .await
            .unwrap_err();
        match err {
            ToolError::Execution(msg) => assert_eq!(msg, "stub failure"),
            other => panic!("expected Execution, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mount_mcp_records_spec_returns_ok() {
        let host = MockToolHost::new();
        let spec = McpServerSpec {
            name: "filesystem".into(),
            transport: McpTransport::Stdio,
            command_or_url: vec!["uvx".into(), "mcp-server-filesystem".into()],
            env: Default::default(),
        };
        host.mount_mcp(spec.clone()).await.unwrap();
        assert_eq!(host.mounted_mcp(), vec![spec]);
    }

    #[tokio::test]
    async fn list_returns_specs_of_registered_native_tools_only() {
        let host = MockToolHost::new();
        host.register_native(CannedTool::new("a", serde_json::json!({})));
        host.register_native(CannedTool::new("b", serde_json::json!({})));
        // mounting MCP should not show up in list() in v1
        host.mount_mcp(McpServerSpec {
            name: "fs".into(),
            transport: McpTransport::Stdio,
            command_or_url: vec!["x".into()],
            env: Default::default(),
        })
        .await
        .unwrap();

        let mut names: Vec<String> = host.list().into_iter().map(|s| s.name).collect();
        names.sort();
        assert_eq!(names, vec!["a".to_owned(), "b".to_owned()]);
    }

    #[tokio::test]
    async fn invocations_records_in_order() {
        let host = MockToolHost::new();
        host.enqueue_response("t", serde_json::json!(1))
            .enqueue_response("t", serde_json::json!(2))
            .enqueue_response("u", serde_json::json!(3));

        let _ = host.invoke("t", serde_json::json!({"first": true})).await;
        let _ = host.invoke("u", serde_json::json!({"middle": true})).await;
        let _ = host.invoke("t", serde_json::json!({"last": true})).await;

        let invs = host.invocations();
        assert_eq!(invs.len(), 3);
        assert_eq!(invs[0].0, "t");
        assert_eq!(invs[1].0, "u");
        assert_eq!(invs[2].0, "t");
        assert_eq!(invs[0].1, serde_json::json!({"first": true}));
    }

    #[tokio::test]
    async fn canned_tool_helper_implements_nativetool() {
        let tool = CannedTool::new("greeter", serde_json::json!({"hi": "there"}));
        assert_eq!(tool.spec().name, "greeter");
        let out = tool.call(serde_json::json!({})).await.unwrap();
        assert_eq!(out, serde_json::json!({"hi": "there"}));
    }

    #[tokio::test]
    async fn cloned_tool_host_shares_state() {
        let a = MockToolHost::new();
        let b = a.clone();
        a.enqueue_response("t", serde_json::json!("hello"));
        let out = b.invoke("t", serde_json::json!({})).await.unwrap();
        assert_eq!(out, serde_json::json!("hello"));
        assert_eq!(a.invocations().len(), 1);
        assert_eq!(b.invocations().len(), 1);
    }
}
