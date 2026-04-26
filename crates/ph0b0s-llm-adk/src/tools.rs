//! `AdkToolHost` — implementation of the `ToolHost` seam trait.
//!
//! v1 design choice: this is a Rust-side registry, not a bridge to adk's
//! tool system. Detection-pack code calls `ToolHost::register_native` to
//! attach `NativeTool` impls and `ToolHost::invoke(name, args)` to dispatch
//! them. We do NOT (yet) thread these into `LlmRequest.tools` for the
//! model to autonomously call — that requires a tool-call loop in the
//! adapter, which the plan explicitly defers.
//!
//! `mount_mcp` records the spec and logs a warning. Actual MCP connection
//! is a TBD per the slice (e) plan; the seam stays stable.
//!
//! Same dispatch policy as `MockToolHost` (canned-first, then native, then
//! `Unknown`) so detection packs can rely on consistent behaviour across
//! tests and real runs.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ph0b0s_core::error::ToolError;
use ph0b0s_core::llm::ToolSpec;
use ph0b0s_core::tools::{McpServerSpec, NativeTool, ToolHost};

#[derive(Default)]
struct State {
    native_tools: Mutex<HashMap<String, Arc<dyn NativeTool>>>,
    canned: Mutex<HashMap<String, VecDeque<Result<serde_json::Value, ToolError>>>>,
    mounted_mcp: Mutex<Vec<McpServerSpec>>,
    invocations: Mutex<Vec<(String, serde_json::Value)>>,
}

#[derive(Clone, Default)]
pub struct AdkToolHost {
    state: Arc<State>,
}

impl AdkToolHost {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed a canned response for a named tool. The first canned entry wins
    /// over any registered `NativeTool` of the same name. Useful in tests.
    pub fn enqueue_response(&self, name: &str, value: serde_json::Value) -> &Self {
        self.state
            .canned
            .lock()
            .expect("canned mutex poisoned")
            .entry(name.to_owned())
            .or_default()
            .push_back(Ok(value));
        self
    }

    /// Seed a canned error for a named tool.
    pub fn enqueue_error(&self, name: &str, err: ToolError) -> &Self {
        self.state
            .canned
            .lock()
            .expect("canned mutex poisoned")
            .entry(name.to_owned())
            .or_default()
            .push_back(Err(err));
        self
    }

    pub fn invocations(&self) -> Vec<(String, serde_json::Value)> {
        self.state
            .invocations
            .lock()
            .expect("invocations poisoned")
            .clone()
    }

    pub fn mounted_mcp(&self) -> Vec<McpServerSpec> {
        self.state
            .mounted_mcp
            .lock()
            .expect("mounted_mcp poisoned")
            .clone()
    }
}

#[async_trait]
impl ToolHost for AdkToolHost {
    fn list(&self) -> Vec<ToolSpec> {
        self.state
            .native_tools
            .lock()
            .expect("native_tools poisoned")
            .values()
            .map(|t| t.spec())
            .collect()
    }

    #[tracing::instrument(skip(self, args), fields(tool = %name))]
    async fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        // 1. Record the invocation (for tests / observability).
        self.state
            .invocations
            .lock()
            .expect("invocations poisoned")
            .push((name.to_owned(), args.clone()));

        // 2. Try canned-first.
        let canned = {
            let mut map = self.state.canned.lock().expect("canned poisoned");
            map.get_mut(name).and_then(|q| q.pop_front())
        };
        if let Some(r) = canned {
            return r;
        }

        // 3. Delegate to a registered native tool.
        let tool = {
            self.state
                .native_tools
                .lock()
                .expect("native_tools poisoned")
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
            .expect("native_tools poisoned")
            .insert(spec.name, tool);
    }

    #[tracing::instrument(skip(self, server), fields(server = %server.name))]
    async fn mount_mcp(&self, server: McpServerSpec) -> Result<(), ToolError> {
        // v1 limitation: actual MCP connection deferred. Record the spec so
        // observability and tests can confirm the request reached the host.
        tracing::warn!(
            server = %server.name,
            "MCP mount recorded but not yet connected (v1 limitation: \
             adk-rust MCP integration TBD)"
        );
        self.state
            .mounted_mcp
            .lock()
            .expect("mounted_mcp poisoned")
            .push(server);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::llm::{ToolSource, ToolSpec};
    use ph0b0s_core::tools::McpTransport;

    /// Minimal native-tool helper for tests.
    struct EchoTool {
        name: String,
        response: serde_json::Value,
    }

    #[async_trait]
    impl NativeTool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: Some("test echo".into()),
                schema: serde_json::json!({"type":"object"}),
                source: ToolSource::Native,
            }
        }
        async fn call(&self, _args: serde_json::Value) -> Result<serde_json::Value, ToolError> {
            Ok(self.response.clone())
        }
    }

    fn echo(name: &str, response: serde_json::Value) -> Arc<dyn NativeTool> {
        Arc::new(EchoTool {
            name: name.to_owned(),
            response,
        })
    }

    #[tokio::test]
    async fn register_native_then_invoke_dispatches_to_tool() {
        let host = AdkToolHost::new();
        host.register_native(echo("hello", serde_json::json!({"hi":"there"})));
        let r = host.invoke("hello", serde_json::json!({})).await.unwrap();
        assert_eq!(r, serde_json::json!({"hi":"there"}));
    }

    #[tokio::test]
    async fn canned_response_overrides_registered_tool() {
        let host = AdkToolHost::new();
        host.register_native(echo("hello", serde_json::json!({"from":"tool"})));
        host.enqueue_response("hello", serde_json::json!({"from":"canned"}));
        let r = host.invoke("hello", serde_json::json!({})).await.unwrap();
        assert_eq!(r, serde_json::json!({"from":"canned"}));
        // After canned drains, falls back to registered tool.
        let r2 = host.invoke("hello", serde_json::json!({})).await.unwrap();
        assert_eq!(r2, serde_json::json!({"from":"tool"}));
    }

    #[tokio::test]
    async fn unknown_tool_returns_unknown_error() {
        let host = AdkToolHost::new();
        let err = host
            .invoke("nope", serde_json::json!({}))
            .await
            .unwrap_err();
        match err {
            ToolError::Unknown(name) => assert_eq!(name, "nope"),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mount_mcp_records_spec_returns_ok() {
        let host = AdkToolHost::new();
        let spec = McpServerSpec {
            name: "fs".into(),
            transport: McpTransport::Stdio,
            command_or_url: vec!["uvx".into(), "mcp-server-filesystem".into()],
            env: Default::default(),
        };
        host.mount_mcp(spec.clone()).await.unwrap();
        assert_eq!(host.mounted_mcp(), vec![spec]);
    }

    #[tokio::test]
    async fn list_returns_specs_of_registered_native_tools() {
        let host = AdkToolHost::new();
        host.register_native(echo("a", serde_json::json!({})));
        host.register_native(echo("b", serde_json::json!({})));
        let mut names: Vec<_> = host.list().into_iter().map(|s| s.name).collect();
        names.sort();
        assert_eq!(names, vec!["a".to_owned(), "b".to_owned()]);
    }

    #[tokio::test]
    async fn invocations_records_in_order() {
        let host = AdkToolHost::new();
        host.enqueue_response("t", serde_json::json!(1))
            .enqueue_response("t", serde_json::json!(2))
            .enqueue_response("u", serde_json::json!(3));
        let _ = host.invoke("t", serde_json::json!({"k": 1})).await.unwrap();
        let _ = host.invoke("u", serde_json::json!({"k": 2})).await.unwrap();
        let _ = host.invoke("t", serde_json::json!({"k": 3})).await.unwrap();
        let invs = host.invocations();
        assert_eq!(
            invs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["t", "u", "t"]
        );
    }

    #[tokio::test]
    async fn cloned_host_shares_state() {
        let a = AdkToolHost::new();
        let b = a.clone();
        a.enqueue_response("e", serde_json::json!("yes"));
        let r = b.invoke("e", serde_json::json!({})).await.unwrap();
        assert_eq!(r, serde_json::json!("yes"));
        assert_eq!(a.invocations().len(), 1);
        assert_eq!(b.invocations().len(), 1);
    }

    #[tokio::test]
    async fn enqueue_error_propagates() {
        let host = AdkToolHost::new();
        host.enqueue_error("t", ToolError::Execution("boom".into()));
        let err = host.invoke("t", serde_json::json!({})).await.unwrap_err();
        match err {
            ToolError::Execution(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected Execution, got {other:?}"),
        }
    }
}
