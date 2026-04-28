//! Hermetic stdio MCP integration test against `fixtures/fake_mcp.py`.
//!
//! Skipped on non-Unix because stdio MCP needs a real subprocess. Skipped if
//! `python3` isn't on PATH (CI installs Python; local devs may not).

#![cfg(unix)]

use std::collections::HashMap;
use std::path::PathBuf;

use ph0b0s_core::tools::{McpServerSpec, McpTransport, ToolHost};
use ph0b0s_llm_adk::AdkToolHost;

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/fake_mcp.py");
    p
}

fn python_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn mount_lists_and_invokes_fake_mcp_ping() {
    if !python_available() {
        eprintln!("skipping: python3 not on PATH");
        return;
    }
    let host = AdkToolHost::new();
    let spec = McpServerSpec {
        name: "fake".into(),
        transport: McpTransport::Stdio,
        command_or_url: vec![
            "python3".into(),
            fixture_path().to_string_lossy().into_owned(),
        ],
        env: HashMap::new(),
    };
    host.mount_mcp(spec).await.expect("mount succeeds");

    let names: Vec<_> = host.list().into_iter().map(|s| s.name).collect();
    assert!(
        names.contains(&"ping".to_owned()),
        "ping not listed: {names:?}"
    );

    let r = host
        .invoke("ping", serde_json::json!({}))
        .await
        .expect("invoke ping");
    // adk_tool::McpToolset typically wraps results — verify the substantive
    // bit is reachable. The exact shape depends on adk's MCP result mapping;
    // accept any shape that contains "pong".
    let s = serde_json::to_string(&r).unwrap();
    assert!(s.contains("pong"), "unexpected result shape: {s}");

    // Clean shutdown — should not panic, should not leave zombies.
    host.shutdown_mcp().await;
}
