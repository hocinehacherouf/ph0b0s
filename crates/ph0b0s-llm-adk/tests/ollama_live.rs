//! Live tests against a local Ollama server. Marked `#[ignore]` so they
//! only run when explicitly requested via `--include-ignored`. The CI
//! `live-ollama` job sets `OLLAMA_HOST=http://localhost:11434` and pulls
//! a tiny model (`qwen2.5-coder:0.5b`) before invoking these.
//!
//! Local dev: `ollama serve` in one terminal, `ollama pull qwen2.5-coder:0.5b`,
//! then `cargo test -p ph0b0s-llm-adk --test ollama_live -- --include-ignored`.

use ph0b0s_core::llm::{ChatMessage, ChatRequest, LlmAgent, StructuredRequest, UserMessage};
use ph0b0s_llm_adk::provider::ollama;

const TEST_MODEL_ENV: &str = "PH0B0S_LIVE_OLLAMA_MODEL";

fn model() -> String {
    std::env::var(TEST_MODEL_ENV).unwrap_or_else(|_| "qwen2.5-coder:0.5b".into())
}

#[tokio::test]
#[ignore = "requires running local Ollama server"]
async fn chat_returns_non_empty_response() {
    let m = model();
    let agent = ollama::build(Some(&m), None).expect("build ollama");
    let req = ChatRequest::new()
        .system("be terse")
        .user("say the single word 'pong' and nothing else");
    let resp = agent.chat(req).await.expect("chat ok");
    assert!(!resp.content.is_empty(), "empty response");
    assert!(resp.usage.tokens_in > 0, "expected token accounting");
}

#[tokio::test]
#[ignore = "requires running local Ollama server"]
async fn structured_emits_parseable_json() {
    let m = model();
    let agent = ollama::build(Some(&m), None).expect("build ollama");
    let req = StructuredRequest {
        messages: vec![ChatMessage::User {
            content: "respond with {\"ok\": true} as JSON, nothing else".into(),
        }],
        schema: serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"]
        }),
        schema_name: "Smoke".into(),
        tools: Vec::new(),
        hints: Default::default(),
    };
    let v = agent.structured(req).await.expect("structured ok");
    assert!(v["ok"].is_boolean(), "expected ok:bool, got {v}");
}

#[tokio::test]
#[ignore = "requires running local Ollama server"]
async fn session_multi_turn_accumulates_usage() {
    let m = model();
    let agent = ollama::build(Some(&m), None).expect("build ollama");
    let mut sess = agent.session(Default::default()).await.expect("session ok");
    let r1 = sess.send(UserMessage::new("hello")).await.unwrap();
    assert!(!r1.content.is_empty());
    let usage_after_1 = sess.usage();
    let r2 = sess.send(UserMessage::new("how are you?")).await.unwrap();
    assert!(!r2.content.is_empty());
    let usage_after_2 = sess.usage();
    assert!(
        usage_after_2.tokens_in >= usage_after_1.tokens_in,
        "tokens_in should be monotonic"
    );
}
