//! Build an `AdkLlmAgent` from environment variables at startup.
//!
//! v1 supports two paths:
//! - `PH0B0S_PROVIDER=mock` → `MockLlm`-backed agent. Optionally loads
//!   canned responses from a JSON file at `PH0B0S_MOCK_RESPONSES`.
//!   Used by the integration test in Step 8 and by anyone wanting a
//!   no-network smoke run.
//! - Real providers (Anthropic / OpenAI / Gemini / Ollama / etc.):
//!   **TBD in v1**. Each adk client constructor takes a provider-specific
//!   config struct; wiring those is a Step-7 follow-up. Returns a clear
//!   error directing the user to set `PH0B0S_PROVIDER=mock` for now.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ph0b0s_llm_adk::AdkLlmAgent;

/// Build the LLM agent for the current process. Returns `(agent, model_id)`.
pub fn build_from_env() -> Result<AdkLlmAgent> {
    if let Ok(name) = std::env::var("PH0B0S_PROVIDER") {
        if name == "mock" {
            return build_mock();
        }
        bail!(
            "PH0B0S_PROVIDER={name:?} not supported in v1. \
             Real providers (anthropic/openai/gemini/ollama) are not yet wired; \
             set PH0B0S_PROVIDER=mock for now."
        );
    }
    bail!(
        "no LLM provider configured. v1 only supports the mock provider — \
         set PH0B0S_PROVIDER=mock (and optionally PH0B0S_MOCK_RESPONSES=<file.json>) \
         to scan without a real model. Real provider wiring is a follow-up."
    )
}

/// Build a `MockLlm`-backed agent. Canned responses are JSON values, one per
/// turn, pulled from `PH0B0S_MOCK_RESPONSES` (a JSON array of strings — each
/// string is the model's text reply). If the env var is unset, the mock will
/// surface "provider returned no responses" as an error on first call.
fn build_mock() -> Result<AdkLlmAgent> {
    let mut mock = adk_rust::model::MockLlm::new("ph0b0s-mock");

    if let Ok(path) = std::env::var("PH0B0S_MOCK_RESPONSES") {
        let body = std::fs::read_to_string(&path)
            .with_context(|| format!("reading PH0B0S_MOCK_RESPONSES={path}"))?;
        let parsed: serde_json::Value = serde_json::from_str(&body)
            .with_context(|| format!("parsing PH0B0S_MOCK_RESPONSES={path}"))?;
        let arr = parsed
            .as_array()
            .ok_or_else(|| anyhow::anyhow!(
                "PH0B0S_MOCK_RESPONSES must be a JSON array"
            ))?;
        for item in arr {
            let text = match item {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            mock = mock.with_response(adk_rust::LlmResponse {
                content: Some(adk_rust::Content::new("model").with_text(text)),
                usage_metadata: Some(adk_rust::UsageMetadata {
                    prompt_token_count: 0,
                    candidates_token_count: 0,
                    total_token_count: 0,
                    ..Default::default()
                }),
                finish_reason: Some(adk_rust::FinishReason::Stop),
                ..Default::default()
            });
        }
    }

    let llm: Arc<dyn adk_rust::Llm> = Arc::new(mock);
    Ok(AdkLlmAgent::new(llm, "ph0b0s-mock").with_role("default"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `unsafe`-wrapped env mutation for tests. Each test must clean up
    /// since env state is shared across cargo's parallel runner.
    struct EnvScope;
    impl EnvScope {
        fn set(k: &str, v: &str) {
            // SAFETY: tests serialize their env mutations within each fn.
            unsafe {
                std::env::set_var(k, v);
            }
        }
        fn unset(k: &str) {
            // SAFETY: tests serialize their env mutations within each fn.
            unsafe {
                std::env::remove_var(k);
            }
        }
    }

    #[test]
    fn unset_provider_returns_helpful_error() {
        EnvScope::unset("PH0B0S_PROVIDER");
        EnvScope::unset("PH0B0S_MOCK_RESPONSES");
        let err = build_from_env().unwrap_err();
        let s = err.to_string();
        assert!(s.contains("mock"), "got: {s}");
    }

    #[test]
    fn unsupported_provider_returns_error() {
        EnvScope::set("PH0B0S_PROVIDER", "weatherbot");
        EnvScope::unset("PH0B0S_MOCK_RESPONSES");
        let err = build_from_env().unwrap_err();
        let s = err.to_string();
        assert!(s.contains("not supported"), "got: {s}");
        EnvScope::unset("PH0B0S_PROVIDER");
    }

    #[test]
    fn mock_with_no_responses_builds_successfully() {
        use ph0b0s_core::llm::LlmAgent as _;
        EnvScope::set("PH0B0S_PROVIDER", "mock");
        EnvScope::unset("PH0B0S_MOCK_RESPONSES");
        let agent = build_from_env().expect("should build");
        assert_eq!(agent.model_id(), "ph0b0s-mock");
        EnvScope::unset("PH0B0S_PROVIDER");
    }

    #[tokio::test]
    async fn mock_loads_responses_from_file() {
        use ph0b0s_core::llm::{ChatRequest, LlmAgent};
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("resp.json");
        std::fs::write(&path, r#"["hello","world"]"#).unwrap();

        EnvScope::set("PH0B0S_PROVIDER", "mock");
        EnvScope::set("PH0B0S_MOCK_RESPONSES", path.to_str().unwrap());
        let agent = build_from_env().unwrap();

        // MockLlm yields all responses on each call; "world" is the last.
        let r = agent.chat(ChatRequest::new().user("hi")).await.unwrap();
        assert_eq!(r.content, "world");

        EnvScope::unset("PH0B0S_PROVIDER");
        EnvScope::unset("PH0B0S_MOCK_RESPONSES");
    }
}
