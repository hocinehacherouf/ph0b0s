//! Mock provider builder. Reads canned responses from `PH0B0S_MOCK_RESPONSES`
//! (a JSON array of strings — each entry is the model's text reply).

use std::sync::Arc;

use crate::AdkLlmAgent;
use crate::error::BuildError;

const MODEL_ID: &str = "ph0b0s-mock";

pub fn build() -> Result<AdkLlmAgent, BuildError> {
    let mut mock = adk_rust::model::MockLlm::new(MODEL_ID);

    if let Ok(path) = std::env::var("PH0B0S_MOCK_RESPONSES") {
        let body = std::fs::read_to_string(&path)
            .map_err(|e| BuildError::Mock(format!("read {path}: {e}")))?;
        let parsed: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| BuildError::Mock(format!("parse {path}: {e}")))?;
        let arr = parsed
            .as_array()
            .ok_or_else(|| BuildError::Mock("PH0B0S_MOCK_RESPONSES must be a JSON array".into()))?;
        for item in arr {
            let text = match item {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            mock = mock.with_response(adk_rust::LlmResponse {
                content: Some(adk_rust::Content::new("model").with_text(text)),
                usage_metadata: Some(adk_rust::UsageMetadata {
                    ..Default::default()
                }),
                finish_reason: Some(adk_rust::FinishReason::Stop),
                ..Default::default()
            });
        }
    }

    let llm: Arc<dyn adk_rust::Llm> = Arc::new(mock);
    Ok(AdkLlmAgent::new(llm, MODEL_ID).with_role("default"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{EnvVarGuard, env_lock, unset_var};
    use ph0b0s_core::llm::{ChatRequest, LlmAgent};

    #[test]
    fn no_responses_file_builds_successfully() {
        let _g = env_lock();
        unset_var("PH0B0S_MOCK_RESPONSES");
        let agent = build().expect("should build");
        assert_eq!(agent.model_id(), MODEL_ID);
    }

    #[tokio::test]
    async fn responses_file_loaded_and_yielded() {
        let agent = {
            let _g = env_lock();
            let td = tempfile::tempdir().unwrap();
            let path = td.path().join("resp.json");
            std::fs::write(&path, r#"["hello","world"]"#).unwrap();
            let _v = EnvVarGuard::set("PH0B0S_MOCK_RESPONSES", path.to_str().unwrap());
            build().unwrap()
            // _g and _v dropped here, before the await below
        };
        // MockLlm yields all responses on each call; "world" is the last.
        let r = agent.chat(ChatRequest::new().user("hi")).await.unwrap();
        assert_eq!(r.content, "world");
    }

    #[test]
    fn malformed_responses_file_returns_mock_error() {
        let _g = env_lock();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("resp.json");
        std::fs::write(&path, "not json").unwrap();
        let _v = EnvVarGuard::set("PH0B0S_MOCK_RESPONSES", path.to_str().unwrap());
        let err = build().unwrap_err();
        assert!(matches!(err, BuildError::Mock(_)));
    }

    #[test]
    fn missing_responses_file_returns_mock_error() {
        let _g = env_lock();
        let _v = EnvVarGuard::set("PH0B0S_MOCK_RESPONSES", "/nonexistent/path/resp.json");
        let err = build().unwrap_err();
        assert!(matches!(err, BuildError::Mock(_)));
    }

    #[test]
    fn non_array_json_returns_mock_error() {
        let _g = env_lock();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("resp.json");
        std::fs::write(&path, r#"{"not":"array"}"#).unwrap();
        let _v = EnvVarGuard::set("PH0B0S_MOCK_RESPONSES", path.to_str().unwrap());
        let err = build().unwrap_err();
        assert!(matches!(err, BuildError::Mock(_)));
    }
}
