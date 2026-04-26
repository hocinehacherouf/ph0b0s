//! Anthropic provider builder. Pinned to adk-rust's `AnthropicClient`.

use std::sync::Arc;

use crate::AdkLlmAgent;
use crate::error::BuildError;
use crate::provider::require_env;

const DEFAULT_MODEL: &str = "claude-sonnet-4-6";

/// Build an `AdkLlmAgent` configured for Anthropic.
///
/// `model` overrides the default. The API key comes from `ANTHROPIC_API_KEY`.
pub fn build(model: Option<&str>) -> Result<AdkLlmAgent, BuildError> {
    let api_key = require_env("ANTHROPIC_API_KEY")?;
    let model_id = model.unwrap_or(DEFAULT_MODEL).to_owned();
    let cfg = adk_rust::model::anthropic::AnthropicConfig::new(&api_key, model_id.clone());
    let client = adk_rust::model::anthropic::AnthropicClient::new(cfg)
        .map_err(|e| BuildError::Adk(e.to_string()))?;
    Ok(AdkLlmAgent::new(Arc::new(client), model_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{env_lock, set_var, unset_var};
    use ph0b0s_core::llm::LlmAgent;

    #[test]
    fn missing_key_returns_missing_key_error() {
        let _g = env_lock();
        unset_var("ANTHROPIC_API_KEY");
        let err = build(None).unwrap_err();
        match err {
            BuildError::MissingKey(k) => assert_eq!(k, "ANTHROPIC_API_KEY"),
            other => panic!("expected MissingKey, got {other:?}"),
        }
    }

    #[test]
    fn happy_path_uses_default_model() {
        let _g = env_lock();
        set_var("ANTHROPIC_API_KEY", "sk-ant-test-key");
        let agent = build(None).expect("should build");
        assert_eq!(agent.model_id(), DEFAULT_MODEL);
        unset_var("ANTHROPIC_API_KEY");
    }

    #[test]
    fn override_model_passes_through() {
        let _g = env_lock();
        set_var("ANTHROPIC_API_KEY", "sk-ant-test-key");
        let agent = build(Some("claude-opus-4-7")).expect("should build");
        assert_eq!(agent.model_id(), "claude-opus-4-7");
        unset_var("ANTHROPIC_API_KEY");
    }
}
