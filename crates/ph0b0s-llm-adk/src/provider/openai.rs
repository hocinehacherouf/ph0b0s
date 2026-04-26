//! OpenAI provider builder. Supports OpenAI-compatible endpoints (Azure,
//! OpenRouter, etc.) via the optional `base_url`.

use std::sync::Arc;

use crate::AdkLlmAgent;
use crate::error::BuildError;
use crate::provider::require_env;

const DEFAULT_MODEL: &str = "gpt-5-mini";

/// Build an `AdkLlmAgent` configured for OpenAI.
///
/// `model` overrides the default. `base_url` enables OpenAI-compatible
/// endpoints (Azure, OpenRouter, Ollama, etc.). The API key comes from
/// `OPENAI_API_KEY`.
pub fn build(model: Option<&str>, base_url: Option<&str>) -> Result<AdkLlmAgent, BuildError> {
    let api_key = require_env("OPENAI_API_KEY")?;
    let model_id = model.unwrap_or(DEFAULT_MODEL).to_owned();
    let cfg = match base_url {
        Some(url) => {
            adk_rust::model::openai::OpenAIConfig::compatible(api_key, url, model_id.clone())
        }
        None => adk_rust::model::openai::OpenAIConfig::new(api_key, model_id.clone()),
    };
    let client = adk_rust::model::openai::OpenAIClient::new(cfg)
        .map_err(|e| BuildError::Adk(e.to_string()))?;
    Ok(AdkLlmAgent::new(Arc::new(client), model_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{EnvVarGuard, env_lock, unset_var};
    use ph0b0s_core::llm::LlmAgent;

    #[test]
    fn missing_key_returns_missing_key_error() {
        let _g = env_lock();
        unset_var("OPENAI_API_KEY");
        let err = build(None, None).unwrap_err();
        assert!(matches!(err, BuildError::MissingKey("OPENAI_API_KEY")));
    }

    #[test]
    fn happy_path_uses_default_model() {
        let _g = env_lock();
        let _v = EnvVarGuard::set("OPENAI_API_KEY", "sk-test");
        let agent = build(None, None).expect("should build");
        assert_eq!(agent.model_id(), DEFAULT_MODEL);
    }

    #[test]
    fn base_url_path_uses_compatible_constructor() {
        let _g = env_lock();
        let _v = EnvVarGuard::set("OPENAI_API_KEY", "sk-test");
        let agent =
            build(Some("gpt-4o-mini"), Some("https://openrouter.ai/api/v1")).expect("should build");
        assert_eq!(agent.model_id(), "gpt-4o-mini");
    }
}
