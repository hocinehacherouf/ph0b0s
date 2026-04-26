//! Gemini provider builder.

use std::sync::Arc;

use crate::AdkLlmAgent;
use crate::error::BuildError;
use crate::provider::require_env;

const DEFAULT_MODEL: &str = "gemini-2.5-flash";

pub fn build(model: Option<&str>) -> Result<AdkLlmAgent, BuildError> {
    let api_key = require_env("GOOGLE_API_KEY")?;
    let model_id = model.unwrap_or(DEFAULT_MODEL).to_owned();
    let client = adk_rust::model::GeminiModel::new(api_key, model_id.clone())
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
        unset_var("GOOGLE_API_KEY");
        let err = build(None).unwrap_err();
        assert!(matches!(err, BuildError::MissingKey("GOOGLE_API_KEY")));
    }

    #[test]
    fn happy_path_uses_default_model() {
        let _g = env_lock();
        let _v = EnvVarGuard::set("GOOGLE_API_KEY", "AIza-test");
        let agent = build(None).expect("should build");
        assert_eq!(agent.model_id(), DEFAULT_MODEL);
    }

    #[test]
    fn override_model_passes_through() {
        let _g = env_lock();
        let _v = EnvVarGuard::set("GOOGLE_API_KEY", "AIza-test");
        let agent = build(Some("gemini-2.5-pro")).expect("should build");
        assert_eq!(agent.model_id(), "gemini-2.5-pro");
    }
}
