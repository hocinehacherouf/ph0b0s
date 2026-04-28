//! Ollama provider builder. Local by default — no API key. `OLLAMA_HOST`
//! env var is read by the env-detection fallback in `provider::build_from_env`,
//! but `build()` itself takes the URL via `base_url`.

use std::sync::Arc;

use crate::AdkLlmAgent;
use crate::error::BuildError;

const DEFAULT_MODEL: &str = "llama3.2:3b";
const DEFAULT_BASE_URL: &str = "http://localhost:11434";

pub fn build(model: Option<&str>, base_url: Option<&str>) -> Result<AdkLlmAgent, BuildError> {
    let model_id = model.unwrap_or(DEFAULT_MODEL).to_owned();
    let url = base_url.unwrap_or(DEFAULT_BASE_URL);
    let cfg = adk_rust::model::ollama::OllamaConfig::with_host(url.to_owned(), model_id.clone());
    let client = adk_rust::model::ollama::OllamaModel::new(cfg)
        .map_err(|e| BuildError::Adk(e.to_string()))?;
    Ok(AdkLlmAgent::new(Arc::new(client), model_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::llm::LlmAgent;

    // No env-locking needed: build() does not read env. The fallback path
    // (which DOES read OLLAMA_HOST) is tested in provider::tests later.

    #[test]
    fn happy_path_uses_default_model_and_url() {
        let agent = build(None, None).expect("should build");
        assert_eq!(agent.model_id(), DEFAULT_MODEL);
    }

    #[test]
    fn override_model_passes_through() {
        let agent = build(Some("qwen2.5-coder:0.5b"), None).expect("should build");
        assert_eq!(agent.model_id(), "qwen2.5-coder:0.5b");
    }

    #[test]
    fn override_base_url_does_not_panic() {
        // We can't assert the base_url is wired without exposing internals,
        // but constructing must succeed.
        let _agent = build(None, Some("http://10.0.0.1:11434")).expect("should build");
    }
}
