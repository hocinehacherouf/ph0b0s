//! Map `adk_core::AdkError` into the seam's `LlmError`.

use ph0b0s_core::error::LlmError;

/// Convert `adk_core::AdkError` into `LlmError`. The mapping is
/// best-effort because adk's errors are stringly-typed at the boundary;
/// we keep the message and tag with `Provider`.
pub(crate) fn map_adk_error(err: adk_rust::AdkError) -> LlmError {
    LlmError::Provider(err.to_string())
}

/// Errors surfaced from the per-provider builders in `crate::provider`.
///
/// `Display` text is what the CLI prints to stderr, so phrase each variant
/// for a human reading the terminal.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    /// A required env var was not set.
    #[error("missing required environment variable: {0}")]
    MissingKey(&'static str),
    /// The TOML config named a provider for which we have no builder.
    #[error("unknown provider: {0:?} (expected one of: anthropic, openai, gemini, ollama, mock)")]
    UnknownProvider(String),
    /// No `[agents.default]` and no provider env var set.
    #[error(
        "no LLM provider configured. Set ANTHROPIC_API_KEY / OPENAI_API_KEY / \
         GOOGLE_API_KEY / OLLAMA_HOST, define [agents.default] in ph0b0s.toml, \
         or set PH0B0S_PROVIDER=mock for a hermetic run."
    )]
    NoProviderConfigured,
    /// The underlying adk-rust client constructor returned an error verbatim.
    #[error("adk-rust client constructor failed: {0}")]
    Adk(String),
    /// `PH0B0S_MOCK_RESPONSES` could not be loaded / parsed.
    #[error("mock-responses file: {0}")]
    Mock(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_adk_error_preserves_message_under_provider_variant() {
        let adk_err = adk_rust::AdkError::model("provider rejected request");
        let mapped = map_adk_error(adk_err);
        match mapped {
            LlmError::Provider(msg) => assert!(
                msg.contains("provider rejected request"),
                "expected adk message in mapped error, got: {msg}"
            ),
            other => panic!("expected Provider variant, got {other:?}"),
        }
    }

    #[test]
    fn build_error_missing_key_displays_var_name() {
        let e = BuildError::MissingKey("ANTHROPIC_API_KEY");
        assert!(e.to_string().contains("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn build_error_unknown_provider_lists_alternatives() {
        let e = BuildError::UnknownProvider("weatherbot".into());
        let s = e.to_string();
        assert!(s.contains("weatherbot"));
        assert!(s.contains("anthropic"));
        assert!(s.contains("ollama"));
    }

    #[test]
    fn build_error_no_provider_mentions_env_vars() {
        let e = BuildError::NoProviderConfigured;
        let s = e.to_string();
        assert!(s.contains("ANTHROPIC_API_KEY"));
        assert!(s.contains("PH0B0S_PROVIDER"));
    }
}
