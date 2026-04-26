//! Build an `AdkLlmAgent` from layered config + env at startup.
//!
//! The actual selection logic + per-provider construction lives in
//! `ph0b0s_llm_adk::provider`. The CLI just hands the typed config in.

use anyhow::Result;
use ph0b0s_llm_adk::{AdkLlmAgent, provider};

use crate::config::Config;

/// Build the LLM agent for the current process.
pub fn build_from_config(config: &Config) -> Result<AdkLlmAgent> {
    provider::build_from_config(config.default_agent().as_ref(), &config.provider_registry())
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// Saves env vars relevant to provider selection and clears them, returning
    /// a guard that restores them on drop. Use this in tests that exercise the
    /// provider dispatcher to make them hermetic against the host shell.
    struct EnvSaver {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvSaver {
        fn new(keys: &[&'static str]) -> Self {
            let saved: Vec<(&'static str, Option<String>)> =
                keys.iter().map(|k| (*k, std::env::var(k).ok())).collect();
            for k in keys {
                // SAFETY: tests are serialized via the calling test's lock.
                unsafe {
                    std::env::remove_var(k);
                }
            }
            Self { saved }
        }
    }

    impl Drop for EnvSaver {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                if let Some(v) = v {
                    // SAFETY: see new().
                    unsafe {
                        std::env::set_var(k, v);
                    }
                } else {
                    // SAFETY: see new().
                    unsafe {
                        std::env::remove_var(k);
                    }
                }
            }
        }
    }

    /// Crashes on missing env, exercises the full mapper path. The actual
    /// selection logic is tested exhaustively in `ph0b0s_llm_adk::provider`.
    #[test]
    fn dispatcher_propagates_no_provider_error() {
        let _saver = EnvSaver::new(&[
            "PH0B0S_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OLLAMA_HOST",
        ]);
        let cfg = Config::default();
        let err = build_from_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("no LLM provider"));
    }
}
