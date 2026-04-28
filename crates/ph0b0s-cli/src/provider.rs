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

    /// Per-binary serialization for env-var mutation in CLI tests. Any test in
    /// this crate that mutates `PH0B0S_PROVIDER` / `*_API_KEY` / `OLLAMA_HOST`
    /// must take this lock first; cargo runs tests in a binary in parallel by
    /// default. Mirrors the adapter-side `env_lock()` pattern.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

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
        let _g = env_lock();
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

    /// Verifies the `EnvSaver::drop` restore-when-set branch: when a var was
    /// already set before the saver captured it, dropping the saver must
    /// re-apply that original value (not just unset).
    #[test]
    fn env_saver_restores_preexisting_value_on_drop() {
        let _g = env_lock();
        const KEY: &str = "PH0B0S_TEST_ENV_SAVER_RESTORE";

        // Pre-set the var BEFORE constructing EnvSaver so drop hits the
        // `Some(v)` arm.
        // SAFETY: tests are serialized via env_lock().
        unsafe {
            std::env::set_var(KEY, "original");
        }

        // Also clear all provider keys so build_from_config errors out
        // predictably while the saver is alive.
        let provider_keys = [
            "PH0B0S_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OLLAMA_HOST",
        ];
        let provider_saver = EnvSaver::new(&provider_keys);

        {
            let _saver = EnvSaver::new(&[KEY]);
            // The saver removed it; while alive, the var is unset.
            assert!(std::env::var(KEY).is_err());

            // Run a no-provider build to exercise the build-error path with
            // a saver alive (mirrors how the real CLI tests use it).
            let cfg = Config::default();
            let err = build_from_config(&cfg).unwrap_err();
            assert!(err.to_string().contains("no LLM provider"));
        }
        // Saver dropped — the original value must be restored.
        assert_eq!(std::env::var(KEY).ok().as_deref(), Some("original"));

        // Cleanup: drop the provider saver explicitly, then unset our key.
        drop(provider_saver);
        // SAFETY: see above.
        unsafe {
            std::env::remove_var(KEY);
        }
    }
}
