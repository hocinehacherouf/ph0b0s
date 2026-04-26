//! Per-provider builders. Each module owns one provider; this `mod.rs`
//! exposes `build_from_env` / `build_from_config` (filled in Task 9) and
//! shared helpers.

use crate::error::BuildError;

pub mod anthropic;
pub mod gemini;
pub mod mock;
pub mod ollama;
pub mod openai;

/// Read `key` from env, returning `BuildError::MissingKey` if absent or empty.
pub(crate) fn require_env(key: &'static str) -> Result<String, BuildError> {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => Ok(v),
        _ => Err(BuildError::MissingKey(key)),
    }
}

use crate::AdkLlmAgent;
use crate::config::{AgentConfig, ProviderRegistry};

/// Top-level entry: choose a provider and build the agent.
///
/// Precedence (highest first):
/// 1. `PH0B0S_PROVIDER` env override (always wins; for ad-hoc CLI runs).
/// 2. Explicit `[agents.default]` in TOML.
/// 3. Env-key precedence fallback (`ANTHROPIC_API_KEY` → ... → `OLLAMA_HOST`).
///
/// Errors out with `BuildError::NoProviderConfigured` if none of the above
/// resolve.
pub fn build_from_config(
    agent: Option<&AgentConfig>,
    providers: &ProviderRegistry,
) -> Result<AdkLlmAgent, BuildError> {
    if let Ok(name) = std::env::var("PH0B0S_PROVIDER") {
        let model = agent.and_then(|a| a.model.as_deref());
        return build_named(&name, providers, model);
    }
    if let Some(a) = agent {
        if !a.provider.is_empty() {
            return build_named(&a.provider, providers, a.model.as_deref());
        }
    }
    build_from_env(providers)
}

/// Env-detection fallback. Iterates canonical env vars in fixed precedence
/// and builds the first that matches.
pub fn build_from_env(providers: &ProviderRegistry) -> Result<AdkLlmAgent, BuildError> {
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return anthropic::build(providers.model_for("anthropic"));
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return openai::build(
            providers.model_for("openai"),
            providers.base_url_for("openai"),
        );
    }
    if std::env::var("GOOGLE_API_KEY").is_ok() {
        return gemini::build(providers.model_for("gemini"));
    }
    if std::env::var("OLLAMA_HOST").is_ok() {
        return ollama::build(
            providers.model_for("ollama"),
            providers.base_url_for("ollama"),
        );
    }
    Err(BuildError::NoProviderConfigured)
}

fn build_named(
    name: &str,
    providers: &ProviderRegistry,
    model_override: Option<&str>,
) -> Result<AdkLlmAgent, BuildError> {
    // Per-call model override beats the registry's default.
    let model = model_override.or(providers.model_for(name));
    match name {
        "anthropic" => anthropic::build(model),
        "openai" => openai::build(model, providers.base_url_for("openai")),
        "gemini" => gemini::build(model),
        "ollama" => ollama::build(model, providers.base_url_for("ollama")),
        "mock" => mock::build(),
        other => Err(BuildError::UnknownProvider(other.to_owned())),
    }
}

// Re-export the env-mutation helpers for sibling test modules in the
// `provider` tree. They're private to the crate.
#[cfg(test)]
pub(crate) use helper_tests::{EnvVarGuard, env_lock, unset_var};

#[cfg(test)]
mod helper_tests {
    use super::*;

    /// Per-test serialization for env-var mutation. All env-touching tests
    /// in the adapter must take this lock first.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn set_var(k: &str, v: &str) {
        // SAFETY: callers hold env_lock().
        unsafe { std::env::set_var(k, v) }
    }

    pub(crate) fn unset_var(k: &str) {
        // SAFETY: callers hold env_lock().
        unsafe { std::env::remove_var(k) }
    }

    /// RAII guard: sets `key=value` on construction; unsets on drop.
    /// Always pair with `env_lock()`.
    pub(crate) struct EnvVarGuard {
        key: &'static str,
    }

    impl EnvVarGuard {
        pub(crate) fn set(key: &'static str, value: &str) -> Self {
            set_var(key, value);
            Self { key }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unset_var(self.key);
        }
    }

    #[test]
    fn require_env_returns_value_when_set() {
        let _g = env_lock();
        let _v = EnvVarGuard::set("PH0B0S_TEST_REQUIRE_ENV_X", "value");
        assert_eq!(require_env("PH0B0S_TEST_REQUIRE_ENV_X").unwrap(), "value");
    }

    #[test]
    fn require_env_rejects_empty_string() {
        let _g = env_lock();
        let _v = EnvVarGuard::set("PH0B0S_TEST_REQUIRE_ENV_Y", "");
        let err = require_env("PH0B0S_TEST_REQUIRE_ENV_Y").unwrap_err();
        match err {
            BuildError::MissingKey(k) => assert_eq!(k, "PH0B0S_TEST_REQUIRE_ENV_Y"),
            other => panic!("expected MissingKey, got {other:?}"),
        }
    }

    #[test]
    fn require_env_returns_missing_key_when_unset() {
        let _g = env_lock();
        unset_var("PH0B0S_TEST_NEVER_SET");
        let err = require_env("PH0B0S_TEST_NEVER_SET").unwrap_err();
        assert!(matches!(err, BuildError::MissingKey(_)));
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::config::{AgentConfig, ProviderConfig, ProviderRegistry};
    use crate::provider::{EnvVarGuard, env_lock, unset_var};
    use ph0b0s_core::llm::LlmAgent;

    fn clear_provider_env() {
        for k in [
            "PH0B0S_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OLLAMA_HOST",
        ] {
            unset_var(k);
        }
    }

    #[test]
    fn ph0b0s_provider_env_wins_over_agent_config() {
        let _g = env_lock();
        clear_provider_env();
        let _v = EnvVarGuard::set("PH0B0S_PROVIDER", "mock");
        // [agents.default] says anthropic, but env override wins.
        let agent_cfg = AgentConfig {
            provider: "anthropic".into(),
            model: None,
        };
        let registry = ProviderRegistry::default();
        let agent = build_from_config(Some(&agent_cfg), &registry).unwrap();
        assert_eq!(agent.model_id(), "ph0b0s-mock");
    }

    #[test]
    fn agent_config_drives_selection_when_no_env_override() {
        let _g = env_lock();
        clear_provider_env();
        let _v = EnvVarGuard::set("ANTHROPIC_API_KEY", "sk-test");
        let agent_cfg = AgentConfig {
            provider: "anthropic".into(),
            model: Some("claude-opus-4-7".into()),
        };
        let agent = build_from_config(Some(&agent_cfg), &ProviderRegistry::default()).unwrap();
        assert_eq!(agent.model_id(), "claude-opus-4-7");
    }

    #[test]
    fn env_fallback_picks_anthropic_when_only_anthropic_set() {
        let _g = env_lock();
        clear_provider_env();
        let _v = EnvVarGuard::set("ANTHROPIC_API_KEY", "sk-test");
        let agent = build_from_config(None, &ProviderRegistry::default()).unwrap();
        // Default Anthropic model.
        assert!(agent.model_id().starts_with("claude-"));
    }

    #[test]
    fn env_fallback_returns_no_provider_when_nothing_set() {
        let _g = env_lock();
        clear_provider_env();
        let err = build_from_config(None, &ProviderRegistry::default()).unwrap_err();
        assert!(matches!(err, BuildError::NoProviderConfigured));
    }

    #[test]
    fn unknown_provider_in_agent_config_errors() {
        let _g = env_lock();
        clear_provider_env();
        let _v = EnvVarGuard::set("PH0B0S_PROVIDER", "weatherbot");
        let err = build_from_config(None, &ProviderRegistry::default()).unwrap_err();
        match err {
            BuildError::UnknownProvider(name) => assert_eq!(name, "weatherbot"),
            other => panic!("expected UnknownProvider, got {other:?}"),
        }
    }

    #[test]
    fn registry_model_override_used_when_agent_config_has_no_model() {
        let _g = env_lock();
        clear_provider_env();
        let _v = EnvVarGuard::set("ANTHROPIC_API_KEY", "sk-test");
        let registry = ProviderRegistry {
            anthropic: Some(ProviderConfig {
                default_model: Some("claude-haiku-4-5".into()),
                base_url: None,
            }),
            ..Default::default()
        };
        let agent_cfg = AgentConfig {
            provider: "anthropic".into(),
            model: None,
        };
        let agent = build_from_config(Some(&agent_cfg), &registry).unwrap();
        assert_eq!(agent.model_id(), "claude-haiku-4-5");
    }

    #[test]
    fn agent_config_model_beats_registry_default() {
        let _g = env_lock();
        clear_provider_env();
        let _v = EnvVarGuard::set("ANTHROPIC_API_KEY", "sk-test");
        let registry = ProviderRegistry {
            anthropic: Some(ProviderConfig {
                default_model: Some("claude-haiku-4-5".into()),
                base_url: None,
            }),
            ..Default::default()
        };
        let agent_cfg = AgentConfig {
            provider: "anthropic".into(),
            model: Some("claude-opus-4-7".into()),
        };
        let agent = build_from_config(Some(&agent_cfg), &registry).unwrap();
        assert_eq!(agent.model_id(), "claude-opus-4-7");
    }
}
