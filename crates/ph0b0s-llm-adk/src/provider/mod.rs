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
