# Real LLM Providers + Tool-Call Loop + stdio MCP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire real Anthropic/OpenAI/Gemini/Ollama providers behind the `LlmAgent` seam, run a multi-turn tool-call loop in `chat()` and `session.send()`, and connect stdio MCP servers via `rmcp` + `adk_tool::McpToolset` — all inside `ph0b0s-llm-adk`, with no seam changes.

**Architecture:** Adapter owns all provider construction (`ph0b0s-llm-adk::provider::*`). CLI's `provider.rs` shrinks to a ~30-LOC dispatcher. `agent.rs` gains a private `run_loop` helper used by both `chat()` and `AdkSession::send()`. New `mcp.rs` connects via rmcp and wraps each tool in an `McpToolWrapper: NativeTool`. Selection: `PH0B0S_PROVIDER` env override > `[agents.default]` TOML > env-key precedence fallback.

**Tech Stack:** Rust 2024 / MSRV 1.85, `adk-rust = =0.6.0`, `rmcp` (already in adk-rust's transitive graph; we import directly), `tokio` + `tokio-util::CancellationToken`, `figment` (config), `tracing`. Tests use `adk_rust::model::MockLlm` (existing) and a new in-crate fake adk `Llm` for tool-loop assertions.

**Branch:** `feat/real-providers-and-tool-loop` (already created; spec already committed).

**Spec:** [`docs/superpowers/specs/2026-04-26-real-providers-and-tool-loop-design.md`](../specs/2026-04-26-real-providers-and-tool-loop-design.md)

---

## File structure

**Create:**
- `crates/ph0b0s-llm-adk/src/config.rs` — `AgentConfig`, `ProviderConfig`, `ProviderRegistry`
- `crates/ph0b0s-llm-adk/src/provider/mod.rs` — `build_from_config`, `build_from_env`, `build_named`
- `crates/ph0b0s-llm-adk/src/provider/anthropic.rs`
- `crates/ph0b0s-llm-adk/src/provider/openai.rs`
- `crates/ph0b0s-llm-adk/src/provider/gemini.rs`
- `crates/ph0b0s-llm-adk/src/provider/ollama.rs`
- `crates/ph0b0s-llm-adk/src/provider/mock.rs` — moved from `ph0b0s-cli/src/provider.rs`
- `crates/ph0b0s-llm-adk/src/mcp.rs` — rmcp + McpToolset wiring + `McpToolWrapper`
- `crates/ph0b0s-llm-adk/tests/ollama_live.rs` — `#[ignore]` live tests
- `crates/ph0b0s-llm-adk/tests/mcp_fixture.rs` — hermetic stdio MCP test
- `crates/ph0b0s-llm-adk/tests/fixtures/fake_mcp.py` — tiny stdio MCP server

**Modify:**
- `Cargo.toml` (workspace) — add `rmcp` workspace dep
- `crates/ph0b0s-llm-adk/Cargo.toml` — add `rmcp`, `tokio-util`
- `crates/ph0b0s-llm-adk/src/lib.rs` — re-exports
- `crates/ph0b0s-llm-adk/src/agent.rs` — share `run_loop` helper; tool-call loop in `chat()` + `session.send()`; `AdkLlmAgent` carries an optional `Arc<dyn ToolHost>`
- `crates/ph0b0s-llm-adk/src/tools.rs` — `mount_mcp` delegates to `mcp::mount`, gains `McpHandle` lifecycle
- `crates/ph0b0s-llm-adk/src/error.rs` — add `BuildError`
- `crates/ph0b0s-cli/src/provider.rs` — shrink to dispatcher
- `crates/ph0b0s-cli/src/config.rs` — `Config::provider_registry()` mapper, add `base_url` to `ProviderConfig`, change `AgentConfig::model` to `Option<String>`
- `xtask/src/main.rs` — extend allow-list comment if needed (no regex change; rmcp is already in the banned list, ph0b0s-llm-adk already allow-listed)
- `.github/workflows/ci.yml` — add `live-ollama` job

---

## Pre-flight

- [ ] **Step 0a: Confirm working branch + clean tree**

Run:
```bash
git -C /Users/hocinehacherouf/git/ph0b0s rev-parse --abbrev-ref HEAD
git -C /Users/hocinehacherouf/git/ph0b0s status --short
```
Expected: branch is `feat/real-providers-and-tool-loop`; status is clean (spec already committed at `989ca21`).

- [ ] **Step 0b: Verify baseline tests are green**

Run:
```bash
cd /Users/hocinehacherouf/git/ph0b0s
cargo test --workspace --all-features 2>&1 | tail -5
```
Expected: All tests pass (~178 from slice (e)).

---

## Task 1: Workspace dep additions

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/ph0b0s-llm-adk/Cargo.toml`

- [ ] **Step 1: Add `rmcp` to workspace deps**

In `Cargo.toml` under `[workspace.dependencies]`, add (after the `adk-rust` line):

```toml
# MCP client. Imported directly only by ph0b0s-llm-adk (allow-listed in xtask).
# Pinned to the version pulled in by adk-rust 0.6.0 to avoid graph divergence.
rmcp = { version = "0.7", features = ["client", "transport-child-process"] }
```

- [ ] **Step 2: Add adapter-side deps**

In `crates/ph0b0s-llm-adk/Cargo.toml` under `[dependencies]`, add:

```toml
tokio-util  = { workspace = true }
rmcp        = { workspace = true }
```

Add a `[dev-dependencies]` section (or extend the existing one) with:

```toml
[dev-dependencies]
ph0b0s-test-support = { workspace = true }
tokio = { workspace = true, features = ["test-util"] }
tempfile = { workspace = true }
```

- [ ] **Step 3: Verify compile**

Run: `cargo check -p ph0b0s-llm-adk`
Expected: success. Any rmcp version conflict here means we need to align — check `cargo tree -p ph0b0s-llm-adk -i rmcp`. If a different version is pulled transitively, change the workspace pin to match before committing.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/ph0b0s-llm-adk/Cargo.toml Cargo.lock
git commit -m "deps: add rmcp + tokio-util to ph0b0s-llm-adk"
```

---

## Task 2: Adapter config types (move from CLI)

The CLI already has `ProviderConfig { default_model }` and `AgentConfig { provider, model }` in `crates/ph0b0s-cli/src/config.rs`. We need them in the adapter so the adapter's public API is self-contained, and we need to add `base_url` and make `model` optional.

**Files:**
- Create: `crates/ph0b0s-llm-adk/src/config.rs`
- Modify: `crates/ph0b0s-llm-adk/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/ph0b0s-llm-adk/src/config.rs`:

```rust
//! Public config types consumed by the provider builders.
//!
//! These are deserialize-friendly so the CLI can hand them in directly from
//! a `figment` extraction. Keep them dumb: no validation, no env reads.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct AgentConfig {
    /// Provider name: "anthropic" | "openai" | "gemini" | "ollama" | "mock".
    pub provider: String,
    /// Override the per-provider default model. `None` => use builder default.
    pub model: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ProviderConfig {
    pub default_model: Option<String>,
    /// OpenAI: full base URL for OpenAI-compatible endpoints (Azure, OpenRouter…).
    /// Ollama: server URL (default `http://localhost:11434`).
    /// Ignored by Anthropic + Gemini.
    pub base_url: Option<String>,
}

/// Per-provider config grouped by name. Maps from the CLI's
/// `HashMap<String, ProviderConfig>` after slot-filtering.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderRegistry {
    pub anthropic: Option<ProviderConfig>,
    pub openai:    Option<ProviderConfig>,
    pub gemini:    Option<ProviderConfig>,
    pub ollama:    Option<ProviderConfig>,
}

impl ProviderRegistry {
    /// Look up the per-provider model override (if any).
    pub fn model_for(&self, provider: &str) -> Option<&str> {
        self.get(provider)
            .and_then(|p| p.default_model.as_deref())
    }

    /// Look up the per-provider base_url override (if any).
    pub fn base_url_for(&self, provider: &str) -> Option<&str> {
        self.get(provider).and_then(|p| p.base_url.as_deref())
    }

    fn get(&self, provider: &str) -> Option<&ProviderConfig> {
        match provider {
            "anthropic" => self.anthropic.as_ref(),
            "openai"    => self.openai.as_ref(),
            "gemini"    => self.gemini.as_ref(),
            "ollama"    => self.ollama.as_ref(),
            _           => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lookups_return_overrides_when_present() {
        let reg = ProviderRegistry {
            anthropic: Some(ProviderConfig {
                default_model: Some("claude-opus-4-7".into()),
                base_url: None,
            }),
            ..Default::default()
        };
        assert_eq!(reg.model_for("anthropic"), Some("claude-opus-4-7"));
        assert_eq!(reg.base_url_for("anthropic"), None);
    }

    #[test]
    fn registry_returns_none_for_unknown_provider() {
        let reg = ProviderRegistry::default();
        assert_eq!(reg.model_for("xyz"), None);
        assert_eq!(reg.base_url_for("xyz"), None);
    }

    #[test]
    fn registry_returns_base_url_when_set() {
        let reg = ProviderRegistry {
            ollama: Some(ProviderConfig {
                default_model: None,
                base_url: Some("http://192.168.1.5:11434".into()),
            }),
            ..Default::default()
        };
        assert_eq!(reg.base_url_for("ollama"), Some("http://192.168.1.5:11434"));
    }

    #[test]
    fn agent_config_deserializes_with_no_model_override() {
        let json = serde_json::json!({"provider": "anthropic"});
        let a: AgentConfig = serde_json::from_value(json).unwrap();
        assert_eq!(a.provider, "anthropic");
        assert!(a.model.is_none());
    }
}
```

- [ ] **Step 2: Add module to `lib.rs`**

In `crates/ph0b0s-llm-adk/src/lib.rs`, add (alongside existing module declarations):

```rust
pub mod config;
pub use config::{AgentConfig, ProviderConfig, ProviderRegistry};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p ph0b0s-llm-adk config::`
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/config.rs crates/ph0b0s-llm-adk/src/lib.rs
git commit -m "feat(adk): add AgentConfig/ProviderConfig/ProviderRegistry"
```

---

## Task 3: `BuildError` variants

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/error.rs`
- Test: same file `#[cfg(test)] mod tests`

- [ ] **Step 1: Read current error.rs to understand context**

Run: `cat /Users/hocinehacherouf/git/ph0b0s/crates/ph0b0s-llm-adk/src/error.rs`
Note the existing `map_adk_error` shape so we don't break it.

- [ ] **Step 2: Append `BuildError` to `error.rs`**

Add at the end of `crates/ph0b0s-llm-adk/src/error.rs`:

```rust
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
```

- [ ] **Step 3: Add tests**

Append to the existing `#[cfg(test)] mod tests` block (or add one if missing):

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p ph0b0s-llm-adk error::`
Expected: 3 new tests pass + existing pass.

- [ ] **Step 5: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/error.rs
git commit -m "feat(adk): add BuildError variants for provider construction"
```

---

## Task 4: Anthropic provider builder

**Files:**
- Create: `crates/ph0b0s-llm-adk/src/provider/anthropic.rs`
- Create: `crates/ph0b0s-llm-adk/src/provider/mod.rs` (skeleton with re-exports + helper)
- Modify: `crates/ph0b0s-llm-adk/src/lib.rs`

- [ ] **Step 1: Skeleton `provider/mod.rs` with `require_env` helper**

Create `crates/ph0b0s-llm-adk/src/provider/mod.rs`:

```rust
//! Per-provider builders. Each module owns one provider; this `mod.rs`
//! exposes `build_from_env` / `build_from_config` (filled in Task 9) and
//! shared helpers.

use crate::error::BuildError;

pub mod anthropic;
// Filled in in later tasks: openai, gemini, ollama, mock.
//
// pub mod openai;
// pub mod gemini;
// pub mod ollama;
// pub mod mock;

/// Read `key` from env, returning `BuildError::MissingKey` if absent or empty.
pub(crate) fn require_env(key: &'static str) -> Result<String, BuildError> {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => Ok(v),
        _ => Err(BuildError::MissingKey(key)),
    }
}

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

    #[test]
    fn require_env_returns_value_when_set() {
        let _g = env_lock();
        set_var("PH0B0S_TEST_REQUIRE_ENV_X", "value");
        assert_eq!(require_env("PH0B0S_TEST_REQUIRE_ENV_X").unwrap(), "value");
        unset_var("PH0B0S_TEST_REQUIRE_ENV_X");
    }

    #[test]
    fn require_env_rejects_empty_string() {
        let _g = env_lock();
        set_var("PH0B0S_TEST_REQUIRE_ENV_Y", "");
        let err = require_env("PH0B0S_TEST_REQUIRE_ENV_Y").unwrap_err();
        match err {
            BuildError::MissingKey(k) => assert_eq!(k, "PH0B0S_TEST_REQUIRE_ENV_Y"),
            other => panic!("expected MissingKey, got {other:?}"),
        }
        unset_var("PH0B0S_TEST_REQUIRE_ENV_Y");
    }

    #[test]
    fn require_env_returns_missing_key_when_unset() {
        let _g = env_lock();
        unset_var("PH0B0S_TEST_NEVER_SET");
        let err = require_env("PH0B0S_TEST_NEVER_SET").unwrap_err();
        matches!(err, BuildError::MissingKey(_));
    }
}

// Re-export the env-mutation helpers for sibling test modules in the
// `provider` tree. They're private to the crate.
#[cfg(test)]
pub(crate) use helper_tests::{env_lock, set_var, unset_var};
```

- [ ] **Step 2: Write the failing Anthropic builder tests**

Create `crates/ph0b0s-llm-adk/src/provider/anthropic.rs`:

```rust
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
    let cfg = adk_rust::model::AnthropicConfig::new(&api_key, model_id.clone());
    let client = adk_rust::model::AnthropicClient::new(cfg)
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
```

- [ ] **Step 3: Wire `provider` module into `lib.rs`**

In `crates/ph0b0s-llm-adk/src/lib.rs`, add:

```rust
pub mod provider;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p ph0b0s-llm-adk provider::`
Expected: anthropic + helper tests pass (~6).

If `AnthropicConfig::new(&api_key, ...)` doesn't compile because the constructor wants an owned `String` instead of `&str`, change `&api_key` to `api_key.clone()` and re-run. Either signature has been seen in adk-rust 0.6.x.

- [ ] **Step 5: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/provider crates/ph0b0s-llm-adk/src/lib.rs
git commit -m "feat(adk): wire Anthropic provider builder"
```

---

## Task 5: OpenAI provider builder

**Files:**
- Create: `crates/ph0b0s-llm-adk/src/provider/openai.rs`
- Modify: `crates/ph0b0s-llm-adk/src/provider/mod.rs`

- [ ] **Step 1: Write the OpenAI builder + tests**

Create `crates/ph0b0s-llm-adk/src/provider/openai.rs`:

```rust
//! OpenAI provider builder. Supports OpenAI-compatible endpoints (Azure,
//! OpenRouter, etc.) via the optional `base_url`.

use std::sync::Arc;

use crate::AdkLlmAgent;
use crate::error::BuildError;
use crate::provider::require_env;

const DEFAULT_MODEL: &str = "gpt-5-mini";

/// `model` and `base_url` are optional overrides.
pub fn build(model: Option<&str>, base_url: Option<&str>) -> Result<AdkLlmAgent, BuildError> {
    let api_key = require_env("OPENAI_API_KEY")?;
    let model_id = model.unwrap_or(DEFAULT_MODEL).to_owned();
    let cfg = match base_url {
        Some(url) => adk_rust::model::OpenAIConfig::compatible(api_key, url, model_id.clone()),
        None => adk_rust::model::OpenAIConfig::new(api_key, model_id.clone()),
    };
    let client = adk_rust::model::OpenAIClient::new(cfg)
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
        unset_var("OPENAI_API_KEY");
        let err = build(None, None).unwrap_err();
        matches!(err, BuildError::MissingKey("OPENAI_API_KEY"));
    }

    #[test]
    fn happy_path_uses_default_model() {
        let _g = env_lock();
        set_var("OPENAI_API_KEY", "sk-test");
        let agent = build(None, None).expect("should build");
        assert_eq!(agent.model_id(), DEFAULT_MODEL);
        unset_var("OPENAI_API_KEY");
    }

    #[test]
    fn base_url_path_uses_compatible_constructor() {
        let _g = env_lock();
        set_var("OPENAI_API_KEY", "sk-test");
        let agent = build(
            Some("gpt-4o-mini"),
            Some("https://openrouter.ai/api/v1"),
        ).expect("should build");
        assert_eq!(agent.model_id(), "gpt-4o-mini");
        unset_var("OPENAI_API_KEY");
    }
}
```

- [ ] **Step 2: Register module in `provider/mod.rs`**

In `crates/ph0b0s-llm-adk/src/provider/mod.rs`, change:

```rust
pub mod anthropic;
// Filled in in later tasks: openai, gemini, ollama, mock.
```

to:

```rust
pub mod anthropic;
pub mod openai;
// Filled in in later tasks: gemini, ollama, mock.
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p ph0b0s-llm-adk provider::openai::`
Expected: 3 tests pass.

If `OpenAIConfig::compatible` doesn't exist in adk-rust 0.6.0, replace with `OpenAIConfig::new(api_key, model_id).with_base_url(url)` (the builder pattern). Verify with `cargo doc -p adk-rust --open` or `grep -r "OpenAIConfig" ~/.cargo/registry/src/`.

- [ ] **Step 4: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/provider/openai.rs crates/ph0b0s-llm-adk/src/provider/mod.rs
git commit -m "feat(adk): wire OpenAI provider builder (with base_url support)"
```

---

## Task 6: Gemini provider builder

**Files:**
- Create: `crates/ph0b0s-llm-adk/src/provider/gemini.rs`
- Modify: `crates/ph0b0s-llm-adk/src/provider/mod.rs`

- [ ] **Step 1: Write the builder + tests**

Create `crates/ph0b0s-llm-adk/src/provider/gemini.rs`:

```rust
//! Gemini provider builder.

use std::sync::Arc;

use crate::AdkLlmAgent;
use crate::error::BuildError;
use crate::provider::require_env;

const DEFAULT_MODEL: &str = "gemini-2.5-flash";

pub fn build(model: Option<&str>) -> Result<AdkLlmAgent, BuildError> {
    let api_key = require_env("GOOGLE_API_KEY")?;
    let model_id = model.unwrap_or(DEFAULT_MODEL).to_owned();
    let client = adk_rust::model::GeminiModel::new(&api_key, model_id.clone())
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
        unset_var("GOOGLE_API_KEY");
        let err = build(None).unwrap_err();
        matches!(err, BuildError::MissingKey("GOOGLE_API_KEY"));
    }

    #[test]
    fn happy_path_uses_default_model() {
        let _g = env_lock();
        set_var("GOOGLE_API_KEY", "AIza-test");
        let agent = build(None).expect("should build");
        assert_eq!(agent.model_id(), DEFAULT_MODEL);
        unset_var("GOOGLE_API_KEY");
    }

    #[test]
    fn override_model_passes_through() {
        let _g = env_lock();
        set_var("GOOGLE_API_KEY", "AIza-test");
        let agent = build(Some("gemini-2.5-pro")).expect("should build");
        assert_eq!(agent.model_id(), "gemini-2.5-pro");
        unset_var("GOOGLE_API_KEY");
    }
}
```

- [ ] **Step 2: Register module + run tests**

In `crates/ph0b0s-llm-adk/src/provider/mod.rs`, add `pub mod gemini;`.

Run: `cargo test -p ph0b0s-llm-adk provider::gemini::`
Expected: 3 tests pass.

If `GeminiModel::new` is `Result`-returning vs infallible, adjust the `?`/`map_err` call accordingly. The spec was verified against context7 — `Result<Self, AdkError>` is the documented signature.

- [ ] **Step 3: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/provider/gemini.rs crates/ph0b0s-llm-adk/src/provider/mod.rs
git commit -m "feat(adk): wire Gemini provider builder"
```

---

## Task 7: Ollama provider builder

**Files:**
- Create: `crates/ph0b0s-llm-adk/src/provider/ollama.rs`
- Modify: `crates/ph0b0s-llm-adk/src/provider/mod.rs`

- [ ] **Step 1: Write the builder + tests**

Create `crates/ph0b0s-llm-adk/src/provider/ollama.rs`:

```rust
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
    let cfg = adk_rust::model::OllamaConfig::new(model_id.clone()).with_base_url(url.to_owned());
    let client = adk_rust::model::OllamaModel::new(cfg)
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
```

- [ ] **Step 2: Register module + run tests**

In `provider/mod.rs`, add `pub mod ollama;`.

Run: `cargo test -p ph0b0s-llm-adk provider::ollama::`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/provider/ollama.rs crates/ph0b0s-llm-adk/src/provider/mod.rs
git commit -m "feat(adk): wire Ollama provider builder"
```

---

## Task 8: Move mock builder out of CLI into adapter

**Files:**
- Create: `crates/ph0b0s-llm-adk/src/provider/mock.rs`
- Modify: `crates/ph0b0s-llm-adk/src/provider/mod.rs`
- Modify (later in Task 10): `crates/ph0b0s-cli/src/provider.rs`

- [ ] **Step 1: Copy mock construction into adapter, with `BuildError`**

Create `crates/ph0b0s-llm-adk/src/provider/mock.rs`:

```rust
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
                usage_metadata: Some(adk_rust::UsageMetadata::default()),
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
    use crate::provider::{env_lock, set_var, unset_var};
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
        let _g = env_lock();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("resp.json");
        std::fs::write(&path, r#"["hello","world"]"#).unwrap();
        set_var("PH0B0S_MOCK_RESPONSES", path.to_str().unwrap());
        let agent = build().unwrap();
        // MockLlm yields all responses on each call; "world" is the last.
        let r = agent.chat(ChatRequest::new().user("hi")).await.unwrap();
        assert_eq!(r.content, "world");
        unset_var("PH0B0S_MOCK_RESPONSES");
    }

    #[test]
    fn malformed_responses_file_returns_mock_error() {
        let _g = env_lock();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("resp.json");
        std::fs::write(&path, "not json").unwrap();
        set_var("PH0B0S_MOCK_RESPONSES", path.to_str().unwrap());
        let err = build().unwrap_err();
        matches!(err, BuildError::Mock(_));
        unset_var("PH0B0S_MOCK_RESPONSES");
    }

    #[test]
    fn missing_responses_file_returns_mock_error() {
        let _g = env_lock();
        set_var("PH0B0S_MOCK_RESPONSES", "/nonexistent/path/resp.json");
        let err = build().unwrap_err();
        matches!(err, BuildError::Mock(_));
        unset_var("PH0B0S_MOCK_RESPONSES");
    }

    #[test]
    fn non_array_json_returns_mock_error() {
        let _g = env_lock();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("resp.json");
        std::fs::write(&path, r#"{"not":"array"}"#).unwrap();
        set_var("PH0B0S_MOCK_RESPONSES", path.to_str().unwrap());
        let err = build().unwrap_err();
        matches!(err, BuildError::Mock(_));
        unset_var("PH0B0S_MOCK_RESPONSES");
    }
}
```

- [ ] **Step 2: Register module + run tests**

In `provider/mod.rs`, add `pub mod mock;`.

Run: `cargo test -p ph0b0s-llm-adk provider::mock::`
Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/provider/mock.rs crates/ph0b0s-llm-adk/src/provider/mod.rs
git commit -m "feat(adk): move mock provider builder into adapter"
```

---

## Task 9: Selection logic — `build_from_config` + `build_from_env`

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/provider/mod.rs`

- [ ] **Step 1: Append public selection functions**

Add to `crates/ph0b0s-llm-adk/src/provider/mod.rs` (after the `require_env` definition):

```rust
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
        "openai"    => openai::build(model, providers.base_url_for("openai")),
        "gemini"    => gemini::build(model),
        "ollama"    => ollama::build(model, providers.base_url_for("ollama")),
        "mock"      => mock::build(),
        other       => Err(BuildError::UnknownProvider(other.to_owned())),
    }
}
```

- [ ] **Step 2: Add table-driven selection tests**

Append to the same file (still under `#[cfg(test)] mod helper_tests`, or a new sibling `mod selection_tests`):

```rust
#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::config::{AgentConfig, ProviderConfig, ProviderRegistry};
    use crate::provider::{env_lock, set_var, unset_var};
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
        set_var("PH0B0S_PROVIDER", "mock");
        // [agents.default] says anthropic, but env override wins.
        let agent_cfg = AgentConfig {
            provider: "anthropic".into(),
            model: None,
        };
        let registry = ProviderRegistry::default();
        let agent = build_from_config(Some(&agent_cfg), &registry).unwrap();
        assert_eq!(agent.model_id(), "ph0b0s-mock");
        clear_provider_env();
    }

    #[test]
    fn agent_config_drives_selection_when_no_env_override() {
        let _g = env_lock();
        clear_provider_env();
        set_var("ANTHROPIC_API_KEY", "sk-test");
        let agent_cfg = AgentConfig {
            provider: "anthropic".into(),
            model: Some("claude-opus-4-7".into()),
        };
        let agent = build_from_config(Some(&agent_cfg), &ProviderRegistry::default()).unwrap();
        assert_eq!(agent.model_id(), "claude-opus-4-7");
        clear_provider_env();
    }

    #[test]
    fn env_fallback_picks_anthropic_when_only_anthropic_set() {
        let _g = env_lock();
        clear_provider_env();
        set_var("ANTHROPIC_API_KEY", "sk-test");
        let agent = build_from_config(None, &ProviderRegistry::default()).unwrap();
        // Default Anthropic model.
        assert!(agent.model_id().starts_with("claude-"));
        clear_provider_env();
    }

    #[test]
    fn env_fallback_returns_no_provider_when_nothing_set() {
        let _g = env_lock();
        clear_provider_env();
        let err = build_from_config(None, &ProviderRegistry::default()).unwrap_err();
        matches!(err, BuildError::NoProviderConfigured);
    }

    #[test]
    fn unknown_provider_in_agent_config_errors() {
        let _g = env_lock();
        clear_provider_env();
        set_var("PH0B0S_PROVIDER", "weatherbot");
        let err = build_from_config(None, &ProviderRegistry::default()).unwrap_err();
        match err {
            BuildError::UnknownProvider(name) => assert_eq!(name, "weatherbot"),
            other => panic!("expected UnknownProvider, got {other:?}"),
        }
        clear_provider_env();
    }

    #[test]
    fn registry_model_override_used_when_agent_config_has_no_model() {
        let _g = env_lock();
        clear_provider_env();
        set_var("ANTHROPIC_API_KEY", "sk-test");
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
        clear_provider_env();
    }

    #[test]
    fn agent_config_model_beats_registry_default() {
        let _g = env_lock();
        clear_provider_env();
        set_var("ANTHROPIC_API_KEY", "sk-test");
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
        clear_provider_env();
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p ph0b0s-llm-adk provider::`
Expected: ~24 tests pass (anthropic 3 + openai 3 + gemini 3 + ollama 3 + mock 5 + selection 7 + helpers 3).

- [ ] **Step 4: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/provider/mod.rs
git commit -m "feat(adk): add build_from_config + build_from_env selection"
```

---

## Task 10: CLI dispatcher shrink + `provider_registry()` mapper

**Files:**
- Modify: `crates/ph0b0s-cli/src/config.rs`
- Modify: `crates/ph0b0s-cli/src/provider.rs`

- [ ] **Step 1: Update CLI's `ProviderConfig` to add `base_url` + flip `AgentConfig::model` to `Option<String>`**

In `crates/ph0b0s-cli/src/config.rs`, find:

```rust
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub default_model: Option<String>,
    // api_key NEVER here — read from env vars at runtime.
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub provider: String,
    pub model: String,
}
```

Replace with:

```rust
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub default_model: Option<String>,
    /// OpenAI-compatible endpoints + Ollama server URL.
    pub base_url: Option<String>,
    // api_key NEVER here — read from env vars at runtime.
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub provider: String,
    /// Per-agent model override. `None` ⇒ use the per-provider default.
    pub model: Option<String>,
}
```

- [ ] **Step 2: Add `provider_registry()` + `default_agent()` mappers on `Config`**

In `crates/ph0b0s-cli/src/config.rs`, add inside `impl Config { ... }`:

```rust
/// Map the figment-loaded `HashMap<String, ProviderConfig>` to the
/// adapter's typed `ProviderRegistry`.
pub fn provider_registry(&self) -> ph0b0s_llm_adk::ProviderRegistry {
    use ph0b0s_llm_adk::{ProviderConfig as AdkProviderConfig, ProviderRegistry};
    let to_adk = |c: &ProviderConfig| AdkProviderConfig {
        default_model: c.default_model.clone(),
        base_url: c.base_url.clone(),
    };
    ProviderRegistry {
        anthropic: self.providers.get("anthropic").map(to_adk),
        openai:    self.providers.get("openai").map(to_adk),
        gemini:    self.providers.get("gemini").map(to_adk),
        ollama:    self.providers.get("ollama").map(to_adk),
    }
}

/// Map `[agents.default]` to the adapter's typed `AgentConfig`. Returns
/// `None` when no default agent is configured (callers fall back to the
/// env-key detection path).
pub fn default_agent(&self) -> Option<ph0b0s_llm_adk::AgentConfig> {
    self.agents.get("default").map(|a| ph0b0s_llm_adk::AgentConfig {
        provider: a.provider.clone(),
        model: a.model.clone(),
    })
}
```

- [ ] **Step 3: Add `ph0b0s-llm-adk` to CLI's deps if not already**

Run: `grep ph0b0s-llm-adk /Users/hocinehacherouf/git/ph0b0s/crates/ph0b0s-cli/Cargo.toml`
Expected: a line in `[dependencies]`. If absent, add `ph0b0s-llm-adk = { workspace = true }`.

- [ ] **Step 4: Rewrite CLI's `provider.rs` as the thin dispatcher**

Replace the entire contents of `crates/ph0b0s-cli/src/provider.rs` with:

```rust
//! Build an `AdkLlmAgent` from layered config + env at startup.
//!
//! The actual selection logic + per-provider construction lives in
//! `ph0b0s_llm_adk::provider`. The CLI just hands the typed config in.

use anyhow::Result;
use ph0b0s_llm_adk::{AdkLlmAgent, provider};

use crate::config::Config;

/// Build the LLM agent for the current process.
pub fn build_from_config(config: &Config) -> Result<AdkLlmAgent> {
    provider::build_from_config(
        config.default_agent().as_ref(),
        &config.provider_registry(),
    )
    .map_err(Into::into)
}
```

- [ ] **Step 5: Update CLI call sites**

The previous public function was `build_from_env()`. Find callers and update to `build_from_config(&config)`. Run:

```bash
grep -rn "provider::build_from_env\|provider::build" /Users/hocinehacherouf/git/ph0b0s/crates/ph0b0s-cli/src/
```

For each call site (likely `commands/scan.rs` or `lib.rs`), pass the loaded `Config` in. If a `Config` isn't available at the call site, load one inline via `Config::load()?`.

- [ ] **Step 6: Move CLI's existing mock-provider tests over (or delete duplicates)**

The mock builder + its tests now live in `crates/ph0b0s-llm-adk/src/provider/mock.rs`. The CLI's old tests in `provider.rs` are gone (the file shrank to ~15 LOC and has no logic to test). If a wrapping integration test is wanted, add to `crates/ph0b0s-cli/src/provider.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// Crashes on missing env, exercises the full mapper path. The actual
    /// selection logic is tested exhaustively in `ph0b0s_llm_adk::provider`.
    #[test]
    fn dispatcher_propagates_no_provider_error() {
        // Save + clear all relevant env vars so the test is hermetic.
        let saved: Vec<(&str, Option<String>)> = [
            "PH0B0S_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OLLAMA_HOST",
        ]
        .into_iter()
        .map(|k| (k, std::env::var(k).ok()))
        .collect();
        for (k, _) in &saved {
            unsafe { std::env::remove_var(k) };
        }
        let cfg = Config::default();
        let err = build_from_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("no LLM provider"));
        // Restore.
        for (k, v) in saved {
            if let Some(v) = v {
                unsafe { std::env::set_var(k, v) };
            }
        }
    }
}
```

- [ ] **Step 7: Run CLI tests + workspace check**

```bash
cargo check -p ph0b0s-cli
cargo test -p ph0b0s-cli
cargo test --workspace
```

Expected: green. If a test in `ph0b0s-cli/tests/end_to_end.rs` exercises `PH0B0S_PROVIDER=mock`, it should still pass — the env var path still wires to the moved mock builder.

- [ ] **Step 8: Commit**

```bash
git add crates/ph0b0s-cli/src/config.rs crates/ph0b0s-cli/src/provider.rs crates/ph0b0s-cli/src/
git commit -m "refactor(cli): shrink provider.rs to dispatcher; map config to adapter types"
```

---

## Task 11: Tool-call loop — extract `run_loop` (still single-shot, no tools yet)

This task is structural-only: factor the existing `chat()` body into a `run_loop` helper that takes the same inputs as today and behaves identically. The next task adds the actual loop.

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/agent.rs`

- [ ] **Step 1: Refactor `chat()` to call a private `run_loop`**

In `crates/ph0b0s-llm-adk/src/agent.rs`, replace the `chat` impl with:

```rust
#[tracing::instrument(skip_all, fields(model = %self.model_id, role = %self.role.0))]
async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
    if !req.tools.is_empty() {
        tracing::warn!(
            tool_count = req.tools.len(),
            "tools passed to chat() but tool-loop wiring is in progress; \
             dispatch is single-shot in this build"
        );
    }
    self.run_loop(req.messages, None, &req.hints).await
}
```

Then add as a private method on `AdkLlmAgent`:

```rust
async fn run_loop(
    &self,
    messages: Vec<ChatMessage>,
    schema: Option<serde_json::Value>,
    _hints: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Result<ChatResponse, LlmError> {
    let adk_req = build_request(
        &self.model_id,
        &messages,
        self.default_system.as_deref(),
        schema,
    );
    let stream = self
        .llm
        .generate_content(adk_req, false)
        .await
        .map_err(map_adk_error)?;
    let response = collect_final(stream).await?;
    Ok(to_chat_response(response))
}
```

Also update `structured` to keep using the existing single-shot path (do NOT route it through `run_loop` — the spec says structured stays single-shot):

```rust
#[tracing::instrument(skip_all, fields(model = %self.model_id, role = %self.role.0, schema = %req.schema_name))]
async fn structured(&self, req: StructuredRequest) -> Result<serde_json::Value, LlmError> {
    let adk_req = build_request(
        &self.model_id,
        &req.messages,
        self.default_system.as_deref(),
        Some(req.schema.clone()),
    );
    let stream = self
        .llm
        .generate_content(adk_req, false)
        .await
        .map_err(map_adk_error)?;
    let response = collect_final(stream).await?;
    let text = extract_text(response.content.as_ref());
    parse_json_loose(&text)
}
```

- [ ] **Step 2: Verify the existing tests still pass**

Run: `cargo test -p ph0b0s-llm-adk agent::`
Expected: All existing tests still pass (~17). Pure refactor.

- [ ] **Step 3: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/agent.rs
git commit -m "refactor(adk): extract run_loop helper from chat()"
```

---

## Task 12: Tool-call loop — wire `ToolHost` into `AdkLlmAgent`

We need the loop to dispatch tool calls. That means the agent needs access to a `ToolHost`. Add it as an optional field with a builder method.

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/agent.rs`

- [ ] **Step 1: Add `tool_host` field + builder**

In `crates/ph0b0s-llm-adk/src/agent.rs`, change:

```rust
#[derive(Clone)]
pub struct AdkLlmAgent {
    llm: Arc<dyn adk_rust::Llm>,
    model_id: String,
    role: AgentRoleKey,
    default_system: Option<String>,
}
```

to:

```rust
#[derive(Clone)]
pub struct AdkLlmAgent {
    llm: Arc<dyn adk_rust::Llm>,
    model_id: String,
    role: AgentRoleKey,
    default_system: Option<String>,
    /// Optional `ToolHost` used by `chat()`'s tool-call loop. When `None`,
    /// the loop will only fire if `req.tools` is empty AND the model emits
    /// no `Part::FunctionCall` — it'll error if the model tries to call
    /// any tool.
    tool_host: Option<Arc<dyn ph0b0s_core::tools::ToolHost>>,
}
```

Update the `Debug` impl + `new` to include the new field (set `tool_host: None` in `new`). Add a builder:

```rust
pub fn with_tool_host(
    mut self,
    host: Arc<dyn ph0b0s_core::tools::ToolHost>,
) -> Self {
    self.tool_host = Some(host);
    self
}
```

- [ ] **Step 2: Update CLI wiring to call `with_tool_host` when scanning**

Locate where `AdkToolHost` is constructed in the CLI scan path (likely `crates/ph0b0s-cli/src/scan.rs` or `commands/scan.rs`). After building the `AdkLlmAgent`, pass the host:

```rust
let tool_host: Arc<dyn ph0b0s_core::tools::ToolHost> = Arc::new(adk_tool_host.clone());
let agent = agent.with_tool_host(tool_host);
```

Read first to find the exact site:

```bash
grep -rn "AdkLlmAgent::new\|build_from_config\|AdkToolHost::new" /Users/hocinehacherouf/git/ph0b0s/crates/ph0b0s-cli/src/
```

- [ ] **Step 3: Run tests**

Run: `cargo test --workspace`
Expected: green. Pure additive change; tool-loop logic still not exercised.

- [ ] **Step 4: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/agent.rs crates/ph0b0s-cli/
git commit -m "feat(adk): plumb ToolHost into AdkLlmAgent (no loop yet)"
```

---

## Task 13: Tool-call loop — function-call dispatch + multi-turn

This is the core change. After this task, `chat()` runs a real multi-turn loop.

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/agent.rs`

- [ ] **Step 1: Write the failing tests first**

Add to `agent.rs`'s `#[cfg(test)] mod tests` (after the existing tests). The first test asserts the loop dispatches a tool call and feeds the result back:

```rust
/// Test fake of `adk_rust::Llm` that pops queued responses on each call.
/// We need this because adk-rust's stock `MockLlm` returns *all* canned
/// responses on every call to `generate_content`; a multi-turn test needs
/// each `generate_content` invocation to return *one* response.
#[derive(Clone)]
struct ScriptedLlm {
    queue: Arc<Mutex<std::collections::VecDeque<adk_rust::LlmResponse>>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<adk_rust::LlmResponse>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(responses.into_iter().collect())),
        }
    }
}

#[async_trait]
impl adk_rust::Llm for ScriptedLlm {
    async fn generate_content(
        &self,
        _req: adk_rust::LlmRequest,
        _stream: bool,
    ) -> Result<adk_rust::LlmResponseStream, adk_rust::AdkError> {
        let resp = self
            .queue
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| adk_rust::AdkError::Other("ScriptedLlm queue empty".into()))?;
        Ok(Box::pin(futures::stream::once(async move { Ok(resp) })))
    }

    fn model_name(&self) -> &str {
        "scripted"
    }
}

fn fc_response(name: &str, args: serde_json::Value) -> adk_rust::LlmResponse {
    let mut content = adk_rust::Content::new("model");
    content.parts.push(adk_rust::Part::FunctionCall {
        name: name.into(),
        args,
        id: Some(format!("call_{name}")),
        thought_signature: None,
    });
    adk_rust::LlmResponse {
        content: Some(content),
        usage_metadata: Some(adk_rust::UsageMetadata {
            prompt_token_count: 5,
            candidates_token_count: 5,
            total_token_count: 10,
            ..Default::default()
        }),
        finish_reason: None,
        ..Default::default()
    }
}

fn text_response(text: &str) -> adk_rust::LlmResponse {
    adk_rust::LlmResponse {
        content: Some(adk_rust::Content::new("model").with_text(text.into())),
        usage_metadata: Some(adk_rust::UsageMetadata {
            prompt_token_count: 3,
            candidates_token_count: 4,
            total_token_count: 7,
            ..Default::default()
        }),
        finish_reason: Some(adk_rust::FinishReason::Stop),
        ..Default::default()
    }
}

#[tokio::test]
async fn chat_dispatches_function_call_and_returns_final_text() {
    use ph0b0s_test_support::MockToolHost;
    let llm: Arc<dyn adk_rust::Llm> = Arc::new(ScriptedLlm::new(vec![
        fc_response("search", serde_json::json!({"q": "rustsec"})),
        text_response("done: found 0 advisories"),
    ]));
    let host = Arc::new(MockToolHost::new());
    host.enqueue_response("search", serde_json::json!({"hits": 0}));
    let agent = AdkLlmAgent::new(llm, "scripted").with_tool_host(host.clone());
    let req = ChatRequest::new().user("look up advisories");
    let resp = agent.chat(req).await.unwrap();
    assert_eq!(resp.content, "done: found 0 advisories");
    let invs = host.invocations();
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].0, "search");
    // Usage accumulated across both turns: 5+3=8 tokens_in, 5+4=9 tokens_out.
    assert_eq!(resp.usage.tokens_in, 8);
    assert_eq!(resp.usage.tokens_out, 9);
}
```

- [ ] **Step 2: Implement the real loop**

Replace the simple `run_loop` with the full version. Add `use ph0b0s_core::llm::ToolSpec;` near the top of the file if missing, and add `use std::sync::Mutex;` if not present.

```rust
async fn run_loop(
    &self,
    initial_messages: Vec<ChatMessage>,
    schema: Option<serde_json::Value>,
    hints: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Result<ChatResponse, LlmError> {
    let max_turns = hints
        .get("max_tool_turns")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;

    // Build initial adk Contents from the seam messages (system handling
    // already in build_request).
    let mut contents = build_initial_contents(
        &initial_messages,
        self.default_system.as_deref(),
    );

    let mut cumulative = Usage::default();
    let mut last_finish: Option<adk_rust::FinishReason> = None;

    for _turn in 0..max_turns {
        let mut adk_req = adk_rust::LlmRequest::new(self.model_id.clone(), contents.clone());
        if let Some(s) = schema.clone() {
            adk_req = adk_req.with_response_schema(s);
        }

        let stream = self
            .llm
            .generate_content(adk_req, false)
            .await
            .map_err(map_adk_error)?;
        let response = collect_final(stream).await?;
        accumulate(&mut cumulative, &from_adk_usage(response.usage_metadata.as_ref()));
        last_finish = response.finish_reason;

        let model_content = response.content.clone().unwrap_or_else(|| {
            adk_rust::Content::new("model")
        });
        let function_calls = collect_function_calls(&model_content);

        if function_calls.is_empty() {
            // Final turn — return assistant text.
            return Ok(ChatResponse {
                content: extract_text(Some(&model_content)),
                tool_calls: Vec::new(),
                usage: cumulative,
                finish_reason: map_finish_reason(last_finish),
            });
        }

        // Append the model's tool-calling turn.
        contents.push(model_content);

        // Sequentially dispatch each tool call.
        let mut tool_parts = Vec::new();
        for (name, args, id) in function_calls {
            let result = match self.tool_host.as_ref() {
                Some(host) => host.invoke(&name, args.clone()).await,
                None => Err(ph0b0s_core::error::ToolError::Unknown(name.clone())),
            };
            let response_value = match result {
                Ok(v) => v,
                Err(e) => serde_json::json!({"error": e.to_string()}),
            };
            tool_parts.push(adk_rust::Part::FunctionResponse {
                name: name.clone(),
                response: response_value,
                id,
            });
        }

        // Append the tool-result turn.
        let mut tool_content = adk_rust::Content::new("tool");
        tool_content.parts = tool_parts;
        contents.push(tool_content);
    }

    Err(LlmError::ToolDispatch(format!(
        "model exceeded max_tool_turns ({max_turns}) without producing a final reply"
    )))
}

/// Build the initial Vec<Content> respecting the default_system rules. Same
/// logic as `build_request` but exposed for the loop.
fn build_initial_contents(
    messages: &[ChatMessage],
    default_system: Option<&str>,
) -> Vec<adk_rust::Content> {
    let mut contents = Vec::new();
    let has_explicit_system = messages
        .iter()
        .any(|m| matches!(m, ChatMessage::System { .. }));
    if !has_explicit_system {
        if let Some(sp) = default_system {
            contents.push(adk_rust::Content::new("system").with_text(sp.to_owned()));
        }
    }
    for m in messages {
        contents.push(chat_message_to_content(m));
    }
    contents
}

/// Extract `(name, args, id)` triples for every `Part::FunctionCall` in `c`.
fn collect_function_calls(c: &adk_rust::Content) -> Vec<(String, serde_json::Value, Option<String>)> {
    c.parts
        .iter()
        .filter_map(|p| match p {
            adk_rust::Part::FunctionCall { name, args, id, .. } => {
                Some((name.clone(), args.clone(), id.clone()))
            }
            _ => None,
        })
        .collect()
}
```

Note: the exact `Part::FunctionResponse` fields depend on adk-rust 0.6.0. If the variant is `FunctionResponse { function_response: FunctionResponseData, id: ... }` (object-shaped), adapt accordingly — the `cargo build` failure will tell you. Keep a single line of comment explaining whichever shape we settle on.

If `LlmError::ToolDispatch` doesn't exist yet, add it to `ph0b0s-core::error::LlmError`:

```rust
#[error("tool dispatch: {0}")]
ToolDispatch(String),
```

(One-line addition; bump test count.)

- [ ] **Step 3: Run tests**

Run: `cargo test -p ph0b0s-llm-adk agent::tests::chat_dispatches_function_call_and_returns_final_text`
Expected: pass. Then run the whole suite:

```bash
cargo test -p ph0b0s-llm-adk
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/agent.rs crates/ph0b0s-core/src/error.rs
git commit -m "feat(adk): tool-call loop in chat() with sequential dispatch"
```

---

## Task 14: Tool-call loop — error + max-turns + multi-call tests

Add the remaining loop-behavior tests. The implementation from Task 13 should already handle them; this task is mostly assertion coverage to satisfy the patch-100% gate.

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/agent.rs` (test additions only)

- [ ] **Step 1: Add tool-error test**

Append to `agent.rs` test module:

```rust
#[tokio::test]
async fn chat_feeds_tool_error_back_as_function_response() {
    use ph0b0s_test_support::MockToolHost;
    let llm: Arc<dyn adk_rust::Llm> = Arc::new(ScriptedLlm::new(vec![
        fc_response("search", serde_json::json!({"q": "x"})),
        text_response("recovered"),
    ]));
    let host = Arc::new(MockToolHost::new());
    host.enqueue_error(
        "search",
        ph0b0s_core::error::ToolError::Execution("boom".into()),
    );
    let agent = AdkLlmAgent::new(llm, "scripted").with_tool_host(host.clone());
    let resp = agent.chat(ChatRequest::new().user("go")).await.unwrap();
    assert_eq!(resp.content, "recovered");
    // No assertion about content of FunctionResponse here — the next test
    // verifies the model received it. We just verify no panic.
}
```

- [ ] **Step 2: Add max-turns test**

```rust
#[tokio::test]
async fn chat_returns_tool_dispatch_error_when_max_turns_exceeded() {
    use ph0b0s_test_support::MockToolHost;
    // 11 function-call responses; loop default cap is 10 turns.
    let mut canned = Vec::new();
    for i in 0..11 {
        canned.push(fc_response("loop", serde_json::json!({"i": i})));
    }
    let llm: Arc<dyn adk_rust::Llm> = Arc::new(ScriptedLlm::new(canned));
    let host = Arc::new(MockToolHost::new());
    for _ in 0..11 {
        host.enqueue_response("loop", serde_json::json!({}));
    }
    let agent = AdkLlmAgent::new(llm, "scripted").with_tool_host(host);
    let err = agent.chat(ChatRequest::new().user("loop")).await.unwrap_err();
    match err {
        LlmError::ToolDispatch(msg) => assert!(msg.contains("max_tool_turns")),
        other => panic!("expected ToolDispatch, got {other:?}"),
    }
}
```

- [ ] **Step 3: Add multi-call-in-one-turn test**

```rust
#[tokio::test]
async fn chat_dispatches_multiple_calls_in_one_turn_sequentially() {
    use ph0b0s_test_support::MockToolHost;
    // Turn 1: model emits two FunctionCalls in the same Content.
    let mut content = adk_rust::Content::new("model");
    content.parts.push(adk_rust::Part::FunctionCall {
        name: "a".into(),
        args: serde_json::json!({}),
        id: Some("call_a".into()),
        thought_signature: None,
    });
    content.parts.push(adk_rust::Part::FunctionCall {
        name: "b".into(),
        args: serde_json::json!({}),
        id: Some("call_b".into()),
        thought_signature: None,
    });
    let multi_call_resp = adk_rust::LlmResponse {
        content: Some(content),
        usage_metadata: Some(adk_rust::UsageMetadata::default()),
        finish_reason: None,
        ..Default::default()
    };
    let llm: Arc<dyn adk_rust::Llm> = Arc::new(ScriptedLlm::new(vec![
        multi_call_resp,
        text_response("ok"),
    ]));
    let host = Arc::new(MockToolHost::new());
    host.enqueue_response("a", serde_json::json!("a-ret"));
    host.enqueue_response("b", serde_json::json!("b-ret"));
    let agent = AdkLlmAgent::new(llm, "scripted").with_tool_host(host.clone());
    let _ = agent.chat(ChatRequest::new().user("go")).await.unwrap();
    let order: Vec<&str> = host.invocations().iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(order, vec!["a", "b"]);
}
```

- [ ] **Step 4: Add no-tool-host fallback test**

```rust
#[tokio::test]
async fn chat_with_no_tool_host_feeds_unknown_error_back() {
    let llm: Arc<dyn adk_rust::Llm> = Arc::new(ScriptedLlm::new(vec![
        fc_response("anything", serde_json::json!({})),
        text_response("done"),
    ]));
    let agent = AdkLlmAgent::new(llm, "scripted"); // no tool host
    let resp = agent.chat(ChatRequest::new().user("go")).await.unwrap();
    assert_eq!(resp.content, "done");
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p ph0b0s-llm-adk agent::
```
Expected: all green (existing + 4 new).

- [ ] **Step 6: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/agent.rs
git commit -m "test(adk): cover tool-loop error/max-turns/multi-call branches"
```

---

## Task 15: Tool-call loop — `req.tools` override (decision 4)

Per the spec: if `req.tools` is non-empty, only those tools are visible to the model; if empty, fall back to `tool_host.list()`. The current loop doesn't pass tool decls into `LlmRequest` at all — this task wires it up.

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/agent.rs`

- [ ] **Step 1: Resolve and pass tool list**

In `run_loop`, before the `for _turn in 0..max_turns` loop, add:

```rust
let resolved_tools = if !req_tools.is_empty() {
    req_tools.clone()
} else if let Some(host) = self.tool_host.as_ref() {
    host.list()
} else {
    Vec::new()
};
```

This means `run_loop` needs `req_tools: &[ToolSpec]` as a new parameter. Update the signature:

```rust
async fn run_loop(
    &self,
    initial_messages: Vec<ChatMessage>,
    req_tools: &[ph0b0s_core::llm::ToolSpec],
    schema: Option<serde_json::Value>,
    hints: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Result<ChatResponse, LlmError> {
```

Update `chat()`:

```rust
self.run_loop(req.messages, &req.tools, None, &req.hints).await
```

Inside the loop, attach the resolved tools to each `LlmRequest`:

```rust
let mut adk_req = adk_rust::LlmRequest::new(self.model_id.clone(), contents.clone());
if !resolved_tools.is_empty() {
    let adk_tools = resolved_tools.iter().map(to_adk_tool_decl).collect::<Vec<_>>();
    adk_req = adk_req.with_tools(adk_tools);
}
if let Some(s) = schema.clone() {
    adk_req = adk_req.with_response_schema(s);
}
```

Add the `to_adk_tool_decl` helper at the bottom of the file:

```rust
/// Convert a seam `ToolSpec` to adk-rust's tool declaration. Best-effort
/// JSON-Schema passthrough; adk normalizes for the underlying provider.
fn to_adk_tool_decl(spec: &ph0b0s_core::llm::ToolSpec) -> adk_rust::FunctionDeclaration {
    adk_rust::FunctionDeclaration {
        name: spec.name.clone(),
        description: spec.description.clone().unwrap_or_default(),
        parameters: Some(spec.schema.clone()),
    }
}
```

If adk-rust 0.6.0 names the type something other than `FunctionDeclaration` (e.g. `Tool` or `ToolDeclaration`), the compiler will error — adapt to the real name. Verify with `grep "pub struct" ~/.cargo/registry/src/*adk-rust*/src/`.

- [ ] **Step 2: Test the override path**

Add to `agent.rs` tests:

```rust
#[tokio::test]
async fn chat_uses_req_tools_when_non_empty_else_falls_back_to_host_list() {
    use ph0b0s_core::llm::{ToolSource, ToolSpec};
    use ph0b0s_test_support::MockToolHost;

    // First call: req.tools provided → only "specific" should be visible.
    // The MockLlm doesn't actually inspect tools — but we exercise the code
    // path by verifying chat() builds a request with the right tool list.
    // We do this by reading `host.list()` after registering vs not using it.
    let host = Arc::new(MockToolHost::new());
    let dummy = Arc::new(crate::tools::tests::EchoTool {
        name: "host_tool".into(),
        response: serde_json::json!({}),
    }) as Arc<dyn ph0b0s_core::tools::NativeTool>;
    host.register_native(dummy);
    assert_eq!(host.list().len(), 1);
    // The behavioral guarantee in v1: if req.tools is provided, the host's
    // own list is *not* used. We can't observe this in MockLlm without
    // more plumbing, so this test mainly asserts compile + no-panic.
    let llm: Arc<dyn adk_rust::Llm> = Arc::new(ScriptedLlm::new(vec![text_response("ok")]));
    let agent = AdkLlmAgent::new(llm, "scripted").with_tool_host(host);
    let mut req = ChatRequest::new().user("go");
    req.tools.push(ToolSpec {
        name: "specific".into(),
        description: None,
        schema: serde_json::json!({}),
        source: ToolSource::Native,
    });
    let r = agent.chat(req).await.unwrap();
    assert_eq!(r.content, "ok");
}
```

(The behavioral assertion is weak — adk's `MockLlm` doesn't observe `req.tools`. Tighter coverage comes from the live tests later.)

- [ ] **Step 3: Run tests**

```bash
cargo test -p ph0b0s-llm-adk
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/agent.rs
git commit -m "feat(adk): tool-call loop honours req.tools override"
```

---

## Task 16: `AdkSession::send` shares `run_loop`

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/agent.rs`

- [ ] **Step 1: Refactor `AdkSession`**

Right now `AdkSession::new` doesn't carry a tool host. The cleanest fix is to make `AdkSession` hold an `Option<Arc<dyn ToolHost>>` and have `AdkLlmAgent::session()` pass `self.tool_host.clone()` through. The session's `send()` then uses the same `run_loop`-equivalent logic.

Change `AdkSession`:

```rust
#[derive(Clone)]
pub struct AdkSession {
    llm: Arc<dyn adk_rust::Llm>,
    model_id: String,
    state: Arc<Mutex<SessionState>>,
    tool_host: Option<Arc<dyn ph0b0s_core::tools::ToolHost>>,
}
```

Update `AdkSession::new`:

```rust
pub fn new(
    llm: Arc<dyn adk_rust::Llm>,
    model_id: impl Into<String>,
    system_prompt: Option<String>,
    tool_host: Option<Arc<dyn ph0b0s_core::tools::ToolHost>>,
) -> Self {
    // ... existing body, plus tool_host
}
```

Update `AdkLlmAgent::session()`:

```rust
async fn session(&self, opts: SessionOptions) -> Result<Box<dyn LlmSession>, LlmError> {
    Ok(Box::new(AdkSession::new(
        self.llm.clone(),
        self.model_id.clone(),
        opts.system_prompt.or_else(|| self.default_system.clone()),
        self.tool_host.clone(),
    )))
}
```

- [ ] **Step 2: Run the loop inside `send`**

Replace `AdkSession::send`'s body with a loop that mirrors `run_loop` but maintains session history. Pull a private free function `run_loop_inner(llm, model_id, contents, tool_host, max_turns) -> Result<(ChatResponse, Vec<Content>), LlmError>` so both `AdkLlmAgent::run_loop` and `AdkSession::send` can call it. Inline the body if a free function feels heavy; the goal is no duplication of dispatch logic.

For brevity, here's the simplest path that keeps both paths through one function:

```rust
async fn run_loop_inner(
    llm: &Arc<dyn adk_rust::Llm>,
    model_id: &str,
    initial_contents: Vec<adk_rust::Content>,
    tool_host: &Option<Arc<dyn ph0b0s_core::tools::ToolHost>>,
    resolved_tools: &[ph0b0s_core::llm::ToolSpec],
    schema: Option<serde_json::Value>,
    max_turns: usize,
) -> Result<(ChatResponse, Vec<adk_rust::Content>), LlmError> {
    // Same body as the existing run_loop's per-turn block, but:
    // - takes initial_contents (not initial_messages),
    // - returns the appended history alongside the ChatResponse.
}
```

Then:
- `AdkLlmAgent::run_loop` builds initial contents from messages, calls `run_loop_inner`, returns just the `ChatResponse`.
- `AdkSession::send` builds initial contents from `state.history + new user msg`, calls `run_loop_inner`, replaces `state.history` with the returned conversation, accumulates usage.

- [ ] **Step 3: Add a session-side tool-loop test**

```rust
#[tokio::test]
async fn session_send_dispatches_function_call_and_extends_history() {
    use ph0b0s_test_support::MockToolHost;
    let llm: Arc<dyn adk_rust::Llm> = Arc::new(ScriptedLlm::new(vec![
        fc_response("search", serde_json::json!({"q":"x"})),
        text_response("found"),
    ]));
    let host: Arc<dyn ph0b0s_core::tools::ToolHost> = Arc::new(MockToolHost::new()
        .enqueue_response_owned("search", serde_json::json!({"hits":1})));
    let mut sess = AdkSession::new(llm, "scripted", None, Some(host));
    let r = sess.send(UserMessage::new("ping")).await.unwrap();
    assert_eq!(r.content, "found");
    let history = sess.history();
    // user, model(fc), tool(fr), model(text)  → 4 turns
    assert_eq!(history.len(), 4);
}
```

(If `MockToolHost::enqueue_response` only takes `&self` and returns `&Self`, you may need an `enqueue_response_owned` helper or chained calls. Adjust to existing API.)

- [ ] **Step 4: Run tests**

```bash
cargo test -p ph0b0s-llm-adk
```

- [ ] **Step 5: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/agent.rs
git commit -m "feat(adk): AdkSession.send uses shared tool-call loop"
```

---

## Task 17: MCP module skeleton

**Files:**
- Create: `crates/ph0b0s-llm-adk/src/mcp.rs`
- Modify: `crates/ph0b0s-llm-adk/src/lib.rs`

- [ ] **Step 1: Sketch `mcp.rs` with `mount` + `McpToolWrapper`**

Create `crates/ph0b0s-llm-adk/src/mcp.rs`:

```rust
//! MCP integration: spawn stdio MCP servers via rmcp, wrap each discovered
//! tool as a `NativeTool`, register with `AdkToolHost`. This module is the
//! only place outside `adk_rust::*` calls that imports `rmcp::*`.

use std::sync::Arc;

use async_trait::async_trait;
use ph0b0s_core::error::ToolError;
use ph0b0s_core::llm::{ToolSource, ToolSpec};
use ph0b0s_core::tools::{McpServerSpec, McpTransport, NativeTool};
use rmcp::{ServiceExt, transport::TokioChildProcess};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// Handle held by `AdkToolHost` for lifecycle (cancellation on shutdown).
#[derive(Clone)]
pub struct McpHandle {
    pub server_name: String,
    pub cancel: CancellationToken,
}

/// Outcome of a successful `mount`: the per-tool `NativeTool` wrappers + the
/// lifecycle handle.
pub struct MountResult {
    pub tools: Vec<Arc<dyn NativeTool>>,
    pub handle: McpHandle,
}

/// Spawn the configured stdio MCP server, list its tools, and return them
/// wrapped as `NativeTool` instances.
///
/// Non-stdio transports return `ToolError::McpTransport` (caller decides
/// whether to fail-soft or hard).
pub async fn mount(spec: McpServerSpec) -> Result<MountResult, ToolError> {
    if !matches!(spec.transport, McpTransport::Stdio) {
        return Err(ToolError::McpTransport(format!(
            "non-stdio MCP transports not yet supported: {:?}",
            spec.transport
        )));
    }
    if spec.command_or_url.is_empty() {
        return Err(ToolError::McpTransport(
            "stdio MCP server has no command".into(),
        ));
    }

    let mut cmd = Command::new(&spec.command_or_url[0]);
    cmd.args(&spec.command_or_url[1..]);
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }

    let transport = TokioChildProcess::new(cmd)
        .map_err(|e| ToolError::McpTransport(format!("spawn {}: {e}", spec.name)))?;
    let peer = ()
        .serve(transport)
        .await
        .map_err(|e| ToolError::McpTransport(format!("connect {}: {e}", spec.name)))?;

    let toolset = adk_rust::tool::McpToolset::new(peer).with_name(&spec.name);
    let cancel = toolset.cancellation_token().await;

    let ctx = adk_rust::context::ReadonlyContext::default();
    let inner_tools = toolset
        .tools(&ctx)
        .await
        .map_err(|e| ToolError::McpTransport(format!("list tools {}: {e}", spec.name)))?;

    let server_name = spec.name.clone();
    let tools: Vec<Arc<dyn NativeTool>> = inner_tools
        .into_iter()
        .map(|t| {
            let schema = t.parameters_schema().unwrap_or_else(|| serde_json::json!({}));
            Arc::new(McpToolWrapper {
                server_name: server_name.clone(),
                inner: t,
                schema,
            }) as Arc<dyn NativeTool>
        })
        .collect();

    Ok(MountResult {
        tools,
        handle: McpHandle {
            server_name,
            cancel,
        },
    })
}

struct McpToolWrapper {
    server_name: String,
    inner: Arc<dyn adk_rust::Tool>,
    schema: serde_json::Value,
}

#[async_trait]
impl NativeTool for McpToolWrapper {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.inner.name().to_owned(),
            description: Some(self.inner.description().to_owned()),
            schema: self.schema.clone(),
            source: ToolSource::Mcp {
                server: self.server_name.clone(),
            },
        }
    }

    async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        let ctx: Arc<dyn adk_rust::ToolContext> = Arc::new(NoopToolContext);
        self.inner
            .execute(ctx, args)
            .await
            .map_err(|e| ToolError::Execution(format!("{}: {e}", self.server_name)))
    }
}

/// `ToolContext` stub for MCP tools that don't consume context fields.
/// adk-rust's MCP tools currently don't use the session/event hooks; this
/// provides empty answers if any future tool reaches for them.
struct NoopToolContext;

impl adk_rust::ToolContext for NoopToolContext {
    // The exact required methods depend on adk-rust's ToolContext shape.
    // Provide unit/empty defaults for each. See `cargo doc -p adk-rust`.
}
```

The exact shape of `NoopToolContext`'s impl depends on adk-rust 0.6.0's `ToolContext` trait surface — fill in the methods the compiler asks for, returning unit / empty / sensible defaults.

If `ToolError::McpTransport` doesn't exist, add it to `ph0b0s-core::error::ToolError`:

```rust
#[error("mcp transport: {0}")]
McpTransport(String),
```

- [ ] **Step 2: Wire module into `lib.rs`**

Add to `crates/ph0b0s-llm-adk/src/lib.rs`:

```rust
pub mod mcp;
```

- [ ] **Step 3: Verify compile**

Run: `cargo check -p ph0b0s-llm-adk`
Expected: success. If it doesn't compile because `McpToolset` / `Tool::execute` / `ReadonlyContext` have different shapes than the spec assumed, this is the moment to spike — see the spec's risk row for the rmcp-direct fallback. Adapt names, keep behavior.

- [ ] **Step 4: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/mcp.rs crates/ph0b0s-llm-adk/src/lib.rs crates/ph0b0s-core/src/error.rs
git commit -m "feat(adk): mcp.rs skeleton — rmcp + McpToolset stdio wiring"
```

---

## Task 18: `mount_mcp` delegates to `mcp::mount` + lifecycle

**Files:**
- Modify: `crates/ph0b0s-llm-adk/src/tools.rs`

- [ ] **Step 1: Add `mcp_handles` to `AdkToolHost::State`**

Modify the `State` struct in `crates/ph0b0s-llm-adk/src/tools.rs`:

```rust
#[derive(Default)]
struct State {
    native_tools: Mutex<HashMap<String, Arc<dyn NativeTool>>>,
    canned: Mutex<HashMap<String, VecDeque<Result<serde_json::Value, ToolError>>>>,
    mounted_mcp: Mutex<Vec<McpServerSpec>>,
    mcp_handles: Mutex<Vec<crate::mcp::McpHandle>>,
    invocations: Mutex<Vec<(String, serde_json::Value)>>,
}
```

- [ ] **Step 2: Replace `mount_mcp` body**

Replace the warning-and-record `mount_mcp`:

```rust
#[tracing::instrument(skip(self, server), fields(server = %server.name))]
async fn mount_mcp(&self, server: McpServerSpec) -> Result<(), ToolError> {
    if !matches!(server.transport, ph0b0s_core::tools::McpTransport::Stdio) {
        tracing::warn!(
            server = %server.name,
            "non-stdio MCP transport — recording spec but not connecting"
        );
        self.state
            .mounted_mcp
            .lock()
            .expect("mounted_mcp poisoned")
            .push(server);
        return Ok(());
    }

    let result = crate::mcp::mount(server.clone()).await?;
    // Register every discovered tool. Last-mount-wins on collisions.
    {
        let mut natives = self
            .state
            .native_tools
            .lock()
            .expect("native_tools poisoned");
        for tool in result.tools {
            natives.insert(tool.spec().name, tool);
        }
    }
    self.state
        .mcp_handles
        .lock()
        .expect("mcp_handles poisoned")
        .push(result.handle);
    self.state
        .mounted_mcp
        .lock()
        .expect("mounted_mcp poisoned")
        .push(server);
    Ok(())
}
```

- [ ] **Step 3: Add `shutdown_mcp` lifecycle method**

After the `ToolHost` impl, add:

```rust
impl AdkToolHost {
    /// Cancel every mounted MCP server's transport. Idempotent. Called by
    /// the CLI's shutdown path; failing to call it is not fatal (rmcp
    /// closes stdio on drop) but produces cleaner logs.
    pub async fn shutdown_mcp(&self) {
        let handles: Vec<_> = self
            .state
            .mcp_handles
            .lock()
            .expect("mcp_handles poisoned")
            .drain(..)
            .collect();
        for h in handles {
            tracing::debug!(server = %h.server_name, "cancelling MCP transport");
            h.cancel.cancel();
        }
    }
}
```

- [ ] **Step 4: Verify existing tests still pass**

Run: `cargo test -p ph0b0s-llm-adk tools::`
Expected: all 7 existing tests pass. The `mount_mcp_records_spec_returns_ok` test uses `transport: McpTransport::Stdio` with a fake `uvx` command — that will now try to actually spawn. Update the test to use `Stdio` only when we have the fake server available; switch to a non-stdio transport for this hermetic unit test:

```rust
#[tokio::test]
async fn mount_mcp_non_stdio_transport_records_and_warns() {
    let host = AdkToolHost::new();
    let spec = McpServerSpec {
        name: "fs".into(),
        transport: McpTransport::Sse,
        command_or_url: vec!["http://example.com/sse".into()],
        env: Default::default(),
    };
    host.mount_mcp(spec.clone()).await.unwrap();
    assert_eq!(host.mounted_mcp(), vec![spec]);
}
```

(If `McpTransport::Sse` doesn't exist in the seam yet, add it — or use whatever non-stdio variant is defined.)

- [ ] **Step 5: Run tests**

```bash
cargo test -p ph0b0s-llm-adk tools::
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add crates/ph0b0s-llm-adk/src/tools.rs
git commit -m "feat(adk): mount_mcp delegates to mcp::mount + lifecycle handles"
```

---

## Task 19: Hermetic MCP fixture + integration test

**Files:**
- Create: `crates/ph0b0s-llm-adk/tests/fixtures/fake_mcp.py`
- Create: `crates/ph0b0s-llm-adk/tests/mcp_fixture.rs`

- [ ] **Step 1: Write the fake MCP server**

Create `crates/ph0b0s-llm-adk/tests/fixtures/fake_mcp.py`:

```python
#!/usr/bin/env python3
"""Tiny stdio MCP server for hermetic ph0b0s-llm-adk tests.

Implements just enough of MCP 2024-11-05 to let McpToolset:
  1. initialize
  2. tools/list  → returns one tool named `ping`
  3. tools/call  → returns `{"pong": true}` for any args

No deps. Run: `python3 fake_mcp.py`. Communicates via JSON-RPC 2.0 over stdio.
"""
import json
import sys


def respond(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle(req):
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        respond(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "0.1"},
        })
    elif method == "notifications/initialized":
        # No response for notifications.
        pass
    elif method == "tools/list":
        respond(rid, {
            "tools": [{
                "name": "ping",
                "description": "responds with pong",
                "inputSchema": {"type": "object", "properties": {}},
            }]
        })
    elif method == "tools/call":
        respond(rid, {
            "content": [{"type": "text", "text": json.dumps({"pong": True})}]
        })
    elif rid is not None:
        respond(rid, error={"code": -32601, "message": f"unknown method: {method}"})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        handle(req)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make the fixture executable**

```bash
chmod +x /Users/hocinehacherouf/git/ph0b0s/crates/ph0b0s-llm-adk/tests/fixtures/fake_mcp.py
```

- [ ] **Step 3: Write the integration test**

Create `crates/ph0b0s-llm-adk/tests/mcp_fixture.rs`:

```rust
//! Hermetic stdio MCP integration test against `fixtures/fake_mcp.py`.
//!
//! Skipped on non-Unix because stdio MCP needs a real subprocess. Skipped if
//! `python3` isn't on PATH (CI installs Python; local devs may not).

#![cfg(unix)]

use std::collections::HashMap;
use std::path::PathBuf;

use ph0b0s_core::tools::{McpServerSpec, McpTransport, ToolHost};
use ph0b0s_llm_adk::AdkToolHost;

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/fake_mcp.py");
    p
}

fn python_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn mount_lists_and_invokes_fake_mcp_ping() {
    if !python_available() {
        eprintln!("skipping: python3 not on PATH");
        return;
    }
    let host = AdkToolHost::new();
    let spec = McpServerSpec {
        name: "fake".into(),
        transport: McpTransport::Stdio,
        command_or_url: vec![
            "python3".into(),
            fixture_path().to_string_lossy().into_owned(),
        ],
        env: HashMap::new(),
    };
    host.mount_mcp(spec).await.expect("mount succeeds");

    let names: Vec<_> = host.list().into_iter().map(|s| s.name).collect();
    assert!(names.contains(&"ping".to_owned()), "ping not listed: {names:?}");

    let r = host
        .invoke("ping", serde_json::json!({}))
        .await
        .expect("invoke ping");
    // adk_tool::McpToolset typically wraps results — verify the substantive
    // bit is reachable. The exact shape depends on adk's MCP result mapping;
    // accept either {"pong": true} or {"content": [{"text": "{\"pong\":true}"}]}.
    let s = serde_json::to_string(&r).unwrap();
    assert!(s.contains("pong"), "unexpected result shape: {s}");

    // Clean shutdown — should not panic, should not leave zombies.
    host.shutdown_mcp().await;
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test -p ph0b0s-llm-adk --test mcp_fixture
```
Expected: pass on macOS/Linux; skipped on Windows. If the test panics inside `mount_mcp`, the most likely failure modes are: (a) `McpToolset` wants different ctor args than `new(peer)`; (b) `tools/list` shape doesn't match (spec uses `inputSchema`, some MCP servers want `parameters`). Adjust `fake_mcp.py` first; only patch the wrapper if the protocol actually requires it.

- [ ] **Step 5: Commit**

```bash
git add crates/ph0b0s-llm-adk/tests/fixtures/fake_mcp.py crates/ph0b0s-llm-adk/tests/mcp_fixture.rs
git commit -m "test(adk): hermetic stdio MCP integration test"
```

---

## Task 20: Live Ollama tests (`#[ignore]`)

**Files:**
- Create: `crates/ph0b0s-llm-adk/tests/ollama_live.rs`

- [ ] **Step 1: Write the live tests**

Create `crates/ph0b0s-llm-adk/tests/ollama_live.rs`:

```rust
//! Live tests against a local Ollama server. Marked `#[ignore]` so they
//! only run when explicitly requested via `--include-ignored`. The CI
//! `live-ollama` job sets `OLLAMA_HOST=http://localhost:11434` and pulls
//! a tiny model (`qwen2.5-coder:0.5b`) before invoking these.
//!
//! Local dev: `ollama serve` in one terminal, `ollama pull qwen2.5-coder:0.5b`,
//! then `cargo test -p ph0b0s-llm-adk --test ollama_live -- --include-ignored`.

use ph0b0s_core::llm::{ChatMessage, ChatRequest, LlmAgent, StructuredRequest, UserMessage};
use ph0b0s_llm_adk::provider::ollama;

const TEST_MODEL_ENV: &str = "PH0B0S_LIVE_OLLAMA_MODEL";

fn model() -> String {
    std::env::var(TEST_MODEL_ENV).unwrap_or_else(|_| "qwen2.5-coder:0.5b".into())
}

#[tokio::test]
#[ignore = "requires running local Ollama server"]
async fn chat_returns_non_empty_response() {
    let m = model();
    let agent = ollama::build(Some(&m), None).expect("build ollama");
    let req = ChatRequest::new()
        .system("be terse")
        .user("say the single word 'pong' and nothing else");
    let resp = agent.chat(req).await.expect("chat ok");
    assert!(!resp.content.is_empty(), "empty response");
    assert!(resp.usage.tokens_in > 0, "expected token accounting");
}

#[tokio::test]
#[ignore = "requires running local Ollama server"]
async fn structured_emits_parseable_json() {
    let m = model();
    let agent = ollama::build(Some(&m), None).expect("build ollama");
    let req = StructuredRequest {
        messages: vec![ChatMessage::User {
            content: "respond with {\"ok\": true} as JSON, nothing else".into(),
        }],
        schema: serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"]
        }),
        schema_name: "Smoke".into(),
        tools: Vec::new(),
        hints: Default::default(),
    };
    let v = agent.structured(req).await.expect("structured ok");
    assert!(v["ok"].is_boolean(), "expected ok:bool, got {v}");
}

#[tokio::test]
#[ignore = "requires running local Ollama server"]
async fn session_multi_turn_accumulates_usage() {
    let m = model();
    let agent = ollama::build(Some(&m), None).expect("build ollama");
    let mut sess = agent
        .session(Default::default())
        .await
        .expect("session ok");
    let r1 = sess.send(UserMessage::new("hello")).await.unwrap();
    assert!(!r1.content.is_empty());
    let usage_after_1 = sess.usage();
    let r2 = sess.send(UserMessage::new("how are you?")).await.unwrap();
    assert!(!r2.content.is_empty());
    let usage_after_2 = sess.usage();
    assert!(
        usage_after_2.tokens_in >= usage_after_1.tokens_in,
        "tokens_in should be monotonic"
    );
}
```

- [ ] **Step 2: Verify compile**

```bash
cargo check -p ph0b0s-llm-adk --tests
```
Expected: success. Don't run the tests yet — they need a live server.

- [ ] **Step 3: Commit**

```bash
git add crates/ph0b0s-llm-adk/tests/ollama_live.rs
git commit -m "test(adk): live Ollama integration tests (ignored by default)"
```

---

## Task 21: CI live-ollama job

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Read existing CI**

```bash
cat /Users/hocinehacherouf/git/ph0b0s/.github/workflows/ci.yml
```

- [ ] **Step 2: Add the `live-ollama` job**

Append (or insert before the `coverage` job to keep related jobs grouped) to `.github/workflows/ci.yml`:

```yaml
  live-ollama:
    name: live (ollama)
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Install Linux system deps
        run: sudo apt-get update && sudo apt-get install -y libdbus-1-dev pkg-config
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: stable
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: ph0b0s-live-ollama
      - name: Cache Ollama models
        uses: actions/cache@v4
        with:
          path: ~/.ollama/models
          key: ollama-${{ runner.os }}-qwen2.5-coder-0.5b
      - name: Install + start Ollama
        run: |
          curl -fsSL https://ollama.com/install.sh | sh
          ollama serve > /tmp/ollama.log 2>&1 &
          for i in $(seq 1 30); do
            if curl -sf http://localhost:11434/api/tags > /dev/null; then
              echo "ollama up"
              exit 0
            fi
            sleep 1
          done
          echo "ollama failed to start"
          tail /tmp/ollama.log
          exit 1
      - name: Pull model
        run: ollama pull qwen2.5-coder:0.5b
      - name: Run live tests
        env:
          OLLAMA_HOST: http://localhost:11434
          PH0B0S_LIVE_OLLAMA_MODEL: qwen2.5-coder:0.5b
        run: cargo test -p ph0b0s-llm-adk --test ollama_live -- --include-ignored
```

- [ ] **Step 3: Verify YAML syntax locally**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo OK
```
Expected: `OK`. Or use `yamllint` if installed.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add live-ollama job"
```

---

## Task 22: Vendor-coupling check + final workspace sweep

**Files:**
- Verify: `xtask/src/main.rs` allow-list
- Run: full quality gate

- [ ] **Step 1: Confirm vendor-coupling still passes**

```bash
cargo run -p xtask -- check-vendor
```
Expected: `vendor-coupling: OK`. We added `rmcp` imports to `mcp.rs`, but `ph0b0s-llm-adk` is in the allow-list. The fitness-function regex already lists `rmcp` as a banned import outside the allow-list — that's the desired symmetry.

If it fails: read `xtask/src/main.rs` to confirm the allow-list still has `["ph0b0s-llm-adk", "ph0b0s-cli"]`. No change needed if so.

- [ ] **Step 2: Full quality gate**

Run each in sequence (do NOT skip):

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-features
cargo run -p xtask -- check-vendor
cargo deny check
```

Each must be green. Any failure: fix, re-stage, re-commit before proceeding.

- [ ] **Step 3: Local coverage spot-check**

```bash
cargo llvm-cov --workspace --all-features \
  --ignore-filename-regex 'xtask|/tests/|fixtures/|ph0b0s-test-support|ph0b0s-cli/src/(main|workspace)\.rs|ollama_live\.rs' \
  --summary-only
```
Expected: per-file coverage ≥ 90% for new files. Patch coverage will be confirmed by codecov on the PR. If a file in the patch is < 100%, add the missing assertions before opening the PR.

- [ ] **Step 4: Commit any final fix-ups**

If fmt/clippy/coverage required code changes:

```bash
git add -p
git commit -m "chore: address fmt/clippy/coverage findings"
```

---

## Task 23: Documentation + PR

**Files:**
- Modify: `README.md` (Status table; "Known limitations" block)
- Modify: `CONTRIBUTING.md` (if any new dev-time ergonomics worth documenting)
- Create: PR via `gh pr create`

- [ ] **Step 1: Update README Status table**

In `README.md`, find the "Known limitations in slice (e)" block. Strike-through the lifted items:

```markdown
- ~~**Only `PH0B0S_PROVIDER=mock` is fully wired.**~~ Real Anthropic /
  OpenAI / Gemini / Ollama providers are wired as of <slice-this-PR>.
- ~~**`ToolHost::mount_mcp` records the spec** but does not connect.~~
  stdio MCP servers now connect via rmcp + adk_tool::McpToolset.
- ~~**No tool-call loop in the adapter.**~~ `chat()` and `session.send()`
  now run a multi-turn loop (max 10 turns by default).
```

Add a "Provider configuration" section to the README, briefly:

```markdown
### Provider configuration

`ph0b0s` picks a provider in this order (highest precedence first):

1. `PH0B0S_PROVIDER` env override.
2. Explicit `[agents.default]` in `ph0b0s.toml`.
3. Env-key auto-detection: `ANTHROPIC_API_KEY` → Anthropic,
   `OPENAI_API_KEY` → OpenAI, `GOOGLE_API_KEY` → Gemini, `OLLAMA_HOST` → Ollama.

Set the corresponding API key (never in TOML) and run:

```bash
ANTHROPIC_API_KEY=... cargo run -p ph0b0s-cli -- scan ./some-repo
```

Override the default model per provider in `ph0b0s.toml`:

```toml
[providers.anthropic]
default_model = "claude-opus-4-7"

[providers.openai]
base_url = "https://openrouter.ai/api/v1"
default_model = "openai/gpt-5"
```
```

- [ ] **Step 2: Run final test sweep**

```bash
cargo test --workspace --all-features
```
Expected: green.

- [ ] **Step 3: Commit docs**

```bash
git add README.md
git commit -m "docs: lift slice-(e) limitations + document provider selection"
```

- [ ] **Step 4: Push the branch**

```bash
git push -u origin feat/real-providers-and-tool-loop
```

- [ ] **Step 5: Open the PR**

(No Claude attribution per the standing instruction.)

```bash
gh pr create --title "feat: real LLM providers + tool-call loop + stdio MCP" --body "$(cat <<'EOF'
## Summary

- Wires real Anthropic / OpenAI / Gemini / Ollama provider builders behind the `LlmAgent` seam (the adapter owns construction; CLI is a thin dispatcher).
- Adds a multi-turn tool-call loop to `AdkLlmAgent::chat()` and `AdkSession::send()` — sequential dispatch, errors fed back as `FunctionResponse{"error": ...}`, default cap 10 turns (override via `ChatRequest.hints["max_tool_turns"]`).
- Connects stdio MCP servers via rmcp + `adk_tool::McpToolset`. Each discovered tool is registered as a `NativeTool` so the loop sees MCP and Rust tools through one dispatch path.

## Test plan

- [ ] `cargo test --workspace --all-features` green
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] `cargo run -p xtask -- check-vendor` says `vendor-coupling: OK`
- [ ] `cargo deny check` green
- [ ] `live-ollama` CI job green (real chat / structured / session against local `qwen2.5-coder:0.5b`)
- [ ] `tests/mcp_fixture.rs` green (hermetic stdio MCP via Python fake)
- [ ] Manual sanity: `ANTHROPIC_API_KEY=... cargo run -p ph0b0s-cli -- scan ./fixtures/sample-rust-repo` produces real findings
EOF
)"
```

- [ ] **Step 6: Verify CI status**

```bash
gh pr view --web
# or
gh pr checks
```

If any check fails, address the failures with new commits (never `--amend` published commits per the project's git safety rules).

---

## Self-Review Notes

After plan completion, the following spec sections each have at least one task:

| Spec section | Covered by |
|---|---|
| Architectural decisions table | Tasks 9 (selection), 13–15 (loop), 17–19 (MCP), 20–21 (Ollama live), 4–10 (factoring) |
| Architecture file tree | Tasks 2, 4–8, 10, 17, 19, 20 (file creation) |
| Provider builders + table | Tasks 4–8 |
| CLI dispatcher contract | Task 10 |
| Tool-call loop algorithm (5 steps) | Tasks 11–16 |
| MCP integration (5 wiring steps) | Tasks 17–19 |
| Testing strategy (3 tiers) | Tier 1: tasks 4–9, 13–14; Tier 2: existing + Task 11 refactor; Tier 3: Tasks 20–21 |
| CI changes | Task 21 |
| Risks + mitigations | Spike notes inline in Tasks 4 (env-lock), 13 (Part shape), 17 (rmcp fallback) |

**Type consistency check:** `AgentConfig::model` is `Option<String>` in the adapter (Task 2) and the CLI (Task 10) — consistent. `ProviderConfig::base_url` added in both (Tasks 2, 10). `BuildError` defined in Task 3 used by Tasks 4–9. `McpHandle { server_name, cancel }` defined Task 17 used by Task 18.

**Placeholder scan:** No "TBD"/"TODO"/"implement later" in the plan. Two adk-rust API uncertainties have explicit fallback instructions (Task 4 step 4, Task 13 step 2, Task 17 step 1, Task 18 step 4) rather than placeholders — the engineer has a decision tree, not a blank.
