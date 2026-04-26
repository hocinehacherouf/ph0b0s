//! Layered configuration loaded by `figment`.
//!
//! Layers (lowest precedence first):
//!   1. compiled-in defaults
//!   2. `~/.config/ph0b0s/config.toml`             (if present)
//!   3. `./ph0b0s.toml`                            (if present)
//!   4. environment variables prefixed with `PH0B0S__`
//!
//! API keys are NEVER read from a TOML file. They come from canonical env
//! vars (`ANTHROPIC_API_KEY`, etc.) or from an OS keyring. `config check`
//! refuses to run if it sees an `api_key` key anywhere in the merged TOML.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use ph0b0s_core::tools::McpServerSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("figment: {0}")]
    Figment(String),
    #[error("api_key found in TOML file (not allowed); read keys from env vars only")]
    ApiKeyInToml,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// The full, merged config used by every CLI subcommand.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub scan: ScanConfig,
    /// Provider definitions keyed by name (`"anthropic"`, `"openai"`, ...).
    pub providers: HashMap<String, ProviderConfig>,
    /// Per-role agent assignments (`"reasoner" -> { provider, model }`).
    pub agents: HashMap<String, AgentConfig>,
    /// Stdio/HTTP MCP servers to mount at startup.
    #[serde(rename = "mcp_servers")]
    pub mcp_servers: Vec<McpServerSpec>,
    /// Per-detector params keyed by detector id. `BTreeMap` for stable
    /// iteration order (matches `ScanRequest::detector_params`).
    pub detectors: BTreeMap<String, serde_json::Value>,
    /// Hard-rule suppressions from config (a `[[suppress]]` array).
    pub suppress: Vec<SuppressRule>,
    pub output: OutputConfig,
    pub storage: StorageConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ScanConfig {
    pub max_parallel: usize,
    pub detector_timeout_s: u64,
    pub strict: bool,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            max_parallel: default_parallelism(),
            detector_timeout_s: 300,
            strict: false,
        }
    }
}

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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub sarif_path: Option<PathBuf>,
    pub markdown_path: Option<PathBuf>,
    pub json_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Path to the SQLite findings DB. `None` ⇒ XDG state dir.
    pub db_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuppressRule {
    pub rule_id: String,
    pub reason: String,
    #[serde(default)]
    pub expires_at: Option<DateTime<Utc>>,
}

fn default_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 4))
        .unwrap_or(2)
}

impl Config {
    /// Load and merge config from compiled-in defaults, `~/.config/ph0b0s/config.toml`,
    /// `./ph0b0s.toml`, and `PH0B0S__` env vars.
    pub fn load() -> Result<Self, ConfigError> {
        let user = user_config_path();
        Self::load_from(user.as_deref(), Some(Path::new("ph0b0s.toml")))
    }

    /// Test-friendly variant: explicit user + project paths.
    pub fn load_from(user: Option<&Path>, project: Option<&Path>) -> Result<Self, ConfigError> {
        let mut fig = Figment::from(Serialized::defaults(Config::default()));
        if let Some(p) = user {
            if p.exists() {
                check_no_api_key(p)?;
                fig = fig.merge(Toml::file(p));
            }
        }
        if let Some(p) = project {
            if p.exists() {
                check_no_api_key(p)?;
                fig = fig.merge(Toml::file(p));
            }
        }
        fig = fig.merge(Env::prefixed("PH0B0S__").split("__"));
        fig.extract()
            .map_err(|e| ConfigError::Figment(e.to_string()))
    }

    /// Effective DB path: `storage.db_path` or
    /// `$XDG_STATE_HOME/ph0b0s/findings.db` (or `~/.local/state/ph0b0s/findings.db`).
    pub fn effective_db_path(&self) -> PathBuf {
        if let Some(p) = &self.storage.db_path {
            return p.clone();
        }
        let base = std::env::var("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                PathBuf::from(home).join(".local").join("state")
            });
        base.join("ph0b0s").join("findings.db")
    }

    /// Effective config with secrets redacted, suitable for `config check`
    /// stdout. Returns canonical JSON.
    pub fn redacted_json(&self) -> serde_json::Value {
        // Only providers/api_keys live as env vars; the struct doesn't carry
        // them. Still, scrub any field literally named api_key just in case
        // an extension serialised one in.
        let mut value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        scrub_api_keys(&mut value);
        value
    }
}

fn user_config_path() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("ph0b0s").join("config.toml"));
    }
    let home = std::env::var("HOME").ok()?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("ph0b0s")
            .join("config.toml"),
    )
}

fn check_no_api_key(path: &Path) -> Result<(), ConfigError> {
    let text = std::fs::read_to_string(path)?;
    // Match `api_key` as a TOML key; deliberately permissive.
    let needle = "api_key";
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some(idx) = trimmed.find(needle) {
            // Only fail if it's at the start of a key (not embedded in a
            // longer identifier or in a string value).
            let prev = trimmed.get(..idx).and_then(|s| s.chars().last());
            let after = trimmed
                .get(idx + needle.len()..)
                .and_then(|s| s.chars().next());
            let is_key_start = prev
                .map(|c| !c.is_alphanumeric() && c != '_')
                .unwrap_or(true);
            let is_assignment = matches!(after, Some('=') | Some(' ') | Some('\t'));
            if is_key_start && is_assignment {
                return Err(ConfigError::ApiKeyInToml);
            }
        }
    }
    Ok(())
}

fn scrub_api_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if k.to_ascii_lowercase().contains("api_key")
                    || k.to_ascii_lowercase().contains("apikey")
                {
                    *v = serde_json::Value::String("<redacted>".into());
                } else {
                    scrub_api_keys(v);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr.iter_mut() {
                scrub_api_keys(v);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(name: &str, body: &str) -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join(name), body).unwrap();
        td
    }

    #[test]
    fn defaults_load_when_no_files() {
        let cfg = Config::load_from(None, None).unwrap();
        assert!(cfg.scan.max_parallel >= 1);
        assert!(!cfg.scan.strict);
    }

    #[test]
    fn project_toml_overrides_defaults() {
        let td = write_tmp(
            "ph0b0s.toml",
            r#"[scan]
max_parallel = 8
strict = true
"#,
        );
        let cfg = Config::load_from(None, Some(&td.path().join("ph0b0s.toml"))).unwrap();
        assert_eq!(cfg.scan.max_parallel, 8);
        assert!(cfg.scan.strict);
    }

    #[test]
    fn api_key_in_toml_is_rejected() {
        let td = write_tmp(
            "ph0b0s.toml",
            r#"[providers.anthropic]
api_key = "sk-..."
"#,
        );
        let err = Config::load_from(None, Some(&td.path().join("ph0b0s.toml"))).unwrap_err();
        assert!(matches!(err, ConfigError::ApiKeyInToml));
    }

    #[test]
    fn comments_with_api_key_are_allowed() {
        let td = write_tmp(
            "ph0b0s.toml",
            r#"# Reminder: api_key goes in env var, not here.
[providers.anthropic]
default_model = "claude-sonnet-4-6"
"#,
        );
        let cfg = Config::load_from(None, Some(&td.path().join("ph0b0s.toml"))).unwrap();
        assert_eq!(
            cfg.providers["anthropic"].default_model,
            Some("claude-sonnet-4-6".into())
        );
    }

    #[test]
    fn redacted_json_replaces_obvious_api_key_fields() {
        let mut v = serde_json::json!({"providers":{"anthropic":{"api_key":"secret"}}});
        scrub_api_keys(&mut v);
        assert_eq!(v["providers"]["anthropic"]["api_key"], "<redacted>");
    }
}
