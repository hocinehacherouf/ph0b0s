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
    pub openai: Option<ProviderConfig>,
    pub gemini: Option<ProviderConfig>,
    pub ollama: Option<ProviderConfig>,
}

impl ProviderRegistry {
    /// Look up the per-provider model override (if any).
    pub fn model_for(&self, provider: &str) -> Option<&str> {
        self.get(provider).and_then(|p| p.default_model.as_deref())
    }

    /// Look up the per-provider base_url override (if any).
    pub fn base_url_for(&self, provider: &str) -> Option<&str> {
        self.get(provider).and_then(|p| p.base_url.as_deref())
    }

    fn get(&self, provider: &str) -> Option<&ProviderConfig> {
        match provider {
            "anthropic" => self.anthropic.as_ref(),
            "openai" => self.openai.as_ref(),
            "gemini" => self.gemini.as_ref(),
            "ollama" => self.ollama.as_ref(),
            _ => None,
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
