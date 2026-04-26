//! Static detector registry. Detection packs are statically compiled in; no
//! dynamic plugin loading in v1.

use ph0b0s_core::detector::Detector;

use crate::config::Config;

/// (detector, params) pair returned to the orchestrator.
pub struct ResolvedDetector {
    pub detector: Box<dyn Detector>,
    pub params: serde_json::Value,
}

pub struct DetectorRegistry {
    factories: Vec<DetectorFactory>,
}

struct DetectorFactory {
    id: &'static str,
    build: fn() -> Box<dyn Detector>,
}

impl DetectorRegistry {
    pub fn builtin() -> Self {
        Self {
            factories: vec![
                DetectorFactory {
                    id: "cargo-audit",
                    build: ph0b0s_detect_cargo_audit::detector,
                },
                DetectorFactory {
                    id: "llm-toy",
                    build: ph0b0s_detect_llm_toy::detector,
                },
            ],
        }
    }

    /// All built-in ids, in registration order.
    pub fn ids(&self) -> Vec<&'static str> {
        self.factories.iter().map(|f| f.id).collect()
    }

    /// Resolve the set of enabled detectors for the run.
    ///
    /// `filter`: explicit `--detector` ids from the CLI, or empty for "all enabled".
    /// `config`: per-detector params plus `enabled` flag in `[detectors.<id>]`.
    pub fn resolve(&self, filter: &[String], config: &Config) -> Vec<ResolvedDetector> {
        let mut out = Vec::new();
        for f in &self.factories {
            if !filter.is_empty() && !filter.iter().any(|x| x == f.id) {
                continue;
            }
            // Check `enabled` flag under `[detectors.<id>]`. Default true if
            // the section is absent. If `enabled` is explicitly false, skip
            // unless --detector named the detector explicitly.
            let params = config
                .detectors
                .get(f.id)
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let explicit_request = !filter.is_empty() && filter.iter().any(|x| x == f.id);
            if !explicit_request && is_disabled(&params) {
                continue;
            }
            out.push(ResolvedDetector {
                detector: (f.build)(),
                params: strip_enabled_field(params),
            });
        }
        out
    }
}

fn is_disabled(v: &serde_json::Value) -> bool {
    v.get("enabled")
        .and_then(|x| x.as_bool())
        .map(|b| !b)
        .unwrap_or(false)
}

/// Detector-side params don't know about `enabled`; remove it before passing
/// the value into `Detector::run`'s `ctx.params`.
fn strip_enabled_field(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        obj.remove("enabled");
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_detector(id: &str, params: serde_json::Value) -> Config {
        let mut cfg = Config::default();
        cfg.detectors.insert(id.into(), params);
        cfg
    }

    #[test]
    fn builtin_registry_lists_both_smoke_detectors() {
        let r = DetectorRegistry::builtin();
        let ids = r.ids();
        assert!(ids.contains(&"cargo-audit"));
        assert!(ids.contains(&"llm-toy"));
    }

    #[test]
    fn empty_filter_returns_all_enabled() {
        let r = DetectorRegistry::builtin();
        let resolved = r.resolve(&[], &Config::default());
        let ids: Vec<_> = resolved.iter().map(|r| r.detector.metadata().id).collect();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn explicit_filter_picks_one() {
        let r = DetectorRegistry::builtin();
        let resolved = r.resolve(&["llm-toy".into()], &Config::default());
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].detector.metadata().id, "llm-toy");
    }

    #[test]
    fn enabled_false_skips_when_no_explicit_filter() {
        let cfg = cfg_with_detector("llm-toy", serde_json::json!({"enabled": false}));
        let r = DetectorRegistry::builtin();
        let resolved = r.resolve(&[], &cfg);
        let ids: Vec<_> = resolved.iter().map(|r| r.detector.metadata().id).collect();
        assert_eq!(ids, vec!["cargo-audit"]);
    }

    #[test]
    fn explicit_filter_overrides_enabled_false() {
        let cfg = cfg_with_detector("llm-toy", serde_json::json!({"enabled": false}));
        let r = DetectorRegistry::builtin();
        let resolved = r.resolve(&["llm-toy".into()], &cfg);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].detector.metadata().id, "llm-toy");
    }

    #[test]
    fn enabled_field_is_stripped_from_params() {
        let cfg = cfg_with_detector(
            "llm-toy",
            serde_json::json!({"enabled": true, "max_files": 5}),
        );
        let r = DetectorRegistry::builtin();
        let resolved = r.resolve(&["llm-toy".into()], &cfg);
        let params = &resolved[0].params;
        assert!(params.get("enabled").is_none());
        assert_eq!(params["max_files"], serde_json::json!(5));
    }
}
