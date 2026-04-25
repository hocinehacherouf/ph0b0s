//! `Finding` is the unified domain model produced by every detector and
//! consumed by the store + reporters. The fingerprint is load-bearing for
//! cross-run dedup and triage memory.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ulid::Ulid;

use crate::severity::Severity;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Location {
    /// File-and-line region inside the workspace, expressed relative to the
    /// workspace root so it round-trips across machines.
    File {
        path: String,
        start_line: u32,
        end_line: u32,
        start_col: Option<u32>,
        end_col: Option<u32>,
    },
    /// Symbolic reference (e.g. a vulnerable package version) — used by SCA
    /// detectors that don't have a single file location.
    Symbolic {
        package: String,
        version: String,
        ecosystem: String, // "crates.io", "pypi", "npm", ...
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Evidence {
    /// Free-form JSON blob (an upstream tool's raw record, etc.).
    Json(serde_json::Value),
    /// A short captured code snippet.
    Snippet { language: Option<String>, text: String },
    /// A reference to an artifact stored on disk relative to the workspace.
    Artifact { path: String, mime: Option<String> },
    /// A short text note from the detector.
    Note(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SanitizationState {
    Raw,
    Sanitized { rules: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuppressionHint {
    pub reason: String,
    /// `true` if the suppression is a hard rule from config (`[[suppress]]`),
    /// `false` if it's an advisory / heuristic.
    pub hard: bool,
}

/// SHA-256 hex prefix over a stable canonical form. Used as the dedup key.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Fingerprint(pub String);

impl Fingerprint {
    /// Compute a fingerprint over the rule, the normalised location, and a
    /// canonical evidence subset. Detectors should use this so equivalent
    /// findings on subsequent runs collapse cleanly.
    pub fn compute(
        rule_id: &str,
        location: &Location,
        canonical_evidence: &[u8],
    ) -> Self {
        let mut h = Sha256::new();
        h.update(rule_id.as_bytes());
        h.update(b"\x1f");
        h.update(canonical_location(location).as_bytes());
        h.update(b"\x1f");
        h.update(canonical_evidence);
        let digest = h.finalize();
        Fingerprint(hex::encode(&digest[..16])) // 128-bit prefix is plenty
    }
}

fn canonical_location(loc: &Location) -> String {
    match loc {
        Location::File { path, start_line, end_line, .. } => {
            // Normalize separators to `/` so the same location on Windows and
            // Unix produces the same fingerprint.
            let p = path.replace('\\', "/");
            format!("file:{}#{}-{}", p, start_line, end_line)
        }
        Location::Symbolic { package, version, ecosystem } => {
            format!("pkg:{ecosystem}:{package}@{version}")
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    pub id: Ulid,
    pub rule_id: String,
    pub detector: String,
    pub severity: Severity,
    pub confidence: Confidence,
    pub title: String,
    pub message: String,
    pub location: Location,
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    pub fingerprint: Fingerprint,
    pub sanitization: SanitizationState,
    #[serde(default)]
    pub suppressions: Vec<SuppressionHint>,
    pub created_at: DateTime<Utc>,
}

impl Finding {
    /// Convenience constructor that fills `id`, `created_at`, and computes
    /// the fingerprint from `rule_id` + `location` + a canonical-evidence blob
    /// the caller produces.
    pub fn new(
        detector: impl Into<String>,
        rule_id: impl Into<String>,
        title: impl Into<String>,
        message: impl Into<String>,
        location: Location,
        severity: Severity,
        confidence: Confidence,
        canonical_evidence: &[u8],
    ) -> Self {
        let rule_id = rule_id.into();
        let fingerprint = Fingerprint::compute(&rule_id, &location, canonical_evidence);
        Self {
            id: Ulid::new(),
            rule_id,
            detector: detector.into(),
            severity,
            confidence,
            title: title.into(),
            message: message.into(),
            location,
            evidence: Vec::new(),
            fingerprint,
            sanitization: SanitizationState::Raw,
            suppressions: Vec::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_evidence(mut self, ev: Evidence) -> Self {
        self.evidence.push(ev);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::severity::Level;

    #[test]
    fn fingerprint_is_stable_across_calls() {
        let loc = Location::File {
            path: "src/lib.rs".into(),
            start_line: 10,
            end_line: 12,
            start_col: None,
            end_col: None,
        };
        let a = Fingerprint::compute("rule.x", &loc, b"e1");
        let b = Fingerprint::compute("rule.x", &loc, b"e1");
        assert_eq!(a, b);
        assert_eq!(a.0.len(), 32); // 16 bytes hex-encoded
    }

    #[test]
    fn fingerprint_changes_with_rule_id() {
        let loc = Location::File {
            path: "src/lib.rs".into(),
            start_line: 10,
            end_line: 12,
            start_col: None,
            end_col: None,
        };
        let a = Fingerprint::compute("rule.x", &loc, b"e1");
        let b = Fingerprint::compute("rule.y", &loc, b"e1");
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_changes_with_location() {
        let l1 = Location::File {
            path: "src/lib.rs".into(),
            start_line: 10,
            end_line: 12,
            start_col: None,
            end_col: None,
        };
        let l2 = Location::File {
            path: "src/lib.rs".into(),
            start_line: 11,
            end_line: 13,
            start_col: None,
            end_col: None,
        };
        let a = Fingerprint::compute("r", &l1, b"");
        let b = Fingerprint::compute("r", &l2, b"");
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_changes_with_evidence() {
        let loc = Location::File {
            path: "src/lib.rs".into(),
            start_line: 10,
            end_line: 12,
            start_col: None,
            end_col: None,
        };
        let a = Fingerprint::compute("r", &loc, b"e1");
        let b = Fingerprint::compute("r", &loc, b"e2");
        assert_ne!(a, b);
    }

    #[test]
    fn fingerprint_normalises_path_separators() {
        let unix = Location::File {
            path: "src/sub/lib.rs".into(),
            start_line: 1,
            end_line: 1,
            start_col: None,
            end_col: None,
        };
        let win = Location::File {
            path: "src\\sub\\lib.rs".into(),
            start_line: 1,
            end_line: 1,
            start_col: None,
            end_col: None,
        };
        assert_eq!(
            Fingerprint::compute("r", &unix, b""),
            Fingerprint::compute("r", &win, b"")
        );
    }

    #[test]
    fn fingerprint_distinguishes_symbolic_packages() {
        let l1 = Location::Symbolic {
            package: "openssl".into(),
            version: "0.10.0".into(),
            ecosystem: "crates.io".into(),
        };
        let l2 = Location::Symbolic {
            package: "openssl".into(),
            version: "0.10.1".into(),
            ecosystem: "crates.io".into(),
        };
        assert_ne!(
            Fingerprint::compute("r", &l1, b""),
            Fingerprint::compute("r", &l2, b"")
        );
    }

    #[test]
    fn finding_round_trips_via_serde() {
        let f = Finding::new(
            "llm-toy",
            "ph0b0s.llm-toy.example",
            "Hardcoded password",
            "found a string literal that looks like a credential",
            Location::File {
                path: "src/main.rs".into(),
                start_line: 12,
                end_line: 12,
                start_col: Some(8),
                end_col: Some(20),
            },
            Severity::Qualitative(Level::High),
            Confidence::Medium,
            b"pwd=hunter2",
        );
        let j = serde_json::to_string(&f).unwrap();
        let f2: Finding = serde_json::from_str(&j).unwrap();
        assert_eq!(f, f2);
    }
}
