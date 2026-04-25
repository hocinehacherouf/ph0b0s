//! Severity is pure data. Detectors that produce CVSS strings own the parse;
//! we only carry the canonical numeric (0.0–10.0) and qualitative bucket.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Severity {
    /// CVSS 3.1 vector + numeric score precomputed by the detector.
    Cvss31 { vector: String, score: f32 },
    /// CVSS 4.0 vector + numeric score precomputed by the detector.
    Cvss40 { vector: String, score: f32 },
    /// Detector has no numeric — just a qualitative bucket.
    Qualitative(Level),
}

impl Severity {
    /// Canonical 0.0–10.0 numeric for SARIF `properties.security-severity`
    /// and for ranking.
    pub fn numeric(&self) -> f32 {
        match self {
            Severity::Cvss31 { score, .. } | Severity::Cvss40 { score, .. } => {
                score.clamp(0.0, 10.0)
            }
            Severity::Qualitative(level) => match level {
                Level::None => 0.0,
                Level::Low => 3.0,
                Level::Medium => 5.5,
                Level::High => 7.5,
                Level::Critical => 9.5,
            },
        }
    }

    /// Map the numeric score back to a qualitative bucket.
    /// Bands match CVSS 3.1 §5.
    pub fn qualitative_bucket(&self) -> Level {
        match self {
            Severity::Qualitative(level) => *level,
            _ => Self::bucket_from_numeric(self.numeric()),
        }
    }

    pub fn bucket_from_numeric(n: f32) -> Level {
        if n <= 0.0 {
            Level::None
        } else if n < 4.0 {
            Level::Low
        } else if n < 7.0 {
            Level::Medium
        } else if n < 9.0 {
            Level::High
        } else {
            Level::Critical
        }
    }

    /// SARIF `result.level` (`error`/`warning`/`note`) for a finding.
    pub fn sarif_level(&self) -> &'static str {
        match self.qualitative_bucket() {
            Level::Critical | Level::High => "error",
            Level::Medium => "warning",
            Level::Low | Level::None => "note",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualitative_buckets_have_stable_numerics() {
        assert_eq!(Severity::Qualitative(Level::None).numeric(), 0.0);
        assert!(Severity::Qualitative(Level::Low).numeric() < 4.0);
        assert!(Severity::Qualitative(Level::Medium).numeric() >= 4.0);
        assert!(Severity::Qualitative(Level::High).numeric() >= 7.0);
        assert!(Severity::Qualitative(Level::Critical).numeric() >= 9.0);
    }

    #[test]
    fn cvss_score_is_clamped() {
        let s = Severity::Cvss31 {
            vector: "AV:N/AC:L/Au:N/C:P/I:P/A:P".into(),
            score: 11.5,
        };
        assert_eq!(s.numeric(), 10.0);
        let s = Severity::Cvss40 {
            vector: "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H".into(),
            score: -2.0,
        };
        assert_eq!(s.numeric(), 0.0);
    }

    #[test]
    fn bucket_band_edges() {
        assert_eq!(Severity::bucket_from_numeric(-1.0), Level::None);
        assert_eq!(Severity::bucket_from_numeric(0.0), Level::None);
        assert_eq!(Severity::bucket_from_numeric(0.1), Level::Low);
        assert_eq!(Severity::bucket_from_numeric(3.9), Level::Low);
        assert_eq!(Severity::bucket_from_numeric(4.0), Level::Medium);
        assert_eq!(Severity::bucket_from_numeric(6.9), Level::Medium);
        assert_eq!(Severity::bucket_from_numeric(7.0), Level::High);
        assert_eq!(Severity::bucket_from_numeric(8.9), Level::High);
        assert_eq!(Severity::bucket_from_numeric(9.0), Level::Critical);
        assert_eq!(Severity::bucket_from_numeric(10.0), Level::Critical);
    }

    #[test]
    fn sarif_level_mapping() {
        assert_eq!(Severity::Qualitative(Level::Critical).sarif_level(), "error");
        assert_eq!(Severity::Qualitative(Level::High).sarif_level(), "error");
        assert_eq!(Severity::Qualitative(Level::Medium).sarif_level(), "warning");
        assert_eq!(Severity::Qualitative(Level::Low).sarif_level(), "note");
        assert_eq!(Severity::Qualitative(Level::None).sarif_level(), "note");
    }

    #[test]
    fn cvss_qualitative_bucket_uses_score() {
        let s = Severity::Cvss31 {
            vector: "AV:N/AC:L/Au:N/C:P/I:P/A:P".into(),
            score: 7.5,
        };
        assert_eq!(s.qualitative_bucket(), Level::High);
        assert_eq!(s.sarif_level(), "error");
    }

    #[test]
    fn round_trips_via_serde() {
        let s = Severity::Qualitative(Level::High);
        let j = serde_json::to_value(&s).unwrap();
        let s2: Severity = serde_json::from_value(j).unwrap();
        assert_eq!(s, s2);

        let s = Severity::Cvss31 {
            vector: "AV:N/AC:L/Au:N/C:P/I:P/A:P".into(),
            score: 7.5,
        };
        let j = serde_json::to_value(&s).unwrap();
        let s2: Severity = serde_json::from_value(j).unwrap();
        assert_eq!(s, s2);
    }
}
