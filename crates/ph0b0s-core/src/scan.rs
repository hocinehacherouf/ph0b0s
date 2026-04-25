//! Scan request, scan result, and the lifetime context detectors run inside.

use std::collections::BTreeMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::error::DetectorError;
use crate::finding::Finding;
use crate::severity::Level;
use crate::target::Target;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorFilter {
    /// Run every detector that's enabled in config.
    #[default]
    All,
    /// Only run the named detectors (matching `DetectorMetadata::id`).
    Only(Vec<String>),
    /// Run all enabled detectors except the named ones.
    Except(Vec<String>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScanOptions {
    pub max_parallel: usize,
    /// Per-detector wall-clock timeout.
    #[serde(with = "humantime_secs")]
    pub detector_timeout: Duration,
    /// If true, any detector failure aborts the run.
    pub strict: bool,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            max_parallel: 4,
            detector_timeout: Duration::from_secs(300),
            strict: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScanRequest {
    pub run_id: Ulid,
    pub target: Target,
    pub detector_filter: DetectorFilter,
    pub options: ScanOptions,
    /// Per-detector params keyed by detector id.
    #[serde(default)]
    pub detector_params: BTreeMap<String, serde_json::Value>,
}

/// Stable, run-scoped context handed to materializers and orchestrator hooks.
/// Detectors get a richer `DetectorCtx` (see `detector.rs`).
#[derive(Debug)]
pub struct ScanCtx {
    pub run_id: Ulid,
    pub started_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScanStats {
    pub by_severity: BTreeMap<Level, u64>,
    pub by_detector: BTreeMap<String, u64>,
    pub total_findings: u64,
    pub total_suppressed: u64,
    pub total_deduped: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd_estimate: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorRunError {
    pub detector_id: String,
    pub message: String,
}

impl From<(&str, &DetectorError)> for DetectorRunError {
    fn from(value: (&str, &DetectorError)) -> Self {
        Self {
            detector_id: value.0.to_owned(),
            message: value.1.to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScanResult {
    pub run_id: Ulid,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub findings: Vec<Finding>,
    pub stats: ScanStats,
    #[serde(default)]
    pub errors: Vec<DetectorRunError>,
}

mod humantime_secs {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}
