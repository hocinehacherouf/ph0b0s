//! SARIF 2.1.0 reporter — primary output for IDE / GitHub Code Scanning.
//!
//! Output round-trips through `serde-sarif`'s typed [`Sarif`] struct, so the
//! emitted JSON is guaranteed to deserialize back as a schema-valid SARIF
//! 2.1.0 log. We build the JSON tree directly because typed-builder's
//! `strip_option` setters don't compose cleanly with conditional fields
//! (start_column/end_column/end_line are all optional). After building the
//! JSON we parse it through `Sarif` to enforce the schema; if you want the
//! typed value, call [`SarifReporter::build`].
//!
//! Mapping (from `Finding` → SARIF):
//!
//! | Field | SARIF target |
//! |---|---|
//! | `rule_id`              | `result.ruleId`, `tool.driver.rules[].id` |
//! | `detector`             | `tool.driver.rules[].properties.detector` |
//! | `severity` (numeric)   | `result.properties["security-severity"]` (string) |
//! | `severity` (level)     | `result.level` (`error`/`warning`/`note`) |
//! | `title`                | `tool.driver.rules[].shortDescription.text` |
//! | `message`              | `result.message.text` |
//! | `location` (file)      | `result.locations[].physicalLocation` |
//! | `location` (symbolic)  | `result.locations[].logicalLocations` + properties |
//! | `evidence`             | `result.properties.evidence` |
//! | `fingerprint`          | `result.fingerprints["ph0b0s/v1"]` |
//! | `confidence`           | `result.properties.confidence` |
//! | `sanitization`         | `result.properties.sanitization` |
//! | `suppressions`         | `result.suppressions[]` (kind=`external`) |

use std::collections::BTreeMap;

use async_trait::async_trait;
use ph0b0s_core::error::ReportError;
use ph0b0s_core::finding::{Confidence, Finding, Location, SuppressionHint};
use ph0b0s_core::report::Reporter;
use ph0b0s_core::scan::ScanResult;
use ph0b0s_core::severity::Level;
use serde_json::{Value, json};
use serde_sarif::sarif::Sarif;
use tokio::io::{AsyncWrite, AsyncWriteExt};

const TOOL_NAME: &str = "ph0b0s";
const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");
const TOOL_INFORMATION_URI: &str = "https://github.com/hocinehacherouf/ph0b0s";
const SARIF_SCHEMA: &str = "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json";
const FINGERPRINT_KEY: &str = "ph0b0s/v1";

#[derive(Default, Clone, Copy)]
pub struct SarifReporter;

impl SarifReporter {
    pub fn new() -> Self {
        Self
    }

    /// Build the typed [`Sarif`] value, validated against the SARIF 2.1.0
    /// schema by `serde-sarif`. Errors only if the JSON we constructed
    /// internally fails to deserialize as `Sarif` — which would indicate a
    /// bug in this reporter, not bad input.
    pub fn build(&self, result: &ScanResult) -> Result<Sarif, ReportError> {
        let value = self.build_value(result);
        serde_json::from_value::<Sarif>(value).map_err(|e| ReportError::InvalidSarif(e.to_string()))
    }

    /// Build the SARIF tree as a `serde_json::Value`. Useful when you want
    /// untyped access (e.g. for snapshot tests).
    pub fn build_value(&self, result: &ScanResult) -> Value {
        let rules = build_rules(&result.findings);
        let sarif_results: Vec<Value> = result.findings.iter().map(build_result).collect();

        json!({
            "$schema": SARIF_SCHEMA,
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": TOOL_NAME,
                            "version": TOOL_VERSION,
                            "informationUri": TOOL_INFORMATION_URI,
                            "rules": rules,
                        }
                    },
                    "results": sarif_results,
                }
            ]
        })
    }

    /// Render to pretty JSON. Always validates round-trip through `Sarif`.
    pub fn render(&self, result: &ScanResult) -> Result<String, ReportError> {
        let value = self.build_value(result);
        let _validated: Sarif = serde_json::from_value(value.clone())
            .map_err(|e| ReportError::InvalidSarif(e.to_string()))?;
        Ok(serde_json::to_string_pretty(&value)?)
    }
}

#[async_trait]
impl Reporter for SarifReporter {
    fn name(&self) -> &'static str {
        "sarif"
    }

    async fn write(
        &self,
        result: &ScanResult,
        sink: &mut (dyn AsyncWrite + Send + Unpin),
    ) -> Result<(), ReportError> {
        let body = self.render(result)?;
        sink.write_all(body.as_bytes()).await?;
        sink.write_all(b"\n").await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JSON tree builders
// ---------------------------------------------------------------------------

fn build_rules(findings: &[Finding]) -> Vec<Value> {
    // One rule per unique rule_id, ordered alphabetically for snapshot
    // stability.
    let mut seen: BTreeMap<String, &Finding> = BTreeMap::new();
    for f in findings {
        seen.entry(f.rule_id.clone()).or_insert(f);
    }
    seen.into_values()
        .map(|f| {
            json!({
                "id": f.rule_id,
                "name": f.rule_id,
                "shortDescription": { "text": f.title },
                "properties": {
                    "detector": f.detector,
                }
            })
        })
        .collect()
}

fn build_result(f: &Finding) -> Value {
    let mut props = serde_json::Map::new();
    // GitHub Code Scanning expects `security-severity` as a STRING containing
    // a number 0.0–10.0.
    props.insert(
        "security-severity".to_owned(),
        Value::String(format!("{:.1}", f.severity.numeric())),
    );
    props.insert(
        "confidence".to_owned(),
        Value::String(confidence_str(f.confidence).to_owned()),
    );
    props.insert(
        "evidence".to_owned(),
        serde_json::to_value(&f.evidence).unwrap_or(Value::Null),
    );
    props.insert(
        "sanitization".to_owned(),
        serde_json::to_value(&f.sanitization).unwrap_or(Value::Null),
    );
    if let Location::Symbolic {
        package,
        version,
        ecosystem,
    } = &f.location
    {
        props.insert("package".to_owned(), Value::String(package.clone()));
        props.insert("version".to_owned(), Value::String(version.clone()));
        props.insert("ecosystem".to_owned(), Value::String(ecosystem.clone()));
    }

    let mut result = serde_json::Map::new();
    result.insert("ruleId".to_owned(), Value::String(f.rule_id.clone()));
    result.insert(
        "level".to_owned(),
        Value::String(severity_to_level(f.severity.qualitative_bucket()).to_owned()),
    );
    result.insert("message".to_owned(), json!({ "text": f.message }));
    result.insert("locations".to_owned(), json!([build_location(&f.location)]));
    result.insert(
        "fingerprints".to_owned(),
        json!({ FINGERPRINT_KEY: f.fingerprint.0 }),
    );
    result.insert("properties".to_owned(), Value::Object(props));

    if !f.suppressions.is_empty() {
        let suppressions: Vec<Value> = f.suppressions.iter().map(build_suppression).collect();
        result.insert("suppressions".to_owned(), Value::Array(suppressions));
    }

    Value::Object(result)
}

fn build_location(loc: &Location) -> Value {
    match loc {
        Location::File {
            path,
            start_line,
            end_line,
            start_col,
            end_col,
        } => {
            let mut region = serde_json::Map::new();
            region.insert("startLine".to_owned(), json!(start_line));
            if start_line != end_line {
                region.insert("endLine".to_owned(), json!(end_line));
            }
            if let Some(c) = start_col {
                region.insert("startColumn".to_owned(), json!(c));
            }
            if let Some(c) = end_col {
                region.insert("endColumn".to_owned(), json!(c));
            }
            json!({
                "physicalLocation": {
                    "artifactLocation": { "uri": path },
                    "region": Value::Object(region),
                }
            })
        }
        Location::Symbolic {
            package,
            version,
            ecosystem,
        } => json!({
            "logicalLocations": [
                {
                    "name": format!("{ecosystem}:{package}@{version}"),
                    "kind": "package",
                }
            ]
        }),
    }
}

fn build_suppression(hint: &SuppressionHint) -> Value {
    json!({
        "kind": "external",
        "justification": hint.reason,
    })
}

fn severity_to_level(level: Level) -> &'static str {
    match level {
        Level::Critical | Level::High => "error",
        Level::Medium => "warning",
        Level::Low | Level::None => "note",
    }
}

fn confidence_str(c: Confidence) -> &'static str {
    match c {
        Confidence::Low => "low",
        Confidence::Medium => "medium",
        Confidence::High => "high",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_test_support::sample_scan_result;

    #[tokio::test]
    async fn render_round_trips_via_serde_sarif() {
        let r = sample_scan_result(3);
        let body = SarifReporter.render(&r).unwrap();
        let parsed: Sarif = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.runs.len(), 1);
        let run = &parsed.runs[0];
        assert_eq!(run.tool.driver.name, TOOL_NAME);
        let n = run.results.as_ref().map(Vec::len).unwrap_or(0);
        assert_eq!(n, 3);
    }

    #[tokio::test]
    async fn rules_table_is_unique_by_rule_id() {
        let r = sample_scan_result(3);
        let value = SarifReporter.build_value(&r);
        let rules = value["runs"][0]["tool"]["driver"]["rules"]
            .as_array()
            .unwrap();
        let ids: std::collections::HashSet<_> =
            rules.iter().map(|r| r["id"].as_str().unwrap()).collect();
        assert_eq!(ids.len(), rules.len(), "rule ids must be unique");
        assert_eq!(ids.len(), 3); // sample_scan_result(3) has 3 distinct rule_ids
    }

    #[tokio::test]
    async fn level_maps_severity_buckets() {
        let r = sample_scan_result(3);
        let value = SarifReporter.build_value(&r);
        let levels: Vec<&str> = value["runs"][0]["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["level"].as_str().unwrap())
            .collect();
        assert_eq!(levels, vec!["note", "warning", "error"]);
    }

    #[tokio::test]
    async fn fingerprint_is_emitted_under_ph0b0s_v1() {
        let r = sample_scan_result(1);
        let value = SarifReporter.build_value(&r);
        let fps = &value["runs"][0]["results"][0]["fingerprints"];
        assert_eq!(fps[FINGERPRINT_KEY], r.findings[0].fingerprint.0);
    }

    #[tokio::test]
    async fn security_severity_is_string_with_one_decimal() {
        let r = sample_scan_result(1);
        let value = SarifReporter.build_value(&r);
        let sev = &value["runs"][0]["results"][0]["properties"]["security-severity"];
        let s = sev.as_str().unwrap();
        // sample_scan_result(1) starts at Level::Low → numeric 3.0
        assert_eq!(s, "3.0");
    }

    #[tokio::test]
    async fn build_validates_against_typed_sarif() {
        let r = sample_scan_result(2);
        let typed = SarifReporter.build(&r).expect("schema-valid");
        assert_eq!(typed.runs.len(), 1);
    }

    #[tokio::test]
    async fn sarif_snapshot_three_findings() {
        let r = sample_scan_result(3);
        // `render` validates round-trip through the typed Sarif value AND
        // produces the exact wire format. Snapshot the wire form via a
        // plain text snapshot to avoid any insta-internal JSON normalisation
        // that would otherwise drift between feature sets.
        let body = SarifReporter.render(&r).unwrap();
        insta::assert_snapshot!("sarif_three_findings", body);
    }

    #[tokio::test]
    async fn write_appends_trailing_newline() {
        let r = sample_scan_result(1);
        let mut buf = Vec::new();
        SarifReporter.write(&r, &mut buf).await.unwrap();
        assert!(buf.ends_with(b"\n"));
    }
}
