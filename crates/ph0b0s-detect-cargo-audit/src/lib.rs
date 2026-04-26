//! Smoke-test detector #2: subprocess wrapper around `cargo audit --json`.
//!
//! Demonstrates the subprocess-detector wiring pattern. Ignores
//! `ctx.agent` entirely — proving the seam doesn't force LLM dependency on
//! detection-pack code.
//!
//! Behavior:
//! - If the workspace has no `Cargo.lock` → returns `Ok(vec![])`. (Library
//!   crates often have no lockfile checked in; that's a normal state.)
//! - If `cargo` or the `audit` subcommand isn't available → returns
//!   `DetectorError::MissingTool`.
//! - Otherwise: runs `cargo audit --json [--no-fetch]`, parses
//!   `vulnerabilities.list[]` from stdout, maps each entry into a
//!   `Finding`. Non-zero exit codes are EXPECTED when vulnerabilities are
//!   present — we always parse stdout first and only error on truly
//!   unparseable output.
//!
//! See `parse_audit_output` for the JSON → `Finding` mapping.

use std::str::FromStr;

use async_trait::async_trait;
use ph0b0s_core::detector::{
    Detector, DetectorCtx, DetectorKind, DetectorMetadata,
};
use ph0b0s_core::error::DetectorError;
use ph0b0s_core::finding::{
    Confidence, Evidence, Finding, Fingerprint, Location, SanitizationState,
};
use ph0b0s_core::severity::{Level, Severity};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

const RULE_ID_PREFIX: &str = "ph0b0s.cargo-audit";
const DETECTOR_ID: &str = "cargo-audit";

#[derive(Default, Clone, Copy)]
pub struct CargoAuditDetector;

impl CargoAuditDetector {
    pub fn new() -> Self {
        Self
    }
}

pub fn detector() -> Box<dyn Detector> {
    Box::new(CargoAuditDetector::new())
}

#[async_trait]
impl Detector for CargoAuditDetector {
    fn metadata(&self) -> DetectorMetadata {
        DetectorMetadata {
            id: DETECTOR_ID.to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            kind: DetectorKind::Subprocess,
            description: "Wraps `cargo audit --json` and ingests RUSTSEC \
                          advisories as findings."
                .to_owned(),
            capabilities: vec!["cargo".to_owned(), "cargo-audit".to_owned()],
        }
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "no_fetch": {
                    "type": "boolean",
                    "default": true,
                    "description": "Skip refreshing the advisory DB \
                                    (passes --no-fetch to cargo audit)."
                },
                "cargo_path": {
                    "type": "string",
                    "description": "Path to the `cargo` binary; defaults \
                                    to looking up `cargo` on PATH."
                }
            }
        })
    }

    #[tracing::instrument(skip_all, fields(detector = DETECTOR_ID, run_id = %ctx.run_id))]
    async fn run(
        &self,
        ctx: &DetectorCtx<'_>,
        cancel: CancellationToken,
    ) -> Result<Vec<Finding>, DetectorError> {
        let lockfile = ctx.workspace.root.join("Cargo.lock");
        if !lockfile.is_file() {
            tracing::debug!(
                "no Cargo.lock at {} — skipping",
                lockfile.display()
            );
            return Ok(Vec::new());
        }

        let params = parse_params(ctx.params)?;
        let stdout = run_cargo_audit(&ctx.workspace.root, &params, cancel).await?;
        parse_audit_output(&stdout)
    }
}

// ---------------------------------------------------------------------------
// Subprocess invocation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Params {
    no_fetch: bool,
    cargo_path: String,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            no_fetch: true,
            cargo_path: "cargo".to_owned(),
        }
    }
}

fn parse_params(raw: &serde_json::Value) -> Result<Params, DetectorError> {
    if raw.is_null() || raw.as_object().map(|o| o.is_empty()).unwrap_or(false) {
        return Ok(Params::default());
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ParamsRaw {
        #[serde(default = "default_no_fetch")]
        no_fetch: bool,
        #[serde(default)]
        cargo_path: Option<String>,
    }
    fn default_no_fetch() -> bool {
        true
    }
    let raw: ParamsRaw = serde_json::from_value(raw.clone()).map_err(|e| {
        DetectorError::InvalidParams(format!("cargo-audit params: {e}"))
    })?;
    Ok(Params {
        no_fetch: raw.no_fetch,
        cargo_path: raw.cargo_path.unwrap_or_else(|| "cargo".to_owned()),
    })
}

async fn run_cargo_audit(
    workspace_root: &std::path::Path,
    params: &Params,
    cancel: CancellationToken,
) -> Result<String, DetectorError> {
    let mut cmd = tokio::process::Command::new(&params.cargo_path);
    cmd.arg("audit").arg("--json");
    if params.no_fetch {
        cmd.arg("--no-fetch");
    }
    cmd.current_dir(workspace_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    let child_fut = cmd.output();
    let output = tokio::select! {
        out = child_fut => out,
        _ = cancel.cancelled() => return Err(DetectorError::Cancelled),
    };

    let output = match output {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(DetectorError::MissingTool(format!(
                "cargo not found at {:?}",
                params.cargo_path
            )));
        }
        Err(e) => return Err(DetectorError::Io(e)),
    };

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    // cargo-audit not installed: cargo emits "no such command: `audit`" on
    // stderr and exits non-zero with no JSON on stdout.
    if stdout.trim().is_empty() {
        if stderr.contains("no such command") || stderr.contains("audit") {
            return Err(DetectorError::MissingTool(format!(
                "cargo-audit not installed; install with `cargo install cargo-audit`. stderr: {}",
                stderr.trim()
            )));
        }
        return Err(DetectorError::Subprocess(format!(
            "cargo audit produced no output; stderr: {}",
            stderr.trim()
        )));
    }

    Ok(stdout)
}

// ---------------------------------------------------------------------------
// JSON → Finding mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AuditReport {
    #[serde(default)]
    vulnerabilities: VulnerabilityList,
}

#[derive(Debug, Default, Deserialize)]
struct VulnerabilityList {
    #[serde(default)]
    list: Vec<RawVulnerability>,
}

#[derive(Debug, Deserialize)]
struct RawVulnerability {
    advisory: RawAdvisory,
    package: RawPackage,
    // We pass-through other fields (`versions`, `affected`) as part of the
    // raw evidence blob; we don't need to model them strictly here.
    #[serde(flatten)]
    extras: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawAdvisory {
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    cvss: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(flatten)]
    extras: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawPackage {
    name: String,
    version: String,
    #[serde(flatten)]
    extras: serde_json::Map<String, serde_json::Value>,
}

/// Parse `cargo audit --json` stdout into `Finding`s.
///
/// Public so tests in this crate (and potentially adapter integration tests)
/// can exercise the parser without spawning a subprocess.
pub fn parse_audit_output(stdout: &str) -> Result<Vec<Finding>, DetectorError> {
    let report: AuditReport = serde_json::from_str(stdout).map_err(|e| {
        DetectorError::Parse(format!("cargo-audit JSON: {e}"))
    })?;

    let mut findings = Vec::with_capacity(report.vulnerabilities.list.len());
    for vuln in report.vulnerabilities.list {
        findings.push(vuln_to_finding(&vuln));
    }
    Ok(findings)
}

fn vuln_to_finding(vuln: &RawVulnerability) -> Finding {
    let rule_id = format!("{}.{}", RULE_ID_PREFIX, vuln.advisory.id);
    let location = Location::Symbolic {
        package: vuln.package.name.clone(),
        version: vuln.package.version.clone(),
        ecosystem: "crates.io".to_owned(),
    };
    let severity = severity_from_advisory(&vuln.advisory);

    // Re-serialize the raw advisory + package as evidence so consumers can
    // see the original record (URL, references, CVSS vector, etc.).
    let raw_value = serde_json::to_value(serialize_raw(vuln))
        .unwrap_or(serde_json::Value::Null);

    let title = if vuln.advisory.title.is_empty() {
        format!("{} ({})", vuln.advisory.id, vuln.package.name)
    } else {
        vuln.advisory.title.clone()
    };

    let message = build_message(vuln);
    let fingerprint = Fingerprint::compute(&rule_id, &location, b"");

    Finding {
        id: ulid::Ulid::new(),
        rule_id,
        detector: DETECTOR_ID.to_owned(),
        severity,
        confidence: Confidence::High,
        title,
        message,
        location,
        evidence: vec![Evidence::Json(raw_value)],
        fingerprint,
        sanitization: SanitizationState::Raw,
        suppressions: Vec::new(),
        created_at: chrono::Utc::now(),
    }
}

fn severity_from_advisory(advisory: &RawAdvisory) -> Severity {
    let Some(vector) = &advisory.cvss else {
        return Severity::Qualitative(Level::Medium);
    };
    match cvss::v3::Base::from_str(vector) {
        Ok(base) => {
            // Score is internally f64; cap-cast to f32 for the seam type.
            // CVSS scores are bounded to 0.0–10.0 so the cast is lossless.
            let score = base.score().value() as f32;
            Severity::Cvss31 {
                vector: vector.clone(),
                score,
            }
        }
        Err(_) => {
            // Unparseable vector: fall back to qualitative based on
            // crude heuristics on the vector severity letter (or Medium).
            Severity::Qualitative(Level::Medium)
        }
    }
}

fn build_message(vuln: &RawVulnerability) -> String {
    let mut msg = String::new();
    if !vuln.advisory.description.is_empty() {
        msg.push_str(vuln.advisory.description.trim());
    } else if !vuln.advisory.title.is_empty() {
        msg.push_str(&vuln.advisory.title);
    } else {
        msg.push_str(&vuln.advisory.id);
    }
    if let Some(url) = &vuln.advisory.url {
        if !msg.is_empty() {
            msg.push_str("\n\n");
        }
        msg.push_str("See: ");
        msg.push_str(url);
    }
    msg
}

/// Re-serialize the raw vulnerability as a JSON value that preserves all
/// pass-through fields. We keep the original shape for traceability.
fn serialize_raw(vuln: &RawVulnerability) -> serde_json::Value {
    let mut advisory = serde_json::Map::new();
    advisory.insert("id".into(), serde_json::Value::String(vuln.advisory.id.clone()));
    advisory.insert(
        "title".into(),
        serde_json::Value::String(vuln.advisory.title.clone()),
    );
    advisory.insert(
        "description".into(),
        serde_json::Value::String(vuln.advisory.description.clone()),
    );
    if let Some(c) = &vuln.advisory.cvss {
        advisory.insert("cvss".into(), serde_json::Value::String(c.clone()));
    }
    if let Some(u) = &vuln.advisory.url {
        advisory.insert("url".into(), serde_json::Value::String(u.clone()));
    }
    for (k, v) in &vuln.advisory.extras {
        advisory.insert(k.clone(), v.clone());
    }

    let mut package = serde_json::Map::new();
    package.insert(
        "name".into(),
        serde_json::Value::String(vuln.package.name.clone()),
    );
    package.insert(
        "version".into(),
        serde_json::Value::String(vuln.package.version.clone()),
    );
    for (k, v) in &vuln.package.extras {
        package.insert(k.clone(), v.clone());
    }

    let mut top = serde_json::Map::new();
    top.insert("advisory".into(), serde_json::Value::Object(advisory));
    top.insert("package".into(), serde_json::Value::Object(package));
    for (k, v) in &vuln.extras {
        top.insert(k.clone(), v.clone());
    }
    serde_json::Value::Object(top)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::llm::{AgentRoleKey, ChatRequest, LlmAgent, LlmSession, SessionOptions, StructuredRequest};
    use ph0b0s_core::error::LlmError;
    use ph0b0s_test_support::{
        deterministic_run_id, temp_workspace, temp_workspace_with, MockToolHost,
    };
    use std::time::{Duration, Instant};

    /// Minimal `LlmAgent` impl that errors on every call — proves the
    /// subprocess detector never touches `ctx.agent`.
    struct PanickingAgent;
    #[async_trait]
    impl LlmAgent for PanickingAgent {
        async fn chat(&self, _: ChatRequest) -> Result<ph0b0s_core::llm::ChatResponse, LlmError> {
            panic!("subprocess detector must not call agent.chat()")
        }
        async fn structured(&self, _: StructuredRequest) -> Result<serde_json::Value, LlmError> {
            panic!("subprocess detector must not call agent.structured()")
        }
        async fn session(&self, _: SessionOptions) -> Result<Box<dyn LlmSession>, LlmError> {
            panic!("subprocess detector must not call agent.session()")
        }
        fn model_id(&self) -> &str { "none" }
        fn role(&self) -> &AgentRoleKey {
            static R: std::sync::OnceLock<AgentRoleKey> = std::sync::OnceLock::new();
            R.get_or_init(|| AgentRoleKey::new("none"))
        }
    }

    const FIXTURE_ONE_VULN: &str = r#"{
        "lockfile": {"dependency-count": 12},
        "settings": {"target_arch": [], "target_os": [], "ignore": [], "informational_warnings": []},
        "vulnerabilities": {
            "found": true,
            "count": 1,
            "list": [
                {
                    "advisory": {
                        "id": "RUSTSEC-2024-0001",
                        "package": "openssl",
                        "title": "Memory corruption in openssl",
                        "description": "OpenSSL 0.10.x suffers from a heap corruption when parsing malformed certificates.",
                        "date": "2024-03-15",
                        "aliases": ["CVE-2024-12345"],
                        "categories": ["memory-corruption"],
                        "keywords": ["ssl"],
                        "cvss": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                        "url": "https://rustsec.org/advisories/RUSTSEC-2024-0001",
                        "references": []
                    },
                    "versions": {"patched": [">= 0.10.55"], "unaffected": []},
                    "affected": null,
                    "package": {
                        "name": "openssl",
                        "version": "0.10.0",
                        "source": "registry+https://github.com/rust-lang/crates.io-index",
                        "checksum": null,
                        "dependencies": []
                    }
                }
            ]
        },
        "warnings": {}
    }"#;

    const FIXTURE_NO_VULNS: &str = r#"{
        "lockfile": {"dependency-count": 12},
        "settings": {"target_arch": [], "target_os": [], "ignore": [], "informational_warnings": []},
        "vulnerabilities": {"found": false, "count": 0, "list": []},
        "warnings": {}
    }"#;

    const FIXTURE_ADVISORY_NO_CVSS: &str = r#"{
        "lockfile": {"dependency-count": 1},
        "settings": {"target_arch": [], "target_os": [], "ignore": [], "informational_warnings": []},
        "vulnerabilities": {
            "found": true,
            "count": 1,
            "list": [
                {
                    "advisory": {
                        "id": "RUSTSEC-2030-0042",
                        "package": "tinyfoo",
                        "title": "tinyfoo bug",
                        "description": "no cvss attached",
                        "date": "2030-01-01",
                        "aliases": [],
                        "categories": [],
                        "keywords": [],
                        "cvss": null,
                        "references": []
                    },
                    "versions": {"patched": [], "unaffected": []},
                    "affected": null,
                    "package": {"name": "tinyfoo", "version": "1.0.0", "source": null, "checksum": null, "dependencies": []}
                }
            ]
        },
        "warnings": {}
    }"#;

    #[test]
    fn parses_no_vulnerabilities_into_empty_list() {
        let findings = parse_audit_output(FIXTURE_NO_VULNS).unwrap();
        assert!(findings.is_empty());
    }

    #[test]
    fn parses_one_vulnerability_into_one_finding() {
        let findings = parse_audit_output(FIXTURE_ONE_VULN).unwrap();
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.rule_id, "ph0b0s.cargo-audit.RUSTSEC-2024-0001");
        assert_eq!(f.detector, "cargo-audit");
        assert_eq!(f.title, "Memory corruption in openssl");
        assert_eq!(f.confidence, Confidence::High);
        match &f.location {
            Location::Symbolic { package, version, ecosystem } => {
                assert_eq!(package, "openssl");
                assert_eq!(version, "0.10.0");
                assert_eq!(ecosystem, "crates.io");
            }
            other => panic!("expected Symbolic location, got {other:?}"),
        }
    }

    #[test]
    fn cvss_is_parsed_into_cvss31_severity() {
        let findings = parse_audit_output(FIXTURE_ONE_VULN).unwrap();
        let f = &findings[0];
        match &f.severity {
            Severity::Cvss31 { vector, score } => {
                assert!(vector.starts_with("CVSS:3.1/"));
                assert!((*score - 9.8).abs() < 0.05, "score was {score}");
            }
            other => panic!("expected Cvss31, got {other:?}"),
        }
        // SARIF level should be `error` for a critical CVSS.
        assert_eq!(f.severity.sarif_level(), "error");
    }

    #[test]
    fn missing_cvss_falls_back_to_qualitative_medium() {
        let findings = parse_audit_output(FIXTURE_ADVISORY_NO_CVSS).unwrap();
        let f = &findings[0];
        match f.severity {
            Severity::Qualitative(Level::Medium) => {}
            ref other => panic!("expected Medium fallback, got {other:?}"),
        }
    }

    #[test]
    fn evidence_includes_raw_advisory_and_package() {
        let findings = parse_audit_output(FIXTURE_ONE_VULN).unwrap();
        let f = &findings[0];
        assert_eq!(f.evidence.len(), 1);
        match &f.evidence[0] {
            Evidence::Json(v) => {
                assert!(v["advisory"]["id"].as_str() == Some("RUSTSEC-2024-0001"));
                assert!(v["package"]["name"].as_str() == Some("openssl"));
                // pass-through extras preserved
                assert!(v["versions"].is_object());
            }
            other => panic!("expected Evidence::Json, got {other:?}"),
        }
    }

    #[test]
    fn fingerprint_is_stable_for_same_advisory_and_package() {
        let f1 = parse_audit_output(FIXTURE_ONE_VULN).unwrap();
        let f2 = parse_audit_output(FIXTURE_ONE_VULN).unwrap();
        assert_eq!(f1[0].fingerprint, f2[0].fingerprint);
        assert_ne!(f1[0].id, f2[0].id, "ids should be unique per call");
    }

    #[test]
    fn parse_garbage_returns_parse_error() {
        let err = parse_audit_output("not json").unwrap_err();
        match err {
            DetectorError::Parse(_) => {}
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_returns_empty_when_lockfile_absent() {
        let ws = temp_workspace().expect("ws");
        let agent = PanickingAgent;
        let tools = MockToolHost::new();
        let ctx = DetectorCtx {
            workspace: &ws,
            target: &ph0b0s_core::target::Target::LocalDirectory {
                path: ws.root.clone(),
            },
            agent: &agent,
            tools: &tools,
            params: &serde_json::Value::Null,
            run_id: deterministic_run_id(),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        let findings = CargoAuditDetector
            .run(&ctx, CancellationToken::new())
            .await
            .expect("should be Ok with no Cargo.lock");
        assert!(findings.is_empty());
    }

    #[tokio::test]
    async fn run_propagates_missing_tool_when_cargo_path_invalid() {
        let ws = temp_workspace_with(&[(
            "Cargo.lock",
            "# minimal lockfile shell\n",
        )])
        .expect("ws");
        let agent = PanickingAgent;
        let tools = MockToolHost::new();
        let ctx = DetectorCtx {
            workspace: &ws,
            target: &ph0b0s_core::target::Target::LocalDirectory {
                path: ws.root.clone(),
            },
            agent: &agent,
            tools: &tools,
            params: &serde_json::json!({"cargo_path": "/nonexistent/cargo-bin"}),
            run_id: deterministic_run_id(),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        let err = CargoAuditDetector
            .run(&ctx, CancellationToken::new())
            .await
            .expect_err("should be MissingTool");
        match err {
            DetectorError::MissingTool(_) => {}
            other => panic!("expected MissingTool, got {other:?}"),
        }
    }

    #[test]
    fn metadata_advertises_subprocess_kind_and_capabilities() {
        let m = CargoAuditDetector.metadata();
        assert_eq!(m.id, "cargo-audit");
        assert_eq!(m.kind, DetectorKind::Subprocess);
        assert!(m.capabilities.iter().any(|c| c == "cargo-audit"));
    }

    #[test]
    fn config_schema_advertises_no_fetch_and_cargo_path() {
        let s = CargoAuditDetector.config_schema();
        let props = &s["properties"];
        assert!(props["no_fetch"].is_object());
        assert!(props["cargo_path"].is_object());
    }

    #[test]
    fn parse_params_defaults_when_null_or_empty() {
        let p1 = parse_params(&serde_json::Value::Null).unwrap();
        assert!(p1.no_fetch);
        assert_eq!(p1.cargo_path, "cargo");
        let p2 = parse_params(&serde_json::json!({})).unwrap();
        assert!(p2.no_fetch);
    }

    #[test]
    fn parse_params_overrides() {
        let p = parse_params(
            &serde_json::json!({"no_fetch": false, "cargo_path": "/usr/local/bin/cargo"}),
        )
        .unwrap();
        assert!(!p.no_fetch);
        assert_eq!(p.cargo_path, "/usr/local/bin/cargo");
    }

    #[test]
    fn parse_params_rejects_unknown_field() {
        let err = parse_params(&serde_json::json!({"unknown": 1})).unwrap_err();
        match err {
            DetectorError::InvalidParams(_) => {}
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }
}
