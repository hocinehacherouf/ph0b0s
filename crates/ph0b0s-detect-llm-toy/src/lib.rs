//! Smoke detector #1: LLM-driven.
//!
//! Walks a bounded set of source files, asks the model for "obvious issues"
//! via `LlmAgent::structured`, and emits findings. Hermetic by design —
//! tests run against `MockLlmAgent` (no real provider, no network).
//!
//! Per-file errors (read fail, agent error, schema mismatch) are logged at
//! `WARN` and skipped; one bad file never aborts the whole scan. Cancellation
//! and deadline are checked once per file.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use async_trait::async_trait;
use ph0b0s_core::detector::{Detector, DetectorCtx, DetectorKind, DetectorMetadata};
use ph0b0s_core::error::DetectorError;
use ph0b0s_core::finding::{
    Confidence, Evidence, Finding, Fingerprint, Location, SanitizationState,
};
use ph0b0s_core::llm::{ChatMessage, StructuredRequest};
use ph0b0s_core::severity::{Level, Severity};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

const RULE_ID_PREFIX: &str = "ph0b0s.llm-toy";
const DETECTOR_ID: &str = "llm-toy";
const SCHEMA_NAME: &str = "LlmToyOutput";
const SYSTEM_PROMPT: &str = "You are a security-focused code reviewer. Read the user-provided \
source file and identify obvious issues — hardcoded credentials, weak crypto, missing input \
validation, command-injection risk, hardcoded URLs to debug endpoints, etc. Reply ONLY in JSON \
matching the provided schema; do not include any text outside the JSON. If you see no issues, \
return {\"issues\": []}.";

const RESPONSE_SCHEMA: &str = r#"{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "issues": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "line": {"type": "integer", "minimum": 1},
          "severity": {"enum": ["low", "medium", "high", "critical"]},
          "message": {"type": "string", "minLength": 1}
        },
        "required": ["line", "severity", "message"]
      }
    }
  },
  "required": ["issues"]
}"#;

const DEFAULT_EXTENSIONS: &[&str] =
    &[".rs", ".py", ".js", ".ts", ".go", ".java", ".rb"];
const DEFAULT_MAX_FILES: usize = 10;
const DEFAULT_MAX_BYTES_PER_FILE: usize = 65_536;
const DEFAULT_MAX_FINDINGS_PER_FILE: usize = 20;
const EXCLUDED_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".git",
    ".venv",
    ".tox",
];

#[derive(Default, Clone, Copy)]
pub struct LlmToyDetector;

impl LlmToyDetector {
    pub fn new() -> Self {
        Self
    }
}

pub fn detector() -> Box<dyn Detector> {
    Box::new(LlmToyDetector::new())
}

#[async_trait]
impl Detector for LlmToyDetector {
    fn metadata(&self) -> DetectorMetadata {
        DetectorMetadata {
            id: DETECTOR_ID.to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            kind: DetectorKind::LlmDriven,
            description: "Walks bounded source files and asks the model \
                          for obvious issues."
                .to_owned(),
            capabilities: vec!["llm:any".to_owned()],
        }
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "max_files": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_FILES},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": DEFAULT_EXTENSIONS
                },
                "max_bytes_per_file": {
                    "type": "integer",
                    "minimum": 1,
                    "default": DEFAULT_MAX_BYTES_PER_FILE
                },
                "max_findings_per_file": {
                    "type": "integer",
                    "minimum": 1,
                    "default": DEFAULT_MAX_FINDINGS_PER_FILE
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
        let params = parse_params(ctx.params)?;
        let files = collect_files(&ctx.workspace.root, &params);

        let mut findings = Vec::new();
        for file in files {
            if cancel.is_cancelled() {
                return Err(DetectorError::Cancelled);
            }
            if Instant::now() >= ctx.deadline {
                return Err(DetectorError::Timeout);
            }

            let rel = file
                .strip_prefix(&ctx.workspace.root)
                .unwrap_or(&file)
                .to_string_lossy()
                .into_owned();

            let body = match read_truncated(&file, params.max_bytes_per_file).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(file = %rel, error = %e, "failed to read source file; skipping");
                    continue;
                }
            };

            let req = build_request(&rel, &body);

            let value = match ctx.agent.structured(req).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(file = %rel, error = %e, "agent structured call failed; skipping");
                    continue;
                }
            };

            let toy: ToyOutput = match serde_json::from_value(value) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(file = %rel, error = %e, "agent response failed schema; skipping");
                    continue;
                }
            };

            let take = toy.issues.len().min(params.max_findings_per_file);
            for issue in toy.issues.into_iter().take(take) {
                findings.push(issue_to_finding(&rel, issue));
            }
        }

        Ok(findings)
    }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Params {
    max_files: usize,
    extensions: Vec<String>,
    max_bytes_per_file: usize,
    max_findings_per_file: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_files: DEFAULT_MAX_FILES,
            extensions: DEFAULT_EXTENSIONS.iter().map(|s| (*s).to_owned()).collect(),
            max_bytes_per_file: DEFAULT_MAX_BYTES_PER_FILE,
            max_findings_per_file: DEFAULT_MAX_FINDINGS_PER_FILE,
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
        #[serde(default)]
        max_files: Option<usize>,
        #[serde(default)]
        extensions: Option<Vec<String>>,
        #[serde(default)]
        max_bytes_per_file: Option<usize>,
        #[serde(default)]
        max_findings_per_file: Option<usize>,
    }
    let r: ParamsRaw = serde_json::from_value(raw.clone()).map_err(|e| {
        DetectorError::InvalidParams(format!("llm-toy params: {e}"))
    })?;
    let mut p = Params::default();
    if let Some(v) = r.max_files {
        if v == 0 {
            return Err(DetectorError::InvalidParams(
                "llm-toy: max_files must be >= 1".into(),
            ));
        }
        p.max_files = v;
    }
    if let Some(v) = r.extensions {
        p.extensions = v;
    }
    if let Some(v) = r.max_bytes_per_file {
        if v == 0 {
            return Err(DetectorError::InvalidParams(
                "llm-toy: max_bytes_per_file must be >= 1".into(),
            ));
        }
        p.max_bytes_per_file = v;
    }
    if let Some(v) = r.max_findings_per_file {
        if v == 0 {
            return Err(DetectorError::InvalidParams(
                "llm-toy: max_findings_per_file must be >= 1".into(),
            ));
        }
        p.max_findings_per_file = v;
    }
    Ok(p)
}

// ---------------------------------------------------------------------------
// File walking
// ---------------------------------------------------------------------------

fn collect_files(root: &std::path::Path, params: &Params) -> Vec<PathBuf> {
    let excluded: HashSet<&str> = EXCLUDED_DIRS.iter().copied().collect();

    let walker = walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(move |e| {
            if e.depth() == 0 {
                return true; // never reject the root, even if its name starts with '.'
            }
            let name = e.file_name().to_string_lossy();
            if name.starts_with('.') {
                return false;
            }
            if e.file_type().is_dir() && excluded.contains(name.as_ref()) {
                return false;
            }
            true
        });

    let mut all: Vec<PathBuf> = walker
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let path = e.path();
            let ext = path.extension().and_then(|s| s.to_str())?;
            let dotted = format!(".{ext}");
            if params.extensions.iter().any(|x| x == &dotted) {
                Some(path.to_path_buf())
            } else {
                None
            }
        })
        .collect();
    all.sort();
    all.truncate(params.max_files);
    all
}

async fn read_truncated(
    path: &std::path::Path,
    max_bytes: usize,
) -> std::io::Result<String> {
    let file = tokio::fs::File::open(path).await?;
    let mut buf = Vec::with_capacity(max_bytes.min(8192));
    file.take(max_bytes as u64).read_to_end(&mut buf).await?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

// ---------------------------------------------------------------------------
// Request / response shapes
// ---------------------------------------------------------------------------

fn build_request(rel_path: &str, body: &str) -> StructuredRequest {
    let schema = serde_json::from_str(RESPONSE_SCHEMA)
        .unwrap_or_else(|_| serde_json::json!({ "type": "object" }));
    let user_text = format!("File: {rel_path}\n\n```\n{body}\n```");
    StructuredRequest {
        messages: vec![
            ChatMessage::System {
                content: SYSTEM_PROMPT.to_owned(),
            },
            ChatMessage::User {
                content: user_text,
            },
        ],
        schema,
        schema_name: SCHEMA_NAME.to_owned(),
        tools: Vec::new(),
        hints: Default::default(),
    }
}

#[derive(Debug, Deserialize)]
struct ToyOutput {
    issues: Vec<ToyIssue>,
}

#[derive(Debug, Deserialize)]
struct ToyIssue {
    line: u32,
    severity: String,
    message: String,
}

// ---------------------------------------------------------------------------
// Mapping issue -> Finding
// ---------------------------------------------------------------------------

fn issue_to_finding(rel_path: &str, issue: ToyIssue) -> Finding {
    let location = Location::File {
        path: rel_path.to_owned(),
        start_line: issue.line.max(1),
        end_line: issue.line.max(1),
        start_col: None,
        end_col: None,
    };
    let level = parse_level(&issue.severity);

    let mut h = Sha256::new();
    h.update(issue.message.as_bytes());
    let prefix = hex::encode(&h.finalize()[..8]);
    let rule_id = format!("{RULE_ID_PREFIX}.{prefix}");

    let title = truncate_chars(&issue.message, 80);
    let fingerprint = Fingerprint::compute(&rule_id, &location, b"");

    Finding {
        id: ulid::Ulid::new(),
        rule_id,
        detector: DETECTOR_ID.to_owned(),
        severity: Severity::Qualitative(level),
        confidence: Confidence::Low,
        title,
        message: issue.message,
        location,
        evidence: vec![Evidence::Note("flagged by llm-toy".to_owned())],
        fingerprint,
        sanitization: SanitizationState::Raw,
        suppressions: Vec::new(),
        created_at: chrono::Utc::now(),
    }
}

fn parse_level(s: &str) -> Level {
    match s.to_ascii_lowercase().as_str() {
        "critical" => Level::Critical,
        "high" => Level::High,
        "medium" => Level::Medium,
        "low" => Level::Low,
        _ => Level::Medium,
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    let total = s.chars().count();
    if total <= max {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::error::LlmError;
    use ph0b0s_core::target::Target;
    use ph0b0s_test_support::{
        deterministic_run_id, temp_workspace, temp_workspace_with, MockLlmAgent,
        MockToolHost,
    };
    use std::time::Duration;

    fn ctx_for_workspace<'a>(
        ws: &'a ph0b0s_core::target::Workspace,
        agent: &'a MockLlmAgent,
        tools: &'a MockToolHost,
        params: &'a serde_json::Value,
        deadline: Instant,
    ) -> (Target, DetectorCtx<'a>) {
        let target = Target::LocalDirectory {
            path: ws.root.clone(),
        };
        // We need the target to outlive the ctx; this helper bundles them.
        // Caller drops both at end of test scope.
        let ctx = DetectorCtx {
            workspace: ws,
            target: Box::leak(Box::new(target.clone())),
            agent,
            tools,
            params,
            run_id: deterministic_run_id(),
            deadline,
        };
        (target, ctx)
    }

    fn issues_response(items: &[(u32, &str, &str)]) -> serde_json::Value {
        let issues: Vec<_> = items
            .iter()
            .map(|(line, sev, msg)| {
                serde_json::json!({
                    "line": line,
                    "severity": sev,
                    "message": msg,
                })
            })
            .collect();
        serde_json::json!({ "issues": issues })
    }

    #[test]
    fn metadata_advertises_llm_driven_kind() {
        let m = LlmToyDetector.metadata();
        assert_eq!(m.id, "llm-toy");
        assert_eq!(m.kind, DetectorKind::LlmDriven);
        assert!(m.capabilities.iter().any(|c| c == "llm:any"));
    }

    #[test]
    fn config_schema_documents_known_fields() {
        let s = LlmToyDetector.config_schema();
        let props = &s["properties"];
        for k in ["max_files", "extensions", "max_bytes_per_file", "max_findings_per_file"] {
            assert!(props[k].is_object(), "missing schema field {k}");
        }
        assert_eq!(s["additionalProperties"], serde_json::Value::Bool(false));
    }

    #[test]
    fn parse_params_defaults_when_null_or_empty() {
        let p1 = parse_params(&serde_json::Value::Null).unwrap();
        assert_eq!(p1.max_files, DEFAULT_MAX_FILES);
        assert_eq!(p1.max_bytes_per_file, DEFAULT_MAX_BYTES_PER_FILE);
        assert_eq!(p1.max_findings_per_file, DEFAULT_MAX_FINDINGS_PER_FILE);
        assert!(p1.extensions.iter().any(|e| e == ".rs"));
        let p2 = parse_params(&serde_json::json!({})).unwrap();
        assert_eq!(p2.max_files, DEFAULT_MAX_FILES);
    }

    #[test]
    fn parse_params_accepts_overrides() {
        let p = parse_params(&serde_json::json!({
            "max_files": 3,
            "extensions": [".rs", ".toml"],
            "max_bytes_per_file": 1024,
            "max_findings_per_file": 5,
        }))
        .unwrap();
        assert_eq!(p.max_files, 3);
        assert_eq!(p.extensions, vec![".rs".to_owned(), ".toml".to_owned()]);
        assert_eq!(p.max_bytes_per_file, 1024);
        assert_eq!(p.max_findings_per_file, 5);
    }

    #[test]
    fn parse_params_rejects_unknown_field() {
        let err = parse_params(&serde_json::json!({"unknown": 1})).unwrap_err();
        match err {
            DetectorError::InvalidParams(_) => {}
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn parse_params_rejects_zero() {
        let err = parse_params(&serde_json::json!({"max_files": 0})).unwrap_err();
        matches!(err, DetectorError::InvalidParams(_));
    }

    #[tokio::test]
    async fn run_with_no_matching_files_returns_empty_and_does_not_call_agent() {
        let ws = temp_workspace_with(&[("README.md", "hi"), ("notes.txt", "x")])
            .expect("ws");
        let agent = MockLlmAgent::new();
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let findings = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert!(findings.is_empty());
        assert!(agent.recorded_structured().is_empty());
    }

    #[tokio::test]
    async fn run_calls_agent_once_per_matching_file_in_sorted_order() {
        let ws = temp_workspace_with(&[
            ("b.py", "print('hi')"),
            ("a.rs", "fn main() {}"),
            ("c.js", "console.log('hi')"),
            ("README.md", "ignored"),
        ])
        .expect("ws");
        let agent = MockLlmAgent::new();
        agent
            .enqueue_structured_ok(issues_response(&[]))
            .enqueue_structured_ok(issues_response(&[]))
            .enqueue_structured_ok(issues_response(&[]));
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let findings = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert!(findings.is_empty());
        let recorded = agent.recorded_structured();
        assert_eq!(recorded.len(), 3, "one structured call per matching file");
        let paths: Vec<_> = recorded
            .iter()
            .map(|r| match &r.messages[1] {
                ChatMessage::User { content } => content.lines().next().unwrap_or("").to_owned(),
                _ => panic!("expected User message at index 1"),
            })
            .collect();
        assert_eq!(
            paths,
            vec![
                "File: a.rs".to_owned(),
                "File: b.py".to_owned(),
                "File: c.js".to_owned(),
            ]
        );
    }

    #[tokio::test]
    async fn run_caps_at_max_files() {
        let ws = temp_workspace_with(&[
            ("a.rs", ""),
            ("b.rs", ""),
            ("c.rs", ""),
            ("d.rs", ""),
        ])
        .expect("ws");
        let agent = MockLlmAgent::new();
        for _ in 0..2 {
            agent.enqueue_structured_ok(issues_response(&[]));
        }
        let tools = MockToolHost::new();
        let params = serde_json::json!({"max_files": 2});
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(agent.recorded_structured().len(), 2);
    }

    #[tokio::test]
    async fn run_skips_excluded_directories() {
        let ws = temp_workspace_with(&[
            ("src/main.rs", "fn main() {}"),
            ("target/foo.rs", "fn ignored() {}"),
            ("node_modules/dep.js", "console.log('skip')"),
            (".git/hooks.rs", "ignored"),
        ])
        .expect("ws");
        let agent = MockLlmAgent::new();
        agent.enqueue_structured_ok(issues_response(&[]));
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        let recorded = agent.recorded_structured();
        assert_eq!(recorded.len(), 1);
        match &recorded[0].messages[1] {
            ChatMessage::User { content } => {
                assert!(content.starts_with("File: src/main.rs"));
            }
            _ => panic!("expected user msg"),
        }
    }

    #[tokio::test]
    async fn run_truncates_files_above_max_bytes() {
        let body: String = "x".repeat(2_000);
        let ws = temp_workspace_with(&[("a.rs", body.as_str())]).expect("ws");
        let agent = MockLlmAgent::new();
        agent.enqueue_structured_ok(issues_response(&[]));
        let tools = MockToolHost::new();
        let params = serde_json::json!({"max_bytes_per_file": 500});
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        let recorded = agent.recorded_structured();
        let user = match &recorded[0].messages[1] {
            ChatMessage::User { content } => content.clone(),
            _ => panic!(),
        };
        // The user message contains the truncated body; the body chars sent
        // are at most max_bytes_per_file.
        // Wrapping is `File: <path>\n\n```\n<body>\n````, so check the body
        // length is <= max_bytes_per_file.
        let body_in_msg: String =
            user.chars().filter(|c| *c == 'x').collect();
        assert!(
            body_in_msg.len() <= 500,
            "expected truncated body <=500, got {}",
            body_in_msg.len()
        );
    }

    #[tokio::test]
    async fn run_returns_findings_from_canned_agent_output() {
        let ws = temp_workspace_with(&[("a.rs",
            "fn main() { let pwd = \"hunter2\"; println!(\"{}\", pwd); }")])
            .expect("ws");
        let agent = MockLlmAgent::new();
        agent.enqueue_structured_ok(issues_response(&[
            (1, "high", "hardcoded password literal"),
            (1, "medium", "println leaks sensitive data"),
        ]));
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let findings = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].title, "hardcoded password literal");
        assert!(findings[0].rule_id.starts_with("ph0b0s.llm-toy."));
        assert_eq!(findings[0].confidence, Confidence::Low);
        assert_eq!(findings[0].detector, "llm-toy");
        // location uses workspace-relative path
        match &findings[0].location {
            Location::File { path, start_line, .. } => {
                assert_eq!(path, "a.rs");
                assert_eq!(*start_line, 1);
            }
            other => panic!("expected File location, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_continues_on_per_file_agent_error() {
        let ws = temp_workspace_with(&[
            ("a.rs", "fn a() {}"),
            ("b.rs", "fn b() {}"),
        ])
        .expect("ws");
        let agent = MockLlmAgent::new();
        agent
            .enqueue_structured_err(LlmError::Provider("boom".into()))
            .enqueue_structured_ok(issues_response(&[(2, "low", "minor lint")]));
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let findings = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(agent.recorded_structured().len(), 2);
    }

    #[tokio::test]
    async fn run_continues_on_per_file_schema_mismatch() {
        let ws = temp_workspace_with(&[
            ("a.rs", "fn a() {}"),
            ("b.rs", "fn b() {}"),
        ])
        .expect("ws");
        let agent = MockLlmAgent::new();
        agent
            .enqueue_structured_ok(serde_json::json!({"unrelated": "shape"}))
            .enqueue_structured_ok(issues_response(&[(3, "high", "real one")]));
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let findings = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].title, "real one");
    }

    #[tokio::test]
    async fn run_returns_cancelled_when_token_pre_cancelled() {
        let ws = temp_workspace_with(&[("a.rs", "fn a() {}")]).expect("ws");
        let agent = MockLlmAgent::new();
        agent.enqueue_structured_ok(issues_response(&[]));
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = LlmToyDetector.run(&ctx, cancel).await.unwrap_err();
        match err {
            DetectorError::Cancelled => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
        assert!(agent.recorded_structured().is_empty());
    }

    #[tokio::test]
    async fn run_returns_timeout_when_deadline_in_past() {
        let ws = temp_workspace_with(&[("a.rs", "fn a() {}")]).expect("ws");
        let agent = MockLlmAgent::new();
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() - Duration::from_secs(1),
        );
        let err = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap_err();
        match err {
            DetectorError::Timeout => {}
            other => panic!("expected Timeout, got {other:?}"),
        }
        assert!(agent.recorded_structured().is_empty());
    }

    #[tokio::test]
    async fn run_caps_at_max_findings_per_file() {
        let ws = temp_workspace_with(&[("a.rs", "fn a() {}")]).expect("ws");
        let agent = MockLlmAgent::new();
        let many: Vec<_> = (1..=10)
            .map(|i| (i, "low", "noise"))
            .collect();
        agent.enqueue_structured_ok(issues_response(&many));
        let tools = MockToolHost::new();
        let params = serde_json::json!({"max_findings_per_file": 3});
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let findings = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(findings.len(), 3);
    }

    #[test]
    fn parse_level_known_strings() {
        assert_eq!(parse_level("low"), Level::Low);
        assert_eq!(parse_level("medium"), Level::Medium);
        assert_eq!(parse_level("HIGH"), Level::High);
        assert_eq!(parse_level("Critical"), Level::Critical);
    }

    #[test]
    fn parse_level_unknown_falls_back_to_medium() {
        assert_eq!(parse_level("urgent"), Level::Medium);
        assert_eq!(parse_level(""), Level::Medium);
    }

    #[test]
    fn truncate_chars_keeps_short_strings_unchanged() {
        assert_eq!(truncate_chars("hello", 80), "hello");
    }

    #[test]
    fn truncate_chars_appends_ellipsis_when_long() {
        let long: String = "x".repeat(100);
        let t = truncate_chars(&long, 80);
        assert_eq!(t.chars().count(), 81); // 80 chars + ellipsis
        assert!(t.ends_with('…'));
    }

    #[test]
    fn truncate_chars_is_char_aware() {
        // 4 multi-byte chars
        let s = "héllø";
        assert_eq!(truncate_chars(s, 4), "héll…");
    }

    #[test]
    fn issue_to_finding_uses_message_hash_for_rule_id() {
        let f1 = issue_to_finding(
            "src/x.rs",
            ToyIssue {
                line: 1,
                severity: "high".into(),
                message: "same message".into(),
            },
        );
        let f2 = issue_to_finding(
            "src/y.rs",
            ToyIssue {
                line: 5,
                severity: "low".into(),
                message: "same message".into(),
            },
        );
        assert_eq!(f1.rule_id, f2.rule_id);
        assert_ne!(f1.fingerprint, f2.fingerprint, "different paths/lines should still differ in fingerprint");
    }

    #[tokio::test]
    async fn run_does_not_panic_when_workspace_is_empty() {
        let ws = temp_workspace().expect("ws");
        let agent = MockLlmAgent::new();
        let tools = MockToolHost::new();
        let params = serde_json::Value::Null;
        let (_target, ctx) = ctx_for_workspace(
            &ws,
            &agent,
            &tools,
            &params,
            Instant::now() + Duration::from_secs(60),
        );
        let findings = LlmToyDetector
            .run(&ctx, CancellationToken::new())
            .await
            .unwrap();
        assert!(findings.is_empty());
        assert!(agent.recorded_structured().is_empty());
    }
}
