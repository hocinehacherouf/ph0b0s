//! Scan orchestrator — implements the lifecycle described in the slice (e)
//! plan.
//!
//! v1 simplifications:
//! - **Sequential** detector execution (no `Semaphore` / `JoinSet`); the
//!   bounded-parallel knob is in config but not honoured yet.
//! - Suppression filter consumes config rules + DB suppressions and merely
//!   tags findings via `SuppressionHint` (no removal from output) — keeps
//!   downstream consumers in control.
//! - `--strict` aborts the run on the first detector error.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ph0b0s_core::detector::DetectorCtx;
use ph0b0s_core::finding::{Finding, SuppressionHint};
use ph0b0s_core::report::Reporter;
use ph0b0s_core::scan::{
    DetectorFilter, DetectorRunError, ScanOptions, ScanRequest, ScanResult, ScanStats,
};
use ph0b0s_core::severity::Level;
use ph0b0s_core::store::FindingStore;
use ph0b0s_core::target::Target;
use ph0b0s_core::tools::ToolHost;
use ph0b0s_llm_adk::{AdkLlmAgent, AdkToolHost};
use ph0b0s_report::{JsonReporter, MarkdownReporter, SarifReporter};
use ph0b0s_storage::SqliteFindingStore;
use tokio_util::sync::CancellationToken;
use ulid::Ulid;

use crate::config::Config;
use crate::registry::DetectorRegistry;
use crate::workspace;

/// CLI arguments for `scan`, plain enough to be tested directly.
#[derive(Debug, Clone)]
pub struct ScanArgs {
    pub target: Target,
    pub output: Option<PathBuf>,
    pub markdown: Option<PathBuf>,
    pub json: Option<PathBuf>,
    pub strict: bool,
    pub detector_filter: Vec<String>,
    pub report_cost: bool,
}

/// Outcome of a `scan` invocation. Returned to `main.rs` for stdout
/// summary and exit-code shaping.
pub struct ScanOutcome {
    pub run_id: Ulid,
    pub result: ScanResult,
    pub deduped: usize,
}

/// Run a full scan. Wires the adapter, builds the request, runs the detector
/// pipeline, persists findings, writes reports.
pub async fn run(
    args: ScanArgs,
    config: Config,
    agent: AdkLlmAgent,
) -> Result<ScanOutcome> {
    // Tools host. v1: only Rust-native tools registered by detection packs;
    // MCP servers are recorded but not connected (adapter limitation).
    let tools = AdkToolHost::new();
    for spec in &config.mcp_servers {
        tools
            .mount_mcp(spec.clone())
            .await
            .with_context(|| format!("mount_mcp {}", spec.name))?;
    }

    // Storage.
    let db_path = config.effective_db_path();
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let store = SqliteFindingStore::open(&db_path)
        .await
        .with_context(|| format!("opening DB at {}", db_path.display()))?;
    let _aborted = store.cleanup_orphan_runs().await?;

    // Build request.
    let request = ScanRequest {
        run_id: Ulid::new(),
        target: args.target.clone(),
        detector_filter: if args.detector_filter.is_empty() {
            DetectorFilter::All
        } else {
            DetectorFilter::Only(args.detector_filter.clone())
        },
        options: ScanOptions {
            max_parallel: config.scan.max_parallel,
            detector_timeout: Duration::from_secs(config.scan.detector_timeout_s),
            strict: args.strict || config.scan.strict,
        },
        detector_params: config.detectors.clone(),
    };
    let run_id = store.begin_run(&request).await?;
    tracing::info!(run_id = %run_id, target = ?args.target, "scan started");

    // Workspace prep.
    let ws = workspace::prepare(&request.target).await?;

    // Detector resolution.
    let registry = DetectorRegistry::builtin();
    let resolved = registry.resolve(&args.detector_filter, &config);

    // Bounded-parallel: deferred to v2; sequential for now.
    let mut all_findings: Vec<Finding> = Vec::new();
    let mut errors: Vec<DetectorRunError> = Vec::new();

    let agent_ref: Arc<AdkLlmAgent> = Arc::new(agent);
    let tools_ref = Arc::new(tools);

    for resolved_det in resolved {
        let id = resolved_det.detector.metadata().id.clone();
        let deadline = Instant::now() + request.options.detector_timeout;
        let ctx = DetectorCtx {
            workspace: &ws,
            target: &request.target,
            agent: agent_ref.as_ref(),
            tools: tools_ref.as_ref(),
            params: &resolved_det.params,
            run_id,
            deadline,
        };
        let cancel = CancellationToken::new();
        let span = tracing::info_span!("detector", id = %id);
        let _g = span.enter();

        match resolved_det.detector.run(&ctx, cancel).await {
            Ok(findings) => {
                tracing::info!(detector = %id, count = findings.len(), "detector finished");
                all_findings.extend(findings);
            }
            Err(e) => {
                tracing::warn!(detector = %id, error = %e, "detector failed");
                let entry = DetectorRunError {
                    detector_id: id.clone(),
                    message: e.to_string(),
                };
                errors.push(entry);
                if request.options.strict {
                    anyhow::bail!("strict mode: detector {id} failed: {e}");
                }
            }
        }
    }

    // Persist findings.
    for f in &all_findings {
        store.record(run_id, f).await?;
    }

    // Dedup pass.
    let deduped = store.dedup(run_id).await?;

    // Apply config-driven suppressions: tag matching findings with a hint.
    // Findings stay in the DB & report; downstream consumers decide whether
    // to filter.
    apply_config_suppressions(&mut all_findings, &config);

    // Stats. (We keep our local copy for stats; the DB has authoritative data.)
    let stats = compute_stats(&all_findings, deduped, errors.len());

    // Persist suppressions hints into a fresh `record` if any were appended?
    // For simplicity, v1 emits the hint only at report time via the
    // in-memory copy. Persistent suppression-hints round-trip is a v2 task.

    store.finish_run(run_id, &stats).await?;

    // Reload from DB so the report uses persisted (post-dedup) data. We
    // overwrite the in-memory findings with the DB view but keep our
    // suppression hints.
    let mut result = store.load_run(run_id).await?;
    overlay_suppressions(&mut result.findings, &all_findings);
    result.errors = errors;

    // Reports.
    write_reports(&args, &config, &result).await?;

    Ok(ScanOutcome {
        run_id,
        result,
        deduped,
    })
}

fn apply_config_suppressions(findings: &mut [Finding], config: &Config) {
    if config.suppress.is_empty() {
        return;
    }
    let now = chrono::Utc::now();
    for f in findings.iter_mut() {
        for rule in &config.suppress {
            if rule.rule_id == f.rule_id {
                if let Some(exp) = rule.expires_at {
                    if exp < now {
                        continue;
                    }
                }
                f.suppressions.push(SuppressionHint {
                    reason: rule.reason.clone(),
                    hard: true,
                });
            }
        }
    }
}

fn overlay_suppressions(persisted: &mut [Finding], local: &[Finding]) {
    for p in persisted.iter_mut() {
        if let Some(l) = local.iter().find(|f| f.id == p.id) {
            p.suppressions = l.suppressions.clone();
        }
    }
}

fn compute_stats(findings: &[Finding], deduped: usize, error_count: usize) -> ScanStats {
    let mut by_severity: BTreeMap<Level, u64> = BTreeMap::new();
    let mut by_detector: BTreeMap<String, u64> = BTreeMap::new();
    let mut total_suppressed: u64 = 0;
    for f in findings {
        let level = f.severity.qualitative_bucket();
        *by_severity.entry(level).or_insert(0) += 1;
        *by_detector.entry(f.detector.clone()).or_insert(0) += 1;
        if !f.suppressions.is_empty() {
            total_suppressed += 1;
        }
    }
    ScanStats {
        by_severity,
        by_detector,
        total_findings: findings.len() as u64,
        total_suppressed,
        total_deduped: deduped as u64,
        tokens_in: 0,
        tokens_out: 0,
        cost_usd_estimate: 0.0,
    }
    .with_errors(error_count as u64)
}

/// Tiny extension trait so we can chain a few extra fields onto ScanStats
/// without adding setters to the seam.
trait ScanStatsExt {
    fn with_errors(self, _n: u64) -> Self;
}
impl ScanStatsExt for ScanStats {
    fn with_errors(self, _n: u64) -> Self {
        // ScanStats has no `errors` count field today; v1 stops here.
        // Kept as a hook for future extension.
        self
    }
}

async fn write_reports(
    args: &ScanArgs,
    config: &Config,
    result: &ScanResult,
) -> Result<()> {
    if let Some(path) = args
        .output
        .as_ref()
        .or(config.output.sarif_path.as_ref())
    {
        write_with(SarifReporter, path, result).await?;
    }
    if let Some(path) = args
        .markdown
        .as_ref()
        .or(config.output.markdown_path.as_ref())
    {
        write_with(MarkdownReporter, path, result).await?;
    }
    if let Some(path) = args.json.as_ref().or(config.output.json_path.as_ref()) {
        write_with(JsonReporter, path, result).await?;
    }
    Ok(())
}

async fn write_with<R: Reporter>(
    reporter: R,
    path: &PathBuf,
    result: &ScanResult,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
    }
    let mut file = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("creating report at {}", path.display()))?;
    reporter
        .write(result, &mut file)
        .await
        .with_context(|| format!("writing {} report", reporter.name()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::finding::{Confidence, Fingerprint, Location, SanitizationState};
    use ph0b0s_core::severity::Severity;

    fn fixture_finding(rule_id: &str, level: Level) -> Finding {
        let location = Location::File {
            path: "src/x.rs".into(),
            start_line: 1,
            end_line: 1,
            start_col: None,
            end_col: None,
        };
        Finding {
            id: Ulid::new(),
            rule_id: rule_id.into(),
            detector: "test".into(),
            severity: Severity::Qualitative(level),
            confidence: Confidence::Medium,
            title: "t".into(),
            message: "m".into(),
            location: location.clone(),
            evidence: Vec::new(),
            fingerprint: Fingerprint::compute(rule_id, &location, b""),
            sanitization: SanitizationState::Raw,
            suppressions: Vec::new(),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn compute_stats_groups_by_severity_and_detector() {
        let findings = vec![
            fixture_finding("r1", Level::Low),
            fixture_finding("r2", Level::Medium),
            fixture_finding("r3", Level::Medium),
        ];
        let stats = compute_stats(&findings, 1, 0);
        assert_eq!(stats.total_findings, 3);
        assert_eq!(stats.total_deduped, 1);
        assert_eq!(*stats.by_severity.get(&Level::Medium).unwrap(), 2);
        assert_eq!(*stats.by_severity.get(&Level::Low).unwrap(), 1);
        assert_eq!(*stats.by_detector.get("test").unwrap(), 3);
    }

    #[test]
    fn apply_config_suppressions_tags_matching_rule_ids() {
        let mut findings = vec![
            fixture_finding("r1", Level::Low),
            fixture_finding("r2", Level::Low),
        ];
        let cfg = Config {
            suppress: vec![crate::config::SuppressRule {
                rule_id: "r1".into(),
                reason: "ok".into(),
                expires_at: None,
            }],
            ..Default::default()
        };
        apply_config_suppressions(&mut findings, &cfg);
        assert_eq!(findings[0].suppressions.len(), 1);
        assert!(findings[0].suppressions[0].hard);
        assert_eq!(findings[0].suppressions[0].reason, "ok");
        assert!(findings[1].suppressions.is_empty());
    }

    #[test]
    fn apply_config_suppressions_respects_expiry() {
        let mut findings = vec![fixture_finding("r1", Level::Low)];
        let cfg = Config {
            suppress: vec![crate::config::SuppressRule {
                rule_id: "r1".into(),
                reason: "expired".into(),
                expires_at: Some(chrono::Utc::now() - chrono::Duration::days(1)),
            }],
            ..Default::default()
        };
        apply_config_suppressions(&mut findings, &cfg);
        assert!(findings[0].suppressions.is_empty());
    }
}
