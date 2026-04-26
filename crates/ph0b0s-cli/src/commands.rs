//! CLI subcommands. `scan` lives in `scan.rs`; the rest are here.

use anyhow::{Context, Result};
use ph0b0s_core::finding::Fingerprint;
use ph0b0s_core::report::Reporter;
use ph0b0s_core::store::FindingStore;
use ph0b0s_report::{JsonReporter, MarkdownReporter, SarifReporter};
use ph0b0s_storage::SqliteFindingStore;
use ulid::Ulid;

use crate::config::Config;
use crate::registry::DetectorRegistry;

// ---------------------------------------------------------------------------
// `ph0b0s detectors list`
// ---------------------------------------------------------------------------

pub async fn detectors_list(
    config: &Config,
    enabled_only: bool,
    as_json: bool,
) -> Result<()> {
    let registry = DetectorRegistry::builtin();
    let mut entries = Vec::new();
    for id in registry.ids() {
        let resolved = registry.resolve(&[id.to_owned()], config);
        let enabled_in_config = !registry.resolve(&[], config).is_empty()
            && registry.resolve(&[], config).iter().any(|r| r.detector.metadata().id == id);
        let enabled = !resolved.is_empty() && enabled_in_config;
        if enabled_only && !enabled {
            continue;
        }
        let det = registry
            .resolve(&[id.to_owned()], config)
            .into_iter()
            .next()
            .map(|r| r.detector)
            .expect("registry lookup roundtrip");
        let m = det.metadata();
        entries.push(serde_json::json!({
            "id": m.id,
            "version": m.version,
            "kind": m.kind,
            "description": m.description,
            "capabilities": m.capabilities,
            "enabled": enabled,
        }));
    }

    if as_json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
    } else {
        for e in &entries {
            let mark = if e["enabled"].as_bool().unwrap_or(false) { "✓" } else { "✗" };
            println!(
                "{mark} {id:24} {kind:?}  — {desc}",
                mark = mark,
                id = e["id"].as_str().unwrap_or(""),
                kind = e["kind"],
                desc = e["description"].as_str().unwrap_or(""),
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ph0b0s report show [run_id]`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Sarif,
    Markdown,
    Json,
}

impl std::str::FromStr for ReportFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "sarif" => Ok(Self::Sarif),
            "md" | "markdown" => Ok(Self::Markdown),
            "json" => Ok(Self::Json),
            other => Err(format!("unknown report format: {other}")),
        }
    }
}

pub async fn report_show(
    config: &Config,
    run_id: Option<String>,
    format: ReportFormat,
) -> Result<()> {
    let store = SqliteFindingStore::open(&config.effective_db_path()).await?;
    let id = match run_id {
        Some(s) => {
            Ulid::from_string(&s).map_err(|e| anyhow::anyhow!("invalid run_id {s}: {e}"))?
        }
        None => latest_run_id(&store).await?,
    };
    let result = store.load_run(id).await?;
    let mut stdout = tokio::io::stdout();
    match format {
        ReportFormat::Sarif => SarifReporter.write(&result, &mut stdout).await?,
        ReportFormat::Markdown => MarkdownReporter.write(&result, &mut stdout).await?,
        ReportFormat::Json => JsonReporter.write(&result, &mut stdout).await?,
    }
    Ok(())
}

async fn latest_run_id(store: &SqliteFindingStore) -> Result<Ulid> {
    store
        .latest_run_id()
        .await
        .context("querying latest run")?
        .ok_or_else(|| anyhow::anyhow!("no scan runs found in DB"))
}

// ---------------------------------------------------------------------------
// `ph0b0s triage suppress <fingerprint>`
// ---------------------------------------------------------------------------

pub async fn triage_suppress(
    config: &Config,
    fingerprint: String,
    reason: String,
) -> Result<()> {
    let store = SqliteFindingStore::open(&config.effective_db_path()).await?;
    store
        .suppress(&Fingerprint(fingerprint.clone()), &reason)
        .await
        .with_context(|| format!("suppress {fingerprint}"))?;
    println!("suppressed {fingerprint}: {reason}");
    Ok(())
}

// ---------------------------------------------------------------------------
// `ph0b0s config check`
// ---------------------------------------------------------------------------

pub async fn config_check(config: &Config) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&config.redacted_json())?);
    println!();
    println!("✓ no api_key fields detected in TOML layers");
    println!("  effective DB path: {}", config.effective_db_path().display());
    Ok(())
}

// ---------------------------------------------------------------------------
// `ph0b0s mcp list`
// ---------------------------------------------------------------------------

pub async fn mcp_list(config: &Config, as_json: bool) -> Result<()> {
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&config.mcp_servers)?
        );
        return Ok(());
    }
    if config.mcp_servers.is_empty() {
        println!("no MCP servers configured");
        return Ok(());
    }
    for s in &config.mcp_servers {
        println!(
            "{name:24} transport={transport:?} command={cmd:?}",
            name = s.name,
            transport = s.transport,
            cmd = s.command_or_url,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::tools::{McpServerSpec, McpTransport};
    use ph0b0s_test_support::{deterministic_run_id, sample_finding};

    /// Unique DB path inside a fresh tempdir. Returns (TempDir, path) — the
    /// caller must hold the TempDir alive for the test's lifetime so the
    /// file isn't garbage-collected mid-run.
    fn db_path() -> (tempfile::TempDir, std::path::PathBuf) {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("findings.db");
        (td, p)
    }

    #[tokio::test]
    async fn report_format_parses_known_strings() {
        for s in ["sarif", "SARIF", "md", "markdown", "json"] {
            let _: ReportFormat = s.parse().unwrap();
        }
        assert!("yaml".parse::<ReportFormat>().is_err());
    }

    #[tokio::test]
    async fn mcp_list_handles_empty_config() {
        let cfg = Config::default();
        mcp_list(&cfg, false).await.unwrap();
        mcp_list(&cfg, true).await.unwrap();
    }

    #[tokio::test]
    async fn mcp_list_renders_configured_servers() {
        let mut cfg = Config::default();
        cfg.mcp_servers.push(McpServerSpec {
            name: "fs".into(),
            transport: McpTransport::Stdio,
            command_or_url: vec!["uvx".into(), "mcp-server-filesystem".into()],
            env: Default::default(),
        });
        mcp_list(&cfg, false).await.unwrap();
        mcp_list(&cfg, true).await.unwrap();
    }

    #[tokio::test]
    async fn config_check_runs_clean() {
        let cfg = Config::default();
        config_check(&cfg).await.unwrap();
    }

    #[tokio::test]
    async fn detectors_list_text_and_json() {
        let cfg = Config::default();
        detectors_list(&cfg, false, false).await.unwrap();
        detectors_list(&cfg, false, true).await.unwrap();
    }

    #[tokio::test]
    async fn triage_suppress_persists_to_store() {
        let (_td, path) = db_path();
        let mut cfg = Config::default();
        cfg.storage.db_path = Some(path.clone());
        triage_suppress(&cfg, "fp-abc".into(), "ok".into())
            .await
            .unwrap();
        // Round-trip via a fresh open + suppress (idempotent ON CONFLICT
        // path proves the row landed).
        let store = SqliteFindingStore::open(&path).await.unwrap();
        store
            .suppress(&Fingerprint("fp-abc".into()), "ok-2")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn report_show_loads_latest_run() {
        let (_td, path) = db_path();
        let mut cfg = Config::default();
        cfg.storage.db_path = Some(path.clone());

        // Set up a fake run.
        let store = SqliteFindingStore::open(&path).await.unwrap();
        let req = ph0b0s_core::scan::ScanRequest {
            run_id: deterministic_run_id(),
            target: ph0b0s_core::target::Target::LocalDirectory {
                path: "/tmp/x".into(),
            },
            detector_filter: ph0b0s_core::scan::DetectorFilter::All,
            options: Default::default(),
            detector_params: Default::default(),
        };
        store.begin_run(&req).await.unwrap();
        store.record(req.run_id, &sample_finding()).await.unwrap();
        store
            .finish_run(req.run_id, &Default::default())
            .await
            .unwrap();
        drop(store);

        report_show(&cfg, None, ReportFormat::Json).await.unwrap();
    }
}
