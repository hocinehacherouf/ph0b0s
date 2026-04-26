//! `ph0b0s` CLI entrypoint. Parses args, loads config, dispatches to the
//! subcommand impls in `commands.rs` and `scan.rs`.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ph0b0s_cli::commands::{self, ReportFormat};
use ph0b0s_cli::config::Config;
use ph0b0s_cli::provider;
use ph0b0s_cli::scan::{self, ScanArgs};
use ph0b0s_core::severity::Level;
use ph0b0s_core::target::Target;

#[derive(Parser, Debug)]
#[command(
    name = "ph0b0s",
    about = "Vendor-neutral agentic AppSec scanner",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Scan a target and emit reports.
    Scan {
        /// Path to a directory to scan.
        path: PathBuf,
        /// Output SARIF path. Defaults to `report.sarif` in the cwd.
        #[arg(long, default_value = "report.sarif")]
        output: Option<PathBuf>,
        /// Also emit a Markdown report at this path.
        #[arg(long)]
        markdown: Option<PathBuf>,
        /// Also emit a JSON report at this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Abort the run on the first detector failure.
        #[arg(long)]
        strict: bool,
        /// Run only the named detector(s); repeat for multiples.
        #[arg(long = "detector", value_name = "ID")]
        detectors: Vec<String>,
        /// Print a per-role token + cost breakdown after the run.
        #[arg(long)]
        report_cost: bool,
    },

    /// List built-in detectors and their enabled state.
    Detectors {
        #[command(subcommand)]
        cmd: DetectorsCmd,
    },

    /// Re-emit a previous scan's report from the DB.
    Report {
        #[command(subcommand)]
        cmd: ReportCmd,
    },

    /// Triage operations on findings.
    Triage {
        #[command(subcommand)]
        cmd: TriageCmd,
    },

    /// Configuration helpers.
    Config {
        #[command(subcommand)]
        cmd: ConfigCmd,
    },

    /// MCP server helpers.
    Mcp {
        #[command(subcommand)]
        cmd: McpCmd,
    },
}

#[derive(Subcommand, Debug)]
enum DetectorsCmd {
    /// List built-in detectors.
    List {
        /// Only show detectors enabled in current config.
        #[arg(long)]
        enabled_only: bool,
        /// Emit JSON instead of a text table.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ReportCmd {
    /// Show a report for a previous run.
    Show {
        /// Run ULID. Defaults to the most recent run.
        run_id: Option<String>,
        /// Output format.
        #[arg(long, default_value = "sarif")]
        format: String,
    },
}

#[derive(Subcommand, Debug)]
enum TriageCmd {
    /// Persist a manual suppression for a finding fingerprint.
    Suppress {
        /// Fingerprint to suppress (hex string from a Finding).
        fingerprint: String,
        /// Reason recorded with the suppression.
        #[arg(long)]
        reason: String,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCmd {
    /// Resolve the effective config (with secrets redacted).
    Check,
}

#[derive(Subcommand, Debug)]
enum McpCmd {
    /// List MCP servers configured for the current project.
    List {
        /// Emit JSON instead of a text table.
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let config = Config::load()?;

    match cli.cmd {
        Cmd::Scan {
            path,
            output,
            markdown,
            json,
            strict,
            detectors,
            report_cost,
        } => {
            let agent = provider::build_from_env()?;
            let target = Target::LocalDirectory { path };
            let args = ScanArgs {
                target,
                output,
                markdown,
                json,
                strict,
                detector_filter: detectors,
                report_cost,
            };
            let outcome = scan::run(args, config, agent).await?;
            print_summary(&outcome);
            Ok(())
        }
        Cmd::Detectors {
            cmd: DetectorsCmd::List { enabled_only, json },
        } => commands::detectors_list(&config, enabled_only, json).await,
        Cmd::Report {
            cmd: ReportCmd::Show { run_id, format },
        } => {
            let f: ReportFormat = format.parse().map_err(|e: String| anyhow::anyhow!("{e}"))?;
            commands::report_show(&config, run_id, f).await
        }
        Cmd::Triage {
            cmd:
                TriageCmd::Suppress {
                    fingerprint,
                    reason,
                },
        } => commands::triage_suppress(&config, fingerprint, reason).await,
        Cmd::Config {
            cmd: ConfigCmd::Check,
        } => commands::config_check(&config).await,
        Cmd::Mcp {
            cmd: McpCmd::List { json },
        } => commands::mcp_list(&config, json).await,
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_env("PH0B0S_LOG")
        .unwrap_or_else(|_| EnvFilter::new("warn,ph0b0s=info"));
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init();
}

fn print_summary(outcome: &scan::ScanOutcome) {
    use std::collections::BTreeMap;

    let r = &outcome.result;
    let mut by_level: BTreeMap<Level, u64> = BTreeMap::new();
    for f in &r.findings {
        *by_level.entry(f.severity.qualitative_bucket()).or_insert(0) += 1;
    }

    eprintln!();
    eprintln!("ph0b0s scan complete");
    eprintln!("  run_id:   {}", outcome.run_id);
    eprintln!("  findings: {}", r.findings.len());
    if outcome.deduped > 0 {
        eprintln!("  deduped:  {}", outcome.deduped);
    }
    for level in [
        Level::Critical,
        Level::High,
        Level::Medium,
        Level::Low,
        Level::None,
    ] {
        let n = by_level.get(&level).copied().unwrap_or(0);
        if n > 0 {
            eprintln!("    {level:?}: {n}");
        }
    }
    if !r.errors.is_empty() {
        eprintln!("  detector errors:");
        for e in &r.errors {
            eprintln!("    {}: {}", e.detector_id, e.message);
        }
    }
}
