//! Markdown reporter — human-readable, GitHub-flavoured.
//!
//! Layout:
//!
//! ```text
//! # ph0b0s scan report
//!
//! - **Run ID:** `<ulid>`
//! - **Started:** <RFC3339>
//! - **Finished:** <RFC3339>
//! - **Findings:** N (X critical, Y high, ...)
//!
//! ## Summary by severity
//!
//! | Severity | Count |
//! |---|---|
//! | Critical | n |
//! | ...      | n |
//!
//! ## Findings
//!
//! ### `<rule_id>` — <title> _(Severity)_
//! ...
//! ```

use std::fmt::Write;

use async_trait::async_trait;
use ph0b0s_core::error::ReportError;
use ph0b0s_core::finding::{Confidence, Finding, Location};
use ph0b0s_core::report::Reporter;
use ph0b0s_core::scan::ScanResult;
use ph0b0s_core::severity::Level;
use tokio::io::{AsyncWrite, AsyncWriteExt};

#[derive(Default, Clone, Copy)]
pub struct MarkdownReporter;

impl MarkdownReporter {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&self, result: &ScanResult) -> Result<String, ReportError> {
        let mut out = String::with_capacity(2048);

        writeln!(out, "# ph0b0s scan report").ok();
        writeln!(out).ok();
        writeln!(out, "- **Run ID:** `{}`", result.run_id).ok();
        writeln!(out, "- **Started:** {}", result.started_at.to_rfc3339()).ok();
        writeln!(out, "- **Finished:** {}", result.finished_at.to_rfc3339()).ok();

        let counts = severity_counts(&result.findings);
        let total = result.findings.len();
        writeln!(out, "- **Findings:** {} ({})", total, counts_summary(&counts)).ok();
        if !result.errors.is_empty() {
            writeln!(out, "- **Detector errors:** {}", result.errors.len()).ok();
        }
        writeln!(out).ok();

        writeln!(out, "## Summary by severity").ok();
        writeln!(out).ok();
        writeln!(out, "| Severity | Count |").ok();
        writeln!(out, "|---|---|").ok();
        for level in [Level::Critical, Level::High, Level::Medium, Level::Low, Level::None] {
            writeln!(
                out,
                "| {} | {} |",
                level_label(level),
                counts.get(level)
            )
            .ok();
        }
        writeln!(out).ok();

        if !result.errors.is_empty() {
            writeln!(out, "## Detector errors").ok();
            writeln!(out).ok();
            for e in &result.errors {
                writeln!(out, "- **{}**: {}", e.detector_id, e.message).ok();
            }
            writeln!(out).ok();
        }

        if total == 0 {
            writeln!(out, "_No findings._").ok();
            return Ok(out);
        }

        writeln!(out, "## Findings").ok();
        writeln!(out).ok();
        for f in &result.findings {
            render_finding(&mut out, f);
        }

        Ok(out)
    }
}

fn render_finding(out: &mut String, f: &Finding) {
    let level = f.severity.qualitative_bucket();
    writeln!(
        out,
        "### `{}` — {} _( {} )_",
        f.rule_id,
        f.title,
        level_label(level)
    )
    .ok();
    writeln!(out).ok();
    writeln!(out, "- **Detector:** {}", f.detector).ok();
    writeln!(out, "- **Confidence:** {}", confidence_label(f.confidence)).ok();
    writeln!(out, "- **Severity (numeric):** {:.1}", f.severity.numeric()).ok();
    writeln!(out, "- **Location:** {}", format_location(&f.location)).ok();
    writeln!(out, "- **Fingerprint:** `{}`", f.fingerprint.0).ok();
    if !f.suppressions.is_empty() {
        writeln!(
            out,
            "- **Suppressions:** {} hint(s)",
            f.suppressions.len()
        )
        .ok();
    }
    writeln!(out).ok();
    writeln!(out, "{}", f.message.trim_end()).ok();
    writeln!(out).ok();
}

fn format_location(loc: &Location) -> String {
    match loc {
        Location::File {
            path,
            start_line,
            end_line,
            ..
        } => {
            if start_line == end_line {
                format!("`{path}:{start_line}`")
            } else {
                format!("`{path}:{start_line}-{end_line}`")
            }
        }
        Location::Symbolic {
            package,
            version,
            ecosystem,
        } => format!("`{ecosystem}:{package}@{version}`"),
    }
}

fn level_label(level: Level) -> &'static str {
    match level {
        Level::Critical => "Critical",
        Level::High => "High",
        Level::Medium => "Medium",
        Level::Low => "Low",
        Level::None => "None",
    }
}

fn confidence_label(c: Confidence) -> &'static str {
    match c {
        Confidence::Low => "Low",
        Confidence::Medium => "Medium",
        Confidence::High => "High",
    }
}

#[derive(Default, Debug)]
struct LevelCounts {
    critical: usize,
    high: usize,
    medium: usize,
    low: usize,
    none: usize,
}

impl LevelCounts {
    fn get(&self, level: Level) -> usize {
        match level {
            Level::Critical => self.critical,
            Level::High => self.high,
            Level::Medium => self.medium,
            Level::Low => self.low,
            Level::None => self.none,
        }
    }
    fn add(&mut self, level: Level) {
        match level {
            Level::Critical => self.critical += 1,
            Level::High => self.high += 1,
            Level::Medium => self.medium += 1,
            Level::Low => self.low += 1,
            Level::None => self.none += 1,
        }
    }
}

fn severity_counts(findings: &[Finding]) -> LevelCounts {
    let mut c = LevelCounts::default();
    for f in findings {
        c.add(f.severity.qualitative_bucket());
    }
    c
}

fn counts_summary(c: &LevelCounts) -> String {
    let mut parts = Vec::new();
    for (level, n) in [
        (Level::Critical, c.critical),
        (Level::High, c.high),
        (Level::Medium, c.medium),
        (Level::Low, c.low),
        (Level::None, c.none),
    ] {
        if n > 0 {
            parts.push(format!("{n} {}", level_label(level).to_lowercase()));
        }
    }
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(", ")
    }
}

#[async_trait]
impl Reporter for MarkdownReporter {
    fn name(&self) -> &'static str {
        "markdown"
    }

    async fn write(
        &self,
        result: &ScanResult,
        sink: &mut (dyn AsyncWrite + Send + Unpin),
    ) -> Result<(), ReportError> {
        let body = self.render(result)?;
        sink.write_all(body.as_bytes()).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_test_support::sample_scan_result;

    #[tokio::test]
    async fn empty_run_is_handled() {
        let r = sample_scan_result(0);
        let body = MarkdownReporter.render(&r).unwrap();
        assert!(body.contains("_No findings._"));
        assert!(!body.contains("## Findings"));
    }

    #[tokio::test]
    async fn render_includes_each_rule_id() {
        let r = sample_scan_result(3);
        let body = MarkdownReporter.render(&r).unwrap();
        for f in &r.findings {
            assert!(body.contains(&f.rule_id), "missing rule_id {}", f.rule_id);
        }
    }

    #[tokio::test]
    async fn render_includes_severity_buckets() {
        let r = sample_scan_result(3);
        let body = MarkdownReporter.render(&r).unwrap();
        // sample_scan_result(3) cycles Low / Medium / High
        assert!(body.contains("| Low | 1 |"));
        assert!(body.contains("| Medium | 1 |"));
        assert!(body.contains("| High | 1 |"));
    }

    #[tokio::test]
    async fn markdown_snapshot_three_findings() {
        let r = sample_scan_result(3);
        let body = MarkdownReporter.render(&r).unwrap();
        insta::assert_snapshot!("markdown_three_findings", body);
    }

    #[tokio::test]
    async fn markdown_snapshot_empty() {
        let r = sample_scan_result(0);
        let body = MarkdownReporter.render(&r).unwrap();
        insta::assert_snapshot!("markdown_empty", body);
    }
}
