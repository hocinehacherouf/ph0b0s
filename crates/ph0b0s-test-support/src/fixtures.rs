//! Reusable fixture data for tests across the workspace.
//!
//! All timestamps and IDs are deterministic so snapshot tests stay stable.

use std::path::PathBuf;

use chrono::{DateTime, TimeZone, Utc};
use ph0b0s_core::error::CoreError;
use ph0b0s_core::finding::{Confidence, Evidence, Finding, Location, SanitizationState};
use ph0b0s_core::scan::{ScanResult, ScanStats};
use ph0b0s_core::severity::{Level, Severity};
use ph0b0s_core::target::{Workspace, WorkspaceGuard};
use ulid::Ulid;

/// Stable, deterministic ULID used across snapshot tests.
pub fn deterministic_run_id() -> Ulid {
    // 16 bytes — chosen arbitrarily but fixed forever.
    Ulid::from_bytes([
        0x01, 0x8F, 0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x12, 0x34, 0x56,
        0x78,
    ])
}

/// Stable timestamp (`2026-04-01T00:00:00Z`) for fixture data.
pub fn fixed_timestamp() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 1, 0, 0, 0).unwrap()
}

/// A single deterministic `Finding` suitable for round-trip and snapshot
/// tests. ID and `created_at` are stable across calls.
pub fn sample_finding() -> Finding {
    let location = Location::File {
        path: "src/main.rs".to_owned(),
        start_line: 12,
        end_line: 12,
        start_col: Some(8),
        end_col: Some(20),
    };
    Finding {
        id: deterministic_run_id(),
        rule_id: "ph0b0s.test.sample".to_owned(),
        detector: "test".to_owned(),
        severity: Severity::Qualitative(Level::Medium),
        confidence: Confidence::Medium,
        title: "Sample finding".to_owned(),
        message: "fixture finding for tests".to_owned(),
        location: location.clone(),
        evidence: vec![Evidence::Note("fixture".to_owned())],
        fingerprint: ph0b0s_core::finding::Fingerprint::compute(
            "ph0b0s.test.sample",
            &location,
            b"fixture",
        ),
        sanitization: SanitizationState::Raw,
        suppressions: Vec::new(),
        created_at: fixed_timestamp(),
    }
}

/// A `ScanResult` with `n` deterministic findings cycling Low/Medium/High
/// severity. Suitable for reporter snapshot tests.
pub fn sample_scan_result(n: usize) -> ScanResult {
    let levels = [Level::Low, Level::Medium, Level::High];

    let started_at = fixed_timestamp();
    let finished_at = fixed_timestamp();

    let mut findings = Vec::with_capacity(n);
    for i in 0..n {
        let level = levels[i % levels.len()];
        let location = Location::File {
            path: format!("src/file_{i}.rs"),
            start_line: (i as u32) + 1,
            end_line: (i as u32) + 1,
            start_col: None,
            end_col: None,
        };
        let rule_id = format!("ph0b0s.test.rule-{i}");
        let fingerprint = ph0b0s_core::finding::Fingerprint::compute(
            &rule_id,
            &location,
            format!("ev-{i}").as_bytes(),
        );
        // Stable per-index ID.
        let mut bytes = [0u8; 16];
        bytes[15] = (i & 0xff) as u8;
        bytes[14] = ((i >> 8) & 0xff) as u8;
        let id = Ulid::from_bytes(bytes);

        findings.push(Finding {
            id,
            rule_id,
            detector: "test".to_owned(),
            severity: Severity::Qualitative(level),
            confidence: Confidence::Medium,
            title: format!("Finding {i}"),
            message: format!("synthetic finding number {i}"),
            location,
            evidence: vec![Evidence::Note(format!("ev-{i}"))],
            fingerprint,
            sanitization: SanitizationState::Raw,
            suppressions: Vec::new(),
            created_at: fixed_timestamp(),
        });
    }

    ScanResult {
        run_id: deterministic_run_id(),
        started_at,
        finished_at,
        findings,
        stats: ScanStats::default(),
        errors: Vec::new(),
    }
}

/// Empty tempdir wrapped in a `Workspace`. Drops the tempdir when the
/// guard goes out of scope.
pub fn temp_workspace() -> Result<Workspace, CoreError> {
    let td = tempfile::TempDir::new()?;
    Ok(Workspace {
        root: td.path().to_path_buf(),
        guard: WorkspaceGuard::Tempdir(td),
    })
}

/// Tempdir-backed `Workspace` seeded with the given relative `(path, body)`
/// pairs. Parent directories are created as needed.
pub fn temp_workspace_with(files: &[(&str, &str)]) -> Result<Workspace, CoreError> {
    let td = tempfile::TempDir::new()?;
    let root: PathBuf = td.path().to_path_buf();
    for (rel, body) in files {
        let full = root.join(rel);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&full, body)?;
    }
    Ok(Workspace {
        root,
        guard: WorkspaceGuard::Tempdir(td),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_run_id_is_stable_across_calls() {
        let a = deterministic_run_id();
        let b = deterministic_run_id();
        assert_eq!(a, b);
    }

    #[test]
    fn fixed_timestamp_is_stable_across_calls() {
        assert_eq!(fixed_timestamp(), fixed_timestamp());
    }

    #[test]
    fn sample_finding_round_trips_via_serde() {
        let f = sample_finding();
        let j = serde_json::to_string(&f).unwrap();
        let f2: Finding = serde_json::from_str(&j).unwrap();
        assert_eq!(f, f2);
    }

    #[test]
    fn sample_finding_uses_stable_id_and_timestamp() {
        let a = sample_finding();
        let b = sample_finding();
        assert_eq!(a.id, b.id);
        assert_eq!(a.created_at, b.created_at);
        assert_eq!(a.fingerprint, b.fingerprint);
    }

    #[test]
    fn sample_scan_result_has_n_findings_and_stable_ids() {
        let r1 = sample_scan_result(5);
        let r2 = sample_scan_result(5);
        assert_eq!(r1.findings.len(), 5);
        assert_eq!(
            r1.findings.iter().map(|f| f.id).collect::<Vec<_>>(),
            r2.findings.iter().map(|f| f.id).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn sample_scan_result_severity_cycles_low_medium_high() {
        let r = sample_scan_result(6);
        let levels: Vec<Level> = r
            .findings
            .iter()
            .map(|f| f.severity.qualitative_bucket())
            .collect();
        assert_eq!(
            levels,
            vec![
                Level::Low,
                Level::Medium,
                Level::High,
                Level::Low,
                Level::Medium,
                Level::High
            ]
        );
    }

    #[test]
    fn temp_workspace_root_exists_and_is_writable() {
        let ws = temp_workspace().expect("temp workspace");
        assert!(ws.root.is_dir());
        let target = ws.root.join("hello.txt");
        std::fs::write(&target, b"hi").expect("write into tempdir");
        assert_eq!(std::fs::read(&target).unwrap(), b"hi");
    }

    #[test]
    fn temp_workspace_with_seeds_listed_files() {
        let ws = temp_workspace_with(&[("a.txt", "alpha"), ("nested/b.txt", "bravo")])
            .expect("seeded workspace");
        assert_eq!(
            std::fs::read_to_string(ws.root.join("a.txt")).unwrap(),
            "alpha"
        );
        assert_eq!(
            std::fs::read_to_string(ws.root.join("nested").join("b.txt")).unwrap(),
            "bravo"
        );
    }

    #[test]
    fn temp_workspace_cleans_up_on_drop() {
        let path = {
            let ws = temp_workspace().expect("ws");
            let p = ws.root.clone();
            assert!(p.is_dir());
            p
        };
        // Tempdir gone after the guard dropped.
        assert!(!path.exists());
    }
}
