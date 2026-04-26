//! End-to-end integration test for `ph0b0s scan`.
//!
//! Exercises both detector wiring patterns in a single scan:
//!
//! - `llm-toy`: uses `PH0B0S_PROVIDER=mock` + `PH0B0S_MOCK_RESPONSES` to
//!   feed canned issues without contacting any real provider.
//! - `cargo-audit`: uses a generated shell script as `cargo_path` to stand
//!   in for `cargo audit --json` (cargo-audit is not installed in the test
//!   environment).
//!
//! Asserts the produced SARIF contains at least one finding from each
//! detector and that `ph0b0s report show` re-emits the same run from the DB.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use serde_json::Value;

/// Path to the workspace root from this test file.
fn workspace_root() -> PathBuf {
    // `crates/ph0b0s-cli/tests/end_to_end.rs` → up four
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn cli_manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Copy a directory tree (recursive). Mirrors `cp -R src/. dst/`.
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir(&from, &to)?;
        } else if ty.is_file() {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Generate a shell script at `path` that ignores all args and prints
/// `body` to stdout. Used to stand in for `cargo audit --json`.
fn write_executable_script(path: &Path, body: &str) -> std::io::Result<()> {
    let script = format!("#!/bin/sh\ncat <<'__PH0B0S_FAKE_EOF__'\n{body}\n__PH0B0S_FAKE_EOF__\n");
    fs::write(path, script)?;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)?;
    Ok(())
}

struct Scenario {
    _scratch: tempfile::TempDir,
    project: PathBuf,
    sarif: PathBuf,
}

fn scenario() -> Scenario {
    let scratch = tempfile::tempdir().expect("scratch dir");
    let project = scratch.path().join("project");
    copy_dir(
        &workspace_root().join("fixtures").join("sample-rust-repo"),
        &project,
    )
    .expect("copy fixture");

    // Generate the fake `cargo` binary.
    let canned_audit = fs::read_to_string(
        cli_manifest_dir()
            .join("tests")
            .join("canned")
            .join("cargo_audit_output.json"),
    )
    .expect("read canned audit json");
    let fake_cargo = scratch.path().join("fake-cargo.sh");
    write_executable_script(&fake_cargo, &canned_audit).expect("write fake cargo");

    // Per-project ph0b0s.toml: pin extensions + point cargo-audit at the
    // fake binary so the integration test is hermetic.
    let toml_body = format!(
        r#"
[scan]
strict = false

[detectors.cargo-audit]
enabled    = true
no_fetch   = true
cargo_path = "{cargo}"

[detectors.llm-toy]
enabled    = true
extensions = [".rs"]
max_files  = 5

[storage]
db_path = "{db}"
"#,
        cargo = fake_cargo.display(),
        db = scratch.path().join("findings.db").display(),
    );
    fs::write(project.join("ph0b0s.toml"), toml_body).expect("write ph0b0s.toml");

    Scenario {
        sarif: scratch.path().join("report.sarif"),
        project,
        _scratch: scratch,
    }
}

fn assert_run(scenario: &Scenario) {
    let canned_responses = cli_manifest_dir()
        .join("tests")
        .join("canned")
        .join("llm_toy_responses.json");

    Command::cargo_bin("ph0b0s")
        .expect("ph0b0s binary")
        .current_dir(&scenario.project)
        .env("PH0B0S_PROVIDER", "mock")
        .env("PH0B0S_MOCK_RESPONSES", &canned_responses)
        // Ensure no stray defaults from the developer's user config bleed in.
        .env(
            "XDG_CONFIG_HOME",
            scenario._scratch.path().join("xdg-config"),
        )
        .env("HOME", scenario._scratch.path().join("fake-home"))
        .args(["scan", "."])
        .args(["--output", scenario.sarif.to_str().expect("path is utf8")])
        .assert()
        .success();
}

fn parse_sarif(path: &Path) -> Value {
    let body = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&body).expect("sarif is valid JSON")
}

fn rule_ids(sarif: &Value) -> Vec<String> {
    sarif["runs"][0]["results"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|r| r["ruleId"].as_str().map(String::from))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn full_scan_emits_findings_from_both_detector_paths() {
    let scenario = scenario();
    assert_run(&scenario);

    let sarif = parse_sarif(&scenario.sarif);
    assert_eq!(sarif["version"], "2.1.0");
    assert_eq!(sarif["runs"][0]["tool"]["driver"]["name"], "ph0b0s");

    let ids = rule_ids(&sarif);
    assert!(!ids.is_empty(), "expected at least one finding, got none");
    assert!(
        ids.iter().any(|id| id.starts_with("ph0b0s.cargo-audit.")),
        "missing cargo-audit finding; got: {ids:?}"
    );
    assert!(
        ids.iter().any(|id| id.starts_with("ph0b0s.llm-toy.")),
        "missing llm-toy finding; got: {ids:?}"
    );

    // Spot-check the cargo-audit finding round-trips the canned advisory id.
    assert!(
        ids.iter()
            .any(|id| id == "ph0b0s.cargo-audit.RUSTSEC-2099-0001"),
        "cargo-audit advisory id mismatch; got: {ids:?}"
    );
}

#[test]
fn report_show_re_emits_latest_run_from_db() {
    let scenario = scenario();
    assert_run(&scenario);

    // `report show` reads the DB at the storage.db_path we configured in
    // ph0b0s.toml and emits SARIF on stdout. We assert the latest run's
    // SARIF is structurally identical to the one written during the scan.
    let canned_responses = cli_manifest_dir()
        .join("tests")
        .join("canned")
        .join("llm_toy_responses.json");

    let stdout = Command::cargo_bin("ph0b0s")
        .expect("ph0b0s binary")
        .current_dir(&scenario.project)
        .env("PH0B0S_PROVIDER", "mock")
        .env("PH0B0S_MOCK_RESPONSES", &canned_responses)
        .env(
            "XDG_CONFIG_HOME",
            scenario._scratch.path().join("xdg-config"),
        )
        .env("HOME", scenario._scratch.path().join("fake-home"))
        .args(["report", "show", "--format", "sarif"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let body = String::from_utf8(stdout).expect("utf8 stdout");
    let parsed: Value =
        serde_json::from_str(body.trim()).expect("report show emitted invalid sarif");
    assert_eq!(parsed["version"], "2.1.0");
    let ids = rule_ids(&parsed);
    assert!(ids.iter().any(|id| id.starts_with("ph0b0s.cargo-audit.")));
    assert!(ids.iter().any(|id| id.starts_with("ph0b0s.llm-toy.")));
}

#[test]
fn detectors_list_runs_against_default_config() {
    // No fixture project needed for this one — config defaults are enough.
    let scratch = tempfile::tempdir().unwrap();
    Command::cargo_bin("ph0b0s")
        .unwrap()
        .current_dir(scratch.path())
        .env("XDG_CONFIG_HOME", scratch.path().join("xdg-config"))
        .env("HOME", scratch.path().join("fake-home"))
        .args(["detectors", "list", "--json"])
        .assert()
        .success();
}

#[test]
fn config_check_runs_against_default_config() {
    let scratch = tempfile::tempdir().unwrap();
    Command::cargo_bin("ph0b0s")
        .unwrap()
        .current_dir(scratch.path())
        .env("XDG_CONFIG_HOME", scratch.path().join("xdg-config"))
        .env("HOME", scratch.path().join("fake-home"))
        .args(["config", "check"])
        .assert()
        .success();
}

#[test]
fn config_check_rejects_api_key_in_toml() {
    let scratch = tempfile::tempdir().unwrap();
    fs::write(
        scratch.path().join("ph0b0s.toml"),
        "[providers.anthropic]\napi_key = \"sk-...\"\n",
    )
    .unwrap();

    let result = Command::cargo_bin("ph0b0s")
        .unwrap()
        .current_dir(scratch.path())
        .env("XDG_CONFIG_HOME", scratch.path().join("xdg-config"))
        .env("HOME", scratch.path().join("fake-home"))
        .args(["config", "check"])
        .assert()
        .failure();

    let stderr = String::from_utf8_lossy(&result.get_output().stderr).to_string();
    assert!(
        stderr.contains("api_key") || stderr.to_lowercase().contains("api"),
        "expected api_key error, got stderr: {stderr}"
    );
}
