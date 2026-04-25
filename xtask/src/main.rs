//! ph0b0s xtask CI helpers.
//!
//! Subcommands:
//!   `check-vendor` — scan workspace `src/` trees and fail if any crate not in
//!   the allow-list imports a vendor or `adk-*` crate.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use regex::Regex;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "xtask", about = "ph0b0s CI helpers")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fail if any non-allow-listed crate imports vendor / adk-* crates.
    CheckVendor,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::CheckVendor => check_vendor(),
    }
}

const ALLOW: &[&str] = &["ph0b0s-llm-adk", "ph0b0s-cli"];

fn check_vendor() -> Result<()> {
    let banned = Regex::new(
        r"(?m)^\s*use\s+(adk_[a-z0-9_]+|anthropic|openai|genai|google_genai|aws_sdk_bedrock|ollama_rs|rmcp)(::|;)",
    )?;

    let workspace_root = workspace_root()?;
    let crates_dir = workspace_root.join("crates");
    let mut violations: Vec<String> = Vec::new();

    for entry in std::fs::read_dir(&crates_dir).context("read crates/")? {
        let dir = entry?.path();
        if !dir.is_dir() {
            continue;
        }
        let crate_name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        if ALLOW.contains(&crate_name.as_str()) {
            continue;
        }
        let src = dir.join("src");
        if !src.exists() {
            continue;
        }
        for f in WalkDir::new(&src)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let body = std::fs::read_to_string(f.path())
                .with_context(|| format!("read {}", f.path().display()))?;
            if banned.is_match(&body) {
                violations.push(format!(
                    "{}: vendor-coupling violation in {}",
                    crate_name,
                    f.path().display()
                ));
            }
        }
    }

    if violations.is_empty() {
        println!("vendor-coupling: OK");
        Ok(())
    } else {
        for v in &violations {
            eprintln!("{v}");
        }
        bail!("{} vendor-coupling violation(s)", violations.len())
    }
}

fn workspace_root() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("current_exe")?;
    let mut p: &Path = &exe;
    while let Some(parent) = p.parent() {
        if parent.join("Cargo.toml").exists() && parent.join("crates").is_dir() {
            return Ok(parent.to_path_buf());
        }
        p = parent;
    }
    let cwd = std::env::current_dir()?;
    if cwd.join("Cargo.toml").exists() {
        return Ok(cwd);
    }
    bail!("could not locate workspace root from exe path or cwd")
}
