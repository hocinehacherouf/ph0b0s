//! Scan targets and the workspace abstraction. Only local-disk variants
//! are wired in v1; the trait already accepts the rest so future slices
//! (HTTP probes, browser sessions) plug in without changing detectors.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::scan::ScanCtx;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Target {
    /// A git repository on disk. If `rev` is set, the materializer performs
    /// a shallow clone of that revision into a tempdir.
    LocalRepo { path: PathBuf, rev: Option<String> },

    /// A directory on disk to scan in place; no clone, no copy.
    LocalDirectory { path: PathBuf },
    // Future variants intentionally omitted from v1; new variants are
    // additive and live behind cargo features once their materializer
    // exists.
}

/// A drop-guarded scan workspace. The guard is responsible for cleaning
/// any tempdir the materializer created.
#[derive(Debug)]
pub struct Workspace {
    pub root: PathBuf,
    /// Holds the tempdir alive for the duration of the scan; for
    /// `LocalDirectory` (in-place scans) this is `None`.
    pub guard: WorkspaceGuard,
}

#[derive(Debug)]
pub enum WorkspaceGuard {
    /// Tempdir owned by this struct; dropping cleans up.
    Tempdir(tempfile::TempDir),
    /// User's directory; nothing to clean.
    InPlace,
}

#[async_trait]
pub trait TargetMaterializer: Send + Sync {
    async fn prepare(
        &self,
        target: &Target,
        ctx: &ScanCtx,
    ) -> Result<Workspace, CoreError>;
}
