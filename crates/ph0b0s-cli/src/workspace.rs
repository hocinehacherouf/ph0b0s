//! Materialize a `Target` into a `Workspace`.
//!
//! v1 supports `Target::LocalDirectory { path }` (in-place, no copy) and
//! `Target::LocalRepo { path, rev }` (in-place if `rev` is None; otherwise
//! shells out to `git clone --depth 1 --branch <rev>` into a tempdir).

use ph0b0s_core::error::CoreError;
use ph0b0s_core::target::{Target, Workspace, WorkspaceGuard};

/// Synchronously prepare a workspace from `target`. Returns an error if the
/// target points at a path that doesn't exist or — for git revisions — if
/// `git` isn't on PATH.
pub async fn prepare(target: &Target) -> Result<Workspace, CoreError> {
    match target {
        Target::LocalDirectory { path } => {
            if !path.is_dir() {
                return Err(CoreError::WorkspacePrep(format!(
                    "not a directory: {}",
                    path.display()
                )));
            }
            Ok(Workspace {
                root: path.clone(),
                guard: WorkspaceGuard::InPlace,
            })
        }
        Target::LocalRepo { path, rev } => match rev {
            None => {
                if !path.is_dir() {
                    return Err(CoreError::WorkspacePrep(format!(
                        "not a directory: {}",
                        path.display()
                    )));
                }
                Ok(Workspace {
                    root: path.clone(),
                    guard: WorkspaceGuard::InPlace,
                })
            }
            Some(rev) => clone_at_rev(path, rev).await,
        },
    }
}

async fn clone_at_rev(
    src: &std::path::Path,
    rev: &str,
) -> Result<Workspace, CoreError> {
    let td = tempfile::TempDir::new()
        .map_err(|e| CoreError::WorkspacePrep(format!("tempdir: {e}")))?;
    let dest = td.path();
    let status = tokio::process::Command::new("git")
        .args(["clone", "--depth", "1", "--branch", rev])
        .arg(src)
        .arg(dest)
        .status()
        .await
        .map_err(|e| CoreError::WorkspacePrep(format!("git clone: {e}")))?;
    if !status.success() {
        return Err(CoreError::WorkspacePrep(format!(
            "git clone exited {status}"
        )));
    }
    Ok(Workspace {
        root: td.path().to_path_buf(),
        guard: WorkspaceGuard::Tempdir(td),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_directory_returns_inplace_workspace() {
        let td = tempfile::tempdir().unwrap();
        let target = Target::LocalDirectory {
            path: td.path().to_path_buf(),
        };
        let ws = prepare(&target).await.unwrap();
        assert_eq!(ws.root, td.path());
        assert!(matches!(ws.guard, WorkspaceGuard::InPlace));
    }

    #[tokio::test]
    async fn missing_directory_returns_error() {
        let target = Target::LocalDirectory {
            path: std::path::PathBuf::from("/no/such/path/ever-1234"),
        };
        let err = prepare(&target).await.unwrap_err();
        assert!(matches!(err, CoreError::WorkspacePrep(_)));
    }

    #[tokio::test]
    async fn local_repo_without_rev_is_inplace() {
        let td = tempfile::tempdir().unwrap();
        let target = Target::LocalRepo {
            path: td.path().to_path_buf(),
            rev: None,
        };
        let ws = prepare(&target).await.unwrap();
        assert_eq!(ws.root, td.path());
        assert!(matches!(ws.guard, WorkspaceGuard::InPlace));
    }
}
