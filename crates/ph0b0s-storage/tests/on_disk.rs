//! On-disk integration test: opens a fresh SQLite file, runs migrations,
//! round-trips a scan, drops the store, reopens the same file, and verifies
//! the data survives.

use ph0b0s_core::scan::{
    DetectorFilter, ScanOptions, ScanRequest, ScanStats,
};
use ph0b0s_core::store::FindingStore;
use ph0b0s_core::target::Target;
use ph0b0s_storage::SqliteFindingStore;
use ph0b0s_test_support::{deterministic_run_id, sample_finding};

#[tokio::test]
async fn on_disk_open_close_reopen_preserves_data() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("findings.db");
    let run_id = deterministic_run_id();

    // First open: migrate, write a finding, finish the run.
    {
        let store = SqliteFindingStore::open(&path)
            .await
            .expect("open fresh db");
        store
            .begin_run(&ScanRequest {
                run_id,
                target: Target::LocalDirectory {
                    path: std::path::PathBuf::from("/tmp/x"),
                },
                detector_filter: DetectorFilter::All,
                options: ScanOptions::default(),
                detector_params: Default::default(),
            })
            .await
            .expect("begin_run");
        store.record(run_id, &sample_finding()).await.expect("record");
        store
            .finish_run(run_id, &ScanStats::default())
            .await
            .expect("finish_run");
    }

    // File on disk should now exist.
    assert!(path.exists(), "expected SQLite file to exist after first open");

    // Reopen and confirm we can load the run back.
    let store2 = SqliteFindingStore::open(&path).await.expect("reopen");
    let loaded = store2.load_run(run_id).await.expect("load_run");
    assert_eq!(loaded.run_id, run_id);
    assert_eq!(loaded.findings.len(), 1);
    assert_eq!(loaded.findings[0], sample_finding());
}

#[tokio::test]
async fn on_disk_cleanup_orphan_runs_runs_at_startup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("findings.db");
    let run_id = deterministic_run_id();

    {
        let store = SqliteFindingStore::open(&path).await.unwrap();
        store
            .begin_run(&ScanRequest {
                run_id,
                target: Target::LocalDirectory {
                    path: std::path::PathBuf::from("/tmp/x"),
                },
                detector_filter: DetectorFilter::All,
                options: ScanOptions::default(),
                detector_params: Default::default(),
            })
            .await
            .unwrap();
        // Do NOT finish — simulate crash by dropping mid-run.
    }

    // Reopen and run the orphan-cleanup pass.
    let store2 = SqliteFindingStore::open(&path).await.unwrap();
    let aborted = store2.cleanup_orphan_runs().await.unwrap();
    assert_eq!(aborted, 1);
}
