//! Persistent finding store. Implementations live in `ph0b0s-storage`.

use async_trait::async_trait;
use ulid::Ulid;

use crate::error::StoreError;
use crate::finding::{Finding, Fingerprint};
use crate::scan::{ScanRequest, ScanResult, ScanStats};

#[async_trait]
pub trait FindingStore: Send + Sync {
    /// Mark any pre-existing run rows that lack a `finished_at` as
    /// aborted. Called once at process startup.
    async fn cleanup_orphan_runs(&self) -> Result<usize, StoreError>;

    async fn begin_run(&self, req: &ScanRequest) -> Result<Ulid, StoreError>;

    async fn record(&self, run_id: Ulid, finding: &Finding) -> Result<(), StoreError>;

    async fn finish_run(&self, run_id: Ulid, stats: &ScanStats) -> Result<(), StoreError>;

    async fn load_run(&self, run_id: Ulid) -> Result<ScanResult, StoreError>;

    /// Collapse identical fingerprints inside `run_id`. Implementations keep
    /// the highest-severity / highest-confidence representative and tag the
    /// others with `superseded_by = <kept_id>` (kept rows are still
    /// retrievable for traceability).
    async fn dedup(&self, run_id: Ulid) -> Result<usize, StoreError>;

    /// Persist a manual suppression for a finding fingerprint.
    async fn suppress(
        &self,
        fingerprint: &Fingerprint,
        reason: &str,
    ) -> Result<(), StoreError>;
}
