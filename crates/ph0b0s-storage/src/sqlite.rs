//! SQLite implementation of the `FindingStore` seam trait.
//!
//! All queries are runtime (`sqlx::query` / `sqlx::query_as`) — no
//! `DATABASE_URL` required at compile time. JSON-shaped fields round-trip
//! through `serde_json` at the boundary; the DB stores the canonical text
//! form. WAL journal mode and `foreign_keys = ON` are set on every
//! connection acquired from the pool.

use std::path::Path;
use std::str::FromStr;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ph0b0s_core::error::StoreError;
use ph0b0s_core::finding::{
    Confidence, Evidence, Finding, Fingerprint, Location, SanitizationState, SuppressionHint,
};
use ph0b0s_core::scan::{DetectorRunError, ScanRequest, ScanResult, ScanStats};
use ph0b0s_core::severity::Severity;
use ph0b0s_core::store::FindingStore;
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};
use sqlx::{ConnectOptions, Row};
use ulid::Ulid;

/// SQLite-backed implementation of `FindingStore`.
#[derive(Clone)]
pub struct SqliteFindingStore {
    pool: SqlitePool,
}

impl SqliteFindingStore {
    /// Open or create a database at `path` and run all embedded migrations.
    /// Sets WAL journal mode and enables foreign keys on every connection.
    #[tracing::instrument(skip(path), fields(path = %path.as_ref().display()))]
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let opts = SqliteConnectOptions::new()
            .filename(path.as_ref())
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .log_statements(tracing::log::LevelFilter::Trace);

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(opts)
            .await
            .map_err(|e| StoreError::Backend(format!("connect: {e}")))?;

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| StoreError::Backend(format!("migrate: {e}")))?;

        Ok(Self { pool })
    }

    /// In-memory database for tests. Foreign keys enabled.
    /// `max_connections = 1` so the in-memory DB is shared across calls.
    pub async fn open_in_memory() -> Result<Self, StoreError> {
        let opts = SqliteConnectOptions::from_str(":memory:")
            .map_err(|e| StoreError::Backend(format!("opts: {e}")))?
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .map_err(|e| StoreError::Backend(format!("connect: {e}")))?;

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| StoreError::Backend(format!("migrate: {e}")))?;

        Ok(Self { pool })
    }

    /// Test/diagnostic helper — returns the underlying pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Most recently started run, if any. Used by `ph0b0s report show` when
    /// the user omits a run id. Lives on the impl (not the trait) because
    /// it's a CLI affordance, not a core storage primitive.
    pub async fn latest_run_id(&self) -> Result<Option<Ulid>, StoreError> {
        let row = sqlx::query("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let Some(row) = row else { return Ok(None) };
        let s: String = row
            .try_get("run_id")
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ulid::from_string(&s)
            .map(Some)
            .map_err(|e| StoreError::Backend(format!("invalid stored run_id: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Conversions between domain types and DB primitives
// ---------------------------------------------------------------------------

fn confidence_str(c: Confidence) -> &'static str {
    match c {
        Confidence::Low => "low",
        Confidence::Medium => "medium",
        Confidence::High => "high",
    }
}

fn parse_confidence(s: &str) -> Result<Confidence, StoreError> {
    match s {
        "low" => Ok(Confidence::Low),
        "medium" => Ok(Confidence::Medium),
        "high" => Ok(Confidence::High),
        other => Err(StoreError::Backend(format!(
            "invalid confidence in DB: {other:?}"
        ))),
    }
}

fn severity_level_str(s: &Severity) -> &'static str {
    use ph0b0s_core::severity::Level;
    match s.qualitative_bucket() {
        Level::None => "none",
        Level::Low => "low",
        Level::Medium => "medium",
        Level::High => "high",
        Level::Critical => "critical",
    }
}

fn parse_ulid(s: &str) -> Result<Ulid, StoreError> {
    Ulid::from_string(s).map_err(|e| StoreError::Backend(format!("invalid ULID {s:?}: {e}")))
}

fn parse_rfc3339(s: &str) -> Result<DateTime<Utc>, StoreError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| StoreError::Backend(format!("invalid RFC3339 {s:?}: {e}")))
}

// ---------------------------------------------------------------------------
// FindingStore impl
// ---------------------------------------------------------------------------

#[async_trait]
impl FindingStore for SqliteFindingStore {
    #[tracing::instrument(skip(self))]
    async fn cleanup_orphan_runs(&self) -> Result<usize, StoreError> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE runs SET status = 'aborted', finished_at = ? \
             WHERE status = 'started'",
        )
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(result.rows_affected() as usize)
    }

    #[tracing::instrument(skip(self, req), fields(run_id))]
    async fn begin_run(&self, req: &ScanRequest) -> Result<Ulid, StoreError> {
        let started_at = Utc::now();
        let request_json = serde_json::to_string(req)?;
        sqlx::query(
            "INSERT INTO runs (run_id, started_at, status, request) \
             VALUES (?, ?, 'started', ?)",
        )
        .bind(req.run_id.to_string())
        .bind(started_at.to_rfc3339())
        .bind(request_json)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        tracing::Span::current().record("run_id", req.run_id.to_string());
        Ok(req.run_id)
    }

    #[tracing::instrument(skip(self, finding), fields(run_id = %run_id, finding_id))]
    async fn record(&self, run_id: Ulid, finding: &Finding) -> Result<(), StoreError> {
        let severity_numeric = f64::from(finding.severity.numeric());
        let severity_level = severity_level_str(&finding.severity);
        let severity_json = serde_json::to_string(&finding.severity)?;
        let location_json = serde_json::to_string(&finding.location)?;
        let evidence_json = serde_json::to_string(&finding.evidence)?;
        let sanitization_json = serde_json::to_string(&finding.sanitization)?;
        let suppressions_json = serde_json::to_string(&finding.suppressions)?;

        sqlx::query(
            "INSERT INTO findings (
                id, run_id, rule_id, detector, fingerprint,
                severity_numeric, severity_level, severity, confidence,
                title, message, location, evidence, sanitization, suppressions,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(finding.id.to_string())
        .bind(run_id.to_string())
        .bind(&finding.rule_id)
        .bind(&finding.detector)
        .bind(&finding.fingerprint.0)
        .bind(severity_numeric)
        .bind(severity_level)
        .bind(severity_json)
        .bind(confidence_str(finding.confidence))
        .bind(&finding.title)
        .bind(&finding.message)
        .bind(location_json)
        .bind(evidence_json)
        .bind(sanitization_json)
        .bind(suppressions_json)
        .bind(finding.created_at.to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| {
            // map UNIQUE PRIMARY KEY collisions into Constraint
            let msg = e.to_string();
            if msg.contains("UNIQUE constraint failed") {
                StoreError::Constraint(msg)
            } else {
                StoreError::Backend(msg)
            }
        })?;
        tracing::Span::current().record("finding_id", finding.id.to_string());
        Ok(())
    }

    #[tracing::instrument(skip(self, stats), fields(run_id = %run_id))]
    async fn finish_run(&self, run_id: Ulid, stats: &ScanStats) -> Result<(), StoreError> {
        let now = Utc::now().to_rfc3339();
        let stats_json = serde_json::to_string(stats)?;
        let result = sqlx::query(
            "UPDATE runs SET status = 'finished', finished_at = ?, stats = ? \
             WHERE run_id = ?",
        )
        .bind(now)
        .bind(stats_json)
        .bind(run_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        if result.rows_affected() == 0 {
            return Err(StoreError::NotFound(run_id.to_string()));
        }
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(run_id = %run_id))]
    async fn load_run(&self, run_id: Ulid) -> Result<ScanResult, StoreError> {
        // 1. Fetch the run row.
        let run_row = sqlx::query(
            "SELECT started_at, finished_at, stats, errors \
             FROM runs WHERE run_id = ?",
        )
        .bind(run_id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?
        .ok_or_else(|| StoreError::NotFound(run_id.to_string()))?;

        let started_at_s: String = run_row
            .try_get("started_at")
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let finished_at_s: Option<String> = run_row
            .try_get("finished_at")
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let stats_s: Option<String> = run_row
            .try_get("stats")
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let errors_s: String = run_row
            .try_get("errors")
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        let started_at = parse_rfc3339(&started_at_s)?;
        let finished_at = match finished_at_s.as_deref() {
            Some(s) => parse_rfc3339(s)?,
            None => started_at,
        };
        let stats: ScanStats = match stats_s.as_deref() {
            Some(s) => serde_json::from_str(s)?,
            None => ScanStats::default(),
        };
        let errors: Vec<DetectorRunError> = serde_json::from_str(&errors_s)?;

        // 2. Fetch all findings for the run, ordered by created_at, then id.
        let rows = sqlx::query(
            "SELECT id, run_id, rule_id, detector, fingerprint, \
                    severity, confidence, title, message, location, \
                    evidence, sanitization, suppressions, created_at \
             FROM findings WHERE run_id = ? \
             ORDER BY created_at ASC, id ASC",
        )
        .bind(run_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;

        let mut findings = Vec::with_capacity(rows.len());
        for row in rows {
            let id_s: String = row
                .try_get("id")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let rule_id: String = row
                .try_get("rule_id")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let detector: String = row
                .try_get("detector")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let fingerprint_s: String = row
                .try_get("fingerprint")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let severity_s: String = row
                .try_get("severity")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let confidence_s: String = row
                .try_get("confidence")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let title: String = row
                .try_get("title")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let message: String = row
                .try_get("message")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let location_s: String = row
                .try_get("location")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let evidence_s: String = row
                .try_get("evidence")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let sanitization_s: String = row
                .try_get("sanitization")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let suppressions_s: String = row
                .try_get("suppressions")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let created_at_s: String = row
                .try_get("created_at")
                .map_err(|e| StoreError::Backend(e.to_string()))?;

            findings.push(Finding {
                id: parse_ulid(&id_s)?,
                rule_id,
                detector,
                fingerprint: Fingerprint(fingerprint_s),
                severity: serde_json::from_str::<Severity>(&severity_s)?,
                confidence: parse_confidence(&confidence_s)?,
                title,
                message,
                location: serde_json::from_str::<Location>(&location_s)?,
                evidence: serde_json::from_str::<Vec<Evidence>>(&evidence_s)?,
                sanitization: serde_json::from_str::<SanitizationState>(&sanitization_s)?,
                suppressions: serde_json::from_str::<Vec<SuppressionHint>>(&suppressions_s)?,
                created_at: parse_rfc3339(&created_at_s)?,
            });
        }

        Ok(ScanResult {
            run_id,
            started_at,
            finished_at,
            findings,
            stats,
            errors,
        })
    }

    #[tracing::instrument(skip(self), fields(run_id = %run_id))]
    async fn dedup(&self, run_id: Ulid) -> Result<usize, StoreError> {
        // Pick a winner per fingerprint (highest severity, then highest
        // confidence, then earliest id as a deterministic tiebreaker).
        // Mark the rest with `superseded_by = winner.id`.
        //
        // Uses SQLite window functions (3.25+, well below sqlx-bundled
        // version).
        let result = sqlx::query(
            "WITH ranked AS (
               SELECT
                 id,
                 fingerprint,
                 ROW_NUMBER() OVER (
                   PARTITION BY fingerprint
                   ORDER BY severity_numeric DESC,
                            CASE confidence
                              WHEN 'high' THEN 3
                              WHEN 'medium' THEN 2
                              WHEN 'low' THEN 1
                              ELSE 0 END DESC,
                            id ASC
                 ) AS rn,
                 FIRST_VALUE(id) OVER (
                   PARTITION BY fingerprint
                   ORDER BY severity_numeric DESC,
                            CASE confidence
                              WHEN 'high' THEN 3
                              WHEN 'medium' THEN 2
                              WHEN 'low' THEN 1
                              ELSE 0 END DESC,
                            id ASC
                 ) AS keeper_id
               FROM findings
               WHERE run_id = ? AND superseded_by IS NULL
             )
             UPDATE findings
                SET superseded_by = (
                    SELECT keeper_id FROM ranked WHERE ranked.id = findings.id
                )
              WHERE findings.id IN (SELECT id FROM ranked WHERE rn > 1)",
        )
        .bind(run_id.to_string())
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(result.rows_affected() as usize)
    }

    #[tracing::instrument(skip(self, reason), fields(fingerprint = %fingerprint.0))]
    async fn suppress(&self, fingerprint: &Fingerprint, reason: &str) -> Result<(), StoreError> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO suppressions (fingerprint, reason, created_at) \
             VALUES (?, ?, ?) \
             ON CONFLICT(fingerprint) DO UPDATE SET reason = excluded.reason",
        )
        .bind(&fingerprint.0)
        .bind(reason)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::finding::{Confidence, Location};
    use ph0b0s_core::scan::{DetectorFilter, ScanOptions, ScanRequest, ScanStats};
    use ph0b0s_core::severity::{Level, Severity};
    use ph0b0s_core::target::Target;
    use ph0b0s_test_support::{
        deterministic_run_id, fixed_timestamp, sample_finding, sample_scan_result,
    };
    use std::path::PathBuf;

    fn sample_request(run_id: Ulid) -> ScanRequest {
        ScanRequest {
            run_id,
            target: Target::LocalDirectory {
                path: PathBuf::from("/tmp/x"),
            },
            detector_filter: DetectorFilter::All,
            options: ScanOptions::default(),
            detector_params: Default::default(),
        }
    }

    async fn fresh_store() -> SqliteFindingStore {
        SqliteFindingStore::open_in_memory()
            .await
            .expect("open in memory")
    }

    #[tokio::test]
    async fn open_in_memory_runs_migrations_and_tables_exist() {
        let store = fresh_store().await;
        let names: Vec<String> = sqlx::query(
            "SELECT name FROM sqlite_master WHERE type = 'table' \
             AND name NOT LIKE '\\_sqlx\\_%' ESCAPE '\\' \
             AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' \
             ORDER BY name",
        )
        .fetch_all(store.pool())
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.try_get::<String, _>("name").unwrap())
        .collect();
        assert_eq!(names, vec!["findings", "runs", "suppressions"]);
    }

    #[tokio::test]
    async fn begin_run_inserts_started_row() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        let req = sample_request(run_id);
        let returned = store.begin_run(&req).await.unwrap();
        assert_eq!(returned, run_id);

        let status: String = sqlx::query("SELECT status FROM runs WHERE run_id = ?")
            .bind(run_id.to_string())
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("status")
            .unwrap();
        assert_eq!(status, "started");
    }

    #[tokio::test]
    async fn record_inserts_finding_row() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();

        let f = sample_finding();
        store.record(run_id, &f).await.unwrap();

        let count: i64 = sqlx::query("SELECT COUNT(*) AS c FROM findings WHERE run_id = ?")
            .bind(run_id.to_string())
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("c")
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn record_round_trip_via_load_run_preserves_finding() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();

        let f = sample_finding();
        store.record(run_id, &f).await.unwrap();
        store
            .finish_run(run_id, &ScanStats::default())
            .await
            .unwrap();

        let result = store.load_run(run_id).await.unwrap();
        assert_eq!(result.run_id, run_id);
        assert_eq!(result.findings.len(), 1);
        assert_eq!(result.findings[0], f);
    }

    #[tokio::test]
    async fn record_duplicate_id_returns_constraint() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();

        let f = sample_finding();
        store.record(run_id, &f).await.unwrap();
        let err = store.record(run_id, &f).await.unwrap_err();
        match err {
            StoreError::Constraint(_) => {}
            other => panic!("expected Constraint, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn finish_run_sets_status_finished_with_stats() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();

        let stats = ScanStats {
            total_findings: 5,
            tokens_in: 100,
            ..Default::default()
        };
        store.finish_run(run_id, &stats).await.unwrap();

        let row = sqlx::query("SELECT status, stats FROM runs WHERE run_id = ?")
            .bind(run_id.to_string())
            .fetch_one(store.pool())
            .await
            .unwrap();
        let status: String = row.try_get("status").unwrap();
        let stats_s: String = row.try_get("stats").unwrap();
        assert_eq!(status, "finished");
        let parsed: ScanStats = serde_json::from_str(&stats_s).unwrap();
        assert_eq!(parsed.total_findings, 5);
        assert_eq!(parsed.tokens_in, 100);
    }

    #[tokio::test]
    async fn finish_run_for_unknown_run_returns_not_found() {
        let store = fresh_store().await;
        let err = store
            .finish_run(deterministic_run_id(), &ScanStats::default())
            .await
            .unwrap_err();
        match err {
            StoreError::NotFound(_) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn load_run_missing_returns_not_found() {
        let store = fresh_store().await;
        let err = store.load_run(deterministic_run_id()).await.unwrap_err();
        match err {
            StoreError::NotFound(_) => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cleanup_orphan_runs_marks_started_as_aborted() {
        let store = fresh_store().await;
        let r1 = Ulid::from_bytes([1; 16]);
        let r2 = Ulid::from_bytes([2; 16]);
        store.begin_run(&sample_request(r1)).await.unwrap();
        store.begin_run(&sample_request(r2)).await.unwrap();
        store.finish_run(r1, &ScanStats::default()).await.unwrap();

        let aborted = store.cleanup_orphan_runs().await.unwrap();
        assert_eq!(aborted, 1);

        let status: String = sqlx::query("SELECT status FROM runs WHERE run_id = ?")
            .bind(r2.to_string())
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("status")
            .unwrap();
        assert_eq!(status, "aborted");
    }

    #[tokio::test]
    async fn dedup_keeps_highest_severity_representative() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();

        // Three findings with the same fingerprint: Low, Medium, High.
        let location = Location::File {
            path: "src/x.rs".into(),
            start_line: 1,
            end_line: 1,
            start_col: None,
            end_col: None,
        };
        let mk = |level: Level, conf: Confidence| -> Finding {
            let f = sample_finding();
            Finding {
                id: Ulid::new(),
                severity: Severity::Qualitative(level),
                confidence: conf,
                location: location.clone(),
                fingerprint: Fingerprint::compute("rule.x", &location, b"ev"),
                rule_id: "rule.x".into(),
                ..f
            }
        };

        let low = mk(Level::Low, Confidence::Medium);
        let med = mk(Level::Medium, Confidence::Medium);
        let high = mk(Level::High, Confidence::High);
        store.record(run_id, &low).await.unwrap();
        store.record(run_id, &med).await.unwrap();
        store.record(run_id, &high).await.unwrap();

        let collapsed = store.dedup(run_id).await.unwrap();
        assert_eq!(collapsed, 2);

        // High wins: low and med both point at high.id; high.superseded_by is NULL
        let row_for = |id: Ulid| {
            let pool = store.pool().clone();
            async move {
                sqlx::query("SELECT superseded_by FROM findings WHERE id = ?")
                    .bind(id.to_string())
                    .fetch_one(&pool)
                    .await
                    .unwrap()
            }
        };
        let high_super: Option<String> = row_for(high.id).await.try_get("superseded_by").unwrap();
        assert!(high_super.is_none());

        let med_super: Option<String> = row_for(med.id).await.try_get("superseded_by").unwrap();
        assert_eq!(med_super.as_deref(), Some(high.id.to_string().as_str()));

        let low_super: Option<String> = row_for(low.id).await.try_get("superseded_by").unwrap();
        assert_eq!(low_super.as_deref(), Some(high.id.to_string().as_str()));
    }

    #[tokio::test]
    async fn dedup_is_idempotent() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();
        let f = sample_finding();
        let f2 = Finding {
            id: Ulid::new(),
            ..f.clone()
        };
        store.record(run_id, &f).await.unwrap();
        store.record(run_id, &f2).await.unwrap();
        let first = store.dedup(run_id).await.unwrap();
        let second = store.dedup(run_id).await.unwrap();
        assert_eq!(first, 1);
        assert_eq!(second, 0);
    }

    #[tokio::test]
    async fn dedup_returns_zero_for_unique_fingerprints() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();
        // sample_scan_result builds findings with distinct fingerprints
        let result = sample_scan_result(3);
        for f in &result.findings {
            store.record(run_id, f).await.unwrap();
        }
        let collapsed = store.dedup(run_id).await.unwrap();
        assert_eq!(collapsed, 0);
    }

    #[tokio::test]
    async fn suppress_inserts_row_keyed_by_fingerprint() {
        let store = fresh_store().await;
        let fp = Fingerprint("abc123".into());
        store.suppress(&fp, "vendored fork").await.unwrap();
        let reason: String = sqlx::query("SELECT reason FROM suppressions WHERE fingerprint = ?")
            .bind(&fp.0)
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("reason")
            .unwrap();
        assert_eq!(reason, "vendored fork");
    }

    #[tokio::test]
    async fn suppress_replaces_existing_reason() {
        let store = fresh_store().await;
        let fp = Fingerprint("abc123".into());
        store.suppress(&fp, "first").await.unwrap();
        store.suppress(&fp, "second").await.unwrap();
        let reason: String = sqlx::query("SELECT reason FROM suppressions WHERE fingerprint = ?")
            .bind(&fp.0)
            .fetch_one(store.pool())
            .await
            .unwrap()
            .try_get("reason")
            .unwrap();
        assert_eq!(reason, "second");

        let count: i64 =
            sqlx::query("SELECT COUNT(*) AS c FROM suppressions WHERE fingerprint = ?")
                .bind(&fp.0)
                .fetch_one(store.pool())
                .await
                .unwrap()
                .try_get("c")
                .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn load_run_returns_findings_in_created_at_order() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();

        // Use sample_scan_result for a deterministic small set.
        let result = sample_scan_result(3);
        for f in &result.findings {
            store.record(run_id, f).await.unwrap();
        }
        store
            .finish_run(run_id, &ScanStats::default())
            .await
            .unwrap();

        let loaded = store.load_run(run_id).await.unwrap();
        assert_eq!(loaded.findings.len(), 3);
        let loaded_ids: Vec<_> = loaded.findings.iter().map(|f| f.id).collect();
        let expected_ids: Vec<_> = result.findings.iter().map(|f| f.id).collect();
        assert_eq!(loaded_ids, expected_ids);
    }

    #[tokio::test]
    async fn load_run_unfinished_uses_started_as_finished_placeholder() {
        let store = fresh_store().await;
        let run_id = deterministic_run_id();
        store.begin_run(&sample_request(run_id)).await.unwrap();
        let r = store.load_run(run_id).await.unwrap();
        // unfinished: finished_at falls back to started_at
        assert_eq!(r.finished_at, r.started_at);
        assert!(r.findings.is_empty());
    }

    #[tokio::test]
    async fn fixed_timestamp_round_trips() {
        // Ensures the RFC3339 (de)serialization preserves the fixture timestamp
        // exactly (no precision loss on the seconds boundary).
        let ts = fixed_timestamp();
        let s = ts.to_rfc3339();
        let parsed = parse_rfc3339(&s).unwrap();
        assert_eq!(parsed, ts);
    }
}
