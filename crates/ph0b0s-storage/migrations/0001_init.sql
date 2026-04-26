-- ph0b0s-storage initial schema.
--
-- Three tables:
--   runs          — one row per scan invocation, with lifecycle state.
--   findings      — one row per Finding, JSON columns for nested fields.
--   suppressions  — global, by-fingerprint, persistent across runs.
--
-- All TEXT columns holding JSON are validated by serde at the boundary; the
-- DB does not enforce JSON shape itself (no jsonb in SQLite).

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT    PRIMARY KEY,
    started_at   TEXT    NOT NULL,                                 -- RFC3339
    finished_at  TEXT,                                             -- RFC3339 or NULL
    status       TEXT    NOT NULL CHECK (status IN ('started', 'finished', 'aborted')),
    request      TEXT    NOT NULL,                                 -- JSON ScanRequest
    stats        TEXT,                                             -- JSON ScanStats (NULL until finished)
    errors       TEXT    NOT NULL DEFAULT '[]'                     -- JSON Vec<DetectorRunError>
);

CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);

CREATE TABLE IF NOT EXISTS findings (
    id                TEXT    PRIMARY KEY,
    run_id            TEXT    NOT NULL,
    rule_id           TEXT    NOT NULL,
    detector          TEXT    NOT NULL,
    fingerprint       TEXT    NOT NULL,
    severity_numeric  REAL    NOT NULL,                            -- 0.0–10.0
    severity_level    TEXT    NOT NULL,                            -- none|low|medium|high|critical
    severity          TEXT    NOT NULL,                            -- JSON Severity
    confidence        TEXT    NOT NULL CHECK (confidence IN ('low','medium','high')),
    title             TEXT    NOT NULL,
    message           TEXT    NOT NULL,
    location          TEXT    NOT NULL,                            -- JSON Location
    evidence          TEXT    NOT NULL DEFAULT '[]',               -- JSON Vec<Evidence>
    sanitization      TEXT    NOT NULL,                            -- JSON SanitizationState
    suppressions      TEXT    NOT NULL DEFAULT '[]',               -- JSON Vec<SuppressionHint>
    superseded_by     TEXT,                                        -- ULID of representative if deduped
    suppressed        INTEGER NOT NULL DEFAULT 0,                  -- bool
    created_at        TEXT    NOT NULL,                            -- RFC3339
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_findings_run_id      ON findings(run_id);
CREATE INDEX IF NOT EXISTS idx_findings_fingerprint ON findings(fingerprint);
CREATE INDEX IF NOT EXISTS idx_findings_rule_id     ON findings(rule_id);
CREATE INDEX IF NOT EXISTS idx_findings_superseded  ON findings(run_id, superseded_by);

CREATE TABLE IF NOT EXISTS suppressions (
    fingerprint  TEXT    PRIMARY KEY,
    reason       TEXT    NOT NULL,
    expires_at   TEXT,                                             -- RFC3339 or NULL = never
    created_at   TEXT    NOT NULL                                  -- RFC3339
);
