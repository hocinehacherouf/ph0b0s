//! ph0b0s reporters.
//!
//! Three implementations of `ph0b0s_core::Reporter`:
//!
//! - [`SarifReporter`] — SARIF 2.1.0, the primary output format. Round-trips
//!   through `serde-sarif` so the JSON is guaranteed to deserialize back as a
//!   schema-valid `Sarif` value.
//! - [`MarkdownReporter`] — human-readable plaintext for terminals and PR
//!   comments.
//! - [`JsonReporter`] — pretty-printed `ScanResult` JSON, useful for piping
//!   into `jq` or for reproducible diffs.

pub mod json;
pub mod markdown;
pub mod sarif;

pub use json::JsonReporter;
pub use markdown::MarkdownReporter;
pub use sarif::SarifReporter;
