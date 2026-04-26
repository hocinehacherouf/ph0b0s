//! ph0b0s SQLite-backed `FindingStore`.
//!
//! Public surface: [`SqliteFindingStore::open`] for on-disk databases and
//! [`SqliteFindingStore::open_in_memory`] for tests. Both run the embedded
//! migrations under `migrations/0001_init.sql` before returning.

pub mod sqlite;

pub use sqlite::SqliteFindingStore;
