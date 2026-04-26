//! `ph0b0s-cli` library surface.
//!
//! Modules are exposed as a library so that integration tests (and any
//! future programmatic embedders) can reuse the orchestrator without going
//! through the binary.

pub mod commands;
pub mod config;
pub mod provider;
pub mod registry;
pub mod scan;
pub mod workspace;
