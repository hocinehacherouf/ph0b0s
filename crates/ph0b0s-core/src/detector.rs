//! The `Detector` trait. Both LLM-driven and subprocess detectors implement
//! it; the orchestrator passes them an `LlmAgent` and a `ToolHost` and they
//! return a vector of `Finding`s.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use ulid::Ulid;

use crate::error::DetectorError;
use crate::finding::Finding;
use crate::llm::LlmAgent;
use crate::target::{Target, Workspace};
use crate::tools::ToolHost;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorKind {
    /// Calls into `ctx.agent`.
    LlmDriven,
    /// Shells out to an external tool; ignores `ctx.agent`.
    Subprocess,
    /// Pure in-process Rust analysis; ignores both `ctx.agent` and
    /// subprocesses.
    Native,
    /// Combination — uses the agent and at least one external tool.
    Hybrid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorMetadata {
    pub id: String,
    pub version: String,
    pub kind: DetectorKind,
    /// One-line human description for `ph0b0s detectors list`.
    pub description: String,
    /// Capability hints the orchestrator may use for routing (e.g. needs
    /// MCP server `filesystem`, needs `cargo` on PATH).
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// Per-run context handed to a detector. Lives only for the duration of
/// `Detector::run`; no detector should hold references past return.
pub struct DetectorCtx<'a> {
    pub workspace: &'a Workspace,
    pub target: &'a Target,
    pub agent: &'a dyn LlmAgent,
    pub tools: &'a dyn ToolHost,
    pub params: &'a serde_json::Value,
    pub run_id: Ulid,
    /// Wall-clock deadline for the whole detector run.
    pub deadline: Instant,
}

#[async_trait]
pub trait Detector: Send + Sync {
    fn metadata(&self) -> DetectorMetadata;

    /// JSON Schema (Draft 2020-12) describing the detector's params. The
    /// orchestrator validates `DetectorCtx::params` against this before
    /// calling `run`.
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "additionalProperties": true })
    }

    async fn run(
        &self,
        ctx: &DetectorCtx<'_>,
        cancel: CancellationToken,
    ) -> Result<Vec<Finding>, DetectorError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial detector implementation that exercises the trait's default
    /// `config_schema` body.
    struct NopDetector;

    #[async_trait]
    impl Detector for NopDetector {
        fn metadata(&self) -> DetectorMetadata {
            DetectorMetadata {
                id: "nop".into(),
                version: "0.0.1".into(),
                kind: DetectorKind::Native,
                description: "test".into(),
                capabilities: vec![],
            }
        }
        async fn run(
            &self,
            _ctx: &DetectorCtx<'_>,
            _cancel: CancellationToken,
        ) -> Result<Vec<Finding>, DetectorError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn default_config_schema_is_open_object() {
        let s = NopDetector.config_schema();
        assert_eq!(s["type"], "object");
        assert_eq!(s["additionalProperties"], true);
    }
}
