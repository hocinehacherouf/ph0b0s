//! Typed errors for module boundaries. CLI glue uses `anyhow` and converts.

use thiserror::Error;

/// Catch-all for ph0b0s-core utilities (target preparation, fingerprint,
/// severity normalisation, etc.). Library callers convert into the
/// module-specific error variants where appropriate.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("workspace prep failed: {0}")]
    WorkspacePrep(String),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum DetectorError {
    #[error("cancelled before completion")]
    Cancelled,

    #[error("timed out before completion")]
    Timeout,

    #[error("missing required tool: {0}")]
    MissingTool(String),

    #[error("invalid params: {0}")]
    InvalidParams(String),

    #[error("llm error: {0}")]
    Llm(#[from] LlmError),

    #[error("tool error: {0}")]
    Tool(#[from] ToolError),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("subprocess failed: {0}")]
    Subprocess(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("provider error: {0}")]
    Provider(String),

    #[error("rate limited (retry-after: {retry_after_secs:?}s)")]
    RateLimited { retry_after_secs: Option<u64> },

    #[error("auth error: {0}")]
    Auth(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("structured-output validation failed: {0}")]
    StructuredValidation(String),

    #[error("tool dispatch failed: {0}")]
    ToolDispatch(String),

    #[error("cancelled")]
    Cancelled,

    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("unknown tool: {0}")]
    Unknown(String),

    #[error("invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("execution failed: {0}")]
    Execution(String),

    #[error("mcp transport error: {0}")]
    McpTransport(String),

    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("backend error: {0}")]
    Backend(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("constraint violation: {0}")]
    Constraint(String),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum ReportError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("invalid sarif: {0}")]
    InvalidSarif(String),

    #[error("other: {0}")]
    Other(String),
}
