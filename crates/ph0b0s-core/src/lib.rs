//! ph0b0s-core: the seam.
//!
//! Domain types and traits that detection-pack crates depend on. Zero
//! coupling to any LLM vendor or to the `adk-rust` adapter. The CLI wires
//! the adapter at startup and passes `&dyn LlmAgent` / `&dyn ToolHost`
//! into detector contexts.

pub mod detector;
pub mod error;
pub mod finding;
pub mod llm;
pub mod report;
pub mod scan;
pub mod severity;
pub mod store;
pub mod target;
pub mod tools;

pub use detector::{Detector, DetectorCtx, DetectorKind, DetectorMetadata};
pub use error::{CoreError, DetectorError, LlmError, ReportError, StoreError, ToolError};
pub use finding::{
    Confidence, Evidence, Finding, Fingerprint, Location, SanitizationState, SuppressionHint,
};
pub use llm::{
    AgentRoleKey, ChatMessage, ChatRequest, ChatResponse, LlmAgent, LlmSession, SessionOptions,
    StructuredRequest, ToolCall, ToolResult, ToolSpec, Usage, UserMessage,
};
pub use report::Reporter;
pub use scan::{DetectorFilter, ScanCtx, ScanOptions, ScanRequest, ScanResult, ScanStats};
pub use severity::{Level, Severity};
pub use store::FindingStore;
pub use target::{Target, TargetMaterializer, Workspace, WorkspaceGuard};
pub use tools::{McpServerSpec, McpTransport, NativeTool, ToolHost};
