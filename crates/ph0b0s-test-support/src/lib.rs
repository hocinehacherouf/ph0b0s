//! ph0b0s-test-support: deterministic mock implementations of the seam
//! traits in `ph0b0s-core`, plus reusable fixture helpers.
//!
//! Intended use:
//!
//! ```ignore
//! use ph0b0s_test_support::{MockLlmAgent, MockToolHost, sample_scan_result};
//!
//! let agent = MockLlmAgent::new();
//! agent.enqueue_chat_text("hello");
//! ```
//!
//! The mocks are `Clone` and share state via `Arc`, so a test can hand the
//! agent or tool host to a system under test as `&dyn LlmAgent` /
//! `&dyn ToolHost` while keeping a separate inspector handle for assertions.

pub mod fixtures;
pub mod mock_llm;
pub mod mock_tools;

pub use fixtures::{
    deterministic_run_id, fixed_timestamp, sample_finding, sample_scan_result, temp_workspace,
    temp_workspace_with,
};
pub use mock_llm::{MockLlmAgent, MockLlmSession};
pub use mock_tools::{CannedTool, MockToolHost};
