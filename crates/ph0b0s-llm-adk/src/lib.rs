//! Adapter binding `adk-rust` to the seam traits in `ph0b0s-core`.
//!
//! THE ONLY CRATE PERMITTED TO IMPORT `adk-*`. Detection-pack crates depend
//! on `ph0b0s-core` only; the CLI wires this adapter at startup and passes
//! `&dyn LlmAgent` / `&dyn ToolHost` into detector contexts.
//!
//! # v1 limitations (documented)
//!
//! - **No tool-call loop.** `LlmAgent::chat` calls the underlying
//!   `adk_core::Llm` once and returns the final assistant turn. If the model
//!   emits a `FunctionCall` `Part`, the adapter does NOT dispatch it back
//!   into `ToolHost`. Smoke detectors don't use tools, so this is fine for
//!   slice (e). Adding the tool loop is a v2 concern.
//! - **MCP mounting deferred.** `ToolHost::mount_mcp` records the spec and
//!   logs a warning; it does not actually connect to the MCP server. The
//!   plan calls this out as a TBD.
//! - **No streaming.** `chat`/`structured` call `generate_content(req, false)`
//!   and collect the final response.
//!
//! See `tests/adapter_smoke.rs` for behaviour fixtures (run against
//! `adk_model::MockLlm`).

pub mod agent;
pub mod config;
pub mod error;
pub mod provider;
pub mod tools;
pub mod usage;

pub use agent::{AdkLlmAgent, AdkSession};
pub use config::{AgentConfig, ProviderConfig, ProviderRegistry};
pub use error::BuildError;
pub use tools::AdkToolHost;
