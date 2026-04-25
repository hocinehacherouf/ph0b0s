//! ph0b0s adapter binding adk-rust to the seam traits in ph0b0s-core.
//!
//! THE ONLY CRATE PERMITTED TO IMPORT `adk-*`. Detection-pack crates depend on
//! `ph0b0s-core` only; the CLI wires this adapter at startup and passes
//! `&dyn LlmAgent` / `&dyn ToolHost` into detector contexts.
//!
//! Implementation lands in a follow-up pass; this stub keeps the workspace
//! buildable while the seam stabilises.
