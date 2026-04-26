//! Map `adk_core::AdkError` into the seam's `LlmError`.

use ph0b0s_core::error::LlmError;

/// Convert `adk_core::AdkError` into `LlmError`. The mapping is
/// best-effort because adk's errors are stringly-typed at the boundary;
/// we keep the message and tag with `Provider`.
pub(crate) fn map_adk_error(err: adk_rust::AdkError) -> LlmError {
    LlmError::Provider(err.to_string())
}
