//! Map `adk_core::AdkError` into the seam's `LlmError`.

use ph0b0s_core::error::LlmError;

/// Convert `adk_core::AdkError` into `LlmError`. The mapping is
/// best-effort because adk's errors are stringly-typed at the boundary;
/// we keep the message and tag with `Provider`.
pub(crate) fn map_adk_error(err: adk_rust::AdkError) -> LlmError {
    LlmError::Provider(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_adk_error_preserves_message_under_provider_variant() {
        let adk_err = adk_rust::AdkError::model("provider rejected request");
        let mapped = map_adk_error(adk_err);
        match mapped {
            LlmError::Provider(msg) => assert!(
                msg.contains("provider rejected request"),
                "expected adk message in mapped error, got: {msg}"
            ),
            other => panic!("expected Provider variant, got {other:?}"),
        }
    }
}
