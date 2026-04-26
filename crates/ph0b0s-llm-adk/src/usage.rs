//! Map adk-rust's `UsageMetadata` into the seam's `Usage`.
//!
//! adk's `prompt_token_count` / `candidates_token_count` are signed `i32`s
//! that may be `0` for providers that don't report usage. Saturating-cast
//! into `u64` for our `tokens_in` / `tokens_out`.

use ph0b0s_core::llm::{CostSource, Usage};

/// Convert an adk `UsageMetadata` into our `Usage`. Negative or unset values
/// clamp to 0. `cost_source = Native` because the provider returned the
/// numbers; `cost_usd_estimate` comes from `cost` if present, else 0.0.
pub(crate) fn from_adk_usage(meta: Option<&adk_rust::UsageMetadata>) -> Usage {
    let Some(meta) = meta else {
        return Usage::default();
    };
    Usage {
        tokens_in: clamp_to_u64(meta.prompt_token_count),
        tokens_out: clamp_to_u64(meta.candidates_token_count),
        cost_usd_estimate: meta.cost.unwrap_or(0.0),
        cost_source: CostSource::Native,
    }
}

#[inline]
fn clamp_to_u64(v: i32) -> u64 {
    if v < 0 { 0 } else { v as u64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_counts_clamp_to_zero() {
        let meta = adk_rust::UsageMetadata {
            prompt_token_count: -5,
            candidates_token_count: -3,
            total_token_count: 0,
            ..Default::default()
        };
        let u = from_adk_usage(Some(&meta));
        assert_eq!(u.tokens_in, 0);
        assert_eq!(u.tokens_out, 0);
    }

    #[test]
    fn populated_metadata_carries_through() {
        let meta = adk_rust::UsageMetadata {
            prompt_token_count: 100,
            candidates_token_count: 200,
            total_token_count: 300,
            cost: Some(0.0123),
            ..Default::default()
        };
        let u = from_adk_usage(Some(&meta));
        assert_eq!(u.tokens_in, 100);
        assert_eq!(u.tokens_out, 200);
        assert!((u.cost_usd_estimate - 0.0123).abs() < 1e-9);
        assert_eq!(u.cost_source, CostSource::Native);
    }

    #[test]
    fn missing_metadata_yields_default_usage() {
        let u = from_adk_usage(None);
        assert_eq!(u, Usage::default());
    }
}
