//! Pretty-printed `ScanResult` JSON. Stable enough for snapshot tests and
//! reproducible CLI diffs.

use async_trait::async_trait;
use ph0b0s_core::error::ReportError;
use ph0b0s_core::report::Reporter;
use ph0b0s_core::scan::ScanResult;
use tokio::io::{AsyncWrite, AsyncWriteExt};

#[derive(Default, Clone, Copy)]
pub struct JsonReporter;

impl JsonReporter {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&self, result: &ScanResult) -> Result<String, ReportError> {
        let s = serde_json::to_string_pretty(result)?;
        Ok(s)
    }
}

#[async_trait]
impl Reporter for JsonReporter {
    fn name(&self) -> &'static str {
        "json"
    }

    async fn write(
        &self,
        result: &ScanResult,
        sink: &mut (dyn AsyncWrite + Send + Unpin),
    ) -> Result<(), ReportError> {
        let body = self.render(result)?;
        sink.write_all(body.as_bytes()).await?;
        sink.write_all(b"\n").await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_test_support::sample_scan_result;

    #[tokio::test]
    async fn render_round_trips_via_serde_json() {
        let r = sample_scan_result(2);
        let body = JsonReporter.render(&r).unwrap();
        let parsed: ScanResult = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.findings.len(), r.findings.len());
        assert_eq!(parsed.findings[0].id, r.findings[0].id);
    }

    #[tokio::test]
    async fn write_appends_trailing_newline() {
        let r = sample_scan_result(1);
        let mut buf = Vec::new();
        JsonReporter.write(&r, &mut buf).await.unwrap();
        assert!(buf.ends_with(b"\n"));
    }

    #[tokio::test]
    async fn json_snapshot() {
        let r = sample_scan_result(3);
        let body = JsonReporter.render(&r).unwrap();
        insta::assert_snapshot!("json_three_findings", body);
    }
}
