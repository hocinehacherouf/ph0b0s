//! Reporter trait. Implementations live in `ph0b0s-report`.

use async_trait::async_trait;
use tokio::io::AsyncWrite;

use crate::error::ReportError;
use crate::scan::ScanResult;

#[async_trait]
pub trait Reporter: Send + Sync {
    fn name(&self) -> &'static str;

    async fn write(
        &self,
        result: &ScanResult,
        sink: &mut (dyn AsyncWrite + Send + Unpin),
    ) -> Result<(), ReportError>;
}
