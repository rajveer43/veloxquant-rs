use anyhow::{anyhow, Result};
use veloxquant::Client;

/// Connects to an existing VeloxQuant runtime and reports its status.
/// Launching a local runtime process directly is planned for a future
/// release.
pub async fn run() -> Result<()> {
    let client = Client::builder().build()?;

    let status = client
        .runtime()
        .health()
        .await
        .map_err(|e| anyhow!("connect to veloxquant runtime: {e}"))?;

    if !status.healthy {
        return Err(anyhow!("veloxquant runtime reported unhealthy status"));
    }

    println!(
        "Connected to VeloxQuant runtime (engine: {}, version: {})",
        status.engine, status.version
    );
    Ok(())
}
