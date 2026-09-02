use anyhow::Result;
use veloxquant::{format_bytes, Client};

pub async fn run() -> Result<()> {
    println!("VeloxQuant Doctor\n");
    println!("✓ Rust runtime available");

    let client = Client::builder().build()?;

    let info = client.system().info().await;
    if info.apple_silicon {
        println!("✓ Apple Silicon detected");
    } else {
        println!(
            "- Apple Silicon not detected (platform: {}/{})",
            info.platform, info.architecture
        );
    }

    if info.total_memory_bytes > 0 {
        println!("✓ {} Unified Memory", format_bytes(info.total_memory_bytes));
    } else {
        println!("- Could not determine total memory");
    }

    let runtime_healthy = match client.runtime().health().await {
        Ok(status) => status.healthy,
        Err(_) => false,
    };

    if runtime_healthy {
        println!("✓ VeloxQuant runtime reachable");
    } else {
        println!("✗ VeloxQuant runtime unreachable");
    }

    println!();
    if runtime_healthy {
        println!("System ready.");
    } else {
        println!("System is ready for local memory analysis; start the VeloxQuant runtime for inference.");
    }

    Ok(())
}
