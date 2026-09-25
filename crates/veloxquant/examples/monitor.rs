//! Live host-memory sampling: publish a sample every second for five
//! seconds and print each one as it arrives. No runtime needed. Run with:
//!
//! ```sh
//! cargo run --example monitor --features monitor
//! ```

use std::time::Duration;

use veloxquant::{format_bytes, Monitor, SystemSampler};

#[tokio::main]
async fn main() {
    let monitor = Monitor::new();
    let mut samples = monitor.subscribe();
    let sampling = monitor.spawn_sampler(SystemSampler::new(), Duration::from_secs(1));

    for _ in 0..5 {
        match samples.recv().await {
            Ok(m) => println!(
                "used {:>10}  available {:>10}",
                format_bytes(m.memory_used_bytes),
                format_bytes(m.memory_available_bytes)
            ),
            Err(e) => eprintln!("subscriber lagged or closed: {e}"),
        }
    }
    sampling.stop().await;
}
