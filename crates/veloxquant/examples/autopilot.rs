//! AutoPilot: pick a model and a compression strategy for this Mac, print
//! every decision, then send one chat message with the planned model.
//!
//! The compression decision comes from the real VeloxQuant-MLX CLI, so this
//! needs the `veloxquant` console script on `PATH` (`pip install
//! veloxquant-mlx`) and an Apple Silicon Mac. The final chat call also needs
//! a runtime at `http://localhost:8765` serving the planned model, e.g.
//! `veloxquant serve --model <model> --method <method> --bits <bits>` using
//! the values printed below. Run with:
//!
//! ```sh
//! cargo run --example autopilot --features autopilot
//! ```

use veloxquant::{AutoPilotConfig, Client, Task};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().auto_detect().build()?;

    let session = client
        .autopilot(AutoPilotConfig {
            task: Some(Task::Chat),
            ..Default::default()
        })
        .await?;

    let plan = session.plan();
    println!("AutoPilot plan:");
    for decision in &plan.decisions {
        println!("  - {decision}");
    }
    println!(
        "\nServe with: veloxquant serve --model {} --method {}{}",
        plan.selected_model.name,
        plan.method,
        plan.bits
            .map(|b| format!(" --bits {b}"))
            .unwrap_or_default()
    );

    match session.chat("Explain KV cache in one sentence.").await {
        Ok(reply) => println!("\n{}", reply.text),
        Err(e) => eprintln!("\n(chat skipped: {e} — is the runtime serving the planned model?)"),
    }
    Ok(())
}
