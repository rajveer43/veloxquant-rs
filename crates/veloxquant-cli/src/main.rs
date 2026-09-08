//! `vq` — the VeloxQuant command-line interface.

mod analyze;
mod benchmark;
mod doctor;
mod models;
mod recommend;
mod serve;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "vq",
    version,
    about = "VeloxQuant CLI — memory intelligence and optimization for local AI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check system readiness for local AI
    Doctor,
    /// Analyze memory requirements for a model
    Analyze {
        /// Model name (checked against the curated registry, else a 7-8B fallback is used)
        model: String,
        /// Context length in tokens
        #[arg(long, default_value_t = 32768)]
        context: usize,
    },
    /// Recommend models and a profile for this hardware
    Recommend,
    /// Benchmark inference performance for a model (requires a running VeloxQuant runtime)
    Benchmark {
        /// Model name to benchmark
        model: String,
        /// Context length in tokens
        #[arg(long, default_value_t = 4096)]
        context: usize,
        /// Prompt to send
        #[arg(long, default_value = "Write a short haiku about the ocean.")]
        prompt: String,
    },
    /// Connect to (or report on) the VeloxQuant runtime
    Serve,
    /// Manage locally downloaded model weights (Hugging Face cache)
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// List models present in the local Hugging Face cache
    List,
    /// Download a model's weights into the local Hugging Face cache
    Pull {
        /// Hugging Face repo id, e.g. "mlx-community/Qwen3-8B-4bit"
        model_id: String,
    },
    /// Delete a model's weights from the local Hugging Face cache
    Delete {
        /// Hugging Face repo id, e.g. "mlx-community/Qwen3-8B-4bit"
        model_id: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Doctor => doctor::run().await,
        Command::Analyze { model, context } => analyze::run(&model, context).await,
        Command::Recommend => recommend::run().await,
        Command::Benchmark {
            model,
            context,
            prompt,
        } => benchmark::run(&model, context, &prompt).await,
        Command::Serve => serve::run().await,
        Command::Models { command } => match command {
            ModelsCommand::List => models::list().await,
            ModelsCommand::Pull { model_id } => models::pull(&model_id).await,
            ModelsCommand::Delete { model_id } => models::delete(&model_id).await,
        },
    }
}
