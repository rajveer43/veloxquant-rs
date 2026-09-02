use anyhow::Result;
use veloxquant::{format_bytes, Client, ModelRecommendationRequest};

pub async fn run() -> Result<()> {
    let client = Client::builder().build()?;

    let info = client.system().info().await;

    println!("Hardware:");
    if !info.cpu_model.is_empty() {
        println!("{}", info.cpu_model);
    } else {
        println!("{}/{}", info.platform, info.architecture);
    }
    if info.total_memory_bytes > 0 {
        println!("{} Unified Memory", format_bytes(info.total_memory_bytes));
    }
    println!();

    let recommendations = client.models().recommend(&ModelRecommendationRequest {
        task: None,
        available_memory_bytes: info.available_memory_bytes,
    });

    println!("Recommended Models:\n");
    if recommendations.is_empty() {
        println!("(none fit available memory)");
    }
    for (i, model) in recommendations.iter().enumerate() {
        println!("{}. {}", i + 1, model.name);
    }
    println!();

    println!("Recommended Profile:\n{}", info.recommended_profile);

    Ok(())
}
