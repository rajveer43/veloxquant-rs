use anyhow::{anyhow, Result};
use veloxquant::PythonInterpreter;
use veloxquant::{delete_local_model, format_bytes, list_local_models, pull_local_model};

pub async fn list() -> Result<()> {
    let interpreter = PythonInterpreter::default();
    let models = list_local_models(&interpreter)
        .await
        .map_err(|e| anyhow!(e))?;

    if models.is_empty() {
        println!("No models found in the local Hugging Face cache.");
        return Ok(());
    }

    println!("Local Models\n");
    for model in models {
        println!("{}", model.id);
        println!("  Size: {}", format_bytes(model.size_bytes));
    }

    Ok(())
}

pub async fn pull(model_id: &str) -> Result<()> {
    let interpreter = PythonInterpreter::default();
    println!("Pulling {model_id}... this can take several minutes for large models.");
    let result = pull_local_model(&interpreter, model_id)
        .await
        .map_err(|e| anyhow!(e))?;

    println!("Pulled {} ({})", result.id, format_bytes(result.size_bytes));
    Ok(())
}

pub async fn delete(model_id: &str) -> Result<()> {
    let interpreter = PythonInterpreter::default();
    let result = delete_local_model(&interpreter, model_id)
        .await
        .map_err(|e| anyhow!(e))?;

    println!(
        "Deleted {} (freed {})",
        result.id,
        format_bytes(result.freed_bytes)
    );
    Ok(())
}
