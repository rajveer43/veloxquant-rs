//! Tool-calling agent example, mirroring
//! `~/Work/personal_projects/veloxquant-sdk/examples/agent.ts`.
//!
//! Registers a simple `get_weather` tool and runs a prompt that should
//! trigger the model to call it. Requires a VeloxQuant (or other
//! OpenAI-compatible) runtime at `http://localhost:8765` serving a model
//! that supports tool calling. Run with:
//!
//! ```sh
//! cargo run --example agent --features agent
//! ```

use serde_json::{json, Value};

use veloxquant::{Agent, AgentRunOptions, Client, Result, Tool};

/// A toy tool that returns a canned "weather report" for any location.
struct GetWeatherTool;

#[async_trait::async_trait]
impl Tool for GetWeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> Option<&str> {
        Some("Gets the current weather for a named location.")
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or place to get the weather for.",
                }
            },
            "required": ["location"],
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let location = args
            .get("location")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        Ok(json!({
            "location": location,
            "condition": "sunny",
            "temperature_celsius": 22,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::builder().auto_detect().build()?;

    let mut agent = Agent::new(client, "mlx-community/Qwen3-8B-4bit");
    agent.tool(Box::new(GetWeatherTool))?;

    let result = agent
        .run(
            "What's the weather like in Lisbon right now?",
            AgentRunOptions::default(),
        )
        .await?;

    println!("Final response: {}", result.text);
    for step in &result.steps {
        println!(
            "  called {} with {} -> {}",
            step.tool_name, step.args, step.result
        );
    }

    Ok(())
}
