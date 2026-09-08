//! Agent with tools pulled from an MCP server, alongside a manually
//! registered tool.
//!
//! Requires a VeloxQuant (or other OpenAI-compatible) runtime at
//! `http://localhost:8765`, and an MCP server reachable over stdio. This
//! example spawns `npx -y @modelcontextprotocol/server-everything` (the
//! reference "everything" test server) as a stand-in — swap the command
//! for whichever MCP server you want to use. Run with:
//!
//! ```sh
//! cargo run --example mcp_agent --features mcp
//! ```

use veloxquant::{Agent, AgentRunOptions, Client};
use veloxquant::{McpServerConfig, McpTransport};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder().auto_detect().build()?;
    let mut agent = Agent::new(client, "mlx-community/Qwen3-8B-4bit");

    agent
        .use_mcp_server(McpServerConfig {
            name: "everything".to_string(),
            transport: McpTransport::Stdio {
                command: "npx".to_string(),
                args: vec![
                    "-y".to_string(),
                    "@modelcontextprotocol/server-everything".to_string(),
                ],
                env: vec![],
            },
        })
        .await?;

    let result = agent
        .run(
            "What tools do you have available?",
            AgentRunOptions::default(),
        )
        .await?;

    println!("Final response: {}", result.text);
    for step in &result.steps {
        println!("  called {} with {}", step.tool_name, step.args);
    }

    Ok(())
}
