//! `Agent`: a single-turn tool-calling loop over a chat-capable [`Client`].
//!
//! Mirrors `@veloxquant/sdk`'s `src/agent.ts`, reusing the OpenAI-compatible
//! `tools`/`tool_calls` wire shape end to end (see
//! [`veloxquant_openai::chat`]) rather than inventing a bespoke schema.
//!
//! Tools are manually registered via [`Agent::tool`], or (with the `mcp`
//! feature) pulled from a Model Context Protocol server via
//! [`Agent::use_mcp_server`](crate::mcp) — both share one name/dispatch
//! namespace inside [`Agent::run`]. There is still no multi-step planning
//! beyond the tool-call round-trip loop (call tools -> feed results back ->
//! repeat until the model stops calling tools or `max_steps` is hit).

use std::collections::HashMap;

use serde_json::Value;

use veloxquant_core::VeloxQuantError;
use veloxquant_openai::{ChatRequest, FunctionCall, FunctionDefinition, Message, ToolCall};

use crate::{Client, Result};

/// A tool an [`Agent`] can call.
///
/// Rust's ownership model makes a trait object (`Box<dyn Tool>`) the
/// idiomatic shape here — not a literal port of TS's
/// object-with-an-`execute`-function-field `ToolSpec` — since `Agent`'s
/// tool registry needs `Vec<Box<dyn Tool>>` object safety.
///
/// Implemented with `#[async_trait::async_trait]` rather than a native
/// `async fn` in the trait: native `async fn` in traits is stable at this
/// workspace's MSRV (1.75) but is not `dyn`-compatible, which this trait
/// needs to support a heterogeneous tool registry.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// The tool's name, used by the model to reference it in a tool call
    /// and as this agent's registration key.
    fn name(&self) -> &str;

    /// A human-readable description of what the tool does, shown to the
    /// model alongside its name and parameters.
    fn description(&self) -> Option<&str> {
        None
    }

    /// JSON Schema describing the tool's arguments object — the
    /// `parameters` field of an OpenAI tool definition (see `agent.ts:9`).
    fn parameters(&self) -> Value;

    /// Executes the tool with the given (already JSON-parsed) arguments,
    /// returning a JSON value to feed back to the model as the tool
    /// result.
    async fn execute(&self, args: Value) -> Result<Value>;
}

/// Options controlling an [`Agent::run`] call.
#[derive(Debug, Clone)]
pub struct AgentRunOptions {
    /// Maximum tool-call round trips before giving up. Default 8 — a
    /// runaway tool loop stops instead of looping forever, matching
    /// `agent.ts:14-15`.
    pub max_steps: usize,
    /// Maximum tokens to generate per chat completion, if overriding the
    /// runtime default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature, if overriding the runtime default.
    pub temperature: Option<f32>,
}

impl Default for AgentRunOptions {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_tokens: None,
            temperature: None,
        }
    }
}

/// A single executed tool call recorded during an [`Agent::run`] call.
#[derive(Debug, Clone)]
pub struct AgentStep {
    /// The name of the tool that was called.
    pub tool_name: String,
    /// The (already JSON-parsed) arguments the model supplied.
    pub args: Value,
    /// The tool's result (or a structured `{"error": ...}` value if the
    /// tool call could not be dispatched or the tool itself failed).
    pub result: Value,
}

/// The result of a completed [`Agent::run`] call.
#[derive(Debug, Clone)]
pub struct AgentRunResult {
    /// The model's final (non-tool-calling) response text.
    pub text: String,
    /// Every tool call executed along the way, in order.
    pub steps: Vec<AgentStep>,
}

/// Single-turn tool-calling agent over a [`Client`] with the `openai`
/// feature enabled.
///
/// Construct with [`Agent::new`]. Register tools with [`Agent::tool`] (and,
/// with the `mcp` feature, [`Agent::use_mcp_server`](crate::mcp)), then call
/// [`Agent::run`].
pub struct Agent {
    client: Client,
    default_model: String,
    tools: HashMap<String, Box<dyn Tool>>,
    #[cfg(feature = "mcp")]
    pub(crate) mcp_sources: Vec<crate::mcp::McpToolSource>,
}

impl Agent {
    /// Creates an agent that sends chat completions to `default_model` via
    /// `client`.
    pub fn new(client: Client, default_model: impl Into<String>) -> Self {
        Self {
            client,
            default_model: default_model.into(),
            tools: HashMap::new(),
            #[cfg(feature = "mcp")]
            mcp_sources: Vec::new(),
        }
    }

    /// Registers a tool on this agent.
    ///
    /// Errors (does not panic or silently overwrite) on a name collision
    /// with an already-registered tool — matching `agent.ts:67-72`'s
    /// `throw`.
    pub fn tool(&mut self, tool: Box<dyn Tool>) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(VeloxQuantError::ToolAlreadyRegistered(name));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Whether a tool with this name is already registered (manually or
    /// via an MCP source).
    pub(crate) fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Registers a tool without checking for a name collision first,
    /// returning `true` if it replaced (rather than added) an entry.
    ///
    /// Used internally by [`crate::mcp`]'s collision-checked
    /// `use_mcp_server`, which performs its own upfront collision scan
    /// across *all* of an MCP source's tools before committing any of them
    /// — see that module's doc comments for why a per-tool `Agent::tool`
    /// call isn't used there.
    pub(crate) fn insert_tool_unchecked(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    fn tool_definitions(&self) -> Vec<veloxquant_openai::ToolDefinition> {
        self.tools
            .values()
            .map(|t| veloxquant_openai::ToolDefinition {
                kind: veloxquant_openai::ToolDefinitionKind::Function,
                function: FunctionDefinition {
                    name: t.name().to_string(),
                    description: t.description().map(str::to_string),
                    parameters: t.parameters(),
                },
            })
            .collect()
    }

    /// Sends `prompt`, executing any tools the model calls and feeding
    /// their results back, until the model responds without calling a tool
    /// or `options.max_steps` round trips are used up (whichever comes
    /// first).
    ///
    /// Mirrors `agent.ts:122-181` exactly:
    /// - A malformed tool-call-arguments JSON from the model does not
    ///   abort the run — a structured error is fed back as the tool result
    ///   and the loop continues (`agent.ts:153-163`).
    /// - A tool call naming an unregistered tool likewise feeds back a
    ///   structured error rather than aborting.
    /// - A tool's own execution error is caught and stringified into the
    ///   tool-result message rather than aborting the whole run
    ///   (`agent.ts:165-170`).
    /// - Exceeding `max_steps` returns
    ///   [`VeloxQuantError::AgentMaxStepsExceeded`] rather than a generic
    ///   timeout (`agent.ts:177-180`).
    pub async fn run(&self, prompt: &str, options: AgentRunOptions) -> Result<AgentRunResult> {
        let chat = self.client.chat()?;
        let tool_defs = self.tool_definitions();
        let mut messages = vec![Message::user(prompt)];
        let mut steps = Vec::new();

        for _ in 0..options.max_steps {
            let request = ChatRequest {
                model: self.default_model.clone(),
                messages: messages.clone(),
                temperature: options.temperature,
                max_tokens: options.max_tokens,
                tools: if tool_defs.is_empty() {
                    None
                } else {
                    Some(tool_defs.clone())
                },
                stream: false,
            };

            let response = chat.send(request).await?;

            let tool_calls = match response.tool_calls {
                Some(calls) if !calls.is_empty() => calls,
                _ => {
                    return Ok(AgentRunResult {
                        text: response.text,
                        steps,
                    });
                }
            };

            messages.push(Message::assistant_with_tool_calls(
                response.text,
                tool_calls.clone(),
            ));

            for call in tool_calls {
                self.dispatch_tool_call(call, &mut messages, &mut steps)
                    .await;
            }
        }

        Err(VeloxQuantError::AgentMaxStepsExceeded {
            max_steps: options.max_steps,
        })
    }

    async fn dispatch_tool_call(
        &self,
        call: ToolCall,
        messages: &mut Vec<Message>,
        steps: &mut Vec<AgentStep>,
    ) {
        let FunctionCall { name, arguments } = call.function;

        let Some(tool) = self.tools.get(&name) else {
            let error = serde_json::json!({
                "error": format!("No tool named \"{name}\" is registered.")
            });
            messages.push(Message::tool(call.id, error.to_string()));
            return;
        };

        let args: Value = match serde_json::from_str(&arguments) {
            Ok(v) => v,
            Err(_) => {
                let error = serde_json::json!({
                    "error": format!("Could not parse arguments as JSON: {arguments}")
                });
                messages.push(Message::tool(call.id, error.to_string()));
                return;
            }
        };

        let result = match tool.execute(args.clone()).await {
            Ok(v) => v,
            Err(e) => serde_json::json!({ "error": e.to_string() }),
        };

        steps.push(AgentStep {
            tool_name: name,
            args,
            result: result.clone(),
        });
        messages.push(Message::tool(call.id, result.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> Option<&str> {
            Some("Echoes its input back")
        }

        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, args: Value) -> Result<Value> {
            Ok(args)
        }
    }

    struct FailingTool;

    #[async_trait::async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            "fail"
        }

        fn parameters(&self) -> Value {
            serde_json::json!({"type": "object"})
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            Err(VeloxQuantError::InvalidRequest("boom".to_string()))
        }
    }

    fn test_client() -> Client {
        Client::builder().build().unwrap()
    }

    #[test]
    fn tool_registration_errors_on_duplicate_name() {
        let mut agent = Agent::new(test_client(), "m");
        agent.tool(Box::new(EchoTool)).unwrap();
        let err = agent.tool(Box::new(EchoTool)).unwrap_err();
        assert!(matches!(err, VeloxQuantError::ToolAlreadyRegistered(name) if name == "echo"));
    }

    #[test]
    fn tool_definitions_reflect_registered_tools() {
        let mut agent = Agent::new(test_client(), "m");
        agent.tool(Box::new(EchoTool)).unwrap();
        let defs = agent.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].function.name, "echo");
        assert_eq!(
            defs[0].function.description.as_deref(),
            Some("Echoes its input back")
        );
    }

    #[tokio::test]
    async fn failing_tool_execution_is_captured_as_error_result_not_propagated() {
        let mut agent = Agent::new(test_client(), "m");
        agent.tool(Box::new(FailingTool)).unwrap();

        let mut messages = Vec::new();
        let mut steps = Vec::new();
        let call = ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: "fail".to_string(),
                arguments: "{}".to_string(),
            },
        };

        agent
            .dispatch_tool_call(call, &mut messages, &mut steps)
            .await;

        assert_eq!(steps.len(), 1);
        assert!(steps[0].result.get("error").is_some());
        assert_eq!(messages.len(), 1);
        assert!(messages[0].content.contains("boom"));
    }

    #[tokio::test]
    async fn malformed_arguments_json_does_not_abort_and_feeds_back_structured_error() {
        let mut agent = Agent::new(test_client(), "m");
        agent.tool(Box::new(EchoTool)).unwrap();

        let mut messages = Vec::new();
        let mut steps = Vec::new();
        let call = ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: "{not valid json".to_string(),
            },
        };

        agent
            .dispatch_tool_call(call, &mut messages, &mut steps)
            .await;

        // No step recorded (arguments never resolved to a tool
        // execution), but the loop must continue: a tool-result message
        // with a structured error was still fed back.
        assert!(steps.is_empty());
        assert_eq!(messages.len(), 1);
        assert!(messages[0].content.contains("Could not parse arguments"));
        assert_eq!(messages[0].role, veloxquant_openai::Role::Tool);
    }

    #[tokio::test]
    async fn unregistered_tool_call_feeds_back_structured_error() {
        let agent = Agent::new(test_client(), "m");

        let mut messages = Vec::new();
        let mut steps = Vec::new();
        let call = ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: "nonexistent".to_string(),
                arguments: "{}".to_string(),
            },
        };

        agent
            .dispatch_tool_call(call, &mut messages, &mut steps)
            .await;

        assert!(steps.is_empty());
        assert_eq!(messages.len(), 1);
        assert!(messages[0].content.contains("No tool named"));
        assert!(messages[0].content.contains("nonexistent"));
        assert!(messages[0].content.contains("is registered"));
    }

    #[tokio::test]
    async fn successful_tool_call_records_a_step_and_feeds_back_result() {
        let mut agent = Agent::new(test_client(), "m");
        agent.tool(Box::new(EchoTool)).unwrap();

        let mut messages = Vec::new();
        let mut steps = Vec::new();
        let call = ToolCall {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: r#"{"x":1}"#.to_string(),
            },
        };

        agent
            .dispatch_tool_call(call, &mut messages, &mut steps)
            .await;

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].tool_name, "echo");
        assert_eq!(steps[0].result, serde_json::json!({"x": 1}));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, veloxquant_openai::Role::Tool);
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_1"));
    }
}

/// End-to-end `Agent::run()` tests against a fake HTTP server standing in
/// for a chat-completions-serving runtime, using the same bare-bones
/// TCP-listener mocking pattern as `veloxquant-openai`/`veloxquant-runtime`'s
/// existing tests (no mocking framework dependency).
#[cfg(test)]
mod run_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    struct NoopTool;

    #[async_trait::async_trait]
    impl Tool for NoopTool {
        fn name(&self) -> &str {
            "noop"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object"})
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            Ok(json!({"ok": true}))
        }
    }

    use serde_json::json;

    /// Starts a server that returns each of `responses` in order (one per
    /// request received), looping the last response for any request beyond
    /// the list, and records the number of requests it has served.
    async fn spawn_sequenced_json_server(
        responses: Vec<&'static str>,
    ) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let counter = request_count.clone();

        tokio::spawn(async move {
            let mut idx = 0usize;
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 8192];
                let _ = socket.read(&mut buf).await;

                let body = responses
                    .get(idx.min(responses.len() - 1))
                    .copied()
                    .unwrap_or("{}");
                idx += 1;
                counter.fetch_add(1, Ordering::SeqCst);

                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(header.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });

        (format!("http://{addr}"), request_count)
    }

    fn client_for(base_url: &str) -> Client {
        Client::builder().runtime_url(base_url).build().unwrap()
    }

    #[tokio::test]
    async fn run_returns_immediately_when_no_tool_calls() {
        let (base_url, requests) =
            spawn_sequenced_json_server(vec![r#"{"id":"1","model":"m","text":"hello there"}"#])
                .await;

        let agent = Agent::new(client_for(&base_url), "m");
        let result = agent.run("hi", AgentRunOptions::default()).await.unwrap();

        assert_eq!(result.text, "hello there");
        assert!(result.steps.is_empty());
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn run_dispatches_a_tool_call_and_feeds_result_back() {
        let (base_url, requests) = spawn_sequenced_json_server(vec![
            r#"{"id":"1","model":"m","text":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"noop","arguments":"{}"}}]}"#,
            r#"{"id":"2","model":"m","text":"done"}"#,
        ])
        .await;

        let mut agent = Agent::new(client_for(&base_url), "m");
        agent.tool(Box::new(NoopTool)).unwrap();

        let result = agent.run("hi", AgentRunOptions::default()).await.unwrap();

        assert_eq!(result.text, "done");
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].tool_name, "noop");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn run_enforces_max_steps() {
        // Every response keeps calling the tool, so the loop should run
        // exactly `max_steps` times and then return AgentMaxStepsExceeded
        // rather than looping forever.
        let (base_url, requests) = spawn_sequenced_json_server(vec![
            r#"{"id":"1","model":"m","text":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"noop","arguments":"{}"}}]}"#,
        ])
        .await;

        let mut agent = Agent::new(client_for(&base_url), "m");
        agent.tool(Box::new(NoopTool)).unwrap();

        let options = AgentRunOptions {
            max_steps: 3,
            ..Default::default()
        };
        let err = agent.run("hi", options).await.unwrap_err();

        assert!(matches!(
            err,
            VeloxQuantError::AgentMaxStepsExceeded { max_steps: 3 }
        ));
        assert_eq!(requests.load(Ordering::SeqCst), 3);
    }
}
