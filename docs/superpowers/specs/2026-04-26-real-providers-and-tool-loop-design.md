# Design: Real LLM providers + tool-call loop + stdio MCP

**Date:** 2026-04-26
**Status:** Approved (awaiting written-spec review)
**Slice:** Follow-on to slice (e) platform skeleton — bundles two adapter-side slices into one PR.

## Context

Slice (e) shipped the platform skeleton with three documented v1 limitations on the `ph0b0s-llm-adk` adapter:

1. **Only `PH0B0S_PROVIDER=mock` is wired.** Real provider construction (Anthropic / OpenAI / Gemini / Ollama) returns a clear "not yet wired" error. Until this lands, every detection-pack downstream is stuck running against canned responses.
2. **No tool-call loop.** `LlmAgent::chat` is single-shot. If the model emits a `Part::FunctionCall`, the adapter logs a warning and returns the call unhandled. Detection packs cannot give the model tools that the model can actually use autonomously.
3. **`ToolHost::mount_mcp` records the spec but doesn't connect.** MCP servers configured in `ph0b0s.toml` are accepted but never spawn.

This slice lifts all three. They are bundled into one PR because:

- All three changes live entirely inside `ph0b0s-llm-adk` (plus a thin CLI dispatcher).
- The seam in `ph0b0s-core` does not change. No detection-pack code recompiles.
- The CLI's vendor-coupling allow-list slot stops being load-bearing for daily work — it stays for startup wiring only.

The intended outcome is that, on merge, a user can:

```bash
ANTHROPIC_API_KEY=sk-ant-... cargo run -p ph0b0s-cli -- scan ./some-repo
```

and the LLM-toy detector produces real model-generated findings, with MCP tools (if mounted) available to the model through the same call path.

## Hard constraints (preserved from slice (e))

1. **Vendor neutrality.** Detection-pack crates may not import any `adk_*` or vendor SDK. Enforced by `xtask check-vendor`. After this slice, only `ph0b0s-llm-adk` and `ph0b0s-cli` import `adk-*` — and the CLI's import shrinks to almost nothing because provider construction moves into the adapter.
2. **API keys never in TOML.** `config check` rejects `api_key` in any TOML layer. New `[providers.<name>]` blocks must follow this rule (only `default_model`, `base_url` — never the secret).
3. **`adk-rust = "=0.6.0"`** stays exact-pinned in the adapter only.
4. **The seam in `ph0b0s-core` does not change.** No new public traits, no breaking changes to existing ones.

## Architectural decisions

These were settled during brainstorming.

| # | Decision | Rationale |
|---|---|---|
| 1 | **All four providers wired** (Anthropic + OpenAI + Gemini + Ollama) + construction-tested. Ollama tested live in CI. Anthropic, OpenAI, Gemini construction-tested only — no API keys in CI. | Two real provider impls is the minimum to validate the abstraction; Ollama is the only free path to live verification. |
| 2 | **TOML-driven provider selection with env-detection fallback.** `[agents.default] provider = "anthropic" model = "..."` is authoritative. If absent, fall back to env-key precedence (`ANTHROPIC_API_KEY` → Anthropic, `OPENAI_API_KEY` → OpenAI, `GOOGLE_API_KEY` → Gemini, `OLLAMA_HOST` → Ollama). `PH0B0S_PROVIDER=<name>` always wins for ad-hoc overrides. **Single global agent** for now; per-role assignment deferred. | Maps to the convention: TOML for config, env for secrets. Per-role wiring would need to thread multiple agents through the orchestrator and let detectors pick by role — significant code with no current consumer. |
| 3 | **`chat()` runs the tool-call loop**, surfacing tool errors to the model as `FunctionResponse{"error": ...}`. Up to 10 turns by default (override via `ChatRequest.hints["max_tool_turns"]`). Sequential dispatch when the model emits multiple `FunctionCall`s in one turn. | Standard pattern in modern Anthropic / OpenAI / adk SDKs. Detection-pack authors get a single response and don't reimplement plumbing. Tool errors as `FunctionResponse` lets the model self-recover from transient failures. |
| 4 | **Tool visibility:** `req.tools` overrides; if empty, fall back to all host-registered tools (native + MCP). | Pragmatic single-axis selector. Smoke runs and one-shot calls get the full toolbox; detectors that want isolation pass an explicit list. |
| 5 | **MCP scope: stdio only via `adk_tool::McpToolset`.** SSE / StreamableHTTP entries get a `WARN` log and are recorded as mounted but no live connection. | ~95% of public MCP servers use stdio. Each transport has its own pre-1.0 adk-rust API surface; we extend without seam churn when an actual user has a non-stdio server. |
| 6 | **Live Ollama tests in a dedicated CI job** with a tiny model (`qwen2.5-coder:0.5b`, ~400 MB), pulled and cached. Other providers are construction-tested only. | Real live coverage of one provider catches integration bugs the mock can't. Constrained to one CI job, parallel with the rest, gated on `test`. |
| 7 | **Architecture factoring:** the adapter owns per-provider construction; the CLI is a thin dispatcher. | Keeps adk knowledge in one crate. CLI's `provider.rs` shrinks from ~140 LOC to ~30 LOC. Adding a new provider becomes one focused PR in one crate. |

## Architecture

```
crates/ph0b0s-llm-adk/                  (the adapter — only crate that touches adk-*)
├── src/
│   ├── lib.rs                          — re-exports + public surface
│   ├── agent.rs                        — AdkLlmAgent + AdkSession
│   │                                     + NEW: tool-call loop in chat() and session.send()
│   ├── tools.rs                        — AdkToolHost
│   │                                     + NEW: mount_mcp delegates to mcp.rs
│   ├── mcp.rs            (NEW)         — rmcp stdio peer + adk_tool::McpToolset wiring
│   ├── config.rs         (NEW)         — AgentConfig, ProviderConfig, ProviderRegistry
│   ├── provider/         (NEW)
│   │   ├── mod.rs                      — build_from_env(), build_from_config()
│   │   ├── anthropic.rs                — build_anthropic(model)
│   │   ├── openai.rs                   — build_openai(model, base_url)
│   │   ├── gemini.rs                   — build_gemini(model)
│   │   ├── ollama.rs                   — build_ollama(model, base_url)
│   │   └── mock.rs                     — build_mock(canned)  (moved out of CLI)
│   ├── error.rs                        — + BuildError variants
│   └── usage.rs                        — unchanged
└── tests/
    ├── ollama_live.rs    (NEW)         — #[ignore] live tests against a local Ollama
    └── fixtures/
        └── fake_mcp.py   (NEW)         — tiny stdio MCP server for hermetic tests

crates/ph0b0s-cli/
├── src/
│   ├── provider.rs                     — shrinks to ~30 LOC dispatcher.
│   │                                     Reads ph0b0s-cli config, calls
│   │                                     ph0b0s_llm_adk::provider::build_from_config()
│   │                                     or ::build_from_env() as fallback.
│   ├── config.rs                       — figment Config gains [providers.<name>]
│   │                                     and [agents.default] mappings; emits the
│   │                                     adapter's typed AgentConfig/ProviderRegistry.
│   └── …                               — no other CLI changes
└── …

.github/workflows/ci.yml                — + live-ollama job (parallel with the rest)
```

**Invariants preserved.**
- `xtask check-vendor`: still passes. `ph0b0s-llm-adk` and `ph0b0s-cli` remain the only crates with `adk_*` imports — but the CLI's import shrinks back to almost nothing (it calls adapter functions, not adk types directly).
- The seam (`LlmAgent`, `LlmSession`, `ToolHost`, etc. in `ph0b0s-core`) does NOT change. No detection-pack code recompiles, none of its tests change.
- v1 limitations being lifted: real provider construction, tool-call loop, MCP stdio mounting. No new limitations introduced.

## Provider builders

Each builder lives in its own file under `ph0b0s-llm-adk/src/provider/`, takes minimal arguments, reads its API key from a canonical env var, and returns `Result<AdkLlmAgent, BuildError>`.

```rust
// ph0b0s-llm-adk/src/provider/anthropic.rs
const DEFAULT_MODEL: &str = "claude-sonnet-4-6";

pub fn build(model: Option<&str>) -> Result<AdkLlmAgent, BuildError> {
    let api_key = require_env("ANTHROPIC_API_KEY")?;
    let model = model.unwrap_or(DEFAULT_MODEL).to_owned();
    let cfg = adk_rust::model::AnthropicConfig::new(api_key, model.clone());
    let client = adk_rust::model::AnthropicClient::new(cfg)
        .map_err(|e| BuildError::Adk(e.to_string()))?;
    Ok(AdkLlmAgent::new(Arc::new(client), model))
}
```

Other providers follow the same shape:

| Provider | API-key env var | Default model | Extra knob |
|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | — |
| OpenAI | `OPENAI_API_KEY` | `gpt-5-mini` | `base_url` (OpenAI-compatible endpoints — Azure, OpenRouter, etc.) |
| Gemini | `GOOGLE_API_KEY` | `gemini-2.5-flash` | — |
| Ollama | (none — local) | `llama3.2:3b` | `base_url` (default `http://localhost:11434`) |
| Mock | (none) | `"ph0b0s-mock"` | optional `PH0B0S_MOCK_RESPONSES` file (existing behavior) |

**Default-model rationale.** Hardcoded per-provider defaults so a fresh user can run `ph0b0s scan .` with just `ANTHROPIC_API_KEY=...` and no TOML. All four defaults are quick / cheap models suitable for smoke runs. Production users override via `[providers.<name>] default_model = "..."`.

**`BuildError` variants** in `error.rs`:

```rust
pub enum BuildError {
    /// Required env var not set.
    MissingKey(&'static str),
    /// Provider name in config didn't match any builder.
    UnknownProvider(String),
    /// No `[agents.default]` and no env var set for any provider.
    NoProviderConfigured,
    /// Underlying adk-rust client constructor failed.
    Adk(String),
    /// Mock-response file unreadable / not JSON / not a JSON array.
    Mock(String),
}
```

The mock path moves out of the CLI: previously in `ph0b0s-cli/src/provider.rs`, now in `ph0b0s-llm-adk/src/provider/mock.rs`. Same env-var contract (`PH0B0S_MOCK_RESPONSES`); same canned-response shape; existing tests move with the code.

## CLI dispatcher + config-shape contract

The adapter exports the canonical config types; the CLI's figment `Config` contains the same shape.

```rust
// ph0b0s-llm-adk/src/config.rs (NEW)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentConfig {
    pub provider: String,                       // "anthropic" | "openai" | "gemini" | "ollama" | "mock"
    pub model: Option<String>,                  // overrides the per-provider default
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub default_model: Option<String>,
    pub base_url: Option<String>,               // honoured by OpenAI + Ollama only
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProviderRegistry {
    pub anthropic: Option<ProviderConfig>,
    pub openai:    Option<ProviderConfig>,
    pub gemini:    Option<ProviderConfig>,
    pub ollama:    Option<ProviderConfig>,
}
```

**Selection algorithm** in `ph0b0s-llm-adk/src/provider/mod.rs`:

```rust
pub fn build_from_config(
    agent: Option<&AgentConfig>,
    providers: &ProviderRegistry,
) -> Result<AdkLlmAgent, BuildError> {
    // 1. PH0B0S_PROVIDER env override always wins (unblocks ad-hoc runs).
    if let Ok(name) = std::env::var("PH0B0S_PROVIDER") {
        return build_named(&name, providers, agent.and_then(|a| a.model.as_deref()));
    }
    // 2. Explicit [agents.default] from TOML.
    if let Some(a) = agent {
        return build_named(&a.provider, providers, a.model.as_deref());
    }
    // 3. Env-key precedence fallback.
    build_from_env(providers)
}

pub fn build_from_env(providers: &ProviderRegistry) -> Result<AdkLlmAgent, BuildError> {
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return anthropic::build(model_for("anthropic", providers));
    }
    if std::env::var("OPENAI_API_KEY").is_ok() {
        return openai::build(model_for("openai", providers), base_url_for("openai", providers));
    }
    if std::env::var("GOOGLE_API_KEY").is_ok() {
        return gemini::build(model_for("gemini", providers));
    }
    if std::env::var("OLLAMA_HOST").is_ok() {
        return ollama::build(model_for("ollama", providers), base_url_for("ollama", providers));
    }
    Err(BuildError::NoProviderConfigured)
}
```

The CLI's `provider.rs` shrinks to:

```rust
// ph0b0s-cli/src/provider.rs (rewritten)
use anyhow::Result;
use ph0b0s_llm_adk::{provider, AdkLlmAgent};
use crate::config::Config;

pub fn build(config: &Config) -> Result<AdkLlmAgent> {
    provider::build_from_config(
        config.agents.get("default"),
        &config.provider_registry(),
    ).map_err(Into::into)
}
```

…where `Config::provider_registry()` is a small mapper that converts the figment-loaded `HashMap<String, ProviderConfig>` (already slice-(e) shape) into the typed `ProviderRegistry` the adapter expects.

**No new env vars** introduced — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_HOST`, `PH0B0S_PROVIDER`, `PH0B0S_MOCK_RESPONSES` are all canonical / already documented.

**`config check` already rejects `api_key` keys** — that guardrail extends naturally to the new `[providers.*]` blocks (no code change needed; the regex scans every line of every TOML layer).

## Tool-call loop algorithm

Lives in `ph0b0s-llm-adk/src/agent.rs`. Replaces the current single-shot path in `LlmAgent::chat` and reused by `LlmSession::send` via a private `run_loop` helper.

```text
chat(req) -> ChatResponse:
    # 1. Resolve which tools the model sees this call. (Decision 4.)
    tools = if req.tools.is_empty():
        tool_host.list()                       # native + MCP-mounted
    else:
        req.tools

    max_turns = req.hints.get("max_tool_turns").as_u64().unwrap_or(10)

    # 2. Build initial conversation from req.messages.
    conversation = req.messages.into_adk_contents(default_system)
    cumulative_usage = Usage::default()

    for turn in 0..max_turns:
        adk_req = LlmRequest {
            model: self.model_id,
            contents: conversation.clone(),
            tools: tools.iter().map(to_adk_tool_decl).collect(),
            config: ...,
        }
        response = collect_final(self.llm.generate_content(adk_req, false).await)?
        cumulative_usage = cumulative_usage + from_adk_usage(response.usage_metadata)

        function_calls = response.content.parts.iter()
            .filter_map(Part::FunctionCall)
            .collect()

        if function_calls.is_empty():
            # Final turn — return the assistant text.
            return ChatResponse {
                content: extract_text(response.content),
                tool_calls: vec![],                   # empty: loop owned them
                usage: cumulative_usage,
                finish_reason: map_finish_reason(response.finish_reason),
            }

        # 3. Append the model's tool-calling turn to history,
        #    then dispatch each call sequentially.
        conversation.push(Content::model(response.content))

        tool_responses = vec![]
        for fc in function_calls:                     # sequential, in emission order
            result = tool_host.invoke(&fc.name, fc.args.clone()).await
            tool_responses.push(match result {
                Ok(value) =>
                    Part::FunctionResponse {
                        function_response: FunctionResponseData::new(&fc.name, value),
                        id: fc.id.clone(),
                    },
                Err(tool_err) =>
                    Part::FunctionResponse {
                        function_response: FunctionResponseData::new(
                            &fc.name,
                            json!({"error": tool_err.to_string()}),
                        ),
                        id: fc.id.clone(),
                    },
            })

        # 4. Append the tool-result turn and loop.
        conversation.push(Content::tool(tool_responses))

    # 5. Loop exhausted.
    return Err(LlmError::ToolDispatch(format!(
        "model exceeded max_tool_turns ({}) without producing a final reply",
        max_turns
    )))
```

**Key invariants.**
- **Sequential dispatch** when the model emits multiple `FunctionCall`s in one turn. Parallel-tool-calling is a v3 optimization; documented but not implemented.
- **Tool errors stay inside the loop** as `FunctionResponse` payloads with `{"error": ...}`. The only way `chat()` itself returns `LlmError::ToolDispatch` is if the loop exhausts `max_turns` — that's a stuck-model signal, not a tool failure.
- **`Usage` accumulates across turns.** Prompt+completion tokens from every `LlmResponse` sum into the final `ChatResponse.usage`. `cost_source` stays `Native` if the provider returned usage on every turn, else degrades to `Estimate`.
- **Tool call IDs preserved** (`fc.id`): OpenAI-style providers require the same id on the matching `FunctionResponse`; Gemini ignores it. Pass-through is safe both ways.
- **`structured()` does NOT run the loop.** It's a single-shot constrained-output call. Tools are permitted in `StructuredRequest`, but the same single-shot semantics as today: if the model emits a `FunctionCall` instead of JSON, that's a `LlmError::StructuredValidation` (the model failed the schema). Documented explicitly as a v1 behavior.
- **`AdkSession::send` shares the same loop** via a private `run_loop` helper — same termination conditions, same tool resolution, same usage accumulation.

## MCP integration

`mount_mcp` becomes a real connection. New `mcp.rs` is the only file that imports from `adk_tool::` and `rmcp::` (no other crate, not even other adapter files, references MCP types directly). Both crates are already covered by the `ph0b0s-llm-adk` allow-list in `xtask check-vendor`.

**Two-step construction** (verified against adk-rust 0.6.0 + rmcp via context7). adk-rust's `McpToolset` wraps a connected rmcp peer; it does not own subprocess spawning. So `mcp.rs` does both:

1. Spawn the MCP server with `rmcp::transport::TokioChildProcess::new(Command)` and connect via `().serve(transport).await` to obtain a peer.
2. Wrap that peer in `adk_tool::McpToolset::new(peer)`, optionally calling `.with_name(server_name)` and `.with_filter(|name| ...)` if the spec carries a tool allow-list (deferred — schema-shaped, not implemented in v1).
3. Enumerate tools via `toolset.tools(&ReadonlyContext::default()).await` — returns `Vec<Arc<dyn adk_rust::Tool>>`. Each Tool exposes `name()`, `description()`, `parameters_schema()`, and `execute(ctx, args)`.

**Wiring.** When `AdkToolHost::mount_mcp(spec)` is called:

1. **Transport gate.** If `spec.transport != Stdio`, log `WARN` (`"non-stdio MCP transports not yet supported"`), append the spec to `mounted_mcp` for observability, return `Ok(())`. SSE / HTTP wait for a future slice.
2. **Spawn + connect.** Build a `tokio::process::Command` from `spec.command_or_url[0]` (program), `spec.command_or_url[1..]` (args), and `spec.env`. Hand it to `TokioChildProcess::new(cmd)?` and `().serve(child).await?` to get a connected peer. Wrap with `McpToolset::new(peer).with_name(spec.name)`.
3. **Enumerate.** Call `toolset.tools(&ctx).await` (initial `tools/list` already fetched at `serve` time). Each discovered tool gets wrapped in an internal `McpToolWrapper` that implements our seam's `NativeTool`:

   ```rust
   struct McpToolWrapper {
       server_name: String,                       // e.g. "filesystem"
       inner:       Arc<dyn adk_rust::Tool>,      // the per-tool handle from McpToolset::tools()
       schema:      serde_json::Value,            // captured from inner.parameters_schema()
   }

   #[async_trait]
   impl NativeTool for McpToolWrapper {
       fn spec(&self) -> ToolSpec {
           ToolSpec {
               name:        self.inner.name().to_owned(),
               description: Some(self.inner.description().to_owned()),
               schema:      self.schema.clone(),
               source:      ToolSource::Mcp { server: self.server_name.clone() },
           }
       }

       async fn call(&self, args: Value) -> Result<Value, ToolError> {
           // adk-rust 0.6 Tool::execute signature: (Arc<dyn ToolContext>, Value) -> Result<Value, AdkError>
           let ctx = Arc::new(NoopToolContext) as Arc<dyn adk_rust::ToolContext>;
           self.inner.execute(ctx, args).await
               .map_err(|e| ToolError::Execution(format!("{}: {e}", self.server_name)))
       }
   }
   ```

   `NoopToolContext` is a tiny in-crate stub for the parts of `ToolContext` that MCP tools don't consume (no session bridging in v1; documented as a known limitation if any MCP server actually needs it).

4. **Register.** Each wrapper goes through the existing `register_native()` so the rest of the system (incl. the tool-call loop's `tool_host.list()` fallback) sees MCP tools and Rust-native tools as equivalent — same dispatch path, same error type.
5. **Lifecycle.** `AdkToolHost` gains `mcp_servers: Mutex<HashMap<String, McpHandle>>`, where `McpHandle { toolset: Arc<McpToolset>, cancel: CancellationToken }`. The cancel token comes from `toolset.cancellation_token().await` and lets us terminate the rmcp peer + subprocess cleanly on shutdown. On drop, we call `cancel.cancel()` and let the rmcp transport's own `Drop` reap the child. Falling back to plain drop works (rmcp closes stdio on drop, server exits on EOF) but emits avoidable `EPIPE` noise — explicit cancel is the documented path.

**Error semantics.**

| Failure mode | `mount_mcp` returns |
|---|---|
| Subprocess failed to spawn (binary not on PATH, EPERM) | `ToolError::McpTransport("failed to start <server>: <err>")` — caller decides whether scan continues |
| Initial `initialize` / `tools/list` request failed or timed out | `ToolError::McpTransport(...)` with adk-tool's message |
| Per-tool `call(...)` failed at runtime | `ToolError::Execution(...)` — flows back through the tool-call loop as a `FunctionResponse{"error": ...}` |

**Known v1 limitations** (documented inline in `mcp.rs`):

- **Stdio only.** SSE / StreamableHTTP record the spec, log a warn, and skip.
- **Name collisions** — if two MCP servers expose the same tool name, the second mount wins (`HashMap` insert). No prefixing in v1; document and move on. v2 candidate: prefix with `<server>__<tool>` when collisions are detected.
- **No live re-discovery.** Tools are listed once at mount time. If an MCP server changes its tool set mid-run, we don't pick that up. Acceptable for a security scanner with a bounded run lifetime.

**`xtask check-vendor` impact:** zero. `ph0b0s-llm-adk` is already allow-listed for `adk_*` imports; `mcp.rs` joins `agent.rs` / `tools.rs` under that allowance.

## Testing strategy

Three test tiers.

**Tier 1 — hermetic unit tests** (run on every `cargo test`, every PR, every push):

| Target | Pattern |
|---|---|
| Per-provider builders (`anthropic`, `openai`, `gemini`, `ollama`) | Env-stubbed (existing `EnvScope` helper extended with a static `Mutex<()>` for cross-test serialization). Asserts: happy path returns an `AdkLlmAgent` with correct `model_id` / `role`; missing key → `BuildError::MissingKey("ANTHROPIC_API_KEY")` etc.; custom model passes through. **No network** — adk client constructors don't ping at construction time. |
| `build_from_config` / `build_from_env` selection logic | Table-driven: vary `[agents.default]` presence + `PH0B0S_PROVIDER` + each canonical env var; assert which builder fires + the resulting `model_id`. |
| Tool-call loop in `chat()` | A new in-crate `FakeAdkLlm` (impl `adk_rust::Llm` directly, in `agent.rs#[cfg(test)]`) returns a canned `Vec<LlmResponse>` per `generate_content` call: e.g. `LlmResponse{ Part::FunctionCall("search", …) }` then `LlmResponse{ Part::Text("ok") }`. Our existing `MockToolHost` (from `ph0b0s-test-support`) carries the canned `"search"` response. Assertions: final `ChatResponse.content == "ok"`; `tool_host.invocations()` shows the call; cumulative `Usage` summed across turns. Plus: max-turns exceeded → `LlmError::ToolDispatch`; tool error → next turn fed a `FunctionResponse{"error": …}`; multiple `FunctionCall`s in one turn → sequential dispatch order preserved. The tool-call loop sits between adk's `Llm` trait and our `ToolHost`, so the mock has to be on the adk side — `MockLlmAgent` from `ph0b0s-test-support` mocks our outer seam, not adk's. |
| MCP wrapper (`McpToolWrapper`) | Hermetic: a tiny Python stdio MCP server in `crates/ph0b0s-llm-adk/tests/fixtures/fake_mcp.py` registering one `ping` tool that returns `{"pong": true}`. Test `mount_mcp` against it, assert `host.list()` exposes `ping`, then `host.invoke("ping", {})` round-trips. Cancel-token cleanup tested by mounting + dropping in scope and asserting the child has exited. `#[cfg(unix)]` because stdio MCP is inherently process-based. |

**Tier 2 — adapter smoke** (already exists from slice (e), still works): the existing `FakeAdkLlm`-backed chat / structured / session tests in `agent.rs` keep proving the adapter shape regardless of provider. The new tool-loop tests extend the same fake.

**Tier 3 — live Ollama** (env-gated, dedicated CI job): `crates/ph0b0s-llm-adk/tests/ollama_live.rs`. Marked `#[ignore]`; only runs with `--include-ignored`. Three tests against a real `qwen2.5-coder:0.5b` model:

```rust
#[tokio::test] #[ignore]
async fn chat_returns_non_empty_response() { /* assert content isn't empty + finish_reason == Stop */ }

#[tokio::test] #[ignore]
async fn structured_emits_parseable_json() { /* schema = {properties:{ok:bool}}; assert v["ok"].is_boolean() */ }

#[tokio::test] #[ignore]
async fn session_multi_turn_accumulates_usage() { /* two sends; assert usage.tokens_in monotonically increases */ }
```

**Live tool-call loop test is deliberately skipped at this tier** — small models (`0.5b`) have unreliable function-calling. The hermetic tests in Tier 1 give deterministic loop coverage; bumping to a tool-capable model in CI would explode runtime + cost. v2 candidate when we have a tool-call-grade live target.

**CI changes** in `.github/workflows/ci.yml`:

```yaml
live-ollama:
  name: live (ollama)
  runs-on: ubuntu-latest
  needs: test                          # only after the matrix tests pass
  steps:
    - uses: actions/checkout@v4
    - name: Install Linux system deps
      run: sudo apt-get update && sudo apt-get install -y libdbus-1-dev pkg-config
    - uses: dtolnay/rust-toolchain@stable
      with: { toolchain: stable }
    - uses: Swatinem/rust-cache@v2
      with: { shared-key: ph0b0s-live-ollama }
    - name: Cache Ollama models
      uses: actions/cache@v4
      with:
        path: ~/.ollama/models
        key: ollama-${{ runner.os }}-qwen2.5-coder-0.5b
    - name: Install + start Ollama
      run: |
        curl -fsSL https://ollama.com/install.sh | sh
        ollama serve > /tmp/ollama.log 2>&1 &
        until curl -sf http://localhost:11434/api/tags > /dev/null; do sleep 1; done
    - name: Pull model
      run: ollama pull qwen2.5-coder:0.5b
    - name: Run live tests
      env:
        OLLAMA_HOST: http://localhost:11434
        PH0B0S_LIVE_OLLAMA_MODEL: qwen2.5-coder:0.5b
      run: cargo test -p ph0b0s-llm-adk --test ollama_live -- --include-ignored
```

Cached model = ~3 min cold start, ~30s warm. Job runs in parallel with `coverage` / `cargo-deny`; gated on `test` so we don't burn Ollama-install time when something earlier already failed.

**Coverage** stays gated at 100% on patch (set by slice (e)). New code:

- Provider builders: 100% (small functions, easy)
- Tool-call loop: high coverage hermetically; max-turns + tool-error + multi-call branches all have explicit tests
- MCP wrapper: covered by the fake-MCP-server test
- Live tests: explicitly excluded from coverage via `--ignore-filename-regex` (live tests run live, don't gate the coverage number)

**Workspace test count** projected: 178 → ~210 (+ ~32 new tests across builders, loop, MCP).

## Non-goals (explicit; documented in code where relevant)

- **Per-role agent assignment.** Single global `AdkLlmAgent` per scan. The seam's `AgentRoleKey` mechanism stays, but the CLI only constructs one agent. Layered on later when an actual detector wants split roles.
- **Non-stdio MCP transports** (SSE, StreamableHTTP). `mount_mcp` warns + records-only.
- **Parallel tool-call dispatch.** When the model emits N `FunctionCall`s in one turn, we dispatch sequentially. v3 candidate.
- **Streaming chat / structured.** Still single-shot. Default `unimplemented!()` slot reserved for `chat_stream()` per the slice (e) plan.
- **Tool-call loop in `structured()`.** Single-shot constrained-output. If the model emits a `FunctionCall` instead of JSON, that's `LlmError::StructuredValidation`.
- **Live tests against Anthropic / OpenAI / Gemini.** No API keys in CI. Construction-tested only at this tier.
- **Tool-name collision prefixing across MCP servers.** Last-mount wins; documented in `mcp.rs`.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| `adk-tool::McpToolset` doesn't surface enough of the underlying rmcp peer for our needs | We already use rmcp directly to connect (see MCP integration section). If `McpToolset::tools()` proves awkward (e.g. `ToolContext` requirements bleed through), we can drop McpToolset entirely and call rmcp's `peer.list_tools()` + `peer.call_tool(...)` directly — same wrapper shape, no seam change. ~half-day spike if needed. |
| Tiny Ollama model output is flaky | Live tests assert structural properties (non-empty, JSON-parseable, monotonic usage), not exact content. Pin model + version (`qwen2.5-coder:0.5b`) so behavior is reproducible across CI runs. |
| Provider constructors do network at init (would break offline tests) | If true, hermetic builder tests would hang. Validate during impl: a 30-second test against each adk-rust client constructor with a fake key. Anthropic / OpenAI / Gemini SDK clients are typically lazy; document if any aren't. |
| Test env-var races | Tests serialize via a static `Mutex<()>`. `EnvScope::lock()` returns a guard; every env-touching test starts with `let _g = ENV_LOCK.lock();`. |
| Ollama install in CI is flaky | Cache `~/.ollama/models` keyed on model+OS; gate live job on `test` job (don't burn ~3 min on Ollama install if something earlier already failed). |
| `BuildError::Adk` swallows useful detail | Keep the original `adk_rust::AdkError`'s `Display` output verbatim in the `String` payload. We already do this for `LlmError::Provider`. |

## Critical files

To be created:

- `crates/ph0b0s-llm-adk/src/config.rs` — `AgentConfig`, `ProviderConfig`, `ProviderRegistry`
- `crates/ph0b0s-llm-adk/src/mcp.rs` — `adk_tool::McpToolset` integration + `McpToolWrapper`
- `crates/ph0b0s-llm-adk/src/provider/mod.rs` — `build_from_env` / `build_from_config`
- `crates/ph0b0s-llm-adk/src/provider/anthropic.rs`
- `crates/ph0b0s-llm-adk/src/provider/openai.rs`
- `crates/ph0b0s-llm-adk/src/provider/gemini.rs`
- `crates/ph0b0s-llm-adk/src/provider/ollama.rs`
- `crates/ph0b0s-llm-adk/src/provider/mock.rs` (moved from `ph0b0s-cli/src/provider.rs`)
- `crates/ph0b0s-llm-adk/tests/ollama_live.rs`
- `crates/ph0b0s-llm-adk/tests/fixtures/fake_mcp.py`

To be modified:

- `crates/ph0b0s-llm-adk/src/lib.rs` — re-exports
- `crates/ph0b0s-llm-adk/src/agent.rs` — tool-call loop in `chat()` and `session.send()` via shared `run_loop`
- `crates/ph0b0s-llm-adk/src/tools.rs` — `mount_mcp` delegates to `mcp::mount`
- `crates/ph0b0s-llm-adk/src/error.rs` — add `BuildError`
- `crates/ph0b0s-cli/src/provider.rs` — shrink to ~30 LOC dispatcher
- `crates/ph0b0s-cli/src/config.rs` — add `provider_registry()` helper that maps figment-loaded values onto `ProviderRegistry`
- `.github/workflows/ci.yml` — add `live-ollama` job

## Verification

```bash
# Workspace tests — should be ~210 passing
cargo test --workspace --all-features

# Adapter-only check (faster iteration during impl)
cargo test -p ph0b0s-llm-adk

# MCP fixture test alone
cargo test -p ph0b0s-llm-adk --test '*' mcp_

# Quality gates (all unchanged from slice (e))
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p xtask -- check-vendor                # must still print "vendor-coupling: OK"
cargo deny check

# Coverage — patch must be 100%; project floor 90%
cargo llvm-cov --workspace --all-features \
  --ignore-filename-regex 'xtask|/tests/|fixtures/|ph0b0s-test-support|ph0b0s-cli/src/(main|workspace)\.rs|ollama_live\.rs' \
  --summary-only

# Manual smoke (with a real key — outside CI)
ANTHROPIC_API_KEY=sk-ant-... cargo run -p ph0b0s-cli -- \
    scan ./fixtures/sample-rust-repo --output /tmp/report.sarif

# Live Ollama (manual or in CI's live-ollama job)
cargo test -p ph0b0s-llm-adk --test ollama_live -- --include-ignored
```

**Done when:**

- All quality gates above are green.
- The end-to-end mock-provider integration test from slice (e) (`crates/ph0b0s-cli/tests/end_to_end.rs`) still passes unchanged — proves no regression on the existing data path.
- A manual scan with `ANTHROPIC_API_KEY` produces a real model-generated finding from the LLM-toy detector (recorded in the PR description as a sanity-check transcript).
- The live-ollama CI job is green on the PR.

## What comes after this slice

When this slice is done, the candidate next slices (per the slice (e) plan) are:

- **Real secrets / SCA detector** replacing the `llm-toy` and/or `cargo-audit` smoke wrappers. Now possible because real LLM providers work.
- **Per-role agent assignment** — let detectors request a specific role (`reasoner`, `triager`, …) and have the CLI construct multiple agents from `[agents.<role>]` blocks. Layered on without seam changes.
- **Bounded-parallel detector execution** — honour the `max_parallel` config knob.
- **Real SAST (CPG + LLM data-flow reasoning)** — Shannon's flagship feature. Multi-week.
- **Tool-call-loop polish** — parallel dispatch when the model emits multiple `FunctionCall`s in one turn; non-stdio MCP transports; tool-name collision prefixing.

Each gets its own brainstorm → spec → plan cycle and plugs into the existing skeleton without modifying it — by depending on `ph0b0s-core` and registering new detectors, or by extending the adapter's public surface.
