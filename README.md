# ph0b0s

[![CI](https://github.com/hocinehacherouf/ph0b0s/actions/workflows/ci.yml/badge.svg)](https://github.com/hocinehacherouf/ph0b0s/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hocinehacherouf/ph0b0s/branch/main/graph/badge.svg)](https://codecov.io/gh/hocinehacherouf/ph0b0s)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**ph0b0s** is a vendor-neutral, agentic application-security (AppSec) scanner
written in Rust. It is inspired by [Shannon Pro][shannon] and aims to combine
several detection modalities (SAST, SCA, secrets, DAST, business-logic
testing) under one platform — without locking detection logic to any specific
LLM vendor.

This repository currently contains **slice (e): the platform skeleton** —
the seam, persistence, reporting, two smoke-test detectors (one LLM-driven
and one subprocess-driven), and a CLI. Real detection categories ship in
follow-on slices that plug into this skeleton without modifying it.

[shannon]: https://github.com/KeygraphHQ/shannon

## Why vendor-neutral

Detection-pack code lives behind a small in-house seam (`LlmAgent`,
`ToolHost`, `Detector`, `FindingStore`, `Reporter` traits in
[`ph0b0s-core`](crates/ph0b0s-core)). The only crate that imports any
LLM-vendor SDK is the adapter
[`ph0b0s-llm-adk`](crates/ph0b0s-llm-adk). A custom CI fitness function
([`xtask check-vendor`](xtask/src/main.rs)) regex-scans every workspace
crate's `src/` and fails the build if any non-allow-listed crate imports
`adk-*` or a vendor SDK directly. Replacing the underlying adapter is a
local change.

## Status

| Component | State |
|---|---|
| `ph0b0s-core` (the seam) | ✅ stable |
| `ph0b0s-test-support` (mocks + fixtures) | ✅ stable |
| `ph0b0s-storage` (SQLite `FindingStore`) | ✅ stable |
| `ph0b0s-report` (SARIF 2.1.0 + Markdown + JSON) | ✅ stable |
| `ph0b0s-detect-cargo-audit` (subprocess detector) | ✅ smoke detector |
| `ph0b0s-detect-llm-toy` (LLM-driven detector) | ✅ smoke detector |
| `ph0b0s-llm-adk` (adk-rust adapter) | ✅ Anthropic + OpenAI + Gemini + Ollama wired; tool-call loop active; stdio MCP supported |
| `ph0b0s-cli` (`ph0b0s` binary) | ✅ TOML-driven provider selection + env-var fallback |

### Current limitations

These are intentional and tracked in the seam doc-comments. They will be
lifted in follow-on slices.

- **Sequential detector execution** (the `max_parallel` config knob exists
  but is not honoured yet).
- **MCP transports: stdio only.** SSE / StreamableHTTP entries are
  recorded with a warning but no live connection.
- **Single global agent per scan.** Per-role agent assignment
  (`reasoner`, `triager`, etc.) is deferred until a detector needs it.
- **Sequential tool dispatch.** When the model emits multiple
  `FunctionCall`s in one turn, we dispatch them one by one.
- **No streaming** in the LLM seam (additive change later).
- **Linux/macOS only.** The end-to-end integration test is gated
  `#[cfg(unix)]`.

### Provider configuration

`ph0b0s` picks a provider in this order (highest precedence first):

1. `PH0B0S_PROVIDER` env override (e.g. `PH0B0S_PROVIDER=mock` for hermetic runs).
2. Explicit `[agents.default]` in `ph0b0s.toml`.
3. Env-key auto-detection: `ANTHROPIC_API_KEY` → Anthropic,
   `OPENAI_API_KEY` → OpenAI, `GOOGLE_API_KEY` → Gemini,
   `OLLAMA_HOST` → Ollama.

Set the corresponding API key (never in TOML) and run:

```bash
ANTHROPIC_API_KEY=... cargo run -p ph0b0s-cli -- scan ./some-repo
```

Override per-provider defaults in `ph0b0s.toml`:

```toml
[providers.anthropic]
default_model = "claude-opus-4-7"

[providers.openai]
base_url = "https://openrouter.ai/api/v1"
default_model = "openai/gpt-5"

[agents.default]
provider = "anthropic"
```

## Quick start

### Prerequisites

- Rust **1.85** or newer (workspace MSRV; CI builds on stable)
- On Linux: `libdbus-1-dev` and `pkg-config` (transitive system deps from
  the adk-rust dependency graph)

```bash
# Linux
sudo apt-get install -y libdbus-1-dev pkg-config

# macOS
# nothing extra — Keychain is built in
```

### Build and test

```bash
git clone https://github.com/hocinehacherouf/ph0b0s.git
cd ph0b0s
cargo test --workspace --all-features
```

### Run a hermetic scan

The mock provider lets you exercise the full pipeline without any API
keys or network access:

```bash
PH0B0S_PROVIDER=mock cargo run -p ph0b0s-cli -- \
    scan ./fixtures/sample-rust-repo --output /tmp/report.sarif
jq '.runs[0].results[] | .ruleId' /tmp/report.sarif
```

Output:
- `ph0b0s.cargo-audit.RUSTSEC-...` from the subprocess detector (real
  `cargo audit` if installed; otherwise a `MissingTool` error)
- `ph0b0s.llm-toy.…` if you also pass `PH0B0S_MOCK_RESPONSES=<file.json>`
  with canned issues

See [`crates/ph0b0s-cli/tests/canned/`](crates/ph0b0s-cli/tests/canned)
for a working canned-responses example.

## CLI surface

```text
ph0b0s scan <path>            --output, --markdown, --json, --strict, --detector ID
ph0b0s detectors list         --enabled-only, --json
ph0b0s report show [run_id]   --format sarif|md|json   (defaults to latest run)
ph0b0s triage suppress <fp>   --reason <text>
ph0b0s config check
ph0b0s mcp list               --json
```

Run `ph0b0s --help` after a `cargo build` for full flag documentation.

## Configuration

Config is layered (lowest → highest precedence):

1. compiled-in defaults
2. `~/.config/ph0b0s/config.toml`
3. `./ph0b0s.toml` (per-project)
4. environment variables prefixed `PH0B0S__` (figment `__` split)

**API keys never appear in TOML.** They're read at runtime from
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc. `ph0b0s config check` rejects
any TOML containing an `api_key` field.

Sample `ph0b0s.toml`:

```toml
[scan]
max_parallel       = 4
detector_timeout_s = 300
strict             = false

[providers.anthropic]
default_model = "claude-sonnet-4-6"

[detectors.cargo-audit]
enabled  = true
no_fetch = true

[detectors.llm-toy]
enabled    = true
extensions = [".rs", ".py", ".js", ".ts"]
max_files  = 10

[[mcp_servers]]
name      = "filesystem"
transport = "stdio"
command   = ["uvx", "mcp-server-filesystem", "--root", "."]

[output]
sarif_path    = "report.sarif"
markdown_path = "report.md"
```

## Architecture overview

```
ph0b0s-core ──► ph0b0s-storage ─────┐
   ▲   ▲                            │
   │   └──► ph0b0s-report ──────────┤
   │                                ▼
   ├── ph0b0s-detect-llm-toy ─────► ph0b0s-cli
   ├── ph0b0s-detect-cargo-audit ──► ph0b0s-cli
   │                                ▲
   └── ph0b0s-llm-adk ──────────────┘
                          (only ph0b0s-cli depends on the adapter)

ph0b0s-test-support  (dev-deps only, used by every other crate's tests)
xtask                (CI helpers; excluded from default-members)
```

| Crate | Responsibility |
|---|---|
| [`ph0b0s-core`](crates/ph0b0s-core) | The seam: domain types + traits. Zero deps on any vendor / adapter. |
| [`ph0b0s-test-support`](crates/ph0b0s-test-support) | `MockLlmAgent`, `MockToolHost`, `CannedTool`, deterministic fixture builders. |
| [`ph0b0s-storage`](crates/ph0b0s-storage) | SQLite `FindingStore` via `sqlx`. Embedded migrations. |
| [`ph0b0s-report`](crates/ph0b0s-report) | SARIF 2.1.0 (round-trips through `serde-sarif`), Markdown, JSON. |
| [`ph0b0s-detect-cargo-audit`](crates/ph0b0s-detect-cargo-audit) | Subprocess wrapper around `cargo audit --json`. |
| [`ph0b0s-detect-llm-toy`](crates/ph0b0s-detect-llm-toy) | LLM-driven smoke detector via the seam. |
| [`ph0b0s-llm-adk`](crates/ph0b0s-llm-adk) | Adapter binding `adk-rust = "=0.6.0"` to the seam. **Only crate permitted to import `adk-*`.** |
| [`ph0b0s-cli`](crates/ph0b0s-cli) | `ph0b0s` binary (clap + figment). Wires adapter + registry + storage + reporters. |
| [`xtask`](xtask) | CI helpers (`check-vendor` fitness function). |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, build/test
workflow, coding conventions, and the project's hard constraints (vendor
neutrality, API keys never in TOML, etc.).

## License

Apache-2.0. See [LICENSE](LICENSE).
