# Contributing to ph0b0s

Thanks for taking a look. ph0b0s is in active early development. This
document covers everything you need to set up locally, run the same
checks CI runs, and submit a clean PR.

## Hard constraints (non-negotiable)

These are enforced at the source, dependency-graph, and supply-chain
levels by CI. Read them before opening a PR — they are easy to violate
unintentionally.

1. **Vendor neutrality.** No detection-pack crate may `use` an LLM-vendor
   crate (`anthropic`, `openai`, `genai`, etc.) or any `adk_*` crate.
   Only [`crates/ph0b0s-llm-adk`](crates/ph0b0s-llm-adk) and
   [`crates/ph0b0s-cli`](crates/ph0b0s-cli) are allow-listed. The
   `xtask check-vendor` job regex-scans every crate's `src/` and fails
   the build on violations.
2. **API keys never appear in TOML.** They're read at runtime from
   canonical env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
   `ph0b0s config check` rejects any TOML containing an `api_key`
   field.
3. **SARIF 2.1.0 emit must round-trip through `serde-sarif`.** A
   snapshot test enforces this; do not bypass.
4. **`adk-rust` is pinned to an exact version (`=0.6.0`)** in the
   adapter only. Bumping it is a focused PR with adapter updates +
   smoke-test verification. Detection packs never depend on it.

## Prerequisites

- **Rust 1.85** or newer. The workspace MSRV is `1.85` (declared in the
  root `Cargo.toml`'s `workspace.package`); CI builds on `stable` to
  pull in transitive deps that need newer rustc (AWS SDK, etc.).
- **Linux** users: install `libdbus-1-dev` and `pkg-config`. Transitive
  via the adk-rust dep graph (keyring backends).
- **macOS** users: nothing extra — Keychain is built in.
- **Optional but useful**:
  - [`cargo-llvm-cov`](https://crates.io/crates/cargo-llvm-cov) for
    local coverage runs.
  - [`cargo-deny`](https://crates.io/crates/cargo-deny) for local
    license / advisory checks.
  - [`cargo-insta`](https://crates.io/crates/cargo-insta) for
    accepting changes to snapshot tests in `ph0b0s-report`.

```bash
# Linux system deps
sudo apt-get install -y libdbus-1-dev pkg-config

# Optional cargo helpers
cargo install cargo-llvm-cov cargo-deny cargo-insta --locked
rustup component add llvm-tools-preview   # required by cargo-llvm-cov
```

## Build and test

The workspace builds and tests with vanilla `cargo`. There is no global
build script.

```bash
git clone https://github.com/hocinehacherouf/ph0b0s.git
cd ph0b0s

# Build everything
cargo build --workspace --all-features

# Run the full test suite (~145 unit + integration tests)
cargo test --workspace --all-features
```

### Per-crate

```bash
# Just the seam
cargo test -p ph0b0s-core

# Just the SQLite store (includes 2 on-disk integration tests in
# crates/ph0b0s-storage/tests/on_disk.rs)
cargo test -p ph0b0s-storage

# Reporter snapshot tests
cargo test -p ph0b0s-report
# If a snapshot legitimately needs to change:
cargo insta review

# End-to-end CLI test (gated #[cfg(unix)])
cargo test -p ph0b0s-cli --test end_to_end
```

### Lints, formatting, and the vendor-coupling guard

These are exactly what CI runs — keep them green locally before you push.

```bash
# Format check (CI uses --check; locally you can just `cargo fmt --all`)
cargo fmt --all -- --check

# Clippy with warnings-as-errors (matches CI)
cargo clippy --workspace --all-targets -- -D warnings

# The vendor-coupling fitness function. Fails if any non-allow-listed
# crate imports a vendor SDK or adk-* crate.
cargo run -p xtask -- check-vendor
```

### Supply-chain check

```bash
cargo deny check
```

This runs the same advisories / bans / licenses / sources checks CI
runs. The configuration lives in [`deny.toml`](deny.toml). If you add a
dependency under a license not yet in the allow list, this is the
fastest place to discover it.

### Coverage

CI uploads to Codecov via `cargo-llvm-cov`. To reproduce locally:

```bash
cargo llvm-cov --workspace --all-features \
  --ignore-filename-regex 'xtask|/tests/|fixtures/|ph0b0s-test-support|ph0b0s-cli/src/(main|workspace)\.rs' \
  --summary-only
```

The `--ignore-filename-regex` mirrors the `ignore:` block in
[`codecov.yml`](codecov.yml) so the local number matches what Codecov
reports.

### Hermetic end-to-end demo

The mock provider lets the full pipeline run without any API keys or
network access:

```bash
PH0B0S_PROVIDER=mock cargo run -p ph0b0s-cli -- \
    scan ./fixtures/sample-rust-repo --output /tmp/report.sarif

jq '.runs[0].tool.driver.name' /tmp/report.sarif        # "ph0b0s"
jq '[.runs[0].results[].ruleId] | unique' /tmp/report.sarif
```

Re-emit a saved report from the SQLite store:

```bash
cargo run -p ph0b0s-cli -- report show --format sarif
```

## Project layout

```
crates/
  ph0b0s-core/           # the seam (domain types + traits, no vendor deps)
  ph0b0s-test-support/   # mocks + deterministic fixtures
  ph0b0s-storage/        # SQLite FindingStore impl
  ph0b0s-report/         # SARIF / Markdown / JSON reporters
  ph0b0s-detect-*/       # detection packs (cargo-audit, llm-toy)
  ph0b0s-llm-adk/        # adk-rust adapter — only crate that touches adk-*
  ph0b0s-cli/            # `ph0b0s` binary (clap + figment + orchestrator)
fixtures/
  sample-rust-repo/      # used by the e2e integration test
xtask/                   # CI helpers (vendor-coupling fitness function)
.github/workflows/ci.yml # CI definition
codecov.yml              # coverage gate config
deny.toml                # cargo-deny config
```

## Adding a new detector

The point of the seam is that new detectors plug in without modifying
existing crates. The full pattern is:

1. **Create a new crate** under `crates/ph0b0s-detect-<your-id>/`
   depending only on `ph0b0s-core` (plus your own deps; **never**
   `adk-rust` or any vendor SDK directly).
2. **Implement `Detector`** from `ph0b0s_core::detector`. Set
   `metadata().kind` to one of `LlmDriven`, `Subprocess`, `Native`, or
   `Hybrid`. Provide a JSON-Schema `config_schema()`.
3. **Honour `cancel: CancellationToken` and `ctx.deadline`** at every
   per-unit-of-work boundary. Per-file errors should `tracing::warn!`
   and continue; only `Cancelled` / `Timeout` should abort the whole
   run.
4. **Use the `Fingerprint::compute(rule_id, &location, evidence)`
   helper** so cross-run dedup works.
5. **Register the detector** in
   [`crates/ph0b0s-cli/src/registry.rs`](crates/ph0b0s-cli/src/registry.rs).
6. **Write tests** that use `MockLlmAgent` / `MockToolHost` from
   `ph0b0s-test-support`. Detection-pack tests must be hermetic — no
   network, no real provider, no installed binaries unless wrapped in
   a generated fake (see
   [`crates/ph0b0s-cli/tests/end_to_end.rs`](crates/ph0b0s-cli/tests/end_to_end.rs)
   for the fake-cargo pattern).
7. **Run the local checks above** before opening a PR. Confirm the
   `xtask check-vendor` step still says `vendor-coupling: OK`.

## Pull-request checklist

Before opening a PR:

- [ ] `cargo fmt --all -- --check` is clean.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean.
- [ ] `cargo test --workspace --all-features` is green.
- [ ] `cargo run -p xtask -- check-vendor` says `vendor-coupling: OK`.
- [ ] `cargo deny check` is green (or you've updated `deny.toml` with a
      reasoned justification).
- [ ] Every new line is covered by a test (`patch.target: 100%` in
      [`codecov.yml`](codecov.yml)). Cover error paths, not just happy
      paths.
- [ ] No API keys, secrets, or vendor crates in any TOML or
      detection-pack `use` statement.
- [ ] PR description briefly explains *why* the change is needed, not
      just *what* changed.
- [ ] Commit messages follow the existing style — short
      `type: imperative summary` (e.g. `feat:`, `fix:`, `ci:`,
      `docs:`, `refactor:`, `test:`).

CI will catch all of the above, but doing it locally is much faster
than waiting on the runner.

## Reporting bugs

Open an issue with:
- The `ph0b0s --version` output.
- A minimal reproduction (`cargo run -p ph0b0s-cli -- ...` invocation).
- The relevant tail of stderr with `PH0B0S_LOG=debug` set if the
  failure isn't obvious.

## Security

Do **not** open public issues for security vulnerabilities. Email the
maintainer directly (see the `authors` field in `Cargo.toml`).
