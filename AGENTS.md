# latticeqcd-rs contributor instructions

Read the shared tensor4all rules from the online
`https://github.com/tensor4all/tensor4all-agent-rules` repository; if
unavailable, use the sibling checkout `../tensor4all-agent-rules`.
Load only the relevant common, Rust, performance, numerical, and docs/test
rules. Then read this repository's `README.md` and `REPOSITORY_RULES.md` before
editing.

This repository is Rust 2021 with Markdown documentation; write source code and
documentation in English. It is implemented through Phase 8 and follows the
exact tenferro `origin/main` revision recorded
in the current migration design/worklog. Keep public APIs small, preserve
numerical tolerances, and keep runtime, extension-module, backend-session, and
AD-rule ownership explicit.

Use CodeGraph first: run `codegraph init` and refresh its index before source
exploration, then trace changed symbols and callers. Do not commit `.codegraph/`.

When a referenced upstream implementation appears buggy, record the candidate
as an issue in this Rust repository with the pinned upstream revision, source
location, reproducer or direct evidence, and the Rust-side decision. Do not file
or patch upstream without the user's separate explicit approval.

Before completion, run the focused tests and the local gate:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo doc --workspace --all-features --no-deps
```

Also run the traced example/smoke test, `git diff --check`, and the migration's
stale-symbol and exact-pin checks. Commit and push only after the required
independent review gate passes.
