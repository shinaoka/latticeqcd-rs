# tenferro `origin/main` migration

- **Date:** 2026-08-18
- **Branch:** `chore/update-tenferro-main`
- **Issue:** latticeqcd-rs #17
- **Pre-design review gate:** `reviewer-flash: Correct-to-merge`
- **tenferro source:** `https://github.com/tensor4all/tenferro-rs.git`
- **Pinned revision:** `c942129974b544225ed963414d7be1300980f901`

## Decisions

- Replaced the removed graph execution and extension registration surface with
  an application-owned `Runtime`, `GraphCompiler`, CPU engine registration, and
  explicit installation of every module from
  `gaugefields::runtime_modules::<CpuBackend>(engine_id)`.
- Kept the three Wilson family IDs and numerical formulas unchanged. Extension
  payloads declare pure effects and fresh outputs, record symbolic equality
  constraints, prepare through the runtime-owned backend session, and return
  typed placement/payload errors without fallback. The implementation follows
  the current `ext/tropical` and `ext/sparse` reference-module patterns.
- Migrated Wilson AD to `SemanticExtensionRuleSet`: action-family linearize
  emits the variable-arity JVP and action-family linear transpose emits force.
  JVP and force have no higher-order semantic rules.
- Removed direct `computegraph`/`tidu` dependencies; computegraph remains only
  as a transitive tenferro dependency. Replaced tensor clones at owning test
  boundaries with explicit `TypedTensor::duplicate()`.
- Restored concise root `AGENTS.md`, kept repository rules downstream-focused,
  and updated current README/design documentation. Older plans/worklogs remain
  historical.

## Verification

Initial RED evidence was `cargo test --workspace --all-features`, which exposed
three current typed-storage/API compile failures. After the focused migrations:

- `cargo check --workspace --all-features` — PASS (parent verification)
- `cargo test --workspace --all-features` — PASS: 76 passed, 1 ignored
- `cargo test --workspace` — PASS: 66 passed, 1 ignored
- `cargo fmt --all` — PASS
- `cargo fmt --all -- --check` — PASS
- `cargo check --workspace` — PASS
- `cargo test --workspace` — PASS: 66 passed, 1 ignored
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — PASS
- `cargo test --workspace --all-features` — PASS: 76 passed, 1 ignored
- `cargo test --doc --workspace --all-features` — PASS: 1 doctest
- `cargo doc --workspace --all-features --no-deps` — PASS
- `cargo run -p gaugefields --example traced_wilson_action --all-features` — PASS:
  `direct=-576 traced=-576 residual=0`
- `git diff --check` — PASS
- stale-symbol `git grep` — PASS for compiled/current setup; remaining matches
  are explicitly historical plans/worklogs plus the current semantic prefix
  `SemanticExtension*`.

The traced action integration test and `traced_wilson_action` example use the
installed runtime modules and preserve the `1e-13` action residual. Existing
finite-difference, Julia fixture, AD, evolution, layout, and transactionality
tolerances were not changed.

## Review follow-up

The post-implementation `reviewer-flash` on the migration implementation diff
was `Correct-to-merge` with exactly two `Minor` findings. This follow-up records
their fixes; it does not represent a review verdict on this later diff.

- `docs/design/ad-convention.md` now states that `ExtensionShapeContext`
  records cross-input extension shape-equality constraints.
- `crates/gaugefields/tests/traced_action.rs` now checks the concrete
  `RuntimeStateSource` → `PrepareError::NoInputIngress` source path, including
  the input index and device placement fields.

## Risks

- The module is a host-reference implementation: direct host kernels reject
  backend storage, while runtime execution materializes through the selected
  backend session. No GPU implementation or implicit transfer is provided.
- The requested exact revision is reproducible but `origin/main` may advance;
  a future update must move all five direct pins and the lockfile atomically.
- The full workspace retains computegraph transitively because current tenferro
  runtime/operation crates require it; it is not an application-owned API.
