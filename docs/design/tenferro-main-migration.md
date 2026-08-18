# Migration to tenferro `origin/main`

Status: implemented
Date: 2026-08-18
Issue: [latticeqcd-rs #17](https://github.com/shinaoka/latticeqcd-rs/issues/17)

## Goal

Move every tenferro workspace dependency from
`51bc0a7bef274e20d08fc054856cb4d74c284cbe` to the current fetched
`origin/main`, `c942129974b544225ed963414d7be1300980f901`, and migrate gaugefields to
the current extension runtime, semantic AD, execution, and tensor ownership
contracts without compatibility shims or numerical behavior changes.

The same PR restores the repository policy entry point. A concise root
`AGENTS.md` directs contributors to the shared tensor4all rules and this
repository's `REPOSITORY_RULES.md`; the latter keeps its existing downstream
rules and adopts only current shared/tenferro rules that apply to a gauge-field
consumer of tenferro.

## Sources and current breakage

The design was checked against:

- `../tenferro-rs` at `c942129974b544225ed963414d7be1300980f901`;
- `../tensor4all-agent-rules/rules/index.md`, common repository, performance,
  docs/test, and Rust numerical/performance rules;
- tenferro `crates/tenferro-internal-ops/src/ext_op.rs` and
  `shape_constraint.rs`;
- tenferro `crates/tenferro-runtime/src/runtime/{extension.rs,extension_provider.rs,snapshot.rs}`;
- tenferro `crates/tenferro-ad/src/{context.rs,semantic_extension.rs}`;
- current extension implementations in tenferro `ext/tropical` and
  `ext/sparse`;
- this repository's `crates/gaugefields/src/{extension.rs,ad.rs}` and Phase
  6--8 design/tests.

A clean scratch `cargo check --workspace --all-features` with only the new pin
fails because the old `HostReferenceRuntime`/`ExtensionExecutor` registration
surface and primitive extension AD rule set were removed, and
`ExtensionOp::infer_output_meta` now takes `ExtensionShapeContext`.

## Design

### 1. Atomic dependency update

Update the five tenferro revisions in the workspace manifest together and let
Cargo regenerate `Cargo.lock`. Verify both the manifest and lock resolve every
tenferro crate to the fetched `c9421299` commit. Do not float on a branch name
or mix revisions.

### 2. Extension payload and shape contract

Retain the three stable payload families and their numerical kernels:

- `gaugefields.wilson_action.v1`;
- `gaugefields.wilson_action_jvp.v1`;
- `gaugefields.wilson_force.v1`.

Migrate each `ExtensionOp` as follows:

| Old contract | Current contract |
|---|---|
| `infer_output_meta(dtypes, shapes)` | `infer_output_meta(&mut ExtensionShapeContext)` |
| known-constant comparisons only | read dtypes/shapes through the context and record same-shape/axis equality constraints |
| `host_reference()` callback | no payload execution callback; execution belongs to an installed extension module |
| implicit/default semantic contract | declare pure effects and fresh outputs explicitly |

The inference callback still validates arity, C64/F64 dtypes, rank, color axes,
seed scalar shape, and tangent correspondence. It additionally uses
`require_same_shape`/`require_axes_equal` (or equivalent context methods) so
symbolic lattice and tangent equality is represented in the semantic program,
removing the obsolete Phase 6 limitation. Concrete execution repeats all
placement, dtype, rank, and exact-shape checks before kernels run.

### 3. Runtime-owned extension module

Replace the removed host-reference callback/registry surface with three
explicit, reusable `ExtensionModule` values returned together by
`runtime_modules`, one for each Wilson family.

Follow the current tenferro reference-module pattern:

1. each `ExtensionModule` owns a validated module ID for one Wilson family;
2. `configure` registers that family's engine and matching planning config;
3. each `ExtensionEngine::prepare` validates family/payload support and returns
   a `PreparedOperationPlan`;
4. each prepared operation stores the operation payload, specialization, and
   backend binding;
5. `PreparedOperationExecutor::execute` downcasts the erased execution context,
   materializes borrowed inputs to compact tensors in the existing backend
   session, and calls the existing shared gauge kernel;
6. unsupported family/payload and missing module remain typed runtime errors;
   there is no eager/CPU fallback or freshly constructed backend.

Expose the narrowest public constructor needed by applications:
`runtime_modules::<B>(engine_id) -> Result<Vec<Arc<dyn ExtensionModule>>, ...>`.
Do not expose engine, prepared-operation, or planning structs. Tests and
examples build one `Runtime`, register
`tenferro_cpu::runtime_engine_registration(&backend)`, install every returned
module, compile with `GraphCompiler`, and execute with `Runtime::run_compiled`.

`GraphCompiler` remains the supported compiler. `GraphExecutor` has been
removed from the current public execution path and must be replaced at all call
sites by application-owned `Runtime`; runtime/module/cache ownership remains
explicit and reusable.

### 4. Semantic AD migration

Replace primitive `ExtensionRuleSet`, `ExtensionLinearizeRule`, and
`ExtensionLinearTransposeRule` with `SemanticExtensionRuleSet`,
`SemanticLinearizeRule`, and `SemanticLinearTransposeRule`.

The mathematical decomposition remains the same, but both semantic rules are
registered under the **primal action family** because current
`semantic_vjp` dispatches `linearize_operation` and then
`linear_transpose_operation` on the same primal operation:

```text
Wilson action --action-family linearize--> Wilson action JVP
Wilson action --action-family linear transpose using primal inputs--> Wilson force
```

The action-family linearize rule:

- downcasts the action payload;
- validates four primal inputs, one active output, and four tangent slots;
- returns an absent tangent when the output is inactive or all inputs are
  inactive;
- appends a JVP extension operation with four primal inputs plus only active
  tangent values through `SemanticProgramBuilder::add_extension`;
- returns one `AdValue` and no custom linearization residuals.

The action-family linear-transpose rule:

- declares `ResidualSpec::input(0).with_input(1).with_input(2).with_input(3)`
  because it reads the four primal action links as tensor operands; these are
  available through `request.primal_inputs()` and are distinct from the empty
  custom `request.residuals()` returned by linearization;
- downcasts the **action** payload and validates four action inputs, one
  cotangent output, and the four-entry active-link mask supplied by semantic
  VJP;
- returns four absent cotangents when no scalar cotangent is present;
- appends one force operation using the four primal action inputs and scalar
  cotangent;
- maps the force outputs to active link input slots and returns `Absent` for
  inactive links.

Do not port the old `WilsonActionJvpOp`-family transpose registration: current
VJP does not dispatch transpose on the emitted JVP operation. The JVP and force
families intentionally have no semantic AD rules, so differentiating a JVP or
the generated force is a typed missing-rule/unsupported higher-order boundary.
Tests assert the typed family/role rather than obsolete error wording.

`ad_rules()` returns `SemanticExtensionRuleSet` containing action-family
linearize and action-family linear-transpose rules. Callers install it with
`AdContext::builder().with_semantic_extension_rules(...)`. Current
`AdContext::jvp_program`/`vjp_program` and traced convenience surfaces perform
the semantic transforms. No direct primal-VJP shortcut is added.

### 5. Current tensor ownership and sessions

Audit all affected code for current tenferro ownership rules rather than adding
clones around compiler errors. `TypedTensor`/`Tensor` ownership moves must be
explicit; borrowed host access uses guarded/read APIs; tensor-sized backend work
runs inside the application-owned backend/runtime session. Preserve the
existing transactionality of four-direction evolution updates and do not add
implicit host/device transfer or dense materialization.

### 6. Policy restoration

Create a short root `AGENTS.md` that:

- reads the online shared tensor4all rules, with sibling checkout fallback;
- loads only relevant common/Rust/performance/numerical/docs files;
- then requires `README.md` and local `REPOSITORY_RULES.md`;
- states source/docs language, repository stage, CodeGraph-first exploration,
  and the local CI commands.

Do not vendor the shared rule corpus. Retain the existing
`REPOSITORY_RULES.md` and add only downstream-relevant current rules:

- source-of-truth and base-branch synchronization;
- exact atomic tenferro `origin/main` revision updates;
- explicit runtime/session/module ownership and no silent extension fallback;
- current semantic extension effects/aliases and semantic AD rule ownership,
  replacing obsolete `host reference runtime` and primitive role-split terms;
- current tensor borrow/ownership discipline (no clone-based workaround), and
  `Runtime`/module language instead of the removed graph executor;
- focused local validation plus CI ownership, and numerical residual/oracle
  requirements.

Do not import tenferro-only GPU provider internals, FFI rules, publication
protocol, coverage thresholds, or repository review-bot routing that this
repository does not implement.

### 7. Documentation and worklog

Update `docs/design/phase-6-8.md` wherever it names removed APIs
(`GraphExecutor`, `HostReferenceRuntime`, primitive role-split AD) so the
durable architecture describes the current module/runtime and semantic AD
contracts. Update README/example setup if it contains the old execution path.
Add a concise `docs/worklogs/2026-08-18-tenferro-main-migration.md` recording
source revisions, decisions, exact commands, residuals, and remaining
hardware limitations. No standalone AI transcript or report is added.

## Expected files

At minimum inspect and likely change:

- `Cargo.toml`, `Cargo.lock`;
- `AGENTS.md`, `REPOSITORY_RULES.md`;
- `crates/gaugefields/src/extension.rs` and its module-local tests;
- `crates/gaugefields/src/ad.rs` and its module-local tests;
- integration tests using `GraphExecutor`, runtime registration, AD context,
  or tensor cloning under `crates/gaugefields/tests/`;
- `crates/gaugefields/examples/traced_wilson_action.rs`;
- `README.md`, `docs/design/phase-6-8.md`, and the migration worklog.

Avoid unrelated refactors and keep existing family IDs, formulas, layouts,
public domain types, and tolerances unchanged.

## Verification

Run focused checks during implementation, then the complete local gate:

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

Also verify:

```bash
git grep -n -E '51bc0a7bef274e20d08fc054856cb4d74c284cbe|GraphExecutor|HostReference(Runtime)?|ExtensionExecutor|ExtensionRuleSet|ExtensionRegistryError|with_extension_rules|ShapeGuardContext|PrimitiveRuleBuilder|register_runtime'
git diff --check
git merge-base --is-ancestor origin/main HEAD
git status --short
```

The grep may find historical prose only when it explicitly labels the old API;
no compiled code or current setup instructions may retain removed names. Update
string-based failure assertions to the current typed runtime/semantic AD
families and roles rather than requiring old phrases such as `missing runtime`
or `unsupported`. Replace computegraph `OperationRole::Linearized` structural
assertions with semantic-program assertions for the JVP family payload, arity,
and ordered active directions. Run the traced action example or equivalent
integration smoke test through an installed runtime module. Numerical AD tests
must continue to report and meet the existing finite-difference/Julia residual
tolerances; tolerances are not relaxed.

Before PR creation, fetch `origin`, rebase or merge current `origin/main` if
needed, rerun the affected/full gates, inspect the full committed diff against
`REPOSITORY_RULES.md`, and verify README accuracy.

## Acceptance criteria

- every tenferro dependency and lock entry resolves to
  `c942129974b544225ed963414d7be1300980f901`;
- default and all-feature builds compile without deprecated compatibility code;
- all three extension families infer symbolic constraints and execute only
  through an explicitly installed current `ExtensionModule`;
- missing module, malformed metadata, wrong placement, and unsupported AD paths
  remain typed errors without panic or fallback;
- first-order Wilson JVP/VJP retains existing finite-difference and Julia
  agreement, including inactive and non-unit cotangent cases;
- all runtime owners/caches/sessions are caller-owned and reused;
- existing eager/evolution behavior and transactionality remain unchanged;
- restored policy files are concise, relevant, and consistent with shared and
  tenferro rules;
- formatting, CI-parity clippy/tests, doctests, docs, smoke checks, and final
  diff checks pass;
- the committed branch is based on current `origin/main` and a PR links issue
  #17 and the worklog.

## Risks

- Extension runtime registration now requires several identity and capability
  types; hide this plumbing behind one module constructor rather than exposing
  it as public API.
- Semantic AD residual/activity ordering is easy to mis-map. Preserve ordered
  four-direction tests, inactive slots, negative/non-unit cotangents, and
  higher-order rejection.
- Current tensor values may no longer be freely cloneable. Fix ownership at the
  owning call boundary; do not duplicate storage to silence borrow errors.
- The CPU reference module materializes compact inputs by design. It must remain
  an explicit host execution boundary with no hidden GPU transfer.
- `origin/main` can advance during the work. The PR targets the fetched commit
  named above; refetch before final verification and update atomically again if
  the requested latest revision has advanced.
