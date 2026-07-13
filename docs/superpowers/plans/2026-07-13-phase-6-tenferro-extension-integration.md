# Phase 6 tenferro Extension Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate gauge storage to typed tenferro tensors and execute the Wilson action through three explicitly registered host-reference extension families.

**Architecture:** Domain wrappers own compact `TypedTensor<Complex64>` and `TypedTensor<f64>` values; erased `Tensor` appears only at fixture and extension ABI boundaries. Direct and traced APIs share prepared host kernels, while `wilson_action_traced` builds an extension graph with `tenferro_runtime::extension::apply` and applications explicitly register three `HostReferenceRuntime` adapters on their `GraphExecutor`.

**Tech Stack:** Rust 2021, tenferro `51bc0a7bef274e20d08fc054856cb4d74c284cbe`, `TypedTensor`, `TracedTensor`, `GraphCompiler`, `GraphExecutor<CpuBackend>`, `ExtensionOp`, `HostReference`, Julia/Gaugefields.jl fixtures.

---

## File structure

- Modify `Cargo.toml`: atomically pin every tenferro crate to `51bc0a7bef274e20d08fc054856cb4d74c284cbe`; retain computegraph `691def2d82fd4367b397d61209449f68e82050b7` and tidu `57a2e7ebe7738ca2f8b5c96f4c6ce4e467b20495`.
- Modify `gaugefields/Cargo.toml`: make runtime and CPU dependencies available to the Phase 6 library, and keep AD-only dependencies behind `autodiff`.
- Modify `gaugefields/src/field.rs`: typed `GaugeLinkTensor`, `TaGaugeField`, erased conversions, compact-host validation.
- Modify `gaugefields/src/error.rs`: structured placement, extension metadata, registration, graph, and execution errors.
- Create `gaugefields/src/kernel.rs`: prepared lattice metadata and shared slice-based action/JVP/force kernels.
- Modify `gaugefields/src/observables.rs`, `gaugefields/src/force.rs`, `gaugefields/src/index.rs`, `gaugefields/src/fixture.rs`: use typed storage and shared preparation without per-site host extraction.
- Replace `gaugefields/src/autodiff.rs` with `gaugefields/src/extension.rs`: the three payloads, metadata validation, host references, traced wrapper, and runtime registration; Phase 7 later adds AD rules.
- Modify `gaugefields/src/lib.rs`: narrow public exports.
- Create `gaugefields/tests/typed_storage.rs`, `gaugefields/tests/extension_metadata.rs`, `gaugefields/tests/traced_action.rs`, and `gaugefields/examples/traced_wilson_action.rs`.
- Create `docs/worklogs/2026-07-13-phase-6.md`; update `docs/design/dependencies.md`, `docs/design/layout.md`, and README/rustdoc capability wording.

### Task 1: Pin the current tenferro snapshot

**Files:**
- Modify: `Cargo.toml`
- Modify: `gaugefields/Cargo.toml`
- Test: `gaugefields/tests/dependency_smoke.rs`

- [ ] **Step 1: Write the failing dependency contract**

Add a source-contract assertion that reads workspace `Cargo.toml`, counts the old revision as zero, counts `51bc0a7bef274e20d08fc054856cb4d74c284cbe` on all tenferro dependency declarations, and asserts the exact computegraph/tidu revisions above.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test dependency_smoke dependency_manifest_uses_phase_6_snapshot -- --exact`

Expected: FAIL because the manifest still contains `f504ba0a8668baca89ab1d4348b9475ff85377b4`.

- [ ] **Step 3: Update dependencies atomically**

Replace every tenferro `rev` with `51bc0a7bef274e20d08fc054856cb4d74c284cbe`. Move `tenferro-runtime` and `tenferro-cpu` into normal crate dependencies because public registration and `exp_ta_update` need them; keep `tenferro-ad`, `tenferro-internal-ops`, computegraph, and tidu optional under `autodiff`.

- [ ] **Step 4: Verify green and dependency uniqueness**

Run: `cargo test -p gaugefields --test dependency_smoke dependency_manifest_uses_phase_6_snapshot -- --exact && cargo tree -p gaugefields --all-features -i tenferro-tensor`

Expected: PASS; the tree reports a single `tenferro-tensor` source at revision `51bc0a7b`.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock gaugefields/Cargo.toml gaugefields/tests/dependency_smoke.rs
git commit -m "build: pin phase 6 tenferro snapshot"
```

### Task 2: Establish typed domain storage

**Files:**
- Modify: `gaugefields/src/field.rs`
- Modify: `gaugefields/src/error.rs`
- Modify: `gaugefields/src/lib.rs`
- Create: `gaugefields/tests/typed_storage.rs`

- [ ] **Step 1: Write typed-boundary tests**

Test the exact signatures:

```rust
let _: fn(TypedTensor<Complex64>, LatticeShape4) -> Result<GaugeLinkTensor, GaugeError> =
    GaugeLinkTensor::from_typed;
let _: fn(Tensor, LatticeShape4) -> Result<GaugeLinkTensor, GaugeError> =
    GaugeLinkTensor::try_from_tensor;
let _: fn([TypedTensor<f64>; 4], LatticeShape4) -> Result<TaGaugeField, GaugeError> =
    TaGaugeField::new;
```

Also assert C64 `[3,3,Lx,Ly,Lz,Lt]`, F64 `[8,Lx,Ly,Lz,Lt]`, mismatched directions, backend-buffer rejection through `host_data()`, and compact column-major round trips. Require hand-written compact `Debug` output that omits values.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test typed_storage`

Expected: compile failure because typed constructors and `TaGaugeField` do not exist.

- [ ] **Step 3: Implement the typed wrappers**

Store `TypedTensor<Complex64>` in `GaugeLinkTensor` and `[TypedTensor<f64>; 4]` in `TaGaugeField`. Convert erased input by matching `Tensor::C64(tensor)` or `Tensor::F64(tensor)` without copying. Expose `typed()`, `typed_mut()`, `into_typed()`, and explicit erased `into_tensor()` conversions; validate rank, exact shape, checked element/byte counts, and `host_data()` before accepting host-only direct kernels.

- [ ] **Step 4: Migrate construction callers**

Change cold fields, fixture loading, force outputs, index load/store, and existing tests from `Tensor::from_vec_col_major`/typed `as_slice::<T>()` to `TypedTensor::<T>::from_vec_col_major`/`host_data()`.

- [ ] **Step 5: Verify green**

Run: `cargo test -p gaugefields --test typed_storage && cargo test -p gaugefields --no-default-features`

Expected: PASS, including all Phase 0–5 tests.

- [ ] **Step 6: Commit**

```bash
git add gaugefields/src gaugefields/tests
git commit -m "refactor: adopt typed gauge storage"
```

### Task 3: Share prepared Wilson kernels

**Files:**
- Create: `gaugefields/src/kernel.rs`
- Modify: `gaugefields/src/observables.rs`
- Modify: `gaugefields/src/force.rs`
- Modify: `gaugefields/src/index.rs`
- Test: `gaugefields/tests/observables.rs`
- Test: `gaugefields/tests/force_finite_difference.rs`

- [ ] **Step 1: Add source and numerical regressions**

Add tests proving direct action, JVP directional contraction, and force share results on cold, `random_2x2x2x2`, and `random_4x4x4x4`. Add a source contract forbidding `load_link(` inside the site loops in `observables.rs` and requiring one `PreparedGaugeField::new` boundary.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test observables shared_prepared_kernel_contract -- --exact`

Expected: FAIL because observables still extracts links per site.

- [ ] **Step 3: Implement preparation and leaf kernels**

Create crate-private `PreparedGaugeField<'a>` containing lattice, `nv`, `[&'a [Complex64];4]`, and plus/minus neighbor tables. Its constructor runs `require_su3`, checked arithmetic, shape/host validation, and borrows each direction once. Move plaquette, staple, action, directional derivative, dense gradient, and TA-force arithmetic behind this prepared record; retain fixed `Mat3` stencils with an `INVARIANT` comment explaining periodic SU(3) parity.

- [ ] **Step 4: Route direct APIs through the shared kernels**

Make `wilson_action`, `plaquette_sum`, `action_gradient`, and `gauge_force` thin validation/output wrappers. Preserve exact output capacities and avoid full-volume staple intermediates.

- [ ] **Step 5: Verify green**

Run: `cargo test -p gaugefields --test observables && cargo test -p gaugefields --test force_finite_difference && cargo test -p gaugefields --test julia_force`

Expected: PASS with reported finite residuals at existing tolerances.

- [ ] **Step 6: Commit**

```bash
git add gaugefields/src/kernel.rs gaugefields/src/observables.rs gaugefields/src/force.rs gaugefields/src/index.rs gaugefields/tests
git commit -m "refactor: share prepared Wilson kernels"
```

### Task 4: Define and validate the extension families

**Files:**
- Delete: `gaugefields/src/autodiff.rs`
- Create: `gaugefields/src/extension.rs`
- Modify: `gaugefields/src/error.rs`
- Create: `gaugefields/tests/extension_metadata.rs`

- [ ] **Step 1: Write metadata and identity tests**

Test constants `gaugefields.wilson_action.v1`, `gaugefields.wilson_action_jvp.v1`, and `gaugefields.wilson_force.v1`; `beta.to_bits()` hash/equality; JVP active directions sorted, unique, and in `0..4`; arities `4`, `4+n`, and `5`; scalar F64 action/JVP output; four C64 force outputs. Cover wrong dtype, rank, color axes, lattice equality, tangent equality, scalar seed dtype/rank, and malformed active directions with `catch_unwind` to prove typed errors.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test extension_metadata`

Expected: compile failure because the three operations do not exist.

- [ ] **Step 3: Implement payloads and symbolic inference**

Implement crate-private `WilsonActionOp`, `WilsonActionJvpOp`, and `WilsonForceOp` using `ExtensionOp`. Hash `beta.to_bits()` plus active direction bytes, downcast in `payload_eq`, return `Ok(None)` from `lower_to_standard_ops`, and validate `SymDim::constant_value()` for the two color axes while comparing all symbolic lattice dimensions structurally.

- [ ] **Step 4: Verify green**

Run: `cargo test -p gaugefields --test extension_metadata`

Expected: PASS for all positive and negative metadata paths.

- [ ] **Step 5: Commit**

```bash
git add gaugefields/src/autodiff.rs gaugefields/src/extension.rs gaugefields/src/error.rs gaugefields/tests/extension_metadata.rs
git commit -m "feat: define Wilson extension families"
```

### Task 5: Add host runtimes, registration, and traced execution

**Files:**
- Modify: `gaugefields/src/extension.rs`
- Modify: `gaugefields/src/lib.rs`
- Create: `gaugefields/tests/traced_action.rs`
- Create: `gaugefields/examples/traced_wilson_action.rs`

- [ ] **Step 1: Write runtime tests**

Construct four `TracedTensor::input_concrete_shape(DType::C64, shape)` inputs, call `wilson_action_traced([&u0,&u1,&u2,&u3], beta)`, compile with `GraphCompiler::compile_with_input_specs`, and run via `GraphExecutor::run_with_inputs`. Register exactly with `executor.register_extension(gaugefields::register_runtime)?`. Assert missing registration contains `missing runtime`, registered cold/random/Julia outputs match direct action, repeating that exact registration call is idempotent, GPU/backend input returns an error mentioning explicit download, and no `GaugeIdentityOp` remains.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test traced_action`

Expected: compile failure because the traced wrapper and registration function do not exist.

- [ ] **Step 3: Implement host execution**

Implement `HostReference` on each payload. Match input `Tensor` variants, validate before extraction, call shared kernels, and return `Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![value])?)` or C64 outputs. Implement:

```rust
pub fn register_runtime<B: TensorBackend + 'static>(
    executor: &mut ExtensionExecutor<B>,
) -> Result<(), ExtensionRuntimeRegistryError>
```

by registering three `Arc::new(HostReferenceRuntime::<B>::new(FAMILY))` values through `executor.registry_mut().register(...)`.
This public function is the closure contract consumed directly as
`graph_executor.register_extension(gaugefields::register_runtime)?`; do not
wrap it in a second application-facing registry API.

- [ ] **Step 4: Implement the public traced wrapper**

Implement the exact public signature from the design, build `Arc<WilsonActionOp>`, call `tenferro_runtime::extension::apply(op, &links)`, and return the single output with a typed crate error if the output count differs.

- [ ] **Step 5: Verify green**

Run: `cargo test -p gaugefields --test traced_action --all-features && cargo run -p gaugefields --example traced_wilson_action --all-features`

Expected: PASS; both test and example use
`executor.register_extension(gaugefields::register_runtime)?`, and the example
prints equal direct and traced action values with residual below `1e-13`.

- [ ] **Step 6: Commit**

```bash
git add gaugefields/src/extension.rs gaugefields/src/lib.rs gaugefields/tests/traced_action.rs gaugefields/examples/traced_wilson_action.rs
git commit -m "feat: execute traced Wilson action"
```

### Task 6: Document and verify Phase 6

**Files:**
- Modify: `README.md`
- Modify: `docs/design/dependencies.md`
- Modify: `docs/design/layout.md`
- Create: `docs/worklogs/2026-07-13-phase-6.md`

- [ ] **Step 1: Update user and reviewer documentation**

Document typed ownership, explicit host placement/download, application-owned `CpuBackend`/`GraphExecutor`, exact registration code, exact revision, shared custom stencil rationale, sources consulted, numerical residuals, commands, and remaining risks. Do not claim AD support until Phase 7.

- [ ] **Step 2: Run the complete Phase 6 gate**

Run:

```bash
GAUGEFIELDS_JL_DIR=/home/shinaoka/tensor4all/Gaugefields.jl julia fixtures/generate.jl
git diff --exit-code -- fixtures
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --no-default-features
cargo test --workspace --all-features
cargo test --workspace --doc --all-features
cargo doc --workspace --all-features --no-deps
cargo tree --workspace --all-features --duplicates
git diff --check
```

Expected: every command exits zero; fixture regeneration is byte-stable.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/design docs/worklogs/2026-07-13-phase-6.md
git commit -m "docs: record phase 6 extension integration"
```
