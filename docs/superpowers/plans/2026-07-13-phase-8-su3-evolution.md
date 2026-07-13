# Phase 8 SU(3) Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Julia-compatible SU(3) exponentiation, normalization, batched link evolution, and crate-private HMC sanity checks.

**Architecture:** Fixed-size `Mat3` code owns the exact Gaugefields.jl Cardano/fallback exponential and normalization projection. Field evolution builds one packed exponential tensor per direction and delegates `E U` to tenferro `dot_general`; a private deterministic leapfrog driver composes the public kernels only for regression tests.

**Tech Stack:** Rust 2021, tenferro `51bc0a7bef274e20d08fc054856cb4d74c284cbe`, `CpuBackend`, `BackendSessionHost`, `TensorDot`, `DotGeneralConfig`, `TypedTensor`, Gaugefields.jl `9e5719970770f4497405a856315c90bef7f74449`, `rand`/`rand_chacha` for deterministic test support.

---

## Required ownership decision before execution

At tenferro `51bc0a7b`, `CpuBackend::dot_general` and `BackendSessionHost::with_backend_session` create a fresh private `GemmAnalysisCache` per call. Persistent contraction analysis reuse is available only through `BackendSessionHost::with_backend_session_cached`, whose cache type is `<CpuBackend as BackendRuntimeCache>::RuntimeCache` and whose concrete `GemmAnalysisCache` is private. Therefore the design's exact public signature `exp_ta_update(&mut CpuBackend, ...)` cannot also guarantee reuse of contraction-analysis caches across calls.

Execution must adopt one explicit resolution before Task 4: retain the designed signature and document that backend context/buffer pool are reused but analysis is per call, or amend the public surface to an application-owned evolution owner that privately stores both `CpuBackend` and its associated runtime cache. The tasks below use the designed signature and do not claim cross-call analysis-cache reuse; changing that choice requires a design amendment before production edits.

## File structure

- Modify `gaugefields/src/mat3.rs`: TA construction helpers, Julia exponential, finite/unitarity/determinant helpers, normalization.
- Create `gaugefields/src/evolution.rs`: public `exp_ta`, `normalize_su3`, and batched `exp_ta_update`.
- Modify `gaugefields/src/field.rs`: final `TaGaugeField` construction/access APIs needed by evolution and momentum updates.
- Modify `gaugefields/src/error.rs`: non-finite, singular normalization, and backend/contraction variants.
- Modify `gaugefields/src/lib.rs`: public numerical exports only.
- Modify `fixtures/generate.jl`; create `fixtures/exp_ta/metadata.json` and `.npy` oracle files.
- Create `gaugefields/tests/exp_ta.rs`, `gaugefields/tests/normalize_su3.rs`, `gaugefields/tests/exp_ta_update.rs`.
- Create `gaugefields/src/hmc_test_support.rs` and `gaugefields/src/hmc_test_support/tests.rs`, compiled only for tests.
- Create `docs/worklogs/2026-07-13-phase-8.md`; update README and evolution design notes.

### Task 1: Generate exact Julia exponential fixtures

**Files:**
- Modify: `fixtures/generate.jl`
- Create: `fixtures/exp_ta/metadata.json`
- Create: `fixtures/exp_ta/random.npy`
- Create: `fixtures/exp_ta/degenerate.npy`
- Create: `fixtures/exp_ta/near_degenerate.npy`
- Create: `gaugefields/tests/exp_ta.rs`

- [ ] **Step 1: Write the failing fixture reader test**

Define fixed coefficient cases including zero, deterministic random values, exact degeneracy, and perturbations on both sides of the Julia fallback threshold. Assert Fortran order, C64 shape `[3,3,ncase]`, commit `9e5719970770f4497405a856315c90bef7f74449`, coefficients, `t`, and an explicit `branch` label for every case.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test exp_ta julia_exp_ta_fixture_has_branch_provenance -- --exact`

Expected: FAIL because `fixtures/exp_ta` does not exist.

- [ ] **Step 3: Extend the clean-checkout generator**

Use Gaugefields.jl's `exptU!` at the pinned source algorithm to emit the cases and metadata without reimplementing the expected values in Rust. Preserve the existing `GAUGEFIELDS_JL_DIR`, clean tracked-worktree rejection, and commit recording.

- [ ] **Step 4: Generate and verify green**

Run: `GAUGEFIELDS_JL_DIR=/home/shinaoka/tensor4all/Gaugefields.jl julia fixtures/generate.jl && cargo test -p gaugefields --test exp_ta julia_exp_ta_fixture_has_branch_provenance -- --exact`

Expected: PASS and metadata records the exact Julia source/revision and branch cases.

- [ ] **Step 5: Commit**

```bash
git add fixtures/generate.jl fixtures/exp_ta gaugefields/tests/exp_ta.rs
git commit -m "test: add Julia SU3 exponential oracle"
```

### Task 2: Port `exp_ta` with analytic and fallback branches

**Files:**
- Modify: `gaugefields/src/mat3.rs`
- Create: `gaugefields/src/evolution.rs`
- Modify: `gaugefields/src/error.rs`
- Modify: `gaugefields/src/lib.rs`
- Modify: `gaugefields/tests/exp_ta.rs`

- [ ] **Step 1: Write algebraic and oracle tests**

Assert the exact public type `fn(f64, &[f64;8]) -> Result<Mat3,GaugeError>`, identity at `t=0`, centered small-`t` derivative against `Mat3::from_gell_mann_coefficients`, all Julia fixture entries, branch coverage counters/hooks under `cfg(test)`, unitarity max residual, determinant-one residual, and rejection of non-finite `t` or coefficients.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test exp_ta`

Expected: compile failure because `exp_ta` does not exist.

- [ ] **Step 3: Implement TA matrix construction**

Add `Mat3::from_gell_mann_coefficients([f64;8])` using the established `A=(i/2) sum_a p_a lambda_a` convention and test exact round trip through `gell_mann_coefficients()`.

- [ ] **Step 4: Implement the Julia algorithm semantically**

Port `TA_gaugefields_4D_serial.jl:603-806`: Cardano eigenvalue parameters, explicit eigenvectors, phase rotation, spectral assembly, and the same degeneracy predicate/fourth-order fallback. Use checked finite tests before branch selection and return structured `GaugeError::NonFiniteSu3Input`; do not call a generic eigensolver.

- [ ] **Step 5: Verify green**

Run: `cargo test -p gaugefields --test exp_ta -- --nocapture`

Expected: PASS; random, degenerate, and near-degenerate maximum residuals meet the fixture-derived tolerance and both branches execute.

- [ ] **Step 6: Commit**

```bash
git add gaugefields/src/mat3.rs gaugefields/src/evolution.rs gaugefields/src/error.rs gaugefields/src/lib.rs gaugefields/tests/exp_ta.rs
git commit -m "feat: add Julia-compatible exp_ta"
```

### Task 3: Add Julia-compatible SU(3) normalization

**Files:**
- Modify: `gaugefields/src/evolution.rs`
- Create: `gaugefields/tests/normalize_su3.rs`

- [ ] **Step 1: Write projection and rejection tests**

Cover identity, controlled drift, random perturbed SU(3), Julia normalization fixture values, zero/dependent columns, NaN, and infinity. Assert post-projection column orthonormality and determinant-one residuals, and assert singular/non-finite structured errors without partial mutation by comparing the original matrix after failure.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test normalize_su3`

Expected: compile failure because `normalize_su3` does not exist.

- [ ] **Step 3: Implement normalization**

Port `gaugefields_4D_nowing.jl:2387-2454`: normalize the first column, orthogonalize/normalize the second, construct the third consistently, and apply determinant-phase correction. Compute into a local `Mat3`, validate finite norms against a documented threshold, and assign to the caller only after success.

- [ ] **Step 4: Verify green**

Run: `cargo test -p gaugefields --test normalize_su3 -- --nocapture`

Expected: PASS with finite max orthogonality and determinant residual reports.

- [ ] **Step 5: Commit**

```bash
git add gaugefields/src/evolution.rs gaugefields/tests/normalize_su3.rs
git commit -m "feat: normalize matrices to SU3"
```

### Task 4: Apply batched link evolution through tenferro

**Files:**
- Modify: `gaugefields/src/evolution.rs`
- Modify: `gaugefields/src/field.rs`
- Create: `gaugefields/tests/exp_ta_update.rs`

- [ ] **Step 1: Write update and delegation tests**

Assert the designed public signature with `CpuBackend`, cold/nontrivial momentum agreement against an explicit site-local `Mat3` reference on `2^4`, unchanged links for `t=0`, shape/lattice mismatch and backend placement errors, and transactional behavior on failure. Add a source contract requiring `BackendSessionHost::with_backend_session` and `dot_general` in `evolution.rs` and forbidding a site loop that multiplies full-field `Mat3` pairs.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --test exp_ta_update`

Expected: compile failure because `exp_ta_update` does not exist.

- [ ] **Step 3: Pack exponentials once per direction**

Borrow each F64 momentum direction once, call `exp_ta` per site, and build compact `TypedTensor<Complex64>` with shape `[3,3,Lx,Ly,Lz,Lt]`. Convert typed values to `Tensor::C64` without copying.

- [ ] **Step 4: Delegate the field product**

Enter one `BackendSessionHost::with_backend_session` closure for the update and call `TensorDot::dot_general` once per direction with:

```rust
DotGeneralConfig {
    lhs_contracting_dims: vec![1],
    rhs_contracting_dims: vec![0],
    lhs_batch_dims: vec![2, 3, 4, 5],
    rhs_batch_dims: vec![2, 3, 4, 5],
}
```

Validate four C64 outputs before replacing any link tensor. Reclaim temporary backend outputs where ownership permits. This reuses the `CpuBackend` context and buffer pool but, under the selected designed signature, does not promise cross-call GEMM-analysis-cache reuse.

- [ ] **Step 5: Verify green**

Run: `cargo test -p gaugefields --test exp_ta_update -- --nocapture`

Expected: PASS with max site/reference residual below the documented tolerance.

- [ ] **Step 6: Commit**

```bash
git add gaugefields/src/evolution.rs gaugefields/src/field.rs gaugefields/tests/exp_ta_update.rs
git commit -m "feat: apply batched SU3 link evolution"
```

### Task 5: Build crate-private leapfrog test support

**Files:**
- Create: `gaugefields/src/hmc_test_support.rs`
- Create: `gaugefields/src/hmc_test_support/tests.rs`
- Modify: `gaugefields/src/lib.rs`
- Modify: `gaugefields/Cargo.toml`

- [ ] **Step 1: Write deterministic HMC tests**

Under `cfg(test)`, create a fixed `ChaCha8Rng` seed and `4^4` hot field. Implement tests for `U(dt/2) -> P(dt) -> U(dt/2)`, momentum negation reversibility, finite values, max unitarity/determinant drift, Hamiltonian error ratios over halved `dt` consistent with global second order, and acceptance greater than `0.5` over a fixed documented short run. Print every residual and acceptance count.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields hmc_test_support::tests -- --nocapture`

Expected: compile failure because the test-only driver does not exist.

- [ ] **Step 3: Implement private momentum and leapfrog helpers**

Use `TaGaugeField` for momentum, `gauge_force` for `P <- P - dt*force/NC`, `exp_ta_update` for link half-steps, and kinetic energy from the documented coefficient normalization. Keep every helper `pub(crate)` inside a `cfg(test)` module; export no sampler, trajectory, or RNG API.

- [ ] **Step 4: Verify green**

Run: `RAYON_NUM_THREADS=1 cargo test -p gaugefields hmc_test_support::tests -- --nocapture`

Expected: PASS; reversibility, second-order ratios, drift bounds, finite checks, and acceptance threshold are all reported.

- [ ] **Step 5: Commit**

```bash
git add gaugefields/Cargo.toml Cargo.lock gaugefields/src/hmc_test_support.rs gaugefields/src/hmc_test_support/tests.rs gaugefields/src/lib.rs
git commit -m "test: add private HMC sanity driver"
```

### Task 6: Document and verify Phase 8

**Files:**
- Modify: `README.md`
- Modify: `docs/design/phase-6-8.md` if the cache-ownership decision changes the durable contract
- Create: `docs/worklogs/2026-07-13-phase-8.md`

- [ ] **Step 1: Record algorithms, ownership, and evidence**

Document exact Julia lines/revision, analytic/fallback rationale, normalization thresholds, explicit CPU placement, batched `dot_general` dimensions, selected cache-ownership resolution, release measurements for packing/update scaling, HMC parameters, residuals, acceptance, commands, and remaining risks. Keep the HMC driver absent from public docs/API listings.

- [ ] **Step 2: Run the complete Phase 8 gate**

Run:

```bash
GAUGEFIELDS_JL_DIR=/home/shinaoka/tensor4all/Gaugefields.jl julia fixtures/generate.jl
git diff --exit-code -- fixtures
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --no-default-features
RAYON_NUM_THREADS=1 cargo test --workspace --all-features -- --nocapture
cargo test --workspace --doc --all-features
cargo doc --workspace --all-features --no-deps
cargo test -p gaugefields --test exp_ta_update
git diff --check
```

Expected: every command exits zero; fixture regeneration is byte-stable; numerical output is finite.

- [ ] **Step 3: Run release scaling evidence**

Run: `RAYON_NUM_THREADS=1 cargo test -p gaugefields --release --test exp_ta_update -- --ignored --nocapture`

Expected: the documented `2^4`, `4^4`, and `8^4` update measurements complete, demonstrate tensor-sized scaling, and confirm the tenferro path.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/design/phase-6-8.md docs/worklogs/2026-07-13-phase-8.md
git commit -m "docs: record phase 8 evolution kernels"
```
