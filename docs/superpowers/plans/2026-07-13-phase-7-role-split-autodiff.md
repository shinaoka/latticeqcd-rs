# Phase 7 Role-Split Autodiff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit first-order JVP and reverse differentiation for traced Wilson action using tenferro's linearize-then-linear-transpose extension rules.

**Architecture:** `WilsonActionLinearize` emits the variable-arity Wilson JVP family for active link directions only. `WilsonJvpTranspose` transposes that linear operation into the Wilson force family, while the force family intentionally has no AD rule so higher-order requests fail explicitly; applications attach `ad_rules()` to their own `AdContext`.

**Tech Stack:** Rust 2021, tenferro `51bc0a7bef274e20d08fc054856cb4d74c284cbe`, `ExtensionLinearizeRule`, `ExtensionLinearTransposeRule`, `PrimitiveRuleBuilder`, `AdContext`, `TracedTensorAdExt`, Julia/Phase 5 numerical oracles.

---

## File structure

- Create `gaugefields/src/ad.rs`: operation downcasts, action linearize rule, JVP transpose rule, `ad_rules()`.
- Modify `gaugefields/src/extension.rs`: crate-private constructors for JVP/force payloads and arbitrary scalar-cotangent host force execution.
- Modify `gaugefields/src/error.rs`: structured AD registration/unsupported differentiation wrappers.
- Modify `gaugefields/src/lib.rs`: export only `ad_rules` under `autodiff`.
- Create `gaugefields/tests/ad_registration.rs`, `gaugefields/tests/traced_jvp.rs`, `gaugefields/tests/traced_gradient.rs`, and `gaugefields/tests/ad_failures.rs`.
- Create `docs/worklogs/2026-07-13-phase-7.md`; update `docs/design/ad-convention.md` and README.

### Task 1: Register the role-split rule set

**Files:**
- Create: `gaugefields/src/ad.rs`
- Modify: `gaugefields/src/lib.rs`
- Create: `gaugefields/tests/ad_registration.rs`

- [ ] **Step 1: Write the failing registration contract**

Assert that `ad_rules()` returns an `ExtensionRuleSet` whose `lookup_linearize("gaugefields.wilson_action.v1")` and `lookup_linear_transpose("gaugefields.wilson_action_jvp.v1")` are present, while no primal-VJP rule exists and no force rule is registered. Build two independent `AdContext::builder().with_extension_rules(ad_rules()?).build()` values and verify both succeed.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --all-features --test ad_registration`

Expected: compile failure because `ad_rules` does not exist.

- [ ] **Step 3: Implement the rule-set boundary**

Create private zero-sized rule structs and:

```rust
pub fn ad_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    ExtensionRuleSet::new()
        .with_linearize(Arc::new(WilsonActionLinearize))?
        .with_linear_transpose(Arc::new(WilsonJvpTranspose))
}
```

Keep rule structs and payload constructors crate-private.

- [ ] **Step 4: Verify green**

Run: `cargo test -p gaugefields --all-features --test ad_registration`

Expected: PASS and independent contexts contain identical rule roles.

- [ ] **Step 5: Commit**

```bash
git add gaugefields/src/ad.rs gaugefields/src/lib.rs gaugefields/tests/ad_registration.rs
git commit -m "feat: register Wilson role-split AD rules"
```

### Task 2: Linearize action into the active-direction JVP

**Files:**
- Modify: `gaugefields/src/ad.rs`
- Modify: `gaugefields/src/extension.rs`
- Create: `gaugefields/tests/traced_jvp.rs`

- [ ] **Step 1: Write active-direction graph tests**

For one, two, and four active links, call `AdContext::jvp` and inspect the resolved graph: exactly one `WilsonActionJvpOp` is emitted, its `active_dirs` equals the sorted active input directions, its arity is `4 + active_dirs.len()`, and inactive tangent tensors are absent. Add wrong rule arity/downcast tests returning `tidu::ADRuleError::InvalidInput`, never panic.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --all-features --test traced_jvp active_direction_payload_omits_inactive_tangents -- --exact`

Expected: FAIL because the action linearize rule has no implementation.

- [ ] **Step 3: Implement `ExtensionLinearizeRule`**

Downcast `WilsonActionOp`, validate four primal inputs/one output/four tangent slots, collect `(mu, LocalValueId)` from `tangent_in.iter().enumerate().filter_map(...)`, and emit `StdTensorOp::Extension(Arc::new(WilsonActionJvpOp::new(beta, active_dirs)?))` through `PrimitiveRuleBuilder::add_operation`. Inputs are four `ValueRef::External(primal_in[mu].clone())` followed by active `ValueRef::Local(tangent)` values; role is `OperationRole::Linearized` with a mask aligned to this emitted input list. Return the scalar output in `vec![Some(id)]`, or `vec![None]` when no tangent is active.

- [ ] **Step 4: Verify green**

Run: `cargo test -p gaugefields --all-features --test traced_jvp active_direction_payload_omits_inactive_tangents -- --exact`

Expected: PASS for all active subsets.

- [ ] **Step 5: Commit**

```bash
git add gaugefields/src/ad.rs gaugefields/src/extension.rs gaugefields/tests/traced_jvp.rs
git commit -m "feat: linearize Wilson action"
```

### Task 3: Validate JVP numerically

**Files:**
- Modify: `gaugefields/tests/traced_jvp.rs`

- [ ] **Step 1: Add independent oracle tests**

Use deterministic nonzero C64 tangents in multiple direction subsets. Compare executed `AdContext::jvp` to (a) the centered direct-action finite difference over `h = [1e-2, 5e-3, 2.5e-3, 1.25e-3]` and select the minimum finite residual, and (b) `sum_mu Re Σ conj(action_gradient_mu) * tangent_mu`. Print absolute and relative residuals and reject non-finite values.

- [ ] **Step 2: Verify red against an intentionally wrong sign**

Temporarily assert the negative inner product and run: `cargo test -p gaugefields --all-features --test traced_jvp jvp_matches_finite_difference_and_gradient_inner_product -- --exact`

Expected: FAIL with a finite residual larger than `1e-8`; restore the correct sign before implementation work continues.

- [ ] **Step 3: Implement JVP host execution**

In `WilsonActionJvpOp::execute`, validate all four links and active tangent shape/placement, borrow each once, and call the shared directional derivative kernel without materializing inactive zeros. Return one scalar `Tensor::F64`.

- [ ] **Step 4: Verify green**

Run: `cargo test -p gaugefields --all-features --test traced_jvp`

Expected: PASS; finite-difference and gradient-inner-product residuals satisfy the documented sweep-derived tolerance.

- [ ] **Step 5: Commit**

```bash
git add gaugefields/src/extension.rs gaugefields/tests/traced_jvp.rs
git commit -m "test: validate Wilson action JVP"
```

### Task 4: Transpose JVP into arbitrary-seed force

**Files:**
- Modify: `gaugefields/src/ad.rs`
- Modify: `gaugefields/src/extension.rs`
- Create: `gaugefields/tests/traced_gradient.rs`

- [ ] **Step 1: Write reverse graph and seed tests**

Use `AdContext::{grad,vjp}`. Inspect that reverse transformation contains `WilsonForceOp`, not a direct primal-VJP. Execute seeds `1.0`, `-2.5`, and `0.25`; assert active outputs only, four-direction `grad` agreement with Phase 5 `action_gradient`, and exact linear scaling within `1e-13` max residual.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --all-features --test traced_gradient`

Expected: FAIL because the JVP family has no linear-transpose rule.

- [ ] **Step 3: Implement `ExtensionLinearTransposeRule`**

Downcast `WilsonActionJvpOp`, validate one optional scalar cotangent, `inputs.len() == 4 + active_dirs.len()`, and the tenferro `active_mask`. Add a crate-private `fixed_transpose_input` helper that matches `PrimitiveTransposeInput::Residual(key)` to `ValueRef::External(key.clone())`, matches `PrimitiveTransposeInput::Linear { primal: Some(primal), .. }` to the primal external value, and returns `ADRuleError::invalid_input` for a linear input without a retained primal. Emit one `WilsonForceOp` with the four fixed primal link inputs plus `ValueRef::Local(cotangent)` through `PrimitiveRuleBuilder`. Return a vector aligned to JVP inputs: `Some(force_output_for_mu)` only for active tangent slots and `None` for primal/non-active slots according to the role contract.

- [ ] **Step 4: Scale the host force by the runtime seed**

Validate a rank-zero F64 fifth input, read its sole finite value, and multiply the shared dense complex gradient kernel by that arbitrary seed before producing four C64 tensors.

- [ ] **Step 5: Verify green**

Run: `cargo test -p gaugefields --all-features --test traced_gradient`

Expected: PASS for all directions and positive, negative, and fractional seeds.

- [ ] **Step 6: Commit**

```bash
git add gaugefields/src/ad.rs gaugefields/src/extension.rs gaugefields/tests/traced_gradient.rs
git commit -m "feat: transpose Wilson JVP to force"
```

### Task 5: Make failure modes explicit

**Files:**
- Create: `gaugefields/tests/ad_failures.rs`

- [ ] **Step 1: Write failure-path tests**

Separate these cases: registered AD rules but missing runtime; registered runtime but `AdContext` without `ad_rules`; malformed tangent dtype/shape; malformed scalar cotangent; and differentiation through `WilsonForceOp`. Assert distinct tenferro runtime/AD error variants or source-preserving messages and wrap every public call in `catch_unwind`.

- [ ] **Step 2: Verify red**

Run: `cargo test -p gaugefields --all-features --test ad_failures`

Expected: FAIL because the force family has not yet been exercised as an intentionally unsupported higher-order path.

- [ ] **Step 3: Preserve the owning error boundary**

Do not introduce a gaugefields AD execution wrapper solely to translate errors: applications call the owning `AdContext` directly. Keep the force family absent from `ad_rules()` and assert that tenferro returns its typed unsupported/missing-linearize error naming `gaugefields.wilson_force.v1`; runtime execution independently returns tenferro's missing-runtime error. Gauge extension host callbacks continue mapping only numerical/ABI validation through their existing source-preserving tenferro tensor errors.

- [ ] **Step 4: Verify green**

Run: `cargo test -p gaugefields --all-features --test ad_failures`

Expected: PASS and no failure path panics.

- [ ] **Step 5: Commit**

```bash
git add gaugefields/tests/ad_failures.rs
git commit -m "fix: classify Wilson AD failures"
```

### Task 6: Document and verify Phase 7

**Files:**
- Modify: `README.md`
- Modify: `docs/design/ad-convention.md`
- Create: `docs/worklogs/2026-07-13-phase-7.md`

- [ ] **Step 1: Document the supported first-order path**

Show explicit runtime registration and `AdContext::builder().with_extension_rules(ad_rules()?)`; state the Hermitian complex convention, active-direction behavior, arbitrary cotangent scaling, and unsupported higher order. Record finite-difference sweep residuals and Julia provenance.

- [ ] **Step 2: Run the complete Phase 7 gate**

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
git diff --check
```

Expected: every command exits zero and all numerical reports are finite.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/design/ad-convention.md docs/worklogs/2026-07-13-phase-7.md
git commit -m "docs: record phase 7 autodiff support"
```
