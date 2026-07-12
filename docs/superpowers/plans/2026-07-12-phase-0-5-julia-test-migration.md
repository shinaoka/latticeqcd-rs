# Phase 0–5 Julia Test Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three missing semantic Rust regressions extracted from Gaugefields.jl's Phase 0–5 initialization and HMC kernel tests.

**Architecture:** Keep Julia-generated field and force parity in the existing fixture tests. Add one integration test file that treats the checked nonzero 2⁴ field as input and verifies coefficient scaling and the test-local Julia momentum-update conversion without adding an integrator API.

**Tech Stack:** Rust 2021, `gaugefields`, tenferro Tensor slices, checked Gaugefields.jl fixtures at commit `9e5719970770f4497405a856315c90bef7f74449`.

---

### Task 1: Add Julia action and force coefficient contracts

**Files:**
- Create: `gaugefields/tests/julia_hmc_kernel_contracts.rs`
- Reference: `docs/superpowers/specs/2026-07-12-phase-0-5-julia-test-migration-design.md`

- [ ] **Step 1: Write a test that initially fails because its local comparison helper is deliberately strict**

Create the integration test with a finite-value component comparator and the three named tests below. During the red run, temporarily use the incorrect Julia momentum sign `+epsilon*dt/NC`; confirm that the nonzero fixture produces a mismatch. Do not commit the incorrect sign.

```rust
use gaugefields::{action_gradient, dsdu, gauge_force, load_fixture, wilson_action};
use num_complex::Complex64;
use std::path::Path;

fn fixture() -> gaugefields::Fixture {
    load_fixture(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/random_2x2x2x2"),
    )
    .unwrap()
}

fn assert_scaled_f64(actual: &[f64], base: &[f64], scale: f64, label: &str) {
    assert_eq!(actual.len(), base.len());
    let mut max_residual = 0.0_f64;
    for (&a, &b) in actual.iter().zip(base) {
        assert!(a.is_finite() && b.is_finite());
        max_residual = max_residual.max((a - scale * b).abs());
    }
    assert!(max_residual < 1e-13, "{label}: max residual={max_residual}");
}

fn assert_scaled_c64(actual: &[Complex64], base: &[Complex64], scale: f64, label: &str) {
    assert_eq!(actual.len(), base.len());
    let mut max_residual = 0.0_f64;
    for (&a, &b) in actual.iter().zip(base) {
        assert!(a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite());
        max_residual = max_residual.max((a - scale * b).norm());
    }
    assert!(max_residual < 1e-13, "{label}: max residual={max_residual}");
}
```

- [ ] **Step 2: Verify the intentionally wrong momentum sign fails**

Run:

```bash
cargo test -p gaugefields --test julia_hmc_kernel_contracts -- --nocapture
```

Expected: `julia_momentum_update_coefficient_matches_p_update` fails with a nonzero maximum residual. Restore the correct negative sign before proceeding.

- [ ] **Step 3: Implement the action beta-scaling test**

Use `beta = fixture.metadata().beta` and `scale` values `[-1.75, 0.0, 2.5]`. Require:

```rust
let base = wilson_action(f.links(), beta).unwrap();
for scale in [-1.75, 0.0, 2.5] {
    let actual = wilson_action(f.links(), scale * beta).unwrap();
    let residual = (actual - scale * base).abs();
    assert!(actual.is_finite() && residual < 1e-12,
            "scale={scale}: action residual={residual}");
}
```

- [ ] **Step 4: Implement derivative beta scaling for every direction and component**

Compute the base `dsdu`, `action_gradient`, and `gauge_force` once. For each scale `[-1.75, 0.0, 2.5]`, recompute all three APIs and compare every C64/F64 component with the helpers above. Labels must contain the API, scale, and direction.

- [ ] **Step 5: Implement the Julia `P_update!` coefficient contract**

Set finite representative values `epsilon = 0.5`, `dt = 0.125`, and calculate only in the test:

```rust
let coefficient = -epsilon * dt / f.links().nc() as f64;
let force = gauge_force(f.links(), f.metadata().beta).unwrap();
for (mu, tensor) in force.tensors().iter().enumerate() {
    let components = tensor.as_slice::<f64>().unwrap();
    let updated: Vec<_> = components.iter().map(|&value| coefficient * value).collect();
    assert_scaled_f64(&updated, components, coefficient, &format!("P_update mu={mu}"));
}
```

Also assert at least one input and updated coefficient is nonzero, ensuring an incorrect sign cannot pass vacuously.

- [ ] **Step 6: Run focused and complete verification**

Run:

```bash
cargo test -p gaugefields --test julia_hmc_kernel_contracts -- --nocapture
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --no-default-features
cargo test --workspace --all-features
cargo doc --workspace --all-features --no-deps
```

Expected: all commands exit zero; the focused suite reports three passing tests with finite residual diagnostics available on failure.

- [ ] **Step 7: Commit the migrated contracts**

```bash
git add gaugefields/tests/julia_hmc_kernel_contracts.rs \
  docs/superpowers/plans/2026-07-12-phase-0-5-julia-test-migration.md
git commit -m "test: port Julia phase 0-5 kernel contracts"
```

### Task 2: Audit traceability and publish

**Files:**
- Verify: `docs/superpowers/specs/2026-07-12-phase-0-5-julia-test-migration-design.md`
- Verify: `gaugefields/tests/julia_hmc_kernel_contracts.rs`

- [ ] Confirm every GitHub link in the design contains commit `9e5719970770f4497405a856315c90bef7f74449` rather than a moving branch.
- [ ] Confirm the implementation adds no production source or public API changes with `git diff --name-only origin/main...HEAD`.
- [ ] Run the full verification commands from Task 1 again on the final commit.
- [ ] Open one PR, wait for all GitHub Actions checks, and merge with a merge commit rather than squash. The PR must state that deferred Phase 6–8 tests are documented, not counted as migrated.
