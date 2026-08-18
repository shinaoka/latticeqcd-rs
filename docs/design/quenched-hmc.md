# Public quenched HMC and Julia trajectory parity

Date: 2026-08-18
Status: implemented

## Goal

Promote the crate's regression-only Wilson HMC into a small public SU(3) API,
using the approved reproducible RNG, and verify one complete leapfrog proposal
against Gaugefields.jl from identical links and momentum.

Reference implementation:

- Gaugefields.jl v0.7.2, commit
  `9e5719970770f4497405a856315c90bef7f74449`;
- `test/HMC_test_nowing.jl` for Hamiltonian, U-P-U leapfrog, and Metropolis
  conventions;
- `src/TA_Gaugefields.jl` and `src/4D/TA_gaugefields_4D_serial.jl` for Gaussian
  coefficient order and kinetic normalization.

## Scope

This change provides:

- Gaussian SU(3) momentum sampling from `ReproducibleRng`;
- kinetic energy and Wilson Hamiltonian evaluation;
- a caller-context-owned, transactional leapfrog trajectory;
- one transactional HMC update with Metropolis rollback;
- a Julia-generated deterministic one-trajectory oracle;
- a deterministic public HMC example.

Heatbath and ensemble statistics are a separate reviewed change. General
SU(N), alternate actions, alternate integrators, adaptation, persistent
momentum, and device-resident HMC are not included.

## Mathematical contract

For every positive direction and site,

```
P = (i/2) sum_{a=1}^8 p_a lambda_a,
p_a ~ Normal(0, 1),
K(P) = (1/2) sum p_a^2,
H(U, P) = -(beta/3) sum_plaquettes Re tr(P_mu_nu) + K(P).
```

One step of size `dt` is the Gaugefields.jl U-P-U leapfrog:

```
U <- exp((dt/2) P) U
P <- P - (dt/3) gauge_force(U, beta)
U <- exp((dt/2) P) U
```

`gauge_force` already includes beta and no `1/NC`; HMC supplies exactly the
single `1/3` factor. No reunitarization occurs inside a trajectory.

The Metropolis probability is `min(1, exp(-delta_h))`, where
`delta_h = H_proposed - H_initial`. One open uniform is consumed for every HMC
update, including downhill proposals, so draw consumption is fixed. Equality
accepts. Rejection restores all four original link fields.

## Public API

Add `crates/gaugefields/src/hmc.rs` and re-export:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HmcParams { /* private fields */ }

impl HmcParams {
    pub fn new(beta: f64, step_size: f64, steps: usize)
        -> Result<Self, GaugeError>;
    pub const fn beta(self) -> f64;
    pub const fn step_size(self) -> f64;
    pub const fn steps(self) -> usize;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HmcOutcome {
    pub accepted: bool,
    pub initial_hamiltonian: f64,
    pub proposed_hamiltonian: f64,
    pub delta_h: f64,
    pub acceptance_probability: f64,
}

pub fn sample_momentum(
    lattice: LatticeShape4,
    rng: &mut ReproducibleRng,
) -> Result<TaGaugeField, GaugeError>;

pub fn kinetic_energy(momentum: &TaGaugeField) -> Result<f64, GaugeError>;

pub fn hamiltonian(
    links: &GaugeLinks,
    momentum: &TaGaugeField,
    beta: f64,
) -> Result<f64, GaugeError>;

pub fn leapfrog_trajectory(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    params: HmcParams,
) -> Result<(), GaugeError>;

pub fn hmc_update(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    params: HmcParams,
    rng: &mut ReproducibleRng,
) -> Result<HmcOutcome, GaugeError>;
```

`HmcParams::new` rejects non-finite beta with the existing `NonFiniteBeta`,
non-finite step size with `NonFiniteStepSize`, non-positive finite step size
with `NonPositiveStepSize`, and zero steps with `ZeroHmcSteps`. All operations
are SU(3), host-placement boundaries and reject mismatched lattices before
numerical work.

`sample_momentum` fills direction 0 through 3, each in compact tensor storage
order, by repeated uncached Box--Muller pairs. Since `8*NV` is even, it consumes
exactly `4*8*NV` raw xoshiro words. For the `2x2x2x2` oracle that is 512 words;
the unconditional Metropolis uniform is word 513, and a test calls
`next_u64()` after `hmc_update` and compares word 514 with fixture metadata.
This proves the final stream position without adding state export.

`kinetic_energy` rejects non-finite inputs and finite inputs whose square sum
leaves the finite range. `hamiltonian` rejects a lattice mismatch and a
non-finite result. Metropolis uses a branch-stable probability: `1` when
`delta_h <= 0`, otherwise `exp(-delta_h)`.

`leapfrog_trajectory` computes on fallible duplicates of links and momentum,
then commits both only after every step succeeds. On any error, both caller
values remain unchanged. `hmc_update` similarly computes on a private proposal;
only acceptance replaces the caller's links. RNG advancement is not rolled
back on an error or rejection and is documented explicitly.

The current `hmc_test_support.rs` arithmetic must move to `hmc.rs`; no second
integrator, momentum arithmetic implementation, or compatibility shim remains.
Remove the now-unused `rand_chacha` dev-dependency and its stale dependency
documentation when the private regression module is removed.

## Deterministic Julia fixture

Add focused generator mode `hmc_trajectory` and
`fixtures/hmc_trajectory/` containing:

- metadata with lattice, beta, step size, steps, the four-word initial RNG
  state, the post-momentum acceptance uniform (word 513), the expected next raw
  word after it (word 514), `H_initial`, `H_proposed`, `delta_h`, acceptance
  probability and decision;
- four Fortran-order `p_initial_mu.npy` arrays and four `p_final_mu.npy` arrays;
- four Fortran-order proposed-link arrays;
- Julia version, Gaugefields.jl version/commit/source paths, formulas, storage
  order, and comparison tolerances.

Use a cold SU(3) `2x2x2x2` field. Starting from one recorded nonzero
four-word xoshiro256++ state, generate momentum in direction/storage order with
the approved scalar open-unit mapping and uncached Box--Muller transform, then
consume the next open uniform for Metropolis. This makes the complete public
`hmc_update` reproducible while still storing the explicit initial momentum for
fault localization. The generator must execute Gaugefields.jl's exported field,
exponential, action-derivative, and TA operations under the same U-P-U formula
as `test/HMC_test_nowing.jl`; it must not encode Rust output as the oracle.
Choose a stable nonzero trajectory and record every output at full `Float64`
precision. This oracle validates its one naturally produced acceptance or
rejection decision. It does not claim to exercise both branches from one RNG
state; a separate Rust contract test uses a second fixed state/configuration to
exercise the opposite branch and bitwise rollback.

The generator runs before the general fixture environment check, but this mode
requires `GAUGEFIELDS_JL_DIR`, verifies the exact reference commit, and exits
after writing only this fixture. Two clean runs must produce an identical
complete fixture-tree checksum.

## Tests and acceptance

Create `crates/gaugefields/tests/quenched_hmc.rs`. Tests must first be added and
shown RED because the public API/fixture is absent, then cover:

1. `sample_momentum` values, direction/storage order, exact raw draw count, and
   finite output;
2. kinetic and Hamiltonian normalization on concrete fields;
3. full Julia parity for sampled initial/final momentum, all proposed links,
   `H_initial`, `H_proposed`, `delta_h`, probability, decision, accepted/rolled
   back links, and the next raw RNG position. Verify word 513 by replaying
   `sample_momentum` from a second RNG initialized with the recorded state and
   then calling `open_unit_f64()`; verify word 514 by calling `next_u64()` after
   the complete `hmc_update`;
4. reversibility and second-order `delta_h` scaling migrated from the private
   regression module;
5. both accepted updates and rejected bitwise rollback, using distinct fixed
   RNG states/configurations when necessary;
6. invalid beta, step size, steps, lattice mismatch, non-finite momentum, and
   injected evolution failure, each with typed errors and transactional inputs;
7. compact `Debug` output without field or RNG state disclosure.

The fixture supplies absolute tolerances based on observed Julia/Rust residuals
with safety margin. They must be tight enough to fail a sign, `1/NC`, ordering,
or half-step error; a blanket relative percentage is not acceptable.

Add `examples/quenched_hmc.rs`, showing caller-owned `CpuEvolutionContext`,
explicit RNG state, several updates, acceptance count, and normalized
plaquette. Update README, layout/dependency docs, and a worklog recording RED,
generator provenance, review gates, and exact verification evidence.

## Verification gates

Before implementation:

- this design receives a recorded `reviewer-flash: Correct-to-merge` verdict;
- all findings are fixed and re-reviewed.

After implementation:

- deterministic focused generator twice with identical artifact checksum;
- `cargo fmt --all -- --check`;
- `cargo check --workspace`;
- `cargo test --workspace`;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
- `cargo test --workspace --all-features`;
- `cargo test --doc --workspace --all-features`;
- `cargo doc --workspace --all-features --no-deps`;
- `cargo run -p gaugefields --example quenched_hmc --all-features`;
- existing traced Wilson example;
- stale private-HMC/placeholder search and `git diff --check`;
- full-diff `reviewer-flash` review, finding fixes, and final re-review.

No PR is opened for this task; it is committed to the Phase 1 branch only.
