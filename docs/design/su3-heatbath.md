# SU(3) Wilson heatbath and plaquette statistics

Date: 2026-08-18
Status: implemented

## Goal

Complete Issue #17 Phase 1 with a public, reproducible SU(3) Wilson heatbath
sweep and demonstrate quenched plaquette agreement with Gaugefields.jl at
multiple beta values within independently estimated statistical uncertainty.

Reference implementation:

- Gaugefields.jl v0.7.2, commit
  `9e5719970770f4497405a856315c90bef7f74449`;
- `src/heatbath/heatbathmodule.jl` for the Kennedy--Pendleton SU(2) update,
  Cabibbo--Marinari SU(3) subgroup order, and direction/even-odd sweep order;
- `test/heatbathtest.jl` and `test/heatbathtest_bare.jl` for the Wilson heatbath
  and normalized plaquette convention.

## Scope

This change provides one Wilson-action SU(3), four-dimensional, CPU/host
heatbath sweep driven by `ReproducibleRng`, plus an example and a three-beta
statistical oracle. It does not add SU(2), general SU(N), improved actions,
overrelaxation, GPU/MPI execution, adaptation, or a generic Markov-chain
framework.

## Public API

Add `crates/gaugefields/src/heatbath.rs` and re-export:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatbathParams { /* private fields */ }

impl HeatbathParams {
    pub fn new(beta: f64, max_attempts: usize) -> Result<Self, GaugeError>;
    pub const fn beta(self) -> f64;
    pub const fn max_attempts(self) -> usize;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeatbathSweepStats {
    pub updated_links: usize,
    pub su2_attempts: usize,
}

pub fn heatbath_sweep(
    links: &mut GaugeLinks,
    params: HeatbathParams,
    rng: &mut ReproducibleRng,
) -> Result<HeatbathSweepStats, GaugeError>;
```

`HeatbathParams::new` rejects non-finite beta with the existing
`NonFiniteBeta`, non-positive finite beta with `NonPositiveHeatbathBeta`, and
zero `max_attempts` with `ZeroHeatbathAttempts`. `heatbath_sweep` requires
host-resident SU(3) links and even positive extents on all four axes. The even
extent restriction is explicit because the reference direction/even-odd
checkerboard assumes a bipartite periodic lattice; an odd periodic extent
would connect equal parities. It returns `OddHeatbathExtent { axis, extent }`
before RNG consumption. Kernel failures use `SingularHeatbathStaple {
direction, site, subgroup }`, `HeatbathNumericalRange { stage }`, and
`HeatbathRejectionLimit { max_attempts }`.

A successful sweep updates exactly `4*NV` links and performs exactly three
fixed SU(2) subgroup updates per link. `su2_attempts` counts total
Kennedy--Pendleton loop iterations across all subgroups, including each
accepting iteration; it therefore equals three per link when every subgroup
accepts on its first try. It is checked for overflow and pinned by a fixed-draw
determinism test.

## Wilson staple and sweep order

Reuse `PreparedGaugeField::force_staple`; do not duplicate periodic topology or
staple arithmetic. For link `U_mu(x)`, the heatbath matrix is
`V = force_staple(x, mu)^dagger`, so the local action is proportional to
`Re tr(U_mu V)`.

The sweep order is deterministic:

1. directions `mu = 0,1,2,3`;
2. even parity, then odd parity;
3. within each parity, ascending compact site index (x fastest, then y, z, t).

For one direction/parity, construct one `PreparedGaugeField`, compute all new
links from that immutable parity snapshot and the shared RNG in site order,
drop the snapshot, then store the updates. Same-parity links are not Wilson
staple neighbors on the required even periodic lattice. Previous directions
and the opposite parity remain immediately visible, matching Gaugefields.jl's
`evaluate_gaugelinks_evenodd!` followed by `map_U!`.

The complete sweep runs on a fallible duplicate of `GaugeLinks` and replaces
the caller only after all directions/parities succeed. A singular staple,
rejection-limit exhaustion, allocation error, or numerical error leaves every
caller link bitwise unchanged. RNG draws already consumed are not rolled back
and this is documented.

Move fallible link duplication to one crate-private field helper shared by HMC
and heatbath; do not retain two cloning implementations or expose cloning as a
new public abstraction. Implement the sweep through one private parameterized
core that accepts a uniform-draw closure and an update-observer closure. The
public wrapper supplies `ReproducibleRng::open_unit_f64` and a no-op observer;
unit tests supply scripted draws and record `(direction, parity, site,
subgroup, attempt)` tuples. Neither seam is exported or conditionally changes
the production algorithm.

## SU(2) Kennedy--Pendleton kernel

For each SU(3) link, use subgroup order `(0,1)`, `(1,2)`, `(0,2)`. Before each
subgroup hit:

1. compute `UV = U*V`;
2. extract the selected 2x2 diagonal block;
3. project it to quaternion form
   `S = [[alpha, -conj(beta)], [beta, conj(alpha)]]`;
4. set `rho = sqrt(|alpha|^2 + |beta|^2)` and
   `V0 = S^dagger/rho`;
5. sample a Kennedy--Pendleton quaternion with
   `k = 2*(beta/3)*rho`;
6. form `K*V0`, project and normalize it to SU(2), embed it in SU(3), and
   left-multiply `U`;
7. after the three hits, apply the existing `normalize_su3` once.

A non-finite or zero `rho`, non-finite `k`, invalid square-root radicand, or
non-finite matrix is a typed heatbath numerical/singular error.

Each rejection attempt consumes four open uniforms in this order:

```text
x      = -log(u1)/k
xprime = -log(u2)/k
delta  = xprime + x*cos(2*pi*u3)^2
accept iff u4^2 <= 1 - delta/2
```

After acceptance, consume two more open uniforms for `phi=2*pi*u5` and
`cos(theta)=2*u6-1`. The sampled quaternion is the Gaugefields.jl convention:

```text
a0 = 1-delta
r  = sqrt(1-a0^2)
a1 = r*cos(phi)*sin(theta)
a2 = r*sin(phi)*sin(theta)
a3 = r*cos(theta)
K  = [[a0+i*a3, a2+i*a1], [-a2+i*a1, a0-i*a3]]
```

## Deliberate corrections to the Julia implementation

The target distribution and update order are the compatibility contract, not
accidental RNG consumption or numerical defects. Rust therefore:

- does not perform the four unused preliminary draws before the Julia rejection
  loop;
- normalizes a projected SU(2) matrix by `sqrt(|alpha|^2+|beta|^2)`, not by the
  unsquared norm used at `heatbathmodule.jl:652-656`;
- rejects zero/singular staples and non-finite intermediates instead of dividing
  by zero;
- uses open uniforms, while Julia's in-loop `rand()` may return exactly zero
  and send `-log(0)` to a non-finite rejection/error path;
- accepts an explicit RNG instead of using global state.

These differences are documented in fixture metadata. They intentionally make
bitwise Julia/Rust heatbath-link parity a non-goal; equilibrium observables are
compared statistically.

## Statistical Julia oracle

Add focused generator mode `heatbath_statistics` and
`fixtures/heatbath_statistics/metadata.json`. The generator must call the real
Gaugefields.jl `Heatbath`/`heatbath!` implementation and
`calculate_Plaquette`; it must not reimplement heatbath or encode Rust results.
It verifies the exact clean reference commit and records Julia/package/source
provenance, the deliberate corrections above, and the complete schedule.

Use three cold-start SU(3) `2x2x2x2` chains:

- beta values `5.5`, `5.7`, and `6.0`;
- independent fixed Julia seeds, one per beta;
- 512 burn-in sweeps;
- 32 blocks of 32 measured sweeps (1024 measurements);
- one normalized plaquette measurement after every measured sweep.

For each beta store all 32 block means, their mean, and
`standard_error = sample_stddev(block_means)/sqrt(32)`. Full and focused fixture
modes regenerate this metadata. Two consecutive runs must produce identical
complete fixture-tree checksums.

The Rust integration test uses distinct fixed nonzero four-word xoshiro states,
the identical lattice/beta/burn-in/block schedule, `max_attempts=100_000`
(matching Julia `Heatbath`'s default `ITERATION_MAX`), and the public
`heatbath_sweep`. For each beta it computes its own 32 block means and standard
error. Acceptance requires:

```text
abs(mean_rust - mean_julia)
    <= 6 * sqrt(se_rust^2 + se_julia^2)
```

Additionally every mean must be finite and in `[0,1]`, every standard error
must be finite, positive, and below `0.03`, and every chain must change from the
cold field. These bounds prevent an inflated or zero uncertainty from making
the comparison vacuous. Fixed schedules and seeds make the test deterministic,
not flaky; the criterion remains a statistical comparison of independent
streams rather than exact trajectory matching.

## Tests and acceptance

Add tests before production implementation and record focused RED evidence.
Tests cover:

1. parameter validation, accessors, compact debug, and rustdoc examples;
2. one fixed-state sweep determinism, exact `4*NV` update count, attempt count,
   finite output, SU(3) unitarity/determinant bounds, and plaquette change;
3. direction/even-odd/site ordering through a test-only injected sampler trace;
4. rejection-limit, singular-subgroup, non-finite/numerical, host-placement,
   unsupported-NC, odd-extent, and allocation-overflow errors;
5. bitwise caller-link transactionality and documented non-rollback RNG position
   on failures;
6. exact subgroup order and corrected square-root SU(2) normalization in
   focused unit tests;
7. the three-beta Julia/Rust block-statistics criterion above;
8. all existing HMC, force, action, fixture, and RNG tests unchanged.

Add `examples/quenched_heatbath.rs` with caller-owned links/RNG, several sweeps,
attempt statistics, and normalized plaquette. Update README, layout/dependency
docs, and a worklog with RED, provenance, review gates, generator checksums,
statistical values, runtime, and exact verification evidence.

## Verification gates

Before implementation:

- this design receives recorded `reviewer-flash: Correct-to-merge`;
- all findings are fixed and re-reviewed.

After implementation:

- focused and default generators twice, with identical complete artifact
  checksums;
- `cargo fmt --all -- --check`;
- `cargo check --workspace`;
- `cargo test --workspace`;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
- `cargo test --workspace --all-features`;
- `cargo test --doc --workspace --all-features`;
- `cargo doc --workspace --all-features --no-deps`;
- heatbath and HMC examples plus the traced Wilson example;
- focused heatbath/HMC tests and a release-mode runtime diagnostic for the
  statistical schedule;
- stale placeholder/general-SUN/duplicate-staple search and `git diff --check`;
- full-diff `reviewer-flash` review, fixes, and final re-review.

No PR is opened for this task. It is committed to the Phase 1 branch only.
