# Phase 4 fermion measurements and frontend

Status: pre-implementation design

## Scope

Complete Issue #17 Phase 4 on the existing CPU/host, single-process, 4D SU(3),
`Complex64` boundary:

1. deterministic pion correlator and chiral-condensate measurements;
2. a strict TOML-driven `latticeqcd` library and binary over implemented Rust
   updates, measurements, ILDG input, and ILDG output;
3. fixed-configuration and short-ensemble Julia/Rust comparisons.

Not in scope: GPU, MPI, SU(N), clover/domain-wall, SLMC, arbitrary Nf HMC,
runtime Remez, custom gauge actions, legacy Julia parameter aliases, JLD2,
BridgeText, hot/instanton starts, or implicit fallback.

## Pinned references

- QCDMeasurements.jl v0.2.13
  `9e04c37bbd68712cf7a749ae5aff10eb6aae4566`
  - `src/measurements/measure_Pion_correlator.jl`
  - `src/measurements/measure_chiral_condensate.jl`
  - `src/parameters/parameters.jl`
- LatticeQCD.jl v1.3.7
  `c09de20aae10f28f6a9c7e84e7711fce94d50915`
  - `src/system/lqcd.jl`
  - `src/system/parameters_TOML.jl`
  - `src/system/parameter_structs.jl`
  - `src/measurements/Measurement_set.jl`
- LatticeDiracOperators.jl v0.6.4
  `bdef628184597815ba3e0cddf2536df767e78a02`

Confirmed/reference bug records are repository-local Issues #23, #27, #29,
and #30. Rust intentionally uses the corrected pion contraction and canonical
Z4 noise. No upstream report or patch is authorized by these records.

## Crate boundary

`measurements` gains an optional `fermions` feature backed by the optional
`dirac-operators` dependency. Its existing gluonic-only default build remains
unchanged. `dirac-operators` does not depend on `measurements`, so the graph is
acyclic.

A new `latticeqcd` crate depends on `gaugefields`, `wilsonloop`,
`dirac-operators`, and `measurements` with `fermions`. It owns parameter parsing,
validation, scheduling, execution reports, and output publication. Lower crates
continue to own numerical kernels.

Only one new external dependency is justified: `toml`, for a standards-compliant
parser integrated with the already-used `serde`. No CLI framework is needed;
the binary accepts exactly one path using `std::env`.

## Task A: fermion measurements

### Minimal field seam

Add `FermionField::point_source(lattice, components, color, component, site)`.
It validates indices and checked allocation before constructing one unit basis
field. This keeps the private physical offset formula in its owning crate.

### Pion correlator

Expose under `measurements::fermions`:

```text
pion_correlator(operator, solver_params) -> PionCorrelator
```

The generic bound is `FermionOperator`; the existing transactional `bicgstab`
solves `D p = b` from zero for each point source at site zero. Source order is
color outer, component inner, matching QCDMeasurements.jl. The result contains:

- `values: Vec<f64>` of length `NT`;
- one `SolverReport` per source in source order.

For each source alpha and every sink color/component beta,

```text
C_pi(t) = sum_xyz sum_alpha,beta |G_beta,alpha(x,y,z,t)|^2.
```

There is no volume, color, spin, or source normalization and no staggered sign.
Every intermediate and result must be finite. The implementation accumulates
directly from each solved field; it does not allocate the Julia six-dimensional
`S` reconstruction.

Issue #29 records that pinned Julia duplicates `G_alpha,alpha` across all beta.
The fixture records both the corrected value and an independently evaluated
legacy value, but only the corrected value is public Rust behavior.

### Chiral condensate

Expose:

```text
stochastic_chiral_condensate(
    operator,
    flavor_factor,
    source_count,
    solver_params,
    rng,
) -> ChiralCondensate
```

The generic operator permits the supported Wilson and staggered host operators;
callers choose the finite positive `flavor_factor`. The Phase 4 frontend uses
`Nf/4` for staggered and does not expose a Wilson chiral measurement because the
pinned public QCDMeasurements parameter path supports staggered only.

Each source is canonical Z4 `{1, i, -1, -i}` generated from the caller-owned
`ReproducibleRng`, one raw word per physical field component in Rust compact
storage order. The mapping is fixed: `k = word & 3`, then
`[1, i, -1, -i][k]`. This intentionally does not reproduce Issue #27's
`k*pi/4` noise. Julia's native draw loop shares this physical order only for a
one-component staggered field; deterministic fixtures therefore store explicit
codes per `(color, component, site)` and never rely on matched cross-language
seeds. For each source, solve `D p = r` from zero and accumulate
`Re(r†p)`. Return:

```text
flavor_factor * mean_r Re(r† D^-1 r) / NV.
```

The result contains the scalar, per-source unnormalized estimates, and solver
reports. `source_count == 0`, non-finite/non-positive factors, incompatible
fields, solver failures, and non-finite arithmetic are typed errors. RNG draws
already consumed are not rolled back.

### Deterministic fixture

Add generator mode and tree `fixtures/fermion_measurements_phase4` using a
nontrivial `[2,2,2,4]` SU(3) field, default anti-periodic temporal boundary,
Wilson `kappa=0.08`, staggered `mass=0.17`, and explicit canonical Z4 codes.
Julia constructs every point/noise source and calls the pinned operators and
`solve_DinvX!`; it computes the corrected contraction independently instead of
calling the buggy high-level pion measurement.

Payloads include links, all point sources and propagators, corrected Wilson and
staggered correlators, legacy correlators, fixed chiral sources/solutions,
per-source inner products, per-side solver diagnostics, layout/boundary metadata,
and all source URLs/revisions. Julia's `bicg` and Rust's `bicgstab` diagnostics
are provenance records and are not compared for iteration-count equality; only
solutions and fresh residuals are cross-language gates. The generator passes
the tight solver tolerance under Julia's correct fermion-specific key: Wilson
`eps_CG`, staggered `eps` (avoiding Issue #23). Rust uses absolute squared
tolerance `1e-24` for point and noise sources, and Julia uses the corresponding
tight absolute squared threshold. The test consumes every declared payload and
metadata field.

Gates:

- source/layout/Z4 mapping exact;
- operator/propagator/measurement max absolute error `<= 2e-12` and relative
  error `<= 2e-10`;
- independently recomputed true relative residual `<= 1e-11`;
- cold/synthetic identity contraction checks;
- typed invalid-input and RNG-advance tests;
- two clean Julia generations with byte-identical complete trees.

## Task B: strict `latticeqcd` frontend

### Parameter schema

All structs use `serde(deny_unknown_fields)`. Physical/numerical/reproducibility
values have no defaults. Only omitted measurement lists, omitted flow, and
omitted output mean disabled.

```toml
schema_version = 1

[physical]
lattice = [4, 4, 4, 4]
beta = 5.7

[initial]
kind = "cold"                 # or "ildg" with path

[fermions]
kind = "quenched"             # or "wilson_nf2" / "staggered_nf2"

[update]
kind = "hmc"                  # or quenched-only "heatbath"

[rng]
state_hex = ["...", "...", "...", "..."]

[control]
first_trajectory = 1
trajectories = 10
thermalization = 2
measure_initial = false
```

`state_hex` words are exactly 16 ASCII hexadecimal digits so all `u64` states
round-trip through TOML. All-zero state is invalid.

Tagged fermion variants:

- `quenched`;
- `wilson_nf2`: `kappa`, explicit `[i8;4]` boundary, solver;
- `staggered_nf2`: `mass`, explicit boundary, `lambda_low`, `lambda_high`,
  solver. Bounds must stay within `[0.0004,64]`.

Tagged update variants:

- `hmc`: finite positive `step_size`, positive `steps`; dispatch is exact:
  `quenched` to `gaugefields::hmc_update`, `wilson_nf2` to
  `dirac_operators::wilson_hmc_update`, and `staggered_nf2` to
  `dirac_operators::staggered_hmc_update`;
- `heatbath`: positive `max_attempts`, quenched only, all extents even; each
  requested update is exactly one `gaugefields::heatbath_sweep`.

No fermion variant is metadata-only: each accepted sea-fermion kind selects its
matching Phase 3 dynamical update. Any other pairing is rejected before RNG use.

A solver has finite positive absolute squared `tolerance` and positive
`max_iterations`.

### Scheduled measurements

`[[measurements]]` entries contain a positive `every` and one kind:

- `plaquette`;
- `polyakov_loop`;
- `clover_topological_charge`;
- `pion_wilson` with valence `kappa`, boundary, solver;
- `pion_staggered` with valence mass, boundary, solver;
- `chiral_staggered` with valence mass, boundary, solver, positive `sources`,
  and positive `flavors` (normalization `flavors/4`). `flavors` is a valence
  estimator normalization and does not alter the configured sea action; values
  greater than four are allowed but carry only that explicit normalization
  meaning.

Valence parameters are explicit rather than silently inherited, so quenched and
dynamical runs support the same measurements. Duplicate measurement kinds with
identical valence parameters are rejected.

Optional `[gradient_flow]` contains positive `every_trajectories`, positive
finite `step_size`, positive `steps`, positive `measure_every_steps`, and a
non-empty unique list limited to the three gluonic measurement kinds. `steps`
must be divisible by `measure_every_steps`. The action is the existing Wilson
plaquette action and is not configurable in Phase 4.

Optional `[output]` contains `directory`, non-empty safe `prefix`, and positive
`every`; only ILDG is supported and therefore no format field is exposed.

### Validation and side effects

Validation order is fixed:

1. UTF-8/TOML syntax and strict deserialization;
2. schema version;
3. lattice, volume, beta, trajectory-count/ID overflow;
4. RNG words;
5. fermion and solver values;
6. update compatibility;
7. measurement and flow schedules;
8. output path/prefix;
9. ILDG read and lattice equality;
10. context construction.

No directory creation, file truncation, backend construction, or RNG draw occurs
before value validation. Unsupported strings and combinations fail explicitly.

### Runner semantics

`run_lqcd(&Params) -> Result<RunReport, RunFailure>` owns one
`ReproducibleRng` and one reusable `CpuEvolutionContext`. It initializes cold
links or reads one strict ILDG configuration, then executes exactly the requested
number of updates.

`first_trajectory >= 1` and `trajectories > 0` are required; checked addition
must represent the final trajectory ID. `thermalization <= trajectories` is
allowed, including a deliberate thermalization-only run with no post-update
measurements. If `measure_initial` is true and `thermalization == 0`,
measurements use ID `first_trajectory - 1`; otherwise requesting initial
measurement is invalid. Initial measurements obey each measurement's ordinary
`trajectory_id % every == 0` interval and never trigger flow or output.

After each update, the update counts as completed even when HMC rejects. Bare
measurements run only after `completed_updates > thermalization` and when
`trajectory_id % every == 0`. Flow and output use their own positive intervals
under the same post-update condition. Stochastic chiral measurement consumes
the same explicit RNG stream in schedule order, matching Julia's single-stream
ownership while making the effect visible in the report ordering.

Failures stop immediately. Lower update functions retain their transactional
link semantics; RNG draws already consumed are not rolled back. `RunFailure`
contains the typed source, trajectory/flow/measurement context, and the partial
`RunReport` so completed work is not lost.

Output is no-clobber: write a unique same-directory temporary ILDG file, publish
with `hard_link` (which fails atomically if the destination exists), then remove
the temporary link. Failure performs best-effort temporary cleanup. The runner
never overwrites an existing configuration.

The report contains requested/completed update counts, accept/reject counts,
per-trajectory update outcome, ordered measurement records, flow records, and
published paths. It does not claim resumability or final RNG-state export.

### Binary and examples

The binary accepts exactly one TOML path, parses and executes it, prints a short
report, and exits nonzero with the typed error chain. No `clap` dependency.
Commit one cold quenched example using a tiny even lattice, one update, known
measurements, and no output; CI runs it and checks its deterministic report.

Tests cover every variant, strict unknown-field rejection, invalid combinations,
validation-before-side-effect, scheduling boundaries, HMC reject/error partial
reports, deterministic RNG position, cold and ILDG starts, no-clobber output,
and all public report records.

## Task C: integration evidence

Add a short independent Julia/Rust ensemble comparison for the corrected
staggered pion correlator and chiral condensate. Predeclared schedule:
`[2,2,2,2]`, beta `5.7`, mass `0.5`, Nf `2`, fixed Phase 3 RHMC tables/bounds,
step size `0.01`, two MD steps, four thermalization trajectories, then sixteen
measurements at interval one, grouped into four consecutive blocks of four.
Julia and Rust use independent recorded canonical-Z4/update streams and do not
share configurations. For each pion timeslice and the chiral scalar, require

```text
abs(mean_rust - mean_julia)
    <= 6 * sqrt(se_rust^2 + se_julia^2)
```

using standard errors of the four block means. Record all per-configuration
values, block means, stream provenance, acceptance, and delta-H metadata. Phase
3's chiral ensemble remains supporting evidence but does not replace this Phase
4 run.

Update README/rustdoc, a checked frontend example, fixture provenance, and a
reviewer-facing worklog. Regenerate the deterministic fixture twice and run a
fresh clean local gate before the final review and PR.

## Review and completion gates

Each task requires recorded pre-implementation `Correct-to-merge`, then a
read-only full-diff review by `reviewer-flash`, fixes, and delta re-review before
the next task. The integrated diff receives another full review.

Final local gates:

```text
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo doc --workspace --all-features --no-deps
cargo run -p gaugefields --example traced_wilson_action --all-features
cargo run -p dirac-operators --example fermions --all-features
cargo run -p latticeqcd -- examples/phase4.toml
```

Also run every other checked example, `git diff --check`, exact tenferro-pin,
stale-symbol, license, provenance, fixture-tree, and generated-artifact audits.
The Phase 4 PR is merged only after hosted CI succeeds; post-merge main CI must
succeed before Issue #17 Phase 4 boxes are checked.
