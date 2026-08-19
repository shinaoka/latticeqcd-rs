# Phase 3 fermions worklog

Status: Tasks A-E and integrated review complete; PR pending

## Base and integration state

- Phase 2 PR: <https://github.com/shinaoka/latticeqcd-rs/pull/20>
- Phase 2 merge commit: `eb66f34d5fc25da2f163f1005a8c632492572d28`
- post-merge CI: <https://github.com/shinaoka/latticeqcd-rs/actions/runs/32171068120> (`success`)
- Phase 3 branch: `feat/phase3-fermions`, fast-forwarded to the merge commit
- Issue #17 Phase 2: all six evidence-backed checklist entries updated to complete

## Pinned references

- LatticeDiracOperators.jl v0.6.4:
  `bdef628184597815ba3e0cddf2536df767e78a02`
- Gaugefields.jl v0.7.2:
  `9e5719970770f4497405a856315c90bef7f74449`
- Wilsonloop.jl v0.1.5 source:
  `e1a617fdedb19b785f89bdeb13c30e53b20743a7`

v0.6.4 is the newest tagged LatticeDiracOperators.jl release compatible with
Gaugefields 0.7. LatticeDiracOperators.jl v1 requires the incompatible
Gaugefields/LatticeMatrices v1 backend and is not used as the numerical oracle.

## Contract reconnaissance

The active v0.6.4 Wilson, CG/BiCGStab, Wilson action/force, one-link staggered,
shifted-CG, and RHMC paths were read before design. The design fixes:

- Julia's Euclidean chiral gamma basis and hopping-normalized Wilson operator,
- color-fast logical fields transposed into Rust's site-contiguous
  `[3,components,NX,NY,NZ,NT]` layout,
- default `[+1,+1,+1,-1]` fermion boundary signs,
- absolute squared solver tolerances plus fresh true-residual verification,
- two Wilson flavors and two-flavor staggered RHMC,
- caller-owned RNG/context and transactional U-P-U HMC,
- the exact Gaugefields zero-based staggered phases,
- the v0.6.4 `[0.0004,64]` rational-coefficient interval.

The Rust API does not reproduce Julia string-key fallback, global RNG,
incomplete force paths, or unsupported factor conventions.

## Reference-defect issues

Previously known Phase 1/2 defects and Phase 3 findings are now durable
`latticeqcd-rs` issues:

- #21 Gaugefields.jl heatbath RNG/normalization/degenerate-input defects
- #22 incomplete Gaugefields.jl ILDG writer
- #23 ignored staggered `eps_CG`
- #24 Wilson `factor` missing from pseudofermion force
- #25 Wilson-clover force multiplied by zero
- #26 negative one-eighth RHMC coefficient inversion defect

New confirmed defects found during implementation will be filed with an exact
revision, source link, reproducer or direct code proof, and Rust-side handling.
Suspicions are reproduced before being filed as facts.

## Review gate

| Task | Design | Pre-review | Post-review |
|---|---|---|---|
| A: field and Wilson operator | `docs/design/phase-3-fermions.md` Task A | Correct-to-merge | Correct-to-merge |
| B: CG and BiCGStab | same, Task B | Correct-to-merge | Correct-to-merge |
| C: Wilson pseudofermion/HMC | same, Task C | Correct-to-merge | Correct-to-merge |
| D: staggered/multi-shift CG | same, Task D | Correct-to-merge | Correct-to-merge |
| E: two-flavor RHMC/integration | same, Task E | Correct-to-merge | Correct-to-merge |
| Integrated Phase 3 | full design | Correct-to-merge | Correct-to-merge |

`reviewer-flash` completed the pre-implementation review and recorded an
overall `Correct-to-merge`, with separate `Correct-to-merge` verdicts for Tasks
A--E. Three Minor documentation findings were folded in before implementation:

- gauge and fermion momentum-force scaling now states the distinct Julia
  `1/NC` treatment,
- the chiral condensate now uses the exact pinned QCDMeasurements.jl stochastic
  estimator `(Nf/4) Re[r†D^-1r]/NV`, without an extra `1/NC`,
- Julia fixture modes now state explicit-array fixed trajectories and explicitly
  seeded ensemble RNG provenance.

The reviewer requested no further design round after these Minor clarifications.

After Task A started, the user required the Rust implementation to remain as
parallel as practical to Julia for direct comparison. The design now maps Rust
modules to Julia source files and requires matching kernel decomposition,
recurrence/intermediate names, source citations, and fixture entrypoint maps,
while retaining Rust safety and ownership contracts. `reviewer-flash` gave the
focused design delta `Correct-to-merge`. Its required conformance action renamed
the uncommitted `operator.rs` draft to the Julia-parallel `wilson.rs`; Minor
clarifications added full Julia source paths and mapped `wilson_hmc.rs` to the
pinned Julia HMC driver and force update.

The first Task A executor timed out before verification. It left an uncommitted
crate/generator/fixture draft; the last reported blocker was a Clippy
`needless_range_loop` finding in the file now named `wilson.rs`. This continuation
audited that draft, fixed the lint and normal-operator transactionality, and
completed the evidence below. Task A subsequently received `Correct-to-merge`.

## Task A completion evidence

### Delivered scope

- Added the `dirac-operators` workspace crate with private, validated host
  `FermionField` storage, `FermionBoundary`, typed `DiracError`, the
  `FermionOperator` trait, borrowed `WilsonDirac`, `WilsonAdjoint`, and
  composed `NormalOperator` (`D†D`).
- Wilson fields accept exactly `[3,4,NX,NY,NZ,NT]`; construction checks rank,
  shape, Complex64 dtype, host placement, finite values, and checked sizes.
  Operator construction checks finite positive `kappa`, `r == 1`, finite gauge
  links, and validated boundary signs. There is one `HostGaugeLinks` view per
  operator and no public raw storage accessor or production panic path.
- `Wx!`/`Wdagx_noclover!` hop ordering, `rminusγ`/`rplusγ` projectors,
  `shift_fermion` boundary wrapping, and `DdagD_Wilson_operator` composition
  remain directly comparable to the pinned Julia decomposition. Scratch output
  is committed only after complete success; the new numerical-range regression
  test covers a failure during the second `D†D` stencil.
- Fixed the reported `clippy::needless_range_loop` in `wilson.rs` with direct
  iterator/enumerate traversal while retaining the explicit Julia layout
  citation. No solver, action, force, HMC, staggered, or RHMC code was added.

### Julia fixture generation and residuals

The generator was run twice with `JULIA_NUM_THREADS=1`,
`LATTICEQCD_JULIA_PROJECT=/tmp/latticeqcd-phase3-julia-env`,
`GAUGEFIELDS_JL_DIR=/home/shinaoka/tensor4all/Gaugefields.jl`,
`LATTICEDIRACOPERATORS_JL_DIR=/home/shinaoka/tensor4all/LatticeDiracOperators.jl`,
and `WILSONLOOP_JL_DIR=/home/shinaoka/tensor4all/Wilsonloop.jl`:

```text
julia --startup-file=no --project=/tmp/latticeqcd-phase3-julia-env fixtures/generate.jl fermions_task_a
```

Both runs produced 19 files and the complete-tree hashes matched:

```text
run 1: a865e5f444062b11025ed0cecc5481ed9dc0a2cf522fbc13cac785c492ed9b3b
run 2: a865e5f444062b11025ed0cecc5481ed9dc0a2cf522fbc13cac785c492ed9b3b
```

The fixture records Gaugefields.jl
`9e5719970770f4497405a856315c90bef7f74449`, LatticeDiracOperators.jl
`bdef628184597815ba3e0cddf2536df767e78a02`, full source paths/functions,
entrypoint mappings, and the exact Julia-to-Rust permutation
`(1,6,2,3,4,5)` from `[3,NX,NY,NZ,NT,4]` to `[3,4,NX,NY,NZ,NT]`.
The generated Julia/Rust layout-converted comparisons had maximum residuals:

```text
D periodic       4.51828035988302737e-16
D† periodic      5.85504509035522313e-16
D†D periodic     5.67506506526714226e-16
D antiperiodic   5.55111512312578270e-16
D† antiperiodic  6.10622663543836097e-16
D†D antiperiodic 5.55804968563355581e-16
```

Independent focused checks reported cold impulse/projector residual `0`,
adjoint inner-product residual `6.35528743231301918e-14`, gamma5 residual `0`,
and cold plane-wave residual `9.50197851991224090e-16`; all are below
`2e-12`.

### Verification from the dedicated target

All commands below used
`CARGO_TARGET_DIR=/home/shinaoka/tensor4all/latticeqcd-rs-phase3/target`
(the workspace keeps `debug = 0` for dev/test profiles):

```text
cargo fmt --all                                      passed
cargo fmt --all -- --check                           passed
cargo check --workspace                              passed; 4 crates
cargo test -p dirac-operators --tests -- --nocapture passed; 12 tests
cargo test --workspace                               passed; 204 passed, 1 ignored, 44 suites
cargo test --workspace --all-features                passed; 214 passed, 1 ignored, 44 suites
cargo test --doc --workspace --all-features          passed; 45 doctests, 4 suites
cargo clippy --workspace --all-targets --all-features -- -D warnings
                                                       passed; no issues
cargo doc --workspace --all-features --no-deps       passed; 4 crates documented
git diff --check                                     passed
```

Every existing example was run with `--all-features`; the exact commands were:

```text
cargo run -p gaugefields --example ildg_roundtrip --all-features
cargo run -p gaugefields --example quenched_heatbath --all-features
cargo run -p gaugefields --example quenched_hmc --all-features
cargo run -p gaugefields --example traced_wilson_action --all-features
cargo run -p measurements --example quenched_measurements --all-features
```

All five passed (`ildg_roundtrip`, `quenched_heatbath`, `quenched_hmc`,
`traced_wilson_action`, and `quenched_measurements`). Temporary comparison
copies under `/tmp/latticeqcd-phase3-fermions-run1` and `run2` were removed;
the required `fixtures/fermions_task_a` and the dedicated Cargo target remain.

### Contract audits

- Exact tenferro pin check: 5 manifest `rev` declarations and 9 matching
  lockfile source lines, all at
  `c942129974b544225ed963414d7be1300980f901`.
- Stale-symbol check passed: no `operator.rs` path or symbol remains in the
  Task A crate; the approved historical rename is retained only in this log.
- License/provenance check passed: all four workspace crates have MIT notices,
  the new crate preserves the pinned Julia copyright notice, reference
  checkouts were clean at their exact revisions, and every Task A numerical
  source has a function/revision citation in Rust or fixture metadata.
- Task-A scope check passed: 7 crate source files, 4 focused test files, and
  no solver/action/force/HMC/staggered/RHMC symbols. No commit, push, PR, or
  sibling/reference checkout was modified.

### Post-implementation review

`reviewer-flash` completed a full-diff review and recorded
`Correct-to-merge` with no Critical or Important finding. Two new Minor
findings were fixed:

- the fixture integration test now parses `metadata.json` and asserts schema,
  lattice, NC, components, kappa, r, boundaries, both pinned commits, layout
  permutation, nine-entry Julia/Rust entrypoint map, tolerance, and all 18
  declared payload files;
- the redundant full-lattice `preflight_neighbors` pass was removed from each
  application. Constructor validation remains once per borrowed link snapshot,
  while the Julia-parallel hot stencil keeps checked neighbor/link access and
  transactional buffered output.

After the fixes, all 12 crate tests plus 6 doctests, focused
all-target/all-feature Clippy, formatting, and `git diff --check` passed. The
delta review confirmed both fixes and recorded `Correct-to-merge`; its one new
Minor count-clarity finding is corrected here from ambiguous "18 focused tests"
to the measured split above.

## Task B completion evidence

### Sources, RED-first fix, and mapping

Task B was implemented in `crates/dirac-operators/src/solvers.rs`, with the
minimal checked-algebra seams in `src/field.rs`, the
`HermitianPositiveOperator` marker and reusable operator workspace in
`src/wilson.rs`, and typed `SolverError` variants in `src/error.rs`. The Rust
solver cites and follows the pinned LatticeDiracOperators.jl v0.6.4
`src/cgmethods.jl` at `bdef628184597815ba3e0cddf2536df767e78a02`:
`cg` lines 768–868 maps to `conjugate_gradient`, and `bicgstab` lines 157–310
maps to `bicgstab`. The recurrence names `r`, `p`, `Ap`, `s`, `t`, `alpha`,
`beta`, and `omega`, including shadow restart and update ordering, remain
visible in the Rust implementation. Julia's global temporary pool, panic
paths, unchecked divisions, and silent non-finite values were not copied.

RED-first verification added `solver_reuses_operator_scratch_across_iterations`;
the focused command first failed because the draft had no reusable operator
workspace method (`apply_into_with_scratch`). The smallest fix added the
caller-owned workspace seam, made Wilson and `D†D` consume it transactionally,
and the same test then passed. Solver-local fields and the two normal-operator
workspace fields are allocated once per solve; the initial guess is committed
only after a fresh true-residual check. `SolverReport` records method,
iterations, initial/recursive/true residual squared, tolerance, maximum
iterations, restart count, and convergence branch.

### Julia fixture and parity

`fixtures/generate.jl fermions_task_b` was audited and run with explicit
formula-generated diagonal SU(3) links, rhs, and zero/nonzero guesses; it uses
no RNG or hidden state. Metadata records all Julia constructor keys
(`Dirac_operator`, `κ`, `r`, `faster version`, `verbose_level`,
`boundarycondition`, `method_CG`, `eps_CG`, `MaxCGstep`), solver keywords
(`eps`, `maxsteps`, `verbose`), layout conversion, source URLs/functions,
pins, tolerances, entrypoint mappings, and all 18 payload files. The Rust
fixture integration test parses every metadata field, checks the complete
payload tree and shapes, and independently recomputes `sum(abs2, rhs - A*x)`.

The generator ran twice with Julia 1.12.5 in the clean external project
`/tmp/latticeqcd-phase3-julia-env`, with explicit pinned checkouts:
Gaugefields.jl `9e5719970770f4497405a856315c90bef7f74449` and
LatticeDiracOperators.jl `bdef628184597815ba3e0cddf2536df767e78a02`.
Complete-tree hashes matched:

```text
run 1: 5a20feddcbb04282e4c840ef20c82a8280dbffd4f18616a9cc203a5418760610
run 2: 5a20feddcbb04282e4c840ef20c82a8280dbffd4f18616a9cc203a5418760610
```

The focused fixture test reported four cases (CG/BiCGStab × zero/nonzero).
Maximum Julia/Rust solution residuals were, in case order
`cg_zero`, `cg_nonzero`, `bicgstab_zero`, `bicgstab_nonzero`,
`1.58882185807825480e-14`, `1.81153563744117622e-14`,
`1.60855953235337831e-14`, and `1.63772008528668903e-14`. Independently
recomputed Rust true relative residuals were
`1.00972069793319903e-12`, `1.17941961034077306e-12`,
`1.18441250224484603e-12`, and `1.02620361311686820e-12`; all are below the
`2e-11` and `1e-11` acceptance thresholds. Rust iterations were 29/29 for CG
and 32/32 for BiCGStab, with zero fixture restarts.

### Focused error and resource checks

The six solver unit tests cover initial convergence, zero/nonzero guesses,
exhaustion, non-finite intermediates, denominator breakdown, successful and
singular shadow restart, stagnation, wrong lattice/components, transactional
output, and stable recurrence/operator scratch destinations. Fixed linear test
operators, not call-count hooks, exercise the restart branches. The focused
`cargo test -p dirac-operators --tests -- --nocapture` run passed 21 tests:
9 crate unit tests, 8 Task A integration tests, 2 Task B integration tests,
and one fixture test in each task. Focused all-target/all-feature Clippy passed
with `-D warnings`.

### Workspace and audit gates

The exact local gates passed from the dedicated worktree target:

```text
cargo fmt --all                                      PASS
cargo fmt --all -- --check                           PASS
cargo check --workspace                              PASS
cargo test --workspace                               PASS (221 passed, 1 ignored)
cargo test --workspace --all-features                PASS (231 passed, 1 ignored)
cargo test --doc --workspace --all-features          PASS (53 passed)
cargo clippy --workspace --all-targets --all-features -- -D warnings
                                                       PASS
cargo doc --workspace --all-features --no-deps       PASS
```

All five existing examples passed with `--all-features`: `ildg_roundtrip`,
`quenched_heatbath`, `quenched_hmc`, `traced_wilson_action`, and
`quenched_measurements`. `git diff --check` passed. The exact tenferro pin
check found five manifest declarations and nine matching lockfile source
lines for `c942129974b544225ed963414d7be1300980f901`; stale-symbol, license,
provenance, and Task B scope checks passed. The three pinned Julia reference
checkouts remained clean, and no temporary comparison artifacts were created.

Task B implementation is complete; independent post-review found no production
code defect and recorded `Correct-to-merge`. No commit, push, PR, issue,
reference checkout, or Task C/D/E implementation was made. Remaining risk is
limited to the mathematical contract of caller-defined
`HermitianPositiveOperator` implementations and the documented default
workspace fallback for custom operators; the built-in Wilson and normal paths
use the preallocated scratch path verified above.

### Task B post-review findings

`reviewer-flash` reported no Critical or Important finding. Three Minor test
coverage findings were fixed without changing the Julia-parallel production
recurrences:

- a fixed identity operator now forces and asserts BiCGStab's
  `IntermediateResidual` branch, solution update, and true residual;
- a deliberately call-varying test operator now forces
  `TrueResidualMismatch` and verifies bitwise transactional output;
- the fixed restart operator now produces a tiny nonzero shadow product, while
  direct boundary assertions pin both sides of the `epsilon * scale` restart
  threshold.

After the fixes, 23 crate tests plus 14 doctests, focused all-target/all-feature
Clippy, formatting, and `git diff --check` passed. The delta re-review confirmed
all three fixes, found no remaining issue, and recorded `Correct-to-merge`.

## Task C completion evidence

### Julia-parallel action, force, and HMC

Task C adds `wilson_action.rs`, parallel to pinned
`src/action/WilsonFermiAction.jl`, and `wilson_hmc.rs`, parallel to
`test/wilsonhmc.jl`. The implementation retains `X=(D†D)^-1 phi`, `Y=D X`,
the two Julia outer-product force terms, and the U-P-U sequence. Momentum uses
the reviewed split: gauge force `-dt/NC`, fermion force `-dt`. HMC consumes one
unconditional Metropolis draw, commits links only on acceptance, and leaves
links unchanged on trajectory failure or rejection; consumed RNG is not rolled
back.

The complex Gaussian sampler consumes one Box--Muller pair per complex
component and scales both real and imaginary parts by `1/sqrt(2)`. Fixed
fixture fields use explicit arrays rather than RNG.

### Deterministic fixture and force checks

`fixtures/fermions_task_c` contains 28 declared payloads plus metadata. Metadata
records both pinned revisions, Julia/Rust entrypoint mappings, layout, solver,
force projection, gauge/fermion scaling, explicit acceptance Xoshiro state, and
every tolerance. The focused generator was run twice from the clean external
Julia project; file-by-file comparison was empty and both complete 29-file tree
hashes were:

```text
run 1: 9462c1e4bf1f46c0929c81fd932f65dbd20f2a2b65168bb65ad8e8a4d92439af
run 2: 9462c1e4bf1f46c0929c81fd932f65dbd20f2a2b65168bb65ad8e8a4d92439af
```

The Rust comparison reported maximum residuals:

```text
action                       4.55e-13
X                            7.16e-15
Y                            6.62e-15
force                        5.11e-14
initial Hamiltonian          4.55e-13
proposed Hamiltonian         4.55e-12
delta-H                      5.23e-12
final momentum               4.12e-13
proposed links               2.89e-15
```

All 512 force coefficients were checked independently by central differences
under `U <- exp(epsilon*T_a)U`. For epsilon `[1e-3,5e-4,2.5e-4]`, maximum
residuals were `[5.00524e-7,1.26267e-7,3.46349e-8]`, with successive ratios
about `[3.964,3.646]`. The quadratic truncation region is explicit and all 512
coefficients pass the selected `5e-4` step below `2e-7`.

### Transaction and verification gates

Eight Task C integration tests cover refresh/action, the public HMC surface,
rejection rollback with acceptance-word consumption, trajectory-error rollback,
reversibility, fixture action/force/trajectory/RNG parity, all-coefficient
finite differences, and complex-normal stream scaling. The combined focused
run passed 5 `task_c` and 3 `task_c_fixture` tests.

The full workspace gates passed on the Task C tree:

```text
cargo fmt --all -- --check                         PASS
cargo check --workspace                            PASS
cargo test --workspace                             PASS (234 passed, 1 ignored)
cargo test --workspace --all-features              PASS (244 passed, 1 ignored)
cargo test --doc --workspace --all-features        PASS (56 passed)
cargo clippy --workspace --all-targets --all-features -- -D warnings
                                                     PASS
cargo doc --workspace --all-features --no-deps     PASS
```

All five existing examples and diff, exact-pin, stale-symbol, license,
provenance, scope, and clean-reference-checkout audits passed. Task D/E code
was not added.

### Task C post-review findings

`reviewer-flash` found no production-code defect but required three Minor
corrections:

- stale pre-final fixture hashes in README, crate rustdoc, and design now use
  the twice-regenerated final hash `9462c1e4...d92439af`;
- the public full-HMC smoke test now deterministically asserts acceptance and
  bitwise link replacement, complementing the existing rejection rollback;
- reversibility now uses an explicit nonzero pseudofermion and asserts its
  fermion force is nonzero before the forward/reverse trajectory.

The five Task C public integration tests pass after these corrections. The
focused delta re-review found no remaining issue and recorded
`Correct-to-merge`.

## Task D finalization evidence

### Julia-parallel mapping and scope

Task D is implementation-complete and independently post-reviewed.
The Rust path stays parallel to the pinned LatticeDiracOperators.jl v0.6.4
sources: `Staggered_Dirac_operator`/`Dx!` map to `StaggeredDirac`, its adjoint
maps to `StaggeredAdjoint`, `DdagD_Staggered_operator` maps to
`StaggeredNormalOperator`, the independently lowered `mass^2 I-K^2` check maps
to `StaggeredClosedNormalOperator`, and `Dirac_operators.shiftedcg` maps to
`multi_shift_cg`. The recurrence retains Julia's `r`, `p`, `q`, `alpha`,
`beta`, `rho_m`, `rho_0`, and `rho_p` order. Zero-based eta phases are
`[1,(-1)^x,(-1)^(x+y),(-1)^(x+y+z)]`; a fermion boundary sign is applied once
when a hop wraps.

No Task E coefficient, action, or RHMC implementation symbols were added.
The Task D scope is limited to `staggered.rs`, the shifted-CG additions and
errors needed by `solvers.rs`, the fixture generator, focused tests, and
metadata/documentation.

### Fixture audit and exact values

The fixture uses lattice `[2,2,2,2]`, one component, mass `0.17`, shifts
`[0.31, 0.0, 0.07]`, absolute squared solver tolerance `1e-24`, and maximum
iterations `2000`. The operator, anti-Hermiticity, and normal-composition
comparison tolerance is `2e-12`; the fresh shifted true-relative residual
criterion is `1e-11`. These tolerances are intentionally distinct: `2e-12`
compares operator components and algebraic identities, while `1e-11` checks
fresh relative residuals for each shifted solve; `1e-24` is the solver's
absolute squared stopping/true-residual gate.

`metadata.json` declares 37 payloads and the fixture tree contains 38 files
including metadata. The integration test consumes every declared payload,
including all `u*`, input/rhs, eta, D, Ddag, K, normal-composition,
normal-closed, and shifted solution arrays, and consumes every metadata and
shifted-report field. The two clean Julia 1.12.5 generations used
`LATTICEQCD_JULIA_PROJECT=/tmp/latticeqcd-phase3-julia-env`,
`JULIA_NUM_THREADS=1`, Gaugefields.jl
`9e5719970770f4497405a856315c90bef7f74449`, LatticeDiracOperators.jl
`bdef628184597815ba3e0cddf2536df767e78a02`, and Wilsonloop.jl
`e1a617fdedb19b785f89bdeb13c30e53b20743a7` from clean checkouts. Both
complete-tree hashes were
`c372e6e56bc05ebc611c6cc3dba5c247eafbc12ca58a0eee2ac3737cdbb08d4b`; the
file count was 38 for each run and the complete directory comparison was
empty.

The fixture's D, Ddag, and K payload parity is bit-exact; eta signs and the
boundary-wrap impulse payload are exact. Normal-composition residuals are
`3.19867204157556452e-17` (periodic) and
`1.66533453693773481e-16` (default anti-periodic). The generated
`normal_closed` payload is bit-exact with its recorded oracle; the meaningful
closed-form-versus-composition residuals are the two nonzero values above. K
anti-Hermiticity is `2.48253415324727312e-16`. The focused cold
checks additionally report adjoint residual `4.44089209850062616e-16`, cold K
anti-Hermiticity `6.93889390390722838e-17`, and closed-normal residual
`7.10542735760100186e-15`.

The Julia shifted reports have initial residual squared
`2.99675440000000037e1`; all shifts converge in 8 iterations on
`updated_residual`. In shift order `0.31`, `0.0`, `0.07`, the exact Julia
recursive/true residual-squared pairs are:

```text
0.31: 1.77459964884789642e-27 / 1.47868086305190983e-30
0.0:  1.19636782643941803e-25 / 7.62807887898313613e-30
0.07: 4.17668148129519829e-26 / 4.43903981763168336e-30
```

Rust recursive/true residual-squared pairs are
`(1.77459964884795812e-27, 1.91374843748898659e-30)`,
`(1.19636782643945936e-25, 8.21640232874482560e-30)`, and
`(4.17668148129534236e-26, 4.17923186813384454e-30)`; differences from the
Julia reports are all below the absolute `1e-24` gate. Rust fresh true
relative residuals are
`2.52706753667624838e-16`, `5.23618850174067806e-16`, and
`3.73441567789983714e-16`.

### Verification and remaining risk

The RED-first metadata audit initially rejected exact true-residual parity
because the Rust and Julia independently recomputed residual squares differ
at approximately `1e-30`, far below the absolute `1e-24` solver gate. The test
now consumes and reports that field while applying the documented absolute
solver tolerance, without weakening the relative solve criterion.

The focused Task D command passed 3 kernel/validation tests and 1 fixture test.
The combined Task A-C regression command and the required workspace gates are
recorded below after they run. No commit, push, issue, branch, or pinned
reference checkout was modified; the fixture and dedicated `target/` are kept.

### Final verification record

The combined focused command
`CARGO_TARGET_DIR=/home/shinaoka/tensor4all/latticeqcd-rs-phase3/target cargo test -p dirac-operators --tests`
passed 36 tests: 12 unit tests, Task A 9 tests, Task B 3 tests, Task C 8
tests, and Task D 4 tests. The first run was stopped by the 120-second shell
timeout while Task C's all-coefficient finite-difference test was still
running; the exact command was rerun with a 600-second timeout and passed,
including that test's 136.64 seconds.

The requested workspace gates passed from the dedicated target:

```text
cargo fmt --all                                      PASS
cargo fmt --all -- --check                           PASS
cargo check --workspace                              PASS
cargo test --workspace                               PASS (245 passed, 1 ignored)
cargo clippy --workspace --all-targets --all-features -- -D warnings
                                                       PASS
cargo test --workspace --all-features                PASS (255 passed, 1 ignored)
cargo test --doc --workspace --all-features          PASS (62 passed)
cargo doc --workspace --all-features --no-deps       PASS
```

All five examples passed with `--all-features`: `ildg_roundtrip`,
`quenched_heatbath`, `quenched_hmc`, `traced_wilson_action`, and
`quenched_measurements`. `git diff --check` passed. The exact tenferro pin
check found five manifest declarations and nine matching lockfile source lines
for `c942129974b544225ed963414d7be1300980f901`. The stale-symbol, MIT license,
Julia provenance, and scope checks passed; the Task D generator slice and
staggered tests contain no Task E coefficient/action/RHMC symbols. Temporary
comparison directories were removed, while `fixtures/fermions_task_d` and the
dedicated `target/` remain. No commit, push, issue, or reference checkout was
modified.

### Task D post-review findings

`reviewer-flash` recorded `Correct-to-merge` with no Critical or Important
finding. Three Minor findings were fixed:

- the component-one Julia fixture helper is now named
  `julia_one_component_field` and explicitly documents why the two
  column-major payloads are identical instead of pretending to transpose;
- normal evidence distinguishes bit-exact recorded payload parity from the
  measured closed-form-versus-composition residuals;
- backward eta coordinates are derived from the checked `minus_site`, removing
  a second hand-written wrap calculation.

Task D focused tests, Clippy, formatting, and diff-check pass after the fixes.
The delta re-review found no remaining issue and recorded `Correct-to-merge`.

## Task E deterministic finalization evidence

The deterministic RHMC slice and independent short Julia/Rust ensemble are
implemented, verified, and independently post-reviewed.

The exact pinned Nf=2 tables are represented with `f64::from_bits`: degree 15
`x^(+1/8)` refresh, degree 15 `x^(-1/8)` action, and degree 10 `x^(-1/4)`
MD-force inverse residues on `[0.0004, 64]`. Unit tests assert every alpha and
beta bit, while the fixture test consumes the complete metadata table. The
4097-point logarithmic-grid maximum absolute errors are:

```text
refresh  2.505791796281187e-9
action   3.9620045022559225e-9
MD force 1.5595609319518644e-5
```

The public deterministic refresh path applies the pinned rational to an
explicit `xi`; action evaluates `X=R_(-1/8)(M)`; force compares every
Julia-parallel `X_j=(M+beta_j I)^-1 phi` and `Y_j=D X_j`. Report metadata,
field payloads, source/layout/provenance metadata, and all declared payload
names are asserted. RHMC U-P-U uses private link/momentum proposals, validates
before mutation, rolls back on validation and post-update solver errors, rolls
back links on rejection, and leaves consumed RNG words consumed. Focused tests
cover rejection rollback/RNG advancement and forward/reverse reversibility.

The all-512 central force FD contract is preserved exactly:

```text
epsilons    [0.32, 0.16, 0.08, 0.04]
max errors  [8.434653210321642e-6, 2.139177378187619e-6,
             5.605769951367093e-7, 1.6563038083509257e-7]
ratios      [3.9429424115674574, 3.816027765580949, 3.384505863660601]
pass count  [291, 442, 510, 512]
selected    0.04; 512/512 below 5e-7
```

`fixtures/fermions_task_e` contains 68 files (67 declared payloads plus
metadata). Two clean Julia 1.12.5 generations were byte-identical, with
complete-tree hash
`9e166b37d2c138a28f6d75395e11dc8f91f910599f0397e5888bbd738ba6d34a`.

### Task E verification gates

- Focused deterministic Task E tests, including all-512 force FD:
  `cargo test -p dirac-operators --test task_e --test task_e_fixture --
  --nocapture` — PASS, 7 tests.
- `cargo fmt --all` and `cargo fmt --all -- --check` — PASS.
- `cargo check --workspace` — PASS.
- `cargo test --workspace` — PASS, 256 passed and 1 ignored after the
  ensemble integration test.
- `cargo test --workspace --all-features` — PASS, 266 passed and 1 ignored.
- `cargo test --doc --workspace --all-features` — PASS, 63 doctests.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — PASS.
- `cargo doc --workspace --all-features --no-deps` — PASS.
- All five existing examples with `--all-features` — PASS.
- `git diff --check`, exact five-manifest/nine-lockline tenferro pin,
  stale-symbol/scope, four MIT license files, clean three-reference-checkout
  provenance, and complete fixture metadata/file audits — PASS.

### Independent Julia/Rust ensemble

The metadata-only Julia fixture `fixtures/fermions_task_e_ensemble` was generated
twice in 15.38 and 15.28 seconds with byte-identical SHA-256
`1467768f1798f8b9662d0419bf49009bf64921cc97b9581968d36f3c1c53d2cf`.
Both languages use the same cold `2^4`, beta `5.7`, mass `0.17`, Nf=2,
`dt=0.001`, two-step schedule, four burn-in trajectories, then three blocks of
four measured trajectories. Their HMC and measurement RNG streams are explicit
and independent.

Julia block results were:

```text
plaquette [0.9999310091, 0.9998880612, 0.9998363070]
           mean 0.9998851258, SE 2.73775e-5
chiral    [0.2478494979, 0.2476694079, 0.2477546679]
           mean 0.2477578579, SE 5.20120e-5
mean delta-H -1.59564e-8; acceptance 12/12
```

Rust used a separately owned canonical Z4 source stream and produced:

```text
plaquette [0.9999326182, 0.9998963669, 0.9998438435]
           mean 0.9998909429, SE 2.57702e-5
chiral    [0.2477514929, 0.2479892078, 0.2473041694]
           mean 0.2476816234, SE 2.00816e-4
mean delta-H -4.36219e-8; acceptance 12/12
```

The differences are `0.154717` combined standard errors for plaquette and
`0.367498` for chiral, both below the fixed six-combined-SE criterion. The
Rust ensemble integration test runs in about 1.03 seconds and passed in both
final workspace matrices. Mean
delta-H and acceptance are reported without pass thresholds. The estimator is
`(Nf/4) Re(r†D^-1r)/NV`; metadata records all schedule, block, source, seed,
normalization, provenance, and per-measurement values.

The pinned Julia `Z4_distribution_fermi!` uses `theta=k*pi/4` instead of the
canonical `k*pi/2` grid. This confirmed upstream defect is recorded as Rust
Issue #27. Neither implementation copies it for this comparison: both use
canonical Z4 with independently seeded streams, and Rust exposes no
compatibility API.

Known risks are limited to the caller's spectral-bound assertion and the
pinned rational approximation outside the checked scalar grid; no runtime
spectrum estimator or coefficient generator was added. No commit, push, branch,
or reference checkout was modified.

## Integrated Phase 3 verification

The final branch is based directly on Phase 2 merge commit
`eb66f34d5fc25da2f163f1005a8c632492572d28` and is 0 behind / 6 commits ahead
of `origin/main` before the final integration-evidence commit.

Every Phase 3 Julia mode was regenerated twice from clean pinned checkouts.
Using one common sorted file-content aggregation command, both runs matched:

```text
fermions_task_a          a865e5f444062b11025ed0cecc5481ed9dc0a2cf522fbc13cac785c492ed9b3b
fermions_task_b          91aa615fb22d1558224e469e12a713315b840edf14d7d037e44dd0239c155d14
fermions_task_c          9462c1e4bf1f46c0929c81fd932f65dbd20f2a2b65168bb65ad8e8a4d92439af
fermions_task_d          baafa16e2cbdf0d9de96c027566886f27918cdff67a73a052bcb48323a79d985
fermions_task_e          14179c53f20d68fb473dd4e127a42a54ce601c92784a8b567aabd8cce5b3c468
fermions_task_e_ensemble 0c50e60697acefff3d6f4fc98d2a660b07826635c596dd899986e4caedf9daae
```

These integration hashes use the same command for all modes; earlier task-local
hashes used their recorded task-local aggregation commands. File-by-file
comparisons are empty in every mode.

After deleting 20.4 GiB of prior target artifacts, the complete final tree was
rebuilt from an empty target. Fresh results:

```text
cargo fmt --all -- --check                         PASS
cargo check --workspace                            PASS
cargo test --workspace                             PASS (256 passed, 1 ignored)
cargo test --workspace --all-features              PASS (266 passed, 1 ignored)
cargo test --doc --workspace --all-features        PASS (63 passed)
cargo clippy --workspace --all-targets --all-features -- -D warnings
                                                     PASS
cargo doc --workspace --all-features --no-deps     PASS
```

Six examples ran successfully with `--all-features`, including the new
`dirac-operators/examples/fermions.rs` Wilson/staggered/RHMC smoke example.
`git diff --check`, exact tenferro pin (five manifest declarations and nine
lockfile source lines), stale-symbol, unsafe, dependency, MIT license,
provenance, known-defect issue, scope, and clean-reference-checkout audits
passed. `rand_chacha` is absent. Rebuilt target size is 18 GiB and is removed
after hosted verification because it is entirely reproducible.

The integrated full-diff review found no code defect and recorded
`Correct-to-merge`. Hosted PR CI remains pending.

### Task E post-review findings

`reviewer-flash` recorded `Correct-to-merge` with no production defect and two
Minor findings. Both are fixed:

- the Julia ensemble now avoids the biased pinned pi/4 source and implements
  canonical Z4 explicitly. Julia and Rust therefore estimate the same chiral
  quantity with independent streams; the regenerated fixture hash is
  `1467768f...c53d2cf` and the chiral difference is `0.367498` combined SE;
- scalar approximation maxima retain exact coefficient-bit contracts but use a
  four-ULP portability bound for libm-dependent `powf` evaluation.

Focused ensemble and scalar-grid tests pass after these fixes. The delta
re-review found no remaining issue and recorded `Correct-to-merge`.
