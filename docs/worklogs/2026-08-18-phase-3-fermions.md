# Phase 3 fermions worklog

Status: Task A complete; Task B implementation pending

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
| B: CG and BiCGStab | same, Task B | Correct-to-merge | pending |
| C: Wilson pseudofermion/HMC | same, Task C | Correct-to-merge | pending |
| D: staggered/multi-shift CG | same, Task D | Correct-to-merge | pending |
| E: two-flavor RHMC/integration | same, Task E | Correct-to-merge | pending |
| Integrated Phase 3 | full design | Correct-to-merge | pending |

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
completed the evidence below. The independent Task A post-review remains pending.

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
