# Phase 3 fermions worklog

Status: Task A implementation pending

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
| A: field and Wilson operator | `docs/design/phase-3-fermions.md` Task A | Correct-to-merge | pending |
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
