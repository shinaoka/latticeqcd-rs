# Phase 4 fermion measurements and frontend worklog

Date: 2026-08-19
Branch: `feat/phase4-observables`
Base: `9af624ddb849204ef9db765561c3e4791c926252`

## Scope

Complete Issue #17 Phase 4: pion correlator, chiral condensate, and the strict
parameter-file-driven `latticeqcd` frontend. Preserve the host 4D SU(3)
`Complex64` boundary and existing Phase 1–3 numerical contracts.

## Sources inspected

- QCDMeasurements.jl v0.2.13
  `9e04c37bbd68712cf7a749ae5aff10eb6aae4566`
- LatticeQCD.jl v1.3.7
  `c09de20aae10f28f6a9c7e84e7711fce94d50915`
- LatticeDiracOperators.jl v0.6.4
  `bdef628184597815ba3e0cddf2536df767e78a02`
- Gaugefields.jl v0.7.2
  `9e5719970770f4497405a856315c90bef7f74449`
- existing Rust Phase 1–3 public APIs and fixtures.

## Upstream-reference findings

- Existing Issue #23: staggered solver tolerance key mismatch.
- Existing Issue #27: Julia `k*pi/4` pseudo-Z4 noise.
- Issue #29: pion reconstruction duplicates source-diagonal values across all
  sink components. Rust will use the corrected full Frobenius contraction.
- Issue #30: LatticeQCD.jl v1.3.7 parser/scheduler and included measurement-set
  bug candidates. Rust will use strict tagged parsing and explicit scheduling.

No upstream submission is authorized or performed.

## Review gate

| Task | Design review | Implementation review | Delta review |
|---|---|---|---|
| A: fermion measurements | round 1 `Correct-to-merge` | pending | pending |
| B: `latticeqcd` frontend | round 2 `Correct-to-merge` | pending | pending |
| C: integrated evidence/docs | round 2 `Correct-to-merge` | pending | pending |
| Integrated diff | n/a | pending | pending |

Round 1 reviewer findings were incorporated before implementation: exact sea
fermion-to-HMC dispatch, raw-word Z4 mapping, explicit fixture code placement,
fermion-specific Julia tolerance keys, per-side solver diagnostics, initial
schedule edge cases, valence flavor meaning, and the predeclared ensemble
schedule/criterion. Round 2 found no blocking issues and returned overall
`Correct-to-merge`; its non-blocking heatbath wording note was also made explicit.

Reviewer: `reviewer-flash` (read-only, different model family from the planned
`luna-implementer`).

## Verification evidence

Pending implementation.
