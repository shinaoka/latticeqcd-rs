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
| A: fermion measurements | round 1 `Correct-to-merge` | full diff `Correct-to-merge` | two delta rounds `Correct-to-merge` |
| B: `latticeqcd` frontend | round 2 `Correct-to-merge` | full diff `Correct-to-merge` | delta `Correct-to-merge` |
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

## Task A verification evidence

- deterministic fixture generated twice from clean pinned checkouts with Julia
  1.12.5; both complete trees SHA256:
  `55ee3a86d2b8d6ef4497c05a5539e5daa53ace71b147382e4e951ef21d570f18`;
- 89 declared payloads exactly matched the 89 non-metadata files;
- maximum Wilson solution error: `5.551792708129603e-17`;
- maximum staggered point-source solution error: `3.351599329828878e-13`;
- maximum staggered chiral solution error: `4.139010604214116e-13`;
- corrected pion error: Wilson `5.551115123125783e-17`, staggered
  `8.12128142513302e-14`;
- chiral per-source scalar error: `8.526512829121202e-14`;
- final chiral scalar error: `3.885780586188048e-16`;
- all values remained below the predeclared `2e-12` absolute and `2e-10`
  relative gates; fresh relative residuals remained below `1e-11`.

Focused gates:

- `cargo check -p measurements --no-default-features`: pass;
- `cargo check -p measurements --all-features`: pass;
- `cargo test -p dirac-operators`: 73 passed;
- `cargo test -p measurements --all-features`: 24 passed;
- `cargo clippy -p dirac-operators -p measurements --all-targets --all-features -- -D warnings`: pass;
- `cargo test --doc -p dirac-operators -p measurements --all-features`: 29 passed;
- `cargo fmt --all -- --check` and `git diff --check`: pass.

Task A post-implementation full-diff review returned `Correct-to-merge`.
Three Minor fixture/test findings were fixed; both delta reviews returned
`Correct-to-merge`, including the final all-pairs/global-Z4-phase coverage.

## Task B acceptance continuation

The existing frontend was continued without changing Task A/C numerical code.
The runner now records the validated lattice and initial RNG words, boxes the
large failure payloads, dispatches heatbath/Wilson/staggered updates with exact
`UpdateKind` records, classifies HMC/chiral validation failures separately, and
publishes zero-padded ILDG names through a no-clobber hard-link plus cleanup
path. Acceptance tests cover execution dispatch, initial/thermalization
scheduling, chiral RNG ordering, wrong-lattice ILDG input, and destination
preservation with no temporary-file residue. The checked binary configuration
is `examples/phase4.toml`.

The focused RED test run failed at compile time with the expected missing
report fields, typed variants, and heatbath kind. After the minimal fixes,
`cargo test -p latticeqcd` passed 24 tests (17 integration tests and 7
frontend doctests), and the exact binary command printed:

```text
completed_updates=1 accepted=1 rejected=0 measurements=2 flows=0 outputs=0
```

Task-local verification passed: formatting, latticeqcd check/test/clippy,
latticeqcd doctests, measurements no-default-features check, binary smoke,
workspace check/test/all-features test/workspace doctests/docs, both existing
traced/fermion examples, exact tenferro-pin/stale-symbol checks, and
`git diff --check`. The Task B full-diff review returned `Correct-to-merge`.
Five Minor strictness/coverage/cleanup/documentation findings were fixed, and
the delta review returned `Correct-to-merge`. No commit or push was made;
Task C and integrated review remain pending.
