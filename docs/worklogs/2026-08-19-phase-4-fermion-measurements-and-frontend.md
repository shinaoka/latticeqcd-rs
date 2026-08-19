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
| C: integrated evidence/docs | round 2 `Correct-to-merge` | round 1 `Not correct-to-merge`; fixed | two delta rounds, final `Correct-to-merge` |
| Integrated diff | n/a | full diff `Correct-to-merge` | delta `Correct-to-merge` |

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
  `af8a045ccfcdf92382806b078edd758d056fad019604c48cf0156811466d7c2d`;
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
the delta review returned `Correct-to-merge`. No commit or push was made.

## Task C completion evidence

Task C adds `fixtures/generate.jl` mode
`fermion_measurements_phase4_ensemble`, the metadata-only fixture tree
`fixtures/fermion_measurements_phase4_ensemble`, and the independent
`crates/measurements/tests/phase4_ensemble.rs` test. The generator uses clean
Julia 1.12.5 plus the pinned Gaugefields.jl
`9e5719970770f4497405a856315c90bef7f74449`, LatticeDiracOperators.jl
`bdef628184597815ba3e0cddf2536df767e78a02`, Wilsonloop.jl
`e1a617fdedb19b785f89bdeb13c30e53b20743a7`, and QCDMeasurements.jl
`9e04c37bbd68712cf7a749ae5aff10eb6aae4566`. It records the fixed Phase 3
RHMC tables and `[0.0004,64]` interval, all 20 Julia trajectory records, all
16 measurements, two explicit canonical-Z4 source code arrays per measurement,
four block summaries, solver/layout/normalization/source provenance, and the
separate Rust update/source stream states. No Rust results or configurations
are written by Julia.

The exact predeclared run is beta `5.7`, mass `0.5`, `Nf=2`, lattice
`[2,2,2,2]`, step size `0.01`, two MD steps, four thermalization trajectories,
and 16 interval-one measurements in four blocks of four. Two clean generator
runs produced the identical complete-tree hash
`d3b0e00fafe54d590b23b3afad6e75214fefaf81a68f5c8c28e13da008c7f657`; the
single metadata payload hash is
`b7c5c3ac4d1042de24dcbbdc2bad591fff95e89b48ba304ebe06e9009db8c9c7`.

The focused Rust test passed 2 tests, including the explicit zero-combined-SE
case. It independently accepted all 4 thermalization and 16 measured updates.
The recorded Julia/Rust block means were:

```text
pion Julia [[0.49098124162968626, 1.9156314138884456],
           [0.4987236819746711, 1.9183354002187256],
           [0.504100784573672, 1.920612344936982],
           [0.5141885944254221, 1.9198501820310567]]
pion Rust  [[0.4895505014032875, 1.9147597875158442],
           [0.4981772675478961, 1.9161905005944495],
           [0.5000324075646916, 1.9160401161006275],
           [0.5102573651938351, 1.9154787445422932]]
chiral Julia [0.6034035045821012, 0.6015137365891218,
              0.6046312233300931, 0.6110815307064406]
chiral Rust  [0.5996102079154657, 0.6065291411402107,
              0.609136369505594, 0.6060743361103856]
delta-H Julia [-1.723232045449663e-4, -1.1397239357791022e-4,
              -1.2471150934345587e-4, -2.1943733928964093e-4]
delta-H Rust  [-1.960644032124037e-4, -1.2465574448583538e-4,
              -1.310893148840364e-4, -1.1461058957706882e-4]
```

Normalized differences (absolute mean difference divided by the combined
standard error of the four block means) were `0.3856782` for pion `t=0`,
`2.609377` for pion `t=1`, and `0.06207383` for the chiral scalar. All are
below the required six-SE gate. With only four consecutive blocks and short
MD separation, these standard errors are an intentionally noisy,
autocorrelated consistency estimate rather than a calibrated physics-precision
claim. The test prints every Rust per-trajectory acceptance/delta-H and measured
pion/chiral value, plus the Julia/Rust blocks and acceptance summaries. Rust
stream states in the metadata are predeclared constants, not Julia-generated
results or configurations. The generator never calls the buggy high-level Julia pion
reconstruction or `Z4_distribution_fermi!`; Issues #27, #29, and #30 are
recorded with the pinned revisions and Rust-side decisions.

Task C full-diff review found one Important stale hash/docs mismatch and two
Minor statistical/metadata clarity issues. All were fixed; after metadata
regeneration, a remaining formatting anomaly was fixed and regenerated again.
Both delta rounds completed, with the final verdict `Correct-to-merge`.
Integrated review remains pending; no push was made.

## Fresh integrated local gate

A fresh target rebuild removed `26.2 GiB` of prior Cargo artifacts, then passed:

```text
cargo fmt --all -- --check                         PASS
cargo check --workspace                            PASS
cargo test --workspace                             PASS (293 passed, 1 ignored)
cargo clippy --workspace --all-targets --all-features -- -D warnings
                                                     PASS
cargo test --workspace --all-features              PASS (303 passed, 1 ignored)
cargo test --doc --workspace --all-features        PASS (71 passed)
cargo doc --workspace --all-features --no-deps     PASS
cargo build --workspace --examples --all-features  PASS
```

All six Rust examples plus the `latticeqcd` binary configuration ran
successfully. Their checked outputs included ILDG cold round-trip, three
heatbath sweeps, quenched HMC `3/3`, traced/direct Wilson action residual `0`,
cold gluonic measurements, Wilson/staggered/RHMC smoke, and frontend
`completed_updates=1 accepted=1 rejected=0 measurements=2 flows=0 outputs=0`.

Integrated self-audits passed: `git diff --check`; five exact tenferro manifest
pins and nine lockfile sources at
`c942129974b544225ed963414d7be1300980f901`; current stale-symbol scan; no new
unsafe/TODO/fallback code; both fixture declaration sets; clean pinned Julia
checkouts; and exact MIT notices for all five crates. The preflight found and
fixed the initially missing `latticeqcd` copy of LatticeQCD.jl's MIT notice and
added pinned attribution to README/rustdoc before integrated review.

The integrated full-diff review returned `Correct-to-merge` with four Minor
findings. The design status, synthetic identity contraction coverage,
dependency policy, and independent acceptance reporting were fixed; Task A was
regenerated twice at its current hash. Integrated delta review returned
`Correct-to-merge` with no findings. The exact post-delta tree was then rebuilt
from another empty target (`24.5 GiB` removed) and produced the counts above;
all seven runnable examples/frontends passed again.
