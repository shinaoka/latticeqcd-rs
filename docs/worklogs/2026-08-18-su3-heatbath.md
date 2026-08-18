# SU(3) heatbath implementation worklog

Date: 2026-08-18

## Review gate

The pre-implementation design is `docs/design/su3-heatbath.md`.
`reviewer-flash` completed the full design review with verdict
`Correct-to-merge` and no Critical or Important findings. Five Minor
clarifications were incorporated: accepting-iteration attempt semantics, named
typed errors, private draw/observer test seams, Julia's closed-zero draw defect,
and the `100_000` statistical rejection limit. Delta re-review returned
`Correct-to-merge` with no findings before implementation.

## RED evidence

The current public heatbath integration test was copied without production
changes into a disposable worktree at pre-heatbath commit
`5773310f13bbb4079e610e3e0072d4f6e673538c`. The command

```text
cargo test -p gaugefields --test heatbath --no-run
```

exited 101 with E0432 for absent `heatbath_sweep`/`HeatbathParams` and E0599
for all six absent typed heatbath error variants. The disposable worktree was
then removed.

## Implementation

Codex Luna/max implemented the reviewed kernel, API, fixtures, tests, example,
and documentation in bounded continuation slices; parent completion audited
the exact math, removed dead test helpers, corrected one provenance path, and
ran final evidence. The change:

- adds validated `HeatbathParams`, `HeatbathSweepStats`, and transactional
  `heatbath_sweep`;
- reuses `HostGaugeLinks::force_staple().adjoint()` rather than duplicating
  topology/staple code;
- updates direction 0 through 3, even then odd parity, x-fastest site order,
  with fixed `(0,1)`, `(1,2)`, `(0,2)` subgroup hits;
- implements the Kennedy--Pendleton rejection law with explicit
  `ReproducibleRng`, accepting-iteration attempt accounting, square-root SU(2)
  normalization, typed singular/numerical/rejection failures, and final SU(3)
  normalization;
- shares one crate-private fallible link duplication helper with HMC;
- adds deterministic/order/error kernel tests, public transactionality tests,
  an independent three-beta statistical comparison, and a public example.

Rust deliberately omits Julia's unused preliminary draws, excludes zero from
uniforms, corrects projected SU(2) normalization, rejects singular/non-finite
intermediates, and injects RNG state explicitly. These distribution-preserving
differences are recorded in fixture metadata; bitwise heatbath trajectory
parity is not claimed.

## Julia statistical oracle

The generator used Gaugefields.jl v0.7.2 at clean commit
`9e5719970770f4497405a856315c90bef7f74449` and calls its real `Heatbath`,
`heatbath!`, and `calculate_Plaquette` operations. Each cold `2^4` chain uses
512 burn-in sweeps and 32 blocks of 32 measured sweeps.

| beta | Julia mean | Julia SE | Rust mean | Rust SE | combined z |
|---:|---:|---:|---:|---:|---:|
| 5.5 | 0.5668710138330755 | 0.0018261482821493 | 0.5652114138489025 | 0.0017967254091717 | 0.647815 |
| 5.7 | 0.5889169801245410 | 0.0017294564524736 | 0.5883776863553273 | 0.0016529248300553 | 0.225427 |
| 6.0 | 0.6192210354252312 | 0.0012255777395790 | 0.6198006291655569 | 0.0014194855196873 | 0.309057 |

All comparisons are below the predeclared six-combined-standard-error bound;
all means and block uncertainties satisfy the independent non-vacuity bounds.
The optimized statistical test itself ran in 0.59 seconds after the cold
release dependency build completed.

Two consecutive focused runs took 13.16 and 13.11 seconds and produced identical
artifacts. Two consecutive default full runs took 21.35 and 22.17 seconds and
also remained identical:

```text
complete fixture tree: 24bae8831f8d0ab27bb06ee6a17c55bc8c2a00bf3f72100a1db7b582e90273bc
heatbath metadata:      644e3e0f5b9f32113c2f8a730893d43c9084347ddc26dd6b182570efa95d3ca9
```

## Verification

Fresh local verification after the final implementation and metadata:

- `cargo fmt --all -- --check` — PASS.
- `cargo check --workspace` — PASS.
- `cargo test --workspace` — PASS: 113 passed, 1 ignored.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — PASS.
- `cargo test --workspace --all-features` — PASS: 123 passed, 1 ignored.
- `cargo test --doc --workspace --all-features` — PASS: 15 passed.
- `cargo doc --workspace --all-features --no-deps` — PASS.
- focused public heatbath tests — PASS: 6 passed.
- focused heatbath statistics — PASS: 1 passed in debug and release.
- focused HMC — PASS: 12 passed; reproducible RNG — PASS: 8 passed.
- `quenched_heatbath` example — PASS: three sweeps, 64 links each, finite
  plaquettes; `quenched_hmc` — PASS: 3/3 accepted; traced Wilson — PASS with
  zero residual.
- focused and full Julia generator determinism — PASS with checksums above.
- release statistical runtime — PASS: 0.59 seconds after compilation.
- stale general-SUN/duplicate-staple/dead-code/placeholder searches — PASS.
- `git diff --check` — PASS.

No commit, push, PR/issue mutation, sibling source edit, general-SUN API,
overrelaxation, compatibility shim, placeholder, or hidden fallback was
introduced by this task.

## Post-implementation review

The complete implementation diff received `reviewer-flash: Correct-to-merge`
with no Critical or Important findings. Its three Minor findings were fixed
before delta re-review: one redundant rejection-limit branch was deleted, the
staple finite check was consolidated in `update_link`, and the Julia
normalization citation was corrected to lines 652--656.
