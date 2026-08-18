# Public quenched HMC implementation worklog

Date: 2026-08-18

## Review gate

The pre-implementation design is `docs/design/quenched-hmc.md`.
`reviewer-flash` completed a full review with verdict `Correct-to-merge` and no
Critical or Important findings. Four Minor clarifications were incorporated:
fixture word 514, explicit error variants, word 513/514 test procedure, and
removal of the obsolete ChaCha test dependency. A delta re-review returned
`Correct-to-merge` with no new findings before implementation began.

## RED evidence

The current public integration test was copied without production changes into
a disposable worktree at pre-HMC commit
`1bf7eda6e1213282a3c18a7e8c67adfa5a567f81`. The command

```text
cargo test -p gaugefields --test quenched_hmc --no-run
```

exited 101. It reported E0432/E0425 for the absent public HMC functions and
types and E0599 for the absent typed HMC error variants. The disposable
worktree was then removed.

## Implementation

Codex Luna/max implemented the reviewed change; parent completion passes fixed
documentation, fixture, and validation gaps after bounded implementation turns.
The change:

- adds validated `HmcParams`, `HmcOutcome`, Gaussian momentum sampling,
  coefficient kinetic energy, Wilson Hamiltonian, transactional U-P-U
  trajectories, and transactional Metropolis updates;
- keeps `CpuEvolutionContext` and `ReproducibleRng` application-owned;
- uses exactly one `-step_size/3` force factor and one unconditional acceptance
  uniform after every completed proposal;
- removes the separate crate-private HMC implementation and its direct ChaCha
  dependency;
- adds a pinned Gaugefields.jl one-trajectory oracle, 12 public contract tests,
  one injected-failure unit test, compiled public examples, and documentation.

`rand` default features are disabled because only its `RngCore`/`Error`
re-exports are needed; `cargo tree -i rand_chacha` confirms that ChaCha is not
resolved.

## Julia oracle

The generator used Gaugefields.jl v0.7.2 at clean commit
`9e5719970770f4497405a856315c90bef7f74449`. It executes exported Gaugefields.jl
field, action, derivative, TA, exponential, multiplication, and substitution
operations. The xoshiro state `[1,2,3,4]` supplies 512 raw words for the compact
`2^4` momentum, word 513 for Metropolis, and fixture word 514 for the next-stream
check.

Two consecutive focused generations produced the same complete fixture-tree
checksum and HMC-directory checksum:

```text
fixture tree: 362eb28321d8d4598d79f44fbe6ff77c549c9ef0e6b44c226bec8f27059e8dcb
HMC fixture:  f55790d945114ce4e003c50443cfd9ea993ebe48736b193c5f9128a9e272d764
```

Two consecutive default full generations also retained the same complete-tree
checksum, proving that full regeneration includes the HMC oracle without drift.

## Verification

Fresh local evidence after the final dependency feature set:

- `cargo fmt --all -- --check` — PASS.
- `cargo check --workspace` — PASS.
- `cargo test --workspace` — PASS: 97 passed, 1 ignored.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — PASS.
- `cargo test --workspace --all-features` — PASS: 107 passed, 1 ignored.
- `cargo test --doc --workspace --all-features` — PASS: 12 passed.
- `cargo doc --workspace --all-features --no-deps` — PASS.
- `cargo test -p gaugefields --test quenched_hmc -- --nocapture` — PASS:
  12 passed.
- `cargo run -p gaugefields --example quenched_hmc --all-features` — PASS:
  `accepted=3/3 normalized_plaquette=0.9873817800273383`.
- `cargo run -p gaugefields --example traced_wilson_action --all-features` —
  PASS: `direct=-576 traced=-576 residual=0`.
- stale private-HMC/direct-ChaCha source search — PASS.
- `cargo tree -i rand_chacha` — package absent.
- `git diff --check` — PASS.

No commit, push, PR/issue mutation, sibling-source edit, compatibility shim,
placeholder, hidden fallback, or heatbath implementation was made in this task.

## Post-implementation review

The complete implementation diff received `reviewer-flash: Correct-to-merge`
with no Critical or Important findings. Its one Minor finding identified a
missing `isize::MAX` byte-range guard before momentum allocation. The guard and
a concrete boundary regression case were added before final delta re-review.
