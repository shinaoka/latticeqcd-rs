# Reproducible RNG foundation

- **Date:** 2026-08-18
- **Branch:** `feat/reproducible-rng`
- **Target:** `latticeqcd-rs-rng`, stacked on PR #18
- **Design:** `docs/design/reproducible-rng.md`
- **Design gate:** pre-implementation `reviewer-flash` verdict `Correct-to-merge`; all five Minor precision items were incorporated into the design before implementation.

## Sources and provenance

- Julia 1.12.5 `stdlib/Random/src/Xoshiro.jl`, revision
  `5fe89b8ddc166260bfcd4a195b305aff0ccad686`:
  `https://github.com/JuliaLang/julia/blob/5fe89b8ddc166260bfcd4a195b305aff0ccad686/stdlib/Random/src/Xoshiro.jl`.
- `rand_xoshiro` 0.6.0 `Xoshiro256PlusPlus::from_seed` and `RngCore`,
  documented at `https://docs.rs/rand_xoshiro/0.6.0`, with the transition
  delegated to the dependency rather than copied.
- `rand` 0.8 `RngCore`/`SeedableRng` traits; no distribution crate or serde
  feature was enabled for the RNG addition.
- Blackman and Vigna, “Scrambled Linear Pseudorandom Number Generators,” ACM
  TOMS 47(4), 2021, and the xoshiro256++ reference vectors.

The fixture records Julia's `(s0, s1, s2, s3)` order, little-endian words,
Julia's auxiliary `s4` exclusion, the scalar raw-word loop, the open-unit
formula, Box--Muller ordering, odd-fill consumption, raw hexadecimal words,
and normal decimal values plus IEEE-754 bits. It uses no Gaugefields.jl
activation on the focused path.

## Implementation decisions

`ReproducibleRng` wraps `rand_xoshiro::Xoshiro256PlusPlus`, rejects only the
all-zero state, and re-exports `GaugeError::InvalidRngState`. State replacement
constructs the replacement before assignment, so a failed reset is
transactional. `RngCore` delegates directly. Uniforms consume one raw word;
normal pairs are uncached; odd fills discard the final sine result. No HMC,
jump, state export, global RNG, distribution, or serialization surface was
added.

## RED/GREEN evidence

The public integration contract was added before `rng.rs` and run with:

```text
cargo test -p gaugefields --test reproducible_rng -- --nocapture
```

RED result: exit 101. Rust reported `E0432` (`no ReproducibleRng in the
root`) and two `E0599` failures (`GaugeError::InvalidRngState` was absent),
with no production implementation present.

After the smallest implementation and fixture were added, the same command
passed: **8 tests passed**.

## Verification

Final verification was run in `/home/shinaoka/tensor4all/latticeqcd-rs-rng`:

- Focused fixture regeneration, twice:
  `julia --startup-file=no fixtures/generate.jl reproducible_rng` — PASS on
  both runs; `fixtures/reproducible_rng/metadata.json` checksum was
  `3226d9557bc734009b74d8dda2feaf7dc8897fbaa35cdb530a6a524e6bc6c9dd` both
  times. A sorted SHA-256 snapshot/diff of the complete `fixtures/` tree was
  unchanged.
- `cargo fmt --all` — PASS.
- `cargo fmt --all -- --check` — PASS.
- `cargo check --workspace` — PASS.
- `cargo test --workspace` — PASS: 80 passed, 1 ignored.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — PASS.
- `cargo test --workspace --all-features` — PASS: 90 passed, 1 ignored.
- `cargo test --doc --workspace --all-features` — PASS: 6 passed.
- `cargo doc --workspace --all-features --no-deps` — PASS.
- `cargo test -p gaugefields --test reproducible_rng -- --nocapture` — PASS:
  8 passed.
- `cargo run -p gaugefields --example traced_wilson_action --all-features` —
  PASS: `direct=-576 traced=-576 residual=0`.
- `cargo tree -p gaugefields --all-features -i tenferro-tensor` — PASS: all
  tenferro packages resolved to `c9421299`.
- stale-symbol source check — PASS; exact tenferro pin check — PASS (five
  manifest revisions and matching lock/tree resolution); `git merge-base
  --is-ancestor origin/main HEAD` — PASS.
- `git diff --check` — PASS.

The focused test's RED command was the same cargo test command above and exited
101 before production implementation, reporting unresolved
`gaugefields::ReproducibleRng` and missing `GaugeError::InvalidRngState`.

No commit, push, PR, branch change, sibling-repository edit, or historical
`docs/superpowers/` edit was made during implementation.

## Post-implementation review

The full implementation diff received `reviewer-flash: Correct-to-merge` with
no Critical or Important findings. Its two Minor documentation findings were
fixed before the final delta re-review: the design status now records
implementation, and the `GaugeError` summary covers RNG-state failures.
