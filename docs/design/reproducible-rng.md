# Cross-language reproducible RNG

Status: implemented
Date: 2026-08-18
Issue: [latticeqcd-rs #17](https://github.com/shinaoka/latticeqcd-rs/issues/17)
Dependency: stacked on [PR #18](https://github.com/shinaoka/latticeqcd-rs/pull/18)

## Goal

Add the Phase 1 random-number foundation needed for deterministic Julia/Rust
HMC comparisons:

- import Julia's four-word Xoshiro256++ state directly;
- reproduce the raw `UInt64` stream with `rand_xoshiro`;
- provide a fully specified Box--Muller standard-normal stream independent of
  Julia's and Rust's built-in normal samplers.

This PR does not expose HMC, sample momentum, or change the existing private HMC
regression driver. Those consumers follow after this primitive is stable.

## References and provenance

The state and transition convention is validated against:

- Julia 1.12.5 commit
  `5fe89b8ddc166260bfcd4a195b305aff0ccad686`,
  `stdlib/Random/src/Xoshiro.jl`, especially `Xoshiro`, `initstate!`, and
  `rand(..., UInt64)`;
- `rand_xoshiro` 0.6.0, `Xoshiro256PlusPlus::from_seed` and `RngCore`;
- David Blackman and Sebastiano Vigna, “Scrambled Linear Pseudorandom Number
  Generators,” ACM TOMS 47(4), 2021, and the xoshiro256++ reference vectors.

Julia's `s0..s3` are the 256-bit xoshiro state. Its `s4` is an auxiliary
splitmix/task-fork state and is not part of the raw xoshiro256++ stream imported
here. Source comments and fixture metadata retain these references; the Rust
implementation delegates the transition to `rand_xoshiro` rather than copying
Julia's transition code.

## Public API

Add `crates/gaugefields/src/rng.rs` and re-export one type:

```rust
pub struct ReproducibleRng { /* Xoshiro256PlusPlus */ }

impl ReproducibleRng {
    pub fn from_state(state: [u64; 4]) -> Result<Self, GaugeError>;
    pub fn set_state(&mut self, state: [u64; 4]) -> Result<(), GaugeError>;
    pub fn open_unit_f64(&mut self) -> f64;
    pub fn standard_normal_pair(&mut self) -> [f64; 2];
    pub fn fill_standard_normals(&mut self, output: &mut [f64]);
}

impl rand::RngCore for ReproducibleRng { ... }
```

`Clone` copies the exact stream position. `Debug` is compact and does not dump
the four-word state. Do not expose another seed abstraction, global/default
constructor, serialization, or state extraction in this PR.

### State import

`state` is ordered exactly as Julia `(s0, s1, s2, s3)`. Convert each word with
`u64::to_le_bytes` into the 32-byte seed accepted by
`Xoshiro256PlusPlus::from_seed`. This produces the same words on every host
endianness because `rand_xoshiro` reads its seed as little-endian words.

The all-zero state is invalid for xoshiro256++. `rand_xoshiro::from_seed`
silently remaps it, which would violate direct-state semantics, so both
`from_state` and `set_state` return a new typed
`GaugeError::InvalidRngState`. A failed `set_state` leaves the old stream
unchanged. Every nonzero state is accepted verbatim.

`RngCore::{next_u32,next_u64,fill_bytes,try_fill_bytes}` delegates directly to
the wrapped generator. There is no cryptographic-RNG claim.

## Uniform and normal stream contract

One open-interval uniform consumes exactly one raw 64-bit word:

```text
u = (Float64(next_u64 >> 12) + 0.5) * 2^-52
```

Using the upper 52 bits matches Julia's native Xoshiro precision, avoids weak
low bits, and guarantees `0 < u < 1` without rejection or variable draw count.

One Box--Muller pair consumes exactly two raw words in this order:

```text
u1 = open_unit_f64()
u2 = open_unit_f64()
r = sqrt(-2 * log(u1))
theta = TAU * u2
pair = [r * cos(theta), r * sin(theta)]
```

There is deliberately no cached spare normal: the four xoshiro words remain the
complete hidden state. `fill_standard_normals` fills pairs in order. For an odd
output length it stores the cosine result and intentionally discards the final
sine result; therefore it consumes `2 * ceil(len/2)` raw words. Empty output
consumes none. Mixing `RngCore`, uniform, pair, and fill calls advances one
shared raw stream in call order.

The formula and draw counts are public compatibility contracts. Transcendental
implementations can differ by a few ulps across platforms, so raw words are
bit-exact while normal fixture comparisons use a fixed small absolute
tolerance rather than claiming cross-platform bit identity.

## Dependencies

- Add workspace `rand = "0.8"` and `rand_xoshiro = "0.6"` normal dependencies.
- Use them from `gaugefields`; remove the duplicate dev-only `rand` declaration.
- Keep `rand_chacha` dev-only for existing private HMC regression tests.
- Add no normal-distribution, serialization, or generic RNG dependency.

## Julia fixture

Extend the authoritative `fixtures/generate.jl` with
`generate_reproducible_rng()` and commit
`fixtures/reproducible_rng/metadata.json`. The generator uses only Julia's
`Random` stdlib but remains in the single repository fixture generator.

A focused invocation

```bash
julia --startup-file=no fixtures/generate.jl reproducible_rng
```

must define and run the stdlib-only RNG generator before the existing
`GAUGEFIELDS_JL_DIR` activation/check, then exit without touching other
fixtures. No other argument is accepted. The existing no-argument invocation
retains its clean Gaugefields.jl checkout requirement and additionally writes
the RNG fixture during a full regeneration.

The metadata records:

- Julia version and `Base.GIT_VERSION_INFO.commit`;
- Julia source file URL and revision;
- algorithm and `rand_xoshiro` version;
- input state `[1, 2, 3, 4]` and state-word ordering;
- first ten raw outputs as padded hexadecimal strings, generated by an explicit
  scalar loop calling `rand(rng, UInt64)` once per word; array `rand`/`rand!`
  and Julia's SIMD bulk Xoshiro stream are prohibited;
- uniform formula, Box--Muller pair ordering, and odd-fill policy;
- ten normal values and their Julia IEEE-754 bit strings.

For `[1,2,3,4]`, the independent xoshiro reference/Julia raw sequence starts:

```text
41943041, 58720359, 3588806011781223, 3591011842654386,
9228616714210784205, 9973669472204895162, 14011001112246962877,
12406186145184390807, 15849039046786891736, 10450023813501588000
```

The integration test reads the fixture rather than duplicating its normal
values in test source. Regeneration must be deterministic under the recorded
Julia revision.

## Tests

Add `crates/gaugefields/tests/reproducible_rng.rs` as the public contract test.
Start with this RED test before implementation.

Required checks:

1. `[1,2,3,4]` matches all ten raw Julia/reference words exactly through the
   public `RngCore` implementation.
2. `open_unit_f64` is strictly inside `(0,1)` for concrete boundary vectors:
   state `[1,0,0,0xffff_ffff_ffff_fffe]` yields first raw word zero and state
   `[0,1,0,0xffff_ffff_ffff_ffff]` yields `u64::MAX`. The results are exactly
   `2^-53` and `1-2^-53`, each consuming one word; no NaN or infinity is
   produced.
3. five normal pairs match the Julia fixture within a fixed, justified
   cross-libm absolute tolerance and remain finite.
4. `fill_standard_normals` preserves pair ordering; lengths 0, 1, 2, 3 cover
   exact draw consumption and the documented discarded odd spare.
5. `set_state` reproduces the stream from the beginning and clears no hidden
   state because none exists.
6. all-zero construction/reset returns `InvalidRngState`; failed reset is
   transactional.
7. `Clone` produces identical subsequent raw/uniform/normal results and compact
   `Debug` omits the numeric state words.
8. `RngCore::next_u32`/`fill_bytes` behavior agrees with a directly constructed
   `rand_xoshiro::Xoshiro256PlusPlus` for the same state.
9. One interleaved public sequence — `next_u64`, `open_unit_f64`,
   `standard_normal_pair`, `fill_standard_normals(&mut [0.0; 1])`, then
   `next_u64` — consumes the documented raw word positions and matches values
   reconstructed from the fixture. This guards the single shared stream and
   absence of hidden cached state.

Do not replace these deterministic contracts with statistical tests. Statistical
quality belongs to xoshiro256++ and later HMC ensemble validation.

## Documentation and records

- Add a short README section showing state import and normal filling.
- Update `docs/design/dependencies.md` with the new direct dependencies and
  Phase 1 status.
- Update `docs/design/layout.md` so its Gaugefields.jl provenance restriction
  remains specific to gauge-field fixtures; the RNG fixture records Julia
  stdlib provenance instead. Historical `docs/superpowers/` specs remain
  immutable records: their statement that Rust does not reproduce the
  StableRNG-based hot-field initializer still applies only to those existing
  gauge-field fixtures, not this new explicit Xoshiro stream.
- Add `docs/worklogs/2026-08-18-reproducible-rng.md` with sources, RED/GREEN
  evidence, exact fixture command, verification, and review gates.
- Public methods require `# Errors` where fallible and runnable examples with
  meaningful value/draw assertions.

## Non-goals

- public HMC sampler, momentum sampler, Metropolis logic, or heatbath;
- changing the crate-private HMC tests from ChaCha/uniform draws;
- matching Julia's built-in `randn` implementation or SIMD bulk mode;
- arbitrary distributions, seeding from a scalar/string/entropy, global or
  thread-local RNGs;
- jump/long-jump APIs, state export/checkpoint serialization, cryptographic use;
- GPU/backend tensor random generation.

## Expected files

- `Cargo.toml`, `Cargo.lock`, `crates/gaugefields/Cargo.toml`;
- `crates/gaugefields/src/{rng.rs,error.rs,lib.rs}`;
- `crates/gaugefields/tests/reproducible_rng.rs`;
- `fixtures/generate.jl`, `fixtures/reproducible_rng/metadata.json`;
- `README.md`, relevant `docs/design/*.md`, and the dated worklog.

## Verification

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo doc --workspace --all-features --no-deps
julia --startup-file=no fixtures/generate.jl reproducible_rng
cargo test -p gaugefields --test reproducible_rng -- --nocapture
git diff --check
```

The focused argument rewrites only `reproducible_rng/metadata.json`; it does
not require `GAUGEFIELDS_JL_DIR` or regenerate existing large fixtures.
Confirm the stacked branch contains PR #18 and no unrelated worktree metadata.

## Acceptance criteria

- nonzero Julia four-word states enter `rand_xoshiro` verbatim and all-zero is
  rejected transactionally;
- the raw stream is bit-exact against Julia/reference vectors;
- uniform mapping, Box--Muller ordering, odd-fill consumption, and absence of a
  hidden cache are documented and tested;
- Julia fixture provenance is complete and reproducible;
- public docs/examples compile and tests cover errors and draw counts;
- existing behavior, private HMC tests, and numerical tolerances are unchanged;
- repository local/CI gates and independent design/diff reviews pass;
- the PR remains a small RNG-foundation change stacked cleanly on PR #18.

## Risks

- Built-in Julia `rand(Float64)` and bulk generation are not this API; docs must
  direct the Julia oracle to the explicit formula.
- Box--Muller transcendental results are platform-libm-sensitive; test tolerance
  must allow only the measured ulp-scale difference while raw parity stays
  exact.
- Adding a cached spare later would change state and draw-consumption semantics
  and requires a new versioned contract.
- `rand_xoshiro` major upgrades can change dependency traits; version bumps
  require rerunning the same Julia/reference vectors.
