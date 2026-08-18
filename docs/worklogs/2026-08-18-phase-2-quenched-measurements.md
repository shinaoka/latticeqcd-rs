# Phase 2 quenched measurements worklog

Status: Task A complete; Task B implementation pending

## Base and scope

- Branch: `feat/phase2-quenched-measurements`
- Base: `origin/main` at `094f96e81165b0732b0619dd084c0f2a1e5f0f0a`
- Issue: <https://github.com/shinaoka/latticeqcd-rs/issues/17>
- Design: `docs/design/phase-2-quenched-measurements.md`

The user selected pure-Rust minimal ILDG, signed Wilson paths, CPU/host SU(3),
isotropic synchronous stout, fixed-step general-action RK3 flow, and both
fixed-field and short statistical Julia comparison.

## Pinned upstream sources

- Gaugefields.jl v0.7.2:
  `9e5719970770f4497405a856315c90bef7f74449`
- Wilsonloop.jl 0.1.5 source:
  `e1a617fdedb19b785f89bdeb13c30e53b20743a7`
- QCDMeasurements.jl v0.2.13:
  `9e04c37bbd68712cf7a749ae5aff10eb6aae4566`

All three reference checkouts were clean when the design was written.

## Reconnaissance

Existing Rust code already owns:

- compact `[3,3,NX,NY,NZ,NT]` host links with x-fastest sites,
- checked periodic coordinates and neighbor lookup,
- fixed-size `Mat3` multiplication, adjoint, TA projection, and coefficient
  conversion,
- Julia-compatible `exp_ta` and cached `exp_ta_update`,
- Wilson plaquette, measurement staple, and six-term force staple,
- Julia-compatible reproducible RNG and heatbath ensembles.

The Phase 2 design reuses those paths. It does not create a second plaquette,
periodic-index, matrix-exponential, or Wilson-staple implementation.

Baseline exact-base verification:

```text
cargo test --workspace --all-features
```

Result: 123 passed, 1 ignored, including 15 doctests.

## Review gate

| Task | Design document | Pre-implementation reviewer | Pre verdict | Post implementation verdict |
|---|---|---|---|---|
| A: host view and ILDG | Phase 2 design, Task A | reviewer-flash | Correct-to-merge | Correct-to-merge |
| B: Wilson paths/actions/force | Phase 2 design, Task B | reviewer-flash | Correct-to-merge | pending |
| C: stout | Phase 2 design, Task C | reviewer-flash | Correct-to-merge | pending |
| D: measurements/flow/validation | Phase 2 design, Task D | reviewer-flash | Correct-to-merge | pending |
| Integrated branch | Phase 2 design | reviewer-flash | Correct-to-merge | pending |

Review rounds before implementation:

- Round 1 broad reviews stopped without verdict after exceeding their bounded
  source-survey budgets; they did not clear any gate.
- Round 2 AB verdict: `Correct-to-merge`, Tasks A and B clear. Its one Important
  contract gap and seven Minor clarifications were folded into the design:
  exact LIME header/binary/checksum contract, split-record rejection, real
  coefficient type, repeated unit steps, direct Julia force parity,
  validate-before-create test, unknown XML handling, and the existing tensor
  interop boundary.
- Round 2 CD verdict: `Changes required`, Tasks C, D, and integrated design
  blocked. Its Important context-ownership gap and all Minor findings were
  folded into the design: `CpuEvolutionContext` type/instance ownership,
  positive unweighted stout staple, negative-rho parity, non-cubic scope,
  context failure state, exact citations, default-generator statistics, and a
  `Q^2` relative ceiling.

- Final delta verdict: `Correct-to-merge`. Tasks A, B, C, D, and the integrated
  pre-implementation gate are all clear with no new blocker.

No implementation had started before this final verdict. Task A began only
after the final verdict was recorded.

## Task A implementation

Task A adds a shared, fallible `HostGaugeLinks` read view and moves the existing
host kernels onto it without changing their numerical operations. The public
`GaugeLinks::host_view` and `GaugeLinks::try_clone` boundaries preserve typed
placement and shape failures. The minimal ILDG implementation is in
`gaugefields::ildg` and uses `std::io` plus `quick-xml` 0.37.5; it does not use
c-lime or claim that the pinned Julia writer produced the container framing.

The reader and writer implement one LIME message with one unsplit
`ildg-format` record and one unsplit `ildg-binary-data` record, 144-byte LIME
headers, 8-byte record padding, strict MB/ME sequencing, ILDG XML version 1.0,
SU(3), precision 64, exact payload length, big-endian finite Float64 values,
and `t,z,y,x,mu,row,column,real/imag` order. Unknown records and unknown nested
XML metadata are skipped, but duplicate/split/missing required records,
multiple messages, malformed/truncated headers or payloads, unsupported known
metadata, overflow, non-finite components, and trailing bytes are rejected.
All validation precedes destination creation or truncation.

The Julia fixture starts from pinned Gaugefields.jl `Reproducible` hot links.
Gaugefields.jl resets `StableRNG(123)` for every direction, so the generator
periodically shifts direction `mu` by one site along lattice axis `mu` using
the same existing helper as the older random fixtures. This preserves every
site-local SU(3) value while making direction swaps detectable.

Fixture SHA-256 values:

- `gauge.ildg`: `2f2ecb02df3b5c8f3c143883701ac90d7480d5c45e955446390da72307a949e0`
- `metadata.json`: `c6ae589154b70a53c3d897e5ca8b1e1f04e25b9d53f51e4bd4fad41f5af44330`
- `u0.npy`: `2d767b33727c6b03ac6e54165f3f5aa6fc1fb4fdcce770522a6add1a314f1e3e`
- `u1.npy`: `8ade4cac7e212c20b9d557f0b12a76e2324f51849ea0c179efd65ebc8b249473`
- `u2.npy`: `1efe6d2a8fd51b4a05284092035cbf3e5d1949b8dc6373dccd1d881ad42e3226`
- `u3.npy`: `dd432d4b95a7ef2ea811d820472e98db07e594a4155269e9fbac33653435a7d8`

Both focused `fixtures/generate.jl ildg` runs produced complete-tree hash
`b7f92f721f23e0a0bcf82c229c699cdaa192acad6a7c2a9949b41d32a15e8e94`.
Both default generator runs, including all Phase 1 fixtures, produced complete
fixture-tree hash
`c0331196411e2948937717ae9259175ce0a3a0780812412616b71d487bcef5e0`.
The exact invocation set `GAUGEFIELDS_JL_DIR` to the clean pinned checkout and
`JULIA_NUM_THREADS=1`.

A repository-external Rust consumer read the Julia fixture and called the
public Rust writer. Its output was byte-identical to `gauge.ildg` at the hash
above. The checked `fixtures/check_ildg_readback.jl` then loaded that Rust output
through pinned Gaugefields.jl `ILDG`/`load_gaugefield!` with explicit dimensions
and compared all four directions, 144 `ComplexF64` values (288 real
components) per direction, bit-exactly against `u*.npy`.

Task A fresh local gates:

- `cargo fmt --all -- --check`: passed.
- `cargo check --workspace`: passed.
- `cargo test -p gaugefields`: 132 passed, 1 ignored, including 19 doctests.
- `cargo test -p gaugefields --all-features`: 142 passed, 1 ignored, including
  19 doctests.
- `cargo test -p gaugefields --doc --all-features`: 19 passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`:
  passed.
- `cargo doc --workspace --all-features --no-deps`: passed.
- `cargo build --workspace --examples --all-features`: passed.
- ILDG integration suite: 5 passed; focused `ildg`-name filter: 11 passed.
- `git diff --check`: passed.

The workspace now sets `profile.dev.debug = 0` and `profile.test.debug = 0` to
avoid retaining hundreds of gigabytes of routine debug artifacts; debugging
can be enabled explicitly through Cargo profile environment overrides. During
Task A, 241.1 GiB of stale shared build artifacts and 71 GiB from an inactive
worktree were removed. A clean workspace check rebuilt the current dependency
set successfully.

The first Task A post-implementation full-diff review by `reviewer-flash`
recorded `Correct-to-merge` with five Minor findings: correct the Julia
component-count prose, classify tensor duplication failures as tensor errors,
reuse the canonical site-index helper, add README discoverability, and mention
the debug-profile tradeoff in the PR. All findings were applied. The delta
re-review found no remaining finding and recorded `Correct-to-merge`; the Task A
post-implementation gate is closed.
