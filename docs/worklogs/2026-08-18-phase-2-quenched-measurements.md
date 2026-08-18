# Phase 2 quenched measurements worklog

Status: Task C complete; Task D implementation pending

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
| B: Wilson paths/actions/force | Phase 2 design, Task B | reviewer-flash | Correct-to-merge | Correct-to-merge |
| C: stout | Phase 2 design, Task C | reviewer-flash | Correct-to-merge | Correct-to-merge |
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

## Task B implementation

Task B adds the downward-only `wilsonloop` crate. `WilsonPath` owns a nonempty
validated sequence of signed unit directions `±1..±4`, checked displacement,
adjoint, plaquette, and both 1x2 rectangle helpers. `LoopTerm` requires a finite
real coefficient and closed path. `LoopAction` is nonempty, `Clone + Debug`, and
precompiles one occurrence table per term. Public evaluation obtains one
`GaugeLinks::host_view()` and never exposes or reindexes tensor storage. Force
execution allocates prefix/suffix matrix scratch once per term and reuses it;
site loops allocate no heap and evaluate each path in linear rather than
quadratic path length.

The scalar convention is exactly `c * sum_x Re tr(W)`. Pinned Julia inserts
`f*W + f*W†`, hence `c=2*f`. Fixture comparison exposed an important
holomorphic-derivative detail that the approved design text had stated too
coarsely: Gaugefields.jl `calc_dSdU` contributes `f=c/2` per occurrence to
`TA(U*dS/dU)`. The implementation and corrected design therefore use
`+(c/2) TA(U*after*before)` for forward occurrences and
`-(c/2) TA(after*before*U†)` for backward occurrences. Under
`U -> exp((i/2) sum_a(v_a lambda_a)t) U`, the independent real scalar action
obeys `dS/dt = -sum_a(F_a v_a)`. The negative flow sign remains owned by the
future RK3 stage coefficients, not by `loop_action_force`.

The only general support added to `gaugefields::TaGaugeField` is checked zero
allocation, checked per-site coefficient accumulation/readback, and existing
tensor interop access. The Wilson implementation reuses `HostGaugeLinks`,
`Mat3`, its TA/Gell-Mann mapping, and `TaGaugeField`; it adds no alternate
matrix, indexing, plaquette, or staple path.

### Task B deterministic oracle

`fixtures/generate.jl wilsonloop_task_b` pins clean checkouts of:

- Gaugefields.jl v0.7.2 `9e5719970770f4497405a856315c90bef7f74449`
- Wilsonloop.jl v0.1.5 `e1a617fdedb19b785f89bdeb13c30e53b20743a7`

The oracle uses the direction-distinct reproducible `2^4` SU(3) links and all
six `(mu,nu)` planes. Every plane contains one plaquette and both 1x2 rectangle
orientations, for 18 Rust terms; coefficients are `c=0.73` and `c=-0.31`
(`f=0.365` and `f=-0.155` in Julia). Thus all four force directions are
nonzero. Rust checks every source link, every Julia `calc_dSdU` matrix, every
stored Julia TA coefficient, and every one of the `4*16*8` Rust force
coefficients. Maximum residuals were:

- Julia `TA(U*calc_dSdU)` versus stored Julia coefficients:
  `4.44089209850062616e-16`
- Rust force versus stored Julia coefficients:
  `6.66133814775093924e-16`
- independent centered Gell-Mann left variations over plaquette and mixed
  plaquette/rectangle actions: `4.10362410718789761e-10`

Both focused generator runs produced Task B tree hash
`c93b425937b798a0026091db768dca9eba4e3886d832c07a39ff28d24a9067d1`.
Both complete default generator runs produced fixture-tree hash
`0338314f862cd4474e9b283d0b47cb454343a3b7221c144ab413da0efc6e5bf9`.
Commands set `GAUGEFIELDS_JL_DIR`, `WILSONLOOP_JL_DIR`, and
`JULIA_NUM_THREADS=1` explicitly.

Task B fixture SHA-256 values:

- `dsdu0.npy`: `bdaadc3c51dac7543a601407fccb9ed328dff9c09b350bd3b2227be2759262c4`
- `dsdu1.npy`: `8972791ff2a8e8b4ce9b1acbd86c9001c40d0e87f56c0770eb7ebf50aa497251`
- `dsdu2.npy`: `25d8840fa61d92efa806476f619ec149b4b2211b5f81c257c808e8c3462b546b`
- `dsdu3.npy`: `77489a69b867f164338ae017dce17c90b9eae394ba6722ee912b08df47643c8d`
- `force_coeff0.npy`: `776ee29be6138d0e767e5e1e951ef4b2d63df20a77b3867c64d0bcb4d99bba9b`
- `force_coeff1.npy`: `c4c367c7bc0c1d2f5861956baf2213667245fba0e11750d6c40683ce04a1c4c1`
- `force_coeff2.npy`: `10bbb3b7b353aeb2bd8bf5a270234caee5d071c103fd0013bc064a5877d87788`
- `force_coeff3.npy`: `7eaee778002bc01f914e51fc07daf7a1901e77dda0a7a81e527185396734f089`
- `metadata.json`: `8239fc58bf70382056e0aabfa9772e6e16dbbc580a6277c417bc10c633f5758b`
- `u0..u3.npy`: exactly the existing direction-distinct `random_2x2x2x2`
  values (`2d767b...`, `8ade4c...`, `1efe6d...`, `dd432d...`).

### Task B local gates

- `cargo fmt --all -- --check`: passed.
- `cargo check --workspace`: passed.
- `cargo test --workspace`: 160 passed, 1 ignored, including 33 doctests.
- `cargo test --workspace --all-features`: 170 passed, 1 ignored, including
  33 doctests.
- `cargo test --workspace --doc --all-features`: 33 passed.
- `cargo test -p wilsonloop --all-features`: 25 passed, including 12 doctests.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`:
  passed.
- `cargo doc --workspace --all-features --no-deps`: passed.
- `cargo build --workspace --examples --all-features`: passed.
- `git diff --check`: passed.

The first Task B full-diff review by `reviewer-flash` recorded
`Correct-to-merge` with two Minor findings: add explicit Gaugefields.jl
attribution to the new crate license and avoid recomputing every before/after
product quadratically. The license now carries both notices, and the force uses
term-owned prefix/suffix scratch allocated outside the site loop. The complete
Rust gate and numerical comparisons remained green. The delta re-review found
no remaining finding and recorded `Correct-to-merge`; the Task B
post-implementation gate is closed.

## Task C implementation

Task C adds `gaugefields::stout_step` and exposes the existing
`HostGaugeLinks::force_staple` as the documented unweighted positive six-term
plaquette staple. The implementation validates finite `rho`, host SU(3)
storage, every input component, and allocation bounds before allocating the TA
field or invoking the backend. It builds all `Q_mu(x)` coefficients from one
unchanged input view, clones only after those generators are complete, and
uses the caller-owned `CpuEvolutionContext` through the existing
`exp_ta_update`. Finite negative and zero `rho` values are valid. Input links
remain bitwise unchanged on success and every failure. Finite input whose
intermediate TA coefficients overflow is classified as
`Su3NumericalRange { operation: "stout_step", stage: "TA coefficients" }`, not
as a non-finite input.

The exact convention is
`C_mu=rho*force_staple`, `Omega=C_mu*U_mu^dagger`, `Q_mu=TA(Omega_mu)`, and
`U'_mu=exp(Q_mu)U_mu`. A direct independent six-term test fixes every staple
orientation and its positive sign with zero residual.

### Task C deterministic oracle

`fixtures/generate.jl stout_task_c` uses clean pinned Gaugefields.jl v0.7.2 at
`9e5719970770f4497405a856315c90bef7f74449` and its `stout_fast.jl`,
`stout_dataset.jl`, `Abstractsmearing.jl`, and `HMCstout_test_nowing.jl`
conventions. It applies one synchronous isotropic layer to the existing
direction-distinct `2^4` field at `rho=0.12` and `rho=-0.07`. Rust compares all
four input directions bit-exactly and every complex output component. Results:

- `rho=0.12`: Julia maximum residual
  `2.41409693502886844e-15`; unitarity `6.32443542595004009e-15`;
  determinant `1.22195478062137133e-15`.
- `rho=-0.07`: Julia maximum residual
  `1.92026915295028663e-15`; unitarity `5.37168993881695668e-15`;
  determinant `8.98688895805620469e-16`.
- direct positive-staple residual: exactly `0.0`.

Both focused generator runs produced Task C tree hash
`4a52e3915d3444f221ec1a8bf970cfa17f405f51a9a678a242d6b4018ba4020e`.
Both complete default generator runs produced fixture-tree hash
`f45a2e94cd564eff96d55ec4ba4b720e8a708676737115ac2ef2c9e3a68a8c51`.
The invocation set `GAUGEFIELDS_JL_DIR`, `WILSONLOOP_JL_DIR`, and
`JULIA_NUM_THREADS=1` explicitly.

Task C fixture SHA-256 values:

- `metadata.json`: `33f2b4e565854619d6c504b7518169e7a2700deb88c5d5652dbb82d663ea192c`
- `stout_plus0.npy`: `8381709f3f2891d9b4265d94e323508c478f78a92f87619bf5f7bbcdf7b15b1f`
- `stout_plus1.npy`: `ad8ed1a5e1fa8dde9f943678efb475e04908c1e7124d6b2a81b28364af2d123f`
- `stout_plus2.npy`: `16de73bc659ef54698cf55ee971e717b9e42fa2d262579e53e8d63e7f18f3e08`
- `stout_plus3.npy`: `de17150715bc3c847c2dd7ae4c56d810ce73663518e47cda92f61d136abe5084`
- `stout_minus0.npy`: `dcf2fe7aca9e0a57f43273b6a016639d126f6a300094a00f7e74d24adca0abfa`
- `stout_minus1.npy`: `3375e4620f99bdd759fe723537a22082541f7d79a9b82e4b3b51d3763e9e7c5b`
- `stout_minus2.npy`: `6c034d9212b27a6e244bbfeb6e44648fbc375cab898d3932879a1737439f772f`
- `stout_minus3.npy`: `d238ec86b46447e8780c441946f18372babec58b8ec01965aa74afd66a922ea2`
- `u0..u3.npy`: bit-exact copies of the established direction-distinct fixture.

### Task C local gates

- Stout integration suite: 7 passed.
- `cargo fmt --all -- --check`: passed.
- `cargo check --workspace`: passed.
- `cargo test --workspace`: 169 passed, 1 ignored, including 35 doctests.
- `cargo test --workspace --all-features`: 179 passed, 1 ignored, including
  35 doctests.
- `cargo test --workspace --doc --all-features`: 35 passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`:
  passed.
- `cargo doc --workspace --all-features --no-deps`: passed.
- `cargo build --workspace --examples --all-features`: passed.
- `git diff --check`: passed.

The first Task C full-diff review by `reviewer-flash` recorded
`Correct-to-merge` with two Minor findings: reuse the established checked
allocation helper and remove a redundant final finiteness scan already
guaranteed by `exp_ta_update`. `force::checked_count` is now the crate-private
shared helper, and the unreachable output rescan is removed. The delta
re-review found no remaining finding and recorded `Correct-to-merge`; the Task C
post-implementation gate is closed.
