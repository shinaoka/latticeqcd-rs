# Phase 2 quenched measurements

Status: approved for implementation

## Goal

Complete Issue #17 Phase 2 with a small CPU-first surface:

- pure-Rust ILDG 1.1 gauge-configuration I/O,
- symbolic signed-direction Wilson paths and general loop-action forces,
- Polyakov loop and clover topological charge,
- synchronous isotropic stout smearing,
- fixed-step third-order Runge--Kutta gradient flow for general loop actions,
- deterministic and frozen-ensemble comparison with the Julia ecosystem.

The existing normalized plaquette, periodic indexing, SU(3) matrix algebra,
Wilson staple, and evolution context are reused rather than reimplemented.

## Accepted scope

The user selected these boundaries before implementation:

- four-dimensional periodic lattices,
- host-resident SU(3) `Complex64` links,
- signed directions `-4..=-1, 1..=4`,
- one isotropic stout coefficient,
- fixed-step RK3 flow,
- deterministic Julia parity plus a short statistical comparison,
- one completed pull request after all Phase 2 work is integrated.

The implementation remains fallible and transactional. Invalid public input is
rejected before allocation, file creation, field mutation, or backend work.

## Non-goals

- GPU or MPI execution,
- general SU(N),
- ILDG Float32, reduced-row storage, multiple configurations, SciDAC records,
  checksums, or parallel LIME,
- Bridge++ or JLD2 I/O,
- adaptive flow integration,
- anisotropic or trainable stout coefficients,
- a path string parser, canonicalizer, optimizer, or expression DSL,
- fermionic measurements,
- compatibility shims for old tenferro APIs.

Unknown non-conflicting LIME metadata records are skipped without interpretation;
the required ILDG records remain strict. The deferred formats and features are
not represented by placeholder APIs.

## Pinned references and provenance

The numerical conventions are validated against exact clean checkouts:

| Project | Revision | Role |
|---|---|---|
| Gaugefields.jl v0.7.2 | `9e5719970770f4497405a856315c90bef7f74449` | field layout, ILDG, Polyakov, stout, RK3 flow |
| Wilsonloop.jl 0.1.5 source | `e1a617fdedb19b785f89bdeb13c30e53b20743a7` | signed paths, adjoints, plaquette/clover loops, staples |
| QCDMeasurements.jl v0.2.13 | `9e04c37bbd68712cf7a749ae5aff10eb6aae4566` | clover topological-charge convention |

All three are MIT-licensed. New crates retain the applicable upstream MIT
notice, source files identify the referenced file/function, and the repository
README credits Akio Tomiya and Yuki Nagai. The topological-charge implementation
also cites arXiv:1509.04259. Stout documentation cites Morningstar and Peardon,
Phys. Rev. D 69, 054501 (2004); gradient-flow documentation cites Lüscher,
JHEP 1008:071 (2010), arXiv:1006.4518.

The implementation follows the algorithms and conventions, but does not copy
Julia-specific allocation, mutation, temporary-pool, assertion, or global-state
machinery.

## Dependency and ownership graph

```text
gaugefields
    ^
    |
wilsonloop
    ^
    |
measurements
```

`measurements` may also depend directly on `gaugefields`. No lower crate depends
on an upper crate. The `CpuEvolutionContext` **type** is already owned and
exported by `gaugefields`; "application-owned" and "caller-owned" below refer
to ownership of each context instance, not to ownership by another crate.
Consequently Task C names only a same-crate type, while `measurements` obtains
the type through its existing downward dependency on `gaugefields`.

### `gaugefields`

Owns:

- `GaugeLinks`, compact `[3,3,NX,NY,NZ,NT]` column-major storage,
- checked periodic site traversal and a validated borrowed host view,
- `Mat3`, traceless anti-Hermitian projection, and SU(3) exponential,
- pure-Rust ILDG I/O,
- fixed plaquette stout smearing.

### `wilsonloop`

Owns:

- signed Wilson paths,
- path adjoint and displacement,
- path/loop evaluation,
- closed real loop actions,
- analytic per-link loop-action force.

It does not own gauge storage, ILDG, stout, flow integration, or measurements.

### `measurements`

Owns:

- Polyakov-loop normalization,
- clover topological-charge construction and normalization,
- gradient-flow parameters and RK3 integration over a `wilsonloop::LoopAction`.

It calls `gaugefields::normalized_plaquette`; it does not wrap or duplicate the
plaquette implementation.

## Shared validated host boundary

The private `PreparedGaugeField` is generalized into a public read-only
`HostGaugeLinks<'a>` returned by:

```rust
impl GaugeLinks {
    pub fn host_view(&self) -> Result<HostGaugeLinks<'_>, GaugeError>;
    pub fn try_clone(&self) -> Result<Self, GaugeError>;
}
```

`HostGaugeLinks` validates SU(3), rank, shape, placement, volume, and strides
once, borrows each direction once, and exposes only:

```rust
pub fn lattice(&self) -> LatticeShape4;
pub fn link(&self, direction: usize, site: usize) -> Result<Mat3, GaugeError>;
pub fn shifted_site(
    &self,
    site: usize,
    direction: usize,
    displacement: isize,
) -> Result<usize, GaugeError>;
```

Existing Wilson kernels use the same type. No downstream crate repeatedly asks
a tensor for host storage in a site loop. `HostGaugeLinks` adds no raw-slice
accessor. The pre-existing `GaugeLinkTensor::typed()` tensor-interop API remains
public for compatibility, but Phase 2 production kernels must use
`host_view()`; a source audit enforces that boundary rather than removing the
intentional tensor API.

## Task A: pure-Rust ILDG 1.1

### Public API

```rust
pub fn read_ildg(path: impl AsRef<Path>) -> Result<GaugeLinks, GaugeError>;
pub fn write_ildg(
    path: impl AsRef<Path>,
    links: &GaugeLinks,
) -> Result<(), GaugeError>;
```

The implementation uses `std::io` for LIME and `quick-xml` only for XML. It
does not bind or execute c-lime. Path-based APIs keep the public surface small;
private reader/writer helpers operate on streams for focused tests.

### Canonical writer

One LIME message contains, in order:

1. `ildg-format`,
2. `ildg-binary-data`.

Every record has the standard 144-byte LIME header: big-endian magic
`0x456789ab`, version 1, the MB/ME flag word, a `u64` payload length, and a
128-byte NUL-padded ASCII record type, followed by payload and 8-byte alignment
padding. `ildg-binary-data` begins immediately with gauge Float64 values: it has
no additional id, length, checksum header, or trailer inside its payload. The
reader requires exactly one binary record and rejects a payload split across
multiple records. The canonical two-record writer sets MB only on
`ildg-format` and ME only on `ildg-binary-data`.

Checksums remain outside the accepted scope: a `scidac-checksum` record, like
other non-conflicting metadata, is streamed past without verification, and the
API documentation does not claim corruption detection beyond structural,
length, finiteness, and I/O checks.

The XML declares ILDG format version 1.0, `su3gauge`, precision 64, and
`lx,ly,lz,lt`. The binary payload is uncompressed big-endian IEEE Float64 in
slowest-to-fastest order:

```text
t, z, y, x, direction, row, column, real/imag
```

Equivalently, x is the fastest site coordinate and the Rust tensor's contiguous
column-major matrix block is transposed at the explicit ILDG row/column
boundary as required by the standard. A Julia fixture detects either color
index being reversed.

### Strict reader contract

Before constructing a field, reject with typed errors:

- invalid LIME magic, version, reserved flags, record type, message begin/end,
  or truncated header/data/padding,
- `ildg-binary-data` before `ildg-format`, duplicate, split, or missing required
  records, a second message/configuration, or trailing bytes after message end,
- oversized XML, malformed XML, duplicate/missing fields, unsupported field,
  version, precision, dimension, or non-positive/overflowing extents,
- payload length mismatch, non-finite components, or unsupported placement.

LIME's 8-byte record padding is consumed but its byte value is not constrained.
XML comments, whitespace, namespaces, and unknown top-level metadata elements
are tolerated; known fields must each occur exactly once, and an unsupported
value of the known `<field>` element is rejected. Unrecognized LIME records in
the one message are streamed past without allocation. The expected payload
length and allocation bounds are checked from XML before reading numeric data.
Float32 is intentionally rejected rather than silently promoted.

All validation that can fail is completed before `write_ildg` creates or
truncates its destination. An I/O failure can leave a partial output file and is
reported with the path; atomic replacement is not claimed.

### Compatibility evidence

A Julia generator writes a standards-complete LIME message using the pinned
Gaugefields.jl layout and c-lime tooling. Rust reads every component bit-exactly.
Rust writes the same field; pinned Julia loads it with explicit dimensions and
matches every component locally. CI additionally checks Rust write/read
bit-exact round trip, a malformed-file table, and that invalid links fail
before a previously absent destination path is created.

## Task B: signed Wilson paths and loop actions

### Public API

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WilsonPath { /* validated non-empty signed steps */ }

impl WilsonPath {
    pub fn new(steps: impl Into<Vec<i8>>) -> Result<Self, WilsonError>;
    pub fn steps(&self) -> &[i8];
    pub fn displacement(&self) -> [isize; 4];
    pub fn is_closed(&self) -> bool;
    pub fn adjoint(&self) -> Self;
}

#[derive(Clone, Debug)]
pub struct LoopTerm { /* finite f64 coefficient and closed path */ }

#[derive(Clone, Debug)]
pub struct LoopAction { /* non-empty terms */ }

pub fn evaluate_path(
    links: &GaugeLinks,
    origin: usize,
    path: &WilsonPath,
) -> Result<Mat3, WilsonError>;

pub fn loop_trace_sum(
    links: &GaugeLinks,
    path: &WilsonPath,
) -> Result<Complex64, WilsonError>;

pub fn loop_action_value(
    links: &GaugeLinks,
    action: &LoopAction,
) -> Result<f64, WilsonError>;

pub fn loop_action_force(
    links: &GaugeLinks,
    action: &LoopAction,
) -> Result<TaGaugeField, WilsonError>;
```

Constructors, not public fields, enforce invariants. A `LoopTerm` coefficient
is a real `f64`; complex action coefficients are not accepted. `LoopTerm` means

```text
coefficient * sum_x Re tr(path at x)
```

so callers never need to append an adjoint path merely to express a real
action. Helpers construct the positive oriented plaquette terms and the two
1x2 rectangle orientations used by Gaugefields.jl. A length-two segment is
represented by repeating its validated unit direction, for example
`[+1,+1,+2,-1,-1,-2]`; direction magnitude always names an axis and never a
fat-step length. No named-string registry is added.

### Path convention

For a forward step `+(mu+1)` at site x, multiply `U_mu(x)` and then move to
`x+mu`. For a backward step `-(mu+1)`, first move to `x-mu` and multiply
`U_mu(x-mu)^dagger`. `adjoint` reverses the sequence and flips every sign.
Every move is periodic.

`WilsonPath` permits open paths because staple derivation requires them.
`LoopTerm` requires exact zero displacement. Empty paths, zero directions,
absolute directions above four, non-finite coefficients, empty actions, and
checked displacement/allocation overflow are typed errors.

### Analytic force

For each occurrence of a link, cyclically rotate the loop product to the
varied link. Under the left variation `U -> exp(t A) U`:

- a forward occurrence contributes `TA(U * after * before)`,
- a backward occurrence contributes `-TA(after * before * U^dagger)`.

Contributions are multiplied by the term coefficient and accumulated into the
existing eight-component `TaGaugeField` convention. This is the **positive**
action gradient: Gaugefields.jl `calc_dSdUμ!` adds `beta * staple`, and
`F_update!` forms `TA(U * dS/dU)` without a minus sign. The negative flow
direction appears only in the RK3 stage coefficients below. For a real Julia
coefficient `f`, `Gradientflow_general` inserts both `f * W` and `f * W†`; this
maps to one Rust term with `coefficient = 2*f` under the documented
`coefficient * Re tr(W)` meaning.

A centered finite-difference test for independent Gell-Mann directions, sites,
and both plaquette and rectangle terms fixes sign and normalization. It checks
the derivative of the documented action directly rather than comparing one
force implementation with another. A pinned Julia `calc_dSdU` fixture also
compares every force component for a multi-term plaquette-plus-rectangle action,
fixing accumulation and the positive-gradient convention end to end.

Path metadata and occurrence tables are compiled once per action construction;
site loops allocate no heap storage.

## Task C: stout smearing

### Public API

```rust
pub fn stout_step(
    context: &mut CpuEvolutionContext,
    links: &GaugeLinks,
    rho: f64,
) -> Result<GaugeLinks, GaugeError>;
```

`rho` must be finite; negative finite values remain mathematically valid. The
function requires host SU(3), computes all generators from the unchanged input,
and returns a new field.

For every link,

```text
C_mu(x)     = rho * sum of six plaquette staples,
Omega_mu(x) = C_mu(x) U_mu(x)^dagger,
Q_mu(x)     = TA(Omega_mu(x)),
U'_mu(x)    = exp(Q_mu(x)) U_mu(x).
```

The existing `HostGaugeLinks::force_staple`, `Mat3::ta`, coefficient mapping,
`exp_ta_update`, and caller-owned `gaugefields::CpuEvolutionContext` are reused.
`force_staple` is pinned here to the unweighted positive geometric sum
`+sum U_nu U_mu U_nu^dagger`: it contains neither beta nor an action-gradient
minus sign and is therefore sign-identical to stout `C_mu/rho`. A direct
term-level test fixes this orientation and sign before the full Julia parity
test. There is no hidden backend construction and no in-place directional
update. Failure leaves the input unchanged. The context is reusable scratch;
its cache/scratch state after a mid-operation backend failure is valid but
unspecified and is not rolled back.

## Task D: measurements and gradient flow

### Public measurement API

```rust
pub fn polyakov_loop(links: &GaugeLinks) -> Result<Complex64, MeasurementError>;
pub fn clover_topological_charge(
    links: &GaugeLinks,
) -> Result<f64, MeasurementError>;
```

The temporal direction is axis 3. The Polyakov loop is

```text
(1 / (NX NY NZ)) sum_xyz tr product_t U_3(x,y,z,t),
```

matching Gaugefields.jl: it is not divided by `NC`.

For every ordered pair `mu != nu`, the clover field is the traceless
anti-Hermitian part of the sum of the four oriented plaquettes around the site.
The epsilon tensor is the ordinary four-dimensional Levi-Civita tensor with
`epsilon(0,1,2,3) = +1` (the zero-based Rust form of Julia
`epsilon(1,2,3,4) = +1`); no separate "extended epsilon" convention is used.
With `C_mu_nu` denoting that sum,

```text
Q = -1/(32 pi^2) * sum_x,mu,nu,rho,sigma
    epsilon(mu,nu,rho,sigma)
    Re tr(TA(C_mu_nu) TA(C_rho_sigma)) / 4^2.
```

The returned value is real. This deliberately implements only the Issue #17
clover definition, not the Julia plaquette or rectangle-improved variants.

### Gradient-flow API

```rust
#[derive(Clone, Copy, Debug)]
pub struct GradientFlowParams { /* positive finite step, nonzero steps */ }

pub fn gradient_flow(
    context: &mut CpuEvolutionContext,
    links: &GaugeLinks,
    action: &LoopAction,
    params: GradientFlowParams,
) -> Result<GaugeLinks, MeasurementError>;
```

The function clones the input fallibly and applies the pinned Gaugefields.jl
RK3 sequence for each step, with `F(U) = loop_action_force(U, action)`:

```text
F0 = F(U)
W1 = exp(-(eps/4) F0) U
F1 = F(W1)
W2 = exp(eps*(-8/9 F1 + 17/36 F0)) W1
F2 = F(W2)
U' = exp(eps*(-3/4 F2 + 8/9 F1 - 17/36 F0)) W2
```

The force combinations are accumulated in coefficient space and each
exponential update uses the caller-owned `gaugefields::CpuEvolutionContext`.
Parameters and the complete input/action contract are validated before cloning.
Failure does not mutate the input. The context is reusable scratch; its
cache/scratch state after a mid-flow failure is valid but unspecified and need
not be restored.

## Validation fixtures

### Deterministic fixture

A focused mode in `fixtures/generate.jl` uses the pinned Julia revisions and a
nontrivial reproducible `2x2x2x2` SU(3) field. It records:

- a standards-complete ILDG file and every original link,
- representative forward, backward, open, plaquette, rectangle, and clover
  path values,
- Polyakov loop and clover topological charge,
- all links after one isotropic stout step,
- all links after one and four RK3 Wilson-flow steps,
- all links after a mixed plaquette-plus-rectangle flow step,
- source files/functions, revisions, parameters, layouts, and tolerances.

The focused and default generators must each produce identical complete fixture
trees on two consecutive runs. The frozen-ensemble block means below are part
of the default generator tree, so the same twice-run check covers them without
a third generator mode. Maximum absolute field residual is fixed before
generation at `5e-12`; scalar residual is `2e-12`. ILDG input links are exact.
Cold-field tests provide exact identities. A non-cubic `2x3x2x4` Rust test fixes
the temporal-axis and periodic-layout behavior of Wilson paths, Polyakov, and
clover; full-field Julia parity on `2x2x2x2` covers stout/flow periodic traversal.

### Frozen-ensemble statistical tier

The existing Phase 1 `2^4`, beta=5.7 heatbath schedule is reused:

- 512 burn-in sweeps,
- 32 blocks,
- 32 measured sweeps per block,
- the same documented Julia and Rust RNG states and sampler parameters.

The Phase 2 generator records independent Julia block means for normalized
plaquette, real and imaginary Polyakov loop, clover `Q`, and `Q^2`. Rust runs
its independent seeded chain and requires, for each non-degenerate observable,

```text
abs(mean_rust - mean_julia)
    <= 6 * sqrt(se_rust^2 + se_julia^2).
```

For the positive `Q^2` observable, the mean difference must additionally be no
more than 25% of the larger nonzero mean; this prevents a large standard error
from making the six-sigma gate vacuous. All values and errors must be finite,
every measured field must differ from cold, and at least the plaquette,
Polyakov magnitude, and `Q^2` block series must have nonzero variance. This is
a deterministic CI test over two independent
but fixed stochastic streams; no network or live Julia installation is needed.
Stout and flow are not given a second stochastic acceptance gate because their
full-field deterministic comparisons are strictly stronger for deterministic
transforms.

## Tests and examples

Minimum coverage:

- malformed LIME/ILDG table plus bit-exact read/write round trip,
- Julia ILDG component parity and local Julia readback of Rust output,
- path validation, adjoint involution, exact displacement, cold identities,
  backward-link and periodic-wrap examples,
- analytic loop force against centered finite differences,
- stout cold identity, direct positive-staple sign/orientation, positive- and
  negative-rho Julia full-field parity, SU(3) residual, transactionality,
- Polyakov and topology cold values, non-cubic temporal convention, Julia
  scalar parity,
- RK3 cold identity, Julia one/four/mixed-action full-field parity, positive
  step validation, and Wilson-flow plaquette monotonicity,
- the frozen-ensemble combined-error test,
- runnable `ildg_roundtrip` and `quenched_measurements` examples with known-value
  assertions.

Public items receive runnable doctests. Private details stay in module-local
unit tests; public contracts use integration tests.

## Implementation and review sequence

No implementation begins until this complete design receives a recorded
`Correct-to-merge` verdict from `reviewer-flash`.

1. Task A: shared host view and ILDG I/O; focused tests; full Task A diff review.
2. Task B: `wilsonloop` paths/actions/force; focused tests; full Task B diff review.
3. Task C: stout; focused tests; full Task C diff review.
4. Task D: measurements/flow/fixtures/statistics/docs; focused tests; full Task D
   diff review.
5. Integrated full-branch review, complete local gate, push, one PR, hosted CI.

Every finding is fixed and re-reviewed before the next independently mergeable
task starts. Each task is committed only after its post-implementation verdict.
The PR is not merged without a separate user request.

## Completion gate

Run on the exact final tree:

```text
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --doc --workspace --all-features
cargo doc --workspace --all-features --no-deps
all checked examples
focused Julia generator twice and default generator twice
focused deterministic and statistical parity tests
git diff --check and provenance/license/artifact/stale-symbol audits
```

The worklog records exact counts, residuals, checksums, reviewer verdicts,
GitHub CI, and the final PR state.
