# Phase 6–8: tenferro graph integration, autodiff, and SU(3) evolution

Status: implemented through Phase 8
Date: 2026-07-13

## 1. Purpose and delivery

This document defines the implementation contract for:

- [Phase 6: tenferro extension integration](https://github.com/shinaoka/gaugefields-rs/issues/8)
- [Phase 7: autodiff rules](https://github.com/shinaoka/gaugefields-rs/issues/9)
- [Phase 8: SU(3) evolution kernels and HMC sanity checks](https://github.com/shinaoka/gaugefields-rs/issues/10)

The phases ship as three sequential PRs. Phase 6 establishes storage and
runtime boundaries, Phase 7 adds differentiation without changing the numerical
formula, and Phase 8 builds evolution kernels on the stabilized tensor/runtime
surface. Each PR must pass independently; a single Phase 6–8 mega-PR is outside
this design.

At design time, tenferro `origin/main` is
[`51bc0a7bef274e20d08fc054856cb4d74c284cbe`](https://github.com/tensor4all/tenferro-rs/commit/51bc0a7bef274e20d08fc054856cb4d74c284cbe).
Implementation must fetch `origin/main`, record the then-current exact revision,
and update all tenferro-family pins atomically. A floating Git branch is not a
reproducible dependency. The role-split extension AD API required here landed
through [tenferro issue #1301](https://github.com/tensor4all/tenferro-rs/issues/1301).

The Julia semantic reference is Gaugefields.jl commit
[`9e5719970770f4497405a856315c90bef7f74449`](https://github.com/shinaoka/Gaugefields.jl/tree/9e5719970770f4497405a856315c90bef7f74449).
In particular, Phase 8 follows:

- [`exptU!` for four-dimensional TA fields](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/TA_gaugefields_4D_serial.jl#L603-L806), including its analytic and fallback branches;
- [`normalize_U!` for NC=3 no-wing fields](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/nowing/gaugefields_4D_nowing.jl#L2387-L2454);
- [`MDstep!`, `U_update!`, and `P_update!`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/HMC_test_nowing.jl#L17-L72) for the test-only leapfrog convention.

## 2. Scope and non-goals

In scope:

1. traced Wilson action execution through an explicitly registered tenferro
   host-reference extension;
2. first-order JVP and reverse gradient via graph-level role-split AD rules;
3. exact Julia-compatible `exp_ta`, SU(3) normalization, and field updates;
4. a crate-private HMC driver used only to validate the public kernels;
5. migration of Phase 0–5 storage and hot paths to the correct typed boundary.

Out of scope:

- a public HMC sampler, integrator framework, trajectory state, or RNG policy;
- gradient flow;
- higher-order derivatives of the force operation;
- GPU kernels or implicit device fallback;
- replacing the exact Julia `exp_ta` algorithm with a generic eigensolver;
- a process-global runtime, backend, extension registry, or AD registry.

## 3. Audit of the Phase 0–5 implementation

The existing formulas are broadly suitable, but the ownership boundary must be
corrected before graph integration.

| Area | Decision | Phase 6 action |
|---|---|---|
| `GaugeLinkTensor` stores erased `Tensor` | Wrong internal boundary | Store `TypedTensor<Complex64>` and validate rank/shape in the wrapper |
| `GaugeForce` is `[Tensor; 4]` | Too erased and duplicates the future momentum type | Replace/unify with `TaGaugeField` backed by `[TypedTensor<f64>; 4]` |
| observables repeatedly call `load_link` | Repeats host-slice access in the site loop | Validate and borrow each direction once, then call a shared slice kernel |
| force borrows slices and computes periodic neighbors | Correct | Retain with constant-size prepared lattice metadata |
| explicit unrolled `Mat3` arithmetic | Appropriate fixed-size SU(3) leaf | Retain; document the fixed-size and Julia-parity rationale |
| periodic action/force stencils | No matching tenferro gather/stencil primitive | Retain as custom host kernels; avoid materialized lattice shifts |
| NPY fixture reader and site indexing | Domain/fixture responsibility | Retain |
| `GaugeIdentityOp` | Only an integration spike | Remove when real Phase 6 families land |
| dependency pins | Behind design-time tenferro main | Update atomically to one exact compatible snapshot |

Direct Phase 0–5 APIs remain host-only. They require neither a graph executor nor
an AD context. Their rustdoc must state that backend buffers require an explicit
download; no direct API may silently instantiate a CPU backend or transfer data.

## 4. Canonical type and ownership boundary

The word “tensor” identifies three distinct roles:

| Role | Type | Reason |
|---|---|---|
| owned numerical storage | `TypedTensor<Complex64>` / `TypedTensor<f64>` | dtype known after construction; avoids repeated erased dispatch |
| borrowed numerical kernel input | typed view or validated slice | validates once and avoids per-site extraction/materialization |
| tenferro extension ABI | `Tensor` | official `HostReference` ABI is dtype-erased and permits mixed C64/F64 inputs/outputs |
| traced graph value | `TracedTensor` | represents graph values and symbolic metadata, not materialized storage |

Use dynamic-rank `TypedTensor<T>` internally. The gauge wrappers enforce the
semantic ranks (links rank 6, Lie-algebra coefficients rank 5) and extents. A
static rank parameter would strengthen isolated constructors but complicate
conversion to and from tenferro's dynamic `Tensor` extension ABI without
improving graph validation.

### 4.1 Data layouts

- `GaugeLinks[mu]`: C64 shape `[3, 3, Lx, Ly, Lz, Lt]`.
- `TaGaugeField[mu]`: F64 shape `[8, Lx, Ly, Lz, Lt]`.
- Both are compact column-major, with color/algebra components leftmost and
  lattice coordinates acting as batch dimensions.
- The four directions must share lattice extents.

`GaugeLinks`, `GaugeLinkTensor`, and `TaGaugeField` own typed tensors. Public
constructors validate dtype (when converting from erased input), rank, all
extents, compact host placement where required, and checked element counts.

## 5. tenferro use policy

The implementation uses tenferro where it supplies the intended abstraction,
not merely at the outermost API.

| Concern | Required mechanism |
|---|---|
| typed storage and views | tenferro `TypedTensor` / typed views |
| extension ABI | tenferro `Tensor` |
| graph construction | `TracedTensor` and graph compiler APIs |
| execution and caching | `GraphExecutor<CpuBackend>` |
| extension dispatch | `HostReferenceRuntime` with explicit registration |
| differentiation | `AdContext` and role-split graph rules |
| batched `exp(tP) U` | tenferro `dot_general` |
| expressible tensor reductions | tenferro standard operations |
| periodic Wilson stencil | custom gauge kernel; no native periodic stencil/gather operation |
| site-local `exp_ta` | custom fixed SU(3) kernel for exact Julia algorithm/fallback parity |
| `normalize_su3` | custom fixed SU(3) projection kernel |

A new custom tensor-sized loop requires a design/work-log justification. It may
not duplicate a tenferro backend operation or introduce hidden materialization.

## 6. Runtime ownership and initialization

There is no public `CpuEngine` type in the target tenferro API. For traced work,
the application owns a `GraphExecutor<CpuBackend>` and an `AdContext`. For
Phase 8 eager evolution, it owns a `CpuEvolutionContext`, which privately owns
a `CpuBackend` and its associated tenferro runtime cache.

They are constructed once per execution owner and reused:

- CLI/simulation: during program initialization, before traced work;
- server: once per worker or bounded executor pool, since execution takes
  mutable access;
- tests: once per test fixture when several graph executions belong together.

They are not constructed per operation and are not process-global singletons.
The executor owns graph runtime caches, extension registration, and workspace.
The AD context separately owns AD rules and transform caches. The evolution
context owns the backend buffer pool and persistent contraction-analysis cache.
CPU parallelism stays under each backend's `CpuContext`.

Illustrative application setup (exact builder names follow the pinned tenferro
revision):

```text
let backend = CpuBackend::new();
let mut executor = GraphExecutor::new(backend);
executor.register_extension(gaugefields::register_runtime)?;

let ad = AdContext::builder()
    .with_extension_rules(gaugefields::ad_rules()?)
    .build()?;

let mut evolution = CpuEvolutionContext::new(CpuBackend::new());
```

The library exposes registration functions but performs no hidden
initialization. The README contains the checked evolution setup using the exact
pinned API.

## 7. Shared numerical-kernel architecture

Direct APIs and extension runtimes must not carry separate Wilson formulas.
Both use this pipeline:

1. public/ABI validation converts raw inputs into prepared metadata and typed
   borrowed inputs;
2. a shared host kernel evaluates action, directional derivative, or force;
3. the direct wrapper returns domain types;
4. `HostReference` wraps typed results into erased `Tensor` variants.

Prepared metadata contains lattice shape, checked site count, and four validated
column-major site strides. Periodic neighbors use O(1) wrap arithmetic, so
auxiliary metadata remains O(1) rather than scaling with lattice volume. Hot
loops borrow storage once. Validation precedes zero-size shortcuts and
allocations.

Architectural review rule: prepared gauge metadata must not contain per-site
collections. Behavioral parity plus the large-lattice constant-size unit
regression enforce this boundary; production source text is not a test API.

Extension execution is intentionally host-reference in Phase 6. Receiving a
backend/GPU tensor returns a typed placement error. Missing extension runtime
registration remains tenferro's explicit missing-capability error and never
falls back to the direct kernel through an alternative dispatch path.

## 8. Phase 6 — extension integration

### 8.1 Public surface

Keep only the user-facing traced operation and registration public:

```text
pub fn wilson_action_traced(
    links: [&TracedTensor; 4],
    beta: f64,
) -> Result<TracedTensor>;

pub fn register_runtime<B: TensorBackend + 'static>(
    executor: &mut ExtensionExecutor<B>,
) -> Result<(), ExtensionRuntimeRegistryError>;
```

The JVP and force op constructors, validation records, runtime callbacks, and
payload structs are `pub(crate)`. If tenferro requires a public registration
contract, expose the narrowest function required rather than the concrete
runtime types.

### 8.2 Operation families

Stable family/version identifiers:

| Family | Inputs | Output | Payload |
|---|---|---|---|
| `gaugefields.wilson_action.v1` | four C64 links | scalar F64 | `beta` |
| `gaugefields.wilson_action_jvp.v1` | four C64 links plus active C64 tangents | scalar F64 | `beta`, ordered active directions |
| `gaugefields.wilson_force.v1` | four C64 links plus scalar F64 cotangent | four C64 link cotangents | `beta` |

Payload identity/hashing uses `beta.to_bits()` rather than lossy formatting.
The JVP active-direction list is sorted, unique, restricted to `0..4`, and part
of structural identity. Its input arity is `4 + active_dirs.len()`, so inactive
tangents are neither materialized nor represented by fake zeros.

Shape inference validates:

- four link inputs are C64 rank 6;
- their first two extents are exactly `3, 3`;
- contradictory known lattice constants are rejected;
- active tangents match their corresponding link dtype and shape;
- the force seed is scalar F64;
- action/JVP output is scalar F64 and force outputs match the four links.

Independent unresolved `TensorAxis` dimensions are accepted during inference:
tenferro cannot yet express equality constraints between separate placeholders
([tenferro-rs #1370](https://github.com/tensor4all/tenferro-rs/issues/1370)).
The executor validates placeholder bindings, and every host reference repeats
exact link/tangent dtype, rank, color-axis, and lattice-shape equality checks
before numerical execution. Phase 6 will adopt graph-level equality guards when
the upstream constraint mechanism becomes available.

One `register_runtime` call registers a `HostReferenceRuntime` for all three
families. It is deliberately shaped as the closure accepted by
`GraphExecutor::register_extension`, so applications call
`executor.register_extension(gaugefields::register_runtime)?`. The underlying
argument is the executor-owned `ExtensionExecutor<B>`. `lower_to_standard_ops`
returns no lowering in this phase. Public eager extension wrappers are not
introduced because the existing direct API is the explicit eager/host surface.

### 8.3 Phase 6 verification gate

- direct and traced action agree on identity, random, and Julia fixture fields;
- all family metadata and shape inference paths have positive and negative tests;
- wrong dtype/rank/color extent/lattice agreement/seed/active direction returns
  a typed error without panic;
- executing without `register_runtime` produces the expected missing-runtime error;
- registration is deterministic and duplicate handling follows tenferro's contract;
- traced execution does not call `GaugeIdentityOp`, which is removed;
- existing Phase 0–5 tests remain green after typed-storage migration.

## 9. Phase 7 — role-split autodiff

### 9.1 Rule graph

The canonical reverse path is deliberately compositional:

```text
Wilson action --linearize--> Wilson action JVP
Wilson action JVP --transpose linear op--> Wilson force
Wilson force --higher-order request--> typed unsupported error
```

The primal action rule emits the variable-arity JVP op with only active input
directions. The transpose rule for that linear JVP consumes a scalar cotangent
and emits cotangents only for active link inputs; tenferro supplies zeros or
absence according to its role contract for inactive/non-differentiable inputs.
The force kernel scales linearly with an arbitrary scalar cotangent, not merely
one.

Do not add an optional direct primal-VJP escape hatch in Phase 7. It would create
a second reverse path and weaken coverage of the intended
linearize-then-transpose architecture. Do not register a force AD rule;
higher-order differentiation is an explicit unsupported capability.

### 9.2 Public surface and ownership

`ad_rules()` returns the extension rule set needed by an application-owned
`AdContext`. Registration is explicit and repeatable for independent contexts;
there is no global registry. Operation-specific rule structs and dispatch
helpers remain crate-private.

### 9.3 Numerical oracle gate

Phase 7 is complete only when numerical behavior, not line coverage alone, is
demonstrated:

- gradient for every `mu` agrees with the Phase 5 `action_gradient` kernel;
- JVP agrees with centered finite differences for multiple random directions;
- JVP also equals the inner product of the Phase 5 gradient and tangent under
  the documented real/complex convention;
- reverse output scales correctly for non-unit and negative cotangent seeds;
- Julia-derived action/force fixtures retain links to their upstream source and
  revision;
- missing runtime and missing AD-rule registration produce distinct typed errors;
- requesting differentiation through the force produces the intentional
  higher-order unsupported error.

Finite-difference step size and tolerance are selected by an error sweep, not a
single convenient value. Failures print max absolute and relative/norm residuals.

## 10. Phase 8 — SU(3) evolution kernels

### 10.1 Public numerical API

The public surface consists of domain kernels, not a sampler framework:

```text
pub struct TaGaugeField { /* [TypedTensor<f64>; 4] */ }

pub struct CpuEvolutionContext { /* CpuBackend + associated runtime cache */ }

impl CpuEvolutionContext {
    pub fn new(backend: CpuBackend) -> Self;
    pub fn backend(&self) -> &CpuBackend;
    pub fn clear_cache(&mut self);
    pub fn cache_stats(&self) -> CacheStats;
}

pub fn exp_ta(t: f64, coeffs: &[f64; 8]) -> Result<Mat3>;
pub fn exp_ta_update(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    t: f64,
    momentum: &TaGaugeField,
) -> Result<()>;
pub fn normalize_su3(matrix: &mut Mat3) -> Result<()>;
```

`CpuEvolutionContext::new(CpuBackend)` initializes the associated
`<CpuBackend as BackendRuntimeCache>::RuntimeCache` with its bounded `Default`.
The cache field remains private. The context has a compact hand-written `Debug`,
`backend()` for read-only backend inspection, `clear_cache()`, and
`cache_stats() -> CacheStats`; it exposes no public mutable-backend escape.

`exp_ta_update` takes the application-owned evolution context, enters
`BackendSessionHost::with_backend_session_cached`, and uses stable contraction
cache slots `0..3` for link directions `mu=0..3`. Repeated same-shape updates
reuse the cache owner, backend context, provider choice, and buffer pool. The
pinned cpu-faer provider routes unconjugated multiplication through its strided
path, so `CacheStats` can remain at zero entries; providers that retain GEMM
analysis may populate those same slots. All four contraction outputs are validated before any link is
replaced, so an error leaves the field unchanged. The operation does not
construct a backend, cache, or graph executor internally. Phase 8 is an
explicit CPU/host API; a later device implementation requires a separately
designed placement-aware surface rather than an implicit transfer.

### 10.2 `exp_ta` algorithm

Port the Julia reference algorithm semantically:

1. construct the traceless anti-Hermitian `3 x 3` matrix from eight real
   coefficients using the established generator normalization;
2. compute the analytic eigenvalue parameters with the Cardano branch;
3. construct explicit eigenvectors and rotate their phases consistently;
4. assemble the exponential;
5. use the Julia fourth-order fallback in the same degenerate/near-degenerate
   region.

This must not be replaced by tenferro `eigh`: tenferro provides decomposition
but not the required matrix exponential, and exact branch/fallback compatibility
is part of the Julia oracle. Scalar fixed-size arithmetic stays in `Mat3`.

### 10.3 Field update and normalization

For all sites and directions, compute site-local `E = exp(t P)` and apply
`U <- E U`. The field multiplication is one batched tenferro `dot_general` per
appropriate packed direction/batch, with color axes as contraction dimensions
and lattice axes as batch dimensions. Do not implement the full field multiply
as a naive downstream loop. Packing or canonicalization, if required by the
selected API, is explicit, same-placement, measured, and recorded in the work
log.

`normalize_su3` projects a `Mat3` back to SU(3) using Julia's NC=3 row
Gram--Schmidt step and conjugated cross-product third row. Norms at or below
`1e-30` are treated as singular. It rejects non-finite or
numerically singular input with a typed error. Field-level normalization may be
crate-private until an external use case justifies another public API.

### 10.4 Test-only HMC driver

The HMC driver is crate-private test support. It uses the symmetric leapfrog
sequence

```text
U(dt/2) -> P(dt) -> U(dt/2)
```

and the Phase 5 action/force plus the public Phase 8 numerical kernels. Momentum
uses `P=(i/2) sum_a p_a lambda_a`, `K=(1/2) sum_a p_a^2`, and
`p <- p - dt*gauge_force/NC`. A fixed
seed makes regression runs reproducible. It exists to test:

- reversibility after momentum negation;
- Hamiltonian error scaling approximately as `dt^2` globally;
- bounded unitarity and determinant drift;
- finite values throughout several short trajectories;
- acceptance greater than `0.5` for a documented small `4^4` sanity setup.

The acceptance threshold is a regression sentinel, not a physics-quality or
performance claim. It does not justify exposing the driver as a public sampler.

### 10.5 Phase 8 oracle and verification gate

- `exp_ta(0, p)` is identity and small-`t` behavior matches the generator;
- random coefficient fixtures agree with linked Julia outputs;
- degenerate and near-degenerate fixtures prove the fallback branch;
- results satisfy unitarity and determinant-one tolerances;
- `exp_ta_update` agrees with a small explicit `Mat3` reference calculation;
- two same-shape updates report stable provider-dependent cache stats, and
  `clear_cache` resets retained entries to zero;
- injected contraction failure at each direction leaves all four links
  bit-for-bit unchanged;
- normalization fixtures cover drift, singular rejection, and non-finite input;
- HMC checks above pass under a fixed seed without becoming public API;
- tests demonstrate that batched update reaches tenferro `dot_general` and does
  not silently use a local full-field multiplication fallback.

## 11. Errors and diagnostics

Extend the crate error enum with structured variants (names finalized during
implementation) for:

- dtype, rank, shape, lattice, and direction mismatches;
- non-host placement at a host-reference boundary;
- invalid/non-finite SU(3) input and normalization singularity;
- unsupported higher-order AD;
- wrapped tenferro graph, runtime-registration, execution, and AD errors.

Diagnostics include the operation/family, expected contract, actual dtype or
shape, and relevant direction. Preserve the tenferro error as a source where
possible. Do not flatten public errors to `String` and do not reinterpret a
missing runtime as a numerical failure.

## 12. Documentation and implementation records

Each phase PR:

1. updates relevant rustdoc and `docs/design/` when the durable contract changes;
2. adds `docs/worklogs/<date>-phase-<n>.md` with sources consulted, decisions,
   alternatives, exact dependency revision, verification, and remaining risks;
3. links Julia source files and revisions from fixture documentation/tests;
4. checks README/examples for stale names and capability claims;
5. records formatting, lint, tests, doctests, and numerical residuals.

The Julia Phase 0–5 semantic-test inventory remains the source for already
ported behavior. Phase 6–8 tests should reuse its fixtures and conventions where
applicable, adding new fixtures only for graph/AD/evolution behavior.

## 13. Cross-phase acceptance checklist

Before Phase 8 is considered complete:

- the dependency graph uses one exact, current, compatible tenferro snapshot;
- internal field storage is typed and extension boundaries alone are erased;
- direct and traced Wilson action share one numerical implementation;
- runtime and AD initialization is explicit and reusable;
- all three extension families have runtime, validation, and failure tests;
- first-order AD passes independent finite-difference and Julia/Phase 5 oracles;
- batched updates use tenferro contraction infrastructure;
- custom kernels have documented domain/parity reasons;
- no public HMC sampler or hidden global runtime has been introduced;
- public docs describe only functionality that has landed.
