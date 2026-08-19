# Phase 3 fermions

Status: approved; Tasks A-D complete, Task E pending

## Goal

Complete Issue #17 Phase 3 on the existing CPU/host SU(3) path:

- apply the four-dimensional Wilson operator and its adjoint,
- solve Wilson and normal systems with checked Krylov methods,
- evaluate the two-flavor Wilson pseudofermion action and force and include it
  in transactional HMC,
- apply the one-link staggered operator,
- run two-flavor staggered RHMC with the pinned rational approximation,
- compare deterministic kernels, forces, trajectories, and short ensembles
  with the pinned Julia ecosystem.

The implementation reuses `GaugeLinks::host_view`, `Mat3`, `TaGaugeField`,
`gauge_force`, `kinetic_energy`, `exp_ta_update`, `ReproducibleRng`, and the
Phase 1/2 gauge ensemble machinery. It does not duplicate gauge storage,
periodic indexing, SU(3) projection/exponentiation, or quenched observables.

## Accepted scope

- four-dimensional, single-process, host-resident SU(3) fields,
- `Complex64` fermions and gauge links,
- periodic spatial and anti-periodic temporal fermion boundary signs by
  default, with explicit per-axis `+1` or `-1` overrides,
- hopping-normalized Wilson operator with `r=1` and finite positive `kappa`,
- absolute squared-residual stopping criteria plus an independently computed
  true residual in every successful solver report,
- two degenerate Wilson flavors,
- one-link staggered fermions,
- two staggered flavors through RHMC,
- fixed-step U-P-U leapfrog and explicit caller-owned RNG/evolution context,
- deterministic Julia parity and a short independent statistical comparison.

## Non-goals

- GPU, MPI, halo storage, or distributed solvers,
- general SU(N), dimensions other than four, or non-`Complex64` fermions,
- Wilson--clover, HISQ, domain-wall, Möbius, or generalized domain-wall
  operators,
- even/odd preconditioning, chronological guesses, mixed precision, or solver
  autotuning,
- arbitrary complex fermion boundary phases,
- Julia's non-unit Wilson `factor` option,
- a Remez coefficient generator or arbitrary-flavor RHMC,
- adaptive molecular dynamics, multiple time scales, Omelyan integration,
  Hasenbusch splitting, or rational-coefficient fitting at runtime.

Deferred capabilities have no placeholder APIs. Domain-wall remains outside the
umbrella issue's Phase 3 scope. The incomplete v0.6.4 clover force, optional
HISQ path, Julia global RNG, fast custom-boundary behavior, ignored staggered
`eps_CG` key, and non-unit `factor` inconsistency are not copied.

## Reference-defect issue ledger

Known upstream-reference defects are preserved as Rust-repository issues rather
than hidden in worklog prose:

- #21: Gaugefields.jl heatbath unused RNG draws, normalization, and degenerate input,
- #22: Gaugefields.jl incomplete ILDG writer,
- #23: ignored staggered `eps_CG`,
- #24: Wilson `factor` absent from pseudofermion force,
- #25: computed clover force multiplied by zero,
- #26: negative one-eighth RHMC coefficient tables not inverted.

Every newly confirmed defect or action/force inconsistency found during Phase 3
gets a non-duplicate `latticeqcd-rs` issue containing the exact upstream
revision and source link, a reproducer or direct code proof, and the Rust-side
handling. A suspicion is not filed as fact; it is first reduced to a reproducer.

## Pinned references and provenance

| Project | Revision | Role |
|---|---|---|
| LatticeDiracOperators.jl v0.6.4 | `bdef628184597815ba3e0cddf2536df767e78a02` | Wilson/staggered operators, solvers, pseudofermion actions/forces, RHMC |
| Gaugefields.jl v0.7.2 | `9e5719970770f4497405a856315c90bef7f74449` | links, staggered phases, gauge force/evolution, Julia fixtures |
| Wilsonloop.jl v0.1.5 source | `e1a617fdedb19b785f89bdeb13c30e53b20743a7` | gauge-action conventions |
| QCDMeasurements.jl v0.2.13 | `9e04c37bbd68712cf7a749ae5aff10eb6aae4566` | staggered chiral-condensate estimator |

LatticeDiracOperators.jl v1 is not the reference because it requires
Gaugefields.jl v1 and LatticeMatrices v1.1; those are incompatible with the
Phase 1/2 reference boundary. v0.6.4 is the newest tagged release whose compat
entry is `Gaugefields = "0.7"`. All referenced projects are MIT-licensed. The
new crate preserves the applicable upstream notice and records source URLs in
fixture metadata.

## Reference-parallel organization

Keep the Rust implementation structurally parallel to pinned Julia v0.6.4 so
reviewers can compare it directly:

| Rust module | Julia reference |
|---|---|
| `field.rs` | `src/AbstractFermions_4D.jl` and family field types |
| `wilson.rs` | `src/WilsonFermion/WilsonFermion.jl` |
| `solvers.rs` | `src/cgmethods.jl` |
| `wilson_action.rs` | `src/action/WilsonFermiAction.jl` |
| `wilson_hmc.rs` | `test/wilsonhmc.jl` plus `src/action/WilsonFermiAction.jl` force updates |
| `staggered.rs` | `src/StaggeredFermion/StaggeredFermion.jl` |
| `staggered_action.rs` | `src/action/StaggeredFermiAction.jl` |
| `rhmc.rs` | `src/rhmc/rhmc.jl` |

Within each kernel, retain the Julia decomposition, hop ordering, projector
names, solver recurrence names, and meaningful intermediate names where doing
so does not violate Rust ownership or the established storage contract. Each
ported function cites the exact Julia function and revision. Fixture metadata
maps every compared Julia entrypoint to its Rust entrypoint and records layout
conversion. Tests compare named intermediate results when that makes an action,
force, or trajectory discrepancy easier to localize.

Parallel organization does not copy Julia global RNG, temporary-pool mutation,
assertions at public boundaries, hidden defaults, known defects, halo storage,
or repeated shifted-field allocation. Rust typed errors, transactional output,
caller-owned state, site-contiguous storage, and one prepared host view remain
mandatory. Mechanical Julia helper layers with no numerical semantics are not
recreated.

## Crate and dependency boundary

Add one crate:

```text
gaugefields
    ^
    |
dirac-operators
```

`dirac-operators` owns fermion storage, Dirac stencils, Krylov solvers,
pseudofermion actions/forces, and dynamical-fermion trajectories. It depends on
`gaugefields`, `num-complex`, `thiserror`, `tenferro-tensor`, and `tenferro-cpu`.
No lower crate depends on it. `measurements` remains independent; integration
tests may use both upper crates.

A full-QCD integrator lives in `dirac-operators`, rather than making
`gaugefields::hmc_update` generic or introducing callbacks into the lower
crate. It uses the existing public gauge evolution and force types. This keeps
the quenched API unchanged and avoids a speculative action framework.

## Fermion storage and indexing

`FermionField` wraps a host `TypedTensor<Complex64>` with private storage and
validated metadata. Its shape is

```text
[3, components, NX, NY, NZ, NT]
```

where `components` is exactly 4 for Wilson and 1 for staggered fields. Color is
fastest, then spin/taste, then the x-fast lattice site. This follows the Rust
rule that contraction dimensions precede lattice batch dimensions and keeps
all components at one site contiguous. Julia fixtures transpose from
`[color,x,y,z,t,spin]`; metadata states that conversion explicitly.

Public construction validates dtype statically, rank, exact component count,
lattice shape, host placement, checked element/byte counts, and every finite
real/imaginary component before use. Public access exposes logical
`component(color, spin, site)` and checked field algebra, not raw slices.
Mutation is private except for checked constructors and solver/operator output.

`FermionBoundary::new([i8; 4])` accepts only `+1` and `-1`; the default is
`[+1,+1,+1,-1]`. A sign is applied exactly once when a hop wraps its axis.
Gauge links remain periodic.

## Wilson convention

Use the Euclidean chiral gamma matrices from
`WilsonFermion/WilsonFermion.jl::mk_gamma`, with

```text
gamma5 = diag(-1,-1,+1,+1)
{gamma_mu,gamma_nu} = 2 delta_mu,nu.
```

For `r=1`, the public hopping-normalized operator is

```text
(D psi)(x) = psi(x)
 - kappa sum_mu [
     (1-gamma_mu) U_mu(x) psi(x+mu)
   + (1+gamma_mu) U_mu†(x-mu) psi(x-mu)
 ].
```

The adjoint exchanges the two projectors:

```text
(D† psi)(x) = psi(x)
 - kappa sum_mu [
     (1+gamma_mu) U_mu(x) psi(x+mu)
   + (1-gamma_mu) U_mu†(x-mu) psi(x-mu)
 ].
```

No `1/(2*kappa)` physical normalization or arbitrary overall `factor` is
provided. `D†D` is composition, not a separately re-derived stencil.
`WilsonDirac` borrows links and stores only validated `kappa` and boundary
signs; reconstructing it after a gauge update is allocation-free.

## Operator and solver API

The minimum public surface is:

```rust
pub struct FermionField { /* private */ }
pub struct FermionBoundary([i8; 4]);
pub trait FermionOperator {
    fn lattice(&self) -> LatticeShape4;
    fn components(&self) -> usize;
    fn apply_into(&self, output: &mut FermionField, input: &FermionField)
        -> Result<(), DiracError>;
    fn apply_into_with_scratch(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        scratch: &mut [FermionField],
    ) -> Result<(), DiracError>;
}
pub struct WilsonDirac<'a> { /* borrowed links, kappa, boundary */ }
pub struct WilsonAdjoint<'a> { /* view */ }
pub struct NormalOperator<O> { /* D†D composition */ }
pub trait HermitianPositiveOperator: FermionOperator { /* marker */ }
pub struct SolverParams { /* tolerance, max_iterations */ }
pub struct SolverReport { /* iterations, recursive and true residuals */ }
pub fn conjugate_gradient(...); // output is the transactional initial guess
pub fn bicgstab(...); // output is the transactional initial guess
```

The trait has multiple real implementations (Wilson, adjoint, normal,
staggered, shifted normal) and is the only abstraction shared by all solvers.
Operator calls permit exact input/output alias rejection rather than hidden
full-field copies. Solvers preallocate scratch once per call and do not allocate
per iteration.

`SolverParams` requires finite positive absolute squared-residual tolerance and
positive maximum iterations. Successful reports include method, iterations,
initial/recursive/true residual squared, tolerance, maximum iterations, restart
count, and convergence branch. Non-finite intermediates, zero/near-zero
denominators, singular shadow restart, stagnation, iteration exhaustion, and
wrong field/operator shape return typed errors. The output is changed only on
success, and the supplied initial guess is otherwise preserved. BiCGStab
includes the pinned shadow-residual restart but rejects a still-singular
restart. CG requires the minimal `HermitianPositiveOperator` marker, already
implemented by `NormalOperator`; BiCGStab accepts the existing general
`FermionOperator` contract. Both solvers allocate their solve-local scratch
once and do not allocate solver fields per iteration; the built-in Wilson and
normal operators consume that caller-owned workspace. CG is used for `D†D` and
shifted staggered normal systems; BiCGStab solves `D`.

Task B is implemented in `crates/dirac-operators/src/solvers.rs`, directly
parallel to LatticeDiracOperators.jl v0.6.4 `src/cgmethods.jl` at
`bdef628184597815ba3e0cddf2536df767e78a02`: Rust `conjugate_gradient` maps to
`Dirac_operators.cg` (lines 768–868), and Rust `bicgstab` maps to
`Dirac_operators.bicgstab` (lines 157–310). The recurrence names and update
ordering (`r`, `p`, `Ap`, `s`, `t`, `alpha`, `beta`, `omega`) are retained;
Rust replaces the Julia temporary pool and panic paths with checked algebra,
typed errors, and transactional commit.

## Wilson pseudofermions and HMC

The accepted Wilson action represents two degenerate flavors:

```text
S_f(phi,U) = phi† (D†D)^-1 phi.
```

For a supplied complex Gaussian `xi`, refresh is `phi = D† xi`, so the action
returns `||xi||²` within solver tolerance. The public sampler uses the explicit
`ReproducibleRng` and scales independent standard-normal real and imaginary
components by `1/sqrt(2)`. Julia RNG bit parity is not claimed; deterministic
operator/action/force fixtures use explicit fixed `xi` and `phi` arrays.

For force evaluation:

```text
X = (D†D)^-1 phi
Y = D X
```

and the raw link derivative follows v0.6.4
`WilsonFermiAction::calc_UdSfdU_fromX!`. It is projected once into the existing
`TaGaugeField` convention. The sign and generator normalization are fixed by
central finite differences under `U <- exp(epsilon*T_a) U`, not by function
names alone.

`wilson_hmc_update` owns no global state. It:

1. samples momentum and pseudofermions from the caller RNG,
2. evaluates gauge + kinetic + fermion Hamiltonians,
3. evolves a private link/momentum proposal with U-P-U leapfrog,
4. updates momentum with the pinned split scaling: the gauge TA force retains
   Julia's `-step_size/NC` integrator factor, while the fermion TA force enters
   with `-step_size` and no extra `1/NC`,
5. consumes one unconditional open-unit Metropolis draw,
6. commits links only on acceptance.

Link and momentum state is transactional on every error and rejection. RNG
draws already consumed are not rolled back. Solver failure aborts the proposal
without partial field mutation.

## One-link staggered convention

A staggered field has one component. With zero-based coordinates,

```text
eta_0(x)=1
eta_1(x)=(-1)^x
eta_2(x)=(-1)^(x+y)
eta_3(x)=(-1)^(x+y+z).
```

Define `U_stag_mu(x)=eta_mu(x) U_mu(x)` and

```text
(K chi)(x) = 1/2 sum_mu [
    U_stag_mu(x) chi(x+mu)
  - U_stag_mu†(x-mu) chi(x-mu)
]
D_st = mass I + K
D_st† = mass I - K
D_st†D_st = mass² I - K².
```

The same explicit boundary signs apply to shifted fermions. `mass` must be
finite and positive. Tests establish `K†=-K` and compare composition with the
closed normal formula.

## Two-flavor RHMC

Only `Nf=2` is accepted initially. Let `M=D_st†D_st`. The v0.6.4 coefficients
on the documented spectral interval `[0.0004,64]` are preserved exactly:

- degree 15 approximations to `M^(+1/8)` and `M^(-1/8)` for refresh/action,
- degree 10 approximation to `M^(-1/4)` for molecular-dynamics force.

Each rational function has the partial-fraction form

```text
R(M)b = alpha0*b + sum_j alpha_j (M + beta_j I)^-1 b.
```

The constants are private provenance-backed data, not a general coefficient
API. `multi_shift_cg` validates finite non-negative shifts and independently
checks every true residual. The RHMC constructor accepts explicit finite
positive claimed spectral bounds and rejects bounds outside the coefficient
interval. It documents that this is a caller assertion, then runs deterministic
extremal Ritz checks in tests; it does not silently estimate or clamp spectra.

Refresh uses `phi=R_(+1/8)(M) xi`; the action is
`||R_(-1/8)(M) phi||²`, approximating `phi† M^(-1/4) phi`. Force combines the
shifted solutions with the inverse degree-10 residues and the pinned staggered
link derivative before one TA projection. The RHMC trajectory otherwise uses
the same transactional U-P-U and unconditional-Metropolis contracts as Wilson
HMC.

## Task DAG and review gates

Each task is independently reviewable. Implementation starts only after its
section of this document has a `Correct-to-merge` pre-review verdict.

1. **Task A: field and Wilson operator**
   - crate, errors, field/boundary contracts, gamma/projector kernels,
     `D`, `D†`, and `D†D`;
   - Julia impulse/full-field fixture, adjoint and gamma5 identities.
2. **Task B: Krylov solvers** *(complete; post-review pending)*
   - CG and BiCGStab, reports, transactional output;
   - dense/small-lattice Julia solutions, true residuals and breakdown tests;
   - `fixtures/fermions_task_b` maps Julia `cg`/`bicgstab` to the Rust
     entrypoints with explicit links, rhs, and zero/nonzero guesses.
3. **Task C: Wilson pseudofermions and HMC**
   - refresh, action, analytic force, combined leapfrog and Metropolis;
   - Julia action/force/one-trajectory comparison and finite differences.
4. **Task D: staggered operator and multi-shift CG** *(implementation complete; post-review pending)*
   - `Staggered_Dirac_operator`/`Dx!` → `StaggeredDirac`, adjoint →
     `StaggeredAdjoint`, `DdagD_Staggered_operator` →
     `StaggeredNormalOperator`, closed `mass² I-K²` →
     `StaggeredClosedNormalOperator`, and `shiftedcg` → `multi_shift_cg`;
   - zero-based eta phases and one-time wrapped-hop boundary signs;
   - Julia impulse/full-field and shifted true-residual fixtures, with every
     declared payload and metadata/report field consumed by the integration test.

   Finalization evidence: the fixture has 37 declared payloads and 38 total
   tree files. Two generations from Julia 1.12.5 and the clean pinned
   Gaugefields.jl `9e5719970770f4497405a856315c90bef7f74449`,
   LatticeDiracOperators.jl `bdef628184597815ba3e0cddf2536df767e78a02`, and
   Wilsonloop.jl `e1a617fdedb19b785f89bdeb13c30e53b20743a7` checkouts matched at
   complete-tree hash
   `c372e6e56bc05ebc611c6cc3dba5c247eafbc12ca58a0eee2ac3737cdbb08d4b`.
   The fixture uses mass `0.17`, shifts `[0.31, 0.0, 0.07]`, absolute squared
   solver tolerance `1e-24`, and 2000 iterations. Operator, anti-Hermiticity,
   and normal-composition comparisons use `2e-12`; fresh shifted true relative
   residuals use `1e-11`. The former compares components and identities; the
   latter is the independent solve gate and is intentionally distinct.
   D/Ddag/K payload parity is bit-exact; normal-composition maxima are
   `3.19867204157556452e-17` (periodic) and
   `1.66533453693773481e-16` (default anti-periodic), K anti-Hermiticity is
   `2.48253415324727312e-16`, and eta/impulse payloads are exact. The initial
   shifted residual squared is `2.99675440000000037e1`; all reports took 8
   updated-residual iterations. Julia recursive/true residual-squared pairs in
   shift order are `(1.77459964884789642e-27, 1.47868086305190983e-30)`,
   `(1.19636782643941803e-25, 7.62807887898313613e-30)`, and
   `(4.17668148129519829e-26, 4.43903981763168336e-30)`. Rust fresh true
   relative residuals are `2.52706753667624838e-16`,
   `5.23618850174067806e-16`, and `3.73441567789983714e-16`.
5. **Task E: two-flavor RHMC and integration**
   - pinned coefficients, refresh/action/force/HMC;
   - rational scalar errors, Julia fixed trajectory, reversibility, and short
     ensemble comparison.

Every task receives a full-diff post-review by `reviewer-flash`; findings are
fixed and re-reviewed before the task commit. The final integrated branch gets
one additional full review.

## Fixture and acceptance contract

A single `fixtures/generate.jl` gains focused modes. Every mode is generated
twice in a clean external Julia project with the exact checkouts above, and
complete-tree hashes must match. Fixed-trajectory modes use explicit checked
arrays for links, momenta, and fermions. Ensemble modes construct and seed a
Julia `Xoshiro` explicitly; fixture metadata records the seed and Julia version.
Rust uses caller-owned `ReproducibleRng`; statistical comparison does not claim
cross-language bitwise RNG parity.

### Task A

- lattice `[2,2,2,2]`, nontrivial fixed SU(3) links and deterministic finite
  spinor values,
- periodic and default anti-periodic boundary cases,
- all `D`, `D†`, and `D†D` components within `2e-12`,
- independent impulse stencil, adjoint inner-product residual, gamma5
  hermiticity, and cold plane-wave checks within `2e-12`.

### Task B *(complete; post-review pending)*

- CG on `D†D` and BiCGStab on `D`, both zero and nonzero initial guesses,
- solution max residual against Julia within `2e-11`,
- Rust true relative residual at most `1e-11`,
- explicit initial convergence, exhaustion, non-finite, breakdown, stagnation,
  singular-shadow-restart, wrong-shape, and wrong-component cases,
- output bitwise unchanged on every error,
- fixture metadata parses every declared payload and records the exact Julia
  parameter keys (`eps`, `maxsteps`, `verbose`) and Rust mapping.

### Task C

- fixed `xi` and `phi`: action and every force coefficient against Julia within
  `2e-10`,
- central finite-difference force residual at most `2e-7`, with quadratic
  epsilon trend,
- one fixed-momentum trajectory: old/new Hamiltonian, delta-H, proposal links,
  accept decision, and RNG position,
- reversibility residual at most `2e-10`, exact rollback on rejection/error.

The Julia-parallel map is `sample_pseudofermions!` →
`WilsonFermiAction::sample_pseudofermion`, `evaluate_FermiAction` →
`WilsonFermiAction::evaluate`, `calc_UdSfdU!` → `WilsonFermiAction::force`,
`MDstep!`/`U_update!`/`P_update!` → `wilson_leapfrog_trajectory`, and
`Traceless_antihermitian_add!` → `Mat3::add_ta_coefficients`. The fixed fixture
uses Julia 1.12.5 and the clean pinned Gaugefields.jl,
LatticeDiracOperators.jl, and Wilsonloop.jl checkouts recorded in its metadata
and generator environment.

Task C finalization evidence is complete-tree hash
`9462c1e4bf1f46c0929c81fd932f65dbd20f2a2b65168bb65ad8e8a4d92439af` on both
regenerations. Julia/Rust maximum residuals were `7.16072334609889539e-15`
for `X`, `6.62422734006908809e-15` for `Y`, `4.54747350886464119e-13` for the
action, and `5.10702591327572009e-14` for the force. The established all-512
finite-difference maxima at epsilons `1e-3`, `5e-4`, and `2.5e-4` were
`5.005235745869641e-7`, `1.262665048074041e-7`, and `3.463491360378157e-8`,
with `512/512` coefficients passing at `5e-4` and ratios
`3.964024943514664` and `3.645642262945268`.

### Task D

- staggered eta signs and boundary wrap impulses bit-exact,
- all `D`, `D†`, and normal components against Julia within `2e-12`,
- anti-Hermiticity and normal-composition residual at most `2e-12`,
- every shifted solution true relative residual at most `1e-11`.

### Task E

- every pinned coefficient bit pattern and scalar approximation error over a
  fixed logarithmic grid,
- fixed refresh, action, force, and proposal against Julia within `2e-9`,
- central force finite-difference residual at most `5e-7`,
- reversibility residual at most `5e-9`, exact rollback on rejection/error,
- independent short Julia/Rust ensembles on the same small lattice; plaquette
  and chiral condensate agree within six combined standard errors, and mean
  delta-H/acceptance are reported without a hidden pass criterion.

The staggered chiral condensate follows pinned QCDMeasurements.jl exactly. For
one Z4 noise vector `r`, it is

```text
(Nf/4) * Re[r† D_st^-1 r] / NV.
```

Thus the exact all-source small-lattice value is
`(Nf/4) * Re Tr(D_st^-1) / NV`; for accepted `Nf=2` the prefactor is `1/2`.
There is no additional `1/NC`. Deterministic tests enumerate the complete
orthonormal source basis; ensemble reporting uses explicitly seeded Z4 noise
and records its sample count.

Fixture metadata records shape conversion, gamma basis, boundaries, kappa or
mass, `Nf`, rational interval/degrees, solver thresholds, RNG handling, source
URLs/functions/revisions, and every tolerance.

## Local and integration gates

For each task run focused tests, formatting, `cargo check --workspace`, default
and all-feature tests, doctests, and `git diff --check`. Before the completed
Phase 3 PR run from a clean target:

```text
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo test --workspace --all-features
cargo test --workspace --doc --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --all-features --no-deps
```

Build and run every example, regenerate every Julia fixture twice, run the
fixed and statistical comparisons, inspect licenses and provenance, reject
TODO/dead compatibility code, and verify the exact tenferro pin. The Phase 3
branch is stacked on PR #20 until that PR is merged; neither PR is merged by
this work.
