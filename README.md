# latticeqcd-rs

CPU-first lattice gauge fields backed by tenferro tensors.

## Attribution and license

The `gaugefields` crate ports algorithms and conventions from
[Gaugefields.jl](https://github.com/shinaoka/Gaugefields.jl), distributed under
the MIT License. Copyright (c) 2022 Akio Tomiya and Yuki Nagai. The original
notice is preserved in [`crates/gaugefields/LICENSE`](crates/gaugefields/LICENSE),
and pinned source revisions are recorded with each Julia oracle. The `wilsonloop`
crate follows the signed-path and plaquette/rectangle conventions of
[Wilsonloop.jl](https://github.com/akio-tomiya/Wilsonloop.jl), distributed under
the MIT License; its applicable notice is preserved in
[`crates/wilsonloop/LICENSE`](crates/wilsonloop/LICENSE). The `measurements`
crate follows the Polyakov and clover conventions of
[QCDMeasurements.jl](https://github.com/akio-tomiya/QCDMeasurements.jl); its notice
is preserved in [`crates/measurements/LICENSE`](crates/measurements/LICENSE).
The `dirac-operators` crate follows Wilson/staggered fermion, solver,
pseudofermion-force, and RHMC conventions from
[LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl),
also MIT-licensed; its notice is preserved in
[`crates/dirac-operators/LICENSE`](crates/dirac-operators/LICENSE).

## Signed Wilson paths and loop actions

The `wilsonloop` crate provides periodic signed unit paths (`-4..=-1` and
`1..=4`), closed plaquette/1x2 rectangle terms, `LoopAction` values with
precompiled occurrence metadata, and host-side action values and forces. A
term means `c * sum_x Re tr(W)`; matching a Julia real coefficient `f` that
adds both `W` and `W†` uses `c = 2f`. Because Julia's `calc_dSdU` is the
holomorphic derivative, each force occurrence uses `c/2 = f`; the corresponding
left variation obeys `dS/dt = -F·v`. The pinned multi-term oracle and all
`4 * 16 * 8` force components are checked in
[`fixtures/wilsonloop_task_b`](fixtures/wilsonloop_task_b).

## Compatibility policy

This development line follows `tenferro-rs` `origin/main`, but every Cargo Git
dependency is locked to exact revision
`c942129974b544225ed963414d7be1300980f901`. Updating to a newer main revision
is an intentional compatibility change: all tenferro pins move together, the
full feature matrix is rebuilt, and fixture/layout tests are rerun.

Until the API stabilizes, compatibility is clean-break rather than emulation:
gaugefields-rs targets its recorded tenferro revision and does not carry shims
for older or newer tenferro APIs.

## Host access and ILDG

`GaugeLinks::host_view()` provides fallible, read-only host access without
exposing tenferro's storage layout. `read_ildg` and `write_ildg` exchange one
4D SU(3), Float64 configuration in a minimal ILDG 1.1 LIME message:

```rust
use gaugefields::{read_ildg, write_ildg};

# fn copy_configuration() -> Result<(), gaugefields::GaugeError> {
let links = read_ildg("input.ildg")?;
write_ildg("output.ildg", &links)?;
# Ok(())
# }
```

The I/O boundary is host-only and rejects malformed framing or XML, unsupported
metadata, wrong payload lengths, non-finite components, and trailing data with
typed errors. It uses big-endian IEEE Float64 values in
`t,z,y,x,direction,row,column,real/imaginary` order. Float32, multiple
configurations, and SciDAC checksum verification are intentionally outside the
minimal API. Run the checked self-contained example with
`cargo run -p gaugefields --example ildg_roundtrip`.

## Reproducible random streams

`ReproducibleRng` imports Julia's four-word `(s0, s1, s2, s3)` xoshiro256++
state in little-endian order. With Julia 1.12.5, dump those 256 state bits
before drawing values:

```julia
using Random
rng = Xoshiro(123)
state = (rng.s0, rng.s1, rng.s2, rng.s3)
@show state
```

Pass the four printed `UInt64` words to Rust in the same order. The Rust wrapper
exposes the raw `RngCore` stream, the exact open-unit mapping, and an uncached
Box--Muller normal stream:

```rust
use gaugefields::ReproducibleRng;
use rand::RngCore;

let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
assert_eq!(rng.next_u64(), 41_943_041);
let mut normals = [0.0; 3];
rng.fill_standard_normals(&mut normals);
assert!(normals.into_iter().all(f64::is_finite));
# Ok::<(), gaugefields::GaugeError>(())
```

An odd normal-fill length discards the final sine result and still consumes a
complete pair. All-zero states are rejected; state replacement is transactional.
There is no global RNG or hidden state export.

## Direct and traced Wilson action

`GaugeLinks` owns four compact host `TypedTensor<Complex64>` values. Direct
observables and force kernels are host-only and return a typed error for backend
storage; download explicitly before calling them.

Traced action execution uses an application-owned runtime, backend engine, and
explicitly installed extension modules:

```rust
use gaugefields::runtime_modules;
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::Runtime;

let backend = CpuBackend::new();
let mut builder = Runtime::builder();
builder.register_engine(runtime_engine_registration(&backend)?)?;
for module in runtime_modules::<CpuBackend>(runtime_engine_id()?)? {
    builder.install_extension_module(module)?;
}
let _runtime = builder.build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

With the `autodiff` feature, applications install the semantic Wilson rules
on each application-owned AD context:

```rust
use gaugefields::ad_rules;
use tenferro_ad::AdContext;

let ad = AdContext::builder()
    .with_semantic_extension_rules(ad_rules()?)?
    .build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The supported path is `Wilson action -> JVP -> linear transpose -> force`.
Only active link directions appear as JVP tangent inputs, and reverse mode
accepts arbitrary finite scalar seeds. There is intentionally no direct primal
VJP or force AD rule, so differentiation through force is a typed unsupported
operation. See `docs/design/ad-convention.md` for the complex convention and
`cargo run -p gaugefields --example traced_wilson_action --all-features` for
checked direct/traced action parity.

## SU(3) evolution

`exp_ta` reproduces Gaugefields.jl's fixed-size SU(3) Cardano algorithm and its
fourth-order near-degenerate fallback. `normalize_su3` performs the matching
NC=3 row projection and rejects non-finite or singular input transactionally.
Finite inputs that overflow coefficient scaling, Cardano intermediates,
fallback powers, or final assembly return the distinct typed
`Su3NumericalRange` error; this phase does not silently clamp or add an
alternative scaling-and-squaring algorithm.
Field updates use an application-owned CPU context:

```rust
use gaugefields::{cold_su3, exp_ta_update, CpuEvolutionContext, LatticeShape4, TaGaugeField};
use tenferro_cpu::CpuBackend;

# fn update(momentum: &TaGaugeField) -> Result<(), gaugefields::GaugeError> {
let mut links = cold_su3(LatticeShape4::new([4, 4, 4, 4])?)?;
let mut evolution = CpuEvolutionContext::new(CpuBackend::new());
exp_ta_update(&mut evolution, &mut links, 0.01, momentum)?;
let stats = evolution.cache_stats();
evolution.clear_cache();
# let _ = stats;
# Ok(())
# }
```

The context reuses its backend, buffer pool, and bounded runtime cache. Stable
slots `0..3` identify the four directions, and all outputs are validated before
the links are replaced. Cache entry counts are provider-dependent: the pinned
cpu-faer unconjugated strided path reports zero retained analysis entries even
though it uses the same cached session and stable slots.

A synchronous isotropic stout step reuses that caller-owned context:

```rust
use gaugefields::{stout_step, CpuEvolutionContext};
use tenferro_cpu::CpuBackend;

# fn smear(links: &gaugefields::GaugeLinks) -> Result<gaugefields::GaugeLinks, gaugefields::GaugeError> {
let mut context = CpuEvolutionContext::new(CpuBackend::new());
let smeared = stout_step(&mut context, links, 0.12)?;
# Ok(smeared)
# }
```

It uses `C_mu = rho *` the unweighted positive six-term plaquette staple,
then `Omega = C_mu * U_mu†`, `Q = TA(Omega)`, and
`U'_mu = exp(Q) * U_mu`. Finite negative `rho` values are valid; every link
uses the unchanged input snapshot and failures leave the input untouched.

## Polyakov and clover measurements

The `measurements` crate exposes fallible host-only
`polyakov_loop`, `clover_topological_charge`, and fixed-step third-order
Runge--Kutta `gradient_flow` operations. Polyakov uses the fourth (temporal)
direction and is normalized by spatial volume, not by `NC`. The clover charge
uses QCDMeasurements.jl's four-loop clover, ordinary
`epsilon(0,1,2,3)=+1`, and `/4^2` normalization. The D1 scalar and beta=5.7
statistical oracles are in
[`fixtures/measurements_task_d1`](fixtures/measurements_task_d1); the D2 flow
oracle is in [`fixtures/gradientflow_task_d2`](fixtures/gradientflow_task_d2).

To regenerate the D1/D2 oracles, activate the clean external Julia reference
project and provide the exact Gaugefields.jl and Wilsonloop.jl checkouts:

```bash
export LATTICEQCD_JULIA_PROJECT=/tmp/latticeqcd-phase2-julia-env
export GAUGEFIELDS_JL_DIR=/path/to/Gaugefields.jl
export WILSONLOOP_JL_DIR=/path/to/Wilsonloop.jl
export JULIA_NUM_THREADS=1
julia --startup-file=no --project="$LATTICEQCD_JULIA_PROJECT" \
  fixtures/generate.jl gradientflow_task_d2
```

D1 and the default generator additionally require
`QCDMEASUREMENTS_JL_DIR`:

```bash
export QCDMEASUREMENTS_JL_DIR=/path/to/QCDMeasurements.jl
julia --startup-file=no --project="$LATTICEQCD_JULIA_PROJECT" \
  fixtures/generate.jl measurements_task_d1
```

The reference project must directly develop the three pinned checkouts and add
`NPZ`; it is external and is not committed to this repository. With no mode
argument, the generator runs each fixture generator once, including both D1
and D2. Older modes that do not use these measurements only require
`GAUGEFIELDS_JL_DIR` (and Wilsonloop.jl where their metadata needs it). Run the
Rust known-value example with
`cargo run -p measurements --example quenched_measurements`.

## Quenched SU(3) heatbath

The host-only Wilson heatbath owns no global state: callers provide validated
parameters and a Julia-compatible four-word RNG. A sweep follows directions
`0..3`, even sites before odd sites, and the fixed SU(2) subgroup order
`(0,1)`, `(1,2)`, `(0,2)`:

```rust
use gaugefields::{
    cold_su3, heatbath_sweep, normalized_plaquette, HeatbathParams, LatticeShape4,
    ReproducibleRng,
};

# fn run() -> Result<(), gaugefields::GaugeError> {
let lattice = LatticeShape4::new([2, 2, 2, 2])?;
let mut links = cold_su3(lattice)?;
let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
let params = HeatbathParams::new(5.7, 100_000)?;
let stats = heatbath_sweep(&mut links, params, &mut rng)?;
println!(
    "updated={} attempts={} plaquette={}",
    stats.updated_links,
    stats.su2_attempts,
    normalized_plaquette(&links)?,
);
# Ok(())
# }
```

The update is transactional for link storage and reports total SU(2) rejection
iterations. It intentionally uses the reviewed open-uniform and square-root
SU(2) normalization corrections rather than promising bitwise trajectory parity
with Gaugefields.jl. Regenerate the pinned three-beta statistical oracle with
`GAUGEFIELDS_JL_DIR=/path/to/Gaugefields.jl julia --startup-file=no fixtures/generate.jl heatbath_statistics`.
Its metadata records all block means, independent Julia seeds, provenance,
schedule, and the fixed six-standard-error comparison criterion. See
`examples/quenched_heatbath.rs` for a runnable loop.

## Wilson Krylov solvers (Phase 3 Task B)

Task B is complete and independently post-reviewed. The
`dirac-operators` crate exposes checked, host-resident `conjugate_gradient`
for the minimal `HermitianPositiveOperator` contract (currently
`NormalOperator = D†D`) and `bicgstab` for the existing general
`FermionOperator` contract. `SolverParams` takes a finite positive absolute
squared-residual tolerance and a positive iteration limit. Each successful
`SolverReport` includes recursive and freshly recomputed true residuals; the
mutable initial guess is committed only after the true-residual gate passes.
Typed failures cover non-finite intermediates, denominator breakdown, shadow
restart singularity, stagnation, exhaustion, and incompatible fields.

The implementation is deliberately parallel to
[LatticeDiracOperators.jl v0.6.4 `cgmethods.jl`](https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/cgmethods.jl): Rust
`conjugate_gradient` maps to `Dirac_operators.cg` (lines 768–868), and Rust
`bicgstab` maps to `Dirac_operators.bicgstab` (lines 157–310), retaining the
`r`, `p`, `Ap`, `s`, `t`, `alpha`, `beta`, and `omega` recurrence names and
update order without copying Julia's global temporary pool or panic paths.
The deterministic fixture `fixtures/fermions_task_b` records the Julia keys
`eps`, `maxsteps`, and `verbose`, maps both entrypoints, compares zero and
nonzero guesses on explicit nontrivial links, and independently recomputes
true residuals.

## Two-flavor Wilson pseudofermions and HMC (Phase 3 Task C)

The `dirac-operators` crate implements the pinned v0.6.4 two-flavor Wilson
contract on the host SU(3) path:

```text
S_f(phi,U) = phi† (D†D)^-1 phi,    phi = D† xi
```

`WilsonFermiAction` samples `xi` from the caller-owned
`ReproducibleRng`; each independent real and imaginary normal is scaled by
`1/sqrt(2)`. Its force solves `X=(D†D)^-1 phi`, `Y=DX`, then applies the
pinned `TA[-kappa P_- U X_plus ⊗ Y + kappa X ⊗ Y_plus† U† P_+]` formula. The
fermion TA force has no `1/NC`; the Julia-compatible gauge momentum update
retains `-step_size/NC` separately.

`wilson_hmc_update` performs a private U-P-U trajectory, evaluates the combined
Hamiltonian, consumes one unconditional open-unit Metropolis draw, and commits
links only on acceptance. `wilson_leapfrog_trajectory` commits both links and
momentum only after the complete trajectory succeeds; rejection and errors leave
caller-owned fields unchanged. The fixed Julia action/X/Y/force/trajectory
oracle is [`fixtures/fermions_task_c`](fixtures/fermions_task_c), generated
against LatticeDiracOperators.jl v0.6.4 and Gaugefields.jl v0.7.2. Task C is
implementation-complete and independently post-reviewed.

Regenerate it twice from the clean external project and verify the complete-tree
hashes:

```bash
export LATTICEQCD_JULIA_PROJECT=/tmp/latticeqcd-phase3-julia-env
export GAUGEFIELDS_JL_DIR=/path/to/Gaugefields.jl
export LATTICEDIRACOPERATORS_JL_DIR=/path/to/LatticeDiracOperators.jl
export WILSONLOOP_JL_DIR=/path/to/Wilsonloop.jl
julia --startup-file=no --project="$LATTICEQCD_JULIA_PROJECT" \
  fixtures/generate.jl fermions_task_c
```

The Julia-parallel entrypoints are `sample_pseudofermions!` →
`WilsonFermiAction::sample_pseudofermion`, `evaluate_FermiAction` →
`WilsonFermiAction::evaluate`, `calc_UdSfdU!` → `WilsonFermiAction::force`,
and `MDstep!`/`U_update!`/`P_update!` →
`wilson_leapfrog_trajectory`; `Traceless_antihermitian_add!` maps to
`Mat3::add_ta_coefficients`. The generator uses Julia 1.12.5 with clean
Gaugefields.jl `9e5719970770f4497405a856315c90bef7f74449`,
LatticeDiracOperators.jl `bdef628184597815ba3e0cddf2536df767e78a02`, and
Wilsonloop.jl `e1a617fdedb19b785f89bdeb13c30e53b20743a7` checkouts.

Finalization evidence: the complete `fermions_task_c` tree hash was
`9462c1e4bf1f46c0929c81fd932f65dbd20f2a2b65168bb65ad8e8a4d92439af` on both
runs. Julia/Rust maximum residuals were `7.16072334609889539e-15` for `X`,
`6.62422734006908809e-15` for `Y`, `4.54747350886464119e-13` for the action,
and `5.10702591327572009e-14` for the force. The all-coefficient central
finite-difference series at epsilons `1e-3`, `5e-4`, and `2.5e-4` was
`5.005235745869641e-7`, `1.262665048074041e-7`, and `3.463491360378157e-8`;
all `512/512` coefficients passed at `5e-4`, with ratios `3.964024943514664`
and `3.645642262945268`.

## One-link staggered fermions and multi-shift CG (Phase 3 Task D)

Task D is implementation-complete and independently post-reviewed.
`StaggeredDirac` maps the pinned LatticeDiracOperators.jl v0.6.4
`Staggered_Dirac_operator`/`Dx!` path, `StaggeredAdjoint` maps its adjoint,
`StaggeredNormalOperator` composes `D†D`, and
`StaggeredClosedNormalOperator` independently lowers `mass² I - K²`.
`multi_shift_cg` maps `Dirac_operators.shiftedcg`, retaining the shared
`r`, `p`, `q`, `alpha`, `beta`, `rho_m`, `rho_0`, and `rho_p` recurrence order
with transactional outputs and fresh true-residual checks. Eta is
`[1,(-1)^x,(-1)^(x+y),(-1)^(x+y+z)]` in zero-based coordinates; the validated
fermion boundary sign is applied once per wrapped hop.

The fixture uses `[2,2,2,2]`, one component, mass `0.17`, shifts
`[0.31, 0.0, 0.07]`, absolute squared solver tolerance `1e-24`, and maximum
iterations `2000`. The operator, anti-Hermiticity, and normal-composition
comparison tolerance is `2e-12`; the independently recomputed shifted true
relative-residual tolerance is `1e-11`. These are distinct gates: the former
compares operator components/identities, while the latter checks each solved
`(D†D + shift I)x = b` after a fresh application.

Regenerate twice from the clean pinned Julia project:

```bash
export LATTICEQCD_JULIA_PROJECT=/tmp/latticeqcd-phase3-julia-env
export GAUGEFIELDS_JL_DIR=/path/to/Gaugefields.jl
export LATTICEDIRACOPERATORS_JL_DIR=/path/to/LatticeDiracOperators.jl
export WILSONLOOP_JL_DIR=/path/to/Wilsonloop.jl
export JULIA_NUM_THREADS=1
julia --startup-file=no --project="$LATTICEQCD_JULIA_PROJECT" \
  fixtures/generate.jl fermions_task_d
```

Finalization evidence: both 38-file trees (37 declared payloads plus metadata)
have complete-tree hash
`c372e6e56bc05ebc611c6cc3dba5c247eafbc12ca58a0eee2ac3737cdbb08d4b`. The
Julia reports have initial residual squared `2.99675440000000037e1`; all three
shifts converged in 8 iterations on the updated-residual branch. In shift order
`0.31`, `0.0`, `0.07`, the Julia recursive/true residual-squared pairs are
`(1.77459964884789642e-27, 1.47868086305190983e-30)`,
`(1.19636782643941803e-25, 7.62807887898313613e-30)`, and
`(4.17668148129519829e-26, 4.43903981763168336e-30)`; Rust true relative
residuals are respectively `2.52706753667624838e-16`,
`5.23618850174067806e-16`, and `3.73441567789983714e-16`.

The fixture test consumes all 37 declared payloads and every metadata/report
field. D, D†, and K payload parity is bit-exact; normal-composition parity is
`3.19867204157556452e-17` (periodic) and `1.66533453693773481e-16`
(default anti-periodic), K anti-Hermiticity is
`2.48253415324727312e-16`, and eta/boundary impulse checks are exact.

## Two-flavor staggered RHMC (Phase 3 Task E)

Task E deterministic integration and the independent short ensemble comparison
are complete. Plaquette and chiral-condensate means differ from Julia by
`0.154717` and `0.367498` combined standard errors. `StaggeredFermiAction`
follows the pinned LatticeDiracOperators.jl
v0.6.4 `StaggeredFermiAction`/`RHMC` path with the exact private Nf=2 degree-15
`x^(+1/8)` refresh and `x^(-1/8)` action tables and degree-10 `x^(-1/4)`
force table on `[0.0004,64]`. The scalar 4097-point logarithmic-grid maximum
absolute errors are `2.505791796281187e-9`, `3.9620045022559225e-9`, and
`1.5595609319518644e-5` (refresh/action/force). Explicit deterministic `xi`
refresh, action `X`, force `X_j/Y_j`, and transactional U-P-U trajectory
payloads are checked against Julia, including validation/error rollback,
rejection rollback, RNG advancement, and reversibility.

`fixtures/fermions_task_e` contains 68 files (67 declared payloads plus
metadata); two clean Julia 1.12.5 generations have identical complete-tree
hash `9e166b37d2c138a28f6d75395e11dc8f91f910599f0397e5888bbd738ba6d34a`.
The all-512 central force FD contract uses epsilons
`[0.32,0.16,0.08,0.04]`, maxima
`[8.434653210321642e-6,2.139177378187619e-6,5.605769951367093e-7,1.6563038083509257e-7]`,
ratios `[3.9429424115674574,3.816027765580949,3.384505863660601]`, and pass
counts `[291,442,510,512]`; selected `0.04` passes all `512/512` below `5e-7`.
Run the checked Wilson/staggered/RHMC smoke example with:

```text
cargo run -p dirac-operators --example fermions --all-features
```

## Quenched SU(3) HMC

HMC is a fixed-step, CPU-first SU(3) API. The caller owns the evolution context
and explicitly imports the four-word Julia-compatible RNG state:

```rust
use gaugefields::{
    cold_su3, hmc_update, normalized_plaquette, CpuEvolutionContext, HmcParams,
    LatticeShape4, ReproducibleRng,
};
use tenferro_cpu::CpuBackend;

# fn run() -> Result<(), gaugefields::GaugeError> {
let lattice = LatticeShape4::new([4, 4, 4, 4])?;
let mut links = cold_su3(lattice)?;
let mut context = CpuEvolutionContext::new(CpuBackend::new());
let params = HmcParams::new(5.7, 0.01, 4)?;
let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
let mut accepted = 0;
for _ in 0..3 {
    accepted += usize::from(hmc_update(&mut context, &mut links, params, &mut rng)?.accepted);
}
println!("accepted={accepted}/3 plaquette={}", normalized_plaquette(&links)?);
# Ok(())
# }
```

Each update uses the exact U-P-U trajectory and an unconditional open-unit
Metropolis draw. Link and momentum inputs are transactional on trajectory
failure; rejected proposals restore all links. RNG advancement is not rolled
back: errors consume only draws completed before the error. Adaptation, alternate
actions, and device-resident HMC are not part of this API.
