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
