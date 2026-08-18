# latticeqcd-rs

CPU-first lattice gauge fields backed by tenferro tensors.

## Compatibility policy

This development line follows `tenferro-rs` `origin/main`, but every Cargo Git
dependency is locked to exact revision
`c942129974b544225ed963414d7be1300980f901`. Updating to a newer main revision
is an intentional compatibility change: all tenferro pins move together, the
full feature matrix is rebuilt, and fixture/layout tests are rerun.

Until the API stabilizes, compatibility is clean-break rather than emulation:
gaugefields-rs targets its recorded tenferro revision and does not carry shims
for older or newer tenferro APIs.

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
back: errors consume only draws completed before the error. Heatbath,
adaptation, alternate actions, and device-resident HMC are not part of this API.
