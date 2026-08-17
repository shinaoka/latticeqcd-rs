# latticeqcd-rs

CPU-first lattice gauge fields backed by tenferro tensors.

## Compatibility policy

This development line follows `tenferro-rs` `origin/main`, but every Cargo Git
dependency is locked to exact revision
`51bc0a7bef274e20d08fc054856cb4d74c284cbe`. Updating to a newer main revision
is an intentional compatibility change: all tenferro pins move together, the
full feature matrix is rebuilt, and fixture/layout tests are rerun.

Until the API stabilizes, compatibility is clean-break rather than emulation:
gaugefields-rs targets its recorded tenferro revision and does not carry shims
for older or newer tenferro APIs.

## Direct and traced Wilson action

`GaugeLinks` owns four compact host `TypedTensor<Complex64>` values. Direct
observables and force kernels are host-only and return a typed error for backend
storage; download explicitly before calling them.

Traced action execution uses an application-owned executor and explicit runtime
registration:

```rust
use gaugefields::{register_runtime, wilson_action_traced};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::GraphExecutor;

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(register_runtime)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

With the `autodiff` feature, applications register the role-split Wilson rules
on each application-owned AD context separately from executor registration:

```rust
use gaugefields::ad_rules;
use tenferro_ad::AdContext;

let ad = AdContext::builder()
    .with_extension_rules(ad_rules()?)
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
though it uses the same cached session and stable slots. No HMC sampler is
public; HMC appears only as deterministic crate-private regression support.
