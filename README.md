# gaugefields-rs

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
