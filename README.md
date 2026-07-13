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

See `cargo run -p gaugefields --example traced_wilson_action --all-features`
for a checked direct/traced parity example. Phase 6 supplies host-reference
extension execution only; autodiff rules are not yet part of the public API.
