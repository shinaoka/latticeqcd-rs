# Dependency policy

All direct tenferro crates are pinned to exact `tenferro-rs` revision
`51bc0a7bef274e20d08fc054856cb4d74c284cbe`. The pin is repeated for the
tensor, runtime, CPU, and optional AD crates so neither default nor `autodiff`
builds follow a moving branch.

The `autodiff` compile spike also has direct, optional workspace dependencies
on `computegraph` at exact revision
`691def2d82fd4367b397d61209449f68e82050b7` and `tidu` at exact revision
`57a2e7ebe7738ca2f8b5c96f4c6ce4e467b20495`. These pins match the revisions
used by the pinned tenferro revision, so the direct compile-spike APIs and
tenferro's dependency graph remain compatible.

The default feature set is deliberately minimal. Phase 6 traced Wilson action
uses the pinned runtime extension ABI in every build. The `autodiff` feature
currently reserves the pinned `tenferro-ad`, `computegraph`, and `tidu` compile
boundary for Phase 7; it does not expose gauge autodiff rules yet.

`tenferro-runtime` and `tenferro-cpu` are direct dependencies because the public
registration closure and checked CPU example use their APIs. The nondefault
`autodiff` feature additionally enables tenferro-ad and its `cpu-faer` graph.

Future accelerator work reserves two backend-specific feature names: `cuda`
and `rocm`. Neither feature is implemented or declared in the manifest today.
The design explicitly forbids an umbrella `gpu` feature because CUDA and ROCm
have different toolchains, availability, and compatibility constraints; users
must eventually select a backend explicitly.

`cargo tree --duplicates` at this lockfile reports only backend/tooling families:
`equator` 0.2/0.4/0.6 (and matching macros), `rand` 0.8/0.9 (and matching
`rand_core`), `syn` 1/2, and `thiserror` 1/2 (and matching proc macros). These
come through faer, npyz, and strided dependencies. There is exactly one source
revision for every tenferro crate.
