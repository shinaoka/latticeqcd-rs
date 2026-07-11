# Dependency policy

All direct tenferro crates are pinned to exact `tenferro-rs` revision
`f504ba0a8668baca89ab1d4348b9475ff85377b4`. The pin is repeated for the
tensor, runtime, CPU, and optional AD crates so neither default nor `autodiff`
builds follow a moving branch.

At that revision, tenferro itself pins `computegraph` to
`691def2d82fd4367b397d61209449f68e82050b7` and `tidu` to
`57a2e7ebe7738ca2f8b5c96f4c6ce4e467b20495`. Cargo resolves those transitive
pins from tenferro's manifest; gaugefields does not override them.

The default feature set is deliberately minimal. The `autodiff` feature only
links the pinned `tenferro-ad` API; it does not promise traced gauge kernels.
Its compile spike implements `tenferro_runtime::extension::ExtensionOp`, calls
`extension::apply`, implements `ExtensionLinearizeRule`, registers it through
`ExtensionRuleSet::with_linearize`, and attaches that set with
`AdContextBuilder::with_extension_rules`. The identity carrier is deliberately
non-executable: it validates the Phase 0 extension and AD registration surface,
not a Phase 6 gauge operation.

`cargo tree --duplicates` at this lockfile reports only backend/tooling families:
`equator` 0.2/0.4/0.6 (and matching macros), `rand` 0.8/0.9 (and matching
`rand_core`), `syn` 1/2, and `thiserror` 1/2 (and matching proc macros). These
come through faer, npyz, and strided dependencies. There is exactly one source
revision for every tenferro crate.
