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

`cargo tree --duplicates` is retained as an audit command. Duplicate versions
in tenferro's backend dependency graph are acceptable when selected by Cargo;
multiple revisions of any tenferro crate are not.
