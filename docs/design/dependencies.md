# Dependency policy

All five direct tenferro crates are pinned atomically to exact tenferro-rs
`origin/main` revision
`c942129974b544225ed963414d7be1300980f901`:

- `tenferro-tensor`
- `tenferro-runtime` with `cpu-faer`
- `tenferro-cpu` with `cpu-faer`
- `tenferro-ad` with `cpu-faer` under `autodiff`
- `tenferro-internal-ops` with `autodiff`

`Cargo.lock` records the same revision for every resolved tenferro package.
The repository has no direct `computegraph` or `tidu` dependency; computegraph
remains only where required transitively by tenferro's current runtime and
operation crates. The default feature set is deliberately minimal, while
`autodiff` adds only `tenferro-ad`.

`tenferro-runtime` and `tenferro-cpu` are direct dependencies because traced
program compilation, the application-owned `Runtime`, backend registration, and
the checked example use their APIs. Extension modules are installed explicitly
into that runtime; no backend or extension fallback is hidden in gaugefields.

Phase 1 reproducible RNG support is implemented with the normal dependencies
`rand` 0.8 and `rand_xoshiro` 0.6. `rand` default features are disabled because
the public wrapper uses only `RngCore` and `Error`; this also avoids pulling the
unused `rand_chacha` standard RNG. HMC uses the wrapper's open-unit and
uncached Box--Muller methods; the SU(3) heatbath uses its open-unit method.
Heatbath sampling adds no distribution, serialization, generic RNG, or sampler
dependency. The heatbath API accepts only the existing `ReproducibleRng`, and
its Julia statistical fixture is produced by the already-pinned external
Gaugefields.jl checkout rather than by a Rust-side Julia or heatbath
reimplementation.

Future accelerator work reserves backend-specific feature names. Neither CUDA
nor ROCm is implemented or declared in this manifest today; adding one requires
an explicit placement-aware design rather than an umbrella feature or implicit
transfer.

`cargo tree --duplicates` reports only dependency-family duplicates pulled by
faer, npyz, and strided support. They are unrelated to tenferro pinning and
must not be removed by an application-layer refactor.
