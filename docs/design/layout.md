# Gauge-field layout and fixture contract

Each directed link owns one compact, column-major
`TypedTensor<Complex64>` of rank six
with shape `[NC, NC, NX, NY, NZ, NT]`. A gauge field contains four independent
tensors, one for each `mu = 0,1,2,3`; direction is not another tensor axis.

For component `(a,b)` at site `(x,y,z,t)`, the zero-based flat offset is

```text
a + NC * (b + NC * (x + NX * (y + NY * (z + NZ * t))))
```

Thus matrix row `a` varies first, sites are x-fastest, and every site is one
contiguous nine-element SU(3) block in column-major matrix order. Owned tensors
must be compact; hidden layout conversion is not part of this contract.

Fixtures are directories containing `u0.npy` through `u3.npy` and
`metadata.json`. NPY files must be NumPy `complex128`, rank six, and have
`fortran_order=true`. The reader rejects C order rather than silently
transposing. Metadata records `nc`, lattice extents, beta, expected observables,
Gaugefields.jl version, and the exact Gaugefields.jl Git commit. Metadata and
all four array shapes must agree.

`fixtures/generate.jl` is the authority for checked fixtures. Gauge-field
fixtures produced by that script against Gaugefields.jl record the exact
Gaugefields.jl provenance; the separate `reproducible_rng` metadata fixture is
stdlib-only and records Julia `Random.Xoshiro` provenance instead. The checked
`random_2x2x2x2` fixture stores exact IEEE-754 reference bits for
every component. Gaugefields.jl's reproducible initializer resets the same RNG
for each direction, so the generator applies a direction-specific periodic
site shift; this preserves the random SU(3) matrices while making direction and
axis swaps observable.

Periodic kernels read neighboring nine-element blocks directly. They do not
allocate tensor-roll buffers and do not expose Gaugefields.jl's mutable shared
`Ushifted` scratch buffer. The checked `shifts_3x2x4x5` fixture materializes
that Julia reference only as an oracle (`gaugefields_4D_nowing.jl:380-454`);
Rust parity is evaluated through `neighbor_site` reads.

Direct kernels validate once through `PreparedGaugeField`, borrow each of the
four host slices once, and precompute checked plus/minus neighbor tables. The
periodic action, measurement staple, dense gradient, and TA force share these
site-local `Mat3` leaves. Dtype-erased `Tensor` values occur only at fixture and
extension ABI boundaries; `TracedTensor` is graph metadata rather than storage.

Regenerate all checked fixtures portably from a clean shell with
`GAUGEFIELDS_JL_DIR=/path/to/Gaugefields.jl julia --startup-file=no fixtures/generate.jl`;
the full mode also regenerates `hmc_trajectory`.
Regenerate only the stdlib RNG metadata with
`julia --startup-file=no fixtures/generate.jl reproducible_rng`; this focused
path does not require a Gaugefields.jl checkout and touches no other fixture.
Regenerate only the HMC oracle with
`GAUGEFIELDS_JL_DIR=/path/to/Gaugefields.jl julia --startup-file=no fixtures/generate.jl hmc_trajectory`;
this focused path requires the pinned, clean Gaugefields.jl checkout and touches
only `hmc_trajectory`. The script activates that checkout, rejects tracked dirty
state, and records the loaded package version and clean checkout commit in every
metadata file.

TA momentum fields use four independent compact, column-major
`TypedTensor<f64>` values of shape `[8, NX, NY, NZ, NT]`. The first axis is
Gell-Mann coefficient order; direction remains separate. `sample_momentum`
fills directions 0 through 3 in that storage order, and the HMC fixture stores
those arrays as `p_initial0.npy`/`p_final0.npy` through direction 3. Proposed
link arrays live beside them as `u_proposed0.npy` through `u_proposed3.npy`.
The HMC fixture metadata records the exact RNG state, draw-position words,
Hamiltonian values, formulas, provenance, and observed absolute comparison
tolerances.

The Julia source roots for this contract are
`src/4D/nowing/gaugefields_4D_nowing.jl:18` (the rank-six storage type),
`:41` (allocation shape), and `:73-79` (component indexing). Construction
enters through `src/AbstractGaugefields.jl:358`; deterministic hot fields use
the `StableRNG(123)` branch at `src/4D/nowing/gaugefields_4D_nowing.jl:253-254`.
