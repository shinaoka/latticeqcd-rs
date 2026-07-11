# Gauge-field layout and fixture contract

Each directed link is one compact, column-major tenferro C64 tensor of rank six
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

`fixtures/generate.jl` is the authority for checked Julia fixtures. Data not
produced by that script against Gaugefields.jl must not claim Julia provenance.
The checked `random_2x2x2x2` fixture stores exact IEEE-754 reference bits for
every component. Gaugefields.jl's reproducible initializer resets the same RNG
for each direction, so the generator applies a direction-specific periodic
site shift; this preserves the random SU(3) matrices while making direction and
axis swaps observable.

Periodic kernels read neighboring nine-element blocks directly. They do not
allocate tensor-roll buffers and do not expose Gaugefields.jl's mutable shared
`Ushifted` scratch buffer. The checked `shifts_3x2x4x5` fixture materializes
that Julia reference only as an oracle (`gaugefields_4D_nowing.jl:380-454`);
Rust parity is evaluated through `neighbor_site` reads.

The Julia source roots for this contract are
`src/4D/nowing/gaugefields_4D_nowing.jl:18` (the rank-six storage type),
`:41` (allocation shape), and `:73-79` (component indexing). Construction
enters through `src/AbstractGaugefields.jl:358`; deterministic hot fields use
the `StableRNG(123)` branch at `src/4D/nowing/gaugefields_4D_nowing.jl:253-254`.
