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
