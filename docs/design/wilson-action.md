# Wilson action and staples

For physical `beta`, the implementation uses `S=-(beta/NC) Σ Re tr P` over
the six unoriented positive planes. Gaugefields.jl may express the same sum
with `beta/2` and both loop orientations. `measurement_staple` is the forward
staple used for the `0.5 Σ Re tr(U V†)` identity; the force staple is a separate
upper-plus-lower object. Direct `Mat3` kernels and precomputed neighbor tables
avoid intermediate rolled tensors and a general Wilson-line engine.
