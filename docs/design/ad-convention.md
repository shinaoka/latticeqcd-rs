# Gauge action derivative conventions

With `S=-(beta/NC)Σ Re tr P` and tenferro's real inner product
`dS=Re tr(G†dU)`, the dense gradient is `G=-(beta/NC)V`, where `V` is the
upper-plus-lower staple. Julia's doubled-orientation plaquette action uses
`beta/2`, so `dsdu=(beta/2)V†` before the HMC momentum update applies
`-epsilon/NC`; hence `G=-(2/NC)dsdu†`. `gauge_force` stores the eight
coefficients of `TA(U dsdu)` under `A=(i/2)Σu_aλ_a`; it adds no integrator
step or extra `1/NC`. Julia's force routines may place the dagger and beta
factor at adjacent call boundaries. For a real scalar objective, JAX's
elementwise complex gradient is `conj(G)` relative to tenferro's Hermitian `G`.

| Quantity | Convention |
| --- | --- |
| Julia/Rust `dsdu` | `(beta/2) V†` |
| tenferro Hermitian dense `G` | `-(2/NC) dsdu† = -(beta/NC)V` |
| JAX complex gradient | `conj(G)` |
| projected coefficients | `coeff(TA(U dsdu))`, with `A=(i/2)Σu_aλ_a` |
| Julia momentum update | adds `(-epsilon/NC) coeff(TA(U dsdu))` |

The factor two is the conversion from Julia's `beta/2` doubled-orientation
loop dataset to the six-plane physical action. The dagger is fixed by the
Hermitian real inner product, not by an array-layout convention.

Phase 7 applies this convention compositionally. Action linearization emits a
JVP containing only tangent slots for active link directions. Its scalar is
`Re sum_mu sum_i conj(G_mu[i]) delta_U_mu[i]`. Linear transpose consumes any
finite real scalar cotangent `c` and returns `c G_mu` for active directions
through the Wilson force extension. Applications explicitly attach `ad_rules()`
to their own `AdContext` through `with_semantic_extension_rules`. They separately install all three
`runtime_modules::<CpuBackend>()` values into their application-owned
`Runtime` using the selected `EngineId`. There is no direct action VJP or force AD rule; higher-order
differentiation through force is intentionally unsupported.

Julia parity uses `1e-13` maximum component error because each component is a
bounded sum of six fixed 3×3 `ComplexF64` products evaluated from the identical
checked fixture; no long volume reduction is involved. Aggregate observables
use `1e-12` relative error because their summation length grows with volume.
Parity failures report the measured maximum or relative residual.

Each force API stores only four column-major site strides and borrows the four
compact input slices once. Periodic neighbors use checked O(1) wrap arithmetic.
A shared site-local `Mat3` staple helper feeds exact-capacity final buffers:
`dsdu` and `action_gradient` construct only their four final C64 tensors, while
`gauge_force` writes coefficient tensors directly and never materializes a
staple, `dsdu` field, or volume-sized neighbor table.

`ExtensionShapeContext` records cross-input shape-equality constraints for
extension inputs. The Wilson extension records equality for all four links;
graph construction and concrete runtime validation reject contradictions.
Known contradictions fail graph construction, while runtime preparation and
payload validation enforce exact concrete equality.
