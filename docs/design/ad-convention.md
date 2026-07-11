# Gauge action derivative conventions

With `S=-(beta/NC)Σ Re tr P` and tenferro's real inner product
`dS=Re tr(G†dU)`, the dense gradient is `G=-(beta/NC)V`, where `V` is the
upper-plus-lower staple. Julia's doubled-orientation plaquette action uses
`beta/2`, so `dsdu=(beta/2)V†` before the HMC momentum update applies
`-epsilon/NC`; hence `G=-(2/NC)dsdu†`. `gauge_force` stores the eight
coefficients of `TA(U dsdu)` under `A=(i/2)Σu_aλ_a`; it adds no integrator
step or extra `1/NC`. Julia's force routines may place the dagger and beta
factor at adjacent call boundaries; JAX-style AD corresponds to the dense `G`.

| Quantity | Convention |
| --- | --- |
| Julia/Rust `dsdu` | `(beta/2) V†` |
| tenferro/JAX dense `G` | `-(2/NC) dsdu† = -(beta/NC)V` |
| projected coefficients | `coeff(TA(U dsdu))`, with `A=(i/2)Σu_aλ_a` |
| Julia momentum update | adds `(-epsilon/NC) coeff(TA(U dsdu))` |

The factor two is the conversion from Julia's `beta/2` doubled-orientation
loop dataset to the six-plane physical action. The dagger is fixed by the
Hermitian real inner product, not by an array-layout convention.
