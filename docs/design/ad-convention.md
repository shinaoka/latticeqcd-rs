# Gauge action derivative conventions

With `S=-(beta/NC)Σ Re tr P` and tenferro's real inner product
`dS=Re tr(G†dU)`, the dense gradient is `G=-(beta/NC)V`, where `V` is the
upper-plus-lower staple. `dsdu` stores `G†`. `gauge_force` stores the eight
coefficients of `TA(U dsdu)` under `A=(i/2)Σu_aλ_a`; it adds no integrator
step or extra `1/NC`. Julia's force routines may place the dagger and beta
factor at adjacent call boundaries; JAX-style AD corresponds to the dense `G`.
