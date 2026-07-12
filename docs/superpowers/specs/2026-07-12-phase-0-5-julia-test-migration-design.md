# Phase 0–5 Julia Test Migration Design

## Goal and selection rule

This design identifies the Gaugefields.jl tests whose behavior belongs to gaugefields-rs Phase 0–5 and records how that behavior is or should be tested in Rust. The objective is semantic coverage, not a line-for-line translation of Julia test files.

Use the Gaugefields.jl source at commit [`9e5719970770f4497405a856315c90bef7f74449`](https://github.com/shinaoka/Gaugefields.jl/tree/9e5719970770f4497405a856315c90bef7f74449/test) as the immutable reference. A Julia test is suitable now only when all operations needed to express its invariant are part of the Phase 0–5 Rust API.

Each source invariant receives one of these dispositions:

- **direct**: express the invariant as an ordinary Rust test without invoking Julia;
- **oracle**: compare Rust results with checked data produced by the pinned Julia implementation;
- **covered**: an existing Rust test already provides equal or stronger evidence;
- **candidate**: add a focused Rust regression because current evidence is incomplete;
- **deferred**: reconsider when the named later phase provides the missing operation;
- **excluded**: outside the CPU-only 4D SU(3), no-wing scope.

## Source inventory and scope boundary

Gaugefields.jl's top-level suite mixes gauge-field primitives with algorithms far beyond Phase 5. The suite inventory is visible in [`test/runtests.jl`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/runtests.jl). For the current scope:

| Julia suite | Disposition | Reason |
|---|---|---|
| [`init.jl`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/init.jl) | inspect and migrate | Contains cold/hot 4D initialization and plaquette normalization contracts used by Phase 1 and 4. |
| [`HMC_test_nowing.jl`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/HMC_test_nowing.jl) | partially migrate | Its gauge action and momentum-force update expose Phase 4–5 conventions; link evolution and trajectories require `exptU`. |
| [`HMC_test.jl`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/HMC_test.jl) | excluded | Wing storage and NC/dimensional variants are not the selected representation. |
| `HMCstout_test_nowing.jl` | deferred | Stout smearing and full trajectory behavior are not implemented. |
| `gradientflow*.jl` | deferred to Phase 8 | Requires exponential link updates and flow integration. |
| `heatbath*.jl` | excluded | Heatbath updates are not on the Phase 0–8 roadmap. |
| `Btest/*`, `test_wilson.jl` | excluded | Background fields and general Wilson-loop evaluation are intentionally out of scope. |
| `sun_embedded_instanton.jl` | excluded | SU(2)-in-SU(N) construction, topology, and NC≠3 are not Phase 0–5 APIs. |
| `gputests/*`, `JACCtest/*`, `MPIJACCtest/*` | excluded | GPU, JACC, and MPI backends are not implemented. |
| `Isingtest.jl`, `scalarnn.jl` | excluded | Different field theories. |

## Traceability matrix

### Phase 0–1: layout, initialization, and fixtures

| Julia invariant | Rust evidence | Disposition |
|---|---|---|
| Cold 4D initialization produces normalized plaquette exactly one: [`Init_cold_4D`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/init.jl#L1-L18) and the cold-start assertions in [`init.jl`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/init.jl#L133-L166). | `field_validation::cold_su3_is_identity_at_every_site_and_periodic`, `observables::cold_observables_have_exact_normalization`, and the checked cold fixture. | **covered/direct + oracle** |
| Reproducible hot 4D initialization yields a stable reference field and plaquette: [`Init_hot_4D`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/init.jl#L21-L41). | `fixtures/random_2x2x2x2`, `fixtures/random_4x4x4x4`, exact-bit fixture checks, and observable parity tests. Rust deliberately does not reproduce Julia's RNG. | **covered/oracle** |
| Julia payload is four independent direction tensors with column-major site/color order. | `layout_contract`, `fixture_roundtrip`, and non-isotropic `shift_parity`. | **covered/oracle** |
| ILDG/binary save and reload in [`Init_ildg_4D`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/init.jl#L44-L65). | No ILDG API. | **excluded** |

No additional Phase 0–1 test is warranted. Existing Rust tests are stronger than the Julia assertions because they also cover invalid dtype, rank, shape, storage order, metadata, overflow, and runtime-NC boundaries.

### Phase 2–3: local matrices and periodic access

Gaugefields.jl does not isolate the Phase 2 `Mat3` operations as top-level unit tests in this suite. The relevant behavior is exercised inside action and update code. Rust should retain independent known-value and algebraic property tests rather than copy large Julia implementation loops. `mat3_properties` already covers multiplication variants, adjoint, trace, TA projection, Gell-Mann coefficients, reconstruction, and scaling. `periodic_index` and `shift_parity` cover the Julia shift convention more directly than the HMC suite.

Disposition: **covered/direct + oracle**. No duplicate Julia-shaped test file should be added.

### Phase 4: plaquette, staple, and Wilson action

| Julia invariant | Rust evidence | Disposition |
|---|---|---|
| Four-dimensional normalization is `1/(6*NV*NC)`: [`HMC_test_4D`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/HMC_test_nowing.jl#L77-L112). | Cold normalization plus checked 2⁴/4⁴ aggregate parity. | **covered** |
| Gauge action uses a plaquette loop and its adjoint with physical beta split as beta/2: [`HMC_test_4D`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/HMC_test_nowing.jl#L114-L121). | `wilson_action`, `docs/design/wilson-action.md`, and Julia aggregate parity. | **covered/oracle** |
| Gauge part of `calc_action` is `-evaluate_GaugeAction/NC`: [`calc_action`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/HMC_test_nowing.jl#L8-L14). | Current action formula and oracle values establish one beta, but no direct beta-scaling/zero-beta regression joins the Julia convention to all Phase 4–5 APIs. | **candidate** |
| Direct plaquette and staple contraction agree. | Rust nontrivial direct-versus-staple test is stronger than the Julia HMC smoke path. | **covered/direct** |

Add one contract test on a nontrivial checked field that verifies `wilson_action(U,0)=0` and `wilson_action(U,k*beta)=k*wilson_action(U,beta)` for positive and negative finite `k`. This is a direct semantic port of how Julia stores the action coefficient, without reproducing `GaugeAction`'s general loop engine.

### Phase 5: dSdU, projected force, and dense gradient

| Julia invariant | Rust evidence | Disposition |
|---|---|---|
| `calc_dSdUμ!` feeds `Uμ*dSdUμ`, then TA coefficients are accumulated with `-epsilon*dt/NC`: [`P_update!`](https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/test/HMC_test_nowing.jl#L55-L75). | `julia_force` checks all `dSdU` and unscaled TA-force components; `force` independently checks the local TA formula. The final momentum coefficient/sign is documented but not directly regression-tested. | **candidate** |
| The plaquette action coefficient is linear in beta, so `dSdU`, projected gauge force, and dense action gradient must scale by the same factor. | One beta is covered by Julia parity and gradient by finite differences. Cross-API beta linearity is implicit rather than tested. | **candidate** |
| Dense gradient obeys the tenferro Hermitian-inner-product convention. | Three independent central finite-difference directions with second-order convergence. | **covered/direct** |

Add two focused semantic tests using the nonzero checked 2⁴ field:

1. **Beta linearity across derivative APIs.** For `k` positive and negative, compare every component of `dsdu(U,k*beta)`, `gauge_force(U,k*beta)`, and `action_gradient(U,k*beta)` with `k` times the result at `beta`; also require exact zero payloads for beta zero. Report maximum residuals.
2. **Julia momentum-update coefficient.** For representative finite `epsilon` and `dt`, verify the coefficient update computed from `gauge_force` is exactly `(-epsilon*dt/NC)` times every stored TA component. This remains a test-local calculation; it must not introduce a premature integrator or momentum-update public API.

These tests supplement rather than duplicate the checked Julia oracle. They make the extracted HMC kernel convention explicit while keeping `exptU` and trajectory behavior deferred.

## Deferred migration gates

Re-evaluate deferred tests when their required APIs land:

- **Phase 6:** traced Wilson action must replay Phase 4 eager outputs on the same fixtures.
- **Phase 7:** VJP must replay the Phase 5 dense gradient and finite-difference contracts.
- **Phase 8:** port `U_update!`, reversible/symplectic step checks, short HMC energy conservation, and flow sanity from `HMC_test_nowing.jl` and `gradientflow_test_nowing.jl`.

Acceptance-ratio assertions such as `ratio > 0.5` are stochastic integration tests, not unit tests. Even in Phase 8 they should be replaced by deterministic reversibility, energy-error scaling, and fixed-seed short-trajectory checks.

## Test organization and acceptance

Create one integration test file, `gaugefields/tests/julia_hmc_kernel_contracts.rs`, for the three candidates above. Keep helper functions private to that test file. Use the existing checked `random_2x2x2x2` fixture, validate all tensor shapes before component comparisons, reject nonfinite values, and include useful maximum-residual diagnostics.

The migration is complete when:

- the three candidate contracts fail against an intentionally wrong sign or coefficient and pass against the implementation;
- no new production API is added;
- all existing Julia oracle, finite-difference, formatting, Clippy, minimal-feature, all-feature, and documentation checks remain green;
- every selected Julia source link is commit-pinned;
- deferred and excluded suites remain explicitly documented rather than silently counted as migrated.
