//! Checked host-resident Wilson and one-link staggered fermion fields and operators.
//!
//! The Wilson convention follows the pinned `LatticeDiracOperators.jl` v0.6.4
//! sources `src/WilsonFermion/WilsonFermion.jl` (`mk_gamma`),
//! `src/WilsonFermion/WilsonFermion_4D.jl`, and
//! `src/WilsonFermion/WilsonFermion_4D_nowing.jl` at
//! `bdef628184597815ba3e0cddf2536df767e78a02`. Gauge storage and periodic
//! matrix access come from the pinned `Gaugefields.jl` v0.7.2-compatible Rust
//! `gaugefields` crate.
//!
//! The Task C Julia-parallel mapping is: `sample_pseudofermions!` to
//! [`WilsonFermiAction::sample_pseudofermion`], `evaluate_FermiAction` to
//! [`WilsonFermiAction::evaluate`], `calc_UdSfdU!` to
//! [`WilsonFermiAction::force`], and `MDstep!`/`U_update!`/`P_update!` to
//! [`wilson_leapfrog_trajectory`]. The pinned force projection
//! `Traceless_antihermitian_add!` maps to `Mat3::add_ta_coefficients` in
//! `gaugefields`; the fixture records the complete entrypoint map and all 28
//! payload files in `fixtures/fermions_task_c`. With Julia 1.12.5, two clean
//! generations matched at tree hash
//! `9462c1e4bf1f46c0929c81fd932f65dbd20f2a2b65168bb65ad8e8a4d92439af`;
//! the recorded Julia/Rust maxima are `7.16072334609889539e-15` for `X`,
//! `6.62422734006908809e-15` for `Y`, `4.54747350886464119e-13` for the
//! action, and `5.10702591327572009e-14` for the force. The established
//! all-512 finite-difference series is
//! `[5.005235745869641e-7, 1.262665048074041e-7, 3.463491360378157e-8]` at
//! epsilons `[1e-3, 5e-4, 2.5e-4]`.
//!
//! Task D is implementation-complete and independently post-reviewed. It
//! keeps the pinned staggered `Staggered_Dirac_operator`/`Dx!`,
//! adjoint, composed `DdagD_Staggered_operator`, and `shiftedcg` entrypoints
//! in [`StaggeredDirac`], [`StaggeredNormalOperator`], and [`multi_shift_cg`].
//! The independently lowered [`StaggeredClosedNormalOperator`] checks
//! `mass² I - K²`. Rust retains the eta, `K`, `D`, `Ddag`, `alpha`, `beta`,
//! `rho_m`, `rho_0`, and `rho_p` names/order while using typed errors, one host
//! view, and reusable scratch. Boundary signs are applied once per wrapped hop.
//! The deterministic fixture is `fixtures/fermions_task_d`: 37 declared
//! payloads plus metadata, with both complete trees hashing to
//! `c372e6e56bc05ebc611c6cc3dba5c247eafbc12ca58a0eee2ac3737cdbb08d4b`.
//! Its operator/identity tolerance is `2e-12`, while the absolute squared
//! solver tolerance is `1e-24` and the fresh shifted true-relative tolerance
//! is `1e-11`; these compare different quantities. D, Ddag, and K payloads are
//! bit-exact, normal-composition maxima are `3.19867204157556452e-17` and
//! `1.66533453693773481e-16` for periodic and default anti-periodic boundaries,
//! and the three Rust shifted true-relative residuals are
//! `[2.52706753667624838e-16, 5.23618850174067806e-16,
//! 3.73441567789983714e-16]` in shift order `[0.31, 0.0, 0.07]`.

//!
//! Task E also checks an independent short Julia/Rust ensemble: plaquette and
//! chiral differences are `0.154717` and `0.367498` combined standard errors.
//! [`StaggeredFermiAction`] follows the pinned v0.6.4 Nf=2 RHMC tables for
//! refresh, action, and MD force, with scalar 4097-point log-grid errors
//! `2.505791796281187e-9`, `3.9620045022559225e-9`, and
//! `1.5595609319518644e-5`. The 68-file `fermions_task_e` fixture matched on
//! two clean generations at complete-tree hash
//! `9e166b37d2c138a28f6d75395e11dc8f91f910599f0397e5888bbd738ba6d34a`.
//! Its all-512 force FD maxima at epsilons `[0.32, 0.16, 0.08, 0.04]` are
//! `[8.434653210321642e-6, 2.139177378187619e-6, 5.605769951367093e-7,
//! 1.6563038083509257e-7]`, with pass counts `[291, 442, 510, 512]` and all
//! coefficients passing at `0.04` below `5e-7`. Deterministic tests cover
//! refresh/action/force `X_j/Y_j`, U-P-U reversibility, validation and
//! transactional rollback, rejection rollback, and RNG advancement.

mod boundary;
mod error;
mod field;
mod rhmc;
mod solvers;
mod staggered;
mod staggered_action;
mod wilson;
mod wilson_action;
mod wilson_hmc;

pub use boundary::FermionBoundary;
pub use error::{DiracError, SolverError};
pub use field::FermionField;
pub use rhmc::{
    staggered_hmc_update, staggered_leapfrog_trajectory, StaggeredHmcOutcome, StaggeredHmcParams,
};
pub use solvers::{
    bicgstab, conjugate_gradient, multi_shift_cg, ConvergenceBranch, MultiShiftSolverReport,
    SolverMethod, SolverParams, SolverReport,
};
pub use staggered::{
    StaggeredAdjoint, StaggeredClosedNormalOperator, StaggeredDirac, StaggeredNormalOperator,
};
pub use staggered_action::{StaggeredActionResult, StaggeredFermiAction, StaggeredForceResult};
pub use wilson::{
    FermionOperator, HermitianPositiveOperator, NormalOperator, WilsonAdjoint, WilsonDirac,
};
pub use wilson_action::{WilsonActionResult, WilsonFermiAction, WilsonForceResult};
pub use wilson_hmc::{
    wilson_hmc_update, wilson_leapfrog_trajectory, WilsonHmcOutcome, WilsonHmcParams,
};
