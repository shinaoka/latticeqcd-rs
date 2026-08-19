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
//! Task D is implementation-complete; independent post-review remains
//! pending. It keeps the pinned staggered `Staggered_Dirac_operator`/`Dx!`,
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

mod boundary;
mod error;
mod field;
mod solvers;
mod staggered;
mod wilson;
mod wilson_action;
mod wilson_hmc;

pub use boundary::FermionBoundary;
pub use error::{DiracError, SolverError};
pub use field::FermionField;
pub use solvers::{
    bicgstab, conjugate_gradient, multi_shift_cg, ConvergenceBranch, MultiShiftSolverReport,
    SolverMethod, SolverParams, SolverReport,
};
pub use staggered::{
    StaggeredAdjoint, StaggeredClosedNormalOperator, StaggeredDirac, StaggeredNormalOperator,
};
pub use wilson::{
    FermionOperator, HermitianPositiveOperator, NormalOperator, WilsonAdjoint, WilsonDirac,
};
pub use wilson_action::{WilsonActionResult, WilsonFermiAction, WilsonForceResult};
pub use wilson_hmc::{
    wilson_hmc_update, wilson_leapfrog_trajectory, WilsonHmcOutcome, WilsonHmcParams,
};
