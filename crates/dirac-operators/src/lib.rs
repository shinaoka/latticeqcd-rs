//! Checked host-resident Wilson fermion fields and operators.
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

mod boundary;
mod error;
mod field;
mod solvers;
mod wilson;
mod wilson_action;
mod wilson_hmc;

pub use boundary::FermionBoundary;
pub use error::{DiracError, SolverError};
pub use field::FermionField;
pub use solvers::{
    bicgstab, conjugate_gradient, ConvergenceBranch, SolverMethod, SolverParams, SolverReport,
};
pub use wilson::{
    FermionOperator, HermitianPositiveOperator, NormalOperator, WilsonAdjoint, WilsonDirac,
};
pub use wilson_action::{WilsonActionResult, WilsonFermiAction, WilsonForceResult};
pub use wilson_hmc::{
    wilson_hmc_update, wilson_leapfrog_trajectory, WilsonHmcOutcome, WilsonHmcParams,
};
