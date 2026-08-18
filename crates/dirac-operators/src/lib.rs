//! Checked host-resident Wilson fermion fields and operators.
//!
//! The Wilson convention follows the pinned `LatticeDiracOperators.jl` v0.6.4
//! sources `src/WilsonFermion/WilsonFermion.jl` (`mk_gamma`),
//! `src/WilsonFermion/WilsonFermion_4D.jl`, and
//! `src/WilsonFermion/WilsonFermion_4D_nowing.jl` at
//! `bdef628184597815ba3e0cddf2536df767e78a02`. Gauge storage and periodic
//! matrix access come from the pinned `Gaugefields.jl` v0.7.2-compatible Rust
//! `gaugefields` crate.

mod boundary;
mod error;
mod field;
mod wilson;

pub use boundary::FermionBoundary;
pub use error::DiracError;
pub use field::FermionField;
pub use wilson::{FermionOperator, NormalOperator, WilsonAdjoint, WilsonDirac};
