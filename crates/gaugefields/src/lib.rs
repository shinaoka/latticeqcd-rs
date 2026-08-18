//! Validated SU(3) lattice gauge fields backed by tenferro tensors.

#[cfg(feature = "autodiff")]
mod ad;
mod error;
mod evolution;
mod extension;
mod field;
mod fixture;
mod force;
mod heatbath;
mod hmc;
mod ildg;
mod index;
mod kernel;
mod mat3;
mod observables;
mod rng;
mod stout;

#[cfg(feature = "autodiff")]
pub use ad::ad_rules;
pub use error::GaugeError;
pub use evolution::{exp_ta, exp_ta_update, normalize_su3, CpuEvolutionContext};
pub use extension::{runtime_modules, wilson_action_traced};
pub use field::{
    cold_su3, require_su3, Boundary, GaugeLinkTensor, GaugeLinks, LatticeShape4, TaGaugeField,
};
pub use fixture::{load_fixture, Fixture, FixtureMetadata};
pub use force::{action_gradient, dsdu, gauge_force};
pub use heatbath::{heatbath_sweep, HeatbathParams, HeatbathSweepStats};
pub use hmc::{
    hamiltonian, hmc_update, kinetic_energy, leapfrog_trajectory, sample_momentum, HmcOutcome,
    HmcParams,
};
pub use ildg::{read_ildg, write_ildg};
pub use index::{coords_from_site_index, load_link, neighbor_site, site_index, store_link};
pub use kernel::HostGaugeLinks;
pub use mat3::Mat3;
pub use observables::{measurement_staple, normalized_plaquette, plaquette_sum, wilson_action};
pub use rng::ReproducibleRng;
pub use stout::stout_step;
