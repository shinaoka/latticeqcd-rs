//! Validated SU(3) lattice gauge fields backed by tenferro tensors.

#[cfg(feature = "autodiff")]
mod ad;
mod error;
mod extension;
mod field;
mod fixture;
mod force;
mod index;
mod kernel;
mod mat3;
mod observables;

#[cfg(feature = "autodiff")]
pub use ad::ad_rules;
pub use error::GaugeError;
pub use extension::{register_runtime, wilson_action_traced};
pub use field::{
    cold_su3, require_su3, Boundary, GaugeLinkTensor, GaugeLinks, LatticeShape4, TaGaugeField,
};
pub use fixture::{load_fixture, Fixture, FixtureMetadata};
pub use force::{action_gradient, dsdu, gauge_force};
pub use index::{coords_from_site_index, load_link, neighbor_site, site_index, store_link};
pub use mat3::Mat3;
pub use observables::{measurement_staple, normalized_plaquette, plaquette_sum, wilson_action};
