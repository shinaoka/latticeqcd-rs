//! Validated SU(3) lattice gauge fields backed by tenferro tensors.

mod error;
mod field;
mod fixture;
mod index;
mod mat3;
mod observables;

#[cfg(feature = "autodiff")]
pub mod autodiff;

pub use error::GaugeError;
pub use field::{cold_su3, require_su3, Boundary, GaugeLinkTensor, GaugeLinks, LatticeShape4};
pub use fixture::{load_fixture, Fixture, FixtureMetadata};
pub use index::{coords_from_site_index, load_link, neighbor_site, site_index, store_link};
pub use mat3::Mat3;
pub use observables::{measurement_staple, normalized_plaquette, plaquette_sum, wilson_action};
