//! Validated SU(3) lattice gauge fields backed by tenferro tensors.

mod error;
mod field;
mod fixture;
mod mat3;

#[cfg(feature = "autodiff")]
pub mod autodiff;

pub use error::GaugeError;
pub use field::{cold_su3, require_su3, Boundary, GaugeLinkTensor, GaugeLinks, LatticeShape4};
pub use fixture::{load_fixture, Fixture, FixtureMetadata};
pub use mat3::Mat3;
