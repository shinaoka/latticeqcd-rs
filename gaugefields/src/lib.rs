//! Validated SU(3) lattice gauge fields backed by tenferro tensors.

mod error;
mod field;
mod fixture;

pub use error::GaugeError;
pub use field::{cold_su3, Boundary, GaugeLinkTensor, GaugeLinks, LatticeShape4};
pub use fixture::{load_fixture, Fixture, FixtureMetadata};

/// Returns the compact column-major offset for `[a,b,x,y,z,t]`.
#[allow(clippy::too_many_arguments)]
pub const fn flat_offset(
    a: usize,
    b: usize,
    x: usize,
    y: usize,
    z: usize,
    t: usize,
    nc: usize,
    nx: usize,
    ny: usize,
    nz: usize,
) -> usize {
    a + nc * (b + nc * (x + nx * (y + ny * (z + nz * t))))
}
