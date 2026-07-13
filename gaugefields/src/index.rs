//! Zero-based x-fast indexing corresponding to Gaugefields.jl's one-based
//! `get_latticeindex` / component access in
//! `src/4D/nowing/gaugefields_4D_nowing.jl:77-88`. Rust site `x + NX*(y +
//! NY*(z + NZ*t))` is exactly Julia's loop order after subtracting one from
//! each Julia coordinate.

use crate::{require_su3, GaugeError, GaugeLinks, LatticeShape4, Mat3};

pub fn site_index(coords: [usize; 4], lattice: LatticeShape4) -> Result<usize, GaugeError> {
    let [nx, ny, nz, _] = lattice.extents();
    for (axis, (&coordinate, &extent)) in coords.iter().zip(lattice.extents().iter()).enumerate() {
        if coordinate >= extent {
            return Err(GaugeError::CoordinateOutOfBounds {
                axis,
                coordinate,
                extent,
            });
        }
    }
    let [x, y, z, t] = coords;
    Ok(x + nx * (y + ny * (z + nz * t)))
}
pub fn coords_from_site_index(
    site: usize,
    lattice: LatticeShape4,
) -> Result<[usize; 4], GaugeError> {
    if site >= lattice.nv() {
        return Err(GaugeError::SiteOutOfBounds {
            site,
            volume: lattice.nv(),
        });
    }
    let [nx, ny, nz, _] = lattice.extents();
    let x = site % nx;
    let q = site / nx;
    let y = q % ny;
    let q = q / ny;
    let z = q % nz;
    let t = q / nz;
    Ok([x, y, z, t])
}
pub fn neighbor_site(
    site: usize,
    direction: usize,
    offset: i64,
    lattice: LatticeShape4,
) -> Result<usize, GaugeError> {
    if direction >= 4 {
        return Err(GaugeError::InvalidDirection { direction });
    }
    let mut c = coords_from_site_index(site, lattice)?;
    let n = lattice.extents()[direction] as i128;
    c[direction] = (c[direction] as i128 + i128::from(offset)).rem_euclid(n) as usize;
    site_index(c, lattice)
}
pub fn load_link(links: &GaugeLinks, direction: usize, site: usize) -> Result<Mat3, GaugeError> {
    require_su3(links)?;
    let link = links
        .links()
        .get(direction)
        .ok_or(GaugeError::InvalidDirection { direction })?;
    if site >= links.lattice().nv() {
        return Err(GaugeError::SiteOutOfBounds {
            site,
            volume: links.lattice().nv(),
        });
    }
    let offset = site.checked_mul(9).ok_or(GaugeError::AllocationOverflow)?;
    Mat3::load(
        link.typed()
            .host_data()
            .map_err(|e| GaugeError::Tensor(e.to_string()))?,
        offset,
    )
}
pub fn store_link(
    links: &mut GaugeLinks,
    direction: usize,
    site: usize,
    value: Mat3,
) -> Result<(), GaugeError> {
    require_su3(links)?;
    let volume = links.lattice().nv();
    if site >= volume {
        return Err(GaugeError::SiteOutOfBounds { site, volume });
    }
    let link = links
        .links_mut()
        .get_mut(direction)
        .ok_or(GaugeError::InvalidDirection { direction })?;
    let offset = site.checked_mul(9).ok_or(GaugeError::AllocationOverflow)?;
    value.store(
        link.typed_mut()
            .host_data_mut()
            .map_err(|e| GaugeError::Tensor(e.to_string()))?,
        offset,
    )
}
