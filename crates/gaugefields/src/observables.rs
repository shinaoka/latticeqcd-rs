use crate::{
    kernel::{validate_beta, PreparedGaugeField},
    GaugeError, GaugeLinkTensor, GaugeLinks,
};
use num_complex::Complex64;
use tenferro_tensor::TypedTensor;

/// Sum of `Re tr P_mu,nu` over six positive planes and all sites.
pub fn plaquette_sum(links: &GaugeLinks) -> Result<f64, GaugeError> {
    PreparedGaugeField::new(links)?.plaquette_sum()
}
/// Plaquette divided by `6 NV NC`.
pub fn normalized_plaquette(links: &GaugeLinks) -> Result<f64, GaugeError> {
    Ok(plaquette_sum(links)? / (6 * links.lattice().nv() * links.nc()) as f64)
}
/// Wilson action `-(beta/NC) sum Re tr P`.
pub fn wilson_action(links: &GaugeLinks, beta: f64) -> Result<f64, GaugeError> {
    validate_beta(beta)?;
    Ok(-beta / (links.nc() as f64) * plaquette_sum(links)?)
}
/// Forward/upper measurement staple, distinct from the force staple.
pub fn measurement_staple(
    links: &GaugeLinks,
    direction: usize,
) -> Result<GaugeLinkTensor, GaugeError> {
    if direction >= 4 {
        return Err(GaugeError::InvalidDirection { direction });
    }
    let prepared = PreparedGaugeField::new(links)?;
    let value_count = links
        .lattice()
        .nv()
        .checked_mul(9)
        .ok_or(GaugeError::AllocationOverflow)?;
    let byte_count = value_count
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(GaugeError::AllocationOverflow)?;
    if byte_count > isize::MAX as usize {
        return Err(GaugeError::AllocationOverflow);
    }
    let mut values = Vec::with_capacity(value_count);
    for site in 0..prepared.nv() {
        values.extend_from_slice(prepared.measurement_staple(site, direction)?.as_array());
    }
    debug_assert_eq!(values.len(), value_count);
    let [nx, ny, nz, nt] = links.lattice().extents();
    let tensor = TypedTensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], values)
        .map_err(|error| GaugeError::Tensor(error.to_string()))?;
    GaugeLinkTensor::from_typed(tensor, links.lattice())
}
