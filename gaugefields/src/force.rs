use crate::{
    kernel::{validate_beta, PreparedGaugeField},
    GaugeError, GaugeLinkTensor, GaugeLinks, Mat3, TaGaugeField,
};
use num_complex::Complex64 as C;
use tenferro_tensor::TypedTensor;
fn checked_count(items_per_site: usize, nv: usize, item_size: usize) -> Result<usize, GaugeError> {
    let count = items_per_site
        .checked_mul(nv)
        .ok_or(GaugeError::AllocationOverflow)?;
    let bytes = count
        .checked_mul(item_size)
        .ok_or(GaugeError::AllocationOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(GaugeError::AllocationOverflow);
    }
    Ok(count)
}
fn complex_output(
    u: &GaugeLinks,
    prepared: &PreparedGaugeField<'_>,
    beta: f64,
    gradient: bool,
    mu: usize,
) -> Result<GaugeLinkTensor, GaugeError> {
    let count = checked_count(9, u.lattice().nv(), std::mem::size_of::<C>())?;
    let [nx, ny, nz, nt] = u.lattice().extents();
    let mut data = Vec::with_capacity(count);
    for s in 0..prepared.nv() {
        let v = prepared.force_staple(s, mu)?;
        let out = if gradient {
            v.scaled(C::new(-beta / u.nc() as f64, 0.0))
        } else {
            v.adjoint().scaled(C::new(0.5 * beta, 0.0))
        };
        data.extend_from_slice(out.as_array());
    }
    debug_assert_eq!(data.len(), count);
    let tensor = TypedTensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], data)
        .map_err(|e| GaugeError::Tensor(e.to_string()))?;
    GaugeLinkTensor::from_typed(tensor, u.lattice())
}
fn complex_outputs(
    u: &GaugeLinks,
    beta: f64,
    gradient: bool,
) -> Result<[GaugeLinkTensor; 4], GaugeError> {
    validate_beta(beta)?;
    let prepared = PreparedGaugeField::new(u)?;
    Ok([
        complex_output(u, &prepared, beta, gradient, 0)?,
        complex_output(u, &prepared, beta, gradient, 1)?,
        complex_output(u, &prepared, beta, gradient, 2)?,
        complex_output(u, &prepared, beta, gradient, 3)?,
    ])
}
/// Julia-compatible plaquette payload `dsdu=(beta/2)V†` (no `1/NC`).
pub fn dsdu(u: &GaugeLinks, beta: f64) -> Result<[GaugeLinkTensor; 4], GaugeError> {
    complex_outputs(u, beta, false)
}
/// Dense complex gradient under `dS=Re tr(G† dU)`.
pub fn action_gradient(u: &GaugeLinks, beta: f64) -> Result<[GaugeLinkTensor; 4], GaugeError> {
    complex_outputs(u, beta, true)
}
/// TA coefficients of `U_mu * dsdu_mu`, without integrator or extra `1/NC` factors.
pub fn gauge_force(u: &GaugeLinks, beta: f64) -> Result<TaGaugeField, GaugeError> {
    validate_beta(beta)?;
    let prepared = PreparedGaugeField::new(u)?;
    let count = checked_count(8, u.lattice().nv(), std::mem::size_of::<f64>())?;
    let [nx, ny, nz, nt] = u.lattice().extents();
    let mut tensors = Vec::with_capacity(4);
    for mu in 0..4 {
        let mut data = Vec::with_capacity(count);
        for s in 0..prepared.nv() {
            let local = prepared.link(mu, s)?.mul(
                prepared
                    .force_staple(s, mu)?
                    .adjoint()
                    .scaled(C::new(0.5 * beta, 0.0)),
            );
            let mut c = [0.0; 8];
            Mat3::add_ta_coefficients(&mut c, 1.0, local);
            data.extend_from_slice(&c);
        }
        debug_assert_eq!(data.len(), count);
        tensors.push(
            TypedTensor::from_vec_col_major(vec![8, nx, ny, nz, nt], data)
                .map_err(|e| GaugeError::Tensor(e.to_string()))?,
        );
    }
    TaGaugeField::new(
        tensors
            .try_into()
            .map_err(|_| GaugeError::Tensor("four force tensors".into()))?,
        u.lattice(),
    )
}
