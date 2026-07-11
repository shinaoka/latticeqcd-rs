use crate::{neighbor_site, require_su3, GaugeError, GaugeLinkTensor, GaugeLinks, Mat3};
use num_complex::Complex64 as C;
use tenferro_tensor::Tensor;
type NeighborTable = Vec<[usize; 4]>;

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
fn neighbors(u: &GaugeLinks) -> Result<(NeighborTable, NeighborTable), GaugeError> {
    let l = u.lattice();
    let mut p = Vec::with_capacity(l.nv());
    let mut m = Vec::with_capacity(l.nv());
    for s in 0..l.nv() {
        p.push([
            neighbor_site(s, 0, 1, l)?,
            neighbor_site(s, 1, 1, l)?,
            neighbor_site(s, 2, 1, l)?,
            neighbor_site(s, 3, 1, l)?,
        ]);
        m.push([
            neighbor_site(s, 0, -1, l)?,
            neighbor_site(s, 1, -1, l)?,
            neighbor_site(s, 2, -1, l)?,
            neighbor_site(s, 3, -1, l)?,
        ]);
    }
    Ok((p, m))
}
fn block(src: &[&[C]], mu: usize, site: usize) -> Result<Mat3, GaugeError> {
    Mat3::load(
        src[mu],
        site.checked_mul(9).ok_or(GaugeError::AllocationOverflow)?,
    )
}
fn site_staple(
    src: &[&[C]],
    p: &NeighborTable,
    m: &NeighborTable,
    s: usize,
    mu: usize,
) -> Result<Mat3, GaugeError> {
    let mut v = Mat3::zero();
    for nu in 0..4 {
        if nu != mu {
            let upper = block(src, nu, s)?
                .mul(block(src, mu, p[s][nu])?)
                .mul_adj_right(block(src, nu, p[s][mu])?);
            let sm = m[s][nu];
            let lower = block(src, nu, sm)?
                .adjoint()
                .mul(block(src, mu, sm)?)
                .mul(block(src, nu, p[sm][mu])?);
            v.add_scaled_real(1.0, upper);
            v.add_scaled_real(1.0, lower);
        }
    }
    Ok(v)
}
fn context(u: &GaugeLinks) -> Result<(Vec<&[C]>, NeighborTable, NeighborTable), GaugeError> {
    require_su3(u)?;
    let src = u
        .links()
        .iter()
        .map(|link| {
            link.tensor()
                .as_slice::<C>()
                .map_err(|e| GaugeError::Tensor(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (p, m) = neighbors(u)?;
    Ok((src, p, m))
}
fn complex_outputs(u: &GaugeLinks, beta: f64, gradient: bool) -> Result<GaugeLinks, GaugeError> {
    let (src, p, m) = context(u)?;
    let count = checked_count(9, u.lattice().nv(), std::mem::size_of::<C>())?;
    let [nx, ny, nz, nt] = u.lattice().extents();
    let mut links = Vec::with_capacity(4);
    for mu in 0..4 {
        let mut data = Vec::with_capacity(count);
        for s in 0..u.lattice().nv() {
            let v = site_staple(&src, &p, &m, s, mu)?;
            let out = if gradient {
                v.scaled(C::new(-beta / u.nc() as f64, 0.0))
            } else {
                v.adjoint().scaled(C::new(0.5 * beta, 0.0))
            };
            data.extend_from_slice(out.as_array());
        }
        debug_assert_eq!(data.len(), count);
        let tensor = Tensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], data)
            .map_err(|e| GaugeError::Tensor(e.to_string()))?;
        links.push(GaugeLinkTensor::new(tensor, u.lattice())?);
    }
    GaugeLinks::new(
        links
            .try_into()
            .map_err(|_| GaugeError::Tensor("four links".into()))?,
    )
}
/// Julia-compatible plaquette payload `dsdu=(beta/2)V†` (no `1/NC`).
pub fn dsdu(u: &GaugeLinks, beta: f64) -> Result<GaugeLinks, GaugeError> {
    complex_outputs(u, beta, false)
}
/// Dense complex gradient under `dS=Re tr(G† dU)`.
pub fn action_gradient(u: &GaugeLinks, beta: f64) -> Result<GaugeLinks, GaugeError> {
    complex_outputs(u, beta, true)
}
/// Four `[8,NX,NY,NZ,NT]` F64 coefficient tensors.
pub struct GaugeForce {
    tensors: [Tensor; 4],
}
impl GaugeForce {
    pub fn tensors(&self) -> &[Tensor; 4] {
        &self.tensors
    }
}
/// TA coefficients of `U_mu * dsdu_mu`, without integrator or extra `1/NC` factors.
pub fn gauge_force(u: &GaugeLinks, beta: f64) -> Result<GaugeForce, GaugeError> {
    let (src, p, m) = context(u)?;
    let count = checked_count(8, u.lattice().nv(), std::mem::size_of::<f64>())?;
    let [nx, ny, nz, nt] = u.lattice().extents();
    let mut tensors = Vec::with_capacity(4);
    for mu in 0..4 {
        let mut data = Vec::with_capacity(count);
        for s in 0..u.lattice().nv() {
            let local = block(&src, mu, s)?.mul(
                site_staple(&src, &p, &m, s, mu)?
                    .adjoint()
                    .scaled(C::new(0.5 * beta, 0.0)),
            );
            let mut c = [0.0; 8];
            Mat3::add_ta_coefficients(&mut c, 1.0, local);
            data.extend_from_slice(&c);
        }
        debug_assert_eq!(data.len(), count);
        tensors.push(
            Tensor::from_vec_col_major(vec![8, nx, ny, nz, nt], data)
                .map_err(|e| GaugeError::Tensor(e.to_string()))?,
        );
    }
    Ok(GaugeForce {
        tensors: tensors
            .try_into()
            .map_err(|_| GaugeError::Tensor("four force tensors".into()))?,
    })
}
