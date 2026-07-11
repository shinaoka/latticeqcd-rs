use crate::{
    cold_su3, load_link, neighbor_site, require_su3, store_link, GaugeError, GaugeLinks, Mat3,
};
use tenferro_tensor::Tensor;
type NeighborTable = Vec<[usize; 4]>;

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
fn staple(u: &GaugeLinks) -> Result<GaugeLinks, GaugeError> {
    require_su3(u)?;
    let (p, m) = neighbors(u)?;
    let mut out = cold_su3(u.lattice())?;
    for s in 0..u.lattice().nv() {
        for mu in 0..4 {
            let mut v = Mat3::zero();
            for nu in 0..4 {
                if nu != mu {
                    let upper = load_link(u, nu, s)?
                        .mul(load_link(u, mu, p[s][nu])?)
                        .mul_adj_right(load_link(u, nu, p[s][mu])?);
                    let sm = m[s][nu];
                    let lower = load_link(u, nu, sm)?
                        .adjoint()
                        .mul(load_link(u, mu, sm)?)
                        .mul(load_link(u, nu, p[sm][mu])?);
                    v.add_scaled_real(1.0, upper);
                    v.add_scaled_real(1.0, lower);
                }
            }
            store_link(&mut out, mu, s, v)?;
        }
    }
    Ok(out)
}
/// Julia-compatible plaquette payload `dsdu=(beta/2)V†` (no `1/NC`).
pub fn dsdu(u: &GaugeLinks, beta: f64) -> Result<GaugeLinks, GaugeError> {
    let v = staple(u)?;
    let mut out = cold_su3(u.lattice())?;
    let f = 0.5 * beta;
    for mu in 0..4 {
        for s in 0..u.lattice().nv() {
            store_link(
                &mut out,
                mu,
                s,
                load_link(&v, mu, s)?
                    .adjoint()
                    .scaled(num_complex::Complex64::new(f, 0.0)),
            )?;
        }
    }
    Ok(out)
}
/// Dense complex gradient under `dS=Re tr(G† dU)`.
pub fn action_gradient(u: &GaugeLinks, beta: f64) -> Result<GaugeLinks, GaugeError> {
    let v = staple(u)?;
    let mut out = cold_su3(u.lattice())?;
    let f = -beta / (u.nc() as f64);
    for mu in 0..4 {
        for s in 0..u.lattice().nv() {
            store_link(
                &mut out,
                mu,
                s,
                load_link(&v, mu, s)?.scaled(num_complex::Complex64::new(f, 0.0)),
            )?;
        }
    }
    Ok(out)
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
    let d = dsdu(u, beta)?;
    let [nx, ny, nz, nt] = u.lattice().extents();
    let mut ts = Vec::with_capacity(4);
    for mu in 0..4 {
        let mut data = vec![0.0; 8 * u.lattice().nv()];
        for s in 0..u.lattice().nv() {
            let mut c = [0.0; 8];
            Mat3::add_ta_coefficients(&mut c, 1.0, load_link(u, mu, s)?.mul(load_link(&d, mu, s)?));
            data[8 * s..8 * s + 8].copy_from_slice(&c);
        }
        ts.push(
            Tensor::from_vec_col_major(vec![8, nx, ny, nz, nt], data)
                .map_err(|e| GaugeError::Tensor(e.to_string()))?,
        );
    }
    Ok(GaugeForce {
        tensors: ts
            .try_into()
            .map_err(|_| GaugeError::Tensor("four force tensors".into()))?,
    })
}
