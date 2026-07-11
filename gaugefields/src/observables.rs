use crate::{
    cold_su3, load_link, neighbor_site, require_su3, store_link, GaugeError, GaugeLinks, Mat3,
};

fn forward(links: &GaugeLinks) -> Result<Vec<[usize; 4]>, GaugeError> {
    let l = links.lattice();
    (0..l.nv())
        .map(|s| {
            Ok([
                neighbor_site(s, 0, 1, l)?,
                neighbor_site(s, 1, 1, l)?,
                neighbor_site(s, 2, 1, l)?,
                neighbor_site(s, 3, 1, l)?,
            ])
        })
        .collect()
}

/// Sum of `Re tr P_mu,nu` over six positive planes and all sites.
pub fn plaquette_sum(links: &GaugeLinks) -> Result<f64, GaugeError> {
    require_su3(links)?;
    let f = forward(links)?;
    let mut sum = 0.0;
    for (s, n) in f.iter().enumerate() {
        for mu in 0..4 {
            for nu in (mu + 1)..4 {
                let p = load_link(links, mu, s)?
                    .mul(load_link(links, nu, n[mu])?)
                    .mul_adj_right(load_link(links, mu, n[nu])?)
                    .mul_adj_right(load_link(links, nu, s)?);
                sum += p.trace().re;
            }
        }
    }
    Ok(sum)
}
/// Plaquette divided by `6 NV NC`.
pub fn normalized_plaquette(links: &GaugeLinks) -> Result<f64, GaugeError> {
    Ok(plaquette_sum(links)? / (6 * links.lattice().nv() * links.nc()) as f64)
}
/// Wilson action `-(beta/NC) sum Re tr P`.
pub fn wilson_action(links: &GaugeLinks, beta: f64) -> Result<f64, GaugeError> {
    Ok(-beta / (links.nc() as f64) * plaquette_sum(links)?)
}
/// Forward/upper measurement staple, distinct from the force staple.
pub fn measurement_staple(links: &GaugeLinks) -> Result<GaugeLinks, GaugeError> {
    require_su3(links)?;
    let f = forward(links)?;
    let mut out = cold_su3(links.lattice())?;
    for (s, n) in f.iter().enumerate() {
        for mu in 0..4 {
            let mut v = Mat3::zero();
            for nu in 0..4 {
                if nu != mu {
                    let term = load_link(links, nu, s)?
                        .mul(load_link(links, mu, n[nu])?)
                        .mul_adj_right(load_link(links, nu, n[mu])?);
                    v.add_scaled_real(1.0, term);
                }
            }
            store_link(&mut out, mu, s, v)?;
        }
    }
    Ok(out)
}
