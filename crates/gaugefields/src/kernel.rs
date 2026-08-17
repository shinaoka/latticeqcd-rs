use crate::{require_su3, GaugeError, GaugeLinks, LatticeShape4, Mat3};
use num_complex::Complex64;
use tenferro_tensor::Tensor;

pub(crate) fn validate_beta(beta: f64) -> Result<(), GaugeError> {
    if beta.is_finite() {
        Ok(())
    } else {
        Err(GaugeError::NonFiniteBeta { found: beta })
    }
}

pub(crate) struct PreparedGaugeField<'a> {
    lattice: LatticeShape4,
    links: [&'a [Complex64]; 4],
    site_strides: [usize; 4],
}

impl<'a> PreparedGaugeField<'a> {
    pub(crate) fn new(links: &'a GaugeLinks) -> Result<Self, GaugeError> {
        require_su3(links)?;
        let lattice = links.lattice();
        let [u0, u1, u2, u3] = links.links();
        let host = |link: &'a crate::GaugeLinkTensor| {
            link.typed()
                .host_data()
                .map_err(|source| GaugeError::placement("PreparedGaugeField::new", source))
        };
        let links = [host(u0)?, host(u1)?, host(u2)?, host(u3)?];
        Self::from_parts(lattice, links)
    }

    pub(crate) fn from_tensors(inputs: [&'a Tensor; 4]) -> Result<Self, GaugeError> {
        let typed = inputs.map(|tensor| match tensor {
            Tensor::C64(value) => Ok(value),
            other => Err(GaugeError::DType {
                found: format!("{:?}", other.dtype()),
            }),
        });
        let [u0, u1, u2, u3] = typed;
        let [u0, u1, u2, u3] = [u0?, u1?, u2?, u3?];
        if u0.rank() != 6 {
            return Err(GaugeError::Rank {
                expected: 6,
                found: u0.rank(),
            });
        }
        let shape = u0.shape();
        if shape[0] != 3 || shape[1] != 3 {
            return Err(GaugeError::Shape {
                expected: vec![3, 3],
                found: shape[..2].to_vec(),
            });
        }
        for (mu, tensor) in [u0, u1, u2, u3].iter().enumerate().skip(1) {
            if tensor.shape() != shape {
                return Err(GaugeError::InconsistentMu { mu });
            }
        }
        let lattice = LatticeShape4::new([shape[2], shape[3], shape[4], shape[5]])?;
        let host = |tensor: &'a tenferro_tensor::TypedTensor<Complex64>| {
            tensor
                .host_data()
                .map_err(|source| GaugeError::placement("PreparedGaugeField::from_tensors", source))
        };
        Self::from_parts(lattice, [host(u0)?, host(u1)?, host(u2)?, host(u3)?])
    }

    fn from_parts(lattice: LatticeShape4, links: [&'a [Complex64]; 4]) -> Result<Self, GaugeError> {
        let [nx, ny, nz, _] = lattice.extents();
        let xy = nx.checked_mul(ny).ok_or(GaugeError::VolumeOverflow)?;
        let xyz = xy.checked_mul(nz).ok_or(GaugeError::VolumeOverflow)?;
        Ok(Self {
            lattice,
            links,
            site_strides: [1, nx, xy, xyz],
        })
    }

    fn shifted_site(&self, site: usize, mu: usize, forward: bool) -> Result<usize, GaugeError> {
        if site >= self.nv() {
            return Err(GaugeError::SiteOutOfBounds {
                site,
                volume: self.nv(),
            });
        }
        let extent = *self
            .lattice
            .extents()
            .get(mu)
            .ok_or(GaugeError::InvalidDirection { direction: mu })?;
        let stride = self.site_strides[mu];
        let coordinate = (site / stride) % extent;
        let wrap = (extent - 1)
            .checked_mul(stride)
            .ok_or(GaugeError::VolumeOverflow)?;
        // INVARIANT: `site < nv`, positive validated extents, and column-major
        // strides make both the adjacent step and boundary wrap stay in 0..nv.
        if forward {
            if coordinate + 1 < extent {
                site.checked_add(stride).ok_or(GaugeError::VolumeOverflow)
            } else {
                site.checked_sub(wrap).ok_or(GaugeError::VolumeOverflow)
            }
        } else if coordinate > 0 {
            site.checked_sub(stride).ok_or(GaugeError::VolumeOverflow)
        } else {
            site.checked_add(wrap).ok_or(GaugeError::VolumeOverflow)
        }
    }

    fn plus_site(&self, site: usize, mu: usize) -> Result<usize, GaugeError> {
        self.shifted_site(site, mu, true)
    }

    fn minus_site(&self, site: usize, mu: usize) -> Result<usize, GaugeError> {
        self.shifted_site(site, mu, false)
    }

    #[cfg(test)]
    fn auxiliary_metadata_bytes(&self) -> usize {
        std::mem::size_of_val(&self.site_strides)
    }

    pub(crate) fn nv(&self) -> usize {
        self.lattice.nv()
    }

    pub(crate) const fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }

    pub(crate) fn link(&self, mu: usize, site: usize) -> Result<Mat3, GaugeError> {
        Mat3::load(
            self.links[mu],
            site.checked_mul(9).ok_or(GaugeError::AllocationOverflow)?,
        )
    }

    pub(crate) fn plaquette_sum(&self) -> Result<f64, GaugeError> {
        let mut sum = 0.0;
        // INVARIANT: validated SU(3) links use compact nine-element site blocks.
        for site in 0..self.nv() {
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let next_mu = self.plus_site(site, mu)?;
                    let next_nu = self.plus_site(site, nu)?;
                    let p = self
                        .link(mu, site)?
                        .mul(self.link(nu, next_mu)?)
                        .mul_adj_right(self.link(mu, next_nu)?)
                        .mul_adj_right(self.link(nu, site)?);
                    sum += p.trace().re;
                }
            }
        }
        Ok(sum)
    }

    pub(crate) fn measurement_staple(&self, site: usize, mu: usize) -> Result<Mat3, GaugeError> {
        let mut staple = Mat3::zero();
        for nu in 0..4 {
            if nu != mu {
                let term = self
                    .link(nu, site)?
                    .mul(self.link(mu, self.plus_site(site, nu)?)?)
                    .mul_adj_right(self.link(nu, self.plus_site(site, mu)?)?);
                staple.add_scaled_real(1.0, term);
            }
        }
        Ok(staple)
    }

    pub(crate) fn force_staple(&self, site: usize, mu: usize) -> Result<Mat3, GaugeError> {
        let mut staple = Mat3::zero();
        for nu in 0..4 {
            if nu != mu {
                let upper = self
                    .link(nu, site)?
                    .mul(self.link(mu, self.plus_site(site, nu)?)?)
                    .mul_adj_right(self.link(nu, self.plus_site(site, mu)?)?);
                let back = self.minus_site(site, nu)?;
                let lower = self
                    .link(nu, back)?
                    .adjoint()
                    .mul(self.link(mu, back)?)
                    .mul(self.link(nu, self.plus_site(back, mu)?)?);
                staple.add_scaled_real(1.0, upper);
                staple.add_scaled_real(1.0, lower);
            }
        }
        Ok(staple)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor_site;

    #[test]
    fn prepared_metadata_is_constant_size_and_neighbors_wrap_exactly() {
        let lattice = LatticeShape4::new([1_000_000, 3, 2, 5]).unwrap();
        let empty = &[][..];
        let prepared = PreparedGaugeField::from_parts(lattice, [empty; 4]).unwrap();
        assert!(prepared.auxiliary_metadata_bytes() <= 8 * std::mem::size_of::<usize>());

        for site in [0, 1, lattice.nv() / 2, lattice.nv() - 1] {
            for mu in 0..4 {
                assert_eq!(
                    prepared.plus_site(site, mu).unwrap(),
                    neighbor_site(site, mu, 1, lattice).unwrap()
                );
                assert_eq!(
                    prepared.minus_site(site, mu).unwrap(),
                    neighbor_site(site, mu, -1, lattice).unwrap()
                );
            }
        }
    }
}
