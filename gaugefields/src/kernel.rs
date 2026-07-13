use crate::{neighbor_site, require_su3, GaugeError, GaugeLinks, LatticeShape4, Mat3};
use num_complex::Complex64;
use tenferro_tensor::Tensor;

type NeighborTable = Vec<[usize; 4]>;

pub(crate) struct PreparedGaugeField<'a> {
    lattice: LatticeShape4,
    links: [&'a [Complex64]; 4],
    plus: NeighborTable,
    minus: NeighborTable,
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
        let neighbor_bytes = lattice
            .nv()
            .checked_mul(std::mem::size_of::<[usize; 4]>())
            .and_then(|n| n.checked_mul(2))
            .ok_or(GaugeError::AllocationOverflow)?;
        if neighbor_bytes > isize::MAX as usize {
            return Err(GaugeError::AllocationOverflow);
        }
        let mut plus = Vec::with_capacity(lattice.nv());
        let mut minus = Vec::with_capacity(lattice.nv());
        for site in 0..lattice.nv() {
            plus.push([
                neighbor_site(site, 0, 1, lattice)?,
                neighbor_site(site, 1, 1, lattice)?,
                neighbor_site(site, 2, 1, lattice)?,
                neighbor_site(site, 3, 1, lattice)?,
            ]);
            minus.push([
                neighbor_site(site, 0, -1, lattice)?,
                neighbor_site(site, 1, -1, lattice)?,
                neighbor_site(site, 2, -1, lattice)?,
                neighbor_site(site, 3, -1, lattice)?,
            ]);
        }
        Ok(Self {
            lattice,
            links,
            plus,
            minus,
        })
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
        for (site, next) in self.plus.iter().enumerate() {
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let p = self
                        .link(mu, site)?
                        .mul(self.link(nu, next[mu])?)
                        .mul_adj_right(self.link(mu, next[nu])?)
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
                    .mul(self.link(mu, self.plus[site][nu])?)
                    .mul_adj_right(self.link(nu, self.plus[site][mu])?);
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
                    .mul(self.link(mu, self.plus[site][nu])?)
                    .mul_adj_right(self.link(nu, self.plus[site][mu])?);
                let back = self.minus[site][nu];
                let lower = self
                    .link(nu, back)?
                    .adjoint()
                    .mul(self.link(mu, back)?)
                    .mul(self.link(nu, self.plus[back][mu])?);
                staple.add_scaled_real(1.0, upper);
                staple.add_scaled_real(1.0, lower);
            }
        }
        Ok(staple)
    }
}
