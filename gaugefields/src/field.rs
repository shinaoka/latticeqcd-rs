use crate::GaugeError;
use num_complex::Complex64;
use tenferro_tensor::{DType, Tensor};

/// Four positive lattice extents ordered `[NX, NY, NZ, NT]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatticeShape4([usize; 4]);

impl LatticeShape4 {
    pub fn new(extents: [usize; 4]) -> Result<Self, GaugeError> {
        if let Some(axis) = extents.iter().position(|&n| n == 0) {
            return Err(GaugeError::InvalidExtent { axis });
        }
        Ok(Self(extents))
    }
    pub const fn extents(self) -> [usize; 4] {
        self.0
    }
    pub fn nv(self) -> usize {
        self.0.iter().product()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Boundary {
    Periodic,
}

/// One validated direction tensor with shape `[3,3,NX,NY,NZ,NT]`.
#[derive(Debug)]
pub struct GaugeLinkTensor {
    tensor: Tensor,
    lattice: LatticeShape4,
    nc: usize,
}

impl GaugeLinkTensor {
    pub fn new(tensor: Tensor, lattice: LatticeShape4) -> Result<Self, GaugeError> {
        if tensor.dtype() != DType::C64 {
            return Err(GaugeError::DType {
                found: format!("{:?}", tensor.dtype()),
            });
        }
        if tensor.shape().len() != 6 {
            return Err(GaugeError::Rank {
                expected: 6,
                found: tensor.shape().len(),
            });
        }
        let nc = tensor.shape()[0];
        if nc == 0 || tensor.shape()[1] != nc {
            return Err(GaugeError::Shape {
                expected: vec![nc, nc],
                found: tensor.shape()[..2].to_vec(),
            });
        }
        let [nx, ny, nz, nt] = lattice.extents();
        let expected = vec![nc, nc, nx, ny, nz, nt];
        if tensor.shape() != expected {
            return Err(GaugeError::Shape {
                expected,
                found: tensor.shape().to_vec(),
            });
        }
        Ok(Self {
            tensor,
            lattice,
            nc,
        })
    }
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    pub const fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }
    pub const fn nc(&self) -> usize {
        self.nc
    }
}

#[derive(Debug)]
pub struct GaugeLinks {
    links: [GaugeLinkTensor; 4],
    boundary: Boundary,
}

impl GaugeLinks {
    pub fn new(links: [GaugeLinkTensor; 4]) -> Result<Self, GaugeError> {
        let lattice = links[0].lattice;
        let nc = links[0].nc;
        if let Some(mu) = links
            .iter()
            .position(|link| link.lattice != lattice || link.nc != nc)
        {
            return Err(GaugeError::InconsistentMu { mu });
        }
        Ok(Self {
            links,
            boundary: Boundary::Periodic,
        })
    }
    pub fn links(&self) -> &[GaugeLinkTensor; 4] {
        &self.links
    }
    pub fn into_links(self) -> [GaugeLinkTensor; 4] {
        self.links
    }
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }
    pub const fn lattice(&self) -> LatticeShape4 {
        self.links[0].lattice
    }
    pub const fn nc(&self) -> usize {
        self.links[0].nc
    }
}

/// Validates the boundary before entering any fixed-size SU(3) kernel.
pub fn require_su3(links: &GaugeLinks) -> Result<(), GaugeError> {
    if links.nc() == 3 {
        Ok(())
    } else {
        Err(GaugeError::UnsupportedNc { found: links.nc() })
    }
}

pub fn cold_su3(lattice: LatticeShape4) -> Result<GaugeLinks, GaugeError> {
    let [nx, ny, nz, nt] = lattice.extents();
    let mut values = vec![Complex64::default(); 9 * lattice.nv()];
    for block in values.chunks_exact_mut(9) {
        for i in 0..3 {
            block[i + 3 * i] = Complex64::new(1.0, 0.0);
        }
    }
    let make = || {
        Tensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], values.clone())
            .map_err(|e| GaugeError::Tensor(e.to_string()))
            .and_then(|t| GaugeLinkTensor::new(t, lattice))
    };
    GaugeLinks::new([make()?, make()?, make()?, make()?])
}
