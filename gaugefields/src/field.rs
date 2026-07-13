use crate::GaugeError;
use num_complex::Complex64;
use std::fmt;
use tenferro_tensor::{Tensor, TypedTensor};

/// Four positive lattice extents ordered `[NX, NY, NZ, NT]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatticeShape4 {
    extents: [usize; 4],
    nv: usize,
}

impl LatticeShape4 {
    pub fn new(extents: [usize; 4]) -> Result<Self, GaugeError> {
        if let Some(axis) = extents.iter().position(|&n| n == 0) {
            return Err(GaugeError::InvalidExtent { axis });
        }
        let nv = extents.iter().try_fold(1usize, |volume, &extent| {
            volume.checked_mul(extent).ok_or(GaugeError::VolumeOverflow)
        })?;
        Ok(Self { extents, nv })
    }
    pub const fn extents(self) -> [usize; 4] {
        self.extents
    }
    pub fn nv(self) -> usize {
        self.nv
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Boundary {
    Periodic,
}

/// One validated C64 direction tensor with shape `[NC,NC,NX,NY,NZ,NT]`.
pub struct GaugeLinkTensor {
    tensor: TypedTensor<Complex64>,
    lattice: LatticeShape4,
    nc: usize,
}

impl GaugeLinkTensor {
    pub fn from_typed(
        tensor: TypedTensor<Complex64>,
        lattice: LatticeShape4,
    ) -> Result<Self, GaugeError> {
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
        let bytes = nc
            .checked_mul(nc)
            .and_then(|n| n.checked_mul(lattice.nv()))
            .and_then(|n| n.checked_mul(std::mem::size_of::<Complex64>()))
            .ok_or(GaugeError::AllocationOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(GaugeError::AllocationOverflow);
        }
        tensor
            .host_data()
            .map_err(|source| GaugeError::Tensor(source.to_string()))?;
        Ok(Self {
            tensor,
            lattice,
            nc,
        })
    }
    pub fn try_from_tensor(tensor: Tensor, lattice: LatticeShape4) -> Result<Self, GaugeError> {
        match tensor {
            Tensor::C64(tensor) => Self::from_typed(tensor, lattice),
            other => Err(GaugeError::DType {
                found: format!("{:?}", other.dtype()),
            }),
        }
    }
    pub fn typed(&self) -> &TypedTensor<Complex64> {
        &self.tensor
    }
    pub(crate) fn typed_mut(&mut self) -> &mut TypedTensor<Complex64> {
        &mut self.tensor
    }
    pub fn into_typed(self) -> TypedTensor<Complex64> {
        self.tensor
    }
    pub fn into_tensor(self) -> Tensor {
        self.tensor.into()
    }
    pub const fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }
    pub const fn nc(&self) -> usize {
        self.nc
    }
}

impl fmt::Debug for GaugeLinkTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaugeLinkTensor")
            .field("shape", &self.tensor.shape())
            .field("lattice", &self.lattice)
            .field("nc", &self.nc)
            .finish()
    }
}

pub struct GaugeLinks {
    links: [GaugeLinkTensor; 4],
    boundary: Boundary,
}

impl fmt::Debug for GaugeLinks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaugeLinks")
            .field("lattice", &self.lattice())
            .field("nc", &self.nc())
            .field("boundary", &self.boundary)
            .finish()
    }
}

/// Four compact host F64 tensors with shape `[8,NX,NY,NZ,NT]`.
pub struct TaGaugeField {
    tensors: [TypedTensor<f64>; 4],
    lattice: LatticeShape4,
}

impl TaGaugeField {
    pub fn new(tensors: [TypedTensor<f64>; 4], lattice: LatticeShape4) -> Result<Self, GaugeError> {
        let [nx, ny, nz, nt] = lattice.extents();
        let expected = vec![8, nx, ny, nz, nt];
        for tensor in &tensors {
            if tensor.rank() != 5 {
                return Err(GaugeError::Rank {
                    expected: 5,
                    found: tensor.rank(),
                });
            }
            if tensor.shape() != expected {
                return Err(GaugeError::Shape {
                    expected: expected.clone(),
                    found: tensor.shape().to_vec(),
                });
            }
            let count = 8usize
                .checked_mul(lattice.nv())
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
                .ok_or(GaugeError::AllocationOverflow)?;
            if count > isize::MAX as usize {
                return Err(GaugeError::AllocationOverflow);
            }
            tensor
                .host_data()
                .map_err(|source| GaugeError::Tensor(source.to_string()))?;
        }
        Ok(Self { tensors, lattice })
    }
    pub fn tensors(&self) -> &[TypedTensor<f64>; 4] {
        &self.tensors
    }
    pub fn into_tensors(self) -> [TypedTensor<f64>; 4] {
        self.tensors
    }
    pub const fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }
}

impl fmt::Debug for TaGaugeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaGaugeField")
            .field("shape", &self.tensors[0].shape())
            .field("lattice", &self.lattice)
            .finish()
    }
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
    pub(crate) fn links_mut(&mut self) -> &mut [GaugeLinkTensor; 4] {
        &mut self.links
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
    let value_count = 9usize
        .checked_mul(lattice.nv())
        .ok_or(GaugeError::AllocationOverflow)?;
    let byte_count = value_count
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(GaugeError::AllocationOverflow)?;
    if byte_count > isize::MAX as usize {
        return Err(GaugeError::AllocationOverflow);
    }
    let mut values = vec![Complex64::default(); value_count];
    for block in values.chunks_exact_mut(9) {
        for i in 0..3 {
            block[i + 3 * i] = Complex64::new(1.0, 0.0);
        }
    }
    let make = || {
        TypedTensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], values.clone())
            .map_err(|e| GaugeError::Tensor(e.to_string()))
            .and_then(|t| GaugeLinkTensor::from_typed(t, lattice))
    };
    GaugeLinks::new([make()?, make()?, make()?, make()?])
}
