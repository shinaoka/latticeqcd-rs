//! Host field layout and algebra parallel to
//! `LatticeDiracOperators.jl/src/AbstractFermions_4D.jl` and
//! `src/WilsonFermion/WilsonFermion_4D_nowing.jl` at revision
//! `bdef628184597815ba3e0cddf2536df767e78a02`.

use crate::DiracError;
use gaugefields::LatticeShape4;
use num_complex::Complex64;
use std::fmt;
use tenferro_tensor::{MemoryKind, Tensor, TypedTensor};

const NC: usize = 3;
const RANK: usize = 6;

/// A validated host-resident fermion field.
///
/// The owned tensor uses compact column-major storage with logical shape
/// `[3, components, NX, NY, NZ, NT]`. Color is the fastest dimension, followed
/// by the component dimension and then the x-fast site. Raw storage is private;
/// use [`FermionField::component`] for checked logical access.
pub struct FermionField {
    tensor: TypedTensor<Complex64>,
    lattice: LatticeShape4,
    components: usize,
    shape: [usize; RANK],
    element_count: usize,
}

impl fmt::Debug for FermionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FermionField")
            .field("lattice", &self.lattice)
            .field("components", &self.components)
            .finish()
    }
}

impl FermionField {
    /// Construct a field from an owned typed tensor after validating all layout
    /// and host-storage invariants.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a wrong rank, shape, component count,
    /// placement, allocation size, or non-finite element.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::FermionField;
    /// use gaugefields::LatticeShape4;
    /// use num_complex::Complex64;
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    /// let tensor = TypedTensor::from_vec_col_major(
    ///     vec![3, 4, 1, 1, 1, 1],
    ///     vec![Complex64::new(0.0, 0.0); 12],
    /// )?;
    /// let field = FermionField::from_typed(tensor, lattice, 4)?;
    /// assert_eq!(field.components(), 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_typed(
        tensor: TypedTensor<Complex64>,
        lattice: LatticeShape4,
        components: usize,
    ) -> Result<Self, DiracError> {
        let shape = validated_shape(lattice, components)?;
        let element_count = checked_size(lattice, components)?;
        let bytes = element_count
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(DiracError::AllocationOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(DiracError::AllocationOverflow);
        }
        if tensor.rank() != RANK {
            return Err(DiracError::Rank {
                expected: RANK,
                found: tensor.rank(),
            });
        }
        let expected = shape.to_vec();
        if tensor.shape() != expected {
            return Err(DiracError::Shape {
                expected,
                found: tensor.shape().to_vec(),
            });
        }
        if !is_host_placement(&tensor.placement().memory_kind)
            || tensor.placement().device.is_some()
        {
            return Err(DiracError::Placement {
                found: format!("{:?}", tensor.placement()),
            });
        }
        let values = tensor.host_data().map_err(|_| DiracError::Placement {
            found: format!("{:?}", tensor.placement()),
        })?;
        if values.len() != element_count {
            return Err(DiracError::StorageInvariant);
        }
        for (offset, value) in values.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(DiracError::NonFinite { offset });
            }
        }
        Ok(Self {
            tensor,
            lattice,
            components,
            shape,
            element_count,
        })
    }

    /// Construct a field from compact column-major values in Rust layout.
    ///
    /// Values must be ordered as `[color, component, x, y, z, t]` in the
    /// tensor's column-major physical order.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the component count, checked size, value
    /// count, or finite-value contract is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::FermionField;
    /// use gaugefields::LatticeShape4;
    /// use num_complex::Complex64;
    ///
    /// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    /// let field = FermionField::from_vec_col_major(
    ///     lattice,
    ///     1,
    ///     vec![Complex64::new(2.0, -1.0); 3],
    /// )?;
    /// assert_eq!(field.component(2, 0, 0)?, Complex64::new(2.0, -1.0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_vec_col_major(
        lattice: LatticeShape4,
        components: usize,
        values: Vec<Complex64>,
    ) -> Result<Self, DiracError> {
        let shape = validated_shape(lattice, components)?;
        let element_count = checked_size(lattice, components)?;
        if values.len() != element_count {
            return Err(DiracError::Shape {
                expected: shape.to_vec(),
                found: vec![values.len()],
            });
        }
        for (offset, value) in values.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(DiracError::NonFinite { offset });
            }
        }
        let tensor = TypedTensor::from_vec_col_major(shape.to_vec(), values)
            .map_err(|error| DiracError::Tensor(error.to_string()))?;
        Self::from_typed(tensor, lattice, components)
    }

    /// Construct a zero-filled host field after checking its allocation size.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation-size or component-count error.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::FermionField;
    /// use gaugefields::LatticeShape4;
    ///
    /// let field = FermionField::zeros(LatticeShape4::new([2, 1, 1, 1])?, 4)?;
    /// assert_eq!(field.len(), 24);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn zeros(lattice: LatticeShape4, components: usize) -> Result<Self, DiracError> {
        let element_count = checked_size(lattice, components)?;
        let bytes = element_count
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(DiracError::AllocationOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(DiracError::AllocationOverflow);
        }
        Self::from_vec_col_major(
            lattice,
            components,
            vec![Complex64::default(); element_count],
        )
    }

    /// Construct a field from a dtype-erased tensor, accepting only C64 host
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns a typed dtype, layout, placement, size, or finite-value error.
    pub fn try_from_tensor(
        tensor: Tensor,
        lattice: LatticeShape4,
        components: usize,
    ) -> Result<Self, DiracError> {
        match tensor {
            Tensor::C64(tensor) => Self::from_typed(tensor, lattice, components),
            other => Err(DiracError::DType {
                found: format!("{:?}", other.dtype()),
            }),
        }
    }

    /// Return the validated lattice shape.
    pub const fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }

    /// Return the validated component count, either one or four.
    pub const fn components(&self) -> usize {
        self.components
    }

    /// Return the logical six-dimensional tensor shape.
    pub const fn shape(&self) -> [usize; RANK] {
        self.shape
    }

    /// Return the number of complex elements in the field.
    pub const fn len(&self) -> usize {
        self.element_count
    }

    /// Return whether the field contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.element_count == 0
    }

    /// Return one logical color/component/site value with checked indices.
    ///
    /// `site` is the x-fast linear site index
    /// `x + NX * (y + NY * (z + NZ * t))`.
    ///
    /// # Errors
    ///
    /// Returns a typed bounds or storage error.
    pub fn component(
        &self,
        color: usize,
        component: usize,
        site: usize,
    ) -> Result<Complex64, DiracError> {
        let offset = self.component_offset(color, component, site)?;
        self.host_data()?
            .get(offset)
            .copied()
            .ok_or(DiracError::StorageInvariant)
    }

    /// Return the Hermitian inner product `self† rhs`.
    ///
    /// Julia: `LinearAlgebra.dot` in `src/AbstractFermions_4D.jl` at revision
    /// `bdef628184597815ba3e0cddf2536df767e78a02`.
    ///
    /// # Errors
    ///
    /// Returns a typed mismatch, placement, storage, or numerical-range error.
    pub fn inner_product(&self, rhs: &Self) -> Result<Complex64, DiracError> {
        self.ensure_compatible(rhs, "inner_product")?;
        let left = self.host_data()?;
        let right = rhs.host_data()?;
        let mut sum = Complex64::default();
        for (a, b) in left.iter().zip(right) {
            sum += a.conj() * *b;
        }
        if !sum.re.is_finite() || !sum.im.is_finite() {
            return Err(DiracError::NumericalRange);
        }
        Ok(sum)
    }

    /// Return the squared Euclidean norm of the field.
    ///
    /// Julia: `LinearAlgebra.dot` in `src/AbstractFermions_4D.jl` at revision
    /// `bdef628184597815ba3e0cddf2536df767e78a02`, specialized to `self†self`.
    ///
    /// # Errors
    ///
    /// Returns a typed placement, storage, or numerical-range error.
    pub fn norm_squared(&self) -> Result<f64, DiracError> {
        let values = self.host_data()?;
        let mut sum = 0.0;
        for value in values {
            sum += value.norm_sqr();
        }
        if !sum.is_finite() {
            return Err(DiracError::NumericalRange);
        }
        Ok(sum)
    }

    /// Return `gamma5 * self` for a four-component Wilson field.
    ///
    /// Julia: `mul_γ5x!` in
    /// `src/WilsonFermion/WilsonFermion_4D_nowing.jl` at revision
    /// `bdef628184597815ba3e0cddf2536df767e78a02`.
    ///
    /// The pinned chiral convention is `gamma5 = diag(-1, -1, +1, +1)`.
    ///
    /// # Errors
    ///
    /// Returns a typed component, storage, or numerical-range error.
    pub fn gamma5(&self) -> Result<Self, DiracError> {
        if self.components != 4 {
            return Err(DiracError::ComponentsMismatch {
                operand: "gamma5",
                expected: 4,
                found: self.components,
            });
        }
        let mut values = self.host_data()?.to_vec();
        for site in 0..self.lattice.nv() {
            for component in 0..4 {
                let sign = if component < 2 { -1.0 } else { 1.0 };
                for color in 0..NC {
                    let offset = self.component_offset(color, component, site)?;
                    let value = values.get_mut(offset).ok_or(DiracError::StorageInvariant)?;
                    *value *= sign;
                }
            }
        }
        Self::from_vec_col_major(self.lattice, self.components, values)
    }

    /// Explicitly duplicate this field.
    pub fn try_clone(&self) -> Result<Self, DiracError> {
        let values = self.host_data()?.to_vec();
        Self::from_vec_col_major(self.lattice, self.components, values)
    }

    pub(crate) fn host_data(&self) -> Result<&[Complex64], DiracError> {
        self.tensor.host_data().map_err(|_| DiracError::Placement {
            found: format!("{:?}", self.tensor.placement()),
        })
    }

    pub(crate) fn ensure_finite(&self) -> Result<(), DiracError> {
        for value in self.host_data()? {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(DiracError::NumericalRange);
            }
        }
        Ok(())
    }

    pub(crate) fn copy_from(&mut self, source: &Self) -> Result<(), DiracError> {
        self.ensure_compatible(source, "copy_from")?;
        source.ensure_finite()?;
        let source_data = source.host_data()?;
        let destination = self.host_data_mut()?;
        destination.copy_from_slice(source_data);
        Ok(())
    }

    pub(crate) fn add_scaled(
        &mut self,
        factor: Complex64,
        source: &Self,
    ) -> Result<(), DiracError> {
        self.ensure_compatible(source, "add_scaled")?;
        if !factor.re.is_finite() || !factor.im.is_finite() {
            return Err(DiracError::NumericalRange);
        }
        let left = self.host_data()?;
        let right = source.host_data()?;
        for (left_value, right_value) in left.iter().zip(right) {
            let value = *left_value + factor * *right_value;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(DiracError::NumericalRange);
            }
        }
        let destination = self.host_data_mut()?;
        for (destination_value, right_value) in destination.iter_mut().zip(right) {
            *destination_value += factor * *right_value;
        }
        Ok(())
    }

    pub(crate) fn add_scaled_self(
        &mut self,
        factor: Complex64,
        source: &Self,
    ) -> Result<(), DiracError> {
        self.ensure_compatible(source, "add_scaled_self")?;
        if !factor.re.is_finite() || !factor.im.is_finite() {
            return Err(DiracError::NumericalRange);
        }
        let left = self.host_data()?;
        let right = source.host_data()?;
        for (left_value, right_value) in left.iter().zip(right) {
            let value = factor * *left_value + *right_value;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(DiracError::NumericalRange);
            }
        }
        let destination = self.host_data_mut()?;
        for (destination_value, right_value) in destination.iter_mut().zip(right) {
            *destination_value = factor * *destination_value + *right_value;
        }
        Ok(())
    }

    pub(crate) fn host_data_mut(&mut self) -> Result<&mut [Complex64], DiracError> {
        let found = format!("{:?}", self.tensor.placement());
        self.tensor
            .host_data_mut()
            .map_err(|_| DiracError::Placement { found })
    }

    pub(crate) fn ensure_compatible(
        &self,
        rhs: &Self,
        operation: &'static str,
    ) -> Result<(), DiracError> {
        if self.lattice != rhs.lattice {
            return Err(DiracError::LatticeMismatch {
                operand: operation,
                expected: self.lattice,
                found: rhs.lattice,
            });
        }
        if self.components != rhs.components {
            return Err(DiracError::ComponentsMismatch {
                operand: operation,
                expected: self.components,
                found: rhs.components,
            });
        }
        let left = self.host_data()?;
        let right = rhs.host_data()?;
        if left.len() != self.element_count || right.len() != rhs.element_count {
            return Err(DiracError::StorageInvariant);
        }
        Ok(())
    }

    fn component_offset(
        &self,
        color: usize,
        component: usize,
        site: usize,
    ) -> Result<usize, DiracError> {
        if color >= NC {
            return Err(DiracError::ColorOutOfBounds { color });
        }
        if component >= self.components {
            return Err(DiracError::ComponentOutOfBounds {
                component,
                components: self.components,
            });
        }
        if site >= self.lattice.nv() {
            return Err(DiracError::SiteOutOfBounds {
                site,
                volume: self.lattice.nv(),
            });
        }
        let component_site = component
            .checked_add(
                self.components
                    .checked_mul(site)
                    .ok_or(DiracError::AllocationOverflow)?,
            )
            .ok_or(DiracError::AllocationOverflow)?;
        color
            .checked_add(
                NC.checked_mul(component_site)
                    .ok_or(DiracError::AllocationOverflow)?,
            )
            .ok_or(DiracError::AllocationOverflow)
    }
}

fn is_host_placement(kind: &MemoryKind) -> bool {
    matches!(kind, MemoryKind::UnpinnedHost | MemoryKind::PinnedHost)
}

fn validated_shape(lattice: LatticeShape4, components: usize) -> Result<[usize; RANK], DiracError> {
    validate_components(components)?;
    let [nx, ny, nz, nt] = lattice.extents();
    Ok([NC, components, nx, ny, nz, nt])
}

fn validate_components(components: usize) -> Result<(), DiracError> {
    if matches!(components, 1 | 4) {
        Ok(())
    } else {
        Err(DiracError::InvalidComponents { found: components })
    }
}

fn checked_size(lattice: LatticeShape4, components: usize) -> Result<usize, DiracError> {
    validate_components(components)?;
    NC.checked_mul(components)
        .and_then(|value| value.checked_mul(lattice.nv()))
        .ok_or(DiracError::AllocationOverflow)
}

#[cfg(test)]
mod tests;
