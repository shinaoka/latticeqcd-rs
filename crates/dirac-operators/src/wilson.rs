//! Host Wilson stencils derived from the pinned v0.6.4
//! `LatticeDiracOperators.jl` `mk_gamma`, `Wx!`, and `Wdagx_noclover!`
//! conventions. Gauge access deliberately stays in `Gaugefields::HostGaugeLinks`.

use crate::{DiracError, FermionBoundary, FermionField};
use gaugefields::{coords_from_site_index, GaugeLinks, HostGaugeLinks, LatticeShape4, Mat3};
use num_complex::Complex64;
use std::fmt;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);
const CI: Complex64 = Complex64::new(0.0, 1.0);
const CNI: Complex64 = Complex64::new(0.0, -1.0);
const CN1: Complex64 = Complex64::new(-1.0, 0.0);
const GAMMA: [[[Complex64; 4]; 4]; 4] = [
    [
        [C0, C0, C0, CNI],
        [C0, C0, CNI, C0],
        [C0, CI, C0, C0],
        [CI, C0, C0, C0],
    ],
    [
        [C0, C0, C0, CN1],
        [C0, C0, C1, C0],
        [C0, C1, C0, C0],
        [CN1, C0, C0, C0],
    ],
    [
        [C0, C0, CNI, C0],
        [C0, C0, C0, CI],
        [CI, C0, C0, C0],
        [C0, CNI, C0, C0],
    ],
    [
        [C0, C0, CN1, C0],
        [C0, C0, C0, CN1],
        [CN1, C0, C0, C0],
        [C0, CN1, C0, C0],
    ],
];

pub(crate) type ColorSpinor = [[Complex64; 4]; 3];

/// The approved common interface for checked fermion operators.
pub trait FermionOperator {
    /// Return the operator's four-dimensional lattice.
    fn lattice(&self) -> LatticeShape4;

    /// Return the number of field components accepted by the operator.
    fn components(&self) -> usize;

    /// Apply the operator without permitting input/output aliasing.
    ///
    /// The output is not changed when validation or arithmetic fails.
    fn apply_into(&self, output: &mut FermionField, input: &FermionField)
        -> Result<(), DiracError>;

    /// Apply the operator using caller-owned reusable workspace.
    ///
    /// The default delegates to [`Self::apply_into`]. Operators with
    /// solve-local scratch requirements override this method so Krylov loops
    /// do not allocate fields per iteration.
    fn apply_into_with_scratch(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        scratch: &mut [FermionField],
    ) -> Result<(), DiracError> {
        let _ = scratch;
        self.apply_into(output, input)
    }
}

/// Marker for the Hermitian-positive contract required by conjugate gradient.
///
/// The marker has no runtime methods: an implementation is responsible for
/// preserving the mathematical contract, while the solver still checks every
/// arithmetic denominator and residual. The composed Wilson `D†D` operator is
/// the first implementation; shifted normal operators can implement the same
/// marker when they are added in a later task.
///
/// # Examples
///
/// ```
/// use dirac_operators::{HermitianPositiveOperator, WilsonDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
///
/// fn accepts_hermitian_positive<O: HermitianPositiveOperator>(_: &O) {}
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let dirac = WilsonDirac::new(&links, 0.1)?;
/// let normal = dirac.normal();
/// accepts_hermitian_positive(&normal);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait HermitianPositiveOperator: FermionOperator {}

/// A borrowed, host-side hopping-normalized Wilson operator.
///
/// The operator uses the pinned Euclidean chiral gamma basis, `r=1`, and
///
/// `D = I - kappa * sum_mu ((1-gamma_mu) U_mu(x) shift_+
/// + (1+gamma_mu) U_mu(x-mu)† shift_-)`.
///
/// It retains one validated [`HostGaugeLinks`] view and never copies gauge
/// storage. Rebuild it after a gauge update to borrow the new link snapshot.
pub struct WilsonDirac<'a> {
    links: HostGaugeLinks<'a>,
    lattice: LatticeShape4,
    kappa: f64,
    boundary: FermionBoundary,
    site_stride: usize,
}

impl fmt::Debug for WilsonDirac<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WilsonDirac")
            .field("lattice", &self.lattice)
            .field("kappa", &self.kappa)
            .field("boundary", &self.boundary)
            .finish()
    }
}

impl<'a> WilsonDirac<'a> {
    /// Construct a Wilson operator with the default spatial-periodic,
    /// temporal-antiperiodic boundary.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid links or non-positive/non-finite
    /// `kappa`.
    pub fn new(links: &'a GaugeLinks, kappa: f64) -> Result<Self, DiracError> {
        Self::with_boundary(links, kappa, FermionBoundary::default())
    }

    /// Construct a Wilson operator with explicit fermion boundary signs.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid links, boundary signs, or
    /// non-positive/non-finite `kappa`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::{FermionBoundary, FermionOperator, WilsonDirac};
    /// use gaugefields::{cold_su3, LatticeShape4};
    ///
    /// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    /// let links = cold_su3(lattice)?;
    /// let operator = WilsonDirac::with_boundary(
    ///     &links,
    ///     0.1,
    ///     FermionBoundary::new([1, 1, 1, 1])?,
    /// )?;
    /// assert_eq!(operator.components(), 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_boundary(
        links: &'a GaugeLinks,
        kappa: f64,
        boundary: FermionBoundary,
    ) -> Result<Self, DiracError> {
        Self::with_r(links, kappa, 1.0, boundary)
    }

    /// Construct a Wilson operator while explicitly validating the supported
    /// Wilson parameter `r=1`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if `r` is non-finite or differs from one, or if
    /// the links, boundary, or `kappa` are invalid.
    pub fn with_r(
        links: &'a GaugeLinks,
        kappa: f64,
        r: f64,
        boundary: FermionBoundary,
    ) -> Result<Self, DiracError> {
        validate_kappa(kappa)?;
        if !r.is_finite() {
            return Err(DiracError::NonFiniteWilsonR { found: r });
        }
        if r != 1.0 {
            return Err(DiracError::UnsupportedWilsonR { found: r });
        }
        let host = links.host_view()?;
        validate_host_links(&host)?;
        let lattice = host.lattice();
        let site_stride = 3usize
            .checked_mul(4)
            .ok_or(DiracError::AllocationOverflow)?;
        Ok(Self {
            links: host,
            lattice,
            kappa,
            boundary,
            site_stride,
        })
    }

    /// Return the hopping parameter.
    pub const fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Return the validated fermion boundary signs.
    pub const fn boundary(&self) -> FermionBoundary {
        self.boundary
    }

    /// Return a borrowed adjoint view of this operator.
    pub fn adjoint(&self) -> WilsonAdjoint<'_, 'a> {
        WilsonAdjoint { parent: self }
    }

    /// Return the composed normal operator `D†D`.
    pub fn normal(&self) -> NormalOperator<&WilsonDirac<'a>> {
        NormalOperator::new(self)
    }

    pub(crate) fn host_links(&self) -> &HostGaugeLinks<'a> {
        &self.links
    }

    fn apply_into_kind(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        adjoint: bool,
    ) -> Result<(), DiracError> {
        // Julia: `Wx!` and `Wdagx_noclover!` in
        // `src/WilsonFermion/WilsonFermion.jl` at revision
        // `bdef628184597815ba3e0cddf2536df767e78a02`; the boolean selects
        // the matching rminus-gamma or rplus-gamma projector path.
        let operation = if adjoint {
            "WilsonAdjoint"
        } else {
            "WilsonDirac"
        };
        self.validate_operands(output, input, operation)?;
        let input_data = input.host_data()?;
        let count = self
            .site_stride
            .checked_mul(self.lattice.nv())
            .ok_or(DiracError::AllocationOverflow)?;
        let mut values = vec![C0; count];
        self.apply_to_data(&mut values, input_data, adjoint)?;
        output.host_data_mut()?.copy_from_slice(&values);
        Ok(())
    }

    fn apply_into_kind_with_scratch(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        adjoint: bool,
        scratch: &mut FermionField,
    ) -> Result<(), DiracError> {
        let operation = if adjoint {
            "WilsonAdjoint"
        } else {
            "WilsonDirac"
        };
        self.validate_operands(output, input, operation)?;
        self.validate_workspace(scratch, operation)?;
        let input_data = input.host_data()?;
        {
            let scratch_data = scratch.host_data_mut()?;
            self.apply_to_data(scratch_data, input_data, adjoint)?;
        }
        output.copy_from(scratch)
    }

    fn apply_to_data(
        &self,
        output: &mut [Complex64],
        input: &[Complex64],
        adjoint: bool,
    ) -> Result<(), DiracError> {
        let expected = self
            .site_stride
            .checked_mul(self.lattice.nv())
            .ok_or(DiracError::AllocationOverflow)?;
        if input.len() != expected || output.len() != expected {
            return Err(DiracError::StorageInvariant);
        }
        for site in 0..self.lattice.nv() {
            let mut result = read_site(input, site, self.site_stride)?;
            for direction in 0..4 {
                let (plus_site, plus_sign) = self.neighbor(site, direction, 1)?;
                let (minus_site, minus_sign) = self.neighbor(site, direction, -1)?;
                let forward = self.links.link(direction, site)?;
                let backward = self.links.link(direction, minus_site)?.adjoint();
                let plus = project_color_spin(
                    forward,
                    read_site(input, plus_site, self.site_stride)?,
                    direction,
                    if adjoint { 1 } else { -1 },
                );
                let minus = project_color_spin(
                    backward,
                    read_site(input, minus_site, self.site_stride)?,
                    direction,
                    if adjoint { -1 } else { 1 },
                );
                let plus_factor = self.kappa * f64::from(plus_sign);
                let minus_factor = self.kappa * f64::from(minus_sign);
                for (result_color, (plus_color, minus_color)) in
                    result.iter_mut().zip(plus.iter().zip(minus.iter()))
                {
                    for (result_value, (plus_value, minus_value)) in result_color
                        .iter_mut()
                        .zip(plus_color.iter().zip(minus_color.iter()))
                    {
                        *result_value -= plus_factor * *plus_value;
                        *result_value -= minus_factor * *minus_value;
                    }
                }
            }
            for spinor in result {
                for value in spinor {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(DiracError::NumericalRange);
                    }
                }
            }
            write_site(output, site, self.site_stride, result)?;
        }
        Ok(())
    }

    fn validate_operands(
        &self,
        output: &FermionField,
        input: &FermionField,
        operation: &'static str,
    ) -> Result<(), DiracError> {
        if std::ptr::eq(output, input) {
            return Err(DiracError::AliasedFields { operation });
        }
        if input.lattice() != self.lattice {
            return Err(DiracError::LatticeMismatch {
                operand: "input",
                expected: self.lattice,
                found: input.lattice(),
            });
        }
        if output.lattice() != self.lattice {
            return Err(DiracError::LatticeMismatch {
                operand: "output",
                expected: self.lattice,
                found: output.lattice(),
            });
        }
        if input.components() != 4 {
            return Err(DiracError::ComponentsMismatch {
                operand: "input",
                expected: 4,
                found: input.components(),
            });
        }
        if output.components() != 4 {
            return Err(DiracError::ComponentsMismatch {
                operand: "output",
                expected: 4,
                found: output.components(),
            });
        }
        let input_data = input.host_data()?;
        let output_data = output.host_data()?;
        let expected = self
            .site_stride
            .checked_mul(self.lattice.nv())
            .ok_or(DiracError::AllocationOverflow)?;
        if input_data.len() != expected || output_data.len() != expected {
            return Err(DiracError::StorageInvariant);
        }
        Ok(())
    }

    fn validate_workspace(
        &self,
        workspace: &FermionField,
        operation: &'static str,
    ) -> Result<(), DiracError> {
        if workspace.lattice() != self.lattice {
            return Err(DiracError::LatticeMismatch {
                operand: operation,
                expected: self.lattice,
                found: workspace.lattice(),
            });
        }
        if workspace.components() != 4 {
            return Err(DiracError::ComponentsMismatch {
                operand: operation,
                expected: 4,
                found: workspace.components(),
            });
        }
        let expected = self
            .site_stride
            .checked_mul(self.lattice.nv())
            .ok_or(DiracError::AllocationOverflow)?;
        if workspace.host_data()?.len() != expected {
            return Err(DiracError::StorageInvariant);
        }
        Ok(())
    }

    // Julia: `shift_fermion`/`shifted_fermion!` in
    // `src/WilsonFermion/WilsonFermion_4D_nowing.jl` at revision
    // `bdef628184597815ba3e0cddf2536df767e78a02`; this checked helper applies
    // the boundary sign exactly at a wrapped one-hop displacement.
    pub(crate) fn neighbor(
        &self,
        site: usize,
        direction: usize,
        displacement: isize,
    ) -> Result<(usize, i8), DiracError> {
        let coordinates = coords_from_site_index(site, self.lattice)?;
        let extent = self
            .lattice
            .extents()
            .get(direction)
            .copied()
            .ok_or(gaugefields::GaugeError::InvalidDirection { direction })?;
        let wraps = match displacement {
            1 => coordinates[direction] == extent - 1,
            -1 => coordinates[direction] == 0,
            _ => false,
        };
        let site = self.links.shifted_site(site, direction, displacement)?;
        let sign = if wraps {
            self.boundary.sign(direction)?
        } else {
            1
        };
        Ok((site, sign))
    }
}

impl FermionOperator for WilsonDirac<'_> {
    fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }

    fn components(&self) -> usize {
        4
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        self.apply_into_kind(output, input, false)
    }

    fn apply_into_with_scratch(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        scratch: &mut [FermionField],
    ) -> Result<(), DiracError> {
        let workspace = scratch.first_mut().ok_or(DiracError::StorageInvariant)?;
        self.apply_into_kind_with_scratch(output, input, false, workspace)
    }
}

/// A borrowed view applying the Hermitian adjoint `D†` of a Wilson operator.
pub struct WilsonAdjoint<'op, 'links> {
    parent: &'op WilsonDirac<'links>,
}

impl fmt::Debug for WilsonAdjoint<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WilsonAdjoint")
            .field("lattice", &self.parent.lattice)
            .field("kappa", &self.parent.kappa)
            .field("boundary", &self.parent.boundary)
            .finish()
    }
}

impl FermionOperator for WilsonAdjoint<'_, '_> {
    fn lattice(&self) -> LatticeShape4 {
        self.parent.lattice
    }

    fn components(&self) -> usize {
        4
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        self.parent.apply_into_kind(output, input, true)
    }

    fn apply_into_with_scratch(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        scratch: &mut [FermionField],
    ) -> Result<(), DiracError> {
        let workspace = scratch.first_mut().ok_or(DiracError::StorageInvariant)?;
        self.parent
            .apply_into_kind_with_scratch(output, input, true, workspace)
    }
}

/// A normal operator formed by composing one Wilson operator with its adjoint.
///
/// Julia: `DdagD_Wilson_operator` and `LinearAlgebra.mul!` in
/// `src/WilsonFermion/WilsonFermion.jl` and `src/Diracoperators.jl` at revision
/// `bdef628184597815ba3e0cddf2536df767e78a02`.
///
/// `NormalOperator::new(&dirac)` performs `D†(D(input))`; it does not use a
/// separately derived stencil and commits only after both scratch evaluations
/// succeed.
pub struct NormalOperator<O> {
    operator: O,
}

impl<O> fmt::Debug for NormalOperator<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NormalOperator").finish()
    }
}

impl<'op, 'links> NormalOperator<&'op WilsonDirac<'links>> {
    /// Construct a normal composition from a borrowed Wilson operator.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::{FermionField, FermionOperator, NormalOperator, WilsonDirac};
    /// use gaugefields::{cold_su3, LatticeShape4};
    ///
    /// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    /// let links = cold_su3(lattice)?;
    /// let dirac = WilsonDirac::new(&links, 0.1)?;
    /// let normal = NormalOperator::new(&dirac);
    /// let input = FermionField::zeros(lattice, 4)?;
    /// let mut output = FermionField::zeros(lattice, 4)?;
    /// normal.apply_into(&mut output, &input)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(operator: &'op WilsonDirac<'links>) -> Self {
        Self { operator }
    }
}

impl<'op, 'links> FermionOperator for NormalOperator<&'op WilsonDirac<'links>> {
    fn lattice(&self) -> LatticeShape4 {
        self.operator.lattice
    }

    fn components(&self) -> usize {
        4
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        let mut scratch = [
            FermionField::zeros(self.operator.lattice, 4)?,
            FermionField::zeros(self.operator.lattice, 4)?,
        ];
        self.apply_into_with_scratch(output, input, &mut scratch)
    }

    fn apply_into_with_scratch(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        scratch: &mut [FermionField],
    ) -> Result<(), DiracError> {
        self.operator
            .validate_operands(output, input, "NormalOperator")?;
        if scratch.len() < 2 {
            return Err(DiracError::StorageInvariant);
        }
        let (first, second) = scratch.split_at_mut(1);
        let temporary = &mut first[0];
        let result = &mut second[0];
        self.operator
            .validate_workspace(temporary, "NormalOperator")?;
        self.operator.validate_workspace(result, "NormalOperator")?;
        let input_data = input.host_data()?;
        {
            let temporary_data = temporary.host_data_mut()?;
            self.operator
                .apply_to_data(temporary_data, input_data, false)?;
        }
        {
            let temporary_data = temporary.host_data()?;
            let result_data = result.host_data_mut()?;
            self.operator
                .apply_to_data(result_data, temporary_data, true)?;
        }
        output.copy_from(result)
    }
}

impl<'op, 'links> HermitianPositiveOperator for NormalOperator<&'op WilsonDirac<'links>> {}

pub(crate) fn validate_kappa(kappa: f64) -> Result<(), DiracError> {
    if !kappa.is_finite() {
        return Err(DiracError::NonFiniteKappa { found: kappa });
    }
    if kappa <= 0.0 {
        return Err(DiracError::NonPositiveKappa { found: kappa });
    }
    Ok(())
}

pub(crate) fn validate_host_links(host: &HostGaugeLinks<'_>) -> Result<(), DiracError> {
    for site in 0..host.lattice().nv() {
        for direction in 0..4 {
            let matrix = host.link(direction, site)?;
            for value in matrix.as_array() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(DiracError::NumericalRange);
                }
            }
        }
    }
    Ok(())
}

fn read_site(
    values: &[Complex64],
    site: usize,
    site_stride: usize,
) -> Result<ColorSpinor, DiracError> {
    let offset = site
        .checked_mul(site_stride)
        .ok_or(DiracError::AllocationOverflow)?;
    let end = offset
        .checked_add(site_stride)
        .ok_or(DiracError::AllocationOverflow)?;
    let block = values
        .get(offset..end)
        .ok_or(DiracError::StorageInvariant)?;
    let mut result = [[C0; 4]; 3];
    // Julia: `AbstractFermions_4D.jl::get_latticeindex_fermion` at
    // revision `bdef628184597815ba3e0cddf2536df767e78a02` keeps color
    // fastest inside each site/component block.
    for (color, result_color) in result.iter_mut().enumerate() {
        for (component, result_value) in result_color.iter_mut().enumerate() {
            *result_value = *block
                .get(color + 3 * component)
                .ok_or(DiracError::StorageInvariant)?;
        }
    }
    Ok(result)
}

fn write_site(
    values: &mut [Complex64],
    site: usize,
    site_stride: usize,
    spinor: ColorSpinor,
) -> Result<(), DiracError> {
    let offset = site
        .checked_mul(site_stride)
        .ok_or(DiracError::AllocationOverflow)?;
    let end = offset
        .checked_add(site_stride)
        .ok_or(DiracError::AllocationOverflow)?;
    let block = values
        .get_mut(offset..end)
        .ok_or(DiracError::StorageInvariant)?;
    for (color, spinor_color) in spinor.iter().enumerate() {
        for (component, &spinor_value) in spinor_color.iter().enumerate() {
            let value = block
                .get_mut(color + 3 * component)
                .ok_or(DiracError::StorageInvariant)?;
            *value = spinor_value;
        }
    }
    Ok(())
}

// Julia: the gauge-color multiplication in
// `src/AbstractFermions_4D.jl::LinearAlgebra.mul!` at revision
// `bdef628184597815ba3e0cddf2536df767e78a02`, followed by the projector in
// `src/WilsonFermion/WilsonFermion.jl::Wx!`/`Wdagx_noclover!`.
pub(crate) fn project_color_spin(
    matrix: Mat3,
    input: ColorSpinor,
    direction: usize,
    gamma_sign: i8,
) -> ColorSpinor {
    let mut color_transformed = [[C0; 4]; 3];
    for component in 0..4 {
        for row in 0..3 {
            color_transformed[row][component] = matrix[(row, 0)] * input[0][component]
                + matrix[(row, 1)] * input[1][component]
                + matrix[(row, 2)] * input[2][component];
        }
    }
    let mut result = [[C0; 4]; 3];
    for (color, result_color) in result.iter_mut().enumerate() {
        *result_color = project_spin(direction, gamma_sign, color_transformed[color]);
    }
    result
}

// Julia: `mk_gamma` constructs `rplusγ` and `rminusγ` in
// `src/WilsonFermion/WilsonFermion.jl` at revision
// `bdef628184597815ba3e0cddf2536df767e78a02`; `gamma_sign` selects one of
// those two matrices for the matching hop.
pub(crate) fn project_spin(
    direction: usize,
    gamma_sign: i8,
    input: [Complex64; 4],
) -> [Complex64; 4] {
    let mut result = [C0; 4];
    for row in 0..4 {
        let mut value = C0;
        for column in 0..4 {
            let mut coefficient = if row == column { C1 } else { C0 };
            if gamma_sign > 0 {
                coefficient += GAMMA[direction][row][column];
            } else {
                coefficient -= GAMMA[direction][row][column];
            }
            value += coefficient * input[column];
        }
        result[row] = value;
    }
    result
}

#[cfg(test)]
mod tests;
