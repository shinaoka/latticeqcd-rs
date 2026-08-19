//! One-link staggered stencils parallel to the pinned LatticeDiracOperators.jl
//! v0.6.4 sources:
//! `src/StaggeredFermion/StaggeredFermion.jl::LinearAlgebra.mul!` (lines
//! 166-243) and `src/StaggeredFermion/StaggeredFermion_4D_nowing.jl::Dx!`
//! (lines 43-80) and `shifted_fermion!` (lines 133-198), revision
//! `bdef628184597815ba3e0cddf2536df767e78a02`.
//!
//! The phase wrapper follows Gaugefields.jl v0.7.2
//! `src/4D/nowing/gaugefields_4D_nowing.jl` (lines 459-504) at revision
//! `9e5719970770f4497405a856315c90bef7f74449`. Rust keeps the same zero-based
//! eta phases and hop order while replacing shifted-field materialization with
//! one borrowed host view and caller-owned field scratch.

use crate::{
    DiracError, FermionBoundary, FermionField, FermionOperator, HermitianPositiveOperator,
};
use gaugefields::{coords_from_site_index, GaugeLinks, HostGaugeLinks, LatticeShape4, Mat3};
use num_complex::Complex64;
use std::fmt;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C_HALF: f64 = 0.5;
const NC: usize = 3;

/// A borrowed, host-resident one-component staggered Dirac operator.
///
/// With zero-based coordinates it applies `D = mass I + K`, where
/// `K = 1/2 sum_mu (Ustag_mu(x) chi(x+mu) -
/// Ustag_mu(x-mu)† chi(x-mu))` and
/// `eta = [1, (-1)^x, (-1)^(x+y), (-1)^(x+y+z)]`. The gauge links are
/// periodic; only fermion hops acquire the validated boundary sign when they
/// wrap an axis.
///
/// # Examples
///
/// ```
/// use dirac_operators::{FermionField, FermionOperator, StaggeredDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
/// use num_complex::Complex64;
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let operator = StaggeredDirac::new(&links, 0.17)?;
/// let input = FermionField::from_vec_col_major(
///     lattice,
///     1,
///     vec![Complex64::new(1.0, 0.0); 3],
/// )?;
/// let mut output = FermionField::zeros(lattice, 1)?;
/// operator.apply_into(&mut output, &input)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StaggeredDirac<'a> {
    links: HostGaugeLinks<'a>,
    lattice: LatticeShape4,
    mass: f64,
    boundary: FermionBoundary,
}

impl fmt::Debug for StaggeredDirac<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaggeredDirac")
            .field("lattice", &self.lattice)
            .field("mass", &self.mass)
            .field("boundary", &self.boundary)
            .finish()
    }
}

impl<'a> StaggeredDirac<'a> {
    /// Construct a staggered operator with spatial-periodic and
    /// temporal-antiperiodic boundaries `[+1,+1,+1,-1]`.
    ///
    /// # Errors
    ///
    /// Returns [`DiracError::NonFiniteMass`] or
    /// [`DiracError::NonPositiveMass`] for an invalid mass, or a gauge/typed
    /// numerical error when the links cannot provide a finite host SU(3) view.
    pub fn new(links: &'a GaugeLinks, mass: f64) -> Result<Self, DiracError> {
        Self::with_boundary(links, mass, FermionBoundary::default())
    }

    /// Construct a staggered operator with explicit per-axis fermion signs.
    ///
    /// # Errors
    ///
    /// Returns [`DiracError::NonFiniteMass`] or
    /// [`DiracError::NonPositiveMass`] for an invalid mass, or a gauge/typed
    /// numerical error when the links cannot provide a finite host SU(3) view.
    pub fn with_boundary(
        links: &'a GaugeLinks,
        mass: f64,
        boundary: FermionBoundary,
    ) -> Result<Self, DiracError> {
        validate_mass(mass)?;
        let host = links.host_view()?;
        crate::wilson::validate_host_links(&host)?;
        let lattice = host.lattice();
        Ok(Self {
            links: host,
            lattice,
            mass,
            boundary,
        })
    }

    /// Return the validated positive mass.
    pub const fn mass(&self) -> f64 {
        self.mass
    }

    /// Return the validated fermion boundary signs.
    pub const fn boundary(&self) -> FermionBoundary {
        self.boundary
    }

    pub(crate) fn host_links(&self) -> &HostGaugeLinks<'a> {
        &self.links
    }

    /// Return a borrowed `D† = mass I - K` view.
    pub fn adjoint(&self) -> StaggeredAdjoint<'_, 'a> {
        StaggeredAdjoint { parent: self }
    }

    /// Return the composed normal operator `D†D`.
    pub fn normal(&self) -> StaggeredNormalOperator<'_, 'a> {
        StaggeredNormalOperator { operator: self }
    }

    /// Return the independently lowered normal stencil `mass² I - K²`.
    pub fn normal_closed_form(&self) -> StaggeredClosedNormalOperator<'_, 'a> {
        StaggeredClosedNormalOperator { operator: self }
    }

    fn apply_into_kind(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        adjoint: bool,
    ) -> Result<(), DiracError> {
        self.validate_operands(
            output,
            input,
            if adjoint {
                "StaggeredAdjoint"
            } else {
                "StaggeredDirac"
            },
        )?;
        let count = NC
            .checked_mul(self.lattice.nv())
            .ok_or(DiracError::AllocationOverflow)?;
        let mut values = vec![C0; count];
        self.apply_to_data(&mut values, input.host_data()?, adjoint)?;
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
            "StaggeredAdjoint"
        } else {
            "StaggeredDirac"
        };
        self.validate_operands(output, input, operation)?;
        self.validate_workspace(scratch, operation)?;
        self.apply_to_data(scratch.host_data_mut()?, input.host_data()?, adjoint)?;
        output.copy_from(scratch)
    }

    fn apply_to_data(
        &self,
        output: &mut [Complex64],
        input: &[Complex64],
        adjoint: bool,
    ) -> Result<(), DiracError> {
        self.apply_hopping_to_data(output, input)?;
        let sign = if adjoint { -1.0 } else { 1.0 };
        for (site, output_site) in output.chunks_exact_mut(NC).enumerate() {
            let offset = site.checked_mul(NC).ok_or(DiracError::AllocationOverflow)?;
            let end = offset
                .checked_add(NC)
                .ok_or(DiracError::AllocationOverflow)?;
            let input_site = input.get(offset..end).ok_or(DiracError::StorageInvariant)?;
            for (output_value, input_value) in output_site.iter_mut().zip(input_site) {
                let value = self.mass * *input_value + sign * *output_value;
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(DiracError::NumericalRange);
                }
                *output_value = value;
            }
        }
        Ok(())
    }

    fn apply_hopping_to_data(
        &self,
        output: &mut [Complex64],
        input: &[Complex64],
    ) -> Result<(), DiracError> {
        let expected = NC
            .checked_mul(self.lattice.nv())
            .ok_or(DiracError::AllocationOverflow)?;
        if output.len() != expected || input.len() != expected {
            return Err(DiracError::StorageInvariant);
        }

        for site in 0..self.lattice.nv() {
            let coordinates = coords_from_site_index(site, self.lattice)?;
            let mut hopping = [C0; NC];
            for direction in 0..4 {
                let (plus_site, plus_boundary_sign) = self.neighbor(site, direction, 1)?;
                let (minus_site, minus_boundary_sign) = self.neighbor(site, direction, -1)?;
                let minus_coordinates = coords_from_site_index(minus_site, self.lattice)?;
                let eta_plus = staggered_eta(direction, coordinates);
                let eta_minus = staggered_eta(direction, minus_coordinates);
                let forward = self.links.link(direction, site)?.scaled(Complex64::new(
                    f64::from(eta_plus * plus_boundary_sign),
                    0.0,
                ));
                let backward =
                    self.links
                        .link(direction, minus_site)?
                        .adjoint()
                        .scaled(Complex64::new(
                            f64::from(eta_minus * minus_boundary_sign),
                            0.0,
                        ));
                let plus_values = multiply_color(forward, read_color(input, plus_site)?);
                let minus_values = multiply_color(backward, read_color(input, minus_site)?);
                for color in 0..NC {
                    hopping[color] += C_HALF * (plus_values[color] - minus_values[color]);
                }
            }
            for (color, value) in hopping.into_iter().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(DiracError::NumericalRange);
                }
                let offset = site
                    .checked_mul(NC)
                    .and_then(|offset| offset.checked_add(color))
                    .ok_or(DiracError::AllocationOverflow)?;
                *output.get_mut(offset).ok_or(DiracError::StorageInvariant)? = value;
            }
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
        if input.components() != 1 {
            return Err(DiracError::ComponentsMismatch {
                operand: "input",
                expected: 1,
                found: input.components(),
            });
        }
        if output.components() != 1 {
            return Err(DiracError::ComponentsMismatch {
                operand: "output",
                expected: 1,
                found: output.components(),
            });
        }
        let expected_len = self
            .lattice
            .nv()
            .checked_mul(NC)
            .ok_or(DiracError::AllocationOverflow)?;
        if input.host_data()?.len() != expected_len || output.host_data()?.len() != expected_len {
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
        if workspace.components() != 1 {
            return Err(DiracError::ComponentsMismatch {
                operand: operation,
                expected: 1,
                found: workspace.components(),
            });
        }
        let expected_len = self
            .lattice
            .nv()
            .checked_mul(NC)
            .ok_or(DiracError::AllocationOverflow)?;
        if workspace.host_data()?.len() != expected_len {
            return Err(DiracError::StorageInvariant);
        }
        Ok(())
    }

    // Julia: `shift_fermion`/`shifted_fermion!` in
    // `StaggeredFermion_4D_nowing.jl` at the pinned revision. Gaugefield
    // periodicity is supplied by `HostGaugeLinks`; this helper adds the
    // fermion sign exactly once for a wrapped one-hop displacement.
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
        let shifted = self.links.shifted_site(site, direction, displacement)?;
        let sign = if wraps {
            self.boundary.sign(direction)?
        } else {
            1
        };
        Ok((shifted, sign))
    }
}

impl FermionOperator for StaggeredDirac<'_> {
    fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }

    fn components(&self) -> usize {
        1
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

/// A borrowed view applying the staggered Hermitian adjoint `D† = mass I - K`.
///
/// # Examples
///
/// ```
/// use dirac_operators::{FermionField, FermionOperator, StaggeredDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let operator = StaggeredDirac::new(&links, 0.17)?;
/// let input = FermionField::zeros(lattice, 1)?;
/// let mut output = FermionField::zeros(lattice, 1)?;
/// operator.adjoint().apply_into(&mut output, &input)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StaggeredAdjoint<'op, 'links> {
    parent: &'op StaggeredDirac<'links>,
}

impl fmt::Debug for StaggeredAdjoint<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaggeredAdjoint")
            .field("lattice", &self.parent.lattice)
            .field("mass", &self.parent.mass)
            .field("boundary", &self.parent.boundary)
            .finish()
    }
}

impl FermionOperator for StaggeredAdjoint<'_, '_> {
    fn lattice(&self) -> LatticeShape4 {
        self.parent.lattice
    }

    fn components(&self) -> usize {
        1
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

/// The composed staggered normal operator `D†D`.
///
/// # Examples
///
/// ```
/// use dirac_operators::{FermionField, FermionOperator, StaggeredDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let operator = StaggeredDirac::new(&links, 0.17)?;
/// let input = FermionField::zeros(lattice, 1)?;
/// let mut output = FermionField::zeros(lattice, 1)?;
/// operator.normal().apply_into(&mut output, &input)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StaggeredNormalOperator<'op, 'links> {
    operator: &'op StaggeredDirac<'links>,
}

impl fmt::Debug for StaggeredNormalOperator<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaggeredNormalOperator")
            .field("lattice", &self.operator.lattice)
            .field("mass", &self.operator.mass)
            .finish()
    }
}

impl FermionOperator for StaggeredNormalOperator<'_, '_> {
    fn lattice(&self) -> LatticeShape4 {
        self.operator.lattice
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        let mut scratch = [
            FermionField::zeros(self.operator.lattice, 1)?,
            FermionField::zeros(self.operator.lattice, 1)?,
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
            .validate_operands(output, input, "StaggeredNormalOperator")?;
        if scratch.len() < 2 {
            return Err(DiracError::StorageInvariant);
        }
        let (first, second) = scratch.split_at_mut(1);
        let temporary = &mut first[0];
        let result = &mut second[0];
        self.operator
            .validate_workspace(temporary, "StaggeredNormalOperator")?;
        self.operator
            .validate_workspace(result, "StaggeredNormalOperator")?;
        self.operator
            .apply_to_data(temporary.host_data_mut()?, input.host_data()?, false)?;
        self.operator
            .apply_to_data(result.host_data_mut()?, temporary.host_data()?, true)?;
        output.copy_from(result)
    }
}

impl HermitianPositiveOperator for StaggeredNormalOperator<'_, '_> {}

/// The independently lowered staggered normal operator `mass² I - K²`.
///
/// # Examples
///
/// ```
/// use dirac_operators::{FermionField, FermionOperator, StaggeredDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let operator = StaggeredDirac::new(&links, 0.17)?;
/// let input = FermionField::zeros(lattice, 1)?;
/// let mut output = FermionField::zeros(lattice, 1)?;
/// operator.normal_closed_form().apply_into(&mut output, &input)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StaggeredClosedNormalOperator<'op, 'links> {
    operator: &'op StaggeredDirac<'links>,
}

impl fmt::Debug for StaggeredClosedNormalOperator<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaggeredClosedNormalOperator")
            .field("lattice", &self.operator.lattice)
            .field("mass", &self.operator.mass)
            .finish()
    }
}

impl FermionOperator for StaggeredClosedNormalOperator<'_, '_> {
    fn lattice(&self) -> LatticeShape4 {
        self.operator.lattice
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        let mut scratch = [
            FermionField::zeros(self.operator.lattice, 1)?,
            FermionField::zeros(self.operator.lattice, 1)?,
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
            .validate_operands(output, input, "StaggeredClosedNormalOperator")?;
        if scratch.len() < 2 {
            return Err(DiracError::StorageInvariant);
        }
        let (first, second) = scratch.split_at_mut(1);
        let temporary = &mut first[0];
        let result = &mut second[0];
        self.operator
            .validate_workspace(temporary, "StaggeredClosedNormalOperator")?;
        self.operator
            .validate_workspace(result, "StaggeredClosedNormalOperator")?;
        self.operator
            .apply_hopping_to_data(temporary.host_data_mut()?, input.host_data()?)?;
        self.operator
            .apply_hopping_to_data(result.host_data_mut()?, temporary.host_data()?)?;
        let input_data = input.host_data()?;
        let result_data = result.host_data_mut()?;
        let mass_squared = self.operator.mass * self.operator.mass;
        if !mass_squared.is_finite() {
            return Err(DiracError::NumericalRange);
        }
        for (result_value, input_value) in result_data.iter_mut().zip(input_data) {
            let value = mass_squared * *input_value - *result_value;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(DiracError::NumericalRange);
            }
            *result_value = value;
        }
        output.copy_from(result)
    }
}

impl HermitianPositiveOperator for StaggeredClosedNormalOperator<'_, '_> {}

fn validate_mass(mass: f64) -> Result<(), DiracError> {
    if !mass.is_finite() {
        return Err(DiracError::NonFiniteMass { found: mass });
    }
    if mass <= 0.0 {
        return Err(DiracError::NonPositiveMass { found: mass });
    }
    Ok(())
}

pub(crate) fn staggered_eta(direction: usize, coordinates: [usize; 4]) -> i8 {
    let count = match direction {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        _ => return 1,
    };
    let odd = coordinates[..count]
        .iter()
        .fold(false, |odd, coordinate| odd ^ (coordinate % 2 != 0));
    if odd {
        -1
    } else {
        1
    }
}

fn read_color(values: &[Complex64], site: usize) -> Result<[Complex64; NC], DiracError> {
    let offset = site.checked_mul(NC).ok_or(DiracError::AllocationOverflow)?;
    let end = offset
        .checked_add(NC)
        .ok_or(DiracError::AllocationOverflow)?;
    let block = values
        .get(offset..end)
        .ok_or(DiracError::StorageInvariant)?;
    block.try_into().map_err(|_| DiracError::StorageInvariant)
}

fn multiply_color(matrix: Mat3, input: [Complex64; NC]) -> [Complex64; NC] {
    [
        matrix[(0, 0)] * input[0] + matrix[(0, 1)] * input[1] + matrix[(0, 2)] * input[2],
        matrix[(1, 0)] * input[0] + matrix[(1, 1)] * input[1] + matrix[(1, 2)] * input[2],
        matrix[(2, 0)] * input[0] + matrix[(2, 1)] * input[1] + matrix[(2, 2)] * input[2],
    ]
}
