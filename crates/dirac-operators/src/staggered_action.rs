//! Two-flavor staggered pseudofermions and the Julia-parallel force.
//!
//! This module follows the pinned LatticeDiracOperators.jl v0.6.4
//! `src/action/StaggeredFermiAction.jl` at revision
//! `bdef628184597815ba3e0cddf2536df767e78a02`: `sample_pseudofermions!`,
//! `evaluate_FermiAction`, `calc_UdSfdU!`, and
//! `calc_UdSfdU_fromX!`.  The pinned `R(M)` partial fractions are owned by
//! [`crate::rhmc`].  Rust replaces Julia's global temporary fields with
//! solve-local preallocated shifted outputs, validates every true residual,
//! and commits no caller-owned field during a failed operation.
//!
//! The accepted API is Nf=2 only:
//!
//! ```text
//! phi = R_(+1/8)(M) xi
//! S_f = ||R_(-1/8)(M) phi||^2
//! X_j = (M + beta_j I)^-1 phi
//! Y_j = D X_j
//! ```
//!
//! `xi` uses one uncached Box--Muller pair for each complex component and
//! multiplies both independent real and imaginary normals by `1/sqrt(2)`.

use crate::rhmc::{
    apply_action_inverse, apply_refresh, solve_md_force_shifts, ClaimedSpectralBounds,
};
use crate::{
    DiracError, FermionBoundary, FermionField, FermionOperator, MultiShiftSolverReport,
    SolverParams, StaggeredDirac,
};
use gaugefields::{coords_from_site_index, GaugeLinks, Mat3, ReproducibleRng, TaGaugeField};
use num_complex::Complex64;
use std::fmt;
use tenferro_tensor::TypedTensor;

const NC: usize = 3;
const INV_SQRT_TWO: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Validated two-flavor staggered pseudofermion parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaggeredFermiAction {
    mass: f64,
    boundary: FermionBoundary,
    spectral_bounds: ClaimedSpectralBounds,
    solver_params: SolverParams,
}

impl StaggeredFermiAction {
    /// Construct the Nf=2 staggered RHMC action.
    ///
    /// `lambda_low` and `lambda_high` are caller assertions that every
    /// eigenvalue of `M=D†D` used by this action lies in the supplied interval.
    /// The assertion must be finite, positive, ordered, and contained in the
    /// pinned coefficient interval `[0.0004, 64]`; this constructor does not
    /// estimate, widen, or clamp it.
    ///
    /// # Errors
    ///
    /// Returns [`DiracError::NonFiniteMass`],
    /// [`DiracError::NonPositiveMass`], a typed spectral-bound validation error,
    /// or a solver-parameter error.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::{FermionBoundary, SolverParams, StaggeredFermiAction};
    ///
    /// let action = StaggeredFermiAction::new(
    ///     0.17,
    ///     FermionBoundary::new([1, 1, 1, -1])?,
    ///     0.0004,
    ///     64.0,
    ///     SolverParams::new(1.0e-24, 2_000)?,
    /// )?;
    /// assert_eq!(action.mass(), 0.17);
    /// # Ok::<(), dirac_operators::DiracError>(())
    /// ```
    pub fn new(
        mass: f64,
        boundary: FermionBoundary,
        lambda_low: f64,
        lambda_high: f64,
        solver_params: SolverParams,
    ) -> Result<Self, DiracError> {
        validate_mass(mass)?;
        Ok(Self {
            mass,
            boundary,
            spectral_bounds: ClaimedSpectralBounds::new(lambda_low, lambda_high)?,
            solver_params,
        })
    }

    /// Return the validated positive staggered mass.
    pub const fn mass(self) -> f64 {
        self.mass
    }

    /// Return the validated fermion boundary signs.
    pub const fn boundary(self) -> FermionBoundary {
        self.boundary
    }

    /// Return the caller-claimed lower eigenvalue bound.
    pub const fn lambda_low(self) -> f64 {
        self.spectral_bounds.lower()
    }

    /// Return the caller-claimed upper eigenvalue bound.
    pub const fn lambda_high(self) -> f64 {
        self.spectral_bounds.upper()
    }

    /// Return the shifted-solver parameters.
    pub const fn solver_params(self) -> SolverParams {
        self.solver_params
    }

    /// Fill one complex Gaussian one-component field from the caller RNG.
    ///
    /// Each real and imaginary standard normal is independently multiplied by
    /// `1/sqrt(2)`, matching the two-flavor complex Gaussian convention used by
    /// the Wilson action. Values are consumed in `[color, site]` storage order.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, tensor, or numerical-range error. The RNG is
    /// advanced for every pair drawn before any later field validation failure.
    pub fn sample_xi(
        &self,
        lattice: gaugefields::LatticeShape4,
        rng: &mut ReproducibleRng,
    ) -> Result<FermionField, DiracError> {
        let count = NC
            .checked_mul(lattice.nv())
            .ok_or(DiracError::AllocationOverflow)?;
        let bytes = count
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(DiracError::AllocationOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(DiracError::AllocationOverflow);
        }
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            let [real, imag] = rng.standard_normal_pair();
            let value = Complex64::new(INV_SQRT_TWO * real, INV_SQRT_TWO * imag);
            if !complex_is_finite(value) {
                return Err(DiracError::NumericalRange);
            }
            values.push(value);
        }
        FermionField::from_vec_col_major(lattice, 1, values)
    }

    /// Refresh `phi = R_(+1/8)(M) xi` for the two-flavor action.
    ///
    /// # Errors
    ///
    /// Returns typed gauge/operator, allocation, tensor, shifted-solver, true
    /// residual, or numerical-range errors. The supplied links and RNG-owned
    /// state are not rolled back; already consumed RNG words remain consumed.
    pub fn sample_pseudofermion(
        &self,
        links: &GaugeLinks,
        rng: &mut ReproducibleRng,
    ) -> Result<FermionField, DiracError> {
        let _ = StaggeredDirac::with_boundary(links, self.mass, self.boundary)?;
        let xi = self.sample_xi(links.lattice(), rng)?;
        self.refresh_pseudofermion(links, &xi)
    }

    /// Apply the pinned refresh rational to an explicit Gaussian field.
    ///
    /// This is the deterministic counterpart of [`Self::sample_pseudofermion`]
    /// and is useful when the caller owns a reproducible pseudofermion source.
    /// It computes `phi = R_(+1/8)(D†D) xi`.
    ///
    /// # Errors
    ///
    /// Returns typed gauge/operator, allocation, tensor, shifted-solver, true
    /// residual, or numerical-range errors. Neither input is changed.
    pub fn refresh_pseudofermion(
        &self,
        links: &GaugeLinks,
        xi: &FermionField,
    ) -> Result<FermionField, DiracError> {
        let dirac = StaggeredDirac::with_boundary(links, self.mass, self.boundary)?;
        validate_field(&dirac, xi, "xi")?;
        Ok(apply_refresh(&dirac.normal(), xi, self.solver_params)?.field)
    }

    /// Evaluate `||R_(-1/8)(M) phi||²` and return the transformed field `X`.
    ///
    /// The result includes every shifted solve report, including its freshly
    /// recomputed true residual.
    ///
    /// # Errors
    ///
    /// Returns typed gauge/operator, allocation, tensor, shifted-solver, true
    /// residual, or numerical-range errors. `phi` and `links` are unchanged.
    pub fn evaluate(
        &self,
        links: &GaugeLinks,
        phi: &FermionField,
    ) -> Result<StaggeredActionResult, DiracError> {
        let dirac = StaggeredDirac::with_boundary(links, self.mass, self.boundary)?;
        if phi.lattice() != dirac.lattice() {
            return Err(DiracError::LatticeMismatch {
                operand: "phi",
                expected: dirac.lattice(),
                found: phi.lattice(),
            });
        }
        validate_field(&dirac, phi, "phi")?;
        let transformed = apply_action_inverse(&dirac.normal(), phi, self.solver_params)?;
        let action = transformed.field.norm_squared()?;
        if !action.is_finite() {
            return Err(DiracError::NumericalRange);
        }
        Ok(StaggeredActionResult {
            action,
            x: transformed.field,
            solver_reports: transformed.reports,
        })
    }

    /// Evaluate the degree-10 MD force from all inverse residues.
    ///
    /// For every residue this retains Julia's `X=(M+beta I)^-1 phi` and
    /// `Y=D X` names. The two outer products are accumulated in one raw `Mat3`
    /// per link and projected once with `Mat3::add_ta_coefficients`.
    ///
    /// # Errors
    ///
    /// Returns typed gauge/operator, allocation, tensor, shifted-solver, true
    /// residual, or numerical-range errors. Neither input is changed.
    pub fn force(
        &self,
        links: &GaugeLinks,
        phi: &FermionField,
    ) -> Result<StaggeredForceResult, DiracError> {
        let dirac = StaggeredDirac::with_boundary(links, self.mass, self.boundary)?;
        if phi.lattice() != dirac.lattice() {
            return Err(DiracError::LatticeMismatch {
                operand: "phi",
                expected: dirac.lattice(),
                found: phi.lattice(),
            });
        }
        validate_field(&dirac, phi, "phi")?;
        let (x, solver_reports) = solve_md_force_shifts(&dirac.normal(), phi, self.solver_params)?;
        let mut y = Vec::with_capacity(x.len());
        for shifted_x in &x {
            let mut shifted_y = FermionField::zeros(dirac.lattice(), 1)?;
            dirac.apply_into(&mut shifted_y, shifted_x)?;
            y.push(shifted_y);
        }
        let force = force_from_shifted_xy(&dirac, &x, &y)?;
        Ok(StaggeredForceResult {
            force,
            x,
            y,
            solver_reports,
        })
    }
}

/// Result of one two-flavor staggered rational action evaluation.
pub struct StaggeredActionResult {
    /// `||X||²`, where `X=R_(-1/8)(M)phi`.
    pub action: f64,
    /// Transformed action field `X=R_(-1/8)(M)phi`.
    pub x: FermionField,
    /// True-residual reports for the degree-15 inverse shifts.
    pub solver_reports: Vec<MultiShiftSolverReport>,
}

impl fmt::Debug for StaggeredActionResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaggeredActionResult")
            .field("action", &self.action)
            .field("x", &self.x)
            .field("solver_reports", &self.solver_reports)
            .finish()
    }
}

/// Result of one degree-10 staggered MD-force evaluation.
pub struct StaggeredForceResult {
    /// TA coefficient field for the staggered pseudofermion force.
    pub force: TaGaugeField,
    /// Shifted solutions `X_j=(M+beta_j I)^-1 phi` in pinned beta order.
    pub x: Vec<FermionField>,
    /// `Y_j=D X_j` in the same pinned shift order.
    pub y: Vec<FermionField>,
    /// True-residual reports for all degree-10 inverse shifts.
    pub solver_reports: Vec<MultiShiftSolverReport>,
}

impl fmt::Debug for StaggeredForceResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StaggeredForceResult")
            .field("force", &self.force)
            .field("x_count", &self.x.len())
            .field("y_count", &self.y.len())
            .field("solver_reports", &self.solver_reports)
            .finish()
    }
}

fn validate_field(
    dirac: &StaggeredDirac<'_>,
    field: &FermionField,
    operand: &'static str,
) -> Result<(), DiracError> {
    if field.lattice() != dirac.lattice() {
        return Err(DiracError::LatticeMismatch {
            operand,
            expected: dirac.lattice(),
            found: field.lattice(),
        });
    }
    if field.components() != 1 {
        return Err(DiracError::ComponentsMismatch {
            operand,
            expected: 1,
            found: field.components(),
        });
    }
    field.ensure_finite()
}

fn force_from_shifted_xy(
    dirac: &StaggeredDirac<'_>,
    x: &[FermionField],
    y: &[FermionField],
) -> Result<TaGaugeField, DiracError> {
    if x.len() != y.len() {
        return Err(DiracError::StorageInvariant);
    }
    let coefficients = crate::rhmc::md_force_coefficients();
    if x.len() != coefficients.len() {
        return Err(DiracError::StorageInvariant);
    }
    for field in x.iter().chain(y) {
        if field.lattice() != dirac.lattice() {
            return Err(DiracError::LatticeMismatch {
                operand: "staggered force field",
                expected: dirac.lattice(),
                found: field.lattice(),
            });
        }
        if field.components() != 1 {
            return Err(DiracError::ComponentsMismatch {
                operand: "staggered force field",
                expected: 1,
                found: field.components(),
            });
        }
        field.ensure_finite()?;
    }

    let count = 8usize
        .checked_mul(dirac.lattice().nv())
        .ok_or(DiracError::AllocationOverflow)?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(DiracError::AllocationOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(DiracError::AllocationOverflow);
    }
    let [nx, ny, nz, nt] = dirac.lattice().extents();
    let links = dirac.host_links();
    let mut tensors = Vec::with_capacity(4);
    for direction in 0..4 {
        let mut data = Vec::with_capacity(count);
        for site in 0..dirac.lattice().nv() {
            let coordinates = coords_from_site_index(site, dirac.lattice())?;
            let (plus_site, boundary_sign) = dirac.neighbor(site, direction, 1)?;
            let eta = f64::from(crate::staggered::staggered_eta(direction, coordinates));
            let us = links
                .link(direction, site)?
                .scaled(Complex64::new(eta, 0.0));
            let mut raw = Mat3::zero();
            for (coefficient, (x_field, y_field)) in coefficients.iter().zip(x.iter().zip(y)) {
                let x_site = load_color(x_field.host_data()?, site)?;
                let x_plus = load_color(x_field.host_data()?, plus_site)?;
                let y_site = load_color(y_field.host_data()?, site)?;
                let y_plus = load_color(y_field.host_data()?, plus_site)?;
                let left = multiply_color(us, x_plus, boundary_sign);
                let projected_y = multiply_color(us, y_plus, boundary_sign);
                let factor = Complex64::new(0.5 * *coefficient, 0.0);
                add_outer_left_right(&mut raw, left, y_site, factor);
                add_outer_right_row(&mut raw, x_site, projected_y, factor);
            }

            let mut coefficient_values = [0.0; 8];
            Mat3::add_ta_coefficients(&mut coefficient_values, 1.0, raw);
            if coefficient_values.iter().any(|value| !value.is_finite()) {
                return Err(DiracError::NumericalRange);
            }
            data.extend_from_slice(&coefficient_values);
        }
        tensors.push(
            TypedTensor::from_vec_col_major(vec![8, nx, ny, nz, nt], data)
                .map_err(|error| DiracError::Tensor(error.to_string()))?,
        );
    }
    TaGaugeField::new(
        tensors
            .try_into()
            .map_err(|_| DiracError::Tensor("staggered force requires four tensors".into()))?,
        dirac.lattice(),
    )
    .map_err(DiracError::from)
}

fn load_color(values: &[Complex64], site: usize) -> Result<[Complex64; NC], DiracError> {
    let offset = site.checked_mul(NC).ok_or(DiracError::AllocationOverflow)?;
    let end = offset
        .checked_add(NC)
        .ok_or(DiracError::AllocationOverflow)?;
    values
        .get(offset..end)
        .ok_or(DiracError::StorageInvariant)?
        .try_into()
        .map_err(|_| DiracError::StorageInvariant)
}

fn multiply_color(matrix: Mat3, input: [Complex64; NC], boundary_sign: i8) -> [Complex64; NC] {
    let sign = Complex64::new(f64::from(boundary_sign), 0.0);
    [
        sign * (matrix[(0, 0)] * input[0] + matrix[(0, 1)] * input[1] + matrix[(0, 2)] * input[2]),
        sign * (matrix[(1, 0)] * input[0] + matrix[(1, 1)] * input[1] + matrix[(1, 2)] * input[2]),
        sign * (matrix[(2, 0)] * input[0] + matrix[(2, 1)] * input[1] + matrix[(2, 2)] * input[2]),
    ]
}

fn add_outer_left_right(
    output: &mut Mat3,
    left: [Complex64; NC],
    right: [Complex64; NC],
    factor: Complex64,
) {
    for row in 0..NC {
        for column in 0..NC {
            output[(row, column)] += factor * left[row] * right[column].conj();
        }
    }
}

fn add_outer_right_row(
    output: &mut Mat3,
    left: [Complex64; NC],
    projected_right: [Complex64; NC],
    factor: Complex64,
) {
    for row in 0..NC {
        for column in 0..NC {
            output[(row, column)] += factor * left[row] * projected_right[column].conj();
        }
    }
}

fn validate_mass(mass: f64) -> Result<(), DiracError> {
    if !mass.is_finite() {
        return Err(DiracError::NonFiniteMass { found: mass });
    }
    if mass <= 0.0 {
        return Err(DiracError::NonPositiveMass { found: mass });
    }
    Ok(())
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
