//! Two-degenerate-flavor Wilson pseudofermions.
//!
//! The decomposition follows the pinned LatticeDiracOperators.jl v0.6.4
//! `src/action/WilsonFermiAction.jl` at revision
//! `bdef628184597815ba3e0cddf2536df767e78a02`: `evaluate_FermiAction`
//! (lines 86--97), `calc_UdSfdU!` (99--136),
//! `calc_UdSfdU_fromX!` (138--234), and `sample_pseudofermions!`
//! (362--377). The Julia temporary pools, global Gaussian stream, assertions,
//! and clover path are intentionally not copied. Gauge matrices, periodic
//! access, and TA projection remain owned by `gaugefields`, following the
//! pinned Gaugefields.jl v0.7.2
//! `src/4D/TA_gaugefields_4D_serial.jl` at revision
//! `9e5719970770f4497405a856315c90bef7f74449`.

use crate::wilson::{
    project_color_spin, validate_kappa, ColorSpinor, FermionOperator, WilsonDirac,
};
use crate::{
    conjugate_gradient, DiracError, FermionBoundary, FermionField, SolverParams, SolverReport,
};
use gaugefields::{GaugeLinks, LatticeShape4, Mat3, ReproducibleRng, TaGaugeField};
use num_complex::Complex64;
use std::fmt;
use tenferro_tensor::TypedTensor;

const INV_SQRT_TWO: f64 = std::f64::consts::FRAC_1_SQRT_2;
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const NC: usize = 3;
const SPIN: usize = 4;

/// Validated two-flavor Wilson pseudofermion parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WilsonFermiAction {
    kappa: f64,
    boundary: FermionBoundary,
    solver_params: SolverParams,
}

impl WilsonFermiAction {
    /// Construct an action with explicit hopping, boundary, and solver values.
    ///
    /// # Errors
    ///
    /// Returns the existing typed kappa or solver-parameter error. Boundary
    /// signs are validated by [`FermionBoundary::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::{FermionBoundary, SolverParams, WilsonFermiAction};
    ///
    /// let action = WilsonFermiAction::new(
    ///     0.1,
    ///     FermionBoundary::new([1, 1, 1, -1])?,
    ///     SolverParams::new(1.0e-20, 256)?,
    ///     )?;
    /// assert_eq!(action.kappa(), 0.1);
    /// # Ok::<(), dirac_operators::DiracError>(())
    /// ```
    pub fn new(
        kappa: f64,
        boundary: FermionBoundary,
        solver_params: SolverParams,
    ) -> Result<Self, DiracError> {
        validate_kappa(kappa)?;
        Ok(Self {
            kappa,
            boundary,
            solver_params,
        })
    }

    /// Return the hopping parameter.
    pub const fn kappa(self) -> f64 {
        self.kappa
    }

    /// Return the validated fermion boundary signs.
    pub const fn boundary(self) -> FermionBoundary {
        self.boundary
    }

    /// Return the explicit solver parameters used for every normal solve.
    pub const fn solver_params(self) -> SolverParams {
        self.solver_params
    }

    /// Fill one Gaussian Wilson field from the caller-owned RNG.
    ///
    /// Independent real and imaginary standard normals are scaled by
    /// `1/sqrt(2)`, so each complex component has unit variance. Values are
    /// consumed in Rust field storage order `[color, spin, site]`.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, tensor, or numerical-range error.
    pub fn sample_xi(
        &self,
        lattice: LatticeShape4,
        rng: &mut ReproducibleRng,
    ) -> Result<FermionField, DiracError> {
        let count = NC
            .checked_mul(SPIN)
            .and_then(|value| value.checked_mul(lattice.nv()))
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
        FermionField::from_vec_col_major(lattice, SPIN, values)
    }

    /// Refresh a two-flavor pseudofermion as `phi = D† xi`.
    ///
    /// # Errors
    ///
    /// Returns a typed gauge/operator, allocation, tensor, or numerical-range
    /// error. The RNG is advanced for the Gaussian field even if the operator
    /// output subsequently fails.
    pub fn sample_pseudofermion(
        &self,
        links: &GaugeLinks,
        rng: &mut ReproducibleRng,
    ) -> Result<FermionField, DiracError> {
        let dirac = WilsonDirac::with_boundary(links, self.kappa, self.boundary)?;
        let xi = self.sample_xi(links.lattice(), rng)?;
        let mut phi = FermionField::zeros(links.lattice(), SPIN)?;
        dirac.adjoint().apply_into(&mut phi, &xi)?;
        Ok(phi)
    }

    /// Evaluate `phi† (D†D)^-1 phi` and retain its normal solution.
    ///
    /// # Errors
    ///
    /// Returns typed field/operator errors or the checked solver's
    /// non-finite, breakdown, stagnation, exhaustion, and true-residual errors.
    /// The supplied pseudofermion is never changed.
    pub fn evaluate(
        &self,
        links: &GaugeLinks,
        phi: &FermionField,
    ) -> Result<WilsonActionResult, DiracError> {
        let (x, solver_report, action) = self.solve(links, phi)?;
        Ok(WilsonActionResult {
            action,
            x,
            solver_report,
        })
    }

    /// Evaluate the action and its analytic TA force.
    ///
    /// The force retains the Julia intermediate names `X=(D†D)^-1 phi` and
    /// `Y=D X`, and projects the two outer products once into the existing
    /// [`TaGaugeField`] convention.
    ///
    /// # Errors
    ///
    /// Returns typed gauge/operator errors or checked solver and numerical-range
    /// failures. Neither input field is changed.
    pub fn force(
        &self,
        links: &GaugeLinks,
        phi: &FermionField,
    ) -> Result<WilsonForceResult, DiracError> {
        let dirac = WilsonDirac::with_boundary(links, self.kappa, self.boundary)?;
        let (x, solver_report, action) = self.solve_with_operator(&dirac, phi)?;
        let mut y = FermionField::zeros(links.lattice(), SPIN)?;
        dirac.apply_into(&mut y, &x)?;
        let force = force_from_x(&dirac, &x, &y)?;
        Ok(WilsonForceResult {
            action,
            force,
            x,
            y,
            solver_report,
        })
    }

    fn solve(
        &self,
        links: &GaugeLinks,
        phi: &FermionField,
    ) -> Result<(FermionField, SolverReport, f64), DiracError> {
        let dirac = WilsonDirac::with_boundary(links, self.kappa, self.boundary)?;
        self.solve_with_operator(&dirac, phi)
    }

    fn solve_with_operator(
        &self,
        dirac: &WilsonDirac<'_>,
        phi: &FermionField,
    ) -> Result<(FermionField, SolverReport, f64), DiracError> {
        let mut x = FermionField::zeros(dirac.lattice(), SPIN)?;
        let report = conjugate_gradient(&mut x, &dirac.normal(), phi, self.solver_params)?;
        let value = phi.inner_product(&x)?;
        if !value.re.is_finite() || !value.im.is_finite() || value.re < 0.0 {
            return Err(DiracError::NumericalRange);
        }
        Ok((x, report, value.re))
    }
}

/// Typed result of one Wilson pseudofermion action evaluation.
pub struct WilsonActionResult {
    /// Real action value `phi† (D†D)^-1 phi`.
    pub action: f64,
    /// Normal-system solution `X=(D†D)^-1 phi`.
    pub x: FermionField,
    /// Checked CG diagnostics for the normal solve.
    pub solver_report: SolverReport,
}

impl fmt::Debug for WilsonActionResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WilsonActionResult")
            .field("action", &self.action)
            .field("x", &self.x)
            .field("solver_report", &self.solver_report)
            .finish()
    }
}

/// Typed result of one Wilson pseudofermion force evaluation.
pub struct WilsonForceResult {
    /// Real pseudofermion action evaluated with the same `X`.
    pub action: f64,
    /// Existing TA coefficient field for the fermion force.
    pub force: TaGaugeField,
    /// `X=(D†D)^-1 phi`.
    pub x: FermionField,
    /// `Y=D X`.
    pub y: FermionField,
    /// Checked CG diagnostics for the normal solve.
    pub solver_report: SolverReport,
}

impl fmt::Debug for WilsonForceResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WilsonForceResult")
            .field("action", &self.action)
            .field("force", &self.force)
            .field("x", &self.x)
            .field("y", &self.y)
            .field("solver_report", &self.solver_report)
            .finish()
    }
}

/// Port of `calc_UdSfdU_fromX!`'s outer-product force assembly.
fn force_from_x(
    dirac: &WilsonDirac<'_>,
    x: &FermionField,
    y: &FermionField,
) -> Result<TaGaugeField, DiracError> {
    if x.lattice() != dirac.lattice() {
        return Err(DiracError::LatticeMismatch {
            operand: "X",
            expected: dirac.lattice(),
            found: x.lattice(),
        });
    }
    if y.lattice() != dirac.lattice() {
        return Err(DiracError::LatticeMismatch {
            operand: "Y",
            expected: dirac.lattice(),
            found: y.lattice(),
        });
    }
    if x.components() != SPIN {
        return Err(DiracError::ComponentsMismatch {
            operand: "X",
            expected: SPIN,
            found: x.components(),
        });
    }
    if y.components() != SPIN {
        return Err(DiracError::ComponentsMismatch {
            operand: "Y",
            expected: SPIN,
            found: y.components(),
        });
    }
    x.ensure_finite()?;
    y.ensure_finite()?;
    let x_data = x.host_data()?;
    let y_data = y.host_data()?;
    let links = dirac.host_links();
    let [nx, ny, nz, nt] = dirac.lattice().extents();
    let count = 8usize
        .checked_mul(dirac.lattice().nv())
        .ok_or(DiracError::AllocationOverflow)?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(DiracError::AllocationOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(DiracError::AllocationOverflow);
    }
    let mut tensors = Vec::with_capacity(4);
    for direction in 0..4 {
        let mut data = Vec::with_capacity(count);
        for site in 0..dirac.lattice().nv() {
            let (plus_site, boundary_sign) = dirac.neighbor(site, direction, 1)?;
            let u = links.link(direction, site)?;
            let x_site = load_color_spinor(x_data, site, dirac.lattice())?;
            let x_plus = load_color_spinor(x_data, plus_site, dirac.lattice())?;
            let y_site = load_color_spinor(y_data, site, dirac.lattice())?;
            let y_plus = load_color_spinor(y_data, plus_site, dirac.lattice())?;

            // With X=(D†D)^(-1)phi and Y=DX, the shared Julia formula is
            // TA[-kappa P_- U X_plus ⊗ Y + kappa X ⊗ Y_plus† U† P_+].
            // The same wrapped-link sign multiplies both terms. There is no
            // gauge 1/NC factor here; Julia applies that factor only in P_update!.
            // Julia: `U*xplus`, `mul_1minusγμx!`, then `temp0_f ⊗ Y'`.
            let left = project_color_spin(u, x_plus, direction, -1);
            // Julia: `Yplus' * U'`, `mul_x1plusγμ!`, then `X ⊗ temp0_f`.
            // P_plus is Hermitian, so this is the conjugate of P_plus*U*Yplus.
            let projected_y = project_color_spin(u, y_plus, direction, 1);
            let mut right = [[ZERO; NC]; SPIN];
            for spin in 0..SPIN {
                for color in 0..NC {
                    right[spin][color] = projected_y[color][spin].conj();
                }
            }

            let sign = f64::from(boundary_sign);
            let mut raw = Mat3::zero();
            add_color_outer(
                &mut raw,
                left,
                y_site,
                Complex64::new(-dirac.kappa() * sign, 0.0),
            );
            add_spin_color_outer(
                &mut raw,
                x_site,
                right,
                Complex64::new(dirac.kappa() * sign, 0.0),
            );

            // `add_ta_coefficients` is the Gaugefields.jl
            // `Traceless_antihermitian_add!` convention for
            // A=(i/2) sum_a coefficients[a] lambda_a.
            let mut coefficients = [0.0; 8];
            Mat3::add_ta_coefficients(&mut coefficients, 1.0, raw);
            if coefficients.iter().any(|value| !value.is_finite()) {
                return Err(DiracError::NumericalRange);
            }
            data.extend_from_slice(&coefficients);
        }
        tensors.push(
            TypedTensor::from_vec_col_major(vec![8, nx, ny, nz, nt], data)
                .map_err(|error| DiracError::Tensor(error.to_string()))?,
        );
    }
    TaGaugeField::new(
        tensors
            .try_into()
            .map_err(|_| DiracError::Tensor("Wilson force requires four tensors".into()))?,
        dirac.lattice(),
    )
    .map_err(DiracError::from)
}

fn load_color_spinor(
    values: &[Complex64],
    site: usize,
    lattice: LatticeShape4,
) -> Result<ColorSpinor, DiracError> {
    let offset = site
        .checked_mul(NC * SPIN)
        .ok_or(DiracError::AllocationOverflow)?;
    if site >= lattice.nv() {
        return Err(DiracError::SiteOutOfBounds {
            site,
            volume: lattice.nv(),
        });
    }
    let block = values
        .get(offset..offset + NC * SPIN)
        .ok_or(DiracError::StorageInvariant)?;
    let mut result = [[ZERO; SPIN]; NC];
    for (color, result_color) in result.iter_mut().enumerate() {
        for (spin, result_value) in result_color.iter_mut().enumerate() {
            *result_value = block[color + NC * spin];
        }
    }
    Ok(result)
}

fn add_color_outer(output: &mut Mat3, left: ColorSpinor, right: ColorSpinor, factor: Complex64) {
    for row in 0..NC {
        for column in 0..NC {
            let mut value = ZERO;
            for spin in 0..SPIN {
                value += left[row][spin] * right[column][spin].conj();
            }
            output[(row, column)] += factor * value;
        }
    }
}

fn add_spin_color_outer(
    output: &mut Mat3,
    left: ColorSpinor,
    right: [[Complex64; NC]; SPIN],
    factor: Complex64,
) {
    for row in 0..NC {
        for column in 0..NC {
            let mut value = ZERO;
            for spin in 0..SPIN {
                value += left[row][spin] * right[spin][column];
            }
            output[(row, column)] += factor * value;
        }
    }
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
