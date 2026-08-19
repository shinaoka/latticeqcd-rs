//! Checked Krylov solvers parallel to the pinned
//! `LatticeDiracOperators.jl/src/cgmethods.jl` at revision
//! `bdef628184597815ba3e0cddf2536df767e78a02`:
//! `Dirac_operators.cg` (lines 768-868) and `Dirac_operators.bicgstab`
//! (lines 157-310), from
//! <https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/cgmethods.jl>.
//!
//! [`conjugate_gradient`] follows the pinned `cg` recurrence and
//! [`bicgstab`] follows the pinned `bicgstab` recurrence, including the
//! shadow-residual restart. Julia's global temporary pool and panic-on-failure
//! paths are deliberately replaced by one solve-local scratch set and typed
//! errors. The caller's initial guess is the output field; it is committed only
//! after a fresh true-residual check succeeds.

use crate::error::SolverError;
use crate::{DiracError, FermionField, FermionOperator, HermitianPositiveOperator};
use num_complex::Complex64;
use std::fmt;

const ONE: Complex64 = Complex64::new(1.0, 0.0);
// NormalOperator needs one intermediate and one final operator workspace.
const OPERATOR_SCRATCH_FIELDS: usize = 2;
const DENOMINATOR_RELATIVE_TOLERANCE: f64 = 64.0 * f64::EPSILON;
const SHADOW_RELATIVE_TOLERANCE: f64 = f64::EPSILON;

/// The solver recurrence used for a successful solve.
///
/// # Examples
///
/// ```
/// use dirac_operators::SolverMethod;
///
/// assert_eq!(SolverMethod::ConjugateGradient.to_string(), "cg");
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverMethod {
    /// The Hermitian-positive conjugate-gradient recurrence.
    ConjugateGradient,
    /// The general-operator BiCGStab recurrence.
    BiCgStab,
}

impl fmt::Display for SolverMethod {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ConjugateGradient => "cg",
            Self::BiCgStab => "bicgstab",
        })
    }
}

/// The point at which a solver's recursive residual crossed its tolerance.
///
/// # Examples
///
/// ```
/// use dirac_operators::ConvergenceBranch;
///
/// assert_eq!(ConvergenceBranch::InitialResidual.to_string(), "initial_residual");
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConvergenceBranch {
    /// The supplied initial guess already met the tolerance.
    InitialResidual,
    /// BiCGStab's intermediate `s` residual met the tolerance.
    IntermediateResidual,
    /// The updated recursive residual met the tolerance.
    UpdatedResidual,
}

impl fmt::Display for ConvergenceBranch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InitialResidual => "initial_residual",
            Self::IntermediateResidual => "intermediate_residual",
            Self::UpdatedResidual => "updated_residual",
        })
    }
}

/// Validated stopping parameters for one Krylov solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverParams {
    tolerance: f64,
    max_iterations: usize,
}

impl SolverParams {
    /// Construct absolute squared-residual stopping parameters.
    ///
    /// # Errors
    ///
    /// Returns [`DiracError::Solver`] with
    /// [`SolverError::InvalidTolerance`](SolverError::InvalidTolerance) when
    /// `tolerance` is non-finite or non-positive, or with
    /// [`SolverError::InvalidMaximumIterations`](SolverError::InvalidMaximumIterations)
    /// when `max_iterations` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::SolverParams;
    ///
    /// let params = SolverParams::new(1.0e-20, 1_000)?;
    /// assert_eq!(params.tolerance(), 1.0e-20);
    /// assert_eq!(params.max_iterations(), 1_000);
    /// # Ok::<(), dirac_operators::DiracError>(())
    /// ```
    pub fn new(tolerance: f64, max_iterations: usize) -> Result<Self, DiracError> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(SolverError::InvalidTolerance.into());
        }
        if max_iterations == 0 {
            return Err(SolverError::InvalidMaximumIterations.into());
        }
        Ok(Self {
            tolerance,
            max_iterations,
        })
    }

    /// Return the absolute squared-residual tolerance.
    pub const fn tolerance(self) -> f64 {
        self.tolerance
    }

    /// Return the positive iteration limit.
    pub const fn max_iterations(self) -> usize {
        self.max_iterations
    }
}

/// Compact diagnostics for a successful solve.
///
/// # Examples
///
/// ```
/// use dirac_operators::{ConvergenceBranch, SolverMethod, SolverReport};
///
/// let report = SolverReport {
///     method: SolverMethod::ConjugateGradient,
///     iterations: 0,
///     recursive_residual_squared: 0.0,
///     initial_residual_squared: 0.0,
///     true_residual_squared: 0.0,
///     tolerance: 1.0e-20,
///     maximum_iterations: 8,
///     restart_count: 0,
///     convergence_branch: ConvergenceBranch::InitialResidual,
/// };
/// assert_eq!(report.true_residual_squared, 0.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverReport {
    /// The recurrence used by the solve.
    pub method: SolverMethod,
    /// Number of completed solver iterations.
    pub iterations: usize,
    /// Residual squared maintained by the recurrence at convergence.
    pub recursive_residual_squared: f64,
    /// Residual squared of the supplied initial guess.
    pub initial_residual_squared: f64,
    /// Freshly recomputed residual squared of the committed output.
    pub true_residual_squared: f64,
    /// Absolute squared-residual stopping tolerance.
    pub tolerance: f64,
    /// Iteration limit supplied to the solve.
    pub maximum_iterations: usize,
    /// Number of BiCGStab shadow-residual restarts.
    pub restart_count: usize,
    /// The recurrence branch that first crossed the tolerance.
    pub convergence_branch: ConvergenceBranch,
}

/// Diagnostics for one solution of a checked shifted-CG solve.
///
/// The `shift` field identifies the equation `(A + shift I)x = b`. The true
/// residual is recomputed with a fresh operator application before the result
/// can be committed.
///
/// # Examples
///
/// ```
/// use dirac_operators::{ConvergenceBranch, MultiShiftSolverReport};
///
/// let report = MultiShiftSolverReport {
///     shift: 0.25,
///     iterations: 4,
///     recursive_residual_squared: 1.0e-28,
///     initial_residual_squared: 1.0,
///     true_residual_squared: 1.0e-29,
///     tolerance: 1.0e-24,
///     maximum_iterations: 64,
///     convergence_branch: ConvergenceBranch::UpdatedResidual,
/// };
/// assert_eq!(report.shift, 0.25);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiShiftSolverReport {
    /// The non-negative shift for this solution.
    pub shift: f64,
    /// Number of completed recurrence iterations for this shift.
    pub iterations: usize,
    /// Residual squared estimated by the shifted recurrence.
    pub recursive_residual_squared: f64,
    /// Residual squared of the supplied initial guess.
    pub initial_residual_squared: f64,
    /// Freshly recomputed residual squared of the committed output.
    pub true_residual_squared: f64,
    /// Absolute squared-residual stopping tolerance.
    pub tolerance: f64,
    /// Iteration limit supplied to the solve.
    pub maximum_iterations: usize,
    /// The recurrence branch that first crossed the tolerance.
    pub convergence_branch: ConvergenceBranch,
}

/// Applies the pinned `shiftedcg` recurrence to all supplied shifts.
///
/// The equations are `(operator + beta[j] I) * output[j] = rhs`. Current output
/// values are checked initial guesses. A zero initial-guess batch uses the
/// Julia-parallel multi-shift recurrence with shared `r`, `p`, `q`, `alpha`,
/// `beta`, `rho_m`, `rho_0`, and `rho_p` updates. Non-zero guesses use checked
/// independent CG recurrences because shifted residuals are not collinear for
/// arbitrary per-shift guesses; all scratch is still allocated before the
/// iteration loops. Shift order is preserved and is not required to be sorted.
///
/// # Errors
///
/// Returns [`DiracError::Solver`] with [`SolverError::ShiftCountMismatch`] or
/// [`SolverError::InvalidShift`] for invalid shift input, or with typed
/// non-finite, breakdown, stagnation, exhaustion, or true-residual failures.
/// Field/operator mismatches are returned directly. All output fields remain
/// unchanged on every error.
///
/// # Examples
///
/// ```
/// use dirac_operators::{multi_shift_cg, FermionField, SolverParams, WilsonDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let dirac = WilsonDirac::new(&links, 0.1)?;
/// let normal = dirac.normal();
/// let rhs = FermionField::zeros(lattice, 4)?;
/// let shifts = [0.0, 0.2];
/// let mut solutions = (0..shifts.len())
///     .map(|_| FermionField::zeros(lattice, 4))
///     .collect::<Result<Vec<_>, _>>()?;
/// let reports = multi_shift_cg(
///     &mut solutions,
///     &normal,
///     &rhs,
///     &shifts,
///     SolverParams::new(1.0e-20, 64)?,
/// )?;
/// assert_eq!(reports.len(), shifts.len());
/// assert!(reports.iter().all(|report| report.iterations == 0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn multi_shift_cg<O: HermitianPositiveOperator>(
    outputs: &mut [FermionField],
    operator: &O,
    rhs: &FermionField,
    shifts: &[f64],
    params: SolverParams,
) -> Result<Vec<MultiShiftSolverReport>, DiracError> {
    validate_multi_shift_inputs(outputs, operator, rhs, shifts)?;
    let mut all_zero = true;
    for output in outputs.iter() {
        if output.norm_squared()? != 0.0 {
            all_zero = false;
            break;
        }
    }
    let mut work = Vec::with_capacity(outputs.len());
    for output in outputs.iter() {
        work.push(output.try_clone()?);
    }
    let reports = if all_zero {
        solve_zero_shifted(&mut work, operator, rhs, shifts, params)?
    } else {
        solve_nonzero_shifted(&mut work, operator, rhs, shifts, params)?
    };
    for (output, result) in outputs.iter_mut().zip(work.iter()) {
        output.copy_from(result).map_err(map_numeric_error)?;
    }
    Ok(reports)
}

/// Applies the pinned `cg` recurrence to a Hermitian-positive operator.
///
/// `output` is both the initial guess and the transactional destination. It is
/// changed only after the result's freshly recomputed true residual is below
/// `params.tolerance`.
///
/// # Errors
///
/// Returns field/operator mismatch errors, allocation or tensor errors, or
/// [`DiracError::Solver`] with a typed non-finite, breakdown, stagnation,
/// exhaustion, or true-residual failure. The output remains unchanged on every
/// error. The operator must implement [`HermitianPositiveOperator`].
///
/// # Examples
///
/// ```
/// use dirac_operators::{conjugate_gradient, FermionField, SolverParams, WilsonDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let dirac = WilsonDirac::new(&links, 0.1)?;
/// let normal = dirac.normal();
/// let rhs = FermionField::zeros(lattice, 4)?;
/// let mut solution = FermionField::zeros(lattice, 4)?;
/// let report = conjugate_gradient(
///     &mut solution,
///     &normal,
///     &rhs,
///     SolverParams::new(1.0e-20, 8)?,
/// )?;
/// assert_eq!(report.iterations, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn conjugate_gradient<O: HermitianPositiveOperator>(
    output: &mut FermionField,
    operator: &O,
    rhs: &FermionField,
    params: SolverParams,
) -> Result<SolverReport, DiracError> {
    validate_solver_inputs(output, operator, rhs)?;

    let mut work = output.try_clone()?;
    let mut r = rhs.try_clone()?;
    let mut temp1 = scratch_field(operator)?;
    let mut operator_scratch = scratch_fields(operator, OPERATOR_SCRATCH_FIELDS)?;
    let mut q = scratch_field(operator)?;
    let mut p = scratch_field(operator)?;

    // Julia `cg`: res = b - A*x, with the same named intermediates.
    checked_copy(&mut r, rhs)?;
    checked_apply(operator, &mut temp1, &work, &mut operator_scratch)?;
    checked_add_scaled(&mut r, -ONE, &temp1)?;
    let initial_residual_squared = checked_norm_squared(&r)?;
    if initial_residual_squared < params.tolerance {
        return finish(
            output,
            &mut work,
            operator,
            rhs,
            FinishScratch {
                temp1: &mut temp1,
                residual: &mut r,
                operator_scratch: &mut operator_scratch,
            },
            FinishState {
                method: SolverMethod::ConjugateGradient,
                iterations: 0,
                initial_residual_squared,
                recursive_residual_squared: initial_residual_squared,
                params,
                restart_count: 0,
                convergence_branch: ConvergenceBranch::InitialResidual,
            },
        );
    }

    checked_copy(&mut p, &r)?;
    let mut c1 = checked_dot(&p, &p)?;
    require_positive(c1)?;
    let mut previous_residual_squared = initial_residual_squared;

    for iterations in 1..=params.max_iterations {
        // Julia `cg`: mul!(q, A, p); c2 = dot(p, q).
        checked_apply(operator, &mut q, &p, &mut operator_scratch)?;
        let c2 = checked_dot(&p, &q)?;
        let q_norm_squared = checked_norm_squared(&q)?;
        require_positive(c2)?;
        let alpha_scale = product_scale(c1.re, q_norm_squared)?;
        let alpha = checked_division(c1, c2, alpha_scale)?;

        // Julia `cg`: x += alpha*p; res -= alpha*q.
        checked_add_scaled(&mut work, alpha, &p)?;
        checked_add_scaled(&mut r, -alpha, &q)?;
        let c3 = checked_dot(&r, &r)?;
        let recursive_residual_squared = require_residual(c3)?;

        if recursive_residual_squared < params.tolerance {
            return finish(
                output,
                &mut work,
                operator,
                rhs,
                FinishScratch {
                    temp1: &mut temp1,
                    residual: &mut r,
                    operator_scratch: &mut operator_scratch,
                },
                FinishState {
                    method: SolverMethod::ConjugateGradient,
                    iterations,
                    initial_residual_squared,
                    recursive_residual_squared,
                    params,
                    restart_count: 0,
                    convergence_branch: ConvergenceBranch::UpdatedResidual,
                },
            );
        }
        if is_stagnant(previous_residual_squared, recursive_residual_squared) {
            return Err(SolverError::Stagnation.into());
        }

        // Julia `cg`: beta = c3/c1; c1 = c3; p = beta*p + res.
        let beta = checked_division(c3, c1, c1.norm())?;
        c1 = c3;
        checked_add_scaled_self(&mut p, beta, &r)?;
        previous_residual_squared = recursive_residual_squared;
    }

    Err(SolverError::Exhaustion.into())
}

/// Applies the pinned `bicgstab` recurrence to a general operator.
///
/// `output` is both the initial guess and the transactional destination. The
/// implementation keeps the Julia names `r`, `p`, `Ap`, `s`, `t`, `alpha`,
/// `beta`, and `omega`, and resets the shadow residual exactly at the pinned
/// near-orthogonality test.
///
/// # Errors
///
/// Returns field/operator mismatch errors, allocation or tensor errors, or
/// [`DiracError::Solver`] with a typed non-finite, denominator breakdown,
/// singular-shadow-restart, stagnation, exhaustion, or true-residual failure.
/// The output remains unchanged on every error.
///
/// # Examples
///
/// ```
/// use dirac_operators::{bicgstab, FermionField, SolverParams, WilsonDirac};
/// use gaugefields::{cold_su3, LatticeShape4};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let dirac = WilsonDirac::new(&links, 0.1)?;
/// let rhs = FermionField::zeros(lattice, 4)?;
/// let mut solution = FermionField::zeros(lattice, 4)?;
/// let report = bicgstab(
///     &mut solution,
///     &dirac,
///     &rhs,
///     SolverParams::new(1.0e-20, 8)?,
/// )?;
/// assert_eq!(report.iterations, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(non_snake_case)]
pub fn bicgstab<O: FermionOperator>(
    output: &mut FermionField,
    operator: &O,
    rhs: &FermionField,
    params: SolverParams,
) -> Result<SolverReport, DiracError> {
    validate_solver_inputs(output, operator, rhs)?;

    let mut work = output.try_clone()?;
    let mut r = rhs.try_clone()?;
    let mut temp1 = scratch_field(operator)?;
    let mut operator_scratch = scratch_fields(operator, OPERATOR_SCRATCH_FIELDS)?;
    let mut rs = scratch_field(operator)?;
    let mut p = scratch_field(operator)?;
    let mut Ap = scratch_field(operator)?;
    let mut s = scratch_field(operator)?;
    let mut t = scratch_field(operator)?;

    // Julia `bicgstab`: r = b - A*x, rs = r, p = r.
    checked_copy(&mut r, rhs)?;
    checked_apply(operator, &mut temp1, &work, &mut operator_scratch)?;
    checked_add_scaled(&mut r, -ONE, &temp1)?;
    checked_copy(&mut rs, &r)?;
    checked_copy(&mut p, &r)?;
    let mut rnorm = checked_norm_squared(&r)?;
    let initial_residual_squared = rnorm;
    if initial_residual_squared < params.tolerance {
        return finish(
            output,
            &mut work,
            operator,
            rhs,
            FinishScratch {
                temp1: &mut temp1,
                residual: &mut r,
                operator_scratch: &mut operator_scratch,
            },
            FinishState {
                method: SolverMethod::BiCgStab,
                iterations: 0,
                initial_residual_squared,
                recursive_residual_squared: initial_residual_squared,
                params,
                restart_count: 0,
                convergence_branch: ConvergenceBranch::InitialResidual,
            },
        );
    }

    let mut restart_count = 0usize;
    let mut previous_residual_squared = initial_residual_squared;
    for iterations in 1..=params.max_iterations {
        // Julia `bicgstab`: c1 = dot(rs, r), followed by the shadow restart.
        let mut c1 = checked_dot(&rs, &r)?;
        let mut rs_norm_squared = checked_norm_squared(&rs)?;
        let mut rho_scale = product_scale(rs_norm_squared, rnorm)?;
        let mut restarted_this_iteration = false;
        if shadow_near_zero(c1, rho_scale)? {
            checked_copy(&mut rs, &r)?;
            checked_copy(&mut p, &r)?;
            c1 = checked_dot(&rs, &r)?;
            restart_count = restart_count
                .checked_add(1)
                .ok_or(SolverError::Exhaustion)?;
            restarted_this_iteration = true;
            rs_norm_squared = rnorm;
            rho_scale = product_scale(rnorm, rnorm)?;
            if shadow_near_zero(c1, rho_scale)? {
                return Err(SolverError::SingularShadowRestart.into());
            }
        }

        // Julia `bicgstab`: Ap = A*p; alpha = c1/dot(rs, Ap).
        checked_apply(operator, &mut Ap, &p, &mut operator_scratch)?;
        let c2 = checked_dot(&rs, &Ap)?;
        let ap_norm_squared = checked_norm_squared(&Ap)?;
        let alpha_scale = product_scale(rs_norm_squared, ap_norm_squared)?;
        let alpha = match checked_division(c1, c2, alpha_scale) {
            Err(DiracError::Solver(SolverError::Breakdown)) if restarted_this_iteration => {
                return Err(SolverError::SingularShadowRestart.into());
            }
            result => result?,
        };

        // Julia `bicgstab`: s = r - alpha*Ap.
        checked_copy(&mut s, &r)?;
        checked_add_scaled(&mut s, -alpha, &Ap)?;
        let snorm = checked_norm_squared(&s)?;
        if snorm < params.tolerance {
            checked_add_scaled(&mut work, alpha, &p)?;
            return finish(
                output,
                &mut work,
                operator,
                rhs,
                FinishScratch {
                    temp1: &mut temp1,
                    residual: &mut r,
                    operator_scratch: &mut operator_scratch,
                },
                FinishState {
                    method: SolverMethod::BiCgStab,
                    iterations,
                    initial_residual_squared,
                    recursive_residual_squared: snorm,
                    params,
                    restart_count,
                    convergence_branch: ConvergenceBranch::IntermediateResidual,
                },
            );
        }

        // Julia `bicgstab`: t = A*s; omega = dot(t,s)/dot(t,t).
        checked_apply(operator, &mut t, &s, &mut operator_scratch)?;
        let d1 = checked_dot(&t, &s)?;
        let d2 = checked_dot(&t, &t)?;
        let omega = checked_division(d1, d2, d2.norm())?;
        check_scalar_denominator(omega, alpha.norm())?;

        // Julia `bicgstab`: r = s - omega*t.
        checked_copy(&mut r, &s)?;
        checked_add_scaled(&mut r, -omega, &t)?;

        // Julia `bicgstab`: x += omega*s; x += alpha*p (ordering retained).
        checked_add_scaled(&mut work, omega, &s)?;
        checked_add_scaled(&mut work, alpha, &p)?;
        rnorm = checked_norm_squared(&r)?;
        if rnorm < params.tolerance {
            return finish(
                output,
                &mut work,
                operator,
                rhs,
                FinishScratch {
                    temp1: &mut temp1,
                    residual: &mut r,
                    operator_scratch: &mut operator_scratch,
                },
                FinishState {
                    method: SolverMethod::BiCgStab,
                    iterations,
                    initial_residual_squared,
                    recursive_residual_squared: rnorm,
                    params,
                    restart_count,
                    convergence_branch: ConvergenceBranch::UpdatedResidual,
                },
            );
        }
        if is_stagnant(previous_residual_squared, rnorm) {
            return Err(SolverError::Stagnation.into());
        }

        // Julia `bicgstab`: beta = (dot(rs,r)/c1)*(alpha/omega).
        let shadow_residual = checked_dot(&rs, &r)?;
        let beta_left = checked_division(shadow_residual, c1, rho_scale)?;
        let beta_right = checked_division(alpha, omega, alpha.norm())?;
        let beta = beta_left * beta_right;
        if !complex_is_finite(beta) {
            return Err(SolverError::NonFiniteIntermediate.into());
        }

        // Julia `bicgstab`: p = beta*p + r; p += -omega*beta*Ap.
        checked_add_scaled_self(&mut p, beta, &r)?;
        let correction = -omega * beta;
        if !complex_is_finite(correction) {
            return Err(SolverError::NonFiniteIntermediate.into());
        }
        checked_add_scaled(&mut p, correction, &Ap)?;
        previous_residual_squared = rnorm;
    }

    Err(SolverError::Exhaustion.into())
}

fn validate_multi_shift_inputs<O: HermitianPositiveOperator>(
    outputs: &[FermionField],
    operator: &O,
    rhs: &FermionField,
    shifts: &[f64],
) -> Result<(), DiracError> {
    if outputs.len() != shifts.len() {
        return Err(SolverError::ShiftCountMismatch {
            outputs: outputs.len(),
            shifts: shifts.len(),
        }
        .into());
    }
    validate_operator_rhs(operator, rhs)?;
    let expected_len = operator
        .components()
        .checked_mul(operator.lattice().nv())
        .and_then(|value| value.checked_mul(3))
        .ok_or(DiracError::AllocationOverflow)?;
    for (index, (&shift, output)) in shifts.iter().zip(outputs).enumerate() {
        if !shift.is_finite() || shift < 0.0 {
            return Err(SolverError::InvalidShift { index }.into());
        }
        if output.lattice() != operator.lattice() {
            return Err(DiracError::LatticeMismatch {
                operand: "output",
                expected: operator.lattice(),
                found: output.lattice(),
            });
        }
        if output.components() != operator.components() {
            return Err(DiracError::ComponentsMismatch {
                operand: "output",
                expected: operator.components(),
                found: output.components(),
            });
        }
        if output.host_data()?.len() != expected_len {
            return Err(DiracError::StorageInvariant);
        }
        output.ensure_finite().map_err(map_numeric_error)?;
    }
    Ok(())
}

fn validate_operator_rhs<O: FermionOperator>(
    operator: &O,
    rhs: &FermionField,
) -> Result<(), DiracError> {
    let components = operator.components();
    if !matches!(components, 1 | 4) {
        return Err(DiracError::InvalidComponents { found: components });
    }
    if rhs.lattice() != operator.lattice() {
        return Err(DiracError::LatticeMismatch {
            operand: "rhs",
            expected: operator.lattice(),
            found: rhs.lattice(),
        });
    }
    if rhs.components() != components {
        return Err(DiracError::ComponentsMismatch {
            operand: "rhs",
            expected: components,
            found: rhs.components(),
        });
    }
    let expected_len = components
        .checked_mul(operator.lattice().nv())
        .and_then(|value| value.checked_mul(3))
        .ok_or(DiracError::AllocationOverflow)?;
    if rhs.host_data()?.len() != expected_len {
        return Err(DiracError::StorageInvariant);
    }
    rhs.ensure_finite().map_err(map_numeric_error)
}

fn solve_zero_shifted<O: HermitianPositiveOperator>(
    work: &mut [FermionField],
    operator: &O,
    rhs: &FermionField,
    shifts: &[f64],
    params: SolverParams,
) -> Result<Vec<MultiShiftSolverReport>, DiracError> {
    let count = shifts.len();
    let mut x = FermionField::zeros(operator.lattice(), operator.components())?;
    let mut r = rhs.try_clone()?;
    let mut p = rhs.try_clone()?;
    let mut q = scratch_field(operator)?;
    let mut operator_scratch = scratch_fields(operator, OPERATOR_SCRATCH_FIELDS)?;
    let mut shifted_p = Vec::with_capacity(count);
    for _ in 0..count {
        shifted_p.push(rhs.try_clone()?);
    }
    let mut rho_m = vec![ONE; count];
    let mut rho_0 = vec![ONE; count];
    let mut rho_p = vec![ONE; count];
    let mut converged = vec![false; count];
    let mut initial_residuals = vec![0.0; count];
    let mut recursive_residuals = vec![0.0; count];
    let mut iterations = vec![0usize; count];
    let mut branches = vec![ConvergenceBranch::InitialResidual; count];
    let initial = checked_norm_squared(&r)?;
    initial_residuals.fill(initial);
    recursive_residuals.fill(initial);
    let mut active = count;

    if initial >= params.tolerance {
        let mut c1 = checked_dot(&p, &p)?;
        require_positive(c1)?;
        let mut alpha_m = ONE;
        let mut beta_m = Complex64::default();
        for iteration in 1..=params.max_iterations {
            // Julia `shiftedcg`: q = A*p; pAp = p⋅q; rr = r⋅r.
            checked_apply(operator, &mut q, &p, &mut operator_scratch)?;
            let p_ap = checked_dot(&p, &q)?;
            let q_norm_squared = checked_norm_squared(&q)?;
            require_positive(p_ap)?;
            let rr = checked_dot(&r, &r)?;
            let rr_squared = require_residual(rr)?;
            let alpha_k = checked_division(rr, p_ap, product_scale(rr_squared, q_norm_squared)?)?;

            // Julia `shiftedcg`: x += alpha*p; r -= alpha*q.
            checked_add_scaled(&mut x, alpha_k, &p)?;
            checked_add_scaled(&mut r, -alpha_k, &q)?;
            let c3 = checked_dot(&r, &r)?;
            require_residual(c3)?;
            let beta_k = checked_division(c3, c1, c1.norm())?;
            checked_add_scaled_self(&mut p, beta_k, &r)?;

            for (index, shift) in shifts.iter().copied().enumerate() {
                if converged[index] {
                    continue;
                }
                let rho_kj = rho_0[index];
                if rho_kj.norm() < params.tolerance {
                    converged[index] = true;
                    active = active.saturating_sub(1);
                    recursive_residuals[index] = 0.0;
                    iterations[index] = iteration;
                    branches[index] = ConvergenceBranch::UpdatedResidual;
                    continue;
                }
                let shift = Complex64::new(shift, 0.0);
                let first_term = rho_m[index] * alpha_m * (ONE + alpha_k * shift);
                let second_term = alpha_k * beta_m * (rho_m[index] - rho_kj);
                let denominator = first_term + second_term;
                let numerator = rho_kj * rho_m[index] * alpha_m;
                let denominator_scale = first_term.norm() + second_term.norm();
                let rho_next = checked_division(numerator, denominator, denominator_scale)?;
                let ratio = checked_division(rho_next, rho_kj, rho_kj.norm())?;
                let alpha_kj = ratio * alpha_k;
                if !complex_is_finite(alpha_kj) {
                    return Err(SolverError::NonFiniteIntermediate.into());
                }
                checked_add_scaled(&mut work[index], alpha_kj, &shifted_p[index])?;
                let beta_kj = ratio * ratio * beta_k;
                if !complex_is_finite(beta_kj) {
                    return Err(SolverError::NonFiniteIntermediate.into());
                }
                shifted_p[index].scale_and_add_scaled(beta_kj, rho_next, &r)?;
                // Julia `shiftedcg` estimates with the residual norm before
                // the current base-CG update (`rr * abs(rho_p)^2`).
                let estimate = rr_squared * rho_next.norm_sqr();
                if !estimate.is_finite() {
                    return Err(SolverError::NonFiniteIntermediate.into());
                }
                rho_p[index] = rho_next;
                recursive_residuals[index] = estimate;
                if estimate < params.tolerance {
                    converged[index] = true;
                    active = active.saturating_sub(1);
                    iterations[index] = iteration;
                    branches[index] = ConvergenceBranch::UpdatedResidual;
                }
            }

            rho_m.copy_from_slice(&rho_0);
            rho_0.copy_from_slice(&rho_p);
            alpha_m = alpha_k;
            beta_m = beta_k;
            c1 = c3;
            if active == 0 {
                break;
            }
        }
        if active != 0 {
            return Err(SolverError::Exhaustion.into());
        }
    }

    finish_multi_shift(
        work,
        operator,
        rhs,
        shifts,
        params,
        MultiShiftState {
            initial_residuals: &initial_residuals,
            recursive_residuals: &recursive_residuals,
            iterations: &iterations,
            branches: &branches,
        },
        &mut operator_scratch,
    )
}

fn solve_nonzero_shifted<O: HermitianPositiveOperator>(
    work: &mut [FermionField],
    operator: &O,
    rhs: &FermionField,
    shifts: &[f64],
    params: SolverParams,
) -> Result<Vec<MultiShiftSolverReport>, DiracError> {
    let count = shifts.len();
    let mut residuals = Vec::with_capacity(count);
    let mut directions = Vec::with_capacity(count);
    for _ in 0..count {
        residuals.push(FermionField::zeros(
            operator.lattice(),
            operator.components(),
        )?);
        directions.push(FermionField::zeros(
            operator.lattice(),
            operator.components(),
        )?);
    }
    let mut applied = scratch_field(operator)?;
    let mut operator_scratch = scratch_fields(operator, OPERATOR_SCRATCH_FIELDS)?;
    let mut initial_residuals = vec![0.0; count];
    let mut recursive_residuals = vec![0.0; count];
    let mut iterations = vec![0usize; count];
    let mut branches = vec![ConvergenceBranch::InitialResidual; count];

    for index in 0..count {
        checked_shifted_apply(
            operator,
            &mut applied,
            &work[index],
            shifts[index],
            &mut operator_scratch,
        )?;
        checked_copy(&mut residuals[index], rhs)?;
        checked_add_scaled(&mut residuals[index], -ONE, &applied)?;
        let initial = checked_norm_squared(&residuals[index])?;
        initial_residuals[index] = initial;
        recursive_residuals[index] = initial;
        if initial < params.tolerance {
            continue;
        }
        checked_copy(&mut directions[index], &residuals[index])?;
        let mut c1 = checked_dot(&directions[index], &residuals[index])?;
        require_positive(c1)?;
        let mut previous_residual_squared = initial;
        let mut converged = false;
        for iteration in 1..=params.max_iterations {
            checked_shifted_apply(
                operator,
                &mut applied,
                &directions[index],
                shifts[index],
                &mut operator_scratch,
            )?;
            let c2 = checked_dot(&directions[index], &applied)?;
            let applied_norm_squared = checked_norm_squared(&applied)?;
            require_positive(c2)?;
            let alpha = checked_division(c1, c2, product_scale(c1.re, applied_norm_squared)?)?;
            checked_add_scaled(&mut work[index], alpha, &directions[index])?;
            checked_add_scaled(&mut residuals[index], -alpha, &applied)?;
            let c3 = checked_dot(&residuals[index], &residuals[index])?;
            let residual_squared = require_residual(c3)?;
            recursive_residuals[index] = residual_squared;
            if residual_squared < params.tolerance {
                iterations[index] = iteration;
                branches[index] = ConvergenceBranch::UpdatedResidual;
                converged = true;
                break;
            }
            if is_stagnant(previous_residual_squared, residual_squared) {
                return Err(SolverError::Stagnation.into());
            }
            let beta = checked_division(c3, c1, c1.norm())?;
            checked_add_scaled_self(&mut directions[index], beta, &residuals[index])?;
            c1 = c3;
            previous_residual_squared = residual_squared;
        }
        if !converged {
            return Err(SolverError::Exhaustion.into());
        }
    }

    finish_multi_shift(
        work,
        operator,
        rhs,
        shifts,
        params,
        MultiShiftState {
            initial_residuals: &initial_residuals,
            recursive_residuals: &recursive_residuals,
            iterations: &iterations,
            branches: &branches,
        },
        &mut operator_scratch,
    )
}

struct MultiShiftState<'a> {
    initial_residuals: &'a [f64],
    recursive_residuals: &'a [f64],
    iterations: &'a [usize],
    branches: &'a [ConvergenceBranch],
}

fn finish_multi_shift<O: HermitianPositiveOperator>(
    work: &[FermionField],
    operator: &O,
    rhs: &FermionField,
    shifts: &[f64],
    params: SolverParams,
    state: MultiShiftState<'_>,
    operator_scratch: &mut [FermionField],
) -> Result<Vec<MultiShiftSolverReport>, DiracError> {
    let mut applied = scratch_field(operator)?;
    let mut residual = scratch_field(operator)?;
    let mut reports = Vec::with_capacity(shifts.len());
    for index in 0..shifts.len() {
        checked_shifted_apply(
            operator,
            &mut applied,
            &work[index],
            shifts[index],
            operator_scratch,
        )?;
        checked_copy(&mut residual, rhs)?;
        checked_add_scaled(&mut residual, -ONE, &applied)?;
        let true_residual_squared = checked_norm_squared(&residual)?;
        if true_residual_squared >= params.tolerance {
            return Err(SolverError::TrueResidualMismatch.into());
        }
        reports.push(MultiShiftSolverReport {
            shift: shifts[index],
            iterations: state.iterations[index],
            recursive_residual_squared: state.recursive_residuals[index],
            initial_residual_squared: state.initial_residuals[index],
            true_residual_squared,
            tolerance: params.tolerance,
            maximum_iterations: params.max_iterations,
            convergence_branch: state.branches[index],
        });
    }
    Ok(reports)
}

fn checked_shifted_apply<O: HermitianPositiveOperator>(
    operator: &O,
    output: &mut FermionField,
    input: &FermionField,
    shift: f64,
    operator_scratch: &mut [FermionField],
) -> Result<(), DiracError> {
    checked_apply(operator, output, input, operator_scratch)?;
    checked_add_scaled(output, Complex64::new(shift, 0.0), input)
}

fn validate_solver_inputs<O: FermionOperator>(
    output: &FermionField,
    operator: &O,
    rhs: &FermionField,
) -> Result<(), DiracError> {
    let lattice = operator.lattice();
    let components = operator.components();
    if !matches!(components, 1 | 4) {
        return Err(DiracError::InvalidComponents { found: components });
    }
    if output.lattice() != lattice {
        return Err(DiracError::LatticeMismatch {
            operand: "output",
            expected: lattice,
            found: output.lattice(),
        });
    }
    if rhs.lattice() != lattice {
        return Err(DiracError::LatticeMismatch {
            operand: "rhs",
            expected: lattice,
            found: rhs.lattice(),
        });
    }
    if output.components() != components {
        return Err(DiracError::ComponentsMismatch {
            operand: "output",
            expected: components,
            found: output.components(),
        });
    }
    if rhs.components() != components {
        return Err(DiracError::ComponentsMismatch {
            operand: "rhs",
            expected: components,
            found: rhs.components(),
        });
    }
    output.ensure_finite().map_err(map_numeric_error)?;
    rhs.ensure_finite().map_err(map_numeric_error)?;
    Ok(())
}

fn scratch_field<O: FermionOperator>(operator: &O) -> Result<FermionField, DiracError> {
    FermionField::zeros(operator.lattice(), operator.components())
}

fn scratch_fields<O: FermionOperator>(
    operator: &O,
    count: usize,
) -> Result<Vec<FermionField>, DiracError> {
    let mut fields = Vec::with_capacity(count);
    for _ in 0..count {
        fields.push(scratch_field(operator)?);
    }
    Ok(fields)
}

fn checked_apply<O: FermionOperator>(
    operator: &O,
    output: &mut FermionField,
    input: &FermionField,
    scratch: &mut [FermionField],
) -> Result<(), DiracError> {
    operator
        .apply_into_with_scratch(output, input, scratch)
        .map_err(map_numeric_error)?;
    output.ensure_finite().map_err(map_numeric_error)
}

fn checked_copy(output: &mut FermionField, input: &FermionField) -> Result<(), DiracError> {
    output.copy_from(input).map_err(map_numeric_error)
}

fn checked_add_scaled(
    output: &mut FermionField,
    factor: Complex64,
    input: &FermionField,
) -> Result<(), DiracError> {
    output.add_scaled(factor, input).map_err(map_numeric_error)
}

fn checked_add_scaled_self(
    output: &mut FermionField,
    factor: Complex64,
    input: &FermionField,
) -> Result<(), DiracError> {
    output
        .add_scaled_self(factor, input)
        .map_err(map_numeric_error)
}

fn checked_dot(left: &FermionField, right: &FermionField) -> Result<Complex64, DiracError> {
    let value = left.inner_product(right).map_err(map_numeric_error)?;
    if !complex_is_finite(value) {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    Ok(value)
}

fn checked_norm_squared(field: &FermionField) -> Result<f64, DiracError> {
    require_residual(checked_dot(field, field)?)
}

fn require_residual(value: Complex64) -> Result<f64, DiracError> {
    if !complex_is_finite(value) {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    if value.re < 0.0 {
        return Err(SolverError::Breakdown.into());
    }
    Ok(value.re)
}

fn require_positive(value: Complex64) -> Result<(), DiracError> {
    if !complex_is_finite(value) {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    if value.re <= 0.0 {
        return Err(SolverError::Breakdown.into());
    }
    Ok(())
}

fn product_scale(left: f64, right: f64) -> Result<f64, DiracError> {
    if !left.is_finite() || !right.is_finite() || left < 0.0 || right < 0.0 {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    let product = left * right;
    if !product.is_finite() {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    Ok(product.sqrt())
}

fn checked_division(
    numerator: Complex64,
    denominator: Complex64,
    scale: f64,
) -> Result<Complex64, DiracError> {
    if !complex_is_finite(numerator) || !complex_is_finite(denominator) || !scale.is_finite() {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    let threshold = DENOMINATOR_RELATIVE_TOLERANCE * scale.max(f64::MIN_POSITIVE);
    if denominator.norm() <= threshold {
        return Err(SolverError::Breakdown.into());
    }
    let quotient = numerator / denominator;
    if !complex_is_finite(quotient) {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    Ok(quotient)
}

fn check_scalar_denominator(value: Complex64, scale: f64) -> Result<(), DiracError> {
    if !complex_is_finite(value) || !scale.is_finite() {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    let threshold = DENOMINATOR_RELATIVE_TOLERANCE * scale.max(f64::MIN_POSITIVE);
    if value.norm() <= threshold {
        return Err(SolverError::Breakdown.into());
    }
    Ok(())
}

fn shadow_near_zero(value: Complex64, scale: f64) -> Result<bool, DiracError> {
    if !complex_is_finite(value) || !scale.is_finite() {
        return Err(SolverError::NonFiniteIntermediate.into());
    }
    Ok(value.norm() <= SHADOW_RELATIVE_TOLERANCE * scale.max(f64::MIN_POSITIVE))
}

fn is_stagnant(previous: f64, current: f64) -> bool {
    let scale = previous.max(f64::MIN_POSITIVE);
    (previous - current).abs() <= DENOMINATOR_RELATIVE_TOLERANCE * scale
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn map_numeric_error(error: DiracError) -> DiracError {
    match error {
        DiracError::NumericalRange | DiracError::NonFinite { .. } => {
            SolverError::NonFiniteIntermediate.into()
        }
        other => other,
    }
}

struct FinishScratch<'a> {
    temp1: &'a mut FermionField,
    residual: &'a mut FermionField,
    operator_scratch: &'a mut [FermionField],
}

struct FinishState {
    method: SolverMethod,
    iterations: usize,
    initial_residual_squared: f64,
    recursive_residual_squared: f64,
    params: SolverParams,
    restart_count: usize,
    convergence_branch: ConvergenceBranch,
}

fn finish<O: FermionOperator>(
    output: &mut FermionField,
    work: &mut FermionField,
    operator: &O,
    rhs: &FermionField,
    scratch: FinishScratch<'_>,
    state: FinishState,
) -> Result<SolverReport, DiracError> {
    // The recursive residual is never trusted as the commit gate.
    checked_apply(operator, scratch.temp1, work, scratch.operator_scratch)?;
    checked_copy(scratch.residual, rhs)?;
    checked_add_scaled(scratch.residual, -ONE, scratch.temp1)?;
    let true_residual_squared = checked_norm_squared(scratch.residual)?;
    if true_residual_squared >= state.params.tolerance {
        return Err(SolverError::TrueResidualMismatch.into());
    }
    output.copy_from(work).map_err(map_numeric_error)?;
    Ok(SolverReport {
        method: state.method,
        iterations: state.iterations,
        recursive_residual_squared: state.recursive_residual_squared,
        initial_residual_squared: state.initial_residual_squared,
        true_residual_squared,
        tolerance: state.params.tolerance,
        maximum_iterations: state.params.max_iterations,
        restart_count: state.restart_count,
        convergence_branch: state.convergence_branch,
    })
}

#[cfg(test)]
mod tests;
