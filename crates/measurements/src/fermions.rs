//! Fermion measurements parallel to the pinned QCDMeasurements.jl contracts.
//!
//! The field layout and solver entrypoint follow
//! `LatticeDiracOperators.jl` v0.6.4 at revision
//! `bdef628184597815ba3e0cddf2536df767e78a02`. The pion source ordering and
//! timeslice contraction follow `QCDMeasurements.jl` v0.2.13 at revision
//! `9e04c37bbd68712cf7a749ae5aff10eb6aae4566`, but this module evaluates the
//! full sink-component Frobenius contraction rather than the pinned
//! source-diagonal reconstruction. Noise uses the caller-owned
//! `gaugefields::ReproducibleRng` and one raw word per compact field element.

use dirac_operators::{
    bicgstab, DiracError, FermionField, FermionOperator, SolverParams, SolverReport,
};
use gaugefields::ReproducibleRng;
use num_complex::Complex64;
use rand::RngCore;

const NC: usize = 3;

/// Errors returned by the fermion measurements.
#[derive(Debug, thiserror::Error)]
pub enum FermionMeasurementError {
    /// The checked Dirac field, operator, or solver rejected an input or
    /// encountered a typed numerical failure.
    #[error(transparent)]
    Dirac(#[from] DiracError),
    /// The requested stochastic flavor normalization is invalid.
    #[error("flavor factor must be finite and positive, found {found}")]
    InvalidFlavorFactor { found: f64 },
    /// A stochastic estimate must contain at least one source.
    #[error("stochastic chiral condensate requires at least one source")]
    ZeroSources,
    /// A measurement accumulation left the finite floating-point range.
    #[error("{operation} accumulated a non-finite value")]
    NumericalRange { operation: &'static str },
}

/// The corrected point-source pion correlator and its per-source diagnostics.
#[derive(Debug)]
pub struct PionCorrelator {
    /// Unnormalized correlator values in temporal timeslice order.
    pub values: Vec<f64>,
    /// Solver diagnostics in color-outer, component-inner source order.
    pub solver_reports: Vec<SolverReport>,
}

/// A stochastic chiral-condensate estimate and its per-source diagnostics.
#[derive(Debug)]
pub struct ChiralCondensate {
    /// Flavor-normalized, volume-normalized condensate estimate.
    pub value: f64,
    /// Unnormalized `Re(r† D⁻¹ r)` estimate for each source.
    pub source_values: Vec<f64>,
    /// Solver diagnostics in source-generation order.
    pub solver_reports: Vec<SolverReport>,
}

/// Compute the corrected full point-source pion correlator.
///
/// A zero initial guess is solved for each source at site zero. Sources are
/// ordered by color outermost and component innermost. The returned value is
///
/// `C(t) = sum_xyz sum_alpha,beta |G_beta,alpha(x,y,z,t)|²`.
///
/// There is no source, component, color, or volume normalization.
///
/// # Errors
///
/// Returns [`FermionMeasurementError::Dirac`] for incompatible fields,
/// solver failure, or non-finite operator arithmetic, and
/// [`FermionMeasurementError::NumericalRange`] for a non-finite contraction.
pub fn pion_correlator<O: FermionOperator>(
    operator: &O,
    solver_params: SolverParams,
) -> Result<PionCorrelator, FermionMeasurementError> {
    validate_components(operator.components())?;
    let lattice = operator.lattice();
    let [nx, ny, nz, nt] = lattice.extents();
    let spatial_volume = nx
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nz))
        .ok_or(DiracError::AllocationOverflow)?;
    let source_count = NC
        .checked_mul(operator.components())
        .ok_or(DiracError::AllocationOverflow)?;
    let mut values = vec![0.0; nt];
    let mut solver_reports = Vec::with_capacity(source_count);

    for color in 0..NC {
        for component in 0..operator.components() {
            let source =
                FermionField::point_source(lattice, operator.components(), color, component, 0)?;
            let mut solution = FermionField::zeros(lattice, operator.components())?;
            let report = bicgstab(&mut solution, operator, &source, solver_params)?;

            for site in 0..lattice.nv() {
                // INVARIANT: site is x-fast, so each contiguous spatial block is one t.
                let timeslice = site / spatial_volume;
                for sink_color in 0..NC {
                    for sink_component in 0..operator.components() {
                        let value = solution.component(sink_color, sink_component, site)?;
                        let contribution = value.norm_sqr();
                        if !contribution.is_finite() {
                            return Err(FermionMeasurementError::NumericalRange {
                                operation: "pion_correlator",
                            });
                        }
                        values[timeslice] += contribution;
                        if !values[timeslice].is_finite() {
                            return Err(FermionMeasurementError::NumericalRange {
                                operation: "pion_correlator",
                            });
                        }
                    }
                }
            }
            solver_reports.push(report);
        }
    }

    Ok(PionCorrelator {
        values,
        solver_reports,
    })
}

/// Compute a stochastic chiral condensate with canonical raw-word Z4 noise.
///
/// For every source, one raw `u64` word is consumed for each physical field
/// element in compact Rust storage order `[color, component, x, y, z, t]`.
/// The low two bits map exactly to `{1, i, -1, -i}`. Each source solves
/// `D p = r` from zero and contributes `Re(r†p)`. The returned value is
/// `flavor_factor * mean(source_values) / NV`.
///
/// `flavor_factor` is caller-selected; for a staggered `Nf` estimator it is
/// conventionally `Nf / 4`.
///
/// # Errors
///
/// Returns a typed validation, field/operator, solver, or finite-arithmetic
/// error. Random words consumed before a later solve failure are not rolled
/// back.
pub fn stochastic_chiral_condensate<O: FermionOperator>(
    operator: &O,
    flavor_factor: f64,
    source_count: usize,
    solver_params: SolverParams,
    rng: &mut ReproducibleRng,
) -> Result<ChiralCondensate, FermionMeasurementError> {
    if !flavor_factor.is_finite() || flavor_factor <= 0.0 {
        return Err(FermionMeasurementError::InvalidFlavorFactor {
            found: flavor_factor,
        });
    }
    if source_count == 0 {
        return Err(FermionMeasurementError::ZeroSources);
    }
    validate_components(operator.components())?;

    let lattice = operator.lattice();
    let element_count = NC
        .checked_mul(operator.components())
        .and_then(|value| value.checked_mul(lattice.nv()))
        .ok_or(DiracError::AllocationOverflow)?;
    let mut source_values = Vec::with_capacity(source_count);
    let mut solver_reports = Vec::with_capacity(source_count);
    let mut total = 0.0;

    for _ in 0..source_count {
        let mut values = Vec::with_capacity(element_count);
        for _ in 0..element_count {
            values.push(canonical_z4(rng.next_u64()));
        }
        let source = FermionField::from_vec_col_major(lattice, operator.components(), values)?;
        let mut solution = FermionField::zeros(lattice, operator.components())?;
        let report = bicgstab(&mut solution, operator, &source, solver_params)?;
        let inner = source.inner_product(&solution)?;
        let estimate = inner.re;
        if !estimate.is_finite() {
            return Err(FermionMeasurementError::NumericalRange {
                operation: "stochastic_chiral_condensate",
            });
        }
        total += estimate;
        if !total.is_finite() {
            return Err(FermionMeasurementError::NumericalRange {
                operation: "stochastic_chiral_condensate",
            });
        }
        source_values.push(estimate);
        solver_reports.push(report);
    }

    let mean = total / source_count as f64;
    let value = flavor_factor * mean / lattice.nv() as f64;
    if !mean.is_finite() || !value.is_finite() {
        return Err(FermionMeasurementError::NumericalRange {
            operation: "stochastic_chiral_condensate",
        });
    }

    Ok(ChiralCondensate {
        value,
        source_values,
        solver_reports,
    })
}

fn validate_components(components: usize) -> Result<(), FermionMeasurementError> {
    if matches!(components, 1 | 4) {
        Ok(())
    } else {
        Err(DiracError::InvalidComponents { found: components }.into())
    }
}

fn canonical_z4(word: u64) -> Complex64 {
    match word & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::canonical_z4;
    use num_complex::Complex64;

    #[test]
    fn raw_words_use_canonical_z4_low_bits() {
        assert_eq!(canonical_z4(0), Complex64::new(1.0, 0.0));
        assert_eq!(canonical_z4(1), Complex64::new(0.0, 1.0));
        assert_eq!(canonical_z4(2), Complex64::new(-1.0, 0.0));
        assert_eq!(canonical_z4(3), Complex64::new(0.0, -1.0));
        assert_eq!(canonical_z4(u64::MAX), Complex64::new(0.0, -1.0));
    }
}
