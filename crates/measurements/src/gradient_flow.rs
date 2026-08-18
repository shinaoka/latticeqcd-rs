//! Fixed-step third-order Runge--Kutta Wilson flow.

use crate::MeasurementError;
use gaugefields::{exp_ta_update, CpuEvolutionContext, GaugeLinks, LatticeShape4, TaGaugeField};
use wilsonloop::{loop_action_force, LoopAction};

/// Validated parameters for fixed-step third-order Runge--Kutta flow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientFlowParams {
    step_size: f64,
    steps: usize,
}

impl GradientFlowParams {
    /// Constructs positive finite flow parameters with at least one step.
    ///
    /// # Errors
    ///
    /// Returns [`MeasurementError::NonFiniteStepSize`] or
    /// [`MeasurementError::NonPositiveStepSize`] for an invalid step size, and
    /// [`MeasurementError::ZeroFlowSteps`] when `steps` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use measurements::{GradientFlowParams, MeasurementError};
    ///
    /// let params = GradientFlowParams::new(0.01, 4)?;
    /// assert_eq!(params.step_size(), 0.01);
    /// assert_eq!(params.steps(), 4);
    /// assert!(matches!(
    ///     GradientFlowParams::new(0.0, 1),
    ///     Err(MeasurementError::NonPositiveStepSize { .. })
    /// ));
    /// # Ok::<(), MeasurementError>(())
    /// ```
    pub fn new(step_size: f64, steps: usize) -> Result<Self, MeasurementError> {
        if !step_size.is_finite() {
            return Err(MeasurementError::NonFiniteStepSize { found: step_size });
        }
        if step_size <= 0.0 {
            return Err(MeasurementError::NonPositiveStepSize { found: step_size });
        }
        if steps == 0 {
            return Err(MeasurementError::ZeroFlowSteps);
        }
        Ok(Self { step_size, steps })
    }

    /// Returns the positive flow step size.
    pub const fn step_size(self) -> f64 {
        self.step_size
    }

    /// Returns the number of RK3 flow steps.
    pub const fn steps(self) -> usize {
        self.steps
    }
}

/// Applies fixed-step third-order Runge--Kutta Wilson flow for a general loop action.
///
/// The positive force is `wilsonloop::loop_action_force`; the negative flow
/// direction is carried only by the RK3 coefficients. The input links remain
/// unchanged on success and failure, while `context` is reused for every
/// exponential update.
///
/// # Errors
///
/// Returns a typed parameter, action, input, Wilson-kernel, numerical, or CPU
/// evolution error. `GradientFlowParams` and `LoopAction` carry their
/// constructor validation; finite input is checked before the input is cloned
/// or the backend is entered, while cloning owns its allocation checks.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, CpuEvolutionContext, LatticeShape4};
/// use measurements::{gradient_flow, GradientFlowParams};
/// use tenferro_cpu::CpuBackend;
/// use wilsonloop::{LoopAction, LoopTerm};
///
/// let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
/// let action = LoopAction::new(vec![LoopTerm::plaquette(1.0, 1, 2)?])?;
/// let params = GradientFlowParams::new(0.01, 1)?;
/// let mut context = CpuEvolutionContext::new(CpuBackend::new());
/// let flowed = gradient_flow(&mut context, &links, &action, params)?;
/// assert_eq!(flowed.lattice(), links.lattice());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gradient_flow(
    context: &mut CpuEvolutionContext,
    links: &GaugeLinks,
    action: &LoopAction,
    params: GradientFlowParams,
) -> Result<GaugeLinks, MeasurementError> {
    crate::validated_view(links)?;
    let mut current = links.try_clone()?;
    let eps = params.step_size();
    for _ in 0..params.steps() {
        let f0 = flow_force(&current, action)?;
        let w1 = evolve(context, &current, -(eps / 4.0), &f0)?;

        let f1 = flow_force(&w1, action)?;
        let w2_force = combine_forces(
            w1.lattice(),
            &[(&f1, -(8.0 / 9.0 * eps)), (&f0, 17.0 / 36.0 * eps)],
        )?;
        let w2 = evolve(context, &w1, 1.0, &w2_force)?;

        let f2 = flow_force(&w2, action)?;
        let final_force = combine_forces(
            w2.lattice(),
            &[
                (&f2, -(3.0 / 4.0 * eps)),
                (&f1, 8.0 / 9.0 * eps),
                (&f0, -(17.0 / 36.0 * eps)),
            ],
        )?;
        current = evolve(context, &w2, 1.0, &final_force)?;
    }
    Ok(current)
}

fn flow_force(links: &GaugeLinks, action: &LoopAction) -> Result<TaGaugeField, MeasurementError> {
    let force = loop_action_force(links, action)?;
    for direction in 0..4 {
        for site in 0..force.lattice().nv() {
            if force
                .site_coefficients(direction, site)?
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(MeasurementError::NumericalRange {
                    operation: "gradient_flow force",
                });
            }
        }
    }
    Ok(force)
}

fn combine_forces(
    lattice: LatticeShape4,
    terms: &[(&TaGaugeField, f64)],
) -> Result<TaGaugeField, MeasurementError> {
    if terms
        .iter()
        .any(|(_, coefficient)| !coefficient.is_finite())
    {
        return Err(MeasurementError::NumericalRange {
            operation: "gradient_flow force combination",
        });
    }
    let mut combined = TaGaugeField::zeros(lattice)?;
    for direction in 0..4 {
        for site in 0..lattice.nv() {
            let mut coefficients = [0.0; 8];
            for &(force, coefficient) in terms {
                let values = force.site_coefficients(direction, site)?;
                for (output, value) in coefficients.iter_mut().zip(values) {
                    *output += coefficient * value;
                }
            }
            if coefficients.iter().any(|value| !value.is_finite()) {
                return Err(MeasurementError::NumericalRange {
                    operation: "gradient_flow force combination",
                });
            }
            combined.add_site_coefficients(direction, site, coefficients)?;
        }
    }
    Ok(combined)
}

fn evolve(
    context: &mut CpuEvolutionContext,
    input: &GaugeLinks,
    t: f64,
    force: &TaGaugeField,
) -> Result<GaugeLinks, MeasurementError> {
    let mut output = input.try_clone()?;
    exp_ta_update(context, &mut output, t, force)?;
    Ok(output)
}
