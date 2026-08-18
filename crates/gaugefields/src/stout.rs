//! Synchronous host SU(3) stout smearing.
//!
//! The convention follows Gaugefields.jl v0.7.2 at commit
//! `9e5719970770f4497405a856315c90bef7f74449`,
//! `src/smearing/stout_fast.jl` (`forward!` and `calc_C!`), with the positive
//! six-term plaquette staple supplied by the existing host kernel.

use crate::{
    exp_ta_update, force::checked_count, CpuEvolutionContext, GaugeError, GaugeLinks,
    HostGaugeLinks, LatticeShape4, Mat3, TaGaugeField,
};
use num_complex::Complex64;

fn validate_allocation(lattice: LatticeShape4) -> Result<(), GaugeError> {
    checked_count(9, lattice.nv(), std::mem::size_of::<Complex64>())?;
    checked_count(8, lattice.nv(), std::mem::size_of::<f64>())?;
    Ok(())
}

fn validate_finite_input(view: &HostGaugeLinks<'_>) -> Result<(), GaugeError> {
    for direction in 0..4 {
        for site in 0..view.lattice().nv() {
            let matrix = view.link(direction, site)?;
            if let Some(component) = matrix
                .as_array()
                .iter()
                .position(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(GaugeError::NonFiniteSu3Input {
                    operation: "stout_step",
                    component,
                });
            }
        }
    }
    Ok(())
}

/// Applies one synchronous isotropic stout step to a host SU(3) field.
///
/// For every unchanged input link, this computes `C = rho * staple`,
/// `Omega = C * U†`, `Q = TA(Omega)`, and `U' = exp(Q) * U`. The six-term
/// staple is the unweighted positive plaquette sum; `rho` may be negative.
/// The caller owns and reuses `context`, and the input field is never mutated.
///
/// # Errors
///
/// Returns a typed error for non-finite `rho`, invalid or non-host input,
/// allocation overflow, non-finite input or intermediate values, and CPU
/// evolution failures. A failure never changes `links`.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, stout_step, CpuEvolutionContext, LatticeShape4};
/// use tenferro_cpu::CpuBackend;
///
/// let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
/// let mut context = CpuEvolutionContext::new(CpuBackend::new());
/// let smeared = stout_step(&mut context, &links, -0.07)?;
/// assert_eq!(smeared.host_view()?.link(0, 0)?.trace().re, 3.0);
/// # Ok::<(), gaugefields::GaugeError>(())
/// ```
pub fn stout_step(
    context: &mut CpuEvolutionContext,
    links: &GaugeLinks,
    rho: f64,
) -> Result<GaugeLinks, GaugeError> {
    if !rho.is_finite() {
        return Err(GaugeError::NonFiniteRho { found: rho });
    }

    let view = links.host_view()?;
    validate_finite_input(&view)?;
    validate_allocation(view.lattice())?;

    let mut momentum = TaGaugeField::zeros(view.lattice())?;
    for direction in 0..4 {
        for site in 0..view.lattice().nv() {
            let staple = view.force_staple(site, direction)?;
            let omega = staple
                .scaled(Complex64::new(rho, 0.0))
                .mul_adj_right(view.link(direction, site)?);
            let mut coefficients = [0.0; 8];
            Mat3::add_ta_coefficients(&mut coefficients, 1.0, omega);
            if coefficients.iter().any(|value| !value.is_finite()) {
                return Err(GaugeError::Su3NumericalRange {
                    operation: "stout_step",
                    stage: "TA coefficients",
                });
            }
            momentum.add_site_coefficients(direction, site, coefficients)?;
        }
    }

    let mut output = links.try_clone()?;
    exp_ta_update(context, &mut output, 1.0, &momentum)?;
    Ok(output)
}
