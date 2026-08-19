//! Host-side Polyakov and clover measurements for `gaugefields`.
//!
//! The Polyakov convention follows the MIT-licensed Gaugefields.jl
//! `calculate_Polyakov_loop` implementation at revision
//! `9e5719970770f4497405a856315c90bef7f74449`. The four fixed clover paths and
//! charge convention follow the MIT-licensed QCDMeasurements.jl
//! `measure_topological_charge.jl` implementation at revision
//! `9e04c37bbd68712cf7a749ae5aff10eb6aae4566`, with Wilsonloop.jl's periodic
//! signed-link convention. The implementation reuses `HostGaugeLinks` and
//! `Mat3`; it does not copy tensor storage or introduce a second path engine.
//!
//! With the optional `fermions` feature, [`fermions::pion_correlator`] solves
//! every color-outer, component-inner point source and accumulates the corrected
//! full sink Frobenius contraction. [`fermions::stochastic_chiral_condensate`]
//! consumes a caller-owned [`gaugefields::ReproducibleRng`] using canonical Z4
//! phases (`word & 3` maps to `1, i, -1, -i`). These are intentionally not the
//! pinned high-level QCDMeasurements.jl pion reconstruction (Issue #29) or its
//! `pi/4` Z4 routine (Issue #27). The Phase 4 ensemble evidence also avoids the
//! pinned LatticeQCD.jl parser/scheduler issue candidates (Issue #30) by owning
//! its fixed schedule in the Julia generator and Rust integration test. The
//! pinned revisions and exact normalization are recorded in the fixture metadata.

#[cfg(feature = "fermions")]
pub mod fermions;
mod gradient_flow;

use gaugefields::{site_index, GaugeLinks, HostGaugeLinks, Mat3};
use num_complex::Complex64;
use std::f64::consts::PI;

pub use gradient_flow::{gradient_flow, GradientFlowParams};

/// Measurement validation and numerical failures.
#[derive(Debug, thiserror::Error)]
pub enum MeasurementError {
    #[error(transparent)]
    Gauge(#[from] gaugefields::GaugeError),
    #[error(transparent)]
    Wilson(#[from] wilsonloop::WilsonError),
    #[error("gradient-flow step size must be finite, found {found}")]
    NonFiniteStepSize { found: f64 },
    #[error("gradient-flow step size must be positive, found {found}")]
    NonPositiveStepSize { found: f64 },
    #[error("gradient flow requires at least one step")]
    ZeroFlowSteps,
    #[error(
        "measurement input at direction {direction}, site {site}, matrix component {component} is non-finite"
    )]
    NonFiniteInput {
        direction: usize,
        site: usize,
        component: usize,
    },
    #[error("{operation} exceeded finite numerical range")]
    NumericalRange { operation: &'static str },
}

/// Computes the temporal Polyakov loop.
///
/// The temporal axis is 3. The result is
/// `sum_xyz tr(product_t U_3(x,y,z,t)) / (NX * NY * NZ)` and is not divided by
/// the number of colors.
///
/// # Errors
///
/// Returns a typed gauge, placement, non-finite-input, or numerical-range
/// error.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, LatticeShape4};
/// use measurements::polyakov_loop;
///
/// let links = cold_su3(LatticeShape4::new([2, 3, 2, 4])?)?;
/// assert_eq!(polyakov_loop(&links)?, num_complex::Complex64::new(3.0, 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn polyakov_loop(links: &GaugeLinks) -> Result<Complex64, MeasurementError> {
    let view = validated_view(links)?;
    let [nx, ny, nz, nt] = view.lattice().extents();
    let spatial_volume = nx
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nz))
        .ok_or(gaugefields::GaugeError::VolumeOverflow)?;

    let mut sum = Complex64::default();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut site = site_index([x, y, z, 0], view.lattice())?;
                let mut product = Mat3::identity();
                for _ in 0..nt {
                    product = product.mul(view.link(3, site)?);
                    if !finite_matrix(&product) {
                        return Err(MeasurementError::NumericalRange {
                            operation: "polyakov_loop",
                        });
                    }
                    site = view.shifted_site(site, 3, 1)?;
                }
                sum += product.trace();
                if !sum.re.is_finite() || !sum.im.is_finite() {
                    return Err(MeasurementError::NumericalRange {
                        operation: "polyakov_loop",
                    });
                }
            }
        }
    }

    let result = sum / spatial_volume as f64;
    if result.re.is_finite() && result.im.is_finite() {
        Ok(result)
    } else {
        Err(MeasurementError::NumericalRange {
            operation: "polyakov_loop",
        })
    }
}

/// Computes the basic clover topological charge.
///
/// For every ordered `mu != nu`, the clover matrix is the traceless
/// anti-Hermitian projection of the four oriented plaquettes around the site.
/// With the ordinary epsilon tensor (`epsilon(0,1,2,3) = +1`), this uses
/// `-1/(32*pi^2)` and the pinned `/4^2` clover normalization. No improved
/// plaquette or rectangle term is included.
///
/// # Errors
///
/// Returns a typed gauge, placement, non-finite-input, or numerical-range
/// error.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, LatticeShape4};
/// use measurements::clover_topological_charge;
///
/// let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
/// assert_eq!(clover_topological_charge(&links)?, 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn clover_topological_charge(links: &GaugeLinks) -> Result<f64, MeasurementError> {
    let view = validated_view(links)?;
    let mut total = 0.0_f64;

    for site in 0..view.lattice().nv() {
        let mut clover = [Mat3::zero(); 16];
        for mu in 0..4 {
            for nu in 0..4 {
                if mu != nu {
                    clover[4 * mu + nu] = clover_matrix(&view, site, mu, nu)?;
                }
            }
        }

        for mu in 0..4 {
            for nu in 0..4 {
                for rho in 0..4 {
                    for sigma in 0..4 {
                        let epsilon = epsilon4([mu, nu, rho, sigma]);
                        if epsilon == 0 {
                            continue;
                        }
                        total += f64::from(epsilon)
                            * clover[4 * mu + nu].real_trace_mul(clover[4 * rho + sigma])
                            / 16.0;
                        if !total.is_finite() {
                            return Err(MeasurementError::NumericalRange {
                                operation: "clover_topological_charge",
                            });
                        }
                    }
                }
            }
        }
    }

    let result = -total / (32.0 * PI * PI);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(MeasurementError::NumericalRange {
            operation: "clover_topological_charge",
        })
    }
}

fn validated_view(links: &GaugeLinks) -> Result<HostGaugeLinks<'_>, MeasurementError> {
    let view = links.host_view()?;
    for direction in 0..4 {
        for site in 0..view.lattice().nv() {
            for (component, value) in view.link(direction, site)?.as_array().iter().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(MeasurementError::NonFiniteInput {
                        direction,
                        site,
                        component,
                    });
                }
            }
        }
    }
    Ok(view)
}

fn finite_matrix(matrix: &Mat3) -> bool {
    matrix
        .as_array()
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

fn clover_matrix(
    view: &HostGaugeLinks<'_>,
    origin: usize,
    mu: usize,
    nu: usize,
) -> Result<Mat3, MeasurementError> {
    let plus_mu = view.shifted_site(origin, mu, 1)?;
    let plus_nu = view.shifted_site(origin, nu, 1)?;
    let minus_mu = view.shifted_site(origin, mu, -1)?;
    let minus_nu = view.shifted_site(origin, nu, -1)?;
    let plus_nu_minus_mu = view.shifted_site(plus_nu, mu, -1)?;
    let minus_mu_minus_nu = view.shifted_site(minus_mu, nu, -1)?;
    let minus_nu_plus_mu = view.shifted_site(minus_nu, mu, 1)?;

    let right_top = view
        .link(mu, origin)?
        .mul(view.link(nu, plus_mu)?)
        .mul(view.link(mu, plus_nu)?.adjoint())
        .mul(view.link(nu, origin)?.adjoint());
    let left_top = view
        .link(nu, origin)?
        .mul(view.link(mu, plus_nu_minus_mu)?.adjoint())
        .mul(view.link(nu, minus_mu)?.adjoint())
        .mul(view.link(mu, minus_mu)?);
    let right_bottom = view
        .link(nu, minus_nu)?
        .adjoint()
        .mul(view.link(mu, minus_nu)?)
        .mul(view.link(nu, minus_nu_plus_mu)?)
        .mul(view.link(mu, origin)?.adjoint());
    let left_bottom = view
        .link(mu, minus_mu)?
        .adjoint()
        .mul(view.link(nu, minus_mu_minus_nu)?.adjoint())
        .mul(view.link(mu, minus_mu_minus_nu)?)
        .mul(view.link(nu, minus_nu)?);

    let mut sum = right_top;
    sum.add_scaled_real(1.0, left_top);
    sum.add_scaled_real(1.0, right_bottom);
    sum.add_scaled_real(1.0, left_bottom);
    if !finite_matrix(&sum) {
        return Err(MeasurementError::NumericalRange {
            operation: "clover_topological_charge",
        });
    }
    let result = sum.ta();
    if finite_matrix(&result) {
        Ok(result)
    } else {
        Err(MeasurementError::NumericalRange {
            operation: "clover_topological_charge",
        })
    }
}

fn epsilon4(indices: [usize; 4]) -> i8 {
    if indices.iter().any(|&index| index >= 4) {
        return 0;
    }
    for left in 0..4 {
        if indices[left + 1..].contains(&indices[left]) {
            return 0;
        }
    }
    let mut inversions = 0;
    for left in 0..4 {
        for right in (left + 1)..4 {
            inversions += usize::from(indices[left] > indices[right]);
        }
    }
    if inversions.is_multiple_of(2) {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::epsilon4;

    #[test]
    fn epsilon_has_ordinary_signs_and_zero_repeats() {
        assert_eq!(epsilon4([0, 1, 2, 3]), 1);
        assert_eq!(epsilon4([1, 0, 2, 3]), -1);
        assert_eq!(epsilon4([0, 2, 1, 3]), -1);
        assert_eq!(epsilon4([3, 2, 1, 0]), 1);
        assert_eq!(epsilon4([0, 0, 1, 2]), 0);
        assert_eq!(epsilon4([0, 1, 2, 2]), 0);
        assert_eq!(epsilon4([0, 1, 4, 3]), 0);
    }
}
