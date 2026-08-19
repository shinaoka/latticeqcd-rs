//! Transactional two-flavor Wilson HMC.
//!
//! The update order follows the pinned LatticeDiracOperators.jl v0.6.4
//! `test/wilsonhmc.jl` at revision `bdef628184597815ba3e0cddf2536df767e78a02`:
//! `calc_action` (lines 37--44), `MDstep!` (46--103), `U_update!`
//! (106--119), `P_update!` (121--132), and `P_update_fermion!`
//! (134--146). Gauge evolution and momentum sampling are reused from the
//! public `gaugefields` crate. Julia's global RNG, temporary pools, assertions,
//! and rollback-by-mutation are not copied. The gauge force and TA projection
//! follow the pinned Gaugefields.jl v0.7.2
//! `src/4D/TA_gaugefields_4D_serial.jl` at revision
//! `9e5719970770f4497405a856315c90bef7f74449`.

use crate::{DiracError, FermionField, SolverReport, WilsonFermiAction};
use gaugefields::{
    exp_ta_update, gauge_force, kinetic_energy, require_su3, sample_momentum, wilson_action,
    CpuEvolutionContext, GaugeLinks, HmcParams as GaugeHmcParams, ReproducibleRng, TaGaugeField,
};
use std::fmt;
use tenferro_tensor::TypedTensor;

/// Validated parameters for one Wilson two-flavor HMC update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WilsonHmcParams {
    gauge: GaugeHmcParams,
    action: WilsonFermiAction,
}

impl WilsonHmcParams {
    /// Construct beta, kappa, leapfrog, boundary, and solver parameters.
    ///
    /// # Errors
    ///
    /// Returns typed beta, step-size, zero-step, kappa, boundary, or solver
    /// validation errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::{FermionBoundary, SolverParams, WilsonHmcParams};
    ///
    /// let params = WilsonHmcParams::new(
    ///     5.7,
    ///     0.13,
    ///     0.01,
    ///     4,
    ///     FermionBoundary::new([1, 1, 1, -1])?,
    ///     SolverParams::new(1.0e-20, 2_000)?,
    /// )?;
    /// assert_eq!(params.beta(), 5.7);
    /// # Ok::<(), dirac_operators::DiracError>(())
    /// ```
    pub fn new(
        beta: f64,
        kappa: f64,
        step_size: f64,
        steps: usize,
        boundary: crate::FermionBoundary,
        solver_params: crate::SolverParams,
    ) -> Result<Self, DiracError> {
        Ok(Self {
            gauge: GaugeHmcParams::new(beta, step_size, steps)?,
            action: WilsonFermiAction::new(kappa, boundary, solver_params)?,
        })
    }

    /// Return beta.
    pub const fn beta(self) -> f64 {
        self.gauge.beta()
    }

    /// Return kappa.
    pub const fn kappa(self) -> f64 {
        self.action.kappa()
    }

    /// Return the positive leapfrog step size.
    pub const fn step_size(self) -> f64 {
        self.gauge.step_size()
    }

    /// Return the number of U-P-U steps.
    pub const fn steps(self) -> usize {
        self.gauge.steps()
    }

    /// Return the explicit Wilson action parameters.
    pub const fn action(self) -> WilsonFermiAction {
        self.action
    }

    /// Return the explicit solver parameters.
    pub const fn solver_params(self) -> crate::SolverParams {
        self.action.solver_params()
    }
}

/// Scalar diagnostics for one Wilson HMC proposal.
pub struct WilsonHmcOutcome {
    /// Whether the private links proposal was committed.
    pub accepted: bool,
    /// Initial gauge, kinetic, and pseudofermion Hamiltonian.
    pub initial_hamiltonian: f64,
    /// Proposed Hamiltonian after the private trajectory.
    pub proposed_hamiltonian: f64,
    /// `proposed_hamiltonian - initial_hamiltonian`.
    pub delta_h: f64,
    /// Branch-stable `min(1, exp(-delta_h))` probability.
    pub acceptance_probability: f64,
    /// Solver report for the initial pseudofermion action.
    pub initial_solver_report: SolverReport,
    /// Solver report for the proposed pseudofermion action.
    pub proposed_solver_report: SolverReport,
}

impl fmt::Debug for WilsonHmcOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WilsonHmcOutcome")
            .field("accepted", &self.accepted)
            .field("initial_hamiltonian", &self.initial_hamiltonian)
            .field("proposed_hamiltonian", &self.proposed_hamiltonian)
            .field("delta_h", &self.delta_h)
            .field("acceptance_probability", &self.acceptance_probability)
            .field("initial_solver_report", &self.initial_solver_report)
            .field("proposed_solver_report", &self.proposed_solver_report)
            .finish()
    }
}

/// Evolve a caller-owned momentum and link state with Wilson U-P-U steps.
///
/// Gauge momentum receives exactly `-dt/NC` times the existing gauge force;
/// the fermion force receives exactly `-dt`, with no additional `1/NC`.
/// Links and momentum are copied into a private proposal and committed only
/// after all force, solve, and evolution calls succeed.
///
/// # Errors
///
/// Returns typed shape, placement, gauge, operator, solver, evolution, or
/// numerical-range errors. Both mutable fields are unchanged on every error.
pub fn wilson_leapfrog_trajectory(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    phi: &FermionField,
    params: WilsonHmcParams,
) -> Result<(), DiracError> {
    validate_state(links, momentum, phi, params.action)?;
    let mut proposed_links = links.try_clone()?;
    let mut proposed_momentum = clone_momentum(momentum)?;
    leapfrog_steps(
        context,
        &mut proposed_links,
        &mut proposed_momentum,
        phi,
        params,
    )?;
    *links = proposed_links;
    *momentum = proposed_momentum;
    Ok(())
}

/// Sample momentum and pseudofermions, make a private Wilson trajectory, and
/// apply one unconditional open-unit Metropolis draw.
///
/// The sampled momentum and pseudofermion are proposal-local values. They are
/// never partially exposed or committed on a reject/error; the caller's links
/// are replaced only after acceptance. RNG words already consumed are never
/// rolled back.
///
/// # Errors
///
/// Returns typed gauge/operator, shape, placement, allocation, solver,
/// evolution, or numerical-range errors. The caller's links remain bitwise
/// unchanged on every error or rejection.
///
/// # Examples
///
/// ```
/// use dirac_operators::{FermionBoundary, SolverParams, WilsonHmcParams, wilson_hmc_update};
/// use gaugefields::{cold_su3, CpuEvolutionContext, LatticeShape4, ReproducibleRng};
/// use tenferro_cpu::CpuBackend;
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let mut links = cold_su3(lattice)?;
/// let mut context = CpuEvolutionContext::new(CpuBackend::new());
/// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let params = WilsonHmcParams::new(
///     5.7, 0.1, 1.0e-4, 1, FermionBoundary::new([1, 1, 1, -1])?,
///     SolverParams::new(1.0e-20, 256)?,
/// )?;
/// let outcome = wilson_hmc_update(&mut context, &mut links, params, &mut rng)?;
/// assert!(outcome.initial_hamiltonian.is_finite());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn wilson_hmc_update(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    params: WilsonHmcParams,
    rng: &mut ReproducibleRng,
) -> Result<WilsonHmcOutcome, DiracError> {
    require_su3(links)?;
    // Validate the complete Wilson boundary before consuming proposal RNG.
    let _ = crate::WilsonDirac::with_boundary(links, params.kappa(), params.action().boundary())?;

    let momentum = sample_momentum(links.lattice(), rng)?;
    let phi = params.action().sample_pseudofermion(links, rng)?;
    validate_state(links, &momentum, &phi, params.action())?;

    let initial_action = params.action().evaluate(links, &phi)?;
    let initial_hamiltonian = hamiltonian(links, &momentum, params.beta(), initial_action.action)?;

    let mut proposed_links = links.try_clone()?;
    let mut proposed_momentum = clone_momentum(&momentum)?;
    leapfrog_steps(
        context,
        &mut proposed_links,
        &mut proposed_momentum,
        &phi,
        params,
    )?;
    let proposed_action = params.action().evaluate(&proposed_links, &phi)?;
    let proposed_hamiltonian = hamiltonian(
        &proposed_links,
        &proposed_momentum,
        params.beta(),
        proposed_action.action,
    )?;
    let delta_h = proposed_hamiltonian - initial_hamiltonian;
    if !delta_h.is_finite() {
        return Err(gaugefields::GaugeError::NonFiniteHamiltonianDelta.into());
    }
    let acceptance_probability = if delta_h <= 0.0 {
        1.0
    } else {
        (-delta_h).exp()
    };
    // This draw is unconditional, including every downhill proposal.
    let draw = rng.open_unit_f64();
    let accepted = draw <= acceptance_probability;
    if accepted {
        *links = proposed_links;
    }
    Ok(WilsonHmcOutcome {
        accepted,
        initial_hamiltonian,
        proposed_hamiltonian,
        delta_h,
        acceptance_probability,
        initial_solver_report: initial_action.solver_report,
        proposed_solver_report: proposed_action.solver_report,
    })
}

fn hamiltonian(
    links: &GaugeLinks,
    momentum: &TaGaugeField,
    beta: f64,
    pseudofermion_action: f64,
) -> Result<f64, DiracError> {
    let gauge = wilson_action(links, beta)?;
    let kinetic = kinetic_energy(momentum)?;
    let value = gauge + kinetic + pseudofermion_action;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(gaugefields::GaugeError::NonFiniteHamiltonian.into())
    }
}

fn validate_state(
    links: &GaugeLinks,
    momentum: &TaGaugeField,
    phi: &FermionField,
    action: WilsonFermiAction,
) -> Result<(), DiracError> {
    require_su3(links)?;
    if links.lattice() != momentum.lattice() {
        return Err(gaugefields::GaugeError::Shape {
            expected: links.lattice().extents().to_vec(),
            found: momentum.lattice().extents().to_vec(),
        }
        .into());
    }
    if phi.lattice() != links.lattice() {
        return Err(DiracError::LatticeMismatch {
            operand: "phi",
            expected: links.lattice(),
            found: phi.lattice(),
        });
    }
    if phi.components() != 4 {
        return Err(DiracError::ComponentsMismatch {
            operand: "phi",
            expected: 4,
            found: phi.components(),
        });
    }
    phi.ensure_finite()?;
    let _ = kinetic_energy(momentum)?;
    let _ = crate::WilsonDirac::with_boundary(links, action.kappa(), action.boundary())?;
    Ok(())
}

fn leapfrog_steps(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    phi: &FermionField,
    params: WilsonHmcParams,
) -> Result<(), DiracError> {
    let dt = params.step_size();
    let half_step = 0.5 * dt;
    let gauge_factor = -dt / links.nc() as f64;
    let fermion_factor = -dt;
    for _ in 0..params.steps() {
        exp_ta_update(context, links, half_step, momentum)?;
        let gauge = gauge_force(links, params.beta())?;
        *momentum = add_scaled(momentum, &gauge, gauge_factor)?;
        let fermion = params.action().force(links, phi)?;
        *momentum = add_scaled(momentum, &fermion.force, fermion_factor)?;
        exp_ta_update(context, links, half_step, momentum)?;
    }
    Ok(())
}

pub(crate) fn add_scaled(
    momentum: &TaGaugeField,
    force: &TaGaugeField,
    factor: f64,
) -> Result<TaGaugeField, DiracError> {
    if momentum.lattice() != force.lattice() {
        return Err(gaugefields::GaugeError::Shape {
            expected: momentum.lattice().extents().to_vec(),
            found: force.lattice().extents().to_vec(),
        }
        .into());
    }
    if !factor.is_finite() {
        return Err(gaugefields::GaugeError::NonFiniteSu3Input {
            operation: "wilson momentum update",
            component: 8,
        }
        .into());
    }
    let tensors = (0..4)
        .map(|mu| {
            let left = momentum.tensors()[mu]
                .host_data()
                .map_err(|source| gaugefields::GaugeError::Tensor(source.to_string()))?;
            let right = force.tensors()[mu]
                .host_data()
                .map_err(|source| gaugefields::GaugeError::Tensor(source.to_string()))?;
            if left.len() != right.len() {
                return Err(gaugefields::GaugeError::Shape {
                    expected: momentum.tensors()[mu].shape().to_vec(),
                    found: force.tensors()[mu].shape().to_vec(),
                });
            }
            let mut values = Vec::with_capacity(left.len());
            for (&p, &f) in left.iter().zip(right) {
                let value = p + factor * f;
                if !p.is_finite() || !f.is_finite() || !value.is_finite() {
                    return Err(gaugefields::GaugeError::NonFiniteMomentum {
                        mu,
                        component: values.len(),
                    });
                }
                values.push(value);
            }
            TypedTensor::from_vec_col_major(momentum.tensors()[mu].shape().to_vec(), values)
                .map_err(|source| gaugefields::GaugeError::Tensor(source.to_string()))
        })
        .collect::<Result<Vec<_>, gaugefields::GaugeError>>()?
        .try_into()
        .map_err(|_| DiracError::Tensor("Wilson momentum requires four tensors".into()))?;
    Ok(TaGaugeField::new(tensors, momentum.lattice())?)
}

pub(crate) fn clone_momentum(momentum: &TaGaugeField) -> Result<TaGaugeField, DiracError> {
    let tensors = (0..4)
        .map(|mu| {
            momentum.tensors()[mu].duplicate().map_err(|source| {
                gaugefields::GaugeError::Evolution {
                    operation: "Wilson momentum duplicate",
                    source,
                }
            })
        })
        .collect::<Result<Vec<_>, gaugefields::GaugeError>>()?
        .try_into()
        .map_err(|_| DiracError::Tensor("Wilson momentum requires four tensors".into()))?;
    Ok(TaGaugeField::new(tensors, momentum.lattice())?)
}
