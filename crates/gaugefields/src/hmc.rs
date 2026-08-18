//! Public quenched SU(3) HMC primitives.
//!
//! The Hamiltonian and U-P-U convention follows Gaugefields.jl v0.7.2 at
//! commit `9e5719970770f4497405a856315c90bef7f74449`, especially
//! `test/HMC_test_nowing.jl` (`calc_action`, `U_update!`, and `P_update!`).
//! Momentum coefficient order follows `src/TA_Gaugefields.jl` and
//! `src/4D/TA_gaugefields_4D_serial.jl`.

use crate::field::duplicate_links;
use crate::{
    exp_ta_update, gauge_force, require_su3, wilson_action, CpuEvolutionContext, GaugeError,
    GaugeLinks, LatticeShape4, ReproducibleRng, TaGaugeField,
};
use tenferro_tensor::TypedTensor;

/// Parameters for a fixed-step quenched SU(3) HMC proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HmcParams {
    beta: f64,
    step_size: f64,
    steps: usize,
}

impl HmcParams {
    /// Constructs validated HMC parameters.
    ///
    /// # Errors
    ///
    /// Returns [`GaugeError::NonFiniteBeta`] for a non-finite `beta`,
    /// [`GaugeError::NonFiniteStepSize`] or [`GaugeError::NonPositiveStepSize`]
    /// for an invalid `step_size`, and [`GaugeError::ZeroHmcSteps`] when
    /// `steps` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaugefields::{GaugeError, HmcParams};
    ///
    /// let params = HmcParams::new(5.7, 0.01, 4)?;
    /// assert_eq!(params.beta(), 5.7);
    /// assert_eq!(params.step_size(), 0.01);
    /// assert_eq!(params.steps(), 4);
    /// # Ok::<(), GaugeError>(())
    /// ```
    pub fn new(beta: f64, step_size: f64, steps: usize) -> Result<Self, GaugeError> {
        if !beta.is_finite() {
            return Err(GaugeError::NonFiniteBeta { found: beta });
        }
        if !step_size.is_finite() {
            return Err(GaugeError::NonFiniteStepSize { found: step_size });
        }
        if step_size <= 0.0 {
            return Err(GaugeError::NonPositiveStepSize { found: step_size });
        }
        if steps == 0 {
            return Err(GaugeError::ZeroHmcSteps);
        }
        Ok(Self {
            beta,
            step_size,
            steps,
        })
    }

    /// Returns the Wilson action coefficient.
    pub const fn beta(self) -> f64 {
        self.beta
    }

    /// Returns the positive leapfrog step size.
    pub const fn step_size(self) -> f64 {
        self.step_size
    }

    /// Returns the number of U-P-U steps.
    pub const fn steps(self) -> usize {
        self.steps
    }
}

/// Energies and the decision for one HMC proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HmcOutcome {
    /// Whether the private proposal replaced the caller's links.
    pub accepted: bool,
    /// Hamiltonian of the input links and sampled momentum.
    pub initial_hamiltonian: f64,
    /// Hamiltonian after the private leapfrog trajectory.
    pub proposed_hamiltonian: f64,
    /// `proposed_hamiltonian - initial_hamiltonian`.
    pub delta_h: f64,
    /// `min(1, exp(-delta_h))`, with downhill proposals assigned `1`.
    pub acceptance_probability: f64,
}

/// Samples four compact SU(3) TA coefficient fields in direction order.
///
/// Every direction is filled by uncached Box--Muller pairs. The coefficient
/// order is the compact `[8, NX, NY, NZ, NT]` order used by [`TaGaugeField`].
///
/// # Errors
///
/// Returns a typed allocation or tensor error if the requested field cannot
/// be constructed from the lattice.
///
/// # Examples
///
/// ```
/// use gaugefields::{sample_momentum, LatticeShape4, ReproducibleRng};
/// use rand::RngCore;
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let mut replay = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let _momentum = sample_momentum(lattice, &mut replay)?;
/// let expected_next = replay.next_u64();
/// let momentum = sample_momentum(lattice, &mut rng)?;
/// assert!(momentum
///     .tensors()
///     .iter()
///     .all(|tensor| tensor.host_data().unwrap().iter().all(|value| value.is_finite())));
/// assert_eq!(rng.next_u64(), expected_next);
/// # Ok::<(), gaugefields::GaugeError>(())
/// ```
pub fn sample_momentum(
    lattice: LatticeShape4,
    rng: &mut ReproducibleRng,
) -> Result<TaGaugeField, GaugeError> {
    let [nx, ny, nz, nt] = lattice.extents();
    let count = 8usize
        .checked_mul(lattice.nv())
        .ok_or(GaugeError::AllocationOverflow)?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(GaugeError::AllocationOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(GaugeError::AllocationOverflow);
    }
    let shape = vec![8, nx, ny, nz, nt];
    let mut tensors = Vec::with_capacity(4);
    for _ in 0..4 {
        let mut values = vec![0.0; count];
        rng.fill_standard_normals(&mut values);
        tensors.push(
            TypedTensor::from_vec_col_major(shape.clone(), values)
                .map_err(|source| GaugeError::Tensor(source.to_string()))?,
        );
    }
    TaGaugeField::new(
        tensors
            .try_into()
            .map_err(|_| GaugeError::Tensor("HMC momentum requires four tensors".into()))?,
        lattice,
    )
}

/// Returns the coefficient-space kinetic energy `1/2 sum p_a^2`.
///
/// # Errors
///
/// Returns a placement error for a non-host tensor, [`GaugeError::NonFiniteMomentum`]
/// for a non-finite coefficient, or [`GaugeError::KineticNumericalRange`] when
/// squaring or summing leaves the finite `f64` range.
///
/// # Examples
///
/// ```
/// use gaugefields::{kinetic_energy, sample_momentum, LatticeShape4, ReproducibleRng};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let momentum = sample_momentum(lattice, &mut rng)?;
/// let kinetic = kinetic_energy(&momentum)?;
/// assert!(kinetic.is_finite() && kinetic > 0.0);
/// # Ok::<(), gaugefields::GaugeError>(())
/// ```
pub fn kinetic_energy(momentum: &TaGaugeField) -> Result<f64, GaugeError> {
    let mut sum = 0.0;
    for (mu, tensor) in momentum.tensors().iter().enumerate() {
        for (component, &value) in tensor
            .host_data()
            .map_err(|source| GaugeError::placement("kinetic_energy", source))?
            .iter()
            .enumerate()
        {
            if !value.is_finite() {
                return Err(GaugeError::NonFiniteMomentum { mu, component });
            }
            let square = value * value;
            if !square.is_finite() {
                return Err(GaugeError::KineticNumericalRange);
            }
            sum += square;
            if !sum.is_finite() {
                return Err(GaugeError::KineticNumericalRange);
            }
        }
    }
    let kinetic = 0.5 * sum;
    if kinetic.is_finite() {
        Ok(kinetic)
    } else {
        Err(GaugeError::KineticNumericalRange)
    }
}

/// Returns the Wilson Hamiltonian `wilson_action(links, beta) + K(momentum)`.
///
/// # Errors
///
/// Returns a shape, SU(3), beta, placement, momentum, or numerical-range error
/// when either input or the resulting Hamiltonian is invalid.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, hamiltonian, sample_momentum, LatticeShape4, ReproducibleRng};
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let links = cold_su3(lattice)?;
/// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let momentum = sample_momentum(lattice, &mut rng)?;
/// assert!(hamiltonian(&links, &momentum, 5.7)?.is_finite());
/// # Ok::<(), gaugefields::GaugeError>(())
/// ```
pub fn hamiltonian(
    links: &GaugeLinks,
    momentum: &TaGaugeField,
    beta: f64,
) -> Result<f64, GaugeError> {
    if links.lattice() != momentum.lattice() {
        return Err(GaugeError::Shape {
            expected: links.lattice().extents().to_vec(),
            found: momentum.lattice().extents().to_vec(),
        });
    }
    require_su3(links)?;
    let value = wilson_action(links, beta)? + kinetic_energy(momentum)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GaugeError::NonFiniteHamiltonian)
    }
}

/// Evolves links and momentum with the exact Gaugefields.jl U-P-U step.
///
/// The caller's links and momentum are replaced only after every requested
/// step succeeds. Evolution may update the caller-owned context cache, but an
/// error leaves both field arguments bit-for-bit unchanged.
///
/// # Errors
///
/// Returns a shape, SU(3), momentum, allocation, evolution, or numerical-range
/// error. The field arguments remain unchanged when an error is returned.
///
/// # Examples
///
/// ```
/// use gaugefields::{
///     cold_su3, kinetic_energy, leapfrog_trajectory, sample_momentum,
///     CpuEvolutionContext, HmcParams, LatticeShape4, ReproducibleRng,
/// };
/// use tenferro_cpu::CpuBackend;
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let mut links = cold_su3(lattice)?;
/// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let mut momentum = sample_momentum(lattice, &mut rng)?;
/// let mut context = CpuEvolutionContext::new(CpuBackend::new());
/// leapfrog_trajectory(
///     &mut context,
///     &mut links,
///     &mut momentum,
///     HmcParams::new(5.7, 0.01, 1)?,
/// )?;
/// assert!(kinetic_energy(&momentum)?.is_finite());
/// # Ok::<(), gaugefields::GaugeError>(())
/// ```
pub fn leapfrog_trajectory(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    params: HmcParams,
) -> Result<(), GaugeError> {
    leapfrog_trajectory_with(
        context,
        links,
        momentum,
        params,
        &mut |context, links, t, momentum| exp_ta_update(context, links, t, momentum),
    )
}

fn leapfrog_trajectory_with<F>(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    params: HmcParams,
    update: &mut F,
) -> Result<(), GaugeError>
where
    F: FnMut(
        &mut CpuEvolutionContext,
        &mut GaugeLinks,
        f64,
        &TaGaugeField,
    ) -> Result<(), GaugeError>,
{
    validate_state(links, momentum)?;
    let mut proposed_links = duplicate_links(links)?;
    let mut proposed_momentum = clone_momentum(momentum)?;
    leapfrog_steps(
        context,
        &mut proposed_links,
        &mut proposed_momentum,
        params,
        update,
    )?;
    *links = proposed_links;
    *momentum = proposed_momentum;
    Ok(())
}

/// Samples one momentum, evolves one private proposal, and applies Metropolis.
///
/// The RNG is advanced for the sampled momentum and the unconditional
/// acceptance draw on a completed update. RNG advancement is not rolled back:
/// an error leaves already-consumed words consumed, and an error before the
/// Metropolis stage does not consume the acceptance draw. Rejection leaves the
/// caller's links unchanged.
///
/// # Errors
///
/// Returns a typed SU(3), allocation, shape, placement, evolution, or
/// Hamiltonian error. The caller's links are unchanged on every error.
///
/// # Examples
///
/// ```
/// use gaugefields::{
///     cold_su3, hmc_update, CpuEvolutionContext, HmcParams, LatticeShape4,
///     ReproducibleRng,
/// };
/// use tenferro_cpu::CpuBackend;
///
/// let lattice = LatticeShape4::new([1, 1, 1, 1])?;
/// let mut links = cold_su3(lattice)?;
/// let mut context = CpuEvolutionContext::new(CpuBackend::new());
/// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let outcome = hmc_update(
///     &mut context,
///     &mut links,
///     HmcParams::new(5.7, 0.01, 1)?,
///     &mut rng,
/// )?;
/// assert!(outcome.delta_h.is_finite());
/// assert!((0.0..=1.0).contains(&outcome.acceptance_probability));
/// # Ok::<(), gaugefields::GaugeError>(())
/// ```
pub fn hmc_update(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    params: HmcParams,
    rng: &mut ReproducibleRng,
) -> Result<HmcOutcome, GaugeError> {
    require_su3(links)?;
    let mut momentum = sample_momentum(links.lattice(), rng)?;
    let initial_hamiltonian = hamiltonian(links, &momentum, params.beta)?;
    let mut proposed_links = duplicate_links(links)?;
    leapfrog_steps(
        context,
        &mut proposed_links,
        &mut momentum,
        params,
        &mut |context, links, t, momentum| exp_ta_update(context, links, t, momentum),
    )?;
    let proposed_hamiltonian = hamiltonian(&proposed_links, &momentum, params.beta)?;
    let delta_h = proposed_hamiltonian - initial_hamiltonian;
    if !delta_h.is_finite() {
        return Err(GaugeError::NonFiniteHamiltonianDelta);
    }
    let acceptance_probability = if delta_h <= 0.0 {
        1.0
    } else {
        (-delta_h).exp()
    };
    let draw = rng.open_unit_f64();
    let accepted = draw <= acceptance_probability;
    if accepted {
        *links = proposed_links;
    }
    Ok(HmcOutcome {
        accepted,
        initial_hamiltonian,
        proposed_hamiltonian,
        delta_h,
        acceptance_probability,
    })
}

fn validate_state(links: &GaugeLinks, momentum: &TaGaugeField) -> Result<(), GaugeError> {
    if links.lattice() != momentum.lattice() {
        return Err(GaugeError::Shape {
            expected: links.lattice().extents().to_vec(),
            found: momentum.lattice().extents().to_vec(),
        });
    }
    require_su3(links)?;
    validate_momentum(momentum)
}

fn validate_momentum(momentum: &TaGaugeField) -> Result<(), GaugeError> {
    for (mu, tensor) in momentum.tensors().iter().enumerate() {
        for (component, &value) in tensor
            .host_data()
            .map_err(|source| GaugeError::placement("HMC momentum validation", source))?
            .iter()
            .enumerate()
        {
            if !value.is_finite() {
                return Err(GaugeError::NonFiniteMomentum { mu, component });
            }
        }
    }
    Ok(())
}

fn leapfrog_steps<F>(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    params: HmcParams,
    update: &mut F,
) -> Result<(), GaugeError>
where
    F: FnMut(
        &mut CpuEvolutionContext,
        &mut GaugeLinks,
        f64,
        &TaGaugeField,
    ) -> Result<(), GaugeError>,
{
    let half_step = 0.5 * params.step_size;
    let momentum_factor = -params.step_size / 3.0;
    for _ in 0..params.steps {
        update(context, links, half_step, momentum)?;
        let force = gauge_force(links, params.beta)?;
        *momentum = add_scaled(momentum, &force, momentum_factor)?;
        update(context, links, half_step, momentum)?;
    }
    Ok(())
}

fn add_scaled(
    momentum: &TaGaugeField,
    force: &TaGaugeField,
    factor: f64,
) -> Result<TaGaugeField, GaugeError> {
    if momentum.lattice() != force.lattice() {
        return Err(GaugeError::Shape {
            expected: momentum.lattice().extents().to_vec(),
            found: force.lattice().extents().to_vec(),
        });
    }
    let tensors = (0..4)
        .map(|mu| {
            let lhs = momentum.tensors()[mu]
                .host_data()
                .map_err(|source| GaugeError::placement("HMC momentum update", source))?;
            let rhs = force.tensors()[mu]
                .host_data()
                .map_err(|source| GaugeError::placement("HMC force update", source))?;
            let mut values = Vec::with_capacity(lhs.len());
            for (component, (&p, &f)) in lhs.iter().zip(rhs).enumerate() {
                if !p.is_finite() || !f.is_finite() {
                    return Err(GaugeError::NonFiniteMomentum { mu, component });
                }
                let value = p + factor * f;
                if !value.is_finite() {
                    return Err(GaugeError::NonFiniteMomentum { mu, component });
                }
                values.push(value);
            }
            TypedTensor::from_vec_col_major(momentum.tensors()[mu].shape().to_vec(), values)
                .map_err(|source| GaugeError::Tensor(source.to_string()))
        })
        .collect::<Result<Vec<_>, GaugeError>>()?
        .try_into()
        .map_err(|_| GaugeError::Tensor("HMC momentum requires four tensors".into()))?;
    TaGaugeField::new(tensors, momentum.lattice())
}

fn clone_momentum(momentum: &TaGaugeField) -> Result<TaGaugeField, GaugeError> {
    let tensors = (0..4)
        .map(|mu| {
            momentum.tensors()[mu]
                .duplicate()
                .map_err(|source| GaugeError::Evolution {
                    operation: "HMC momentum duplicate",
                    source,
                })
        })
        .collect::<Result<Vec<_>, GaugeError>>()?
        .try_into()
        .map_err(|_| GaugeError::Tensor("HMC momentum requires four tensors".into()))?;
    TaGaugeField::new(tensors, momentum.lattice())
}

#[cfg(test)]
mod tests;
