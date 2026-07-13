use crate::{
    exp_ta_update, gauge_force, wilson_action, CpuEvolutionContext, GaugeError, GaugeLinkTensor,
    GaugeLinks, LatticeShape4, TaGaugeField,
};
use rand::Rng;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::TypedTensor;

pub(crate) fn random_momentum(
    lattice: LatticeShape4,
    rng: &mut impl Rng,
    scale: f64,
) -> Result<TaGaugeField, GaugeError> {
    let [nx, ny, nz, nt] = lattice.extents();
    let tensors = std::array::from_fn(|_| {
        let values = (0..8 * lattice.nv())
            .map(|_| scale * (2.0 * rng.gen::<f64>() - 1.0))
            .collect();
        TypedTensor::from_vec_col_major(vec![8, nx, ny, nz, nt], values).unwrap()
    });
    TaGaugeField::new(tensors, lattice)
}

pub(crate) fn clone_links(links: &GaugeLinks) -> Result<GaugeLinks, GaugeError> {
    let lattice = links.lattice();
    let copies = std::array::from_fn(|mu| {
        GaugeLinkTensor::from_typed(links.links()[mu].typed().clone(), lattice).unwrap()
    });
    GaugeLinks::new(copies)
}

pub(crate) fn negate(momentum: &TaGaugeField) -> Result<TaGaugeField, GaugeError> {
    scaled_add(momentum, None, -1.0)
}

fn scaled_add(
    momentum: &TaGaugeField,
    force: Option<&TaGaugeField>,
    force_factor: f64,
) -> Result<TaGaugeField, GaugeError> {
    let lattice = momentum.lattice();
    let tensors = std::array::from_fn(|mu| {
        let p = momentum.tensors()[mu].host_data().unwrap();
        let values = match force {
            Some(force) => p
                .iter()
                .zip(force.tensors()[mu].host_data().unwrap())
                .map(|(p, f)| p + force_factor * f)
                .collect(),
            None => p.iter().map(|p| force_factor * p).collect(),
        };
        TypedTensor::from_vec_col_major(momentum.tensors()[mu].shape().to_vec(), values).unwrap()
    });
    TaGaugeField::new(tensors, lattice)
}

pub(crate) fn leapfrog_step(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    beta: f64,
    dt: f64,
) -> Result<(), GaugeError> {
    exp_ta_update(context, links, 0.5 * dt, momentum)?;
    let force = gauge_force(links, beta)?;
    *momentum = scaled_add(momentum, Some(&force), -dt / links.nc() as f64)?;
    exp_ta_update(context, links, 0.5 * dt, momentum)
}

pub(crate) fn trajectory(
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    beta: f64,
    dt: f64,
    steps: usize,
) -> Result<(), GaugeError> {
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    for _ in 0..steps {
        leapfrog_step(&mut context, links, momentum, beta, dt)?;
    }
    Ok(())
}

pub(crate) fn kinetic(momentum: &TaGaugeField) -> f64 {
    // `P=(i/2) sum p_a lambda_a` and `tr(lambda_a lambda_b)=2 delta_ab`;
    // the coefficient-space Gaussian convention is `K=1/2 sum p_a^2`.
    0.5 * momentum
        .tensors()
        .iter()
        .flat_map(|tensor| tensor.host_data().unwrap())
        .map(|value| value * value)
        .sum::<f64>()
}

pub(crate) fn hamiltonian(
    links: &GaugeLinks,
    momentum: &TaGaugeField,
    beta: f64,
) -> Result<f64, GaugeError> {
    Ok(wilson_action(links, beta)? + kinetic(momentum))
}

#[cfg(test)]
mod tests;
