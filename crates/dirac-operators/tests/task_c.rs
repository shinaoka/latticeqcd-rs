use dirac_operators::{
    FermionBoundary, FermionField, FermionOperator, SolverParams, WilsonDirac, WilsonFermiAction,
    WilsonHmcParams,
};
use gaugefields::{
    cold_su3, sample_momentum, CpuEvolutionContext, GaugeLinks, LatticeShape4, ReproducibleRng,
    TaGaugeField,
};
use num_complex::Complex64;
use rand::RngCore;
use std::error::Error;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::TypedTensor;

#[test]
fn task_c_refresh_action_force_and_hmc_surface_is_present() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let links = cold_su3(lattice)?;
    let solver = SolverParams::new(1.0e-20, 256)?;
    let action = WilsonFermiAction::new(0.1, FermionBoundary::new([1, 1, 1, 1])?, solver)?;
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    let phi = action.sample_pseudofermion(&links, &mut rng)?;
    let evaluated = action.evaluate(&links, &phi)?;
    let force = action.force(&links, &phi)?;
    assert!(evaluated.action.is_finite());
    assert_eq!(force.force.lattice(), lattice);
    assert_eq!(force.x.components(), 4);

    let params = WilsonHmcParams::new(
        5.7,
        0.1,
        1.0e-4,
        1,
        FermionBoundary::new([1, 1, 1, 1])?,
        solver,
    )?;
    let mut links = links;
    let before = link_bits(&links)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let outcome = dirac_operators::wilson_hmc_update(&mut context, &mut links, params, &mut rng)?;
    assert!(outcome.initial_hamiltonian.is_finite());
    assert!(outcome.accepted);
    assert_ne!(link_bits(&links)?, before);
    Ok(())
}

fn action() -> Result<WilsonFermiAction, Box<dyn Error>> {
    Ok(WilsonFermiAction::new(
        0.13,
        FermionBoundary::new([1, 1, 1, -1])?,
        SolverParams::new(1.0e-20, 2_000)?,
    )?)
}

fn hmc_params(step_size: f64, steps: usize) -> Result<WilsonHmcParams, Box<dyn Error>> {
    Ok(WilsonHmcParams::new(
        5.7,
        0.13,
        step_size,
        steps,
        FermionBoundary::new([1, 1, 1, -1])?,
        SolverParams::new(1.0e-20, 2_000)?,
    )?)
}

fn link_bits(links: &GaugeLinks) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut bits = Vec::new();
    for link in links.links() {
        for value in link.typed().host_data()? {
            bits.extend([value.re.to_bits(), value.im.to_bits()]);
        }
    }
    Ok(bits)
}

fn momentum(lattice: LatticeShape4, scale: f64) -> Result<TaGaugeField, Box<dyn Error>> {
    let [nx, ny, nz, nt] = lattice.extents();
    let shape = vec![8, nx, ny, nz, nt];
    let count = 8 * lattice.nv();
    let tensors = (0..4)
        .map(|direction| {
            let values = (0..count)
                .map(|index| scale * (1 + direction + index) as f64)
                .collect();
            Ok(TypedTensor::from_vec_col_major(shape.clone(), values)?)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    Ok(TaGaugeField::new(
        tensors.try_into().map_err(|_| "four momentum tensors")?,
        lattice,
    )?)
}

fn negate_momentum(momentum: &TaGaugeField) -> Result<TaGaugeField, Box<dyn Error>> {
    let tensors = momentum
        .tensors()
        .iter()
        .map(|tensor| {
            let values = tensor.host_data()?.iter().map(|value| -*value).collect();
            Ok(TypedTensor::from_vec_col_major(
                tensor.shape().to_vec(),
                values,
            )?)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    Ok(TaGaugeField::new(
        tensors.try_into().map_err(|_| "four momentum tensors")?,
        momentum.lattice(),
    )?)
}

fn max_momentum_reversal(
    actual: &TaGaugeField,
    initial: &TaGaugeField,
) -> Result<f64, Box<dyn Error>> {
    let mut maximum = 0.0_f64;
    for (actual_tensor, initial_tensor) in actual.tensors().iter().zip(initial.tensors()) {
        for (actual_value, initial_value) in actual_tensor
            .host_data()?
            .iter()
            .zip(initial_tensor.host_data()?)
        {
            maximum = maximum.max((actual_value + initial_value).abs());
        }
    }
    Ok(maximum)
}

fn max_link_difference(left: &GaugeLinks, right: &GaugeLinks) -> Result<f64, Box<dyn Error>> {
    let mut maximum = 0.0_f64;
    for (left_link, right_link) in left.links().iter().zip(right.links()) {
        for (left_value, right_value) in left_link
            .typed()
            .host_data()?
            .iter()
            .zip(right_link.typed().host_data()?)
        {
            maximum = maximum.max((*left_value - *right_value).norm());
        }
    }
    Ok(maximum)
}

#[test]
fn task_c_refresh_uses_dagger_and_action_matches_gaussian_norm() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let links = cold_su3(lattice)?;
    let action = action()?;
    let state = [41, 43, 47, 53];
    let mut sampled_rng = ReproducibleRng::from_state(state)?;
    let phi = action.sample_pseudofermion(&links, &mut sampled_rng)?;
    let mut replay_rng = ReproducibleRng::from_state(state)?;
    let xi = action.sample_xi(lattice, &mut replay_rng)?;
    assert_eq!(sampled_rng.next_u64(), replay_rng.next_u64());

    let dirac = WilsonDirac::with_boundary(&links, action.kappa(), action.boundary())?;
    let mut expected_phi = FermionField::zeros(lattice, 4)?;
    dirac.adjoint().apply_into(&mut expected_phi, &xi)?;
    for site in 0..lattice.nv() {
        for component in 0..4 {
            for color in 0..3 {
                assert_eq!(
                    phi.component(color, component, site)?,
                    expected_phi.component(color, component, site)?
                );
            }
        }
    }
    let evaluated = action.evaluate(&links, &phi)?;
    let residual = (evaluated.action - xi.norm_squared()?).abs();
    eprintln!("refresh action residual={residual:.17e}");
    assert!(residual <= 2.0e-10);
    Ok(())
}

#[test]
fn task_c_rejection_rolls_back_links_and_consumes_one_acceptance_word() -> Result<(), Box<dyn Error>>
{
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let before = link_bits(&links)?;
    let params = hmc_params(0.5, 1)?;
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    let mut replay = rng.clone();
    let _sampled_momentum = sample_momentum(lattice, &mut replay)?;
    let _sampled_xi = params.action().sample_xi(lattice, &mut replay)?;
    let _acceptance_draw = replay.open_unit_f64();
    let expected_next = replay.next_u64();

    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let outcome = dirac_operators::wilson_hmc_update(&mut context, &mut links, params, &mut rng)?;
    assert!(!outcome.accepted);
    assert!(outcome.delta_h > 0.0);
    assert!((0.0..1.0).contains(&outcome.acceptance_probability));
    assert_eq!(link_bits(&links)?, before);
    assert_eq!(rng.next_u64(), expected_next);
    Ok(())
}

#[test]
fn task_c_trajectory_error_rolls_back_both_fields() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let before_links = link_bits(&links)?;
    let mut momentum = momentum(lattice, 1.0e-3)?;
    let before_momentum: Vec<Vec<u64>> = momentum
        .tensors()
        .iter()
        .map(|tensor| {
            tensor
                .host_data()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        })
        .collect();
    let phi = FermionField::zeros(lattice, 4)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let result = dirac_operators::wilson_leapfrog_trajectory(
        &mut context,
        &mut links,
        &mut momentum,
        &phi,
        hmc_params(f64::MAX, 1)?,
    );
    assert!(result.is_err());
    assert_eq!(link_bits(&links)?, before_links);
    for (tensor, expected) in momentum.tensors().iter().zip(before_momentum) {
        let actual: Vec<u64> = tensor
            .host_data()?
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[test]
fn task_c_u_p_u_trajectory_is_reversible() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let initial_links = links.try_clone()?;
    let initial_momentum = momentum(lattice, 1.0e-3)?;
    let mut forward_momentum = momentum(lattice, 1.0e-3)?;
    let phi = FermionField::from_vec_col_major(
        lattice,
        4,
        (0..12)
            .map(|index| Complex64::new(0.01 * (index + 1) as f64, -0.004 * (2 * index + 1) as f64))
            .collect(),
    )?;
    assert!(action()?.force(&links, &phi)?.force.tensors()[0]
        .host_data()?
        .iter()
        .any(|value| *value != 0.0));
    let params = hmc_params(0.01, 2)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    dirac_operators::wilson_leapfrog_trajectory(
        &mut context,
        &mut links,
        &mut forward_momentum,
        &phi,
        params,
    )?;
    let mut reverse_momentum = negate_momentum(&forward_momentum)?;
    dirac_operators::wilson_leapfrog_trajectory(
        &mut context,
        &mut links,
        &mut reverse_momentum,
        &phi,
        params,
    )?;
    let link_residual = max_link_difference(&links, &initial_links)?;
    let momentum_residual = max_momentum_reversal(&reverse_momentum, &initial_momentum)?;
    eprintln!(
        "trajectory reversibility link={link_residual:.17e} momentum={momentum_residual:.17e}"
    );
    assert!(link_residual <= 2.0e-10);
    assert!(momentum_residual <= 2.0e-10);
    Ok(())
}
