use gaugefields::{
    cold_su3, exp_ta_update, hamiltonian, hmc_update, kinetic_energy, leapfrog_trajectory,
    CpuEvolutionContext, GaugeError, GaugeLinkTensor, GaugeLinks, HmcOutcome, HmcParams,
    LatticeShape4, Mat3, ReproducibleRng, TaGaugeField,
};
use num_complex::Complex64;
use rand::RngCore;
use serde_json::Value;
use std::{fs, path::Path};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::TypedTensor;

const JULIA_COMMIT: &str = "9e5719970770f4497405a856315c90bef7f74449";

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/hmc_trajectory")
}

fn fixture_metadata() -> Value {
    serde_json::from_slice(&fs::read(fixture_dir().join("metadata.json")).unwrap()).unwrap()
}

fn fixture_state(metadata: &Value) -> [u64; 4] {
    metadata["initial_rng_state"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn read_f64(name: &str, shape: &[u64]) -> Vec<f64> {
    let bytes = fs::read(fixture_dir().join(name)).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(npy.order(), npyz::Order::Fortran);
    assert_eq!(npy.shape(), shape);
    npy.into_vec::<f64>().unwrap()
}

fn read_c64(name: &str, shape: &[u64]) -> Vec<Complex64> {
    let bytes = fs::read(fixture_dir().join(name)).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(npy.order(), npyz::Order::Fortran);
    assert_eq!(npy.shape(), shape);
    npy.into_vec::<Complex64>().unwrap()
}

fn assert_f64_close(actual: &[f64], expected: &[f64], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let mut max_residual = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite(), "{label}: non-finite value");
        max_residual = max_residual.max((a - b).abs());
    }
    assert!(
        max_residual <= tolerance,
        "{label}: max residual {max_residual:e} > {tolerance:e}"
    );
}

fn assert_c64_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let mut max_residual = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite(),
            "{label}: non-finite value"
        );
        max_residual = max_residual.max((a - b).norm());
    }
    assert!(
        max_residual <= tolerance,
        "{label}: max residual {max_residual:e} > {tolerance:e}"
    );
}

fn constant_momentum(lattice: LatticeShape4, value: f64) -> TaGaugeField {
    let [nx, ny, nz, nt] = lattice.extents();
    let tensors = std::array::from_fn(|_| {
        TypedTensor::from_vec_col_major(vec![8, nx, ny, nz, nt], vec![value; 8 * lattice.nv()])
            .unwrap()
    });
    TaGaugeField::new(tensors, lattice).unwrap()
}

fn scale_momentum(momentum: &TaGaugeField, scale: f64) -> TaGaugeField {
    let tensors = std::array::from_fn(|mu| {
        let values = momentum.tensors()[mu]
            .host_data()
            .unwrap()
            .iter()
            .map(|value| scale * value)
            .collect();
        TypedTensor::from_vec_col_major(momentum.tensors()[mu].shape().to_vec(), values).unwrap()
    });
    TaGaugeField::new(tensors, momentum.lattice()).unwrap()
}

fn negate_momentum(momentum: &TaGaugeField) -> TaGaugeField {
    scale_momentum(momentum, -1.0)
}

fn clone_links(links: &GaugeLinks) -> GaugeLinks {
    let lattice = links.lattice();
    let copies = std::array::from_fn(|mu| {
        GaugeLinkTensor::from_typed(links.links()[mu].typed().duplicate().unwrap(), lattice)
            .unwrap()
    });
    GaugeLinks::new(copies).unwrap()
}

fn identity_links(lattice: LatticeShape4, nc: usize) -> GaugeLinks {
    let [nx, ny, nz, nt] = lattice.extents();
    let mut values = vec![Complex64::default(); nc * nc * lattice.nv()];
    for block in values.chunks_exact_mut(nc * nc) {
        for diagonal in 0..nc {
            block[diagonal + nc * diagonal] = Complex64::new(1.0, 0.0);
        }
    }
    let make = || {
        GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(vec![nc, nc, nx, ny, nz, nt], values.clone()).unwrap(),
            lattice,
        )
        .unwrap()
    };
    GaugeLinks::new([make(), make(), make(), make()]).unwrap()
}

fn link_residual(lhs: &GaugeLinks, rhs: &GaugeLinks) -> f64 {
    (0..4)
        .flat_map(|mu| {
            lhs.links()[mu]
                .typed()
                .host_data()
                .unwrap()
                .iter()
                .zip(rhs.links()[mu].typed().host_data().unwrap())
                .map(|(a, b)| (*a - *b).norm())
        })
        .fold(0.0, f64::max)
}

fn momentum_residual(lhs: &TaGaugeField, rhs: &TaGaugeField) -> f64 {
    lhs.tensors()
        .iter()
        .zip(rhs.tensors())
        .flat_map(|(a, b)| {
            a.host_data()
                .unwrap()
                .iter()
                .zip(b.host_data().unwrap())
                .map(|(a, b)| (a - b).abs())
        })
        .fold(0.0, f64::max)
}

fn assert_links_bitwise_equal(actual: &GaugeLinks, expected: &GaugeLinks) {
    for mu in 0..4 {
        for (actual, expected) in actual.links()[mu]
            .typed()
            .host_data()
            .unwrap()
            .iter()
            .zip(expected.links()[mu].typed().host_data().unwrap())
        {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits(), "mu={mu} real");
            assert_eq!(actual.im.to_bits(), expected.im.to_bits(), "mu={mu} imag");
        }
    }
}

fn assert_momentum_bitwise_equal(actual: &TaGaugeField, expected: &TaGaugeField) {
    for mu in 0..4 {
        for (actual, expected) in actual.tensors()[mu]
            .host_data()
            .unwrap()
            .iter()
            .zip(expected.tensors()[mu].host_data().unwrap())
        {
            assert_eq!(actual.to_bits(), expected.to_bits(), "mu={mu}");
        }
    }
}

fn su3_drift(links: &GaugeLinks) -> (f64, f64) {
    let mut unitary = 0.0_f64;
    let mut determinant = 0.0_f64;
    for link in links.links() {
        for block in link.typed().host_data().unwrap().chunks_exact(9) {
            let matrix = Mat3::load(block, 0).unwrap();
            let product = matrix.adjoint().mul(matrix);
            for column in 0..3 {
                for row in 0..3 {
                    let expected = if row == column { 1.0 } else { 0.0 };
                    unitary = unitary.max((product[(row, column)] - expected).norm());
                }
            }
            let a = matrix.as_array();
            let det = a[0] * (a[4] * a[8] - a[7] * a[5]) - a[3] * (a[1] * a[8] - a[7] * a[2])
                + a[6] * (a[1] * a[5] - a[4] * a[2]);
            determinant = determinant.max((det - Complex64::new(1.0, 0.0)).norm());
        }
    }
    (unitary, determinant)
}

#[test]
fn params_and_public_debug_are_typed_and_compact() {
    assert!(matches!(
        HmcParams::new(f64::NAN, 0.1, 1),
        Err(GaugeError::NonFiniteBeta { .. })
    ));
    assert!(matches!(
        HmcParams::new(5.7, f64::NAN, 1),
        Err(GaugeError::NonFiniteStepSize { .. })
    ));
    assert!(matches!(
        HmcParams::new(5.7, 0.0, 1),
        Err(GaugeError::NonPositiveStepSize { .. })
    ));
    assert!(matches!(
        HmcParams::new(5.7, -0.1, 1),
        Err(GaugeError::NonPositiveStepSize { .. })
    ));
    assert!(matches!(
        HmcParams::new(5.7, 0.1, 0),
        Err(GaugeError::ZeroHmcSteps)
    ));

    let params = HmcParams::new(5.7, 0.01, 2).unwrap();
    assert_eq!(params.beta(), 5.7);
    assert_eq!(params.step_size(), 0.01);
    assert_eq!(params.steps(), 2);
    assert!(!format!("{params:?}").contains("GaugeLinks"));
    assert!(!format!("{params:?}").contains("ReproducibleRng"));
    let outcome = HmcOutcome {
        accepted: true,
        initial_hamiltonian: 1.0,
        proposed_hamiltonian: 1.0,
        delta_h: 0.0,
        acceptance_probability: 1.0,
    };
    assert!(!format!("{outcome:?}").contains("state"));
}

#[test]
fn sample_momentum_matches_julia_and_consumes_exactly_four_times_eight_times_nv_words() {
    let metadata = fixture_metadata();
    assert_eq!(metadata["gaugefields_jl_version"], "0.7.2");
    assert_eq!(metadata["gaugefields_jl_commit"], JULIA_COMMIT);
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    let state = fixture_state(&metadata);
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let mut rng = ReproducibleRng::from_state(state).unwrap();
    let momentum = gaugefields::sample_momentum(lattice, &mut rng).unwrap();
    let tolerance = metadata["comparison_tolerance"].as_f64().unwrap();
    for mu in 0..4 {
        let expected = read_f64(&format!("p_initial{mu}.npy"), &[8, 2, 2, 2, 2]);
        let actual = momentum.tensors()[mu].host_data().unwrap();
        assert_f64_close(actual, &expected, tolerance, &format!("p_initial mu={mu}"));
    }
    assert!(momentum.tensors().iter().all(|tensor| tensor
        .host_data()
        .unwrap()
        .iter()
        .all(|value| value.is_finite())));

    let mut replay = ReproducibleRng::from_state(state).unwrap();
    let _ = gaugefields::sample_momentum(lattice, &mut replay).unwrap();
    let uniform = replay.open_unit_f64();
    assert_eq!(
        uniform.to_bits(),
        metadata["acceptance_uniform_bits"].as_u64().unwrap()
    );
}

#[test]
fn kinetic_and_hamiltonian_use_coefficient_normalization() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let momentum = constant_momentum(lattice, 1.0);
    assert_eq!(kinetic_energy(&momentum)?, 16.0);
    let links = cold_su3(lattice)?;
    assert_eq!(hamiltonian(&links, &momentum, 6.0)?, -20.0);
    Ok(())
}

#[test]
fn sample_momentum_checks_allocation_overflow_before_allocating() -> Result<(), GaugeError> {
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    for nx in [
        usize::MAX,
        isize::MAX as usize / (8 * std::mem::size_of::<f64>()) + 1,
    ] {
        let lattice = LatticeShape4::new([nx, 1, 1, 1])?;
        assert!(matches!(
            gaugefields::sample_momentum(lattice, &mut rng),
            Err(GaugeError::AllocationOverflow)
        ));
    }
    Ok(())
}

#[test]
fn hmc_update_rejects_non_su3_before_consuming_rng() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut links = identity_links(lattice, 2);
    let params = HmcParams::new(5.7, 0.01, 1)?;
    let state = [41, 43, 47, 53];
    let mut rng = ReproducibleRng::from_state(state)?;
    let mut replay = ReproducibleRng::from_state(state)?;
    assert!(matches!(
        hmc_update(
            &mut CpuEvolutionContext::new(CpuBackend::new()),
            &mut links,
            params,
            &mut rng,
        ),
        Err(GaugeError::UnsupportedNc { found: 2 })
    ));
    assert_eq!(rng.next_u64(), replay.next_u64());
    Ok(())
}

#[test]
fn hmc_error_is_transactional_and_keeps_completed_rng_draws() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let before = clone_links(&links);
    let params = HmcParams::new(f64::MAX, 0.01, 1)?;
    let state = [59, 61, 67, 71];
    let mut rng = ReproducibleRng::from_state(state)?;
    let mut replay = ReproducibleRng::from_state(state)?;
    assert!(matches!(
        hmc_update(
            &mut CpuEvolutionContext::new(CpuBackend::new()),
            &mut links,
            params,
            &mut rng,
        ),
        Err(GaugeError::NonFiniteHamiltonian)
    ));
    assert_links_bitwise_equal(&links, &before);
    let _ = gaugefields::sample_momentum(lattice, &mut replay)?;
    assert_eq!(rng.next_u64(), replay.next_u64());
    Ok(())
}

#[test]
fn trajectory_numerical_overflow_is_transactional() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let before_links = clone_links(&links);
    let mut momentum = constant_momentum(lattice, f64::MAX);
    let before_momentum = scale_momentum(&momentum, 1.0);
    let result = leapfrog_trajectory(
        &mut CpuEvolutionContext::new(CpuBackend::new()),
        &mut links,
        &mut momentum,
        HmcParams::new(5.7, 0.01, 1)?,
    );
    assert!(matches!(result, Err(GaugeError::Su3NumericalRange { .. })));
    assert_links_bitwise_equal(&links, &before_links);
    assert_momentum_bitwise_equal(&momentum, &before_momentum);
    Ok(())
}

#[test]
fn julia_trajectory_matches_momentum_links_energies_decision_and_rng_position() {
    let metadata = fixture_metadata();
    let state = fixture_state(&metadata);
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let params = HmcParams::new(
        metadata["beta"].as_f64().unwrap(),
        metadata["step_size"].as_f64().unwrap(),
        metadata["steps"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let tolerance = metadata["comparison_tolerance"].as_f64().unwrap();
    let hamiltonian_tolerance = metadata["hamiltonian_tolerance"].as_f64().unwrap();

    let mut rng = ReproducibleRng::from_state(state).unwrap();
    let initial_momentum = gaugefields::sample_momentum(lattice, &mut rng).unwrap();
    let mut momentum = scale_momentum(&initial_momentum, 1.0);
    let mut links = cold_su3(lattice).unwrap();
    let initial_links = clone_links(&links);
    let initial_hamiltonian = hamiltonian(&links, &momentum, params.beta()).unwrap();
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    leapfrog_trajectory(&mut context, &mut links, &mut momentum, params).unwrap();
    let proposed_hamiltonian = hamiltonian(&links, &momentum, params.beta()).unwrap();
    let delta_h = proposed_hamiltonian - initial_hamiltonian;
    let acceptance_probability = if delta_h <= 0.0 {
        1.0
    } else {
        (-delta_h).exp()
    };
    let uniform = rng.open_unit_f64();

    assert_f64_close(
        &[initial_hamiltonian],
        &[metadata["initial_hamiltonian"].as_f64().unwrap()],
        hamiltonian_tolerance,
        "H_initial",
    );
    assert_f64_close(
        &[proposed_hamiltonian],
        &[metadata["proposed_hamiltonian"].as_f64().unwrap()],
        hamiltonian_tolerance,
        "H_proposed",
    );
    assert_f64_close(
        &[delta_h],
        &[metadata["delta_h"].as_f64().unwrap()],
        hamiltonian_tolerance,
        "delta_h",
    );
    assert_f64_close(
        &[acceptance_probability],
        &[metadata["acceptance_probability"].as_f64().unwrap()],
        hamiltonian_tolerance,
        "acceptance probability",
    );
    assert_eq!(
        uniform.to_bits(),
        metadata["acceptance_uniform_bits"].as_u64().unwrap()
    );

    for mu in 0..4 {
        let expected_initial = read_f64(&format!("p_initial{mu}.npy"), &[8, 2, 2, 2, 2]);
        assert_f64_close(
            initial_momentum.tensors()[mu].host_data().unwrap(),
            &expected_initial,
            tolerance,
            &format!("trajectory p_initial mu={mu}"),
        );
        let expected_final = read_f64(&format!("p_final{mu}.npy"), &[8, 2, 2, 2, 2]);
        assert_f64_close(
            momentum.tensors()[mu].host_data().unwrap(),
            &expected_final,
            tolerance,
            &format!("trajectory p_final mu={mu}"),
        );
        let expected_link = read_c64(&format!("u_proposed{mu}.npy"), &[3, 3, 2, 2, 2, 2]);
        assert_c64_close(
            links.links()[mu].typed().host_data().unwrap(),
            &expected_link,
            tolerance,
            &format!("trajectory U mu={mu}"),
        );
    }

    let mut hmc_links = cold_su3(lattice).unwrap();
    let mut hmc_rng = ReproducibleRng::from_state(state).unwrap();
    let mut hmc_context = CpuEvolutionContext::new(CpuBackend::new());
    let outcome = hmc_update(&mut hmc_context, &mut hmc_links, params, &mut hmc_rng).unwrap();
    assert_eq!(outcome.accepted, metadata["accepted"].as_bool().unwrap());
    assert_f64_close(
        &[outcome.initial_hamiltonian],
        &[initial_hamiltonian],
        hamiltonian_tolerance,
        "outcome H_initial",
    );
    assert_f64_close(
        &[outcome.proposed_hamiltonian],
        &[proposed_hamiltonian],
        hamiltonian_tolerance,
        "outcome H_proposed",
    );
    assert_f64_close(
        &[outcome.delta_h],
        &[delta_h],
        hamiltonian_tolerance,
        "outcome delta_h",
    );
    assert_eq!(
        outcome.acceptance_probability.to_bits(),
        acceptance_probability.to_bits()
    );
    let next_raw = hmc_rng.next_u64();
    assert_eq!(next_raw, metadata["next_raw_word"].as_u64().unwrap());
    if outcome.accepted {
        assert!(link_residual(&hmc_links, &links) <= tolerance);
    } else {
        assert_links_bitwise_equal(&hmc_links, &initial_links);
    }
}

#[test]
fn accepted_and_rejected_updates_have_distinct_fixed_contracts() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let accepted_params = HmcParams::new(5.7, 1e-8, 1)?;
    let mut accepted_links = cold_su3(lattice)?;
    let accepted_before = clone_links(&accepted_links);
    let mut accepted_rng = ReproducibleRng::from_state([7, 11, 13, 17])?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let accepted = hmc_update(
        &mut context,
        &mut accepted_links,
        accepted_params,
        &mut accepted_rng,
    )?;
    assert!(accepted.accepted, "{accepted:?}");
    assert!(link_residual(&accepted_links, &accepted_before) > 0.0);

    let rejected_lattice = LatticeShape4::new([2, 1, 1, 1])?;
    let rejected_params = HmcParams::new(5.7, 4.0, 1)?;
    let mut rejected_links = cold_su3(rejected_lattice)?;
    let rejected_before = clone_links(&rejected_links);
    let mut rejected_rng = ReproducibleRng::from_state([19, 23, 29, 31])?;
    let rejected = hmc_update(
        &mut context,
        &mut rejected_links,
        rejected_params,
        &mut rejected_rng,
    )?;
    assert!(!rejected.accepted, "{rejected:?}");
    assert_links_bitwise_equal(&rejected_links, &rejected_before);
    Ok(())
}

#[test]
fn invalid_state_and_mismatch_errors_are_transactional() -> Result<(), GaugeError> {
    let links_lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let momentum_lattice = LatticeShape4::new([2, 1, 1, 1])?;
    let mut links = cold_su3(links_lattice)?;
    let mut momentum = constant_momentum(momentum_lattice, 1.0);
    let links_before = clone_links(&links);
    let momentum_before = scale_momentum(&momentum, 1.0);
    assert!(matches!(
        hamiltonian(&links, &momentum, 5.7),
        Err(GaugeError::Shape { .. })
    ));
    let params = HmcParams::new(5.7, 0.01, 1)?;
    let result = leapfrog_trajectory(
        &mut CpuEvolutionContext::new(CpuBackend::new()),
        &mut links,
        &mut momentum,
        params,
    );
    assert!(matches!(result, Err(GaugeError::Shape { .. })));
    assert_links_bitwise_equal(&links, &links_before);
    assert_eq!(momentum_residual(&momentum, &momentum_before), 0.0);

    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut bad_momentum = constant_momentum(lattice, 0.0);
    let bad_tensors = std::array::from_fn(|mu| {
        let mut values = bad_momentum.tensors()[mu].host_data().unwrap().to_vec();
        if mu == 0 {
            values[0] = f64::NAN;
        }
        TypedTensor::from_vec_col_major(bad_momentum.tensors()[mu].shape().to_vec(), values)
            .unwrap()
    });
    bad_momentum = TaGaugeField::new(bad_tensors, lattice)?;
    assert!(matches!(
        kinetic_energy(&bad_momentum),
        Err(GaugeError::NonFiniteMomentum { .. })
    ));
    let overflow_momentum = constant_momentum(lattice, f64::MAX);
    assert!(matches!(
        kinetic_energy(&overflow_momentum),
        Err(GaugeError::KineticNumericalRange)
    ));
    let mut links = cold_su3(lattice)?;
    let links_before = clone_links(&links);
    let momentum_before = scale_momentum(&bad_momentum, 1.0);
    let result = leapfrog_trajectory(
        &mut CpuEvolutionContext::new(CpuBackend::new()),
        &mut links,
        &mut bad_momentum,
        params,
    );
    assert!(matches!(result, Err(GaugeError::NonFiniteMomentum { .. })));
    assert_links_bitwise_equal(&links, &links_before);
    assert_momentum_bitwise_equal(&bad_momentum, &momentum_before);
    Ok(())
}

#[test]
fn reversibility_preserves_links_momentum_and_su3() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([4, 4, 4, 4])?;
    let mut links = cold_su3(lattice)?;
    let mut rng = ReproducibleRng::from_state([0x484d_435f_5355_3308, 2, 3, 5])?;
    let hot_momentum = scale_momentum(&gaugefields::sample_momentum(lattice, &mut rng)?, 1.5);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    exp_ta_update(&mut context, &mut links, 1.0, &hot_momentum)?;
    let initial = clone_links(&links);
    let mut momentum = scale_momentum(&gaugefields::sample_momentum(lattice, &mut rng)?, 0.35);
    let original_momentum = scale_momentum(&momentum, 1.0);
    let params = HmcParams::new(5.7, 0.01, 4)?;
    leapfrog_trajectory(&mut context, &mut links, &mut momentum, params)?;
    momentum = negate_momentum(&momentum);
    leapfrog_trajectory(&mut context, &mut links, &mut momentum, params)?;
    momentum = negate_momentum(&momentum);
    let link_error = link_residual(&links, &initial);
    let momentum_error = momentum_residual(&momentum, &original_momentum);
    let (unitary, determinant) = su3_drift(&links);
    eprintln!(
        "reversibility links={link_error:e} momentum={momentum_error:e} unitarity={unitary:e} determinant={determinant:e}"
    );
    assert!(link_error < 2e-10);
    assert!(momentum_error < 2e-10);
    assert!(unitary < 2e-10 && determinant < 2e-10);
    assert!(hamiltonian(&links, &momentum, 5.7)?.is_finite());
    Ok(())
}

#[test]
fn energy_error_has_second_order_global_scaling() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([4, 4, 4, 4])?;
    let mut initial = cold_su3(lattice)?;
    let mut rng = ReproducibleRng::from_state([0x484d_435f_454e_4552, 7, 11, 13])?;
    let hot_momentum = scale_momentum(&gaugefields::sample_momentum(lattice, &mut rng)?, 1.5);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    exp_ta_update(&mut context, &mut initial, 1.0, &hot_momentum)?;
    let base_momentum = scale_momentum(&gaugefields::sample_momentum(lattice, &mut rng)?, 0.2);
    let before = hamiltonian(&initial, &base_momentum, 5.7)?;
    let mut errors = Vec::new();
    for (step_size, steps) in [(0.02, 4), (0.01, 8), (0.005, 16)] {
        let mut links = clone_links(&initial);
        let mut momentum = scale_momentum(&base_momentum, 1.0);
        leapfrog_trajectory(
            &mut context,
            &mut links,
            &mut momentum,
            HmcParams::new(5.7, step_size, steps)?,
        )?;
        let error = (hamiltonian(&links, &momentum, 5.7)? - before).abs();
        eprintln!("energy step_size={step_size:e} steps={steps} abs_dh={error:e}");
        errors.push(error);
    }
    let ratios = [errors[0] / errors[1], errors[1] / errors[2]];
    eprintln!("energy ratios={ratios:?}");
    assert!(ratios.iter().all(|ratio| (2.5..6.5).contains(ratio)));
    Ok(())
}
