use dirac_operators::{
    FermionBoundary, FermionField, SolverParams, WilsonFermiAction, WilsonHmcParams,
};
use gaugefields::{
    exp_ta, kinetic_energy, load_link, store_link, wilson_action, CpuEvolutionContext,
    GaugeLinkTensor, GaugeLinks, LatticeShape4, ReproducibleRng, TaGaugeField,
};
use num_complex::Complex64;
use rand::RngCore;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::TypedTensor;

const FORCE_TOLERANCE: f64 = 2.0e-10;
const TRAJECTORY_TOLERANCE: f64 = 2.0e-10;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/fermions_task_c")
}

fn metadata() -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(
        fixture_dir().join("metadata.json"),
    )?)?)
}

fn read_f64(name: &str, shape: &[u64]) -> Result<Vec<f64>> {
    let bytes = fs::read(fixture_dir().join(name))?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    if npy.order() != npyz::Order::Fortran || npy.shape() != shape {
        return Err(format!("bad f64 NPY metadata for {name}").into());
    }
    Ok(npy.into_vec::<f64>()?)
}

fn read_c64(name: &str, shape: &[u64]) -> Result<Vec<Complex64>> {
    let bytes = fs::read(fixture_dir().join(name))?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    if npy.order() != npyz::Order::Fortran || npy.shape() != shape {
        return Err(format!("bad c64 NPY metadata for {name}").into());
    }
    Ok(npy.into_vec::<Complex64>()?)
}

fn assert_f64_close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let residual = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("{label}: max_abs_residual={residual:.17e}");
    assert!(residual <= tolerance, "{label}: residual={residual:e}");
}

fn assert_c64_close(label: &str, actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let residual = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max);
    eprintln!("{label}: max_abs_residual={residual:.17e}");
    assert!(residual <= tolerance, "{label}: residual={residual:e}");
}

fn field_values(field: &FermionField) -> Result<Vec<Complex64>> {
    let mut values = Vec::with_capacity(field.len());
    for site in 0..field.lattice().nv() {
        for component in 0..field.components() {
            for color in 0..3 {
                values.push(field.component(color, component, site)?);
            }
        }
    }
    Ok(values)
}

fn load_links() -> Result<GaugeLinks> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = (0..4)
        .map(|mu| {
            let values = read_c64(&format!("u{mu}.npy"), &[3, 3, 2, 2, 2, 2])?;
            let tensor = TypedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values)?;
            Ok(GaugeLinkTensor::from_typed(tensor, lattice)?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GaugeLinks::new(
        links.try_into().map_err(|_| "four links")?,
    )?)
}

fn load_field(name: &str) -> Result<FermionField> {
    let values = read_c64(name, &[3, 4, 2, 2, 2, 2])?;
    Ok(FermionField::from_vec_col_major(
        LatticeShape4::new([2, 2, 2, 2])?,
        4,
        values,
    )?)
}

fn julia_field_values(name: &str) -> Result<Vec<Complex64>> {
    let values = read_c64(name, &[3, 2, 2, 2, 2, 4])?;
    let volume = 16;
    let mut rust_values = Vec::with_capacity(values.len());
    for site in 0..volume {
        for component in 0..4 {
            for color in 0..3 {
                rust_values.push(values[color + 3 * (site + volume * component)]);
            }
        }
    }
    Ok(rust_values)
}

fn load_momentum(prefix: &str) -> Result<TaGaugeField> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let tensors = (0..4)
        .map(|mu| {
            Ok(TypedTensor::from_vec_col_major(
                vec![8, 2, 2, 2, 2],
                read_f64(&format!("{prefix}{mu}.npy"), &[8, 2, 2, 2, 2])?,
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TaGaugeField::new(
        tensors.try_into().map_err(|_| "four momentum")?,
        lattice,
    )?)
}

fn hamiltonian(links: &GaugeLinks, momentum: &TaGaugeField, beta: f64, action: f64) -> Result<f64> {
    Ok(wilson_action(links, beta)? + kinetic_energy(momentum)? + action)
}

fn metadata_params(meta: &Value) -> Result<(WilsonFermiAction, WilsonHmcParams)> {
    let solver = SolverParams::new(
        meta["solver_parameters"]["tolerance"]
            .as_f64()
            .ok_or("tolerance")?,
        meta["solver_parameters"]["max_iterations"]
            .as_u64()
            .ok_or("max iterations")? as usize,
    )?;
    let boundary = FermionBoundary::new([1, 1, 1, -1])?;
    let action = WilsonFermiAction::new(meta["kappa"].as_f64().ok_or("kappa")?, boundary, solver)?;
    let hmc = WilsonHmcParams::new(
        meta["beta"].as_f64().ok_or("beta")?,
        meta["kappa"].as_f64().ok_or("kappa")?,
        meta["trajectory"]["step_size"].as_f64().ok_or("step")?,
        meta["trajectory"]["steps"].as_u64().ok_or("steps")? as usize,
        boundary,
        solver,
    )?;
    Ok((action, hmc))
}

fn assert_declared_fixture_files(meta: &Value) -> Result<()> {
    let mut declared = Vec::new();
    for value in meta["files"].as_array().ok_or("files")? {
        declared.push(value.as_str().ok_or("file name")?.to_owned());
    }
    let mut actual = Vec::new();
    for entry in fs::read_dir(fixture_dir())? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "metadata.json" {
            continue;
        }
        assert!(entry.file_type()?.is_file(), "unexpected non-file {name}");
        actual.push(name);
    }
    declared.sort();
    actual.sort();
    assert_eq!(
        declared, actual,
        "metadata files must cover the complete payload tree"
    );
    Ok(())
}

#[test]
fn generated_task_c_metadata_payloads_action_force_and_trajectory_match() -> Result<()> {
    let meta = metadata()?;
    assert_eq!(meta["schema"], "fermions_task_c.v1");
    assert_eq!(meta["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(meta["nc"], 3);
    assert_eq!(meta["components"], 4);
    assert_eq!(meta["beta"], 5.7);
    assert_eq!(meta["kappa"], 0.13);
    assert_eq!(meta["r"], 1.0);
    assert_eq!(meta["boundaries"], serde_json::json!([1, 1, 1, -1]));
    assert_eq!(meta["solver_parameters"]["tolerance"], 1.0e-20);
    assert_eq!(meta["solver_parameters"]["max_iterations"], 2_000);
    assert_eq!(
        meta["solver_parameters"]["julia_operator_keys"],
        serde_json::json!([
            "Dirac_operator",
            "κ",
            "r",
            "faster version",
            "verbose_level",
            "boundarycondition",
            "method_CG",
            "eps_CG",
            "MaxCGstep"
        ])
    );
    assert_eq!(
        meta["solver_parameters"]["julia_solver_keywords"],
        serde_json::json!(["eps", "maxsteps", "verbose"])
    );
    assert_eq!(
        meta["gaugefields_jl"],
        serde_json::json!({
            "package": "Gaugefields.jl",
            "version": "0.7.2",
            "commit": "9e5719970770f4497405a856315c90bef7f74449",
            "clean": true
        })
    );
    assert_eq!(
        meta["latticediracoperators_jl"],
        serde_json::json!({
            "package": "LatticeDiracOperators.jl",
            "version": "0.6.4",
            "commit": "bdef628184597815ba3e0cddf2536df767e78a02",
            "clean": true
        })
    );
    assert_eq!(
        meta["layout"]["permutation"],
        serde_json::json!([1, 6, 2, 3, 4, 5])
    );
    assert_eq!(
        meta["source_urls"],
        serde_json::json!([
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/action/WilsonFermiAction.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/test/wilsonhmc.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/WilsonFermion/WilsonFermion.jl",
            "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/TA_gaugefields_4D_serial.jl"
        ])
    );
    assert_eq!(
        meta["source_functions"],
        serde_json::json!([
            "sample_pseudofermions!",
            "evaluate_FermiAction",
            "calc_UdSfdU!",
            "calc_UdSfdU_fromX!",
            "MDstep!",
            "U_update!",
            "P_update!",
            "P_update_fermion!",
            "Traceless_antihermitian_add!"
        ])
    );
    assert_eq!(
        meta["entrypoint_map"],
        serde_json::json!([
            {"julia": "sample_pseudofermions!", "julia_source": "src/action/WilsonFermiAction.jl:362-377", "rust": "WilsonFermiAction::sample_pseudofermion"},
            {"julia": "evaluate_FermiAction", "julia_source": "src/action/WilsonFermiAction.jl:86-97", "rust": "WilsonFermiAction::evaluate"},
            {"julia": "calc_UdSfdU!", "julia_source": "src/action/WilsonFermiAction.jl:99-136", "rust": "WilsonFermiAction::force"},
            {"julia": "calc_UdSfdU_fromX!", "julia_source": "src/action/WilsonFermiAction.jl:138-234", "rust": "wilson_action.rs::force_from_x"},
            {"julia": "MDstep!/U_update!/P_update!/P_update_fermion!", "julia_source": "test/wilsonhmc.jl:46-146", "rust": "wilson_hmc.rs::wilson_hmc_update"},
            {"julia": "Traceless_antihermitian_add!", "julia_source": "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/TA_gaugefields_4D_serial.jl:181-269", "rust": "Mat3::add_ta_coefficients"}
        ])
    );
    assert_eq!(meta["construction"], "explicit diagonal SU(3) links, fixed xi, phi, and coefficient-space momentum; no global RNG; the acceptance draw uses explicit Julia Xoshiro state");
    assert_eq!(meta["pseudofermion_refresh"]["flavors"], 2);
    assert_eq!(meta["pseudofermion_refresh"]["formula"], "phi = D† xi");
    assert_eq!(
        meta["pseudofermion_refresh"]["complex_normal_scale"],
        "1/sqrt(2) per independent real and imaginary standard normal"
    );
    assert_eq!(
        meta["pseudofermion_refresh"]["fixture_xi"],
        "fixed array; sampler parity is checked separately in Rust"
    );
    assert_eq!(meta["force_convention"]["x"], "(D†D)^-1 phi");
    assert_eq!(meta["force_convention"]["y"], "D x");
    assert_eq!(
        meta["force_convention"]["raw_formula"],
        "-kappa Pminus U Xplus outer Y + kappa X outer (Yplus† U† Pplus)"
    );
    assert_eq!(
        meta["force_convention"]["wrapped_link_sign"],
        "applied to both terms exactly once"
    );
    assert_eq!(
        meta["force_convention"]["projection"],
        "Gaugefields.jl Traceless_antihermitian_add!; A=(i/2) sum_a c_a lambda_a"
    );
    assert_eq!(meta["force_convention"]["y"], "D x");
    assert_eq!(
        meta["force_convention"]["gauge_1_over_nc"],
        "not applied here"
    );
    assert_eq!(meta["momentum_update_scaling"]["gauge"], "-step_size/NC");
    assert_eq!(meta["momentum_update_scaling"]["fermion"], "-step_size");
    assert_eq!(
        meta["comparison"]["finite_difference_epsilons"],
        serde_json::json!([1.0e-3, 5.0e-4, 2.5e-4])
    );
    assert_eq!(meta["files"].as_array().map(Vec::len), Some(28));
    assert_declared_fixture_files(&meta)?;
    assert!(meta["trajectory"]["initial_hamiltonian"].is_number());
    assert!(meta["trajectory"]["proposed_hamiltonian"].is_number());
    assert!(meta["trajectory"]["delta_h"].is_number());
    assert_eq!(meta["trajectory"]["step_size"], 0.002);
    assert_eq!(meta["trajectory"]["steps"], 2);
    assert_eq!(
        meta["trajectory"]["acceptance_rng_state"],
        serde_json::json!([4846228630232126559u64, 17, 29, 43])
    );
    assert_eq!(
        meta["trajectory"]["acceptance_uniform_bits"],
        4606228581418356287u64
    );
    assert_eq!(meta["trajectory"]["next_raw_word"], 16493285115757197478u64);
    assert_eq!(meta["trajectory"]["accepted"], true);
    assert_eq!(meta["comparison"]["field_max_abs_tolerance"], 2.0e-10);
    assert_eq!(meta["comparison"]["force_max_abs_tolerance"], 2.0e-10);
    assert_eq!(meta["comparison"]["action_tolerance"], 2.0e-10);
    assert_eq!(
        meta["comparison"]["force_finite_difference_tolerance"],
        2.0e-7
    );
    for file in meta["files"].as_array().ok_or("files")? {
        let name = file.as_str().ok_or("file name")?;
        assert!(fixture_dir().join(name).is_file(), "missing {name}");
    }
    let links = load_links()?;
    let (action, hmc) = metadata_params(&meta)?;
    let xi = load_field("xi_rust.npy")?;
    let phi = load_field("phi_rust.npy")?;

    assert_c64_close(
        "xi Rust payload layout",
        &field_values(&xi)?,
        &read_c64("xi_rust.npy", &[3, 4, 2, 2, 2, 2])?,
        0.0,
    );
    assert_c64_close(
        "xi Julia-to-Rust layout",
        &field_values(&xi)?,
        &julia_field_values("xi_julia.npy")?,
        FORCE_TOLERANCE,
    );
    assert_c64_close(
        "phi Rust payload layout",
        &field_values(&phi)?,
        &read_c64("phi_rust.npy", &[3, 4, 2, 2, 2, 2])?,
        0.0,
    );
    assert_c64_close(
        "phi Julia-to-Rust layout",
        &field_values(&phi)?,
        &julia_field_values("phi_julia.npy")?,
        FORCE_TOLERANCE,
    );

    let evaluated = action.evaluate(&links, &phi)?;
    assert_f64_close(
        "action",
        &[evaluated.action],
        &[meta["action"].as_f64().ok_or("action")?],
        2.0e-10,
    );
    assert_c64_close(
        "X Rust payload parity",
        &field_values(&evaluated.x)?,
        &read_c64("x_rust.npy", &[3, 4, 2, 2, 2, 2])?,
        FORCE_TOLERANCE,
    );
    assert_c64_close(
        "X Julia parity",
        &field_values(&evaluated.x)?,
        &julia_field_values("x_julia.npy")?,
        FORCE_TOLERANCE,
    );
    let force = action.force(&links, &phi)?;
    assert_c64_close(
        "Y Rust payload parity",
        &field_values(&force.y)?,
        &read_c64("y_rust.npy", &[3, 4, 2, 2, 2, 2])?,
        FORCE_TOLERANCE,
    );
    assert_c64_close(
        "Y Julia parity",
        &field_values(&force.y)?,
        &julia_field_values("y_julia.npy")?,
        FORCE_TOLERANCE,
    );
    for mu in 0..4 {
        assert_f64_close(
            &format!("force mu={mu}"),
            force.force.tensors()[mu].host_data()?,
            &read_f64(&format!("force{mu}.npy"), &[8, 2, 2, 2, 2])?,
            FORCE_TOLERANCE,
        );
    }

    let mut links_trajectory = load_links()?;
    let initial_links = links_trajectory.try_clone()?;
    let mut momentum = load_momentum("p_initial")?;
    let initial_action = action.evaluate(&links_trajectory, &phi)?.action;
    let initial_h = hamiltonian(&links_trajectory, &momentum, hmc.beta(), initial_action)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    dirac_operators::wilson_leapfrog_trajectory(
        &mut context,
        &mut links_trajectory,
        &mut momentum,
        &phi,
        hmc,
    )?;
    let proposed_action = action.evaluate(&links_trajectory, &phi)?.action;
    let proposed_h = hamiltonian(&links_trajectory, &momentum, hmc.beta(), proposed_action)?;
    assert_f64_close(
        "H_initial",
        &[initial_h],
        &[meta["trajectory"]["initial_hamiltonian"]
            .as_f64()
            .ok_or("H initial")?],
        TRAJECTORY_TOLERANCE,
    );
    assert_f64_close(
        "H_proposed",
        &[proposed_h],
        &[meta["trajectory"]["proposed_hamiltonian"]
            .as_f64()
            .ok_or("H proposed")?],
        TRAJECTORY_TOLERANCE,
    );
    assert_f64_close(
        "delta H",
        &[proposed_h - initial_h],
        &[meta["trajectory"]["delta_h"].as_f64().ok_or("delta H")?],
        TRAJECTORY_TOLERANCE,
    );
    for mu in 0..4 {
        assert_f64_close(
            &format!("p_final mu={mu}"),
            momentum.tensors()[mu].host_data()?,
            &read_f64(&format!("p_final{mu}.npy"), &[8, 2, 2, 2, 2])?,
            TRAJECTORY_TOLERANCE,
        );
        assert_c64_close(
            &format!("U proposed mu={mu}"),
            links_trajectory.links()[mu].typed().host_data()?,
            &read_c64(&format!("u_proposed{mu}.npy"), &[3, 3, 2, 2, 2, 2])?,
            TRAJECTORY_TOLERANCE,
        );
    }

    let state: [u64; 4] = meta["trajectory"]["acceptance_rng_state"]
        .as_array()
        .ok_or("rng state")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "rng word"))
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| "rng state length")?;
    let mut rng = ReproducibleRng::from_state(state)?;
    let uniform = rng.open_unit_f64();
    assert_eq!(
        uniform.to_bits(),
        meta["trajectory"]["acceptance_uniform_bits"]
            .as_u64()
            .ok_or("uniform bits")?
    );
    assert_eq!(
        uniform.to_bits(),
        meta["trajectory"]["acceptance_uniform"]
            .as_f64()
            .ok_or("uniform")?
            .to_bits()
    );
    assert_eq!(
        rng.next_u64(),
        meta["trajectory"]["next_raw_word"]
            .as_u64()
            .ok_or("next word")?
    );
    assert_eq!(
        uniform
            <= meta["trajectory"]["acceptance_probability"]
                .as_f64()
                .ok_or("probability")?,
        meta["trajectory"]["accepted"].as_bool().ok_or("decision")?
    );
    assert_c64_close(
        "rejected links remain old",
        initial_links.links()[0].typed().host_data()?,
        load_links()?.links()[0].typed().host_data()?,
        0.0,
    );
    Ok(())
}

fn left_shifted_links(
    base: &GaugeLinks,
    direction: usize,
    site: usize,
    generator: usize,
    epsilon: f64,
) -> Result<GaugeLinks> {
    let mut links = base.try_clone()?;
    let mut coefficients = [0.0; 8];
    coefficients[generator] = 1.0;
    let left = exp_ta(epsilon, &coefficients)?;
    let link = load_link(&links, direction, site)?;
    store_link(&mut links, direction, site, left.mul(link))?;
    Ok(links)
}

#[test]
fn all_force_directions_sites_and_generators_match_central_difference_with_quadratic_trend(
) -> Result<()> {
    const FD_TOLERANCE: f64 = 2.0e-7;
    const SELECTED_EPSILON_INDEX: usize = 1;
    let meta = metadata()?;
    let links = load_links()?;
    let phi = load_field("phi_rust.npy")?;
    let (action, _) = metadata_params(&meta)?;
    let force = action.force(&links, &phi)?.force;
    let epsilons = [1.0e-3, 5.0e-4, 2.5e-4];
    let mut max_residuals = [0.0_f64; 3];
    let mut max_locations = [(0, 0, 0, 0.0_f64); 3];
    let mut selected_location = (0, 0, 0, 0.0_f64);
    let mut selected_location_errors = [0.0_f64; 3];
    let mut coefficient_count = 0;
    let mut selected_pass_count = 0;
    for direction in 0..4 {
        for site in 0..links.lattice().nv() {
            for generator in 0..8 {
                coefficient_count += 1;
                let coefficient = force.tensors()[direction].host_data()?[generator + 8 * site];
                let mut errors = [0.0; 3];
                let mut selected_candidate = false;
                for (index, epsilon) in epsilons.into_iter().enumerate() {
                    let plus = action
                        .evaluate(
                            &left_shifted_links(&links, direction, site, generator, epsilon)?,
                            &phi,
                        )?
                        .action;
                    let minus = action
                        .evaluate(
                            &left_shifted_links(&links, direction, site, generator, -epsilon)?,
                            &phi,
                        )?
                        .action;
                    let derivative = (plus - minus) / (2.0 * epsilon);
                    errors[index] = (derivative - coefficient).abs();
                    if errors[index] > max_residuals[index] {
                        max_residuals[index] = errors[index];
                        max_locations[index] = (direction, site, generator, coefficient);
                        if index == SELECTED_EPSILON_INDEX {
                            selected_candidate = true;
                        }
                    }
                }
                if selected_candidate {
                    selected_location = (direction, site, generator, coefficient);
                    selected_location_errors = errors;
                }
                if errors[SELECTED_EPSILON_INDEX] <= FD_TOLERANCE {
                    selected_pass_count += 1;
                } else {
                    panic!(
                        "mu={direction} site={site} generator={generator} coeff={coefficient:e} selected_epsilon={}: errors={errors:?}",
                        epsilons[SELECTED_EPSILON_INDEX]
                    );
                }
            }
        }
    }
    assert_eq!(coefficient_count, 4 * 16 * 8);
    assert_eq!(selected_pass_count, 4 * 16 * 8);
    assert!(max_residuals[SELECTED_EPSILON_INDEX] <= FD_TOLERANCE);
    let quadratic_ratios = [
        max_residuals[0] / max_residuals[1],
        max_residuals[1] / max_residuals[2],
    ];
    eprintln!(
        "all force finite-difference epsilons={epsilons:?} max_residuals={max_residuals:?} locations={max_locations:?}"
    );
    eprintln!(
        "finite-difference selected epsilon={} tolerance={FD_TOLERANCE:.1e} pass_count={selected_pass_count}/{coefficient_count} worst_location={selected_location:?} series={selected_location_errors:?} quadratic_ratios={quadratic_ratios:?}",
        epsilons[SELECTED_EPSILON_INDEX]
    );
    assert!(max_residuals[1] < max_residuals[0]);
    assert!(max_residuals[2] < max_residuals[1]);
    assert!(quadratic_ratios
        .iter()
        .all(|ratio| (2.0..8.0).contains(ratio)));
    Ok(())
}

#[test]
fn sampler_scales_both_complex_normal_parts_and_consumes_one_pair_per_component() -> Result<()> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let action = WilsonFermiAction::new(
        0.13,
        FermionBoundary::new([1, 1, 1, -1])?,
        SolverParams::new(1.0e-20, 256)?,
    )?;
    let state = [41, 43, 47, 53];
    let mut rng = ReproducibleRng::from_state(state)?;
    let xi = action.sample_xi(lattice, &mut rng)?;
    let mut replay = ReproducibleRng::from_state(state)?;
    for site in 0..lattice.nv() {
        for component in 0..4 {
            for color in 0..3 {
                let pair = replay.standard_normal_pair();
                let expected = Complex64::new(
                    std::f64::consts::FRAC_1_SQRT_2 * pair[0],
                    std::f64::consts::FRAC_1_SQRT_2 * pair[1],
                );
                assert_eq!(xi.component(color, component, site)?, expected);
            }
        }
    }
    assert_eq!(rng.next_u64(), replay.next_u64());
    Ok(())
}
