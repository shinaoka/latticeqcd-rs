use dirac_operators::{
    FermionBoundary, FermionField, SolverParams, StaggeredFermiAction, StaggeredHmcParams,
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

const FIELD_TOLERANCE: f64 = 2.0e-9;
const REPORT_TOLERANCE: f64 = 1.0e-24;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/fermions_task_e")
}

fn metadata() -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(
        fixture_dir().join("metadata.json"),
    )?)?)
}

fn read_npy<T: npyz::Deserialize>(name: &str) -> Result<(Vec<T>, Vec<usize>)> {
    let bytes = fs::read(fixture_dir().join(name))?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    if npy.order() != npyz::Order::Fortran {
        return Err(format!("{name} is not Fortran order").into());
    }
    let shape = npy
        .shape()
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((npy.into_vec::<T>()?, shape))
}

fn read_c64(name: &str, shape: &[usize]) -> Result<Vec<Complex64>> {
    let (values, found) = read_npy::<Complex64>(name)?;
    if found != shape {
        return Err(format!("{name} shape {found:?}, expected {shape:?}").into());
    }
    Ok(values)
}

fn read_f64(name: &str, shape: &[usize]) -> Result<Vec<f64>> {
    let (values, found) = read_npy::<f64>(name)?;
    if found != shape {
        return Err(format!("{name} shape {found:?}, expected {shape:?}").into());
    }
    Ok(values)
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

fn julia_field_values(name: &str) -> Result<Vec<Complex64>> {
    let values = read_c64(name, &[3, 2, 2, 2, 2, 1])?;
    Ok(values)
}

fn assert_c64_close(label: &str, actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let residual = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max);
    eprintln!("{label}: max_abs_residual={residual:.17e}");
    assert!(residual <= tolerance, "{label}: residual={residual:e}");
}

fn assert_f64_close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let residual = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    eprintln!("{label}: max_abs_residual={residual:.17e}");
    assert!(residual <= tolerance, "{label}: residual={residual:e}");
}

fn load_links() -> Result<GaugeLinks> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = (0..4)
        .map(|direction| {
            let values = read_c64(&format!("u{direction}.npy"), &[3, 3, 2, 2, 2, 2])?;
            let tensor = TypedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values)?;
            Ok(GaugeLinkTensor::from_typed(tensor, lattice)?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GaugeLinks::new(
        links.try_into().map_err(|_| "four links")?,
    )?)
}

fn load_field(name: &str) -> Result<FermionField> {
    Ok(FermionField::from_vec_col_major(
        LatticeShape4::new([2, 2, 2, 2])?,
        1,
        read_c64(name, &[3, 1, 2, 2, 2, 2])?,
    )?)
}

fn load_momentum(prefix: &str) -> Result<TaGaugeField> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let tensors = (0..4)
        .map(|direction| {
            Ok(TypedTensor::from_vec_col_major(
                vec![8, 2, 2, 2, 2],
                read_f64(&format!("{prefix}{direction}.npy"), &[8, 2, 2, 2, 2])?,
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

fn assert_declared_files(meta: &Value) -> Result<()> {
    let mut declared = meta["files"]
        .as_array()
        .ok_or("files must be an array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "file must be a string")
                })
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut actual = fs::read_dir(fixture_dir())?
        .map(|entry| -> Result<String> { Ok(entry?.file_name().to_string_lossy().into_owned()) })
        .collect::<Result<Vec<_>>>()?;
    actual.retain(|name| name != "metadata.json");
    declared.sort();
    actual.sort();
    assert_eq!(
        declared, actual,
        "metadata must cover the complete payload tree"
    );
    for name in declared {
        assert!(fixture_dir().join(&name).is_file(), "missing {name}");
    }
    Ok(())
}

fn assert_coefficient_table(meta: &Value, name: &str, degree: usize, power: f64, role: &str) {
    let table = &meta["coefficient_tables"][name];
    assert_eq!(table["role"], role);
    assert_eq!(table["degree"], degree);
    assert_eq!(table["power"].as_f64(), Some(power));
    let alpha = table["alpha"].as_array().unwrap();
    let beta = table["beta"].as_array().unwrap();
    let bits = table["bits"].as_array().unwrap();
    assert_eq!(alpha.len(), degree);
    assert_eq!(beta.len(), degree);
    assert_eq!(bits.len(), 1 + 2 * degree);
    let numbers = std::iter::once(table["alpha0"].as_f64().unwrap())
        .chain(alpha.iter().map(|value| value.as_f64().unwrap()))
        .chain(beta.iter().map(|value| value.as_f64().unwrap()));
    for (number, bit) in numbers.zip(bits) {
        let text = bit.as_str().unwrap();
        assert!(text.starts_with("0x"));
        let expected = u64::from_str_radix(&text[2..], 16).unwrap();
        let difference = number.to_bits().abs_diff(expected);
        assert!(
            difference <= 1,
            "{name} coefficient {text} differs by {difference} ulps"
        );
    }
}

fn report_file() -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(
        fixture_dir().join("rational_reports.json"),
    )?)?)
}

fn assert_report_group(name: &str, degree: usize, beta: &[f64]) -> Result<()> {
    let report_file = report_file()?;
    let group = &report_file[name];
    assert!(group["initial_residual_squared"]
        .as_f64()
        .unwrap()
        .is_finite());
    let reports = group["reports"].as_array().unwrap();
    assert_eq!(reports.len(), degree);
    for (report, expected_shift) in reports.iter().zip(beta) {
        assert_eq!(report["shift"].as_f64(), Some(*expected_shift));
        assert_eq!(report["absolute_squared_tolerance"], 1.0e-24);
        assert_eq!(report["maximum_iterations"], 2_000);
        assert_eq!(report["convergence_branch"], "updated_residual");
        assert!(report["iterations"].as_u64().unwrap() > 0);
        assert!(report["recursive_residual_squared"]
            .as_f64()
            .unwrap()
            .is_finite());
        assert!(report["true_residual_squared"].as_f64().unwrap() < REPORT_TOLERANCE);
    }
    Ok(())
}

fn assert_reports_match(
    name: &str,
    actual: &[dirac_operators::MultiShiftSolverReport],
) -> Result<()> {
    let group = report_file()?[name].clone();
    let reports = group["reports"]
        .as_array()
        .ok_or("reports must be an array")?;
    assert_eq!(reports.len(), actual.len());
    for (report, actual) in reports.iter().zip(actual) {
        assert!((actual.shift - report["shift"].as_f64().unwrap()).abs() <= 1.0e-18);
        assert_eq!(
            actual.iterations,
            report["iterations"].as_u64().unwrap() as usize
        );
        assert!(
            (actual.initial_residual_squared - group["initial_residual_squared"].as_f64().unwrap())
                .abs()
                <= 1.0e-15
        );
        assert!(
            (actual.recursive_residual_squared
                - report["recursive_residual_squared"].as_f64().unwrap())
            .abs()
                <= 1.0e-24
        );
        assert!(
            (actual.true_residual_squared - report["true_residual_squared"].as_f64().unwrap())
                .abs()
                <= 1.0e-24
        );
        assert_eq!(
            actual.tolerance,
            report["absolute_squared_tolerance"].as_f64().unwrap()
        );
        assert_eq!(
            actual.maximum_iterations,
            report["maximum_iterations"].as_u64().unwrap() as usize
        );
        assert_eq!(
            actual.convergence_branch.to_string(),
            report["convergence_branch"].as_str().unwrap()
        );
    }
    Ok(())
}

#[test]
fn generated_task_e_metadata_coefficients_reports_and_payloads_match() -> Result<()> {
    let meta = metadata()?;
    assert_eq!(meta["schema"], "fermions_task_e.v1");
    assert_eq!(meta["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(meta["nc"], 3);
    assert_eq!(meta["components"], 1);
    assert_eq!(meta["nf"], 2);
    assert_eq!(meta["mass"], 0.17);
    assert_eq!(meta["xi_scale"], 0.25);
    assert_eq!(meta["boundaries"], serde_json::json!([1, 1, 1, -1]));
    assert_eq!(
        meta["spectral_bounds"],
        serde_json::json!({
            "claimed_lower": 0.0004,
            "claimed_upper": 64.0,
            "table_lower": 0.0004,
            "table_upper": 64.0,
            "caller_assertion": true
        })
    );
    assert_eq!(
        meta["degrees"],
        serde_json::json!({"refresh": 15, "action": 15, "md_force": 10})
    );
    assert_eq!(
        meta["solver_parameters"]["absolute_squared_tolerance"],
        1.0e-24
    );
    assert_eq!(meta["solver_parameters"]["max_iterations"], 2_000);
    assert_eq!(
        meta["solver_parameters"]["julia_solver_keywords"],
        serde_json::json!(["eps", "maxsteps", "verbose"])
    );
    assert_coefficient_table(&meta, "refresh", 15, 0.125, "private coeffs_18");
    assert_coefficient_table(&meta, "action", 15, -0.125, "private coeffs_m18");
    assert_coefficient_table(&meta, "md_force", 10, -0.25, "private coeffs_m14_n10");
    assert_eq!(meta["scalar_log_grid"]["points"], 4_097);
    assert!(
        meta["scalar_log_grid"]["max_abs_error"]["refresh"]
            .as_f64()
            .unwrap()
            < 2.6e-9
    );
    assert!(
        meta["scalar_log_grid"]["max_abs_error"]["action"]
            .as_f64()
            .unwrap()
            < 4.1e-9
    );
    assert!(
        meta["scalar_log_grid"]["max_abs_error"]["md_force"]
            .as_f64()
            .unwrap()
            < 1.6e-5
    );
    assert_eq!(
        meta["scalar_log_grid"]["spacing"],
        "lambda_low*exp(log(lambda_high/lambda_low)*i/(points-1)); endpoints exact"
    );
    assert_eq!(
        meta["scalar_log_grid"]["powers"],
        serde_json::json!({"refresh": 0.125, "action": -0.125, "md_force": -0.25})
    );
    assert_eq!(
        meta["comparison"]["finite_difference_epsilons"],
        serde_json::json!([0.32, 0.16, 0.08, 0.04])
    );
    assert_eq!(
        meta["comparison"]["finite_difference_series"],
        serde_json::json!({
            "epsilons": [0.32, 0.16, 0.08, 0.04],
            "max_residuals": [
                8.434653210321642e-6,
                2.139177378187619e-6,
                5.605769951367093e-7,
                1.6563038083509257e-7
            ],
            "global_max_ratios": [
                3.9429424115674574,
                3.816027765580949,
                3.384505863660601
            ],
            "pass_counts": [291, 442, 510, 512],
            "tolerance": 5.0e-7,
            "selected_epsilon": 0.04,
            "selected_pass_count": 512,
            "coefficient_count": 512,
            "construction": "central U <- exp(epsilon*T_a)U; finite differences of StaggeredFermiAction::evaluate; global maxima over 4*16*8 coefficients"
        })
    );
    assert_eq!(
        meta["rational_form"],
        "R(M)b=alpha0*b+sum_j alpha_j*(M+beta_j I)^-1*b"
    );
    assert_eq!(meta["coefficient_roles"]["refresh"], "x^(+1/8) degree-15");
    assert_eq!(meta["coefficient_roles"]["action"], "x^(-1/8) degree-15");
    assert_eq!(
        meta["coefficient_roles"]["md_force"],
        "x^(-1/4) degree-10 inverse residues; alpha0 has no link derivative"
    );
    assert_eq!(
        meta["gaugefields_jl"]["commit"],
        "9e5719970770f4497405a856315c90bef7f74449"
    );
    assert_eq!(
        meta["latticediracoperators_jl"]["commit"],
        "bdef628184597815ba3e0cddf2536df767e78a02"
    );
    assert_eq!(
        meta["gaugefields_jl"],
        serde_json::json!({"package": "Gaugefields.jl", "version": "0.7.2", "commit": "9e5719970770f4497405a856315c90bef7f74449", "clean": true})
    );
    assert_eq!(
        meta["latticediracoperators_jl"],
        serde_json::json!({"package": "LatticeDiracOperators.jl", "version": "0.6.4", "commit": "bdef628184597815ba3e0cddf2536df767e78a02", "clean": true})
    );
    assert_eq!(
        meta["source_functions"],
        serde_json::json!([
            "StaggeredFermiAction",
            "sample_pseudofermions!",
            "evaluate_FermiAction",
            "calc_UdSfdU!",
            "calc_UdSfdU_fromX!",
            "RHMC",
            "shiftedcg",
            "Traceless_antihermitian_add!",
            "MDstep!",
            "U_update!",
            "P_update!",
            "P_update_fermion!"
        ])
    );
    assert_eq!(
        meta["source_urls"],
        serde_json::json!([
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/action/StaggeredFermiAction.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/rhmc/rhmc.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/StaggeredFermion/StaggeredFermion.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/StaggeredFermion/StaggeredFermion_4D_nowing.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/cgmethods.jl",
            "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/TA_gaugefields_4D_serial.jl"
        ])
    );
    assert_eq!(
        meta["entrypoint_map"],
        serde_json::json!([
            {"julia": "sample_pseudofermions!", "julia_source": "src/action/StaggeredFermiAction.jl:176-276", "rust": "staggered_action.rs::StaggeredFermiAction::sample_pseudofermion"},
            {"julia": "evaluate_FermiAction", "julia_source": "src/action/StaggeredFermiAction.jl:98-142", "rust": "staggered_action.rs::StaggeredFermiAction::evaluate"},
            {"julia": "calc_UdSfdU!/calc_UdSfdU_fromX!", "julia_source": "src/action/StaggeredFermiAction.jl:278-422", "rust": "staggered_action.rs::StaggeredFermiAction::force/force_from_shifted_xy"},
            {"julia": "RHMC", "julia_source": "src/rhmc/rhmc.jl:1-1294", "rust": "rhmc.rs::typed private coefficient roles"},
            {"julia": "shiftedcg", "julia_source": "src/cgmethods.jl:872-968", "rust": "solvers.rs::multi_shift_cg"},
            {"julia": "MDstep!/U_update!/P_update!/P_update_fermion!", "julia_source": "test/wilsonhmc.jl:46-146", "rust": "rhmc.rs::staggered_hmc_update/staggered_leapfrog_trajectory"},
            {"julia": "Traceless_antihermitian_add!", "julia_source": "Gaugefields.jl/src/4D/TA_gaugefields_4D_serial.jl:181-269", "rust": "Mat3::add_ta_coefficients"}
        ])
    );
    assert_eq!(
        meta["layout"]["permutation"],
        serde_json::json!([1, 6, 2, 3, 4, 5])
    );
    assert_eq!(meta["layout"]["julia_shape"], "[3,NX,NY,NZ,NT,1]");
    assert_eq!(meta["layout"]["rust_shape"], "[3,1,NX,NY,NZ,NT]");
    assert_eq!(
        meta["layout"]["conversion"],
        "permutedims(array, (1,6,2,3,4,5))"
    );
    assert_eq!(
        meta["layout"]["force"],
        "Float64 Fortran [gell_mann_component,x,y,z,t]"
    );
    assert_eq!(meta["refresh"]["flavors"], 2);
    assert_eq!(meta["refresh"]["formula"], "phi=R_(+1/8)(M)xi");
    assert_eq!(
        meta["refresh"]["complex_normal_scale"],
        "1/sqrt(2) per independent real and imaginary standard normal"
    );
    assert_eq!(meta["refresh"]["xi"], "explicit deterministic field");
    assert_eq!(meta["action"]["formula"], "||R_(-1/8)(M)phi||^2");
    assert_eq!(meta["action"]["x_name"], "X=R_(-1/8)(M)phi");
    assert_eq!(meta["force"]["x_name"], "X_j=(M+beta_j I)^-1 phi");
    assert_eq!(meta["force"]["y_name"], "Y_j=D X_j");
    assert_eq!(meta["force"]["outer_products"], 2);
    assert_eq!(meta["force"]["projection_count"], 1);
    assert_eq!(
        meta["force"]["wrapped_boundary_sign"],
        "applied once to each shifted plus field"
    );
    assert_eq!(meta["force"]["gauge_1_over_nc"], "not applied here");
    assert_eq!(meta["trajectory"]["beta"], 5.7);
    assert_eq!(meta["trajectory"]["steps"], 2);
    assert_eq!(meta["trajectory"]["accepted"], true);
    assert!(meta["trajectory"]["acceptance_probability"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(meta["trajectory"]["acceptance_uniform"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(meta["trajectory"]["initial_hamiltonian"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(meta["trajectory"]["proposed_hamiltonian"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(meta["trajectory"]["delta_h"].as_f64().unwrap().is_finite());
    assert_eq!(meta["comparison"]["field_max_abs_tolerance"], 2.0e-9);
    assert_eq!(meta["comparison"]["action_tolerance"], 2.0e-9);
    assert_eq!(meta["comparison"]["force_tolerance"], 2.0e-9);
    assert_eq!(meta["comparison"]["finite_difference_tolerance"], 5.0e-7);
    assert_eq!(meta["comparison"]["reversibility_tolerance"], 5.0e-9);
    assert_eq!(
        meta["comparison"]["criterion"],
        "maximum absolute payload residual; central force FD; reversibility"
    );
    assert_eq!(meta["generator"]["script"], "fixtures/generate.jl");
    assert_eq!(meta["generator"]["mode"], "fermions_task_e");
    assert_eq!(
        meta["generator"]["randomness"],
        "none for fixed xi/phi/momentum; explicit Xoshiro only for acceptance word"
    );
    assert_declared_files(&meta)?;

    let refresh_beta = meta["coefficient_tables"]["refresh"]["beta"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let action_beta = meta["coefficient_tables"]["action"]["beta"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let force_beta = meta["coefficient_tables"]["md_force"]["beta"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_report_group("refresh", 15, &refresh_beta)?;
    assert_report_group("action", 15, &action_beta)?;
    assert_report_group("force", 10, &force_beta)?;

    let links = load_links()?;
    let lattice = links.lattice();
    let action = StaggeredFermiAction::new(
        meta["mass"].as_f64().unwrap(),
        FermionBoundary::new([1, 1, 1, -1])?,
        meta["spectral_bounds"]["claimed_lower"].as_f64().unwrap(),
        meta["spectral_bounds"]["claimed_upper"].as_f64().unwrap(),
        SolverParams::new(1.0e-24, 2_000)?,
    )?;
    let xi = load_field("xi_rust.npy")?;
    let phi = load_field("phi_rust.npy")?;
    let refreshed = action.refresh_pseudofermion(&links, &xi)?;
    assert_c64_close(
        "refresh phi",
        &field_values(&refreshed)?,
        &julia_field_values("phi_julia.npy")?,
        FIELD_TOLERANCE,
    );
    assert_c64_close(
        "xi Rust payload",
        &field_values(&xi)?,
        &read_c64("xi_rust.npy", &[3, 1, 2, 2, 2, 2])?,
        0.0,
    );
    assert_c64_close(
        "xi Julia parity",
        &field_values(&xi)?,
        &julia_field_values("xi_julia.npy")?,
        FIELD_TOLERANCE,
    );
    assert_c64_close(
        "phi Rust payload",
        &field_values(&phi)?,
        &read_c64("phi_rust.npy", &[3, 1, 2, 2, 2, 2])?,
        0.0,
    );
    assert_c64_close(
        "phi Julia parity",
        &field_values(&phi)?,
        &julia_field_values("phi_julia.npy")?,
        FIELD_TOLERANCE,
    );

    let evaluated = action.evaluate(&links, &phi)?;
    assert_f64_close(
        "action",
        &[evaluated.action],
        &[meta["action"]["value"].as_f64().unwrap()],
        FIELD_TOLERANCE,
    );
    assert_c64_close(
        "action X",
        &field_values(&evaluated.x)?,
        &julia_field_values("action_x_julia.npy")?,
        FIELD_TOLERANCE,
    );
    assert_eq!(evaluated.solver_reports.len(), 15);
    assert_reports_match("action", &evaluated.solver_reports)?;
    for report in &evaluated.solver_reports {
        assert!(report.true_residual_squared < REPORT_TOLERANCE);
    }

    let force = action.force(&links, &phi)?;
    assert_eq!(force.x.len(), 10);
    assert_eq!(force.y.len(), 10);
    assert_reports_match("force", &force.solver_reports)?;
    for index in 0..10 {
        assert_c64_close(
            &format!("force X {index}"),
            &field_values(&force.x[index])?,
            &julia_field_values(&format!("force_x{index}_julia.npy"))?,
            FIELD_TOLERANCE,
        );
        assert_c64_close(
            &format!("force Y {index}"),
            &field_values(&force.y[index])?,
            &julia_field_values(&format!("force_y{index}_julia.npy"))?,
            FIELD_TOLERANCE,
        );
        assert!(force.solver_reports[index].true_residual_squared < REPORT_TOLERANCE);
    }
    for direction in 0..4 {
        assert_f64_close(
            &format!("force direction {direction}"),
            force.force.tensors()[direction].host_data()?,
            &read_f64(&format!("force{direction}.npy"), &[8, 2, 2, 2, 2])?,
            FIELD_TOLERANCE,
        );
    }

    let mut links_trajectory = links.try_clone()?;
    let mut momentum = load_momentum("p_initial")?;
    let params = StaggeredHmcParams::new(
        5.7,
        0.17,
        meta["trajectory"]["step_size"].as_f64().unwrap(),
        meta["trajectory"]["steps"].as_u64().unwrap() as usize,
        FermionBoundary::new([1, 1, 1, -1])?,
        0.0004,
        64.0,
        SolverParams::new(1.0e-24, 2_000)?,
    )?;
    let initial_action = action.evaluate(&links_trajectory, &phi)?.action;
    let initial_hamiltonian = hamiltonian(&links_trajectory, &momentum, 5.7, initial_action)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    dirac_operators::staggered_leapfrog_trajectory(
        &mut context,
        &mut links_trajectory,
        &mut momentum,
        &phi,
        params,
    )?;
    let proposed_action = action.evaluate(&links_trajectory, &phi)?.action;
    let proposed_hamiltonian = hamiltonian(&links_trajectory, &momentum, 5.7, proposed_action)?;
    assert_f64_close(
        "trajectory H initial",
        &[initial_hamiltonian],
        &[meta["trajectory"]["initial_hamiltonian"].as_f64().unwrap()],
        FIELD_TOLERANCE,
    );
    assert_f64_close(
        "trajectory H proposed",
        &[proposed_hamiltonian],
        &[meta["trajectory"]["proposed_hamiltonian"].as_f64().unwrap()],
        FIELD_TOLERANCE,
    );
    assert_f64_close(
        "trajectory delta H",
        &[proposed_hamiltonian - initial_hamiltonian],
        &[meta["trajectory"]["delta_h"].as_f64().unwrap()],
        FIELD_TOLERANCE,
    );
    for direction in 0..4 {
        assert_f64_close(
            &format!("trajectory p final {direction}"),
            momentum.tensors()[direction].host_data()?,
            &read_f64(&format!("p_final{direction}.npy"), &[8, 2, 2, 2, 2])?,
            FIELD_TOLERANCE,
        );
        assert_c64_close(
            &format!("trajectory U proposed {direction}"),
            links_trajectory.links()[direction].typed().host_data()?,
            &read_c64(&format!("u_proposed{direction}.npy"), &[3, 3, 2, 2, 2, 2])?,
            FIELD_TOLERANCE,
        );
    }

    let state: [u64; 4] = meta["trajectory"]["acceptance_rng_state"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "acceptance RNG state length")?;
    let mut acceptance_rng = ReproducibleRng::from_state(state)?;
    let uniform = acceptance_rng.open_unit_f64();
    assert_eq!(
        uniform.to_bits(),
        meta["trajectory"]["acceptance_uniform_bits"]
            .as_u64()
            .unwrap()
    );
    assert_eq!(
        acceptance_rng.next_u64(),
        meta["trajectory"]["next_raw_word"].as_u64().unwrap()
    );
    assert_eq!(
        uniform
            <= meta["trajectory"]["acceptance_probability"]
                .as_f64()
                .unwrap(),
        meta["trajectory"]["accepted"].as_bool().unwrap()
    );
    assert_eq!(lattice.nv(), 16);
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

fn action_value(
    action: &StaggeredFermiAction,
    links: &GaugeLinks,
    phi: &FermionField,
) -> Result<f64> {
    Ok(action.evaluate(links, phi)?.action)
}

#[test]
fn staggered_force_matches_all_link_generators_and_has_quadratic_fd_trend() -> Result<()> {
    const SELECTED_EPSILON_INDEX: usize = 3;
    let meta = metadata()?;
    let series = meta["comparison"]["finite_difference_series"]
        .as_object()
        .ok_or("missing finite-difference metadata series")?;
    let fd_tolerance = series["tolerance"].as_f64().unwrap();
    assert_eq!(fd_tolerance, 5.0e-7);
    let links = load_links()?;
    let phi = load_field("phi_rust.npy")?;
    let action = StaggeredFermiAction::new(
        0.17,
        FermionBoundary::new([1, 1, 1, -1])?,
        0.0004,
        64.0,
        SolverParams::new(1.0e-24, 2_000)?,
    )?;
    let force = action.force(&links, &phi)?.force;
    let epsilons = [3.2e-1, 1.6e-1, 8.0e-2, 4.0e-2];
    let expected_epsilons = series["epsilons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let expected_max_residuals = series["max_residuals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let expected_ratios = series["global_max_ratios"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let expected_pass_counts = series["pass_counts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    assert_eq!(expected_epsilons.as_slice(), epsilons);
    let mut max_residuals = [0.0_f64; 4];
    let mut pass_counts = [0usize; 4];
    let mut count = 0;
    for direction in 0..4 {
        for site in 0..links.lattice().nv() {
            for generator in 0..8 {
                count += 1;
                let coefficient = force.tensors()[direction].host_data()?[generator + 8 * site];
                let mut errors = [0.0; 4];
                for (index, epsilon) in epsilons.into_iter().enumerate() {
                    let plus = action_value(
                        &action,
                        &left_shifted_links(&links, direction, site, generator, epsilon)?,
                        &phi,
                    )?;
                    let minus = action_value(
                        &action,
                        &left_shifted_links(&links, direction, site, generator, -epsilon)?,
                        &phi,
                    )?;
                    errors[index] = ((plus - minus) / (2.0 * epsilon) - coefficient).abs();
                    max_residuals[index] = max_residuals[index].max(errors[index]);
                    if errors[index] <= fd_tolerance {
                        pass_counts[index] += 1;
                    }
                }
                if errors[SELECTED_EPSILON_INDEX] > fd_tolerance {
                    panic!("force FD failed at ({direction},{site},{generator}): {errors:?}");
                }
            }
        }
    }
    let ratios = [
        max_residuals[0] / max_residuals[1],
        max_residuals[1] / max_residuals[2],
        max_residuals[2] / max_residuals[3],
    ];
    eprintln!("Task E force FD count={count} passes={}/{count} residuals={max_residuals:?} ratios={ratios:?} pass_counts={pass_counts:?}", pass_counts[SELECTED_EPSILON_INDEX]);
    assert_eq!(count, 512);
    assert_eq!(expected_pass_counts.as_slice(), pass_counts);
    assert_eq!(pass_counts[SELECTED_EPSILON_INDEX], count);
    assert!(pass_counts[SELECTED_EPSILON_INDEX - 1] < count);
    assert_eq!(
        series["selected_epsilon"].as_f64(),
        Some(epsilons[SELECTED_EPSILON_INDEX])
    );
    assert_eq!(series["selected_pass_count"].as_u64(), Some(count as u64));
    assert!(max_residuals[2] > fd_tolerance);
    assert!(max_residuals[SELECTED_EPSILON_INDEX] <= fd_tolerance);
    assert_f64_close(
        "FD max residual metadata",
        &max_residuals,
        &expected_max_residuals,
        1.0e-12,
    );
    assert_f64_close(
        "FD global ratio metadata",
        &ratios,
        &expected_ratios,
        1.0e-10,
    );
    assert!(ratios.iter().all(|ratio| (2.0..8.0).contains(ratio)));
    Ok(())
}

#[test]
fn staggered_trajectory_is_reversible_and_rollback_is_transactional() -> Result<()> {
    let links = load_links()?;
    let phi = load_field("phi_rust.npy")?;
    let action = StaggeredFermiAction::new(
        0.17,
        FermionBoundary::new([1, 1, 1, -1])?,
        0.0004,
        64.0,
        SolverParams::new(1.0e-24, 2_000)?,
    )?;
    let params = StaggeredHmcParams::new(
        5.7,
        0.17,
        0.001,
        2,
        FermionBoundary::new([1, 1, 1, -1])?,
        0.0004,
        64.0,
        SolverParams::new(1.0e-24, 2_000)?,
    )?;
    let mut forward_links = links.try_clone()?;
    let initial_links = links.try_clone()?;
    let mut forward_momentum = load_momentum("p_initial")?;
    let initial_momentum = load_momentum("p_initial")?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    dirac_operators::staggered_leapfrog_trajectory(
        &mut context,
        &mut forward_links,
        &mut forward_momentum,
        &phi,
        params,
    )?;
    let mut reverse_momentum = negate_momentum(&forward_momentum)?;
    dirac_operators::staggered_leapfrog_trajectory(
        &mut context,
        &mut forward_links,
        &mut reverse_momentum,
        &phi,
        params,
    )?;
    let link_residual = max_link_difference(&forward_links, &initial_links)?;
    let momentum_residual = max_momentum_reversal(&reverse_momentum, &initial_momentum)?;
    eprintln!("Task E reversibility link={link_residual:.17e} momentum={momentum_residual:.17e}");
    assert!(link_residual <= 5.0e-9);
    assert!(momentum_residual <= 5.0e-9);

    let mut bad_links = links.try_clone()?;
    let old_links = bad_links.try_clone()?;
    let mut bad_momentum = load_momentum("p_initial")?;
    let old_momentum = load_momentum("p_initial")?;
    let bad_phi = FermionField::zeros(LatticeShape4::new([1, 1, 1, 1])?, 1)?;
    let error = dirac_operators::staggered_leapfrog_trajectory(
        &mut context,
        &mut bad_links,
        &mut bad_momentum,
        &bad_phi,
        params,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        dirac_operators::DiracError::LatticeMismatch { .. }
    ));
    assert_eq!(
        bad_links.links()[0].typed().host_data()?,
        old_links.links()[0].typed().host_data()?
    );
    assert_eq!(
        bad_momentum.tensors()[0].host_data()?,
        old_momentum.tensors()[0].host_data()?
    );
    assert_eq!(action.lambda_low(), 0.0004);

    let failing_params = StaggeredHmcParams::new(
        5.7,
        0.17,
        0.001,
        2,
        FermionBoundary::new([1, 1, 1, -1])?,
        0.0004,
        64.0,
        SolverParams::new(1.0e-24, 1)?,
    )?;
    let mut failed_links = links.try_clone()?;
    let failed_links_before = failed_links.try_clone()?;
    let mut failed_momentum = load_momentum("p_initial")?;
    let failed_momentum_before = load_momentum("p_initial")?;
    assert!(dirac_operators::staggered_leapfrog_trajectory(
        &mut context,
        &mut failed_links,
        &mut failed_momentum,
        &phi,
        failing_params,
    )
    .is_err());
    assert_eq!(
        failed_links.links()[0].typed().host_data()?,
        failed_links_before.links()[0].typed().host_data()?
    );
    assert_eq!(
        failed_momentum.tensors()[0].host_data()?,
        failed_momentum_before.tensors()[0].host_data()?
    );
    Ok(())
}

fn negate_momentum(momentum: &TaGaugeField) -> Result<TaGaugeField> {
    let tensors = (0..4)
        .map(|direction| {
            let values = momentum.tensors()[direction]
                .host_data()?
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>();
            Ok(TypedTensor::from_vec_col_major(
                momentum.tensors()[direction].shape().to_vec(),
                values,
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TaGaugeField::new(
        tensors.try_into().map_err(|_| "four momentum")?,
        momentum.lattice(),
    )?)
}

fn max_link_difference(left: &GaugeLinks, right: &GaugeLinks) -> Result<f64> {
    let mut maximum = 0.0_f64;
    for direction in 0..4 {
        for (a, b) in left.links()[direction]
            .typed()
            .host_data()?
            .iter()
            .zip(right.links()[direction].typed().host_data()?)
        {
            maximum = maximum.max((*a - *b).norm());
        }
    }
    Ok(maximum)
}

fn max_momentum_reversal(actual: &TaGaugeField, initial: &TaGaugeField) -> Result<f64> {
    let mut maximum = 0.0_f64;
    for direction in 0..4 {
        for (actual, initial) in actual.tensors()[direction]
            .host_data()?
            .iter()
            .zip(initial.tensors()[direction].host_data()?)
        {
            maximum = maximum.max((actual + initial).abs());
        }
    }
    Ok(maximum)
}

#[test]
fn staggered_sampler_and_full_hmc_advance_one_shared_rng_stream() -> Result<()> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let links = gaugefields::cold_su3(lattice)?;
    let action = StaggeredFermiAction::new(
        0.17,
        FermionBoundary::new([1, 1, 1, -1])?,
        0.0004,
        64.0,
        SolverParams::new(1.0e-20, 256)?,
    )?;
    let state = [41, 43, 47, 53];
    let mut rng = ReproducibleRng::from_state(state)?;
    let xi = action.sample_xi(lattice, &mut rng)?;
    let mut replay = ReproducibleRng::from_state(state)?;
    for color in 0..3 {
        let pair = replay.standard_normal_pair();
        let expected = Complex64::new(
            std::f64::consts::FRAC_1_SQRT_2 * pair[0],
            std::f64::consts::FRAC_1_SQRT_2 * pair[1],
        );
        assert_eq!(xi.component(color, 0, 0)?, expected);
    }
    assert_eq!(rng.next_u64(), replay.next_u64());

    let params = StaggeredHmcParams::new(
        5.7,
        0.17,
        0.0001,
        1,
        FermionBoundary::new([1, 1, 1, -1])?,
        0.0004,
        64.0,
        SolverParams::new(1.0e-20, 256)?,
    )?;
    let mut links = links;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let mut hmc_rng = ReproducibleRng::from_state([101, 103, 107, 109])?;
    let mut expected_rng = ReproducibleRng::from_state([101, 103, 107, 109])?;
    let _ = gaugefields::sample_momentum(lattice, &mut expected_rng)?;
    let _ = params.action().sample_xi(lattice, &mut expected_rng)?;
    let outcome =
        dirac_operators::staggered_hmc_update(&mut context, &mut links, params, &mut hmc_rng)?;
    assert!(outcome.initial_hamiltonian.is_finite());
    let expected_draw = expected_rng.open_unit_f64();
    assert!(expected_draw.is_sign_positive());
    assert_eq!(hmc_rng.next_u64(), expected_rng.next_u64());
    Ok(())
}
