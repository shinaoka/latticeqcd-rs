use dirac_operators::{
    multi_shift_cg, FermionBoundary, FermionField, FermionOperator, SolverParams, StaggeredDirac,
};
use gaugefields::{GaugeLinkTensor, GaugeLinks, LatticeShape4};
use num_complex::Complex64;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tenferro_tensor::TypedTensor;

type C = Complex64;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/fermions_task_d")
}

fn read_npy<T: npyz::Deserialize>(name: &str) -> Result<(Vec<T>, Vec<usize>), Box<dyn Error>> {
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
        .collect::<Result<Vec<_>, _>>()?;
    Ok((npy.into_vec::<T>()?, shape))
}

fn read_c64(name: &str, shape: &[usize]) -> Result<Vec<C>, Box<dyn Error>> {
    let (values, found) = read_npy::<C>(name)?;
    if found != shape {
        return Err(format!("{name} shape {found:?}, expected {shape:?}").into());
    }
    Ok(values)
}

fn read_f64(name: &str, shape: &[usize]) -> Result<Vec<f64>, Box<dyn Error>> {
    let (values, found) = read_npy::<f64>(name)?;
    if found != shape {
        return Err(format!("{name} shape {found:?}, expected {shape:?}").into());
    }
    Ok(values)
}

fn field_values(field: &FermionField) -> Result<Vec<C>, Box<dyn Error>> {
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

fn field_from_values(
    lattice: LatticeShape4,
    values: Vec<C>,
) -> Result<FermionField, Box<dyn Error>> {
    Ok(FermionField::from_vec_col_major(lattice, 1, values)?)
}

fn julia_one_component_field(values: &[C], shape: &[usize]) -> Result<Vec<C>, Box<dyn Error>> {
    if shape != [3, 2, 2, 2, 2, 1] {
        return Err(format!("unexpected Julia field shape {shape:?}").into());
    }
    // With a singleton component axis, Julia `[3,NX,NY,NZ,NT,1]` and Rust
    // `[3,1,NX,NY,NZ,NT]` have the same color-fast column-major payload.
    Ok(values.to_vec())
}

fn max_abs(left: &[C], right: &[C]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

fn assert_close(label: &str, left: &[C], right: &[C], tolerance: f64) {
    let residual = max_abs(left, right);
    eprintln!("{label}: max_abs_residual={residual:.17e}");
    assert!(residual <= tolerance, "{label}: residual={residual:e}");
}

fn assert_scalar_close(label: &str, actual: f64, expected: f64) {
    let scale = actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE);
    let residual = (actual - expected).abs();
    eprintln!("{label}: residual={residual:.17e}, actual={actual:.17e}, expected={expected:.17e}");
    assert!(residual <= 1.0e-8 * scale, "{label}: residual={residual:e}");
}

fn load_links() -> Result<GaugeLinks, Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let mut links = Vec::with_capacity(4);
    for direction in 0..4 {
        let values = read_c64(&format!("u{direction}.npy"), &[3, 3, 2, 2, 2, 2])?;
        let tensor = TypedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values)?;
        links.push(GaugeLinkTensor::from_typed(tensor, lattice)?);
    }
    Ok(GaugeLinks::new(
        links.try_into().map_err(|_| "four links")?,
    )?)
}

fn load_field(name: &str) -> Result<FermionField, Box<dyn Error>> {
    Ok(FermionField::from_vec_col_major(
        LatticeShape4::new([2, 2, 2, 2])?,
        1,
        read_c64(name, &[3, 1, 2, 2, 2, 2])?,
    )?)
}

fn assert_declared_files(metadata: &Value) -> Result<(), Box<dyn Error>> {
    let mut declared = metadata["files"]
        .as_array()
        .ok_or("files must be an array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or("file must be a string")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut actual = fs::read_dir(fixture_dir())?
        .map(|entry| -> Result<String, Box<dyn Error>> {
            Ok(entry?.file_name().to_string_lossy().into_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    declared.sort();
    actual.retain(|name| name != "metadata.json");
    actual.sort();
    assert_eq!(declared, actual);
    for name in declared {
        let path = fixture_dir().join(&name);
        assert!(path.is_file(), "missing declared payload {name}");
        match name.as_str() {
            "eta.npy" => assert_eq!(read_npy::<f64>(&name)?.1, [4, 2, 2, 2, 2]),
            "eta_impulses.npy" => assert_eq!(read_npy::<f64>(&name)?.1, [4, 2, 2]),
            "shifted_reports.json" => {
                let _: Value = serde_json::from_slice(&fs::read(path)?)?;
            }
            _ if name.ends_with("_julia.npy") => {
                assert_eq!(read_npy::<C>(&name)?.1, [3, 2, 2, 2, 2, 1]);
            }
            _ if name.ends_with("_rust.npy") => {
                assert_eq!(read_npy::<C>(&name)?.1, [3, 1, 2, 2, 2, 2]);
            }
            _ => assert_eq!(read_npy::<C>(&name)?.1, [3, 3, 2, 2, 2, 2]),
        }
    }
    Ok(())
}

fn relative_residual<O: FermionOperator>(
    operator: &O,
    solution: &FermionField,
    rhs: &FermionField,
    shift: f64,
) -> Result<f64, Box<dyn Error>> {
    let mut applied = FermionField::zeros(solution.lattice(), solution.components())?;
    operator.apply_into(&mut applied, solution)?;
    let mut residual = field_values(rhs)?;
    for ((residual_value, applied_value), solution_value) in residual
        .iter_mut()
        .zip(field_values(&applied)?)
        .zip(field_values(solution)?)
    {
        *residual_value -= applied_value + shift * solution_value;
    }
    let norm = residual.iter().map(C::norm_sqr).sum::<f64>();
    Ok((norm / rhs.norm_squared()?).sqrt())
}

#[test]
fn generated_task_d_fixture_parses_everything_and_matches_rust() -> Result<(), Box<dyn Error>> {
    let directory = fixture_dir();
    let metadata: Value = serde_json::from_slice(&fs::read(directory.join("metadata.json"))?)?;
    assert_eq!(metadata["schema"], "fermions_task_d.v1");
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["nc"], 3);
    assert_eq!(metadata["components"], 1);
    assert_eq!(metadata["mass"], 0.17);
    assert_eq!(
        metadata["boundaries"],
        serde_json::json!({
            "periodic": [1, 1, 1, 1],
            "default_antiperiodic": [1, 1, 1, -1]
        })
    );
    assert_eq!(metadata["shifts"], serde_json::json!([0.31, 0.0, 0.07]));
    assert_eq!(
        metadata["solver_parameters"],
        serde_json::json!({
            "absolute_squared_tolerance": 1.0e-24,
            "max_iterations": 2_000,
            "julia_solver_keywords": ["eps", "maxsteps", "verbose"]
        })
    );
    assert_eq!(
        metadata["gaugefields_jl"],
        serde_json::json!({
            "package": "Gaugefields.jl",
            "version": "0.7.2",
            "commit": "9e5719970770f4497405a856315c90bef7f74449",
            "clean": true
        })
    );
    assert_eq!(
        metadata["latticediracoperators_jl"],
        serde_json::json!({
            "package": "LatticeDiracOperators.jl",
            "version": "0.6.4",
            "commit": "bdef628184597815ba3e0cddf2536df767e78a02",
            "clean": true
        })
    );
    assert_eq!(
        metadata["layout"]["permutation"],
        serde_json::json!([1, 6, 2, 3, 4, 5])
    );
    assert_eq!(
        metadata["source_urls"],
        serde_json::json!([
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/StaggeredFermion/StaggeredFermion.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/StaggeredFermion/StaggeredFermion_4D_nowing.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/cgmethods.jl",
            "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/nowing/gaugefields_4D_nowing.jl"
        ])
    );
    assert_eq!(
        metadata["source_functions"],
        serde_json::json!([
            "Staggered_Dirac_operator",
            "Dx!",
            "shift_fermion",
            "shifted_fermion!",
            "staggered_U",
            "DdagD_Staggered_operator",
            "LinearAlgebra.mul!",
            "shiftedcg"
        ])
    );
    assert_eq!(
        metadata["entrypoint_map"],
        serde_json::json!([
            {"julia": "Staggered_Dirac_operator", "julia_source": "src/StaggeredFermion/StaggeredFermion.jl:25-78", "rust": "staggered.rs::StaggeredDirac::with_boundary"},
            {"julia": "Dx!", "julia_source": "src/StaggeredFermion/StaggeredFermion_4D_nowing.jl:43-80", "rust": "staggered.rs::StaggeredDirac::apply_hopping_to_data"},
            {"julia": "staggered_U", "julia_source": "Gaugefields.jl/src/4D/nowing/gaugefields_4D_nowing.jl:459-504", "rust": "staggered.rs::staggered_eta + Mat3::scaled"},
            {"julia": "shift_fermion/shifted_fermion!", "julia_source": "src/StaggeredFermion/StaggeredFermion_4D_nowing.jl:99-198", "rust": "staggered.rs::StaggeredDirac::neighbor"},
            {"julia": "LinearAlgebra.mul! + DdagD_Staggered_operator", "julia_source": "src/StaggeredFermion/StaggeredFermion.jl:166-243", "rust": "staggered.rs::StaggeredNormalOperator + StaggeredClosedNormalOperator"},
            {"julia": "Dirac_operators.shiftedcg", "julia_source": "src/cgmethods.jl:872-968", "rust": "solvers.rs::multi_shift_cg"}
        ])
    );
    assert_eq!(
        metadata["source_functions"].as_array().map(Vec::len),
        Some(8)
    );
    assert_eq!(metadata["entrypoint_map"].as_array().map(Vec::len), Some(6));
    assert_eq!(
        metadata["layout"],
        serde_json::json!({
            "julia_shape": "[3,NX,NY,NZ,NT,1]",
            "rust_shape": "[3,1,NX,NY,NZ,NT]",
            "conversion": "permutedims(array, (1, 6, 2, 3, 4, 5))",
            "permutation": [1, 6, 2, 3, 4, 5],
            "site_order": "x fastest"
        })
    );
    assert_eq!(
        metadata["eta"]["formula"],
        serde_json::json!([
            "eta_0=1",
            "eta_1=(-1)^x",
            "eta_2=(-1)^(x+y)",
            "eta_3=(-1)^(x+y+z)"
        ])
    );
    assert_eq!(metadata["eta"]["coordinates"], "zero-based");
    assert_eq!(
        metadata["eta"]["files"],
        serde_json::json!(["eta.npy", "eta_impulses.npy"])
    );
    assert_eq!(
        metadata["eta"]["impulse_layout"],
        "[direction, lower_or_upper_source, eta_or_default_boundary_sign]"
    );
    assert_eq!(
        metadata["eta"]["wrap_sign"],
        "boundary sign applied once on each wrapped fermion hop"
    );
    assert_eq!(
        metadata["construction"],
        "explicit diagonal nontrivial SU(3) links, one-component input and rhs from fixed formulas; no RNG or global state"
    );
    assert_eq!(
        metadata["normal"],
        serde_json::json!({
            "composition": "Ddag(D(x))",
            "closed_form": "mass^2*x-K(K(x))",
            "antihermitian_identity": "Kdag=-K"
        })
    );
    assert_eq!(
        metadata["generator"],
        serde_json::json!({
            "script": "fixtures/generate.jl",
            "mode": "fermions_task_d",
            "randomness": "none"
        })
    );
    assert_eq!(metadata["files"].as_array().map(Vec::len), Some(37));
    assert_eq!(
        metadata["comparison"],
        serde_json::json!({
            "operator_max_abs_tolerance": 2.0e-12,
            "antihermiticity_tolerance": 2.0e-12,
            "normal_composition_tolerance": 2.0e-12,
            "shifted_true_relative_residual_tolerance": 1.0e-11,
            "criterion": "maximum absolute complex-component operator residual and fresh relative shifted residual"
        })
    );
    let solver_tolerance = metadata["solver_parameters"]["absolute_squared_tolerance"]
        .as_f64()
        .ok_or("solver tolerance")?;
    let max_iterations = metadata["solver_parameters"]["max_iterations"]
        .as_u64()
        .ok_or("maximum iterations")? as usize;
    let operator_tolerance = metadata["comparison"]["operator_max_abs_tolerance"]
        .as_f64()
        .ok_or("operator tolerance")?;
    let antihermiticity_tolerance = metadata["comparison"]["antihermiticity_tolerance"]
        .as_f64()
        .ok_or("antihermiticity tolerance")?;
    let normal_composition_tolerance = metadata["comparison"]["normal_composition_tolerance"]
        .as_f64()
        .ok_or("normal composition tolerance")?;
    let shifted_true_relative_tolerance = metadata["comparison"]
        ["shifted_true_relative_residual_tolerance"]
        .as_f64()
        .ok_or("shifted true residual tolerance")?;
    assert_declared_files(&metadata)?;

    let eta = read_f64("eta.npy", &[4, 2, 2, 2, 2])?;
    for direction in 0..4 {
        for site in 0..16 {
            let x: usize = site % 2;
            let y: usize = (site / 2) % 2;
            let z: usize = (site / 4) % 2;
            let expected = match direction {
                0 => 1.0,
                1 => {
                    if x.is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                2 => {
                    if (x + y).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                3 => {
                    if (x + y + z).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => unreachable!(),
            };
            assert_eq!(eta[direction + 4 * site], expected);
        }
    }
    let eta_impulses = read_f64("eta_impulses.npy", &[4, 2, 2])?;
    for direction in 0..4 {
        for side in 0..2 {
            let coordinate: [usize; 3] = if side == 0 {
                [0, 0, 0]
            } else {
                match direction {
                    0 => [1, 0, 0],
                    1 => [0, 1, 0],
                    2 => [0, 0, 1],
                    3 => [0, 0, 0],
                    _ => unreachable!(),
                }
            };
            let phase = match direction {
                0 => 1.0,
                1 => {
                    if coordinate[0].is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                2 => {
                    if (coordinate[0] + coordinate[1]).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                3 => {
                    if (coordinate[0] + coordinate[1] + coordinate[2]).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => unreachable!(),
            };
            assert_eq!(eta_impulses[direction + 4 * side], phase);
            assert_eq!(
                eta_impulses[direction + 4 * side + 8],
                if direction == 3 { -1.0 } else { 1.0 }
            );
        }
    }

    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = load_links()?;
    let input = load_field("input_rust.npy")?;
    let rhs = load_field("rhs_rust.npy")?;
    assert_close(
        "input Julia-to-Rust layout",
        &field_values(&input)?,
        &julia_one_component_field(
            &read_c64("input_julia.npy", &[3, 2, 2, 2, 2, 1])?,
            &[3, 2, 2, 2, 2, 1],
        )?,
        0.0,
    );
    assert_close(
        "rhs Julia-to-Rust layout",
        &field_values(&rhs)?,
        &julia_one_component_field(
            &read_c64("rhs_julia.npy", &[3, 2, 2, 2, 2, 1])?,
            &[3, 2, 2, 2, 2, 1],
        )?,
        0.0,
    );

    for (case, operator) in [
        (
            "periodic",
            StaggeredDirac::with_boundary(&links, 0.17, FermionBoundary::new([1, 1, 1, 1])?),
        ),
        ("default_antiperiodic", StaggeredDirac::new(&links, 0.17)),
    ] {
        let operator = operator?;
        let mut d = FermionField::zeros(lattice, 1)?;
        let mut ddag = FermionField::zeros(lattice, 1)?;
        let mut normal = FermionField::zeros(lattice, 1)?;
        let mut closed = FermionField::zeros(lattice, 1)?;
        operator.apply_into(&mut d, &input)?;
        operator.adjoint().apply_into(&mut ddag, &input)?;
        operator.normal().apply_into(&mut normal, &input)?;
        operator
            .normal_closed_form()
            .apply_into(&mut closed, &input)?;
        let k = field_from_values(
            lattice,
            field_values(&d)?
                .into_iter()
                .zip(field_values(&ddag)?)
                .map(|(left, right)| 0.5 * (left - right))
                .collect(),
        )?;
        assert_close(
            &format!("normal composition {case} vs closed form"),
            &field_values(&normal)?,
            &field_values(&closed)?,
            normal_composition_tolerance,
        );
        for (label, value) in [
            ("d", &d),
            ("ddag", &ddag),
            ("k", &k),
            ("normal_composition", &normal),
            ("normal_closed", &closed),
        ] {
            let expected = read_c64(&format!("{label}_{case}_rust.npy"), &[3, 1, 2, 2, 2, 2])?;
            assert_close(
                &format!("{label} {case} Rust payload"),
                &field_values(value)?,
                &expected,
                operator_tolerance,
            );
            let julia = read_c64(&format!("{label}_{case}_julia.npy"), &[3, 2, 2, 2, 2, 1])?;
            assert_close(
                &format!("{label} {case} Julia parity"),
                &field_values(value)?,
                &julia_one_component_field(&julia, &[3, 2, 2, 2, 2, 1])?,
                operator_tolerance,
            );
        }
    }

    let default_operator = StaggeredDirac::new(&links, 0.17)?;
    let mut d_input = FermionField::zeros(lattice, 1)?;
    let mut ddag_input = FermionField::zeros(lattice, 1)?;
    default_operator.apply_into(&mut d_input, &input)?;
    default_operator
        .adjoint()
        .apply_into(&mut ddag_input, &input)?;
    let k_input = field_from_values(
        lattice,
        field_values(&d_input)?
            .into_iter()
            .zip(field_values(&ddag_input)?)
            .map(|(left, right)| 0.5 * (left - right))
            .collect(),
    )?;
    let other = field_from_values(
        lattice,
        (0..input.len())
            .map(|index| C::new(0.013 * (index + 1) as f64, -0.007 * index as f64))
            .collect(),
    )?;
    let mut d_other = FermionField::zeros(lattice, 1)?;
    let mut ddag_other = FermionField::zeros(lattice, 1)?;
    default_operator.apply_into(&mut d_other, &other)?;
    default_operator
        .adjoint()
        .apply_into(&mut ddag_other, &other)?;
    let k_other = field_from_values(
        lattice,
        field_values(&d_other)?
            .into_iter()
            .zip(field_values(&ddag_other)?)
            .map(|(left, right)| 0.5 * (left - right))
            .collect(),
    )?;
    let antihermitian_residual =
        (k_input.inner_product(&other)? + input.inner_product(&k_other)?).norm();
    eprintln!("K antihermitian residual={antihermitian_residual:.17e}");
    assert!(antihermitian_residual <= antihermiticity_tolerance);

    let reports_meta: Value =
        serde_json::from_slice(&fs::read(directory.join("shifted_reports.json"))?)?;
    let reports = reports_meta["reports"].as_array().ok_or("reports")?;
    assert_eq!(reports.len(), 3);
    let expected_initial_residual = reports_meta["initial_residual_squared"]
        .as_f64()
        .ok_or("initial residual")?;
    let normal = default_operator.normal();
    let mut solutions = (0..3)
        .map(|_| FermionField::zeros(lattice, 1))
        .collect::<Result<Vec<_>, _>>()?;
    let shifts = [0.31, 0.0, 0.07];
    let rust_reports = multi_shift_cg(
        &mut solutions,
        &normal,
        &rhs,
        &shifts,
        SolverParams::new(solver_tolerance, max_iterations)?,
    )?;
    assert_eq!(rust_reports.len(), reports.len());
    for (index, report) in rust_reports.iter().enumerate() {
        assert_eq!(report.shift, shifts[index]);
        assert_eq!(report.maximum_iterations, max_iterations);
        assert_eq!(report.tolerance, solver_tolerance);
        assert_eq!(report.convergence_branch.to_string(), "updated_residual");
        assert_eq!(
            report.iterations,
            reports[index]["iterations"].as_u64().unwrap() as usize,
            "shift {index}: Rust report={:?}, Julia report={:?}",
            report,
            reports[index]
        );
        assert_scalar_close(
            &format!("shift {index} initial residual"),
            report.initial_residual_squared,
            expected_initial_residual,
        );
        assert_scalar_close(
            &format!("shift {index} recursive residual"),
            report.recursive_residual_squared,
            reports[index]["recursive_residual_squared"]
                .as_f64()
                .ok_or("recursive residual")?,
        );
        let julia_true_residual = reports[index]["true_residual_squared"]
            .as_f64()
            .ok_or("true residual")?;
        assert!(julia_true_residual.is_finite());
        let true_residual_difference = (report.true_residual_squared - julia_true_residual).abs();
        eprintln!(
            "shift {index} true residual: Rust={:.17e}, Julia={:.17e}, difference={true_residual_difference:.17e}",
            report.true_residual_squared, julia_true_residual
        );
        assert!(
            true_residual_difference <= solver_tolerance,
            "shift {index}: true residual difference={true_residual_difference:e}"
        );
        assert!(report.true_residual_squared < solver_tolerance);
        assert_eq!(
            reports[index]["absolute_squared_tolerance"],
            metadata["solver_parameters"]["absolute_squared_tolerance"]
        );
        assert_eq!(
            reports[index]["maximum_iterations"],
            serde_json::json!(max_iterations)
        );
        assert_eq!(reports[index]["convergence_branch"], "updated_residual");
        let relative = relative_residual(&normal, &solutions[index], &rhs, shifts[index])?;
        eprintln!(
            "shift {} true relative residual={relative:.17e}",
            shifts[index]
        );
        assert!(relative <= shifted_true_relative_tolerance);
        let expected = read_c64(&format!("shift{index}_rust.npy"), &[3, 1, 2, 2, 2, 2])?;
        assert_close(
            &format!("shift {index} Rust payload"),
            &field_values(&solutions[index])?,
            &expected,
            operator_tolerance,
        );
        let expected_julia = read_c64(&format!("shift{index}_julia.npy"), &[3, 2, 2, 2, 2, 1])?;
        assert_close(
            &format!("shift {index} Julia parity"),
            &field_values(&solutions[index])?,
            &julia_one_component_field(&expected_julia, &[3, 2, 2, 2, 2, 1])?,
            operator_tolerance,
        );
        assert_eq!(reports[index]["shift"].as_f64(), Some(shifts[index]));
        assert!(reports[index]["true_residual_squared"].is_number());
    }
    Ok(())
}
