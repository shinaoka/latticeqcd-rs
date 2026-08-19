#![cfg(feature = "fermions")]

use dirac_operators::{
    bicgstab, FermionBoundary, FermionField, FermionOperator, SolverMethod, SolverParams,
    StaggeredDirac, WilsonDirac,
};
use gaugefields::{GaugeLinkTensor, GaugeLinks, LatticeShape4, ReproducibleRng};
use measurements::fermions::{pion_correlator, stochastic_chiral_condensate};
use num_complex::Complex64;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tenferro_tensor::TypedTensor;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const NC: usize = 3;
const TOLERANCE: f64 = 1.0e-24;
const MAX_ITERATIONS: usize = 2_000;

type C = Complex64;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/fermion_measurements_phase4")
}

type NpyPayload = (String, Vec<usize>, Vec<u8>);

fn parse_npy(path: &Path) -> Result<NpyPayload, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err(format!("{} is not an NPY file", path.display()).into());
    }
    let major = *bytes.get(6).ok_or("missing NPY major version")?;
    let (header_len, data_start) = match major {
        1 => (
            u16::from_le_bytes(
                bytes
                    .get(8..10)
                    .ok_or("missing NPY v1 header length")?
                    .try_into()?,
            ) as usize,
            10usize,
        ),
        2 | 3 => (
            u32::from_le_bytes(
                bytes
                    .get(8..12)
                    .ok_or("missing NPY v2 header length")?
                    .try_into()?,
            ) as usize,
            12usize,
        ),
        other => return Err(format!("unsupported NPY version {other}").into()),
    };
    let header_end = data_start
        .checked_add(header_len)
        .ok_or("NPY header overflow")?;
    let header = std::str::from_utf8(
        bytes
            .get(data_start..header_end)
            .ok_or("short NPY header")?,
    )?;
    if !header.contains("'fortran_order': True") {
        return Err(format!("unsupported NPY layout: {header}").into());
    }
    let descr = if header.contains("'descr': '<c16'") {
        "<c16"
    } else if header.contains("'descr': '<f8'") {
        "<f8"
    } else {
        return Err(format!("unsupported NPY dtype: {header}").into());
    };
    let shape_text = header
        .split_once("'shape': (")
        .ok_or("NPY shape missing")?
        .1
        .split_once(')')
        .ok_or("NPY shape terminator missing")?
        .0;
    let shape = shape_text
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<usize>().map_err(|error| error.into()))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or("NPY shape overflow")?;
    let item_size = if descr == "<c16" { 16 } else { 8 };
    let payload = bytes.get(header_end..).ok_or("NPY payload missing")?;
    if payload.len()
        != count
            .checked_mul(item_size)
            .ok_or("NPY byte count overflow")?
    {
        return Err("NPY payload length mismatch".into());
    }
    Ok((descr.to_owned(), shape, payload.to_owned()))
}

fn read_npy_complex(path: &Path) -> Result<(Vec<C>, Vec<usize>), Box<dyn Error>> {
    let (descr, shape, payload) = parse_npy(path)?;
    if descr != "<c16" {
        return Err(format!("{} is not complex data", path.display()).into());
    }
    let mut values = Vec::with_capacity(payload.len() / 16);
    for chunk in payload.chunks_exact(16) {
        values.push(C::new(
            f64::from_le_bytes(chunk[0..8].try_into()?),
            f64::from_le_bytes(chunk[8..16].try_into()?),
        ));
    }
    Ok((values, shape))
}

fn read_npy_f64(path: &Path) -> Result<(Vec<f64>, Vec<usize>), Box<dyn Error>> {
    let (descr, shape, payload) = parse_npy(path)?;
    if descr != "<f8" {
        return Err(format!("{} is not f64 data", path.display()).into());
    }
    let mut values = Vec::with_capacity(payload.len() / 8);
    for chunk in payload.chunks_exact(8) {
        values.push(f64::from_le_bytes(chunk.try_into()?));
    }
    Ok((values, shape))
}

fn load_scalar_payload(dir: &Path, name: &str) -> Result<(Vec<f64>, Vec<usize>), Box<dyn Error>> {
    let (julia, julia_shape) = read_npy_f64(&dir.join(format!("{name}_julia.npy")))?;
    let (rust, rust_shape) = read_npy_f64(&dir.join(format!("{name}_rust.npy")))?;
    assert_eq!(julia_shape, rust_shape);
    assert_real_close(&format!("{name} Julia/Rust scalar"), &julia, &rust, 0.0);
    Ok((rust, rust_shape))
}

fn transpose_julia_field(values: &[C], shape: &[usize]) -> Result<Vec<C>, Box<dyn Error>> {
    if shape.len() != 6 || shape[0] != NC {
        return Err(format!("unexpected Julia field shape {shape:?}").into());
    }
    let [_, nx, ny, nz, nt, components] = <[usize; 6]>::try_from(shape)?;
    let expected = NC * nx * ny * nz * nt * components;
    if values.len() != expected {
        return Err("Julia field payload length mismatch".into());
    }
    let volume = nx * ny * nz * nt;
    let mut result = vec![C0; expected];
    for site in 0..volume {
        for component in 0..components {
            for color in 0..NC {
                let julia_offset = color + NC * (site + volume * component);
                let rust_offset = color + NC * (component + components * site);
                result[rust_offset] = values[julia_offset];
            }
        }
    }
    Ok(result)
}

fn field_values(field: &FermionField) -> Result<Vec<C>, Box<dyn Error>> {
    let mut values = Vec::with_capacity(field.len());
    for site in 0..field.lattice().nv() {
        for component in 0..field.components() {
            for color in 0..NC {
                values.push(field.component(color, component, site)?);
            }
        }
    }
    Ok(values)
}

fn max_abs(left: &[C], right: &[C]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

fn relative_norm(left: &[C], right: &[C]) -> f64 {
    let difference_squared = left
        .iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>();
    let expected_squared = right.iter().map(|value| value.norm_sqr()).sum::<f64>();
    if expected_squared == 0.0 {
        if difference_squared == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (difference_squared / expected_squared).sqrt()
    }
}

fn assert_complex_close(label: &str, actual: &[C], expected: &[C], abs_tolerance: f64) {
    let residual = max_abs(actual, expected);
    let relative = relative_norm(actual, expected);
    eprintln!("{label}: max_abs={residual:.17e}, relative_norm={relative:.17e}");
    assert!(
        residual <= abs_tolerance,
        "{label} absolute residual {residual:.3e}"
    );
    assert!(
        relative <= 2.0e-10,
        "{label} relative residual {relative:.3e}"
    );
}

fn assert_real_close(label: &str, actual: &[f64], expected: &[f64], abs_tolerance: f64) {
    let residual = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("{label}: max_abs={residual:.17e}");
    assert!(residual <= abs_tolerance, "{label} residual {residual:.3e}");
}

fn load_links(dir: &Path) -> Result<GaugeLinks, Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 4])?;
    let expected_shape = vec![NC, NC, 2, 2, 2, 4];
    let mut links = Vec::with_capacity(4);
    for direction in 0..4 {
        let (values, shape) = read_npy_complex(&dir.join(format!("u{direction}.npy")))?;
        assert_eq!(shape, expected_shape);
        let tensor = TypedTensor::from_vec_col_major(shape, values)
            .map_err(|error| format!("link tensor construction failed: {error}"))?;
        links.push(GaugeLinkTensor::from_typed(tensor, lattice)?);
    }
    let links: [GaugeLinkTensor; 4] = links
        .try_into()
        .map_err(|_| "fixture must contain four link directions")?;
    Ok(GaugeLinks::new(links)?)
}

fn load_field_payload(
    dir: &Path,
    name: &str,
    lattice: LatticeShape4,
    components: usize,
) -> Result<(FermionField, Vec<C>), Box<dyn Error>> {
    let (julia, julia_shape) = read_npy_complex(&dir.join(format!("{name}_julia.npy")))?;
    let (rust, rust_shape) = read_npy_complex(&dir.join(format!("{name}_rust.npy")))?;
    assert_eq!(julia_shape, vec![3, 2, 2, 2, 4, components]);
    assert_eq!(rust_shape, vec![3, components, 2, 2, 2, 4]);
    let transposed = transpose_julia_field(&julia, &julia_shape)?;
    assert_complex_close(
        &format!("{name} Julia/Rust layout"),
        &transposed,
        &rust,
        0.0,
    );
    let field = FermionField::from_vec_col_major(lattice, components, rust.clone())?;
    Ok((field, rust))
}

fn relative_residual<O: FermionOperator>(
    operator: &O,
    solution: &FermionField,
    rhs: &FermionField,
) -> Result<f64, Box<dyn Error>> {
    let mut applied = FermionField::zeros(operator.lattice(), operator.components())?;
    operator.apply_into(&mut applied, solution)?;
    let rhs_values = field_values(rhs)?;
    let applied_values = field_values(&applied)?;
    let numerator = rhs_values
        .iter()
        .zip(&applied_values)
        .map(|(rhs, applied)| (*rhs - *applied).norm_sqr())
        .sum::<f64>();
    let denominator = rhs_values.iter().map(|value| value.norm_sqr()).sum::<f64>();
    Ok(numerator.sqrt() / denominator.sqrt())
}

fn consume_report(report: &dirac_operators::SolverReport) {
    assert_eq!(report.method, SolverMethod::BiCgStab);
    assert!(report.iterations <= report.maximum_iterations);
    assert!(report.recursive_residual_squared.is_finite());
    assert!(report.initial_residual_squared.is_finite());
    assert!(report.true_residual_squared.is_finite());
    assert!(report.tolerance.is_finite());
    assert!(report.maximum_iterations > 0);
    assert!(report.restart_count <= report.iterations);
    assert!(matches!(
        report.convergence_branch,
        dirac_operators::ConvergenceBranch::InitialResidual
            | dirac_operators::ConvergenceBranch::IntermediateResidual
            | dirac_operators::ConvergenceBranch::UpdatedResidual
    ));
}

fn consume_julia_diagnostic(report: &Value) {
    assert_eq!(report["method"].as_str(), Some("bicgstab"));
    for key in [
        "iterations",
        "recursive_residual_squared",
        "initial_residual_squared",
        "true_residual_squared",
        "target_residual_squared",
        "maximum_iterations",
        "restart_count",
        "convergence_branch",
    ] {
        assert!(
            !report[key].is_null(),
            "missing Julia diagnostic field {key}"
        );
    }
    assert!(report["recursive_residual_squared"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(report["initial_residual_squared"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(report["true_residual_squared"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(report["target_residual_squared"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert!(report["iterations"].as_u64().is_some());
    assert!(report["maximum_iterations"].as_u64().unwrap() > 0);
    assert!(report["restart_count"].as_u64().is_some());
    assert!(report["convergence_branch"].as_str().is_some());
}

fn source_indices(kind: &str) -> Vec<(usize, usize)> {
    match kind {
        "wilson" => (0..NC)
            .flat_map(|color| (0..4).map(move |component| (color, component)))
            .collect(),
        "staggered" => (0..NC).map(|color| (color, 0)).collect(),
        _ => panic!("unknown fermion kind"),
    }
}

fn legacy_correlator(
    solutions: &[FermionField],
    indices: &[(usize, usize)],
) -> Result<Vec<f64>, Box<dyn Error>> {
    let [nx, ny, nz, nt] = solutions[0].lattice().extents();
    let spatial_volume = nx * ny * nz;
    let components = solutions[0].components();
    let mut values = vec![0.0; nt];
    for (solution, &(source_color, source_component)) in solutions.iter().zip(indices) {
        for site in 0..solution.lattice().nv() {
            let value = solution.component(source_color, source_component, site)?;
            values[site / spatial_volume] += (NC * components) as f64 * value.norm_sqr();
        }
    }
    Ok(values)
}

fn corrected_from_solutions(solutions: &[FermionField]) -> Result<Vec<f64>, Box<dyn Error>> {
    let [nx, ny, nz, nt] = solutions[0].lattice().extents();
    let spatial_volume = nx * ny * nz;
    let mut values = vec![0.0; nt];
    for solution in solutions {
        for site in 0..solution.lattice().nv() {
            for component in 0..solution.components() {
                for color in 0..NC {
                    values[site / spatial_volume] +=
                        solution.component(color, component, site)?.norm_sqr();
                }
            }
        }
    }
    Ok(values)
}

fn run_point_kind(
    dir: &Path,
    links: &GaugeLinks,
    kind: &str,
    metadata: &Value,
) -> Result<(), Box<dyn Error>> {
    let lattice = links.lattice();
    let components = if kind == "wilson" { 4 } else { 1 };
    let boundary = FermionBoundary::new([1, 1, 1, -1])?;
    let solver = SolverParams::new(
        metadata["solver_parameters"]["absolute_squared_tolerance"]
            .as_f64()
            .unwrap(),
        metadata["solver_parameters"]["max_iterations"]
            .as_u64()
            .unwrap() as usize,
    )?;
    let indices = source_indices(kind);
    let mut actual_solutions = Vec::with_capacity(indices.len());

    if kind == "wilson" {
        let operator = WilsonDirac::with_boundary(
            links,
            metadata["wilson"]["kappa"].as_f64().unwrap(),
            boundary,
        )?;
        let result = pion_correlator(&operator, solver)?;
        let (expected_values, expected_shape) = load_scalar_payload(dir, "pion_wilson_corrected")?;
        assert_eq!(expected_shape, vec![4]);
        assert_real_close(
            "Wilson corrected pion",
            &result.values,
            &expected_values,
            2.0e-12,
        );
        assert_eq!(result.solver_reports.len(), indices.len());
        for (source_number, &(color, component)) in indices.iter().enumerate() {
            let (expected_source, _) = load_field_payload(
                dir,
                &format!("wilson_point_source_c{color}_s{component}"),
                lattice,
                components,
            )?;
            let (_, expected_values) = load_field_payload(
                dir,
                &format!("wilson_propagator_c{color}_s{component}"),
                lattice,
                components,
            )?;
            let source = FermionField::point_source(lattice, components, color, component, 0)?;
            assert_complex_close(
                &format!("Wilson source {source_number}"),
                &field_values(&source)?,
                &field_values(&expected_source)?,
                2.0e-12,
            );
            let mut solution = FermionField::zeros(lattice, components)?;
            let report = bicgstab(&mut solution, &operator, &source, solver)?;
            consume_report(&report);
            assert_complex_close(
                &format!("Wilson solution {source_number}"),
                &field_values(&solution)?,
                &expected_values,
                2.0e-12,
            );
            assert!(relative_residual(&operator, &solution, &source)? <= 1.0e-11);
            actual_solutions.push(solution);
        }
        let corrected = corrected_from_solutions(&actual_solutions)?;
        assert_real_close(
            "Wilson independent corrected pion",
            &corrected,
            &result.values,
            2.0e-12,
        );
        let legacy = legacy_correlator(&actual_solutions, &indices)?;
        let (expected_legacy, _) = load_scalar_payload(dir, "pion_wilson_legacy")?;
        assert_real_close("Wilson legacy pion", &legacy, &expected_legacy, 2.0e-12);
        assert!(max_abs_real(&corrected, &legacy) > 1.0e-8);
        let reports = metadata["solver_diagnostics"]["julia"]["wilson_reports"]
            .as_array()
            .unwrap();
        assert_eq!(reports.len(), indices.len());
        for report in reports {
            consume_julia_diagnostic(report);
        }
    } else {
        let operator = StaggeredDirac::with_boundary(
            links,
            metadata["staggered"]["mass"].as_f64().unwrap(),
            boundary,
        )?;
        let result = pion_correlator(&operator, solver)?;
        let (expected_values, expected_shape) =
            load_scalar_payload(dir, "pion_staggered_corrected")?;
        assert_eq!(expected_shape, vec![4]);
        assert_real_close(
            "Staggered corrected pion",
            &result.values,
            &expected_values,
            2.0e-12,
        );
        assert_eq!(result.solver_reports.len(), indices.len());
        for (source_number, &(color, component)) in indices.iter().enumerate() {
            let (expected_source, _) = load_field_payload(
                dir,
                &format!("staggered_point_source_c{color}_s{component}"),
                lattice,
                components,
            )?;
            let (_, expected_values) = load_field_payload(
                dir,
                &format!("staggered_propagator_c{color}_s{component}"),
                lattice,
                components,
            )?;
            let source = FermionField::point_source(lattice, components, color, component, 0)?;
            assert_complex_close(
                &format!("Staggered source {source_number}"),
                &field_values(&source)?,
                &field_values(&expected_source)?,
                2.0e-12,
            );
            let mut solution = FermionField::zeros(lattice, components)?;
            let report = bicgstab(&mut solution, &operator, &source, solver)?;
            consume_report(&report);
            assert_complex_close(
                &format!("Staggered solution {source_number}"),
                &field_values(&solution)?,
                &expected_values,
                2.0e-12,
            );
            assert!(relative_residual(&operator, &solution, &source)? <= 1.0e-11);
            actual_solutions.push(solution);
        }
        let corrected = corrected_from_solutions(&actual_solutions)?;
        assert_real_close(
            "Staggered independent corrected pion",
            &corrected,
            &result.values,
            2.0e-12,
        );
        let legacy = legacy_correlator(&actual_solutions, &indices)?;
        let (expected_legacy, _) = load_scalar_payload(dir, "pion_staggered_legacy")?;
        assert_real_close("Staggered legacy pion", &legacy, &expected_legacy, 2.0e-12);
        assert!(max_abs_real(&corrected, &legacy) > 1.0e-8);
        let reports = metadata["solver_diagnostics"]["julia"]["staggered_reports"]
            .as_array()
            .unwrap();
        assert_eq!(reports.len(), indices.len());
        for report in reports {
            consume_julia_diagnostic(report);
        }
    }
    Ok(())
}

fn max_abs_real(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn canonical_z4(code: u64) -> C {
    match code & 3 {
        0 => C::new(1.0, 0.0),
        1 => C::new(0.0, 1.0),
        2 => C::new(-1.0, 0.0),
        _ => C::new(0.0, -1.0),
    }
}

fn explicit_noise_field(
    lattice: LatticeShape4,
    codes: &[u64],
) -> Result<FermionField, Box<dyn Error>> {
    let mut values = Vec::with_capacity(NC * lattice.nv());
    for site in 0..lattice.nv() {
        for color in 0..NC {
            values.push(canonical_z4(codes[color + NC * site]));
        }
    }
    Ok(FermionField::from_vec_col_major(lattice, 1, values)?)
}

fn consume_metadata(metadata: &Value, dir: &Path) -> Result<Vec<Vec<u64>>, Box<dyn Error>> {
    assert_eq!(
        metadata["schema"].as_str(),
        Some("fermion_measurements_phase4.v1")
    );
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 4]));
    assert_eq!(metadata["nc"].as_u64(), Some(3));
    assert_eq!(metadata["wilson"]["components"].as_u64(), Some(4));
    assert_eq!(metadata["wilson"]["kappa"].as_f64(), Some(0.08));
    assert_eq!(
        metadata["wilson"]["boundary"],
        serde_json::json!([1, 1, 1, -1])
    );
    assert_eq!(metadata["staggered"]["components"].as_u64(), Some(1));
    assert_eq!(metadata["staggered"]["mass"].as_f64(), Some(0.17));
    assert_eq!(
        metadata["staggered"]["boundary"],
        serde_json::json!([1, 1, 1, -1])
    );
    assert_eq!(metadata["chiral"]["operator"].as_str(), Some("staggered"));
    assert_eq!(metadata["chiral"]["flavor_factor"].as_f64(), Some(0.5));
    assert_eq!(metadata["chiral"]["source_count"].as_u64(), Some(3));
    assert_eq!(
        metadata["chiral"]["codes_file"].as_str(),
        Some("staggered_chiral_codes.json")
    );
    assert_eq!(
        metadata["chiral"]["code_order"].as_str(),
        Some("source, site, component, color; each source array is Rust [color, component, x, y, z, t] order")
    );
    assert_eq!(
        metadata["chiral"]["mapping"],
        serde_json::json!([
            "word & 3 = 0 -> 1",
            "word & 3 = 1 -> i",
            "word & 3 = 2 -> -1",
            "word & 3 = 3 -> -i"
        ])
    );
    assert_eq!(
        metadata["solver_parameters"]["absolute_squared_tolerance"].as_f64(),
        Some(TOLERANCE)
    );
    assert_eq!(
        metadata["solver_parameters"]["max_iterations"].as_u64(),
        Some(MAX_ITERATIONS as u64)
    );
    assert_eq!(
        metadata["solver_parameters"]["wilson_tolerance_key"].as_str(),
        Some("eps_CG")
    );
    assert_eq!(
        metadata["solver_parameters"]["staggered_tolerance_key"].as_str(),
        Some("eps")
    );
    assert_eq!(
        metadata["solver_parameters"]["julia_solver_keywords"],
        serde_json::json!(["eps", "maxsteps", "verbose"])
    );
    assert_eq!(
        metadata["solver_parameters"]["rust_solver"].as_str(),
        Some("bicgstab")
    );
    for (field, package, version, commit) in [
        (
            "gaugefields_jl",
            "Gaugefields.jl",
            "0.7.2",
            "9e5719970770f4497405a856315c90bef7f74449",
        ),
        (
            "latticediracoperators_jl",
            "LatticeDiracOperators.jl",
            "0.6.4",
            "bdef628184597815ba3e0cddf2536df767e78a02",
        ),
        (
            "qcdmeasurements_jl",
            "QCDMeasurements.jl",
            "0.2.13",
            "9e04c37bbd68712cf7a749ae5aff10eb6aae4566",
        ),
    ] {
        assert_eq!(metadata[field]["package"].as_str(), Some(package));
        assert_eq!(metadata[field]["version"].as_str(), Some(version));
        assert_eq!(metadata[field]["commit"].as_str(), Some(commit));
        assert_eq!(metadata[field]["clean"].as_bool(), Some(true));
    }
    assert_eq!(
        metadata["source_urls"],
        serde_json::json!([
            "https://github.com/akio-tomiya/QCDMeasurements.jl/blob/9e04c37bbd68712cf7a749ae5aff10eb6aae4566/src/measurements/measure_Pion_correlator.jl",
            "https://github.com/akio-tomiya/QCDMeasurements.jl/blob/9e04c37bbd68712cf7a749ae5aff10eb6aae4566/src/measurements/measure_chiral_condensate.jl",
            "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/AbstractGaugefields.jl",
            "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/nowing/gaugefields_4D_nowing.jl",
            "https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/Diracoperators.jl",
            "https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/WilsonFermion/WilsonFermion_4D_nowing.jl",
            "https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/StaggeredFermion/StaggeredFermion_4D_nowing.jl",
            "https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/cgmethods.jl"
        ])
    );
    assert_eq!(
        metadata["source_functions"],
        serde_json::json!([
            "Initialize_Gaugefields",
            "Wilson_Dirac_operator",
            "Staggered_Dirac_operator",
            "solve_DinvX!",
            "bicgstab",
            "QCDMeasurements pion source ordering and contraction"
        ])
    );
    assert_eq!(metadata["entrypoint_map"].as_array().unwrap().len(), 4);
    for entrypoint in metadata["entrypoint_map"].as_array().unwrap() {
        assert!(entrypoint["julia"].as_str().is_some());
        assert!(entrypoint["julia_source"].as_str().is_some());
        assert!(entrypoint["rust"].as_str().is_some());
    }
    assert_eq!(
        metadata["layout"]["permutation"],
        serde_json::json!([1, 6, 2, 3, 4, 5])
    );
    assert_eq!(
        metadata["layout"]["source_site"],
        serde_json::json!([0, 0, 0, 0])
    );
    assert_eq!(
        metadata["layout"]["julia_shape"].as_str(),
        Some("[3,NX,NY,NZ,NT,components]")
    );
    assert_eq!(
        metadata["layout"]["rust_shape"].as_str(),
        Some("[3,components,NX,NY,NZ,NT]")
    );
    assert_eq!(
        metadata["layout"]["conversion"].as_str(),
        Some("permutedims(array, (1, 6, 2, 3, 4, 5))")
    );
    assert_eq!(metadata["layout"]["site_order"].as_str(), Some("x fastest"));
    assert_eq!(
        metadata["contraction"]["high_level_pion_reconstruction_called"].as_bool(),
        Some(false)
    );
    assert_eq!(
        metadata["contraction"]["corrected"].as_str(),
        Some("sum_xyz sum_alpha,beta abs2(G_beta,alpha)")
    );
    assert_eq!(
        metadata["contraction"]["legacy"].as_str(),
        Some("source-diagonal value duplicated across every sink color/component")
    );
    assert_eq!(
        metadata["contraction"]["staggered_sign"].as_str(),
        Some("none")
    );
    assert_eq!(
        metadata["contraction"]["normalization"].as_str(),
        Some("none for pion; flavor_factor * mean(Re(rdagger p)) / NV for chiral")
    );
    let julia_diagnostics = &metadata["solver_diagnostics"]["julia"];
    assert_eq!(julia_diagnostics["method"].as_str(), Some("bicgstab"));
    assert_eq!(
        julia_diagnostics["reports_are_provenance_only"].as_bool(),
        Some(true)
    );
    assert_eq!(
        julia_diagnostics["iteration_counts_compared"].as_bool(),
        Some(false)
    );
    assert_eq!(
        julia_diagnostics["wilson_reports"]
            .as_array()
            .unwrap()
            .len(),
        12
    );
    assert_eq!(
        julia_diagnostics["staggered_reports"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        julia_diagnostics["chiral_reports"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    for group in ["wilson_reports", "staggered_reports", "chiral_reports"] {
        for report in julia_diagnostics[group].as_array().unwrap() {
            consume_julia_diagnostic(report);
        }
    }
    let rust_diagnostics = &metadata["solver_diagnostics"]["rust"];
    assert_eq!(
        rust_diagnostics["method"].as_str(),
        Some("dirac_operators::bicgstab")
    );
    assert_eq!(
        rust_diagnostics["reports"].as_str(),
        Some("PionCorrelator::solver_reports and ChiralCondensate::solver_reports")
    );
    assert_eq!(
        rust_diagnostics["iteration_counts_compared"].as_bool(),
        Some(false)
    );
    assert_eq!(
        rust_diagnostics["true_residual_gate"].as_str(),
        Some("fresh relative residual <= 1e-11")
    );
    assert_eq!(
        rust_diagnostics["fields"],
        serde_json::json!([
            "method",
            "iterations",
            "recursive_residual_squared",
            "initial_residual_squared",
            "true_residual_squared",
            "tolerance",
            "maximum_iterations",
            "restart_count",
            "convergence_branch"
        ])
    );
    let comparison = &metadata["comparison"];
    assert_eq!(
        comparison["payload_max_abs_tolerance"].as_f64(),
        Some(2.0e-12)
    );
    assert_eq!(
        comparison["payload_relative_tolerance"].as_f64(),
        Some(2.0e-10)
    );
    assert_eq!(
        comparison["true_relative_residual_tolerance"].as_f64(),
        Some(1.0e-11)
    );
    assert_eq!(
        comparison["criterion"].as_str(),
        Some("solutions and corrected measurements compare; solver iteration counts are provenance only")
    );
    assert_eq!(
        metadata["generator"]["script"].as_str(),
        Some("fixtures/generate.jl")
    );
    assert_eq!(
        metadata["generator"]["mode"].as_str(),
        Some("fermion_measurements_phase4")
    );
    assert_eq!(
        metadata["generator"]["julia_version"].as_str(),
        Some("1.12.5")
    );
    assert_eq!(
        metadata["chiral"]["rng_state"],
        serde_json::json!([11, 22, 33, 44])
    );
    assert_eq!(
        metadata["generator"]["randomness"].as_str(),
        Some("fixed Xoshiro state; generated canonical Z4 codes are stored explicitly")
    );

    let files = metadata["files"].as_array().unwrap();
    assert_eq!(files.len(), 89);
    let mut declared = files
        .iter()
        .map(|file| {
            file.as_str()
                .map(str::to_owned)
                .ok_or("fixture filename is not a string")
        })
        .collect::<Result<Vec<_>, _>>()?;
    declared.sort();
    let mut actual = fs::read_dir(dir)?
        .map(|entry| -> Result<String, Box<dyn Error>> {
            let name = entry?.file_name();
            Ok(name
                .into_string()
                .map_err(|_| "fixture filename is not UTF-8")?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    actual.retain(|name| name != "metadata.json");
    actual.sort();
    assert_eq!(
        actual, declared,
        "fixture payload set differs from metadata"
    );
    for name in declared {
        assert!(dir.join(&name).is_file(), "missing fixture payload {name}");
        let _ = fs::read(dir.join(name))?;
    }
    let codes: Vec<Vec<u64>> = serde_json::from_slice(&fs::read(
        dir.join(metadata["chiral"]["codes_file"].as_str().unwrap()),
    )?)?;
    assert_eq!(codes.len(), 3);
    assert!(codes.iter().all(|codes| codes.len() == NC * 2 * 2 * 2 * 4));
    assert!(codes.iter().flatten().all(|code| *code < 4));
    Ok(codes)
}

#[test]
fn generated_fixture_consumes_all_payloads_and_matches_measurements() -> Result<(), Box<dyn Error>>
{
    let dir = fixture_dir();
    let metadata: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)?;
    let codes = consume_metadata(&metadata, &dir)?;
    let links = load_links(&dir)?;
    run_point_kind(&dir, &links, "wilson", &metadata)?;
    run_point_kind(&dir, &links, "staggered", &metadata)?;

    let lattice = links.lattice();
    let boundary = FermionBoundary::new([1, 1, 1, -1])?;
    let operator = StaggeredDirac::with_boundary(
        &links,
        metadata["staggered"]["mass"].as_f64().unwrap(),
        boundary,
    )?;
    let solver = SolverParams::new(TOLERANCE, MAX_ITERATIONS)?;
    let mut source_values = Vec::with_capacity(codes.len());
    let mut solutions = Vec::with_capacity(codes.len());
    for (index, codes) in codes.iter().enumerate() {
        let (expected_source, _) = load_field_payload(
            &dir,
            &format!("staggered_chiral_source_{index}"),
            lattice,
            1,
        )?;
        let (_, expected_values) = load_field_payload(
            &dir,
            &format!("staggered_chiral_solution_{index}"),
            lattice,
            1,
        )?;
        let source = explicit_noise_field(lattice, codes)?;
        assert_complex_close(
            &format!("chiral source {index}"),
            &field_values(&source)?,
            &field_values(&expected_source)?,
            0.0,
        );
        let mut solution = FermionField::zeros(lattice, 1)?;
        let report = bicgstab(&mut solution, &operator, &source, solver)?;
        consume_report(&report);
        assert_complex_close(
            &format!("chiral solution {index}"),
            &field_values(&solution)?,
            &expected_values,
            2.0e-12,
        );
        assert!(relative_residual(&operator, &solution, &source)? <= 1.0e-11);
        source_values.push(source.inner_product(&solution)?.re);
        solutions.push(solution);
    }
    let (expected_source_values, expected_source_shape) =
        load_scalar_payload(&dir, "staggered_chiral_source_values")?;
    assert_eq!(expected_source_shape, vec![3]);
    assert_real_close(
        "chiral source values",
        &source_values,
        &expected_source_values,
        2.0e-12,
    );
    let value =
        0.5 * source_values.iter().sum::<f64>() / source_values.len() as f64 / lattice.nv() as f64;
    let (expected_value, expected_value_shape) =
        load_scalar_payload(&dir, "staggered_chiral_value")?;
    assert_eq!(expected_value_shape, vec![1]);
    assert_real_close("chiral value", &[value], &expected_value, 2.0e-12);

    let state = metadata["chiral"]["rng_state"]
        .as_array()
        .unwrap()
        .iter()
        .map(|word| word.as_u64().ok_or("invalid chiral RNG state"))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "chiral RNG state must have four words")?;
    let mut rng = ReproducibleRng::from_state(state)?;
    let measured = stochastic_chiral_condensate(
        &operator,
        metadata["chiral"]["flavor_factor"].as_f64().unwrap(),
        metadata["chiral"]["source_count"].as_u64().unwrap() as usize,
        solver,
        &mut rng,
    )?;
    assert_real_close(
        "public chiral source values",
        &measured.source_values,
        &expected_source_values,
        2.0e-12,
    );
    assert_real_close(
        "public chiral value",
        &[measured.value],
        &expected_value,
        2.0e-12,
    );
    assert_eq!(measured.solver_reports.len(), codes.len());
    for report in &measured.solver_reports {
        consume_report(report);
    }

    assert!(solutions
        .iter()
        .all(|solution| solution.norm_squared().unwrap().is_finite()));
    let reports = metadata["solver_diagnostics"]["julia"]["chiral_reports"]
        .as_array()
        .unwrap();
    assert_eq!(reports.len(), codes.len());
    for report in reports {
        consume_julia_diagnostic(report);
    }
    Ok(())
}

#[test]
fn fixture_has_nontrivial_corrected_and_legacy_contractions() -> Result<(), Box<dyn Error>> {
    let dir = fixture_dir();
    for kind in ["wilson", "staggered"] {
        let (corrected, shape) = load_scalar_payload(&dir, &format!("pion_{kind}_corrected"))?;
        let (legacy, legacy_shape) = load_scalar_payload(&dir, &format!("pion_{kind}_legacy"))?;
        assert_eq!(shape, vec![4]);
        assert_eq!(legacy_shape, shape);
        assert!(max_abs_real(&corrected, &legacy) > 1.0e-8);
        assert!(corrected.iter().all(|value| value.is_finite()));
        assert!(legacy.iter().all(|value| value.is_finite()));
    }
    Ok(())
}

#[test]
fn fixture_codes_are_explicit_and_distinct() -> Result<(), Box<dyn Error>> {
    let dir = fixture_dir();
    let codes: Vec<Vec<u64>> =
        serde_json::from_slice(&fs::read(dir.join("staggered_chiral_codes.json"))?)?;
    assert_eq!(codes.len(), 3);
    for left in 0..codes.len() {
        for right in (left + 1)..codes.len() {
            assert_ne!(codes[left], codes[right]);
            assert!(!(0..4).any(|phase| {
                codes[left]
                    .iter()
                    .zip(&codes[right])
                    .all(|(a, b)| (a + phase) % 4 == *b)
            }));
        }
    }
    Ok(())
}
