use dirac_operators::{
    bicgstab, conjugate_gradient, FermionField, FermionOperator, SolverParams, WilsonDirac,
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/fermions_task_b")
}

fn read_npy_complex(path: &Path) -> Result<(Vec<C>, Vec<usize>), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err(format!("{} is not an NPY file", path.display()).into());
    }
    let major = *bytes.get(6).ok_or("missing NPY major version")?;
    let (header_len, data_start) = match major {
        1 => (
            u16::from_le_bytes(bytes[8..10].try_into()?) as usize,
            10usize,
        ),
        2 | 3 => (
            u32::from_le_bytes(bytes[8..12].try_into()?) as usize,
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
    if !header.contains("'descr': '<c16'") || !header.contains("'fortran_order': True") {
        return Err(format!("unsupported NPY header: {header}").into());
    }
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
    let payload = bytes.get(header_end..).ok_or("NPY payload missing")?;
    if payload.len() != count.checked_mul(16).ok_or("NPY byte count overflow")? {
        return Err("NPY payload length mismatch".into());
    }
    let mut values = Vec::with_capacity(count);
    for chunk in payload.chunks_exact(16) {
        values.push(C::new(
            f64::from_le_bytes(chunk[0..8].try_into()?),
            f64::from_le_bytes(chunk[8..16].try_into()?),
        ));
    }
    Ok((values, shape))
}

fn transpose_julia_field(values: &[C], shape: &[usize]) -> Result<Vec<C>, Box<dyn Error>> {
    if shape != [3, 2, 2, 2, 2, 4] {
        return Err(format!("unexpected Julia field shape {shape:?}").into());
    }
    let nv = 2 * 2 * 2 * 2;
    let mut result = vec![C::default(); values.len()];
    for site in 0..nv {
        for component in 0..4 {
            for color in 0..3 {
                let julia_offset = color + 3 * (site + nv * component);
                let rust_offset = color + 3 * (component + 4 * site);
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
            for color in 0..3 {
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

fn assert_close(label: &str, left: &[C], right: &[C], tolerance: f64) {
    let residual = max_abs(left, right);
    eprintln!("{label}: max_abs_residual={residual:.17e}");
    assert!(residual <= tolerance, "{label} residual {residual:.3e}");
}

fn assert_scalar_close(label: &str, actual: f64, expected: f64) {
    let tolerance = 2.0e-11 * expected.abs().max(1.0).max(actual.abs());
    let residual = (actual - expected).abs();
    eprintln!("{label}: residual={residual:.17e}");
    assert!(residual <= tolerance, "{label} residual {residual:.3e}");
}

fn load_links(dir: &Path) -> Result<GaugeLinks, Box<dyn Error>> {
    let mut links = Vec::with_capacity(4);
    for direction in 0..4 {
        let (values, shape) = read_npy_complex(&dir.join(format!("u{direction}.npy")))?;
        if shape != [3, 3, 2, 2, 2, 2] {
            return Err(format!("unexpected link shape {shape:?}").into());
        }
        let lattice = LatticeShape4::new([2, 2, 2, 2])?;
        let tensor = TypedTensor::from_vec_col_major(shape, values)?;
        links.push(GaugeLinkTensor::from_typed(tensor, lattice)?);
    }
    Ok(GaugeLinks::new(
        links.try_into().map_err(|_| "four links")?,
    )?)
}

fn load_field(dir: &Path, name: &str) -> Result<FermionField, Box<dyn Error>> {
    let (values, shape) = read_npy_complex(&dir.join(name))?;
    if shape != [3, 4, 2, 2, 2, 2] {
        return Err(format!("unexpected Rust field shape {shape:?}").into());
    }
    Ok(FermionField::from_vec_col_major(
        LatticeShape4::new([2, 2, 2, 2])?,
        4,
        values,
    )?)
}

fn load_julia_field(dir: &Path, name: &str) -> Result<Vec<C>, Box<dyn Error>> {
    let (values, shape) = read_npy_complex(&dir.join(name))?;
    transpose_julia_field(&values, &shape)
}

fn true_residual_squared<O: FermionOperator>(
    operator: &O,
    solution: &FermionField,
    rhs: &FermionField,
) -> Result<f64, Box<dyn Error>> {
    let mut applied = FermionField::zeros(solution.lattice(), solution.components())?;
    operator.apply_into(&mut applied, solution)?;
    let solution_values = field_values(&applied)?;
    let rhs_values = field_values(rhs)?;
    Ok(rhs_values
        .iter()
        .zip(solution_values)
        .map(|(rhs_value, applied_value)| (*rhs_value - applied_value).norm_sqr())
        .sum())
}

#[test]
fn generated_task_b_fixture_covers_metadata_layout_solutions_and_reports(
) -> Result<(), Box<dyn Error>> {
    let dir = fixture_dir();
    let metadata: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)?;
    assert_eq!(metadata["schema"], "fermions_task_b.v1");
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["nc"], 3);
    assert_eq!(metadata["components"], 4);
    assert_eq!(metadata["kappa"], 0.13);
    assert_eq!(metadata["r"], 1.0);
    assert_eq!(metadata["boundaries"], serde_json::json!([1, 1, 1, -1]));
    assert_eq!(
        metadata["cases"].as_object().map(|cases| cases.len()),
        Some(4)
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
        metadata["solver_parameters"],
        serde_json::json!({
            "tolerance": 1.0e-20,
            "max_iterations": 2_000,
            "julia_operator_keys": [
                "Dirac_operator", "κ", "r", "faster version", "verbose_level",
                "boundarycondition", "method_CG", "eps_CG", "MaxCGstep"
            ],
            "julia_solver_keywords": ["eps", "maxsteps", "verbose"]
        })
    );
    assert_eq!(
        metadata["source_functions"],
        serde_json::json!([
            "cg",
            "bicgstab",
            "DdagD_operator",
            "LinearAlgebra.mul!",
            "LinearAlgebra.dot"
        ])
    );
    assert_eq!(
        metadata["layout"],
        serde_json::json!({
            "julia_shape": "[3,NX,NY,NZ,NT,4]",
            "rust_shape": "[3,4,NX,NY,NZ,NT]",
            "conversion": "permutedims(array, (1, 6, 2, 3, 4, 5))",
            "permutation": [1, 6, 2, 3, 4, 5],
            "site_order": "x fastest"
        })
    );
    assert_eq!(
        metadata["source_urls"],
        serde_json::json!([
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/cgmethods.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/Diracoperators.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/WilsonFermion/WilsonFermion.jl"
        ])
    );
    assert_eq!(
        metadata["entrypoint_map"],
        serde_json::json!([
            {"julia": "Dirac_operators.cg", "julia_source": "src/cgmethods.jl:768-868", "rust": "conjugate_gradient"},
            {"julia": "Dirac_operators.bicgstab", "julia_source": "src/cgmethods.jl:157-310", "rust": "bicgstab"},
            {"julia": "DdagD_operator", "julia_source": "src/Diracoperators.jl:151-169", "rust": "NormalOperator"},
            {"julia": "LinearAlgebra.mul!", "julia_source": "src/Diracoperators.jl:415-430", "rust": "FermionOperator::apply_into"},
            {"julia": "LinearAlgebra.dot", "julia_source": "src/cgmethods.jl:20-48", "rust": "FermionField::inner_product + checked algebra"}
        ])
    );
    assert_eq!(
        metadata["construction"],
        "explicit diagonal SU(3) links, rhs, and zero/nonzero guesses from fixed formulas; no RNG or global state"
    );
    assert_eq!(
        metadata["comparison"],
        serde_json::json!({
            "solution_max_abs_tolerance": 2e-11,
            "rust_true_relative_residual_tolerance": 1e-11,
            "criterion": "fresh sum(abs2, b-A*x) independent of recursive residual"
        })
    );
    assert_eq!(
        metadata["generator"],
        serde_json::json!({
            "script": "fixtures/generate.jl",
            "mode": "fermions_task_b",
            "randomness": "none"
        })
    );

    let fixture_files = metadata["files"]
        .as_array()
        .ok_or("fixture files must be an array")?;
    assert_eq!(fixture_files.len(), 18);
    let mut declared_files = fixture_files
        .iter()
        .map(|file| {
            file.as_str()
                .ok_or("fixture filename must be a string")
                .map(str::to_owned)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut actual_files = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| "non-UTF-8 fixture filename")?;
            if name != "metadata.json" {
                actual_files.push(name);
            }
        }
    }
    declared_files.sort();
    actual_files.sort();
    assert_eq!(declared_files, actual_files);
    for file in fixture_files {
        let name = file.as_str().ok_or("fixture filename must be a string")?;
        assert!(dir.join(name).is_file(), "missing fixture file {name}");
        let (_, shape) = read_npy_complex(&dir.join(name))?;
        let expected_shape = if name.starts_with('u') {
            vec![3, 3, 2, 2, 2, 2]
        } else if name.ends_with("_julia.npy") {
            vec![3, 2, 2, 2, 2, 4]
        } else {
            vec![3, 4, 2, 2, 2, 2]
        };
        assert_eq!(shape, expected_shape, "unexpected payload shape for {name}");
    }

    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = load_links(&dir)?;
    let dirac = WilsonDirac::with_boundary(
        &links,
        0.13,
        dirac_operators::FermionBoundary::new([1, 1, 1, -1])?,
    )?;
    let normal = dirac.normal();
    let (rhs_rust, rhs_shape) = read_npy_complex(&dir.join("rhs_rust.npy"))?;
    assert_eq!(rhs_shape, vec![3, 4, 2, 2, 2, 2]);
    let (rhs_julia, rhs_julia_shape) = read_npy_complex(&dir.join("rhs_julia.npy"))?;
    assert_close(
        "rhs layout",
        &rhs_rust,
        &transpose_julia_field(&rhs_julia, &rhs_julia_shape)?,
        0.0,
    );
    let rhs = FermionField::from_vec_col_major(lattice, 4, rhs_rust)?;

    for guess_name in ["zero", "nonzero"] {
        let guess_rust = load_field(&dir, &format!("guess_{guess_name}_rust.npy"))?;
        let guess_julia = load_julia_field(&dir, &format!("guess_{guess_name}_julia.npy"))?;
        assert_close(
            &format!("guess {guess_name} layout"),
            &field_values(&guess_rust)?,
            &guess_julia,
            0.0,
        );
    }

    let params = SolverParams::new(1.0e-20, 2_000)?;
    for (case_name, method, operator_name, guess_name) in [
        ("cg_zero", "cg", "DdagD", "zero"),
        ("cg_nonzero", "cg", "DdagD", "nonzero"),
        ("bicgstab_zero", "bicgstab", "D", "zero"),
        ("bicgstab_nonzero", "bicgstab", "D", "nonzero"),
    ] {
        let guess = load_field(&dir, &format!("guess_{guess_name}_rust.npy"))?;
        let before = field_values(&guess)?;
        let mut solution = guess;
        let report = if method == "cg" {
            conjugate_gradient(&mut solution, &normal, &rhs, params)?
        } else {
            bicgstab(&mut solution, &dirac, &rhs, params)?
        };
        assert_ne!(field_values(&solution)?, before);
        let expected = load_field(&dir, &format!("{case_name}_solution_rust.npy"))?;
        assert_close(
            &format!("{case_name} Rust fixture"),
            &field_values(&solution)?,
            &field_values(&expected)?,
            2.0e-11,
        );
        assert_close(
            &format!("{case_name} Julia parity"),
            &field_values(&solution)?,
            &load_julia_field(&dir, &format!("{case_name}_solution_julia.npy"))?,
            2.0e-11,
        );

        let case = &metadata["cases"][case_name];
        assert_eq!(case["method"], method);
        assert_eq!(case["guess"], guess_name);
        assert_eq!(case["operator"], operator_name);
        assert_eq!(
            case["tolerance"],
            metadata["solver_parameters"]["tolerance"]
        );
        assert_eq!(
            case["maximum_iterations"],
            metadata["solver_parameters"]["max_iterations"]
        );
        assert_eq!(report.method.to_string(), method);
        assert_eq!(
            report.tolerance,
            metadata["solver_parameters"]["tolerance"].as_f64().unwrap()
        );
        assert_eq!(
            report.maximum_iterations,
            metadata["solver_parameters"]["max_iterations"]
                .as_u64()
                .unwrap() as usize
        );
        assert_eq!(
            report.iterations,
            case["iterations"].as_u64().unwrap() as usize
        );
        assert_eq!(
            report.restart_count,
            case["restart_count"].as_u64().unwrap() as usize
        );
        assert_eq!(
            report.convergence_branch.to_string(),
            case["convergence_branch"]
        );
        assert_scalar_close(
            &format!("{case_name} initial residual"),
            report.initial_residual_squared,
            case["initial_residual_squared"].as_f64().unwrap(),
        );
        assert_scalar_close(
            &format!("{case_name} recursive residual"),
            report.recursive_residual_squared,
            case["recursive_residual_squared"].as_f64().unwrap(),
        );
        assert_scalar_close(
            &format!("{case_name} true residual"),
            report.true_residual_squared,
            case["true_residual_squared"].as_f64().unwrap(),
        );
        let true_squared = if method == "cg" {
            true_residual_squared(&normal, &solution, &rhs)?
        } else {
            true_residual_squared(&dirac, &solution, &rhs)?
        };
        assert_scalar_close(
            &format!("{case_name} report true residual"),
            report.true_residual_squared,
            true_squared,
        );
        let relative = (true_squared / rhs.norm_squared()?).sqrt();
        eprintln!("{case_name}: true_relative_residual={relative:.17e}");
        assert!(relative <= 1.0e-11);
    }
    Ok(())
}
