use dirac_operators::{FermionBoundary, FermionField, FermionOperator, WilsonDirac};
use gaugefields::{GaugeLinkTensor, GaugeLinks, LatticeShape4};
use num_complex::Complex64;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tenferro_tensor::TypedTensor;

type C = Complex64;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/fermions_task_a")
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
        for spin in 0..4 {
            for color in 0..3 {
                let julia_offset = color + 3 * (site + nv * spin);
                let rust_offset = color + 3 * (spin + 4 * site);
                result[rust_offset] = values[julia_offset];
            }
        }
    }
    Ok(result)
}

fn field_values(field: &FermionField) -> Result<Vec<C>, Box<dyn Error>> {
    let mut result = Vec::with_capacity(field.len());
    for site in 0..field.lattice().nv() {
        for component in 0..field.components() {
            for color in 0..3 {
                result.push(field.component(color, component, site)?);
            }
        }
    }
    Ok(result)
}

fn max_abs(left: &[C], right: &[C]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

fn assert_fixture_close(label: &str, left: &[C], right: &[C], tolerance: f64) {
    let residual = max_abs(left, right);
    eprintln!("{label}: max_abs_residual={residual:.17e}");
    assert!(residual <= tolerance, "{label} residual {residual:.3e}");
}

fn load_links(dir: &Path) -> Result<GaugeLinks, Box<dyn Error>> {
    let mut links = Vec::with_capacity(4);
    for direction in 0..4 {
        let (values, shape) = read_npy_complex(&dir.join(format!("u{direction}.npy")))?;
        if shape != [3, 3, 2, 2, 2, 2] {
            return Err(format!("unexpected link shape {shape:?}").into());
        }
        let current = LatticeShape4::new([shape[2], shape[3], shape[4], shape[5]])?;
        let tensor = TypedTensor::from_vec_col_major(shape, values)?;
        links.push(GaugeLinkTensor::from_typed(tensor, current)?);
    }
    Ok(GaugeLinks::new(
        links.try_into().map_err(|_| "four links")?,
    )?)
}

#[test]
fn generated_task_a_fixture_matches_rust_and_records_layout_conversion(
) -> Result<(), Box<dyn Error>> {
    let dir = fixture_dir();
    let metadata: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)?;
    assert_eq!(metadata["schema"], "fermions_task_a.v1");
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["nc"], 3);
    assert_eq!(metadata["components"], 4);
    assert_eq!(metadata["kappa"], 0.13);
    assert_eq!(metadata["r"], 1.0);
    assert_eq!(
        metadata["boundaries"],
        serde_json::json!({
            "periodic": [1, 1, 1, 1],
            "antiperiodic": [1, 1, 1, -1]
        })
    );
    assert_eq!(
        metadata["gaugefields_jl"]["commit"],
        "9e5719970770f4497405a856315c90bef7f74449"
    );
    assert_eq!(
        metadata["latticediracoperators_jl"]["commit"],
        "bdef628184597815ba3e0cddf2536df767e78a02"
    );
    assert_eq!(
        metadata["layout"]["permutation"],
        serde_json::json!([1, 6, 2, 3, 4, 5])
    );
    assert_eq!(metadata["entrypoint_map"].as_array().map(Vec::len), Some(9));
    let tolerance = metadata["comparison"]["component_max_abs_tolerance"]
        .as_f64()
        .ok_or("fixture tolerance must be numeric")?;
    assert_eq!(tolerance, 2e-12);
    let fixture_files = metadata["files"]
        .as_array()
        .ok_or("fixture files must be an array")?;
    assert_eq!(fixture_files.len(), 18);
    for file in fixture_files {
        let name = file.as_str().ok_or("fixture filename must be a string")?;
        assert!(dir.join(name).is_file(), "missing fixture file {name}");
    }

    let (input_julia, input_julia_shape) = read_npy_complex(&dir.join("input_julia.npy"))?;
    let (input_rust, input_rust_shape) = read_npy_complex(&dir.join("input_rust.npy"))?;
    assert_eq!(input_rust_shape, vec![3, 4, 2, 2, 2, 2]);
    assert_fixture_close(
        "input transpose",
        &transpose_julia_field(&input_julia, &input_julia_shape)?,
        &input_rust,
        tolerance,
    );

    let links = load_links(&dir)?;
    let lattice = links.lattice();
    let input = FermionField::from_vec_col_major(lattice, 4, input_rust.clone())?;
    let kappa = 0.13;
    for (name, signs) in [("periodic", [1, 1, 1, 1]), ("antiperiodic", [1, 1, 1, -1])] {
        let operator = WilsonDirac::with_boundary(&links, kappa, FermionBoundary::new(signs)?)?;
        let mut d = FermionField::zeros(lattice, 4)?;
        let mut ddag = FermionField::zeros(lattice, 4)?;
        let mut normal = FermionField::zeros(lattice, 4)?;
        operator.apply_into(&mut d, &input)?;
        operator.adjoint().apply_into(&mut ddag, &input)?;
        operator.normal().apply_into(&mut normal, &input)?;
        for (label, value) in [("d", d), ("ddag", ddag), ("ddagd", normal)] {
            let (expected, expected_shape) =
                read_npy_complex(&dir.join(format!("{label}_{name}_rust.npy")))?;
            assert_eq!(expected_shape, vec![3, 4, 2, 2, 2, 2]);
            assert_fixture_close(
                &format!("{label} {name}"),
                &field_values(&value)?,
                &expected,
                tolerance,
            );
            let (julia, julia_shape) =
                read_npy_complex(&dir.join(format!("{label}_{name}_julia.npy")))?;
            assert_fixture_close(
                &format!("{label} {name} transpose"),
                &transpose_julia_field(&julia, &julia_shape)?,
                &expected,
                tolerance,
            );
        }
    }
    Ok(())
}
