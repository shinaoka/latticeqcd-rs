use gaugefields::{normalize_su3, GaugeError, Mat3};
use npyz::Order;
use num_complex::Complex64;
use serde_json::Value;
use std::{fs, path::Path};

const JULIA_COMMIT: &str = "9e5719970770f4497405a856315c90bef7f74449";

fn assert_bitwise_eq(actual: Mat3, expected: Mat3) {
    for (actual, expected) in actual.as_array().iter().zip(expected.as_array()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
}

#[test]
fn normalization_matches_pinned_julia_oracles() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/normalize_su3");
    let metadata: Value =
        serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["gaugefields_jl_commit"], JULIA_COMMIT);
    assert_eq!(metadata["source_function"], "normalize_U!");
    assert_eq!(
        metadata["source_file"],
        "src/4D/nowing/gaugefields_4D_nowing.jl"
    );
    assert_eq!(metadata["lattice"], serde_json::json!([1, 1, 1, 1]));
    let cases = metadata["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);

    let read = |name| {
        let bytes = fs::read(dir.join(name)).unwrap();
        let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
        assert_eq!(npy.order(), Order::Fortran);
        assert_eq!(npy.shape(), &[3, 3, cases.len() as u64]);
        npy.into_vec::<Complex64>().unwrap()
    };
    let inputs = read("input.npy");
    let expected = read("expected.npy");
    for (index, name) in cases.iter().enumerate() {
        let mut actual = Mat3::from_array(inputs[9 * index..9 * (index + 1)].try_into().unwrap());
        normalize_su3(&mut actual).unwrap();
        let oracle = &expected[9 * index..9 * (index + 1)];
        let residual = actual
            .as_array()
            .iter()
            .zip(oracle)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        assert!(residual < 3e-15, "case={name} residual={residual}");
    }
}

#[test]
fn every_normalization_failure_is_typed_and_transactional() {
    let dependent = Mat3::from_array([
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(6.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);
    let zero_row0 = Mat3::from_array([
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let cases = [(dependent, 1usize), (zero_row0, 0usize)];
    for (input, row) in cases {
        let mut actual = input;
        assert!(matches!(
            normalize_su3(&mut actual),
            Err(GaugeError::SingularSu3Normalization { row: actual_row }) if actual_row == row
        ));
        assert_bitwise_eq(actual, input);
    }

    for input in [
        Mat3::from_array([Complex64::new(f64::NAN, 0.0); 9]),
        Mat3::from_array([Complex64::new(f64::INFINITY, 0.0); 9]),
    ] {
        let mut actual = input;
        assert!(matches!(
            normalize_su3(&mut actual),
            Err(GaugeError::NonFiniteSu3Input { .. })
        ));
        assert_bitwise_eq(actual, input);
    }
}
