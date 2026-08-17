use gaugefields::{exp_ta, GaugeError, Mat3};
use npyz::Order;
use num_complex::Complex64;
use serde_json::Value;
use std::{fs, path::Path};

const JULIA_COMMIT: &str = "9e5719970770f4497405a856315c90bef7f74449";

#[test]
fn julia_exp_ta_fixture_has_branch_provenance() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/exp_ta");
    let metadata: Value =
        serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["gaugefields_jl_commit"], JULIA_COMMIT);
    assert_eq!(metadata["source_function"], "exptU!");
    let cases = metadata["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    assert!(cases.iter().any(|case| case["name"] == "zero"));
    assert!(cases.iter().any(|case| case["branch"] == "analytic"));
    assert!(cases.iter().any(|case| case["branch"] == "fallback"));
    assert!(cases.iter().any(|case| case["name"] == "near_below"));
    assert!(cases.iter().any(|case| case["name"] == "near_above"));
    assert!(cases.iter().any(|case| case["name"] == "balanced_pair"));
    assert!(metadata["balanced_oracle"]
        .as_str()
        .unwrap()
        .contains("csum cancellation"));
    for case in cases {
        assert_eq!(case["coefficients"].as_array().unwrap().len(), 8);
        assert!(case["t"].as_f64().unwrap().is_finite());
    }

    let bytes = fs::read(dir.join("expected.npy")).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(npy.order(), Order::Fortran);
    assert_eq!(npy.shape(), &[3, 3, cases.len() as u64]);
    assert_eq!(npy.into_vec::<Complex64>().unwrap().len(), 9 * cases.len());
}

#[test]
fn exp_ta_matches_julia_branches_and_su3_invariants() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/exp_ta");
    let metadata: Value =
        serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap()).unwrap();
    let bytes = fs::read(dir.join("expected.npy")).unwrap();
    let expected = npyz::NpyFile::new(&bytes[..])
        .unwrap()
        .into_vec::<Complex64>()
        .unwrap();
    for (index, case) in metadata["cases"].as_array().unwrap().iter().enumerate() {
        let coefficients: [f64; 8] = case["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let actual = exp_ta(case["t"].as_f64().unwrap(), &coefficients).unwrap();
        let oracle = Mat3::from_array(expected[9 * index..9 * (index + 1)].try_into().unwrap());
        let max_residual = actual
            .as_array()
            .iter()
            .zip(oracle.as_array())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        let unitary = actual.adjoint().mul(actual);
        let oracle_unitary = oracle.adjoint().mul(oracle);
        let unitary_residual = unitary
            .as_array()
            .iter()
            .zip(Mat3::identity().as_array())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        let oracle_unitary_residual = oracle_unitary
            .as_array()
            .iter()
            .zip(Mat3::identity().as_array())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        let a = actual;
        let det = a[(0, 0)] * (a[(1, 1)] * a[(2, 2)] - a[(1, 2)] * a[(2, 1)])
            - a[(0, 1)] * (a[(1, 0)] * a[(2, 2)] - a[(1, 2)] * a[(2, 0)])
            + a[(0, 2)] * (a[(1, 0)] * a[(2, 1)] - a[(1, 1)] * a[(2, 0)]);
        let o = oracle;
        let oracle_det = o[(0, 0)] * (o[(1, 1)] * o[(2, 2)] - o[(1, 2)] * o[(2, 1)])
            - o[(0, 1)] * (o[(1, 0)] * o[(2, 2)] - o[(1, 2)] * o[(2, 0)])
            + o[(0, 2)] * (o[(1, 0)] * o[(2, 1)] - o[(1, 1)] * o[(2, 0)]);
        assert!(
            max_residual < 2e-12,
            "case={} branch={} residual={max_residual}",
            case["name"],
            case["branch"]
        );
        let invariant_tolerance = if case["name"] == "near_above" {
            oracle_unitary_residual + 1e-12
        } else if case["branch"] == "fallback" {
            1e-2
        } else {
            2e-12
        };
        assert!(
            unitary_residual < invariant_tolerance,
            "case={} branch={} unitary_residual={unitary_residual} oracle_unitary_residual={oracle_unitary_residual} element_residual={max_residual}",
            case["name"],
            case["branch"]
        );
        let determinant_tolerance = if case["name"] == "near_above" {
            (oracle_det - Complex64::new(1.0, 0.0)).norm() + 1e-12
        } else {
            invariant_tolerance
        };
        assert!(
            (det - Complex64::new(1.0, 0.0)).norm() < determinant_tolerance,
            "case={} branch={} det={det}",
            case["name"],
            case["branch"]
        );
    }

    assert!(matches!(
        exp_ta(f64::NAN, &[0.0; 8]),
        Err(GaugeError::NonFiniteSu3Input { .. })
    ));
    let mut nonfinite = [0.0; 8];
    nonfinite[3] = f64::INFINITY;
    assert!(matches!(
        exp_ta(1.0, &nonfinite),
        Err(GaugeError::NonFiniteSu3Input { .. })
    ));
}

#[test]
fn exp_ta_identity_small_t_derivative_and_ta_roundtrip() {
    let coefficients = [0.31, -0.27, 0.19, 0.41, -0.13, 0.23, -0.37, 0.29];
    assert_eq!(exp_ta(0.0, &coefficients).unwrap(), Mat3::identity());
    let generator = Mat3::from_gell_mann_coefficients(coefficients);
    let roundtrip = generator.gell_mann_coefficients();
    for (actual, expected) in roundtrip.iter().zip(coefficients) {
        assert!((actual - expected).abs() < 1e-15);
    }
    let h = 1e-6;
    let plus = exp_ta(h, &coefficients).unwrap();
    let minus = exp_ta(-h, &coefficients).unwrap();
    let max_residual = plus
        .as_array()
        .iter()
        .zip(minus.as_array())
        .zip(generator.as_array())
        .map(|((plus, minus), expected)| ((*plus - *minus) / (2.0 * h) - expected).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_residual < 1e-10,
        "small-t derivative residual={max_residual}"
    );
}

#[test]
fn cancelling_coefficients_are_nonidentity_and_have_the_correct_derivative() {
    let coefficients = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let actual = exp_ta(1.0, &coefficients).unwrap();
    let distance = actual
        .as_array()
        .iter()
        .zip(Mat3::identity().as_array())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    assert!(distance > 0.1, "balanced generator returned identity");

    let h = 1e-6;
    let plus = exp_ta(h, &coefficients).unwrap();
    let minus = exp_ta(-h, &coefficients).unwrap();
    let generator = Mat3::from_gell_mann_coefficients(coefficients);
    let residual = plus
        .as_array()
        .iter()
        .zip(minus.as_array())
        .zip(generator.as_array())
        .map(|((plus, minus), expected)| ((*plus - *minus) / (2.0 * h) - expected).norm())
        .fold(0.0_f64, f64::max);
    assert!(residual < 1e-10, "balanced derivative residual={residual}");
}

#[test]
fn finite_inputs_that_overflow_numerical_range_are_rejected() {
    let mut scaling_overflow = [0.0; 8];
    scaling_overflow[0] = f64::MAX;
    let result = exp_ta(f64::MAX, &scaling_overflow);
    assert!(matches!(result, Err(GaugeError::Su3NumericalRange { .. })));

    let mut cardano_overflow = [0.0; 8];
    cardano_overflow[0] = f64::MAX.sqrt() * 1.5;
    let result = exp_ta(1.0, &cardano_overflow);
    assert!(matches!(result, Err(GaugeError::Su3NumericalRange { .. })));
}
