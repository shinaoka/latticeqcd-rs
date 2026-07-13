use gaugefields::{normalize_su3, GaugeError, Mat3};
use num_complex::Complex64;

fn determinant(a: Mat3) -> Complex64 {
    a[(0, 0)] * (a[(1, 1)] * a[(2, 2)] - a[(1, 2)] * a[(2, 1)])
        - a[(0, 1)] * (a[(1, 0)] * a[(2, 2)] - a[(1, 2)] * a[(2, 0)])
        + a[(0, 2)] * (a[(1, 0)] * a[(2, 1)] - a[(1, 1)] * a[(2, 0)])
}

#[test]
fn normalization_projects_drift_transactionally() {
    let mut matrix = Mat3::from_array([
        Complex64::new(1.02, 0.01),
        Complex64::new(0.03, -0.02),
        Complex64::new(-0.01, 0.04),
        Complex64::new(0.02, 0.01),
        Complex64::new(0.97, -0.03),
        Complex64::new(0.05, 0.01),
        Complex64::new(-0.04, 0.02),
        Complex64::new(0.01, -0.02),
        Complex64::new(1.01, 0.03),
    ]);
    normalize_su3(&mut matrix).unwrap();
    let residual = matrix.mul(matrix.adjoint());
    let max_residual = residual
        .as_array()
        .iter()
        .zip(Mat3::identity().as_array())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        max_residual < 2e-15,
        "orthogonality residual={max_residual}"
    );
    assert!((determinant(matrix) - Complex64::new(1.0, 0.0)).norm() < 2e-15);

    for invalid in [
        Mat3::zero(),
        Mat3::from_array([Complex64::new(f64::NAN, 0.0); 9]),
        Mat3::from_array([Complex64::new(f64::INFINITY, 0.0); 9]),
    ] {
        let mut value = invalid;
        let original = value;
        let result = normalize_su3(&mut value);
        assert!(matches!(
            result,
            Err(GaugeError::NonFiniteSu3Input { .. })
                | Err(GaugeError::SingularSu3Normalization { .. })
        ));
        for (actual, expected) in value.as_array().iter().zip(original.as_array()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }
}
