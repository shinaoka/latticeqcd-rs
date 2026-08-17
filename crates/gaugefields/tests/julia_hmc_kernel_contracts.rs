use gaugefields::{action_gradient, dsdu, gauge_force, load_fixture, wilson_action, Fixture};
use num_complex::Complex64;
use std::{fs, path::Path};

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2")
}

fn fixture() -> Fixture {
    load_fixture(fixture_dir()).unwrap()
}

fn assert_scaled_f64(actual: &[f64], base: &[f64], scale: f64, label: &str) {
    assert_eq!(actual.len(), base.len());
    let mut max_residual = 0.0_f64;
    for (&a, &b) in actual.iter().zip(base) {
        assert!(a.is_finite() && b.is_finite(), "{label}: nonfinite input");
        max_residual = max_residual.max((a - scale * b).abs());
    }
    assert!(max_residual < 1e-13, "{label}: max residual={max_residual}");
}

fn assert_scaled_c64(actual: &[Complex64], base: &[Complex64], scale: f64, label: &str) {
    assert_eq!(actual.len(), base.len());
    let mut max_residual = 0.0_f64;
    for (&a, &b) in actual.iter().zip(base) {
        assert!(
            a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite(),
            "{label}: nonfinite input"
        );
        max_residual = max_residual.max((a - scale * b).norm());
    }
    assert!(max_residual < 1e-13, "{label}: max residual={max_residual}");
}

#[test]
fn julia_gauge_action_coefficient_is_linear_in_beta() {
    let f = fixture();
    let beta = f.metadata().beta;
    let base = wilson_action(f.links(), beta).unwrap();
    for scale in [-1.75, 0.0, 2.5] {
        let actual = wilson_action(f.links(), scale * beta).unwrap();
        let residual = (actual - scale * base).abs();
        assert!(
            actual.is_finite() && residual < 1e-12,
            "scale={scale}: action residual={residual}"
        );
    }
}

#[test]
fn julia_derivative_payloads_are_linear_in_beta() {
    let f = fixture();
    let beta = f.metadata().beta;
    let base_dsdu = dsdu(f.links(), beta).unwrap();
    let base_gradient = action_gradient(f.links(), beta).unwrap();
    let base_force = gauge_force(f.links(), beta).unwrap();

    for scale in [-1.75, 0.0, 2.5] {
        let actual_dsdu = dsdu(f.links(), scale * beta).unwrap();
        let actual_gradient = action_gradient(f.links(), scale * beta).unwrap();
        let actual_force = gauge_force(f.links(), scale * beta).unwrap();
        for mu in 0..4 {
            assert_scaled_c64(
                actual_dsdu[mu].typed().host_data().unwrap(),
                base_dsdu[mu].typed().host_data().unwrap(),
                scale,
                &format!("dsdu scale={scale} mu={mu}"),
            );
            assert_scaled_c64(
                actual_gradient[mu].typed().host_data().unwrap(),
                base_gradient[mu].typed().host_data().unwrap(),
                scale,
                &format!("gradient scale={scale} mu={mu}"),
            );
            assert_scaled_f64(
                actual_force.tensors()[mu].host_data().unwrap(),
                base_force.tensors()[mu].host_data().unwrap(),
                scale,
                &format!("gauge_force scale={scale} mu={mu}"),
            );
            if scale == 0.0 {
                assert!(actual_dsdu[mu]
                    .typed()
                    .host_data()
                    .unwrap()
                    .iter()
                    .all(|value| *value == Complex64::new(0.0, 0.0)));
                assert!(actual_gradient[mu]
                    .typed()
                    .host_data()
                    .unwrap()
                    .iter()
                    .all(|value| *value == Complex64::new(0.0, 0.0)));
                assert!(actual_force.tensors()[mu]
                    .host_data()
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0));
            }
        }
    }
}

#[test]
fn julia_momentum_update_coefficient_matches_p_update() {
    let f = fixture();
    let epsilon = 0.5;
    let dt = 0.125;
    let expected_coefficient = -epsilon * dt / f.links().nc() as f64;
    let force = gauge_force(f.links(), f.metadata().beta).unwrap();
    let mut saw_nonzero_input = false;
    let mut saw_nonzero_update = false;
    for (mu, tensor) in force.tensors().iter().enumerate() {
        let components = tensor.host_data().unwrap();
        let bytes = fs::read(fixture_dir().join(format!("momentum_delta{mu}.npy"))).unwrap();
        let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
        assert_eq!(npy.order(), npyz::Order::Fortran);
        assert_eq!(npy.shape(), vec![8, 2, 2, 2, 2]);
        let julia_delta = npy.into_vec::<f64>().unwrap();
        assert_eq!(components.len(), julia_delta.len());
        let updated: Vec<_> = components
            .iter()
            .map(|&value| expected_coefficient * value)
            .collect();
        saw_nonzero_input |= components.iter().any(|value| *value != 0.0);
        saw_nonzero_update |= updated.iter().any(|value| *value != 0.0);
        assert_scaled_f64(&updated, &julia_delta, 1.0, &format!("P_update mu={mu}"));
    }
    assert!(saw_nonzero_input && saw_nonzero_update);
}
