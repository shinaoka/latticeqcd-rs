use gaugefields::Mat3;
use num_complex::Complex64 as C;

fn sample(seed: f64) -> Mat3 {
    Mat3::from_array(std::array::from_fn(|i| {
        C::new(seed + i as f64, seed - 0.25 * i as f64)
    }))
}
fn close(a: Mat3, b: Mat3) {
    for (x, y) in a.as_array().iter().zip(b.as_array()) {
        assert!((*x - *y).norm() < 1e-12, "{x:?} {y:?}");
    }
}

#[test]
fn basic_matrix_kernels_match_naive_reference() {
    let a = sample(1.0);
    let b = sample(-0.5);
    let mut naive = [C::default(); 9];
    for j in 0..3 {
        for i in 0..3 {
            for k in 0..3 {
                naive[i + 3 * j] += a[(i, k)] * b[(k, j)];
            }
        }
    }
    close(a.mul(b), Mat3::from_array(naive));
    close(a.mul_adj_right(b), a.mul(b.adjoint()));
    close(a.mul_adj_left(b), a.adjoint().mul(b));
    close(a.mul(b).adjoint(), b.adjoint().mul(a.adjoint()));
    assert!((a.real_trace_mul(b) - a.mul(b).trace().re).abs() < 1e-12);
    close(a.adjoint().adjoint(), a);
    assert_eq!(Mat3::identity().trace(), C::new(3.0, 0.0));
    assert_eq!(Mat3::zero().trace(), C::default());
}

#[test]
fn load_store_and_scaled_additions_are_exact() {
    let a = sample(2.0);
    let mut buf = vec![C::new(99.0, 0.0); 20];
    a.store(&mut buf, 5).unwrap();
    assert_eq!(Mat3::load(&buf, 5).unwrap(), a);
    let mut x = Mat3::zero();
    x.add_scaled_real(2.0, a);
    close(x, a.scaled(C::new(2.0, 0.0)));
    x.add_scaled_complex(C::new(0.0, 1.0), a);
    close(x, a.scaled(C::new(2.0, 1.0)));
}

#[test]
fn ta_projection_is_antihermitian_traceless_and_basis_roundtrips() {
    let ta = sample(0.3).ta();
    close(ta.adjoint(), ta.scaled(C::new(-1.0, 0.0)));
    assert!(ta.trace().norm() < 1e-12);
    let coeff = [0.2, -0.3, 0.5, 0.7, -0.11, 0.13, -0.17, 0.19];
    let h = Mat3::hermitian_from_gell_mann(coeff);
    close(h.adjoint(), h);
    let anti = h.scaled(C::new(0.0, 1.0));
    let got = anti.gell_mann_coefficients();
    for i in 0..8 {
        assert!((got[i] - coeff[i]).abs() < 1e-12);
    }
    let mut accumulated = Mat3::zero();
    accumulated.add_gell_mann_factor(coeff, C::new(0.0, 1.0));
    close(accumulated, anti);
}
