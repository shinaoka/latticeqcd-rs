use gaugefields::Mat3;
use num_complex::Complex64 as C;
fn sample(s: f64) -> Mat3 {
    Mat3(std::array::from_fn(|i| {
        C::new(s + i as f64, 2.0 * s - 0.25 * i as f64)
    }))
}
fn close(a: Mat3, b: Mat3) {
    for i in 0..9 {
        assert!((a.0[i] - b.0[i]).norm() < 1e-12, "{i}");
    }
}
fn naive(a: Mat3, b: Mat3, al: bool, br: bool) -> Mat3 {
    let mut o = [C::default(); 9];
    for j in 0..3 {
        for i in 0..3 {
            for k in 0..3 {
                o[i + 3 * j] += if al { a[(k, i)].conj() } else { a[(i, k)] }
                    * if br { b[(j, k)].conj() } else { b[(k, j)] };
            }
        }
    }
    Mat3(o)
}
fn ta(v: Mat3) -> Mat3 {
    let mut o = [C::default(); 9];
    for j in 0..3 {
        for i in 0..3 {
            o[i + 3 * j] = (v[(i, j)] - v[(j, i)].conj()) * 0.5;
        }
    }
    let t = (o[0] + o[4] + o[8]) / 3.0;
    o[0] -= t;
    o[4] -= t;
    o[8] -= t;
    Mat3(o)
}
fn coeff(v: Mat3) -> [f64; 8] {
    let a = ta(v);
    [
        a[(0, 1)].im + a[(1, 0)].im,
        a[(0, 1)].re - a[(1, 0)].re,
        a[(0, 0)].im - a[(1, 1)].im,
        a[(0, 2)].im + a[(2, 0)].im,
        a[(0, 2)].re - a[(2, 0)].re,
        a[(1, 2)].im + a[(2, 1)].im,
        a[(1, 2)].re - a[(2, 1)].re,
        (a[(0, 0)].im + a[(1, 1)].im - 2.0 * a[(2, 2)].im) / 3f64.sqrt(),
    ]
}
#[test]
fn independent_matrix_helpers() {
    let a = sample(1.0);
    let b = sample(-0.7);
    assert_eq!(
        a.adjoint(),
        Mat3([
            a.0[0].conj(),
            a.0[3].conj(),
            a.0[6].conj(),
            a.0[1].conj(),
            a.0[4].conj(),
            a.0[7].conj(),
            a.0[2].conj(),
            a.0[5].conj(),
            a.0[8].conj()
        ])
    );
    assert_eq!(a.trace(), a.0[0] + a.0[4] + a.0[8]);
    close(a.mul(b), naive(a, b, false, false));
    close(a.mul_adj_left(b), naive(a, b, true, false));
    close(a.mul_adj_right(b), naive(a, b, false, true));
    close(a.mul(b).adjoint(), b.adjoint().mul(a.adjoint()));
    let mut r = C::default();
    for i in 0..3 {
        for k in 0..3 {
            r += a[(i, k)] * b[(k, i)];
        }
    }
    assert!((a.real_trace_mul(b) - r.re).abs() < 1e-12);
    let projected = a.ta();
    close(projected, ta(a));
    assert!(projected.trace().norm() < 1e-12);
    for j in 0..3 {
        for i in 0..3 {
            assert!((projected[(i, j)] + projected[(j, i)].conj()).norm() < 1e-12);
        }
    }
}
#[test]
fn storage_and_scaled_add() {
    let a = sample(2.0);
    let mut buf = vec![C::new(99.0, 0.0); 20];
    a.store(&mut buf, 5).unwrap();
    assert_eq!(&buf[5..14], &a.0);
    assert_eq!(Mat3::load(&buf, 5).unwrap(), a);
    let raw = std::array::from_fn(|i| C::new(i as f64, 10.0 - i as f64));
    let wrapped = Mat3::from_array(raw);
    assert_eq!(wrapped.as_array(), &raw);
    assert_eq!(Mat3::zero().0, [C::default(); 9]);
    assert_eq!(
        Mat3::identity().0,
        [
            C::new(1.0, 0.0),
            C::default(),
            C::default(),
            C::default(),
            C::new(1.0, 0.0),
            C::default(),
            C::default(),
            C::default(),
            C::new(1.0, 0.0),
        ]
    );
    let scale = C::new(-0.25, 0.5);
    let scaled = a.scaled(scale);
    for i in 0..9 {
        assert_eq!(scaled.0[i], scale * a.0[i]);
    }
    let mut x = sample(-2.0);
    let s = x;
    x.add_scaled_real(1.25, a);
    for i in 0..9 {
        assert_eq!(x.0[i], s.0[i] + 1.25 * a.0[i]);
    }
    let s = x;
    x.add_scaled_complex(C::new(-0.5, 0.75), a);
    for i in 0..9 {
        assert_eq!(x.0[i], s.0[i] + C::new(-0.5, 0.75) * a.0[i]);
    }
}
#[test]
fn gell_mann_known_values_and_factor_extract() {
    let c = [0.2, -0.3, 0.5, 0.7, -0.11, 0.13, -0.17, 0.19];
    let r = 3f64.sqrt();
    let expected = Mat3([
        C::new(c[2] + c[7] / r, 0.0),
        C::new(c[0], c[1]),
        C::new(c[3], c[4]),
        C::new(c[0], -c[1]),
        C::new(-c[2] + c[7] / r, 0.0),
        C::new(c[5], c[6]),
        C::new(c[3], -c[4]),
        C::new(c[5], -c[6]),
        C::new(-2.0 * c[7] / r, 0.0),
    ]);
    let h = Mat3::hermitian_from_gell_mann(c);
    close(h, expected);
    let got = h.scaled(C::new(0.0, 0.5)).gell_mann_coefficients();
    for i in 0..8 {
        assert!((got[i] - c[i]).abs() < 1e-12);
    }
    let v = sample(0.37);
    let direct = coeff(v);
    assert_eq!(v.gell_mann_coefficients(), direct);
    let mut out = [1.0; 8];
    Mat3::add_ta_coefficients(&mut out, -0.4, v);
    for i in 0..8 {
        assert!((out[i] - (1.0 - 0.4 * direct[i])).abs() < 1e-12);
    }
}
