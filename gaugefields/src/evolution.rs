use crate::{GaugeError, Mat3};
use num_complex::Complex64 as C;

fn taylor_four(v: Mat3) -> Mat3 {
    let v2 = v.mul(v);
    let v3 = v2.mul(v);
    let v4 = v2.mul(v2);
    let mut out = Mat3::identity();
    out.add_scaled_complex(C::new(0.0, 1.0), v);
    out.add_scaled_real(-0.5, v2);
    out.add_scaled_complex(C::new(0.0, -1.0 / 6.0), v3);
    out.add_scaled_real(1.0 / 24.0, v4);
    out
}

fn julia_eigenvector(v: Mat3, eigenvalue: f64) -> ([C; 3], f64) {
    let v1 = v[(0, 0)].re;
    let v3 = v[(0, 1)].re;
    let v4 = v[(0, 1)].im;
    let v5 = v[(0, 2)].re;
    let v6 = v[(0, 2)].im;
    let v9 = v[(1, 1)].re;
    let v11 = v[(1, 2)].re;
    let v12 = v[(1, 2)].im;
    let w1 = v5 * (v9 - eigenvalue) - v3 * v11 + v4 * v12;
    let w2 = -v6 * (v9 - eigenvalue) + v4 * v11 + v3 * v12;
    let w3 = (v1 - eigenvalue) * v11 - v3 * v5 - v4 * v6;
    let w4 = -(v1 - eigenvalue) * v12 - v4 * v5 + v3 * v6;
    let w5 = -(v1 - eigenvalue) * (v9 - eigenvalue) + v3 * v3 + v4 * v4;
    let norm2 = w1 * w1 + w2 * w2 + w3 * w3 + w4 * w4 + w5 * w5;
    ([C::new(w1, w2), C::new(w3, w4), C::new(w5, 0.0)], norm2)
}

/// Exponentiate `t * (i/2) Σ coeffs[a] λ_a` with the Gaugefields.jl branches.
pub fn exp_ta(t: f64, coeffs: &[f64; 8]) -> Result<Mat3, GaugeError> {
    if !t.is_finite() {
        return Err(GaugeError::NonFiniteSu3Input {
            operation: "exp_ta",
            component: 8,
        });
    }
    if let Some(component) = coeffs.iter().position(|value| !value.is_finite()) {
        return Err(GaugeError::NonFiniteSu3Input {
            operation: "exp_ta",
            component,
        });
    }
    let scaled = coeffs.map(|value| 0.5 * t * value);
    if scaled.iter().sum::<f64>() == 0.0 {
        return Ok(Mat3::identity());
    }
    let v = Mat3::hermitian_from_gell_mann(scaled);
    let trv3 = v.trace().re / 3.0;
    let v1 = v[(0, 0)].re;
    let v3 = v[(0, 1)].re;
    let v4 = v[(0, 1)].im;
    let v5 = v[(0, 2)].re;
    let v6 = v[(0, 2)].im;
    let v9 = v[(1, 1)].re;
    let v11 = v[(1, 2)].re;
    let v12 = v[(1, 2)].im;
    let v17 = v[(2, 2)].re;
    let cofac = v1 * v9 - v3 * v3 - v4 * v4 + v1 * v17 - v5 * v5 - v6 * v6 + v9 * v17
        - v11 * v11
        - v12 * v12;
    let det = v1 * v9 * v17
        - v1 * (v11 * v11 + v12 * v12)
        - v9 * (v5 * v5 + v6 * v6)
        - v17 * (v3 * v3 + v4 * v4)
        + 2.0 * (v5 * (v3 * v11 - v4 * v12) + v6 * (v3 * v12 + v4 * v11));
    let p3 = cofac / 3.0 - trv3 * trv3;
    let q = trv3 * cofac - det - 2.0 * trv3.powi(3);
    let x = (-4.0 * p3).sqrt() + 1e-100;
    let arg = (q / (x * p3)).clamp(-1.0, 1.0);
    let theta = arg.acos() / 3.0;
    let e1 = x * theta.cos() + trv3;
    let e2 = x * (theta + 2.0 * std::f64::consts::PI / 3.0).cos() + trv3;
    let e3 = 3.0 * trv3 - e1 - e2;
    let raw = [
        julia_eigenvector(v, e1),
        julia_eigenvector(v, e2),
        julia_eigenvector(v, e3),
    ];
    if raw.iter().any(|(_, norm2)| *norm2 < 1e-24) {
        return Ok(taylor_four(v));
    }
    let vectors = raw.map(|(mut vector, norm2)| {
        let scale = norm2.sqrt().recip();
        vector.iter_mut().for_each(|value| *value *= scale);
        vector
    });
    let eigenvalues = [e1, e2, e3];
    let mut out = Mat3::zero();
    for row in 0..3 {
        for col in 0..3 {
            out[(row, col)] = (0..3)
                .map(|k| {
                    vectors[k][row].conj() * C::from_polar(1.0, eigenvalues[k]) * vectors[k][col]
                })
                .sum();
        }
    }
    Ok(out)
}

/// Project a finite nonsingular matrix to SU(3) using Julia row completion.
pub fn normalize_su3(matrix: &mut Mat3) -> Result<(), GaugeError> {
    if let Some(component) = matrix
        .as_array()
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(GaugeError::NonFiniteSu3Input {
            operation: "normalize_su3",
            component,
        });
    }
    let mut row0 = [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]];
    let mut row1 = [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]];
    let norm0 = row0.iter().map(|value| value.norm_sqr()).sum::<f64>();
    if !norm0.is_finite() || norm0 <= 1e-30 {
        return Err(GaugeError::SingularSu3Normalization { row: 0 });
    }
    let projection = row1
        .iter()
        .zip(row0)
        .map(|(second, first)| *second * first.conj())
        .sum::<C>()
        / norm0;
    for (second, first) in row1.iter_mut().zip(row0) {
        *second -= projection * first;
    }
    let norm1 = row1.iter().map(|value| value.norm_sqr()).sum::<f64>();
    if !norm1.is_finite() || norm1 <= 1e-30 {
        return Err(GaugeError::SingularSu3Normalization { row: 1 });
    }
    let inv0 = norm0.sqrt().recip();
    let inv1 = norm1.sqrt().recip();
    row0.iter_mut().for_each(|value| *value *= inv0);
    row1.iter_mut().for_each(|value| *value *= inv1);
    let row2 = [
        (row0[1] * row1[2] - row0[2] * row1[1]).conj(),
        (row0[2] * row1[0] - row0[0] * row1[2]).conj(),
        (row0[0] * row1[1] - row0[1] * row1[0]).conj(),
    ];
    let mut projected = Mat3::zero();
    for column in 0..3 {
        projected[(0, column)] = row0[column];
        projected[(1, column)] = row1[column];
        projected[(2, column)] = row2[column];
    }
    *matrix = projected;
    Ok(())
}
