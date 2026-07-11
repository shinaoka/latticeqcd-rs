use gaugefields::{
    action_gradient, load_link, store_link, wilson_action, GaugeLinkTensor, GaugeLinks,
    LatticeShape4, Mat3,
};
use num_complex::Complex64 as C;
use tenferro_tensor::Tensor;
fn field(sign: f64, h: f64) -> GaugeLinks {
    let l = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    let mut links = Vec::new();
    for mu in 0..4 {
        let mut d = vec![C::default(); 18];
        for s in 0..2 {
            for j in 0..3 {
                for i in 0..3 {
                    d[9 * s + i + 3 * j] = C::new(
                        if i == j {
                            1.0
                        } else {
                            0.02 * (1 + i + j + mu) as f64
                        },
                        0.01 * (1 + s + i + 2 * j + mu) as f64,
                    );
                }
            }
        }
        links.push(
            GaugeLinkTensor::new(
                Tensor::from_vec_col_major(vec![3, 3, 2, 1, 1, 1], d).unwrap(),
                l,
            )
            .unwrap(),
        );
    }
    let mut u = GaugeLinks::new(links.try_into().unwrap()).unwrap();
    let dir = Mat3(std::array::from_fn(|i| {
        C::new((i + 1) as f64 / 17.0, -((i + 2) as f64) / 23.0)
    }));
    let x = load_link(&u, 1, 1).unwrap();
    store_link(
        &mut u,
        1,
        1,
        Mat3(std::array::from_fn(|i| x.0[i] + sign * h * dir.0[i])),
    )
    .unwrap();
    let y = load_link(&u, 2, 0).unwrap();
    store_link(
        &mut u,
        2,
        0,
        Mat3(std::array::from_fn(|i| y.0[i] + sign * h * dir.0[i])),
    )
    .unwrap();
    let z = load_link(&u, 0, 0).unwrap();
    store_link(
        &mut u,
        0,
        0,
        Mat3(std::array::from_fn(|i| z.0[i] + sign * h * dir.0[i])),
    )
    .unwrap();
    u
}
#[test]
fn dense_gradient_matches_second_order_central_difference() {
    let beta = 5.7;
    let base = field(0.0, 0.0);
    let g = load_link(&action_gradient(&base, beta).unwrap(), 1, 1).unwrap();
    let d = Mat3(std::array::from_fn(|i| {
        C::new((i + 1) as f64 / 17.0, -((i + 2) as f64) / 23.0)
    }));
    let g2 = load_link(&action_gradient(&base, beta).unwrap(), 2, 0).unwrap();
    let g3 = load_link(&action_gradient(&base, beta).unwrap(), 0, 0).unwrap();
    let analytic = g.adjoint().mul(d).trace().re
        + g2.adjoint().mul(d).trace().re
        + g3.adjoint().mul(d).trace().re;
    let mut errors = Vec::new();
    for h in [0.1, 0.05, 0.025] {
        let fd = (wilson_action(&field(1.0, h), beta).unwrap()
            - wilson_action(&field(-1.0, h), beta).unwrap())
            / (2.0 * h);
        errors.push((fd - analytic).abs());
    }
    assert!(
        errors[1] < errors[0] && errors[2] < errors[1],
        "analytic={analytic} errors={errors:?}"
    );
    assert!(errors[2] < 1e-2, "errors={errors:?}");
}
