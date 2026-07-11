use gaugefields::{
    action_gradient, load_link, store_link, wilson_action, GaugeLinkTensor, GaugeLinks,
    LatticeShape4, Mat3,
};
use num_complex::Complex64 as C;
use tenferro_tensor::Tensor;
fn supports(case: usize) -> [(usize, usize); 3] {
    match case {
        0 => [(0, 0), (1, 1), (0, 2)],
        1 => [(0, 1), (1, 0), (0, 3)],
        _ => [(0, 2), (1, 3), (0, 0)],
    }
}
fn direction(case: usize) -> Mat3 {
    Mat3(std::array::from_fn(|i| {
        C::new(
            (i + 1 + case) as f64 / 17.0,
            -((2 * i + 2 + case) as f64) / 23.0,
        )
    }))
}
fn field(case: usize, sign: f64, h: f64) -> GaugeLinks {
    let l = LatticeShape4::new([2, 2, 1, 1]).unwrap();
    let mut links = Vec::new();
    for mu in 0..4 {
        let mut d = vec![C::default(); 36];
        for s in 0..4 {
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
                Tensor::from_vec_col_major(vec![3, 3, 2, 2, 1, 1], d).unwrap(),
                l,
            )
            .unwrap(),
        );
    }
    let mut u = GaugeLinks::new(links.try_into().unwrap()).unwrap();
    let dir = direction(case);
    for (mu, site) in supports(case) {
        let x = load_link(&u, mu, site).unwrap();
        store_link(
            &mut u,
            mu,
            site,
            Mat3(std::array::from_fn(|i| x.0[i] + sign * h * dir.0[i])),
        )
        .unwrap();
    }
    u
}
#[test]
fn dense_gradient_matches_second_order_central_difference() {
    let beta = 5.7;
    for case in 0..3 {
        let base = field(case, 0.0, 0.0);
        let grad = action_gradient(&base, beta).unwrap();
        let d = direction(case);
        let analytic = supports(case)
            .into_iter()
            .map(|(mu, site)| {
                load_link(&grad, mu, site)
                    .unwrap()
                    .adjoint()
                    .mul(d)
                    .trace()
                    .re
            })
            .sum::<f64>();
        let mut errors = Vec::new();
        for h in [0.1, 0.05, 0.025] {
            let fd = (wilson_action(&field(case, 1.0, h), beta).unwrap()
                - wilson_action(&field(case, -1.0, h), beta).unwrap())
                / (2.0 * h);
            errors.push((fd - analytic).abs());
        }
        assert!(
            errors[1] < errors[0] && errors[2] < errors[1],
            "case={case} analytic={analytic} errors={errors:?}"
        );
        let ratios = [errors[0] / errors[1], errors[1] / errors[2]];
        assert!(
            ratios.iter().all(|&r| (3.5..=4.5).contains(&r)),
            "case={case} analytic={analytic} errors={errors:?} ratios={ratios:?}"
        );
        assert!(
            errors[2] < 1e-2,
            "case={case} errors={errors:?} ratios={ratios:?}"
        );
    }
}
