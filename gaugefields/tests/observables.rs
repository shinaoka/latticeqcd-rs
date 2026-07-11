use gaugefields::{
    cold_su3, measurement_staple, normalized_plaquette, plaquette_sum, wilson_action, LatticeShape4,
};

#[test]
fn cold_observables_have_exact_normalization() {
    let l = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let u = cold_su3(l).unwrap();
    assert_eq!(plaquette_sum(&u).unwrap(), 6.0 * l.nv() as f64 * 3.0);
    assert!((normalized_plaquette(&u).unwrap() - 1.0).abs() < 1e-14);
    assert_eq!(
        wilson_action(&u, 6.0).unwrap(),
        -6.0 / 3.0 * plaquette_sum(&u).unwrap()
    );
}
#[test]
fn direct_sum_equals_measurement_staple_contraction() {
    let l = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let u = cold_su3(l).unwrap();
    let v = measurement_staple(&u).unwrap();
    let mut sum = 0.0;
    for mu in 0..4 {
        for site in 0..l.nv() {
            sum += gaugefields::load_link(&u, mu, site)
                .unwrap()
                .mul_adj_right(gaugefields::load_link(&v, mu, site).unwrap())
                .trace()
                .re;
        }
    }
    assert!((plaquette_sum(&u).unwrap() - 0.5 * sum).abs() < 1e-12);
}
