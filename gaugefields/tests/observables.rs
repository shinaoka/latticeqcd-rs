use gaugefields::{
    cold_su3, load_fixture, measurement_staple, normalized_plaquette, plaquette_sum, wilson_action,
    GaugeError, LatticeShape4, Mat3,
};
use std::path::Path;

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
    let fixture =
        load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/random_2x2x2x2"))
            .unwrap();
    let u = fixture.links();
    let l = u.lattice();
    let mut sum = 0.0;
    for mu in 0..4 {
        let v = measurement_staple(u, mu).unwrap();
        let data = v.typed().host_data().unwrap();
        for site in 0..l.nv() {
            sum += gaugefields::load_link(u, mu, site)
                .unwrap()
                .mul_adj_right(Mat3::load(data, 9 * site).unwrap())
                .trace()
                .re;
        }
    }
    assert!((plaquette_sum(u).unwrap() - 0.5 * sum).abs() < 1e-12);
    assert!(matches!(
        measurement_staple(u, 4),
        Err(GaugeError::InvalidDirection { direction: 4 })
    ));
}

#[test]
fn shared_prepared_kernel_contract() {
    let observables = include_str!("../src/observables.rs");
    let force = include_str!("../src/force.rs");
    assert!(observables.contains("PreparedGaugeField::new"));
    assert!(!observables.contains("load_link("));
    assert!(force.contains("PreparedGaugeField::new"));
    assert!(!force.contains("fn site_staple"));
}
