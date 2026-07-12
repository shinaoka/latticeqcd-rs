use gaugefields::{
    action_gradient, cold_su3, dsdu, gauge_force, GaugeError, GaugeLinkTensor, GaugeLinks,
    LatticeShape4,
};
use num_complex::Complex64;
use std::path::Path;

#[test]
fn complex_derivative_functions_have_fixed_direction_array_boundary() {
    let _: fn(&GaugeLinks, f64) -> Result<[GaugeLinkTensor; 4], GaugeError> = dsdu;
    let _: fn(&GaugeLinks, f64) -> Result<[GaugeLinkTensor; 4], GaugeError> = action_gradient;
}
#[test]
fn cold_force_quantities_have_expected_shapes_and_values() {
    let l = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let u = cold_su3(l).unwrap();
    let d = dsdu(&u, 6.0).unwrap();
    for link in &d {
        for block in link
            .tensor()
            .as_slice::<Complex64>()
            .unwrap()
            .chunks_exact(9)
        {
            for j in 0..3 {
                for i in 0..3 {
                    assert!(
                        (block[i + 3 * j] - Complex64::new(if i == j { 18.0 } else { 0.0 }, 0.0))
                            .norm()
                            < 1e-12
                    );
                }
            }
        }
    }
    let f = gauge_force(&u, 6.0).unwrap();
    for t in f.tensors() {
        assert_eq!(t.shape(), &[8, 2, 2, 2, 2]);
        assert!(t.as_slice::<f64>().unwrap().iter().all(|x| x.abs() < 1e-12));
    }
    let g = action_gradient(&u, 6.0).unwrap();
    assert_eq!(g[0].tensor().shape(), &[3, 3, 2, 2, 2, 2]);
}

#[test]
fn nonzero_force_coefficients_match_independent_local_ta_formula() {
    let f = gaugefields::load_fixture(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/random_2x2x2x2"),
    )
    .unwrap();
    let d = dsdu(f.links(), f.metadata().beta).unwrap();
    let force = gauge_force(f.links(), f.metadata().beta).unwrap();
    let dsdu0 = gaugefields::Mat3::load(d[0].tensor().as_slice::<Complex64>().unwrap(), 0).unwrap();
    let a = gaugefields::load_link(f.links(), 0, 0)
        .unwrap()
        .mul(dsdu0)
        .ta();
    let expected = [
        a[(0, 1)].im + a[(1, 0)].im,
        a[(0, 1)].re - a[(1, 0)].re,
        a[(0, 0)].im - a[(1, 1)].im,
        a[(0, 2)].im + a[(2, 0)].im,
        a[(0, 2)].re - a[(2, 0)].re,
        a[(1, 2)].im + a[(2, 1)].im,
        a[(1, 2)].re - a[(2, 1)].re,
        (a[(0, 0)].im + a[(1, 1)].im - 2.0 * a[(2, 2)].im) / 3f64.sqrt(),
    ];
    let actual = &force.tensors()[0].as_slice::<f64>().unwrap()[..8];
    assert!(expected.iter().any(|x| x.abs() > 1e-8));
    for i in 0..8 {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-13,
            "i={i} actual={} expected={}",
            actual[i],
            expected[i]
        );
    }
}
