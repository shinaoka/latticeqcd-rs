use gaugefields::{action_gradient, cold_su3, dsdu, gauge_force, LatticeShape4};
use num_complex::Complex64;
#[test]
fn cold_force_quantities_have_expected_shapes_and_values() {
    let l = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let u = cold_su3(l).unwrap();
    let d = dsdu(&u, 6.0).unwrap();
    for link in d.links() {
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
    assert_eq!(g.links()[0].tensor().shape(), &[3, 3, 2, 2, 2, 2]);
}
