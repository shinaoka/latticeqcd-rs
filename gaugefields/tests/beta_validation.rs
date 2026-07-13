use gaugefields::{
    action_gradient, cold_su3, dsdu, gauge_force, wilson_action, wilson_action_traced, GaugeError,
    LatticeShape4,
};
use tenferro_runtime::{DType, TracedTensor};

#[test]
fn every_public_beta_boundary_rejects_nonfinite_values() {
    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let traced: [TracedTensor; 4] = std::array::from_fn(|_| {
        TracedTensor::input_concrete_shape(DType::C64, &[3, 3, 1, 1, 1, 1]).unwrap()
    });
    for beta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            wilson_action(&links, beta),
            Err(GaugeError::NonFiniteBeta { .. })
        ));
        assert!(matches!(
            dsdu(&links, beta),
            Err(GaugeError::NonFiniteBeta { .. })
        ));
        assert!(matches!(
            action_gradient(&links, beta),
            Err(GaugeError::NonFiniteBeta { .. })
        ));
        assert!(matches!(
            gauge_force(&links, beta),
            Err(GaugeError::NonFiniteBeta { .. })
        ));
        assert!(matches!(
            wilson_action_traced([&traced[0], &traced[1], &traced[2], &traced[3]], beta,),
            Err(GaugeError::NonFiniteBeta { .. })
        ));
    }
}
