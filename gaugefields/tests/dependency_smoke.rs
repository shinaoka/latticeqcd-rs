use num_complex::Complex64;
use tenferro_tensor::Tensor;

#[test]
fn pinned_tenferro_constructs_column_major_c64_tensor() {
    let values = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let tensor = Tensor::from_vec_col_major(vec![2, 1], values.clone()).unwrap();

    assert_eq!(tensor.shape(), &[2, 1]);
    assert_eq!(tensor.as_slice::<Complex64>().unwrap(), values);
    let _ = std::any::TypeId::of::<tenferro_runtime::TracedTensor>();
    let _ = std::any::TypeId::of::<tenferro_cpu::CpuBackend>();
}

#[cfg(feature = "autodiff")]
#[test]
fn autodiff_dependency_is_linked() {
    let op = gaugefields::autodiff::GaugeIdentityOp::new();
    let input = tenferro_runtime::TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let outputs = tenferro_runtime::extension::apply(std::sync::Arc::new(op), &[&input]).unwrap();
    assert_eq!(outputs.len(), 1);
    let rules = gaugefields::autodiff::extension_rules().unwrap();
    let ad = tenferro_ad::AdContext::builder()
        .with_extension_rules(rules)
        .build()
        .unwrap();
    assert!(ad
        .extension_rules()
        .is_linearize_registered("gaugefields.identity.v1"));
}
