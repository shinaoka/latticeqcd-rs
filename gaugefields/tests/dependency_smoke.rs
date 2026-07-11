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
    let _ = std::any::TypeId::of::<tenferro_ad::EagerTensor>();
}
