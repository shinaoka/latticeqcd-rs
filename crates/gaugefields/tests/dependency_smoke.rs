use num_complex::Complex64;
use tenferro_tensor::Tensor;

#[test]
fn dependency_manifest_uses_current_tenferro_snapshot() {
    let manifest =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml")).unwrap();
    let current = "c942129974b544225ed963414d7be1300980f901";
    assert_eq!(manifest.matches(current).count(), 5);
}

#[test]
fn pinned_tenferro_constructs_column_major_c64_tensor() {
    let values = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let tensor = Tensor::from_vec_col_major(vec![2, 1], values.clone()).unwrap();

    assert_eq!(tensor.shape(), &[2, 1]);
    assert_eq!(tensor.as_slice::<Complex64>().unwrap(), values);
    let _ = std::any::TypeId::of::<tenferro_runtime::TracedTensor>();
    let _ = std::any::TypeId::of::<tenferro_cpu::CpuBackend>();
}
