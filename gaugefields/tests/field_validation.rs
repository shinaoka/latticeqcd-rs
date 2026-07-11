use gaugefields::{
    cold_su3, require_su3, Boundary, GaugeError, GaugeLinkTensor, GaugeLinks, LatticeShape4,
};
use num_complex::Complex64;
use tenferro_tensor::Tensor;

#[test]
fn lattice_shape_rejects_every_zero_extent() {
    for axis in 0..4 {
        let mut dims = [2, 3, 4, 5];
        dims[axis] = 0;
        assert!(
            matches!(LatticeShape4::new(dims), Err(GaugeError::InvalidExtent { axis: a }) if a == axis)
        );
    }
}

#[test]
fn lattice_and_cold_allocation_overflow_are_typed_errors() {
    assert!(matches!(
        LatticeShape4::new([usize::MAX, 2, 1, 1]),
        Err(GaugeError::VolumeOverflow)
    ));
    let lattice = LatticeShape4::new([usize::MAX / 8 + 1, 1, 1, 1]).unwrap();
    assert_eq!(lattice.nv(), usize::MAX / 8 + 1);
    assert!(matches!(
        cold_su3(lattice),
        Err(GaugeError::AllocationOverflow)
    ));
}

#[test]
fn link_validation_rejects_dtype_rank_and_shape() {
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let wrong_dtype =
        Tensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], vec![0.0_f64; 144]).unwrap();
    assert!(matches!(
        GaugeLinkTensor::new(wrong_dtype, lattice),
        Err(GaugeError::DType { .. })
    ));
    let wrong_rank =
        Tensor::from_vec_col_major(vec![3, 3, 16], vec![Complex64::default(); 144]).unwrap();
    assert!(matches!(
        GaugeLinkTensor::new(wrong_rank, lattice),
        Err(GaugeError::Rank { .. })
    ));
    let wrong_shape =
        Tensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 3], vec![Complex64::default(); 216])
            .unwrap();
    assert!(matches!(
        GaugeLinkTensor::new(wrong_shape, lattice),
        Err(GaugeError::Shape { .. })
    ));
}

#[test]
fn four_links_must_have_consistent_lattices() {
    let a = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let b = cold_su3(LatticeShape4::new([2, 1, 1, 1]).unwrap()).unwrap();
    let [a0, a1, a2, _] = a.into_links();
    let [b0, _, _, _] = b.into_links();
    assert!(matches!(
        GaugeLinks::new([a0, a1, a2, b0]),
        Err(GaugeError::InconsistentMu { mu: 3 })
    ));
}

#[test]
fn cold_su3_is_identity_at_every_site_and_periodic() {
    let lattice = LatticeShape4::new([2, 1, 1, 2]).unwrap();
    assert_eq!(lattice.nv(), 4);
    let links = cold_su3(lattice).unwrap();
    assert_eq!(links.boundary(), Boundary::Periodic);
    for link in links.links() {
        for block in link
            .tensor()
            .as_slice::<Complex64>()
            .unwrap()
            .chunks_exact(9)
        {
            for b in 0..3 {
                for a in 0..3 {
                    assert_eq!(block[a + 3 * b], Complex64::new((a == b) as u8 as f64, 0.0));
                }
            }
        }
    }
}

#[test]
fn runtime_nc_is_validated_across_mu_and_only_su3_boundary_rejects_nc2() {
    let lattice = LatticeShape4::new([1, 1, 1, 1]).unwrap();
    let make = |nc| {
        GaugeLinkTensor::new(
            Tensor::from_vec_col_major(
                vec![nc, nc, 1, 1, 1, 1],
                vec![Complex64::default(); nc * nc],
            )
            .unwrap(),
            lattice,
        )
        .unwrap()
    };
    let links = GaugeLinks::new([make(2), make(2), make(2), make(2)]).unwrap();
    assert_eq!(links.nc(), 2);
    assert!(matches!(
        require_su3(&links),
        Err(GaugeError::UnsupportedNc { found: 2 })
    ));
    assert!(matches!(
        GaugeLinks::new([make(2), make(2), make(2), make(3)]),
        Err(GaugeError::InconsistentMu { mu: 3 })
    ));
}
