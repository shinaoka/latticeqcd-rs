use gaugefields::{GaugeError, GaugeLinkTensor, LatticeShape4, TaGaugeField};
use num_complex::Complex64;
use tenferro_tensor::{
    BackendStorageHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
    StorageBuffer, Tensor, TypedTensor,
};

#[test]
fn typed_storage_has_the_phase_6_public_boundary() {
    let _: fn(TypedTensor<Complex64>, LatticeShape4) -> Result<GaugeLinkTensor, GaugeError> =
        GaugeLinkTensor::from_typed;
    let _: fn(Tensor, LatticeShape4) -> Result<GaugeLinkTensor, GaugeError> =
        GaugeLinkTensor::try_from_tensor;
    let _: fn([TypedTensor<f64>; 4], LatticeShape4) -> Result<TaGaugeField, GaugeError> =
        TaGaugeField::new;
}

#[test]
fn host_only_typed_constructors_return_structured_placement_errors() {
    let lattice = LatticeShape4::new([1, 1, 1, 1]).unwrap();
    let placement = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
        cpu_affinity: None,
    };
    let link = TypedTensor::from_buffer_col_major(
        vec![3, 3, 1, 1, 1, 1],
        StorageBuffer::Backend(Box::new(BackendStorageHandle::<Complex64>::new_with_len(
            1, 9,
        ))),
        placement.clone(),
    )
    .unwrap();
    assert!(matches!(
        GaugeLinkTensor::from_typed(link, lattice),
        Err(GaugeError::Placement { .. })
    ));

    let ta = || {
        TypedTensor::from_buffer_col_major(
            vec![8, 1, 1, 1, 1],
            StorageBuffer::Backend(Box::new(BackendStorageHandle::<f64>::new_with_len(2, 8))),
            placement.clone(),
        )
        .unwrap()
    };
    assert!(matches!(
        TaGaugeField::new([ta(), ta(), ta(), ta()], lattice),
        Err(GaugeError::Placement { .. })
    ));
}

#[test]
fn zero_ta_field_has_checked_site_coefficient_access() {
    let lattice = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    let mut field = TaGaugeField::zeros(lattice).unwrap();
    field.add_site_coefficients(1, 1, [0.25; 8]).unwrap();
    assert_eq!(field.site_coefficients(1, 1).unwrap(), [0.25; 8]);
    assert!(matches!(
        field.add_site_coefficients(4, 0, [0.0; 8]),
        Err(gaugefields::GaugeError::InvalidDirection { direction: 4 })
    ));
    assert!(matches!(
        field.site_coefficients(0, 2),
        Err(gaugefields::GaugeError::SiteOutOfBounds { site: 2, volume: 2 })
    ));
}

#[test]
fn typed_storage_validates_shapes_and_has_compact_debug() {
    let lattice = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    let link = GaugeLinkTensor::from_typed(
        TypedTensor::from_vec_col_major(vec![3, 3, 2, 1, 1, 1], vec![Complex64::default(); 18])
            .unwrap(),
        lattice,
    )
    .unwrap();
    assert_eq!(link.typed().shape(), &[3, 3, 2, 1, 1, 1]);
    assert!(!format!("{link:?}").contains("0.0"));

    let ta = TypedTensor::from_vec_col_major(vec![8, 2, 1, 1, 1], vec![0.0; 16]).unwrap();
    let field = TaGaugeField::new(
        [
            ta.duplicate().unwrap(),
            ta.duplicate().unwrap(),
            ta.duplicate().unwrap(),
            ta,
        ],
        lattice,
    )
    .unwrap();
    assert_eq!(field.tensors()[0].shape(), &[8, 2, 1, 1, 1]);
    assert!(!format!("{field:?}").contains("0.0"));

    let wrong = TypedTensor::from_vec_col_major(vec![8, 2], vec![0.0; 16]).unwrap();
    assert!(matches!(
        TaGaugeField::new(
            [
                wrong.duplicate().unwrap(),
                wrong.duplicate().unwrap(),
                wrong.duplicate().unwrap(),
                wrong,
            ],
            lattice
        ),
        Err(GaugeError::Rank {
            expected: 5,
            found: 2
        })
    ));
}
