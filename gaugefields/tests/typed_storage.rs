use gaugefields::{GaugeError, GaugeLinkTensor, LatticeShape4, TaGaugeField};
use num_complex::Complex64;
use tenferro_tensor::{Tensor, TypedTensor};

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
    let field = TaGaugeField::new([ta.clone(), ta.clone(), ta.clone(), ta], lattice).unwrap();
    assert_eq!(field.tensors()[0].shape(), &[8, 2, 1, 1, 1]);
    assert!(!format!("{field:?}").contains("0.0"));

    let wrong = TypedTensor::from_vec_col_major(vec![8, 2], vec![0.0; 16]).unwrap();
    assert!(matches!(
        TaGaugeField::new(
            [wrong.clone(), wrong.clone(), wrong.clone(), wrong],
            lattice
        ),
        Err(GaugeError::Rank {
            expected: 5,
            found: 2
        })
    ));
}
