use super::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use tenferro_runtime::extension::ExtensionOp;
use tenferro_runtime::{DType, SymDim};
use tenferro_tensor::{Tensor, TypedTensor};

fn link_shape() -> Vec<SymDim> {
    [3, 3, 2, 2, 2, 2].into_iter().map(SymDim::from).collect()
}

#[test]
fn family_identity_uses_beta_bits_and_active_directions() {
    let action = WilsonActionOp::new(6.0).unwrap();
    assert_eq!(action.family_id(), WILSON_ACTION_FAMILY);
    let mut a = DefaultHasher::new();
    action.payload_hash(&mut a);
    let mut b = DefaultHasher::new();
    WilsonActionOp::new(6.0).unwrap().payload_hash(&mut b);
    assert_eq!(a.finish(), b.finish());
    assert!(action.payload_eq(&WilsonActionOp::new(6.0).unwrap()));
    assert!(!action.payload_eq(&WilsonActionOp::new(-6.0).unwrap()));

    let jvp = WilsonActionJvpOp::new(6.0, vec![0, 3]).unwrap();
    assert_eq!(jvp.family_id(), WILSON_ACTION_JVP_FAMILY);
    assert_eq!(jvp.active_dirs, [0, 3]);
    assert!(WilsonActionJvpOp::new(6.0, vec![3, 0]).is_err());
    assert!(WilsonActionJvpOp::new(6.0, vec![0, 0]).is_err());
    assert!(WilsonActionJvpOp::new(6.0, vec![4]).is_err());
    assert_eq!(
        WilsonForceOp::new(6.0).unwrap().family_id(),
        WILSON_FORCE_FAMILY
    );
    for beta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(WilsonActionOp::new(beta).is_err());
        assert!(WilsonActionJvpOp::new(beta, vec![0]).is_err());
        assert!(WilsonForceOp::new(beta).is_err());
    }
}

#[test]
fn families_infer_exact_symbolic_contracts() {
    let shape = link_shape();
    let shapes = [&shape[..], &shape[..], &shape[..], &shape[..]];
    let action = WilsonActionOp::new(6.0).unwrap();
    assert_eq!(action.input_count(), 4);
    assert_eq!(action.output_count(), 1);
    assert_eq!(
        action.infer_output_meta(&[DType::C64; 4], &shapes).unwrap(),
        vec![(DType::F64, vec![])]
    );

    let jvp = WilsonActionJvpOp::new(6.0, vec![1, 3]).unwrap();
    let jvp_shapes = [
        &shape[..],
        &shape[..],
        &shape[..],
        &shape[..],
        &shape[..],
        &shape[..],
    ];
    assert_eq!(jvp.input_count(), 6);
    assert_eq!(
        jvp.infer_output_meta(&[DType::C64; 6], &jvp_shapes)
            .unwrap(),
        vec![(DType::F64, vec![])]
    );

    let scalar: [SymDim; 0] = [];
    let force_shapes = [&shape[..], &shape[..], &shape[..], &shape[..], &scalar];
    let force = WilsonForceOp::new(6.0).unwrap();
    assert_eq!(force.input_count(), 5);
    assert_eq!(force.output_count(), 4);
    assert_eq!(
        force
            .infer_output_meta(
                &[DType::C64, DType::C64, DType::C64, DType::C64, DType::F64],
                &force_shapes,
            )
            .unwrap(),
        vec![(DType::C64, shape.clone()); 4]
    );
}

#[test]
fn metadata_rejects_wrong_dtype_rank_color_lattice_tangent_and_seed() {
    let shape = link_shape();
    let shapes = [&shape[..], &shape[..], &shape[..], &shape[..]];
    assert!(WilsonActionOp::new(6.0)
        .unwrap()
        .infer_output_meta(&[DType::F64, DType::C64, DType::C64, DType::C64], &shapes)
        .is_err());
    let rank_five = vec![SymDim::from(3); 5];
    assert!(WilsonActionOp::new(6.0)
        .unwrap()
        .infer_output_meta(&[DType::C64; 4], &[&rank_five, &shape, &shape, &shape],)
        .is_err());
    let mut wrong_color = shape.clone();
    wrong_color[1] = SymDim::from(2);
    assert!(WilsonActionOp::new(6.0)
        .unwrap()
        .infer_output_meta(&[DType::C64; 4], &[&wrong_color, &shape, &shape, &shape],)
        .is_err());
    let mut wrong_lattice = shape.clone();
    wrong_lattice[5] = SymDim::from(3);
    assert!(WilsonActionOp::new(6.0)
        .unwrap()
        .infer_output_meta(&[DType::C64; 4], &[&shape, &shape, &shape, &wrong_lattice],)
        .is_err());
}

#[test]
fn jvp_and_force_host_references_execute_registered_contracts() {
    let links = crate::cold_su3(crate::LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let erased: [Tensor; 4] =
        std::array::from_fn(|mu| Tensor::C64(links.links()[mu].typed().clone()));
    let zero_tangent: Tensor =
        TypedTensor::from_vec_col_major(vec![3, 3, 1, 1, 1, 1], vec![Complex64::default(); 9])
            .unwrap()
            .into();
    let jvp_inputs = [
        &erased[0],
        &erased[1],
        &erased[2],
        &erased[3],
        &zero_tangent,
    ];
    let jvp = WilsonActionJvpOp::new(6.0, vec![2])
        .unwrap()
        .execute(&jvp_inputs)
        .unwrap();
    assert_eq!(jvp[0].as_slice::<f64>().unwrap(), &[0.0]);

    let seed: Tensor = TypedTensor::from_vec_col_major(vec![], vec![0.0_f64])
        .unwrap()
        .into();
    let force_inputs = [&erased[0], &erased[1], &erased[2], &erased[3], &seed];
    let force = WilsonForceOp::new(6.0)
        .unwrap()
        .execute(&force_inputs)
        .unwrap();
    assert_eq!(force.len(), 4);
    for output in force {
        assert_eq!(output.shape(), &[3, 3, 1, 1, 1, 1]);
        assert!(output
            .as_slice::<Complex64>()
            .unwrap()
            .iter()
            .all(|value| *value == Complex64::default()));
    }
}
