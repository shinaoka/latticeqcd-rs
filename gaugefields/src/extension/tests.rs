use super::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::path::Path;
#[cfg(feature = "autodiff")]
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::ExtensionOp;
use tenferro_runtime::{DType, SymDim};
#[cfg(feature = "autodiff")]
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::{
    Buffer, BufferHandle, DeviceId, DeviceKind, Error as TensorError, GpuBackendKind, MemoryKind,
    Placement, Tensor, TypedTensor,
};

type InferredMeta = tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>>;

fn link_shape() -> Vec<SymDim> {
    [3, 3, 2, 2, 2, 2].into_iter().map(SymDim::from).collect()
}

fn assert_invalid_without_panic(result: std::thread::Result<InferredMeta>) {
    assert!(result.is_ok(), "metadata inference panicked");
    assert!(result.unwrap().is_err());
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

    let jvp = WilsonActionJvpOp::new(6.0, vec![2]).unwrap();
    let jvp_shapes = [&shape[..], &shape, &shape, &shape, &shape];
    assert_invalid_without_panic(std::panic::catch_unwind(|| {
        jvp.infer_output_meta(
            &[DType::C64, DType::C64, DType::C64, DType::C64, DType::F64],
            &jvp_shapes,
        )
    }));
    assert_invalid_without_panic(std::panic::catch_unwind(|| {
        jvp.infer_output_meta(
            &[DType::C64; 5],
            &[&shape, &shape, &shape, &shape, &rank_five],
        )
    }));
    let mut wrong_tangent_shape = shape.clone();
    wrong_tangent_shape[5] = SymDim::from(7);
    assert_invalid_without_panic(std::panic::catch_unwind(|| {
        jvp.infer_output_meta(
            &[DType::C64; 5],
            &[&shape, &shape, &shape, &shape, &wrong_tangent_shape],
        )
    }));

    let scalar: [SymDim; 0] = [];
    let force = WilsonForceOp::new(6.0).unwrap();
    assert_invalid_without_panic(std::panic::catch_unwind(|| {
        force.infer_output_meta(&[DType::C64; 5], &[&shape, &shape, &shape, &shape, &scalar])
    }));
    let rank_one_seed = [SymDim::from(1)];
    assert_invalid_without_panic(std::panic::catch_unwind(|| {
        force.infer_output_meta(
            &[DType::C64, DType::C64, DType::C64, DType::C64, DType::F64],
            &[&shape, &shape, &shape, &shape, &rank_one_seed],
        )
    }));
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

#[test]
fn host_reference_rejects_exact_link_shape_mismatch_without_panicking() {
    let common: Tensor =
        TypedTensor::from_vec_col_major(vec![3, 3, 1, 1, 1, 1], vec![Complex64::default(); 9])
            .unwrap()
            .into();
    let mismatched: Tensor =
        TypedTensor::from_vec_col_major(vec![3, 3, 2, 1, 1, 1], vec![Complex64::default(); 18])
            .unwrap()
            .into();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        WilsonActionOp::new(6.0)
            .unwrap()
            .execute(&[&common, &common, &common, &mismatched])
    }));
    assert!(result.is_ok(), "host-reference validation panicked");
    let error = result.unwrap().unwrap_err().to_string();
    assert!(error.contains("different lattice shape"), "{error}");
}

fn random_fixture_links() -> crate::GaugeLinks {
    let fixture = crate::load_fixture(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/random_2x2x2x2"),
    )
    .unwrap();
    let lattice = fixture.links().lattice();
    crate::GaugeLinks::new(std::array::from_fn(|mu| {
        crate::GaugeLinkTensor::from_typed(fixture.links().links()[mu].typed().clone(), lattice)
            .unwrap()
    }))
    .unwrap()
}

fn tangent(mu: usize, len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            Complex64::new(
                (1 + i + 3 * mu) as f64 / 211.0,
                -((2 + 2 * i + mu) as f64) / 307.0,
            )
        })
        .collect()
}

fn perturbed_links(
    base: &crate::GaugeLinks,
    active_dirs: &[usize],
    tangents: &[Vec<Complex64>],
    scale: f64,
) -> crate::GaugeLinks {
    let links = std::array::from_fn(|mu| {
        let mut data = base.links()[mu].typed().host_data().unwrap().to_vec();
        if let Some(index) = active_dirs.iter().position(|&active| active == mu) {
            for (value, delta) in data.iter_mut().zip(&tangents[index]) {
                *value += scale * delta;
            }
        }
        crate::GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(base.links()[mu].typed().shape().to_vec(), data)
                .unwrap(),
            base.lattice(),
        )
        .unwrap()
    });
    crate::GaugeLinks::new(links).unwrap()
}

#[test]
fn random_fixture_jvp_matches_finite_difference_and_direct_gradient() {
    let base = random_fixture_links();
    let beta = 5.7;
    let active_dirs = [0, 2];
    let tangents = active_dirs
        .iter()
        .map(|&mu| tangent(mu, 9 * base.lattice().nv()))
        .collect::<Vec<_>>();
    let erased: [Tensor; 4] =
        std::array::from_fn(|mu| Tensor::C64(base.links()[mu].typed().clone()));
    let tangent_tensors = tangents
        .iter()
        .map(|values| {
            Tensor::C64(
                TypedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values.clone()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let inputs = [
        &erased[0],
        &erased[1],
        &erased[2],
        &erased[3],
        &tangent_tensors[0],
        &tangent_tensors[1],
    ];
    let actual = WilsonActionJvpOp::new(beta, active_dirs.to_vec())
        .unwrap()
        .execute(&inputs)
        .unwrap()[0]
        .as_slice::<f64>()
        .unwrap()[0];
    let gradient = crate::action_gradient(&base, beta).unwrap();
    let direct = active_dirs
        .iter()
        .enumerate()
        .map(|(index, &mu)| {
            gradient[mu]
                .typed()
                .host_data()
                .unwrap()
                .iter()
                .zip(&tangents[index])
                .map(|(g, delta)| (g.conj() * delta).re)
                .sum::<f64>()
        })
        .sum::<f64>();
    let h = 1e-6;
    let finite_difference =
        (crate::wilson_action(&perturbed_links(&base, &active_dirs, &tangents, h), beta).unwrap()
            - crate::wilson_action(&perturbed_links(&base, &active_dirs, &tangents, -h), beta)
                .unwrap())
            / (2.0 * h);
    assert!(
        (actual - direct).abs() < 1e-11,
        "actual={actual} direct={direct} residual={}",
        (actual - direct).abs()
    );
    assert!(
        (actual - finite_difference).abs() < 1e-7,
        "actual={actual} finite_difference={finite_difference} residual={}",
        (actual - finite_difference).abs()
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn traced_all_direction_jvp_graph_matches_sum_and_finite_difference_sweep() {
    let base = random_fixture_links();
    let beta = 5.7;
    let active_dirs = [0, 1, 2, 3];
    let tangents = active_dirs
        .iter()
        .map(|&mu| tangent(mu, 9 * base.lattice().nv()))
        .collect::<Vec<_>>();
    let links: [TracedTensor; 4] = std::array::from_fn(|mu| {
        TracedTensor::from_vec_col_major(
            base.links()[mu].typed().shape().to_vec(),
            base.links()[mu].typed().host_data().unwrap().to_vec(),
        )
        .unwrap()
    });
    let tangent_tensors = tangents
        .iter()
        .map(|values| {
            TracedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values.clone()).unwrap()
        })
        .collect::<Vec<_>>();
    let inputs = [
        &links[0],
        &links[1],
        &links[2],
        &links[3],
        &tangent_tensors[0],
        &tangent_tensors[1],
        &tangent_tensors[2],
        &tangent_tensors[3],
    ];
    let traced = apply(
        Arc::new(WilsonActionJvpOp::new(beta, active_dirs.to_vec()).unwrap()),
        &inputs,
    )
    .unwrap()
    .into_iter()
    .next()
    .unwrap();
    let program = GraphCompiler::new().compile(&traced).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(crate::register_runtime)
        .unwrap();
    let actual = executor.run(&program).unwrap().as_slice::<f64>().unwrap()[0];
    let gradient = crate::action_gradient(&base, beta).unwrap();
    let direct = active_dirs
        .iter()
        .enumerate()
        .map(|(index, &mu)| {
            gradient[mu]
                .typed()
                .host_data()
                .unwrap()
                .iter()
                .zip(&tangents[index])
                .map(|(g, delta)| (g.conj() * delta).re)
                .sum::<f64>()
        })
        .sum::<f64>();
    let sweep = [1e-3, 2.5e-4, 6.25e-5, 1.5625e-5].map(|h| {
        let fd = (crate::wilson_action(&perturbed_links(&base, &active_dirs, &tangents, h), beta)
            .unwrap()
            - crate::wilson_action(&perturbed_links(&base, &active_dirs, &tangents, -h), beta)
                .unwrap())
            / (2.0 * h);
        (h, fd, (actual - fd).abs())
    });
    let best = sweep.iter().min_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
    assert!(actual.is_finite() && direct.is_finite() && best.1.is_finite());
    assert!(
        (actual - direct).abs() < 1e-10,
        "actual={actual} direct={direct} residual={}",
        (actual - direct).abs()
    );
    assert!(
        best.2 < 1e-6,
        "actual={actual} best_h={} fd={} residual={} sweep={sweep:?}",
        best.0,
        best.1,
        best.2
    );
}

#[test]
fn random_fixture_force_callback_matches_seeded_direct_gradient() {
    let base = random_fixture_links();
    let beta = 5.7;
    let gradient = crate::action_gradient(&base, beta).unwrap();
    let erased: [Tensor; 4] =
        std::array::from_fn(|mu| Tensor::C64(base.links()[mu].typed().clone()));
    for seed in [1.0, -2.5, 0.25] {
        let seed_tensor: Tensor = TypedTensor::from_vec_col_major(vec![], vec![seed])
            .unwrap()
            .into();
        let outputs = WilsonForceOp::new(beta)
            .unwrap()
            .execute(&[&erased[0], &erased[1], &erased[2], &erased[3], &seed_tensor])
            .unwrap();
        for mu in 0..4 {
            for (index, (actual, expected)) in outputs[mu]
                .as_slice::<Complex64>()
                .unwrap()
                .iter()
                .zip(gradient[mu].typed().host_data().unwrap())
                .enumerate()
            {
                let residual = (*actual - seed * expected).norm();
                assert!(
                    residual < 1e-13,
                    "seed={seed} mu={mu} index={index} actual={actual} expected={} residual={residual}",
                    seed * expected
                );
            }
        }
    }
}

fn device_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
    }
}

fn device_c64(shape: Vec<usize>, len: usize) -> Tensor {
    Tensor::C64(
        TypedTensor::from_buffer_col_major(
            shape,
            Buffer::Backend(Arc::new(BufferHandle::new_with_len(31, len))),
            device_placement(),
        )
        .unwrap(),
    )
}

fn device_f64(shape: Vec<usize>, len: usize) -> Tensor {
    Tensor::F64(
        TypedTensor::from_buffer_col_major(
            shape,
            Buffer::Backend(Arc::new(BufferHandle::new_with_len(32, len))),
            device_placement(),
        )
        .unwrap(),
    )
}

#[test]
fn callback_abi_preserves_placement_variants_and_types_domain_errors() {
    let shape = vec![3, 3, 1, 1, 1, 1];
    let host_link: Tensor =
        TypedTensor::from_vec_col_major(shape.clone(), vec![Complex64::default(); 9])
            .unwrap()
            .into();
    let device_link = device_c64(shape.clone(), 9);
    let action_error = WilsonActionOp::new(6.0)
        .unwrap()
        .execute(&[&device_link, &device_link, &device_link, &device_link])
        .unwrap_err();
    assert!(matches!(
        action_error,
        TensorError::BackendFailure {
            op: "TypedTensor::host_data",
            ..
        }
    ));

    let device_tangent = device_c64(shape, 9);
    let jvp_error = WilsonActionJvpOp::new(6.0, vec![1])
        .unwrap()
        .execute(&[
            &host_link,
            &host_link,
            &host_link,
            &host_link,
            &device_tangent,
        ])
        .unwrap_err();
    assert!(matches!(
        jvp_error,
        TensorError::BackendFailure {
            op: "TypedTensor::host_data",
            ..
        }
    ));

    let device_seed = device_f64(vec![], 1);
    let force_error = WilsonForceOp::new(6.0)
        .unwrap()
        .execute(&[&host_link, &host_link, &host_link, &host_link, &device_seed])
        .unwrap_err();
    assert!(matches!(
        force_error,
        TensorError::BackendFailure {
            op: "TypedTensor::host_data",
            ..
        }
    ));

    assert!(matches!(
        abi_error(
            WILSON_ACTION_FAMILY,
            GaugeError::InvalidDirection { direction: 4 }
        ),
        TensorError::InvalidConfig {
            op: WILSON_ACTION_FAMILY,
            ..
        }
    ));
}
