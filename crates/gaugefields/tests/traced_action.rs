use gaugefields::{load_fixture, runtime_modules, wilson_action, wilson_action_traced};
use std::{error::Error as _, path::Path};
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{DType, GraphCompiler, Runtime, Tensor, TracedTensor};
use tenferro_tensor::{
    BackendStorageHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
    StorageBuffer, TypedTensor,
};

fn runtime(install_modules: bool) -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&backend).unwrap())
        .unwrap();
    if install_modules {
        for module in runtime_modules::<CpuBackend>(runtime_engine_id().unwrap()).unwrap() {
            builder.install_extension_module(module).unwrap();
        }
    }
    builder.build().unwrap()
}

fn traced_inputs(shape: &[usize]) -> [TracedTensor; 4] {
    std::array::from_fn(|_| TracedTensor::input_concrete_shape(DType::C64, shape).unwrap())
}

#[test]
fn unresolved_symbolic_lattice_mismatch_reaches_runtime_and_is_rejected() {
    let traced = std::array::from_fn::<_, 4, _>(|_| {
        TracedTensor::input_symbolic_shape(DType::C64, 6).unwrap()
    });
    let output =
        wilson_action_traced([&traced[0], &traced[1], &traced[2], &traced[3]], 6.0).unwrap();
    let common_shape = [3, 3, 1, 1, 1, 1];
    let mismatched_shape = [3, 3, 2, 1, 1, 1];
    let specs = traced
        .iter()
        .map(|input| (input, DType::C64, &common_shape[..]))
        .collect::<Vec<_>>();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&output, &specs)
        .unwrap();
    let common = Tensor::C64(
        TypedTensor::from_vec_col_major(
            common_shape.to_vec(),
            vec![num_complex::Complex64::default(); 9],
        )
        .unwrap(),
    );
    let mismatched = Tensor::C64(
        TypedTensor::from_vec_col_major(
            mismatched_shape.to_vec(),
            vec![num_complex::Complex64::default(); 18],
        )
        .unwrap(),
    );
    let inputs = [&common, &common, &common, &mismatched];
    let runtime = runtime(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.run_compiled(&program, &inputs)
    }));
    assert!(result.is_ok(), "runtime validation panicked");
    let error = result.unwrap().unwrap_err();
    assert!(matches!(
        error,
        tenferro_runtime::Error::PlaceholderShapeMismatch { .. }
    ));
}

#[test]
fn installed_module_rejects_backend_links_without_panicking() {
    let shape = [3, 3, 1, 1, 1, 1];
    let traced = traced_inputs(&shape);
    let output =
        wilson_action_traced([&traced[0], &traced[1], &traced[2], &traced[3]], 6.0).unwrap();
    let specs = traced
        .iter()
        .map(|input| (input, DType::C64, &shape[..]))
        .collect::<Vec<_>>();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&output, &specs)
        .unwrap();
    let device = Tensor::C64(
        TypedTensor::from_buffer_col_major(
            shape.to_vec(),
            StorageBuffer::Backend(Box::new(BackendStorageHandle::new_with_len(7, 9))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
                cpu_affinity: None,
            },
        )
        .unwrap(),
    );
    let inputs = [&device, &device, &device, &device];
    let runtime = runtime(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.run_compiled(&program, &inputs)
    }));
    assert!(result.is_ok(), "runtime validation panicked");
    let error = result.unwrap().unwrap_err();
    assert!(matches!(
        &error,
        tenferro_runtime::Error::RuntimeStateSource {
            op: "Runtime::run_compiled",
            phase: tenferro_runtime::ErrorPhase::Execution,
            ..
        }
    ));
    let prepare_error = error
        .source()
        .and_then(|source| source.source())
        .and_then(|source| source.downcast_ref::<tenferro_runtime::PrepareError>())
        .expect("runtime state source should retain PrepareError");
    assert!(matches!(
        prepare_error,
        tenferro_runtime::PrepareError::NoInputIngress {
            input_index: 0,
            placement,
        } if placement.memory_kind == MemoryKind::Device
            && placement.device
                == Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                })
            && placement.cpu_affinity.is_none()
    ));
}

#[test]
fn traced_action_requires_installed_module_and_matches_direct_fixture() {
    let fixture =
        load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2"))
            .unwrap();
    let shape = [3, 3, 2, 2, 2, 2];
    let traced = traced_inputs(&shape);
    let output = wilson_action_traced(
        [&traced[0], &traced[1], &traced[2], &traced[3]],
        fixture.metadata().beta,
    )
    .unwrap();
    let specs = traced
        .iter()
        .map(|input| (input, DType::C64, &shape[..]))
        .collect::<Vec<_>>();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&output, &specs)
        .unwrap();
    let values: [Tensor; 4] = std::array::from_fn(|mu| {
        Tensor::C64(fixture.links().links()[mu].typed().duplicate().unwrap())
    });
    let inputs = values.iter().collect::<Vec<_>>();

    let missing = runtime(false).run_compiled(&program, &inputs).unwrap_err();
    let missing_message = missing.to_string();
    assert!(matches!(
        &missing,
        tenferro_runtime::Error::Unsupported { .. }
            | tenferro_runtime::Error::RuntimeState { .. }
            | tenferro_runtime::Error::RuntimeStateSource { .. }
    ));
    assert!(missing_message.contains("gaugefields.wilson_action.v1"));

    let actual = runtime(true).run_compiled(&program, &inputs).unwrap()[0]
        .as_slice::<f64>()
        .unwrap()[0];
    let expected = wilson_action(fixture.links(), fixture.metadata().beta).unwrap();
    assert!((actual - expected).abs() < 1e-13);
}
