use gaugefields::{load_fixture, register_runtime, wilson_action, wilson_action_traced};
use std::path::Path;
use std::sync::Arc;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::{
    Buffer, BufferHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, TypedTensor,
};

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
    let bindings = [
        (&traced[0], &common),
        (&traced[1], &common),
        (&traced[2], &common),
        (&traced[3], &mismatched),
    ];
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(register_runtime).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        executor.run_with_inputs(&program, &bindings)
    }));
    assert!(result.is_ok(), "runtime validation panicked");
    let error = result.unwrap().unwrap_err().to_string();
    assert!(error.contains("binding shape mismatch"), "{error}");
}

#[test]
fn registered_host_runtime_rejects_backend_links_without_panicking() {
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
            Buffer::Backend(Arc::new(BufferHandle::new_with_len(7, 9))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
            },
        )
        .unwrap(),
    );
    let bindings = traced
        .iter()
        .map(|input| (input, &device))
        .collect::<Vec<_>>();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(register_runtime).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        executor.run_with_inputs(&program, &bindings)
    }));
    let error = result.unwrap().unwrap_err().to_string();
    assert!(error.contains("download"), "{error}");
}

#[test]
fn traced_action_requires_registration_and_matches_direct_fixture() {
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
    let values: [Tensor; 4] =
        std::array::from_fn(|mu| Tensor::C64(fixture.links().links()[mu].typed().clone()));
    let bindings = traced.iter().zip(&values).collect::<Vec<_>>();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let missing = executor.run_with_inputs(&program, &bindings).unwrap_err();
    assert!(missing.to_string().contains("missing runtime"));

    executor.register_extension(register_runtime).unwrap();
    executor.register_extension(register_runtime).unwrap();
    let actual = executor
        .run_with_inputs(&program, &bindings)
        .unwrap()
        .as_slice::<f64>()
        .unwrap()[0];
    let expected = wilson_action(fixture.links(), fixture.metadata().beta).unwrap();
    assert!((actual - expected).abs() < 1e-13);
}
