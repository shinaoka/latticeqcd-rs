use gaugefields::{cold_su3, runtime_modules, wilson_action, wilson_action_traced, LatticeShape4};
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::{DType, GraphCompiler, Runtime, Tensor, TracedTensor};

#[test]
fn current_runtime_module_executes_traced_action() {
    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let shape = [3, 3, 1, 1, 1, 1];
    let traced: [TracedTensor; 4] =
        std::array::from_fn(|_| TracedTensor::input_concrete_shape(DType::C64, &shape).unwrap());
    let action =
        wilson_action_traced([&traced[0], &traced[1], &traced[2], &traced[3]], 6.0).unwrap();
    let specs = traced
        .iter()
        .map(|input| (input, DType::C64, &shape[..]))
        .collect::<Vec<_>>();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&action, &specs)
        .unwrap();
    let values: [Tensor; 4] =
        std::array::from_fn(|mu| Tensor::C64(links.links()[mu].typed().duplicate().unwrap()));
    let inputs = values.iter().collect::<Vec<_>>();

    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&backend).unwrap())
        .unwrap();
    for module in runtime_modules::<CpuBackend>(runtime_engine_id().unwrap()).unwrap() {
        builder.install_extension_module(module).unwrap();
    }
    let runtime = builder.build().unwrap();

    let traced_value = runtime.run_compiled(&program, &inputs).unwrap();
    assert_eq!(traced_value.len(), 1);
    let expected = wilson_action(&links, 6.0).unwrap();
    assert!((traced_value[0].as_slice::<f64>().unwrap()[0] - expected).abs() < 1e-13);
}
