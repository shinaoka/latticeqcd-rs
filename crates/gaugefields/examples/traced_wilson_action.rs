use gaugefields::{cold_su3, register_runtime, wilson_action, wilson_action_traced, LatticeShape4};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let links = cold_su3(LatticeShape4::new([2, 2, 2, 2])?)?;
    let shape = [3, 3, 2, 2, 2, 2];
    let traced: [TracedTensor; 4] =
        std::array::from_fn(|_| TracedTensor::input_concrete_shape(DType::C64, &shape).unwrap());
    let action = wilson_action_traced([&traced[0], &traced[1], &traced[2], &traced[3]], 6.0)?;
    let specs = traced
        .iter()
        .map(|input| (input, DType::C64, &shape[..]))
        .collect::<Vec<_>>();
    let program = GraphCompiler::new().compile_with_input_specs(&action, &specs)?;
    let values: [Tensor; 4] =
        std::array::from_fn(|mu| Tensor::C64(links.links()[mu].typed().clone()));
    let bindings = traced.iter().zip(&values).collect::<Vec<_>>();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(register_runtime)?;
    let traced_value = executor
        .run_with_inputs(&program, &bindings)?
        .as_slice::<f64>()?[0];
    let direct = wilson_action(&links, 6.0)?;
    let residual = (traced_value - direct).abs();
    println!("direct={direct} traced={traced_value} residual={residual}");
    assert!(residual < 1e-13);
    Ok(())
}
