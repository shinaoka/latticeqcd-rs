#![cfg(feature = "autodiff")]

use gaugefields::{
    action_gradient, ad_rules, load_fixture, register_runtime, wilson_action_traced,
};
use std::path::Path;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

fn execute(tensor: &TracedTensor) -> tenferro_tensor::Tensor {
    let program = GraphCompiler::new().compile(tensor).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(register_runtime).unwrap();
    executor.run(&program).unwrap()
}

#[test]
fn reverse_mode_matches_every_direct_gradient_and_arbitrary_seed() {
    let fixture =
        load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2"))
            .unwrap();
    let beta = fixture.metadata().beta;
    let expected = action_gradient(fixture.links(), beta).unwrap();
    let links: [TracedTensor; 4] = std::array::from_fn(|mu| {
        TracedTensor::from_vec_col_major(
            fixture.links().links()[mu].typed().shape().to_vec(),
            fixture.links().links()[mu]
                .typed()
                .host_data()
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    });
    let action = wilson_action_traced([&links[0], &links[1], &links[2], &links[3]], beta).unwrap();
    let ad = AdContext::builder()
        .with_extension_rules(ad_rules().unwrap())
        .build()
        .unwrap();

    for seed in [1.0, -2.5, 0.25] {
        let cotangent = TracedTensor::from_vec_col_major(vec![], vec![seed]).unwrap();
        for mu in 0..4 {
            let traced = ad.vjp(&action, &links[mu], &cotangent).unwrap();
            let force_nodes = traced
                .graph()
                .operations()
                .iter()
                .filter(|node| {
                    matches!(
                        &node.operation,
                        StdTensorOp::Extension(op)
                            if op.family_id() == "gaugefields.wilson_force.v1"
                    )
                })
                .count();
            assert_eq!(force_nodes, 1, "seed={seed} mu={mu}");
            let actual = execute(&traced);
            let actual = actual.as_slice::<num_complex::Complex64>().unwrap();
            let expected = expected[mu].typed().host_data().unwrap();
            let mut max_residual = 0.0_f64;
            for (actual, expected) in actual.iter().zip(expected) {
                max_residual = max_residual.max((*actual - seed * expected).norm());
            }
            assert!(
                max_residual < 1e-13,
                "seed={seed} mu={mu} max_residual={max_residual}"
            );
        }
    }

    let unrelated = TracedTensor::from_vec_col_major(
        vec![3, 3, 2, 2, 2, 2],
        vec![num_complex::Complex64::default(); 144],
    )
    .unwrap();
    assert!(ad.grad_optional(&action, &unrelated).unwrap().is_none());
}
