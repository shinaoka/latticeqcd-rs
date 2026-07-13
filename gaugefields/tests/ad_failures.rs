#![cfg(feature = "autodiff")]

use gaugefields::{ad_rules, wilson_action_traced};
use num_complex::Complex64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

fn links() -> [TracedTensor; 4] {
    std::array::from_fn(|mu| {
        TracedTensor::from_vec_col_major(
            vec![3, 3, 1, 1, 1, 1],
            (0..9)
                .map(|i| Complex64::new((1 + i + mu) as f64 / 13.0, i as f64 / 19.0))
                .collect(),
        )
        .unwrap()
    })
}

fn context() -> AdContext {
    AdContext::builder()
        .with_extension_rules(ad_rules().unwrap())
        .build()
        .unwrap()
}

#[test]
fn missing_runtime_and_missing_rules_are_distinct_without_panicking() {
    let links = links();
    let action = wilson_action_traced([&links[0], &links[1], &links[2], &links[3]], 5.7).unwrap();
    let tangent = TracedTensor::from_vec_col_major(
        vec![3, 3, 1, 1, 1, 1],
        vec![Complex64::new(0.25, -0.5); 9],
    )
    .unwrap();

    let missing_rules = catch_unwind(AssertUnwindSafe(|| {
        AdContext::builder()
            .build()
            .unwrap()
            .jvp(&action, &links[0], &tangent)
    }));
    assert!(missing_rules.is_ok());
    let missing_rules = missing_rules.unwrap().unwrap_err().to_string();
    assert!(missing_rules.contains("gaugefields.wilson_action.v1"));

    let jvp = context().jvp(&action, &links[0], &tangent).unwrap();
    let program = GraphCompiler::new().compile(&jvp).unwrap();
    let missing_runtime = catch_unwind(AssertUnwindSafe(|| {
        GraphExecutor::new(CpuBackend::new()).run(&program)
    }));
    assert!(missing_runtime.is_ok());
    let missing_runtime = missing_runtime.unwrap().unwrap_err().to_string();
    assert!(
        missing_runtime.contains("missing runtime"),
        "{missing_runtime}"
    );
    assert_ne!(missing_rules, missing_runtime);
}

#[test]
fn malformed_tangent_and_seed_return_errors_without_panicking() {
    let links = links();
    let action = wilson_action_traced([&links[0], &links[1], &links[2], &links[3]], 5.7).unwrap();
    let ad = context();
    let wrong_tangent = TracedTensor::from_vec_col_major(vec![9], vec![1.0_f64; 9]).unwrap();
    let tangent_result = catch_unwind(AssertUnwindSafe(|| {
        ad.jvp(&action, &links[0], &wrong_tangent)
    }));
    assert!(tangent_result.is_ok());
    assert!(tangent_result.unwrap().is_err());

    for seed in [
        TracedTensor::from_vec_col_major(vec![], vec![1.0_f32]).unwrap(),
        TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ] {
        let seed_result = catch_unwind(AssertUnwindSafe(|| ad.vjp(&action, &links[0], &seed)));
        assert!(seed_result.is_ok());
        assert!(seed_result.unwrap().is_err());
    }
}

#[test]
fn higher_order_force_differentiation_is_typed_unsupported_without_panicking() {
    let links = links();
    let action = wilson_action_traced([&links[0], &links[1], &links[2], &links[3]], 5.7).unwrap();
    let ad = context();
    let gradient = ad.grad(&action, &links[0]).unwrap();
    let tangent = TracedTensor::from_vec_col_major(
        vec![3, 3, 1, 1, 1, 1],
        vec![Complex64::new(0.25, -0.5); 9],
    )
    .unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| ad.jvp(&gradient, &links[0], &tangent)));
    assert!(result.is_ok());
    let error = result.unwrap().unwrap_err().to_string();
    assert!(error.contains("gaugefields.wilson_force.v1"), "{error}");
    assert!(error.contains("unsupported"), "{error}");
}
