#![cfg(feature = "autodiff")]

use computegraph::OperationRole;
use gaugefields::{ad_rules, wilson_action_traced};
use num_complex::Complex64;
use tenferro_ad::AdContext;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::TracedTensor;

fn link(value: f64) -> TracedTensor {
    TracedTensor::from_vec_col_major(
        vec![3, 3, 1, 1, 1, 1],
        (0..9)
            .map(|i| Complex64::new(value + i as f64 / 17.0, i as f64 / 23.0))
            .collect(),
    )
    .unwrap()
}

#[test]
fn active_direction_payload_omits_inactive_tangents() {
    let source = link(1.0);
    let fixed_one = link(2.0);
    let fixed_three = link(3.0);
    let tangent = link(0.25);
    let action = wilson_action_traced([&source, &fixed_one, &source, &fixed_three], 5.7).unwrap();
    let ad = AdContext::builder()
        .with_extension_rules(ad_rules().unwrap())
        .build()
        .unwrap();
    let jvp = ad.jvp(&action, &source, &tangent).unwrap();

    let matching = jvp
        .graph()
        .operations()
        .iter()
        .filter_map(|node| match &node.operation {
            StdTensorOp::Extension(op) if op.family_id() == "gaugefields.wilson_action_jvp.v1" => {
                Some((op.input_count(), node.inputs.len(), &node.role))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);
    assert_eq!(matching[0].0, 6);
    assert_eq!(matching[0].1, 6);
    assert_eq!(
        matching[0].2,
        &OperationRole::Linearized {
            active_mask: vec![false, false, false, false, true, true]
        }
    );
}
