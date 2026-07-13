use super::*;
use computegraph::{GraphOperation, ValueKey};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tenferro_ops::input_key::TensorInputKey;

#[derive(Default)]
struct CaptureBuilder {
    emissions: usize,
}

impl PrimitiveRuleBuilder for CaptureBuilder {
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        _inputs: Vec<ValueRef<StdTensorOp>>,
        _role: OperationRole,
    ) -> Vec<LocalValueId> {
        self.emissions += 1;
        (0..operation.output_count()).collect()
    }
}

fn key(id: u64) -> ValueKey<StdTensorOp> {
    ValueKey::Input(TensorInputKey::User { id })
}

fn transpose_inputs(count: usize) -> Vec<PrimitiveTransposeInput<StdTensorOp>> {
    (0..count)
        .map(|index| PrimitiveTransposeInput::Residual(key(index as u64)))
        .collect()
}

fn assert_invalid_without_panic(call: impl FnOnce() -> ADRuleResult<Vec<Option<LocalValueId>>>) {
    let result = catch_unwind(AssertUnwindSafe(call));
    assert!(result.is_ok(), "AD rule panicked");
    assert!(matches!(
        result.unwrap(),
        Err(ADRuleError::InvalidInput { .. })
    ));
}

#[test]
fn transpose_rejects_every_nonexact_activity_mask_before_emission() {
    let op = WilsonActionJvpOp::new(5.7, vec![0, 2]).unwrap();
    let inputs = transpose_inputs(6);
    for mask in [
        vec![false, false, false, false, true],
        vec![false, false, false, false, true, true, true],
        vec![true, false, false, false, true, true],
        vec![false, false, false, false, false, true],
        vec![false, false, false, false, true, false],
    ] {
        let mut builder = CaptureBuilder::default();
        assert_invalid_without_panic(|| {
            WilsonJvpTranspose.linear_transpose(
                &op,
                &mut builder,
                &[Some(7)],
                &inputs,
                &mask,
                &mut ShapeGuardContext::default(),
            )
        });
        assert_eq!(builder.emissions, 0, "mask={mask:?}");
    }
}

#[test]
fn private_rules_reject_malformed_contracts_without_panicking_or_emitting() {
    let action = WilsonActionOp::new(5.7).unwrap();
    let force = WilsonForceOp::new(5.7).unwrap();
    let primals = (0..4).map(key).collect::<Vec<_>>();
    let outputs = vec![key(10)];
    let tangents = vec![Some(20), None, Some(22), None];
    let mut linearize_cases: Vec<(
        &dyn ExtensionOp,
        &[ValueKey<StdTensorOp>],
        &[ValueKey<StdTensorOp>],
        &[Option<LocalValueId>],
    )> = vec![
        (&force, &primals, &outputs, &tangents),
        (&action, &primals[..3], &outputs, &tangents),
        (&action, &primals, &[], &tangents),
        (&action, &primals, &outputs, &tangents[..3]),
    ];
    for (op, primal_in, primal_out, tangent_in) in linearize_cases.drain(..) {
        let mut builder = CaptureBuilder::default();
        assert_invalid_without_panic(|| {
            WilsonActionLinearize.linearize(
                op,
                &mut builder,
                primal_in,
                primal_out,
                tangent_in,
                &mut ShapeGuardContext::default(),
            )
        });
        assert_eq!(builder.emissions, 0);
    }

    let jvp = WilsonActionJvpOp::new(5.7, vec![0, 2]).unwrap();
    let exact_mask = [false, false, false, false, true, true];
    let inputs = transpose_inputs(6);
    for cotangents in [vec![], vec![Some(7), Some(8)]] {
        let mut builder = CaptureBuilder::default();
        assert_invalid_without_panic(|| {
            WilsonJvpTranspose.linear_transpose(
                &jvp,
                &mut builder,
                &cotangents,
                &inputs,
                &exact_mask,
                &mut ShapeGuardContext::default(),
            )
        });
        assert_eq!(builder.emissions, 0);
    }

    let mut builder = CaptureBuilder::default();
    assert_invalid_without_panic(|| {
        WilsonJvpTranspose.linear_transpose(
            &action,
            &mut builder,
            &[Some(7)],
            &inputs,
            &exact_mask,
            &mut ShapeGuardContext::default(),
        )
    });
    assert_eq!(builder.emissions, 0);

    let mut short_builder = CaptureBuilder::default();
    assert_invalid_without_panic(|| {
        WilsonJvpTranspose.linear_transpose(
            &jvp,
            &mut short_builder,
            &[Some(7)],
            &inputs[..5],
            &exact_mask[..5],
            &mut ShapeGuardContext::default(),
        )
    });
    assert_eq!(short_builder.emissions, 0);

    let mut missing_primal = inputs.clone();
    missing_primal[0] = PrimitiveTransposeInput::Linear {
        key: key(30),
        primal: None,
    };
    let mut retained_builder = CaptureBuilder::default();
    assert_invalid_without_panic(|| {
        WilsonJvpTranspose.linear_transpose(
            &jvp,
            &mut retained_builder,
            &[Some(7)],
            &missing_primal,
            &exact_mask,
            &mut ShapeGuardContext::default(),
        )
    });
    assert_eq!(retained_builder.emissions, 0);
}
