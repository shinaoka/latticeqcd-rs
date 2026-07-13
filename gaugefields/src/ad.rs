use crate::extension::{
    WilsonActionJvpOp, WilsonActionOp, WILSON_ACTION_FAMILY, WILSON_ACTION_JVP_FAMILY,
};
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use std::sync::Arc;
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{
    ExtensionLinearTransposeRule, ExtensionLinearizeRule, ExtensionOp, ExtensionRegistryError,
    ExtensionRuleSet, ShapeGuardContext,
};
use tidu::{ADRuleError, ADRuleKind, ADRuleResult, PrimitiveTransposeInput};

#[derive(Debug)]
struct WilsonActionLinearize;

impl ExtensionLinearizeRule for WilsonActionLinearize {
    fn family_id(&self) -> &'static str {
        WILSON_ACTION_FAMILY
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let action = op
            .as_any()
            .downcast_ref::<WilsonActionOp>()
            .ok_or_else(|| {
                ADRuleError::invalid_input(
                    WILSON_ACTION_FAMILY,
                    ADRuleKind::Jvp,
                    "action payload downcast failed",
                )
            })?;
        if primal_in.len() != 4 || primal_out.len() != 1 || tangent_in.len() != 4 {
            return Err(ADRuleError::invalid_input(
                WILSON_ACTION_FAMILY,
                ADRuleKind::Jvp,
                "expected four primal inputs, one output, and four tangent slots",
            ));
        }
        let active_dirs = tangent_in
            .iter()
            .enumerate()
            .filter_map(|(mu, tangent)| tangent.is_some().then_some(mu))
            .collect::<Vec<_>>();
        if active_dirs.is_empty() {
            return Ok(vec![None]);
        }
        let mut inputs = Vec::with_capacity(4 + active_dirs.len());
        inputs.extend(primal_in.iter().cloned().map(ValueRef::External));
        for &mu in &active_dirs {
            let tangent = tangent_in[mu].ok_or_else(|| {
                ADRuleError::invalid_input(
                    WILSON_ACTION_FAMILY,
                    ADRuleKind::Jvp,
                    "active tangent slot disappeared during graph construction",
                )
            })?;
            inputs.push(ValueRef::Local(tangent));
        }
        let jvp = WilsonActionJvpOp::new(action.beta(), active_dirs).map_err(|error| {
            ADRuleError::invalid_input(WILSON_ACTION_FAMILY, ADRuleKind::Jvp, error.to_string())
        })?;
        let active_mask = std::iter::repeat_n(false, 4)
            .chain(std::iter::repeat_n(true, inputs.len() - 4))
            .collect();
        let outputs = builder.add_operation(
            StdTensorOp::Extension(Arc::new(jvp)),
            inputs,
            OperationRole::Linearized { active_mask },
        );
        outputs
            .first()
            .copied()
            .map(|output| vec![Some(output)])
            .ok_or_else(|| {
                ADRuleError::invalid_input(
                    WILSON_ACTION_FAMILY,
                    ADRuleKind::Jvp,
                    "JVP operation emitted no output",
                )
            })
    }
}

#[derive(Debug)]
struct WilsonJvpTranspose;

impl ExtensionLinearTransposeRule for WilsonJvpTranspose {
    fn family_id(&self) -> &'static str {
        WILSON_ACTION_JVP_FAMILY
    }

    fn linear_transpose(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        _cotangent_out: &[Option<LocalValueId>],
        _inputs: &[PrimitiveTransposeInput<StdTensorOp>],
        _active_mask: &[bool],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Err(ADRuleError::unsupported(
            WILSON_ACTION_JVP_FAMILY,
            ADRuleKind::Transpose,
        ))
    }
}

/// Return a fresh explicit rule set for first-order Wilson action autodiff.
pub fn ad_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    ExtensionRuleSet::new()
        .with_linearize(Arc::new(WilsonActionLinearize))?
        .with_linear_transpose(Arc::new(WilsonJvpTranspose))
}
