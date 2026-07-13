use crate::extension::{WILSON_ACTION_FAMILY, WILSON_ACTION_JVP_FAMILY};
use computegraph::types::{LocalValueId, ValueKey};
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
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        _tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Err(ADRuleError::unsupported(
            WILSON_ACTION_FAMILY,
            ADRuleKind::Jvp,
        ))
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
