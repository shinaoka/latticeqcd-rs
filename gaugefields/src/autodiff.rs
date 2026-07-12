//! Minimal extension carrier and AD-rule registration spike.
//!
//! This proves the pinned tenferro extension surface can carry a gaugefields
//! operation and register an AD rule. It intentionally defines identity
//! metadata only; no gauge kernel execution is provided in Phase 0.

use computegraph::types::{LocalValueId, ValueKey};
use std::{any::Any, hash::Hasher, sync::Arc};
use tenferro_ad::extension::{ExtensionLinearizeRule, ExtensionOp, ExtensionRuleSet};
use tenferro_ops::{
    ad::PrimitiveRuleBuilder, std_tensor_op::StdTensorOp, ShapeGuardContext, SymDim,
};
use tenferro_tensor::DType;
use tidu::ADRuleResult;

/// Compile-spike carrier for a future gaugefields extension family.
#[derive(Clone, Debug, Default)]
pub struct GaugeIdentityOp;

impl GaugeIdentityOp {
    pub const fn new() -> Self {
        Self
    }
}

impl ExtensionOp for GaugeIdentityOp {
    fn family_id(&self) -> &'static str {
        "gaugefields.identity.v1"
    }
    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        1
    }
    fn output_count(&self) -> usize {
        1
    }
    fn infer_output_meta(
        &self,
        dtypes: &[DType],
        shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if dtypes.len() != 1 || shapes.len() != 1 {
            return Err(tenferro_tensor::Error::InvalidConfig {
                op: "gaugefields.identity.v1",
                message: format!(
                    "expected one dtype and one shape, got {} and {}",
                    dtypes.len(),
                    shapes.len()
                ),
            });
        }
        Ok(vec![(dtypes[0], shapes[0].to_vec())])
    }
}

#[derive(Debug)]
struct IdentityLinearize;

impl ExtensionLinearizeRule for IdentityLinearize {
    fn family_id(&self) -> &'static str {
        "gaugefields.identity.v1"
    }
    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        if _primal_in.len() != 1 || _primal_out.len() != 1 || tangent_in.len() != 1 {
            return Err(tidu::ADRuleError::invalid_input(
                self.family_id(),
                tidu::ADRuleKind::Jvp,
                format!(
                    "expected one primal input, output, and tangent; got {}, {}, and {}",
                    _primal_in.len(),
                    _primal_out.len(),
                    tangent_in.len()
                ),
            ));
        }
        Ok(vec![tangent_in[0]])
    }
}

/// Builds the Phase 0 rule set for explicit attachment to an `AdContext`.
pub fn extension_rules() -> Result<ExtensionRuleSet, tenferro_ad::extension::ExtensionRegistryError>
{
    ExtensionRuleSet::new().with_linearize(Arc::new(IdentityLinearize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use computegraph::types::{OperationRole, ValueRef};
    use tenferro_ad::extension::ExtensionLinearizeRule;
    use tenferro_ops::ad::PrimitiveRuleBuilder;

    struct NoopBuilder;
    impl PrimitiveRuleBuilder for NoopBuilder {
        fn add_operation(
            &mut self,
            _: StdTensorOp,
            _: Vec<ValueRef<StdTensorOp>>,
            _: OperationRole,
        ) -> Vec<LocalValueId> {
            unreachable!()
        }
    }

    #[test]
    fn extension_metadata_rejects_wrong_arity() {
        let err = GaugeIdentityOp::new()
            .infer_output_meta(&[], &[])
            .unwrap_err();
        assert!(matches!(err, tenferro_tensor::Error::InvalidConfig { .. }));
    }

    #[test]
    fn linearize_rule_rejects_wrong_arity() {
        let mut builder = NoopBuilder;
        let mut ctx = ShapeGuardContext::default();
        let err = IdentityLinearize
            .linearize(&GaugeIdentityOp, &mut builder, &[], &[], &[], &mut ctx)
            .unwrap_err();
        assert!(matches!(err, tidu::ADRuleError::InvalidInput { .. }));
    }
}
