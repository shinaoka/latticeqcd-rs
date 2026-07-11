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
        Ok(vec![tangent_in[0]])
    }
}

/// Builds the Phase 0 rule set for explicit attachment to an `AdContext`.
pub fn extension_rules() -> Result<ExtensionRuleSet, tenferro_ad::extension::ExtensionRegistryError>
{
    ExtensionRuleSet::new().with_linearize(Arc::new(IdentityLinearize))
}
