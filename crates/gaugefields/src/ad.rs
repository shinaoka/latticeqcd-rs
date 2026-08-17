use crate::extension::{WilsonActionJvpOp, WilsonActionOp, WilsonForceOp, WILSON_ACTION_FAMILY};
use std::sync::Arc;
use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
};
use tenferro_runtime::program::SemanticProgramBuilder;

fn invariant(role: SemanticAdRuleRole, message: impl Into<String>) -> SemanticAdError {
    SemanticAdError::Invariant {
        family_id: WILSON_ACTION_FAMILY,
        role,
        message: message.into(),
    }
}

fn action_payload(
    op: &dyn tenferro_ops::ext_op::ExtensionOp,
    role: SemanticAdRuleRole,
) -> Result<&WilsonActionOp, SemanticAdError> {
    op.as_any()
        .downcast_ref::<WilsonActionOp>()
        .ok_or_else(|| SemanticAdError::Unsupported {
            family_id: WILSON_ACTION_FAMILY,
            role,
            message: "Wilson action semantic AD received an incompatible payload".into(),
        })
}

fn validate_linearize_request(
    request: SemanticLinearizeRequest<'_>,
) -> Result<(), SemanticAdError> {
    if request.primal_inputs().len() != 4
        || request.primal_outputs().len() != 1
        || request.tangent_inputs().len() != 4
        || request.active_outputs().len() != 1
    {
        return Err(invariant(
            SemanticAdRuleRole::Linearize,
            "expected four primal inputs, one output, and four tangent slots",
        ));
    }
    Ok(())
}

fn validate_transpose_request(
    request: SemanticLinearTransposeRequest<'_>,
) -> Result<(), SemanticAdError> {
    if request.primal_inputs().len() != 4
        || request.primal_outputs().len() != 1
        || request.cotangent_outputs().len() != 1
        || request.active_inputs().len() != 4
        || !request.residuals().is_empty()
    {
        return Err(invariant(
            SemanticAdRuleRole::LinearTranspose,
            "expected four primal inputs, one output, one cotangent, and four active links",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct WilsonActionLinearize;

impl SemanticLinearizeRule for WilsonActionLinearize {
    fn family_id(&self) -> &'static str {
        WILSON_ACTION_FAMILY
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        validate_linearize_request(request)?;
        let action = action_payload(request.op(), SemanticAdRuleRole::Linearize)?;
        if !request.active_outputs()[0] {
            return Ok(SemanticLinearizeResult::new([AdValue::Absent], []));
        }
        let active_dirs = request
            .tangent_inputs()
            .iter()
            .enumerate()
            .filter_map(|(mu, tangent)| matches!(tangent, AdValue::Value(_)).then_some(mu))
            .collect::<Vec<_>>();
        if active_dirs.is_empty() {
            return Ok(SemanticLinearizeResult::new([AdValue::Absent], []));
        }
        let mut inputs = request.primal_inputs().to_vec();
        inputs.extend(active_dirs.iter().filter_map(|&mu| {
            request
                .tangent_inputs()
                .get(mu)
                .copied()
                .and_then(AdValue::value)
        }));
        let jvp = WilsonActionJvpOp::new(action.beta(), active_dirs).map_err(|error| {
            SemanticAdError::Unsupported {
                family_id: WILSON_ACTION_FAMILY,
                role: SemanticAdRuleRole::Linearize,
                message: error.to_string(),
            }
        })?;
        let outputs = builder.add_extension(Arc::new(jvp), &inputs)?;
        let output = outputs.first().copied().ok_or_else(|| {
            invariant(
                SemanticAdRuleRole::Linearize,
                "Wilson action JVP emitted no output",
            )
        })?;
        Ok(SemanticLinearizeResult::new([AdValue::Value(output)], []))
    }
}

#[derive(Debug)]
struct WilsonActionLinearTranspose;

impl SemanticLinearTransposeRule for WilsonActionLinearTranspose {
    fn family_id(&self) -> &'static str {
        WILSON_ACTION_FAMILY
    }

    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::input(0)
            .with_input(1)
            .with_input(2)
            .with_input(3)
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        validate_transpose_request(request)?;
        let action = action_payload(request.op(), SemanticAdRuleRole::LinearTranspose)?;
        let Some(cotangent) = request.cotangent_outputs()[0].value() else {
            return Ok(vec![AdValue::Absent; 4].into_boxed_slice());
        };
        let force =
            WilsonForceOp::new(action.beta()).map_err(|error| SemanticAdError::Unsupported {
                family_id: WILSON_ACTION_FAMILY,
                role: SemanticAdRuleRole::LinearTranspose,
                message: error.to_string(),
            })?;
        let mut inputs = request.primal_inputs().to_vec();
        inputs.push(cotangent);
        let outputs = builder.add_extension(Arc::new(force), &inputs)?;
        if outputs.len() != 4 {
            return Err(invariant(
                SemanticAdRuleRole::LinearTranspose,
                "Wilson force emitted an invalid output count",
            ));
        }
        Ok(outputs
            .iter()
            .enumerate()
            .map(|(mu, &output)| {
                if request.active_inputs()[mu] {
                    AdValue::Value(output)
                } else {
                    AdValue::Absent
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }
}

/// Return a fresh semantic rule set for first-order Wilson action autodiff.
pub fn ad_rules() -> Result<SemanticExtensionRuleSet, SemanticExtensionRegistryError> {
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(WilsonActionLinearize))?
        .with_linear_transpose(Arc::new(WilsonActionLinearTranspose))
}

#[cfg(test)]
mod tests;
