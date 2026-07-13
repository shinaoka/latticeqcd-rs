use std::{any::Any, hash::Hasher, sync::Arc};
use tenferro_runtime::extension::ExtensionOp;
use tenferro_runtime::{DType, SymDim};

pub(crate) const WILSON_ACTION_FAMILY: &str = "gaugefields.wilson_action.v1";
pub(crate) const WILSON_ACTION_JVP_FAMILY: &str = "gaugefields.wilson_action_jvp.v1";
pub(crate) const WILSON_FORCE_FAMILY: &str = "gaugefields.wilson_force.v1";

fn invalid(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::InvalidConfig {
        op,
        message: message.into(),
    }
}

fn validate_links(
    op: &'static str,
    dtypes: &[DType],
    shapes: &[&[SymDim]],
) -> tenferro_tensor::Result<Vec<SymDim>> {
    if dtypes.len() < 4 || shapes.len() < 4 {
        return Err(invalid(op, "expected four link inputs"));
    }
    let reference = shapes[0];
    for mu in 0..4 {
        if dtypes[mu] != DType::C64 {
            return Err(invalid(
                op,
                format!("link {mu} must be C64, found {:?}", dtypes[mu]),
            ));
        }
        if shapes[mu].len() != 6 {
            return Err(invalid(
                op,
                format!("link {mu} must have rank 6, found {}", shapes[mu].len()),
            ));
        }
        if shapes[mu][0].constant_value() != Some(3) || shapes[mu][1].constant_value() != Some(3) {
            return Err(invalid(op, format!("link {mu} color axes must be [3,3]")));
        }
        if shapes[mu] != reference {
            return Err(invalid(op, format!("link {mu} lattice shape differs")));
        }
    }
    Ok(reference.to_vec())
}

macro_rules! impl_payload_identity {
    ($ty:ty, $family:expr, $extra:expr) => {
        fn family_id(&self) -> &'static str {
            $family
        }
        fn payload_hash(&self, hasher: &mut dyn Hasher) {
            hasher.write_u64(self.beta);
            $extra(self, hasher);
        }
        fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
            other.as_any().downcast_ref::<Self>() == Some(self)
        }
        fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
            Arc::new(self.clone())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WilsonActionOp {
    beta: u64,
}

impl WilsonActionOp {
    pub(crate) fn new(beta: f64) -> Self {
        Self {
            beta: beta.to_bits(),
        }
    }
}

impl ExtensionOp for WilsonActionOp {
    impl_payload_identity!(Self, WILSON_ACTION_FAMILY, |_, _: &mut dyn Hasher| {});
    fn input_count(&self) -> usize {
        4
    }
    fn output_count(&self) -> usize {
        1
    }
    fn infer_output_meta(
        &self,
        dtypes: &[DType],
        shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if dtypes.len() != 4 || shapes.len() != 4 {
            return Err(invalid(WILSON_ACTION_FAMILY, "expected exactly four links"));
        }
        validate_links(WILSON_ACTION_FAMILY, dtypes, shapes)?;
        Ok(vec![(DType::F64, vec![])])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WilsonActionJvpOp {
    beta: u64,
    active_dirs: Vec<usize>,
}

impl WilsonActionJvpOp {
    pub(crate) fn new(beta: f64, active_dirs: Vec<usize>) -> tenferro_tensor::Result<Self> {
        if active_dirs.iter().any(|&mu| mu >= 4)
            || active_dirs.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                WILSON_ACTION_JVP_FAMILY,
                "active directions must be sorted, unique, and in 0..4",
            ));
        }
        Ok(Self {
            beta: beta.to_bits(),
            active_dirs,
        })
    }
    pub(crate) fn active_dirs(&self) -> &[usize] {
        &self.active_dirs
    }
}

impl ExtensionOp for WilsonActionJvpOp {
    impl_payload_identity!(
        Self,
        WILSON_ACTION_JVP_FAMILY,
        |op: &Self, hasher: &mut dyn Hasher| {
            for &mu in &op.active_dirs {
                hasher.write_usize(mu);
            }
        }
    );
    fn input_count(&self) -> usize {
        4 + self.active_dirs.len()
    }
    fn output_count(&self) -> usize {
        1
    }
    fn infer_output_meta(
        &self,
        dtypes: &[DType],
        shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if dtypes.len() != self.input_count() || shapes.len() != self.input_count() {
            return Err(invalid(WILSON_ACTION_JVP_FAMILY, "wrong JVP input arity"));
        }
        validate_links(WILSON_ACTION_JVP_FAMILY, dtypes, shapes)?;
        for (index, &mu) in self.active_dirs.iter().enumerate() {
            let tangent = 4 + index;
            if dtypes[tangent] != DType::C64 || shapes[tangent] != shapes[mu] {
                return Err(invalid(
                    WILSON_ACTION_JVP_FAMILY,
                    format!("tangent for direction {mu} must match its link"),
                ));
            }
        }
        Ok(vec![(DType::F64, vec![])])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WilsonForceOp {
    beta: u64,
}

impl WilsonForceOp {
    pub(crate) fn new(beta: f64) -> Self {
        Self {
            beta: beta.to_bits(),
        }
    }
}

impl ExtensionOp for WilsonForceOp {
    impl_payload_identity!(Self, WILSON_FORCE_FAMILY, |_, _: &mut dyn Hasher| {});
    fn input_count(&self) -> usize {
        5
    }
    fn output_count(&self) -> usize {
        4
    }
    fn infer_output_meta(
        &self,
        dtypes: &[DType],
        shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if dtypes.len() != 5 || shapes.len() != 5 {
            return Err(invalid(
                WILSON_FORCE_FAMILY,
                "expected four links and one seed",
            ));
        }
        let link_shape = validate_links(WILSON_FORCE_FAMILY, dtypes, shapes)?;
        if dtypes[4] != DType::F64 || !shapes[4].is_empty() {
            return Err(invalid(
                WILSON_FORCE_FAMILY,
                "force seed must be scalar F64",
            ));
        }
        Ok(vec![(DType::C64, link_shape); 4])
    }
}

#[cfg(test)]
mod tests;
