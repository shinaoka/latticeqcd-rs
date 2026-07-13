use crate::{
    kernel::{validate_beta, PreparedGaugeField},
    GaugeError, Mat3,
};
use num_complex::Complex64;
use std::{any::Any, hash::Hasher, sync::Arc};
use tenferro_runtime::extension::{
    apply, ExtensionExecutor, ExtensionOp, ExtensionRuntimeRegistryError, HostReference,
    HostReferenceRuntime,
};
use tenferro_runtime::{DType, SymDim, TracedTensor};
use tenferro_tensor::{Tensor, TensorBackend, TypedTensor};

pub(crate) const WILSON_ACTION_FAMILY: &str = "gaugefields.wilson_action.v1";
pub(crate) const WILSON_ACTION_JVP_FAMILY: &str = "gaugefields.wilson_action_jvp.v1";
pub(crate) const WILSON_FORCE_FAMILY: &str = "gaugefields.wilson_force.v1";

fn invalid(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::InvalidConfig {
        op,
        message: message.into(),
    }
}

fn abi_error(op: &'static str, error: GaugeError) -> tenferro_tensor::Error {
    match error {
        GaugeError::Placement { source, .. } => source,
        domain => invalid(op, domain.to_string()),
    }
}

fn checked_output_count(nv: usize) -> tenferro_tensor::Result<usize> {
    let count = 9usize
        .checked_mul(nv)
        .ok_or_else(|| invalid("gaugefields extension", "output element count overflow"))?;
    let bytes = count
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or_else(|| invalid("gaugefields extension", "output byte count overflow"))?;
    if bytes > isize::MAX as usize {
        return Err(invalid(
            "gaugefields extension",
            "output exceeds supported address range",
        ));
    }
    Ok(count)
}

fn gradient_tensors(
    prepared: &PreparedGaugeField<'_>,
    beta: f64,
    seed: f64,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let count = checked_output_count(prepared.nv())?;
    let mut outputs = Vec::with_capacity(4);
    for mu in 0..4 {
        let mut data = Vec::with_capacity(count);
        for site in 0..prepared.nv() {
            let gradient = prepared
                .force_staple(site, mu)
                .map_err(|error| abi_error(WILSON_FORCE_FAMILY, error))?
                .scaled(Complex64::new(-beta * seed / 3.0, 0.0));
            data.extend_from_slice(gradient.as_array());
        }
        let [nx, ny, nz, nt] = prepared.lattice().extents();
        outputs.push(TypedTensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], data)?.into());
    }
    Ok(outputs)
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
        if shapes[mu][0]
            .constant_value()
            .is_some_and(|value| value != 3)
            || shapes[mu][1]
                .constant_value()
                .is_some_and(|value| value != 3)
        {
            return Err(invalid(op, format!("link {mu} color axes must be [3,3]")));
        }
        if shapes[mu].iter().zip(reference).any(|(actual, expected)| {
            matches!(
                (actual.constant_value(), expected.constant_value()),
                (Some(actual), Some(expected)) if actual != expected
            )
        }) {
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
    pub(crate) fn new(beta: f64) -> tenferro_tensor::Result<Self> {
        validate_beta(beta).map_err(|error| abi_error(WILSON_ACTION_FAMILY, error))?;
        Ok(Self {
            beta: beta.to_bits(),
        })
    }

    pub(crate) fn beta(&self) -> f64 {
        f64::from_bits(self.beta)
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
    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for WilsonActionOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        if inputs.len() != 4 || !f64::from_bits(self.beta).is_finite() {
            return Err(invalid(
                WILSON_ACTION_FAMILY,
                "invalid action inputs or beta",
            ));
        }
        let prepared =
            PreparedGaugeField::from_tensors(inputs.try_into().map_err(|_| {
                invalid(WILSON_ACTION_FAMILY, "expected exactly four link tensors")
            })?)
            .map_err(|error| abi_error(WILSON_ACTION_FAMILY, error))?;
        let value = -f64::from_bits(self.beta) / 3.0
            * prepared
                .plaquette_sum()
                .map_err(|error| abi_error(WILSON_ACTION_FAMILY, error))?;
        Ok(vec![
            TypedTensor::from_vec_col_major(vec![], vec![value])?.into()
        ])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WilsonActionJvpOp {
    beta: u64,
    active_dirs: Vec<usize>,
}

impl WilsonActionJvpOp {
    pub(crate) fn new(beta: f64, active_dirs: Vec<usize>) -> tenferro_tensor::Result<Self> {
        validate_beta(beta).map_err(|error| abi_error(WILSON_ACTION_JVP_FAMILY, error))?;
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
            if dtypes[tangent] != DType::C64
                || shapes[tangent].len() != shapes[mu].len()
                || shapes[tangent]
                    .iter()
                    .zip(shapes[mu])
                    .any(|(actual, expected)| {
                        matches!(
                            (actual.constant_value(), expected.constant_value()),
                            (Some(actual), Some(expected)) if actual != expected
                        )
                    })
            {
                return Err(invalid(
                    WILSON_ACTION_JVP_FAMILY,
                    format!("tangent for direction {mu} must match its link"),
                ));
            }
        }
        Ok(vec![(DType::F64, vec![])])
    }
    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for WilsonActionJvpOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        if inputs.len() != self.input_count() || !f64::from_bits(self.beta).is_finite() {
            return Err(invalid(
                WILSON_ACTION_JVP_FAMILY,
                "invalid JVP inputs or beta",
            ));
        }
        let links: [&Tensor; 4] = inputs[..4]
            .try_into()
            .map_err(|_| invalid(WILSON_ACTION_JVP_FAMILY, "expected four links"))?;
        let prepared = PreparedGaugeField::from_tensors(links)
            .map_err(|error| abi_error(WILSON_ACTION_JVP_FAMILY, error))?;
        let mut value = 0.0;
        for (index, &mu) in self.active_dirs.iter().enumerate() {
            let tangent = match inputs[4 + index] {
                Tensor::C64(tensor) if tensor.shape() == inputs[mu].shape() => {
                    tensor.host_data().map_err(|source| {
                        abi_error(
                            WILSON_ACTION_JVP_FAMILY,
                            GaugeError::placement("WilsonActionJvpOp::execute", source),
                        )
                    })?
                }
                _ => {
                    return Err(invalid(
                        WILSON_ACTION_JVP_FAMILY,
                        format!("tangent for direction {mu} must be matching host C64"),
                    ))
                }
            };
            for site in 0..prepared.nv() {
                let gradient = prepared
                    .force_staple(site, mu)
                    .map_err(|error| abi_error(WILSON_ACTION_JVP_FAMILY, error))?
                    .scaled(Complex64::new(-f64::from_bits(self.beta) / 3.0, 0.0));
                let delta = Mat3::load(tangent, site * 9)
                    .map_err(|error| abi_error(WILSON_ACTION_JVP_FAMILY, error))?;
                value += gradient.adjoint().mul(delta).trace().re;
            }
        }
        Ok(vec![
            TypedTensor::from_vec_col_major(vec![], vec![value])?.into()
        ])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WilsonForceOp {
    beta: u64,
}

impl WilsonForceOp {
    pub(crate) fn new(beta: f64) -> tenferro_tensor::Result<Self> {
        validate_beta(beta).map_err(|error| abi_error(WILSON_FORCE_FAMILY, error))?;
        Ok(Self {
            beta: beta.to_bits(),
        })
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
        validate_links(WILSON_FORCE_FAMILY, dtypes, shapes)?;
        if dtypes[4] != DType::F64 || !shapes[4].is_empty() {
            return Err(invalid(
                WILSON_FORCE_FAMILY,
                "force seed must be scalar F64",
            ));
        }
        Ok((0..4).map(|mu| (DType::C64, shapes[mu].to_vec())).collect())
    }
    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for WilsonForceOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        if inputs.len() != 5 || !f64::from_bits(self.beta).is_finite() {
            return Err(invalid(WILSON_FORCE_FAMILY, "invalid force inputs or beta"));
        }
        let links: [&Tensor; 4] = inputs[..4]
            .try_into()
            .map_err(|_| invalid(WILSON_FORCE_FAMILY, "expected four links"))?;
        let seed = match inputs[4] {
            Tensor::F64(tensor) if tensor.shape().is_empty() => tensor
                .host_data()
                .map_err(|source| {
                    abi_error(
                        WILSON_FORCE_FAMILY,
                        GaugeError::placement("WilsonForceOp::execute", source),
                    )
                })?
                .first()
                .copied()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(WILSON_FORCE_FAMILY, "seed must be finite scalar F64"))?,
            _ => return Err(invalid(WILSON_FORCE_FAMILY, "seed must be scalar F64")),
        };
        let prepared = PreparedGaugeField::from_tensors(links)
            .map_err(|error| abi_error(WILSON_FORCE_FAMILY, error))?;
        gradient_tensors(&prepared, f64::from_bits(self.beta), seed)
    }
}

/// Register all gaugefields host-reference extension families on an executor.
pub fn register_runtime<B: TensorBackend + 'static>(
    executor: &mut ExtensionExecutor<B>,
) -> Result<(), ExtensionRuntimeRegistryError> {
    let _jvp_constructor = WilsonActionJvpOp::new;
    let _force_constructor = WilsonForceOp::new;
    for family in [
        WILSON_ACTION_FAMILY,
        WILSON_ACTION_JVP_FAMILY,
        WILSON_FORCE_FAMILY,
    ] {
        executor
            .registry_mut()
            .register(Arc::new(HostReferenceRuntime::<B>::new(family)))?;
    }
    Ok(())
}

/// Build a traced Wilson action; runtime registration remains explicit.
pub fn wilson_action_traced(
    links: [&TracedTensor; 4],
    beta: f64,
) -> Result<TracedTensor, GaugeError> {
    validate_beta(beta)?;
    let op = WilsonActionOp::new(beta).map_err(|error| GaugeError::Tensor(error.to_string()))?;
    let outputs = apply(Arc::new(op), &links).map_err(GaugeError::Graph)?;
    if outputs.len() != 1 {
        return Err(GaugeError::Tensor(
            "Wilson action must produce one output".into(),
        ));
    }
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| GaugeError::Tensor("Wilson action output is missing".into()))
}

#[cfg(test)]
mod tests;
