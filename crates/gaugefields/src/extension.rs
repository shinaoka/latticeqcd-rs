use crate::{
    kernel::{validate_beta, HostGaugeLinks},
    GaugeError, Mat3,
};
use num_complex::Complex64;
use std::{any::Any, fmt, hash::Hasher, marker::PhantomData, sync::Arc};
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::extension::apply;
use tenferro_runtime::{
    CoreCapabilityKind, DType, EngineId, ErasedExecutionContext, Error as RuntimeError, ErrorPhase,
    ExecutionContextIdentity, ExtensionCacheStore, ExtensionEngine, ExtensionModule,
    ExtensionModuleId, ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest,
    PrepareCapability, PrepareError, PreparedOperation, PreparedOperationBinding,
    PreparedOperationExecutor, PreparedOperationPlan, ProviderContractError,
    Result as RuntimeResult, RuntimeConfigError, SpecializationProjection, UnsupportedReason,
};
use tenferro_tensor::{Tensor, TensorBackend, TensorRead, TypedTensor};

pub(crate) const WILSON_ACTION_FAMILY: &str = "gaugefields.wilson_action.v1";
pub(crate) const WILSON_ACTION_JVP_FAMILY: &str = "gaugefields.wilson_action_jvp.v1";
pub(crate) const WILSON_FORCE_FAMILY: &str = "gaugefields.wilson_force.v1";

fn invalid(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op, "configuration", message)
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
    prepared: &HostGaugeLinks<'_>,
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
    ctx: &mut ExtensionShapeContext<'_>,
    op: &'static str,
) -> tenferro_tensor::Result<Vec<SymDim>> {
    let reference = ctx.input_shape(0)?.to_vec();
    if reference.len() != 6 {
        return Err(invalid(
            op,
            format!("link 0 must have rank 6, found {}", reference.len()),
        ));
    }
    if reference[0]
        .constant_value()
        .is_some_and(|value| value != 3)
        || reference[1]
            .constant_value()
            .is_some_and(|value| value != 3)
    {
        return Err(invalid(op, "link 0 color axes must be [3,3]"));
    }
    for mu in 0..4 {
        if ctx.input_dtype(mu)? != DType::C64 {
            return Err(invalid(op, format!("link {mu} must be C64")));
        }
        let shape = ctx.input_shape(mu)?;
        if shape.len() != 6 {
            return Err(invalid(
                op,
                format!("link {mu} must have rank 6, found {}", shape.len()),
            ));
        }
        if shape[0].constant_value().is_some_and(|value| value != 3)
            || shape[1].constant_value().is_some_and(|value| value != 3)
        {
            return Err(invalid(op, format!("link {mu} color axes must be [3,3]")));
        }
        if shape.iter().zip(&reference).any(|(actual, expected)| {
            matches!(
                (actual.constant_value(), expected.constant_value()),
                (Some(actual), Some(expected)) if actual != expected
            )
        }) {
            return Err(invalid(op, format!("link {mu} lattice shape differs")));
        }
        ctx.require_same_shape(0, mu)?;
    }
    Ok(reference)
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
            other.family_id() == $family && other.as_any().downcast_ref::<Self>() == Some(self)
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

    #[cfg(feature = "autodiff")]
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

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        validate_links(ctx, WILSON_ACTION_FAMILY)?;
        Ok(vec![(DType::F64, vec![])])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WilsonActionJvpOp {
    beta: u64,
    active_dirs: Vec<usize>,
}

impl WilsonActionJvpOp {
    #[cfg(any(feature = "autodiff", test))]
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

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        validate_links(ctx, WILSON_ACTION_JVP_FAMILY)?;
        for (index, &mu) in self.active_dirs.iter().enumerate() {
            let tangent = 4 + index;
            if ctx.input_dtype(tangent)? != DType::C64 {
                return Err(invalid(
                    WILSON_ACTION_JVP_FAMILY,
                    format!("tangent for direction {mu} must be C64"),
                ));
            }
            ctx.require_same_shape(mu, tangent)?;
        }
        Ok(vec![(DType::F64, vec![])])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WilsonForceOp {
    beta: u64,
}

impl WilsonForceOp {
    #[cfg(any(feature = "autodiff", test))]
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

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let links = validate_links(ctx, WILSON_FORCE_FAMILY)?;
        if ctx.input_dtype(4)? != DType::F64 || !ctx.input_shape(4)?.is_empty() {
            return Err(invalid(
                WILSON_FORCE_FAMILY,
                "force seed must be scalar F64",
            ));
        }
        Ok((0..4).map(|_| (DType::C64, links.clone())).collect())
    }
}

fn unsupported_payload(family_id: &'static str) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported(
        "gaugefields.extension",
        format!("extension family {family_id:?} has no gaugefields payload"),
    )
}

fn execute_payload(
    family_id: &'static str,
    op: &dyn ExtensionOp,
    inputs: &[&Tensor],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match family_id {
        WILSON_ACTION_FAMILY => {
            let action = op
                .as_any()
                .downcast_ref::<WilsonActionOp>()
                .ok_or_else(|| unsupported_payload(family_id))?;
            if inputs.len() != 4 || !f64::from_bits(action.beta).is_finite() {
                return Err(invalid(family_id, "invalid action inputs or beta"));
            }
            let links: [&Tensor; 4] = inputs
                .try_into()
                .map_err(|_| invalid(family_id, "expected exactly four link tensors"))?;
            let prepared =
                HostGaugeLinks::from_tensors(links).map_err(|error| abi_error(family_id, error))?;
            let value = -f64::from_bits(action.beta) / 3.0
                * prepared
                    .plaquette_sum()
                    .map_err(|error| abi_error(family_id, error))?;
            Ok(vec![
                TypedTensor::from_vec_col_major(vec![], vec![value])?.into()
            ])
        }
        WILSON_ACTION_JVP_FAMILY => {
            let jvp = op
                .as_any()
                .downcast_ref::<WilsonActionJvpOp>()
                .ok_or_else(|| unsupported_payload(family_id))?;
            if inputs.len() != jvp.input_count() || !f64::from_bits(jvp.beta).is_finite() {
                return Err(invalid(family_id, "invalid JVP inputs or beta"));
            }
            let links: [&Tensor; 4] = inputs[..4]
                .try_into()
                .map_err(|_| invalid(family_id, "expected four links"))?;
            let prepared =
                HostGaugeLinks::from_tensors(links).map_err(|error| abi_error(family_id, error))?;
            let mut value = 0.0;
            for (index, &mu) in jvp.active_dirs.iter().enumerate() {
                let tangent = match inputs[4 + index] {
                    Tensor::C64(tensor) if tensor.shape() == inputs[mu].shape() => {
                        tensor.host_data().map_err(|source| {
                            abi_error(
                                family_id,
                                GaugeError::placement("WilsonActionJvpOp::execute", source),
                            )
                        })?
                    }
                    _ => {
                        return Err(invalid(
                            family_id,
                            format!("tangent for direction {mu} must be matching host C64"),
                        ))
                    }
                };
                for site in 0..prepared.nv() {
                    let gradient = prepared
                        .force_staple(site, mu)
                        .map_err(|error| abi_error(family_id, error))?
                        .scaled(Complex64::new(-f64::from_bits(jvp.beta) / 3.0, 0.0));
                    let delta = Mat3::load(tangent, site * 9)
                        .map_err(|error| abi_error(family_id, error))?;
                    value += gradient.adjoint().mul(delta).trace().re;
                }
            }
            Ok(vec![
                TypedTensor::from_vec_col_major(vec![], vec![value])?.into()
            ])
        }
        WILSON_FORCE_FAMILY => {
            let force = op
                .as_any()
                .downcast_ref::<WilsonForceOp>()
                .ok_or_else(|| unsupported_payload(family_id))?;
            if inputs.len() != 5 || !f64::from_bits(force.beta).is_finite() {
                return Err(invalid(family_id, "invalid force inputs or beta"));
            }
            let links: [&Tensor; 4] = inputs[..4]
                .try_into()
                .map_err(|_| invalid(family_id, "expected four links"))?;
            let seed = match inputs[4] {
                Tensor::F64(tensor) if tensor.shape().is_empty() => tensor
                    .host_data()
                    .map_err(|source| {
                        abi_error(
                            family_id,
                            GaugeError::placement("WilsonForceOp::execute", source),
                        )
                    })?
                    .first()
                    .copied()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| invalid(family_id, "seed must be finite scalar F64"))?,
                _ => return Err(invalid(family_id, "seed must be scalar F64")),
            };
            let prepared =
                HostGaugeLinks::from_tensors(links).map_err(|error| abi_error(family_id, error))?;
            gradient_tensors(&prepared, f64::from_bits(force.beta), seed)
        }
        _ => Err(unsupported_payload(family_id)),
    }
}

struct GaugeReferenceEngine<B: TensorBackend + 'static> {
    family_id: &'static str,
    engine_id: EngineId,
    _backend: PhantomData<fn() -> B>,
}

struct GaugeReferenceModule<B: TensorBackend + 'static> {
    family_id: &'static str,
    module_id: ExtensionModuleId,
    engine_id: EngineId,
    _backend: PhantomData<fn() -> B>,
}

#[derive(Debug)]
struct GaugePlanningConfig {
    family_id: &'static str,
}

struct GaugePreparedOperation<B: TensorBackend + 'static> {
    family_id: &'static str,
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    op: Arc<dyn ExtensionOp>,
    _backend: PhantomData<fn() -> B>,
}

fn module_supports(family_id: &'static str, op: &dyn ExtensionOp) -> bool {
    match family_id {
        WILSON_ACTION_FAMILY => op.as_any().is::<WilsonActionOp>(),
        WILSON_ACTION_JVP_FAMILY => op.as_any().is::<WilsonActionJvpOp>(),
        WILSON_FORCE_FAMILY => op.as_any().is::<WilsonForceOp>(),
        _ => false,
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for GaugeReferenceEngine<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GaugeReferenceEngine")
            .field("family_id", &self.family_id)
            .field("engine_id", &self.engine_id)
            .field("backend_type", &std::any::type_name::<B>())
            .finish()
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for GaugeReferenceModule<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GaugeReferenceModule")
            .field("family_id", &self.family_id)
            .field("module_id", &self.module_id)
            .field("engine_id", &self.engine_id)
            .field("backend_type", &std::any::type_name::<B>())
            .finish()
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for GaugePreparedOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GaugePreparedOperation")
            .field("family_id", &self.family_id)
            .field("binding", &self.binding)
            .field("specialization", &self.specialization)
            .field("backend_type", &std::any::type_name::<B>())
            .finish_non_exhaustive()
    }
}

impl<B: TensorBackend + 'static> ExtensionEngine for GaugeReferenceEngine<B> {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<B>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> std::result::Result<PrepareCapability, PrepareError> {
        if request.operation().family_id() != self.family_id {
            return Err(PrepareError::ProviderContract {
                source: ProviderContractError::WrongOperationFamily {
                    expected: CoreCapabilityKind::Elementwise,
                    operation: self.family_id,
                },
            });
        }
        if !module_supports(self.family_id, request.operation()) {
            return Ok(PrepareCapability::Unsupported(
                UnsupportedReason::Operation {
                    operation: self.family_id,
                },
            ));
        }
        let prepared = Arc::new(GaugePreparedOperation::<B> {
            family_id: self.family_id,
            binding: request.binding().clone(),
            specialization: request.specialization().clone(),
            op: request.operation().clone_arc(),
            _backend: PhantomData,
        });
        Ok(PrepareCapability::Prepared(
            PreparedOperationPlan::executable(prepared.clone(), prepared),
        ))
    }
}

impl ExtensionPlanningConfig for GaugePlanningConfig {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write(self.family_id.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.family_id == self.family_id)
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

impl<B: TensorBackend + 'static> PreparedOperation for GaugePreparedOperation<B> {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

impl<B: TensorBackend + 'static> PreparedOperationExecutor for GaugePreparedOperation<B> {
    fn execute(
        &self,
        context: &mut ErasedExecutionContext<'_>,
        extension_caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> RuntimeResult<Vec<Tensor>> {
        let backend = context
            .downcast_mut::<B>(self.binding.context_identity())
            .map_err(|source| {
                RuntimeError::runtime_state_source(
                    "gaugefields.extension",
                    ErrorPhase::Execution,
                    source,
                )
            })?;
        let mut execution =
            tenferro_runtime::ExtensionExecutionContext::new(backend, extension_caches);
        let materialized_inputs = execution.backend_mut().with_backend_session(|session| {
            inputs
                .iter()
                .cloned()
                .map(|input| session.to_contiguous_read(input))
                .collect::<tenferro_tensor::Result<Vec<_>>>()
        })?;
        let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
        Ok(execute_payload(
            self.family_id,
            self.op.as_ref(),
            &input_refs,
        )?)
    }
}

impl<B: TensorBackend + 'static> ExtensionModule for GaugeReferenceModule<B> {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> std::result::Result<(), tenferro_runtime::ExtensionModuleError> {
        registrar.register_engine(Arc::new(GaugeReferenceEngine::<B> {
            family_id: self.family_id,
            engine_id: self.engine_id.clone(),
            _backend: PhantomData,
        }))?;
        registrar.register_planning_config(
            self.engine_id.clone(),
            Arc::new(GaugePlanningConfig {
                family_id: self.family_id,
            }),
        )?;
        Ok(())
    }
}

fn runtime_module_for_family<B: TensorBackend + 'static>(
    family_id: &'static str,
    engine_id: EngineId,
) -> std::result::Result<Arc<dyn ExtensionModule>, RuntimeConfigError> {
    Ok(Arc::new(GaugeReferenceModule::<B> {
        family_id,
        module_id: ExtensionModuleId::new(format!("{family_id}.module"))?,
        engine_id,
        _backend: PhantomData,
    }))
}

/// Build the three explicitly installable Wilson extension modules.
///
/// Each returned module owns one extension family and the selected runtime
/// engine. Applications install all three modules into their owned `Runtime`.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when a generated module identifier is invalid.
///
/// # Examples
///
/// ```
/// use gaugefields::runtime_modules;
/// use tenferro_cpu::{runtime_engine_id, CpuBackend};
///
/// let modules = runtime_modules::<CpuBackend>(runtime_engine_id()?)?;
/// assert_eq!(modules.len(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn runtime_modules<B: TensorBackend + 'static>(
    engine_id: EngineId,
) -> std::result::Result<Vec<Arc<dyn ExtensionModule>>, RuntimeConfigError> {
    Ok(vec![
        runtime_module_for_family::<B>(WILSON_ACTION_FAMILY, engine_id.clone())?,
        runtime_module_for_family::<B>(WILSON_ACTION_JVP_FAMILY, engine_id.clone())?,
        runtime_module_for_family::<B>(WILSON_FORCE_FAMILY, engine_id)?,
    ])
}

/// Build a traced Wilson action; runtime installation remains explicit.
pub fn wilson_action_traced(
    links: [&tenferro_runtime::TracedTensor; 4],
    beta: f64,
) -> Result<tenferro_runtime::TracedTensor, GaugeError> {
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
