use std::path::PathBuf;

/// Gauge-field, HMC, heatbath, fixture, runtime, and RNG-state failures.
#[derive(Debug, thiserror::Error)]
pub enum GaugeError {
    #[error("reproducible RNG state must not be all zero")]
    InvalidRngState,
    #[error("lattice extent on axis {axis} must be positive")]
    InvalidExtent { axis: usize },
    #[error("lattice volume overflows usize")]
    VolumeOverflow,
    #[error("gauge tensor allocation size overflows the supported address range")]
    AllocationOverflow,
    #[error("matrix block at offset {offset} exceeds buffer length {len}")]
    MatrixBlockOutOfBounds { offset: usize, len: usize },
    #[error("coordinate {coordinate} on axis {axis} exceeds extent {extent}")]
    CoordinateOutOfBounds {
        axis: usize,
        coordinate: usize,
        extent: usize,
    },
    #[error("site {site} exceeds lattice volume {volume}")]
    SiteOutOfBounds { site: usize, volume: usize },
    #[error("direction {direction} is outside 0..4")]
    InvalidDirection { direction: usize },
    #[error("expected C64 tensor, found {found}")]
    DType { found: String },
    #[error("expected rank {expected}, found {found}")]
    Rank { expected: usize, found: usize },
    #[error("expected shape {expected:?}, found {found:?}")]
    Shape {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("link direction {mu} has a different lattice shape")]
    InconsistentMu { mu: usize },
    #[error("SU(3) kernel requires NC=3, found {found}")]
    UnsupportedNc { found: usize },
    #[error("fixture direction {mu} is not Fortran ordered")]
    NpyOrder { mu: usize },
    #[error("fixture direction {mu} has unsupported dtype: {detail}")]
    NpyDType { mu: usize, detail: String },
    #[error("fixture direction {mu} expected rank 6, found {found}")]
    NpyRank { mu: usize, found: usize },
    #[error("fixture metadata disagrees with data: {detail}")]
    MetadataMismatch { detail: String },
    #[error("invalid fixture metadata: {0}")]
    Metadata(#[source] serde_json::Error),
    #[error("could not read fixture file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid NPY in direction {mu}: {detail}")]
    Npy { mu: usize, detail: String },
    #[error("ILDG I/O failed for {path}: {source}")]
    IldgIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid ILDG/LIME structure: {detail}")]
    IldgFormat { detail: &'static str },
    #[error("invalid ILDG XML metadata: {detail}")]
    IldgXml { detail: &'static str },
    #[error("invalid ILDG binary payload: {detail}")]
    IldgPayload { detail: &'static str },
    #[error(
        "non-finite ILDG component at direction {direction}, site {site}, component {component}"
    )]
    IldgNonFinite {
        direction: usize,
        site: usize,
        component: usize,
    },
    #[error("tenferro tensor construction failed: {0}")]
    Tensor(String),
    #[error("{operation} requires host placement: {source}")]
    Placement {
        operation: &'static str,
        #[source]
        source: tenferro_tensor::Error,
    },
    #[error("beta must be finite, found {found}")]
    NonFiniteBeta { found: f64 },
    #[error("stout rho must be finite, found {found}")]
    NonFiniteRho { found: f64 },
    #[error("heatbath beta must be positive, found {found}")]
    NonPositiveHeatbathBeta { found: f64 },
    #[error("heatbath requires at least one rejection attempt")]
    ZeroHeatbathAttempts,
    #[error("heatbath requires an even extent on axis {axis}, found {extent}")]
    OddHeatbathExtent { axis: usize, extent: usize },
    #[error(
        "heatbath staple is singular at direction {direction}, site {site}, subgroup {subgroup}"
    )]
    SingularHeatbathStaple {
        direction: usize,
        site: usize,
        subgroup: usize,
    },
    #[error("heatbath exceeded finite numerical range during {stage}")]
    HeatbathNumericalRange { stage: &'static str },
    #[error("heatbath rejection limit exhausted after {max_attempts} attempts")]
    HeatbathRejectionLimit { max_attempts: usize },
    #[error("HMC step size must be finite, found {found}")]
    NonFiniteStepSize { found: f64 },
    #[error("HMC step size must be positive, found {found}")]
    NonPositiveStepSize { found: f64 },
    #[error("HMC requires at least one leapfrog step")]
    ZeroHmcSteps,
    #[error("HMC momentum at direction {mu}, component {component} is non-finite")]
    NonFiniteMomentum { mu: usize, component: usize },
    #[error("HMC kinetic-energy square sum exceeded finite range")]
    KineticNumericalRange,
    #[error("HMC Hamiltonian is non-finite")]
    NonFiniteHamiltonian,
    #[error("HMC Hamiltonian difference is non-finite")]
    NonFiniteHamiltonianDelta,
    #[error("{operation} received non-finite SU(3) input at component {component}")]
    NonFiniteSu3Input {
        operation: &'static str,
        component: usize,
    },
    #[error("{operation} exceeded finite SU(3) numerical range during {stage}")]
    Su3NumericalRange {
        operation: &'static str,
        stage: &'static str,
    },
    #[error("SU(3) normalization is singular at row {row}")]
    SingularSu3Normalization { row: usize },
    #[error("{operation} failed in tenferro evolution: {source}")]
    Evolution {
        operation: &'static str,
        #[source]
        source: tenferro_tensor::Error,
    },
    #[error("tenferro graph construction failed: {0}")]
    Graph(#[source] tenferro_runtime::Error),
}

impl GaugeError {
    pub(crate) fn placement(operation: &'static str, source: tenferro_tensor::Error) -> Self {
        Self::Placement { operation, source }
    }
}
