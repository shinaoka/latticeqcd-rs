/// Closed set of solver failures returned by the checked Krylov entrypoints.
///
/// # Examples
///
/// ```
/// use dirac_operators::{DiracError, SolverError};
///
/// let error = DiracError::from(SolverError::Exhaustion);
/// assert!(error.to_string().contains("maximum iterations"));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SolverError {
    #[error("solver tolerance must be finite and positive")]
    InvalidTolerance,
    #[error("solver maximum iterations must be positive")]
    InvalidMaximumIterations,
    #[error("solver encountered a non-finite intermediate")]
    NonFiniteIntermediate,
    #[error("solver denominator is zero or near-zero")]
    Breakdown,
    #[error("BiCGStab shadow residual remained singular after restart")]
    SingularShadowRestart,
    #[error("solver stagnated")]
    Stagnation,
    #[error("solver exhausted its maximum iterations")]
    Exhaustion,
    #[error("recursive convergence did not pass a fresh true-residual check")]
    TrueResidualMismatch,
}

/// Failures returned by validated fermion fields and Wilson operators.
#[derive(Debug, thiserror::Error)]
pub enum DiracError {
    #[error("fermion tensor must have rank {expected}, found {found}")]
    Rank { expected: usize, found: usize },
    #[error("fermion tensor shape is {found:?}, expected {expected:?}")]
    Shape {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("fermion component count must be exactly 1 or 4, found {found}")]
    InvalidComponents { found: usize },
    #[error("fermion field {operand} has {found} components, expected {expected}")]
    ComponentsMismatch {
        operand: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("fermion field {operand} has lattice {found:?}, expected {expected:?}")]
    LatticeMismatch {
        operand: &'static str,
        expected: gaugefields::LatticeShape4,
        found: gaugefields::LatticeShape4,
    },
    #[error("fermion color index {color} is outside 0..3")]
    ColorOutOfBounds { color: usize },
    #[error("fermion component index {component} is outside 0..{components}")]
    ComponentOutOfBounds { component: usize, components: usize },
    #[error("fermion site index {site} exceeds lattice volume {volume}")]
    SiteOutOfBounds { site: usize, volume: usize },
    #[error("fermion field storage is inconsistent with its validated metadata")]
    StorageInvariant,
    #[error("fermion tensor requires host placement; found {found}")]
    Placement { found: String },
    #[error("fermion tensor contains a non-finite value at physical offset {offset}")]
    NonFinite { offset: usize },
    #[error("fermion field allocation size overflows the supported address range")]
    AllocationOverflow,
    #[error("fermion tensor has unsupported dtype {found}")]
    DType { found: String },
    #[error("{operation} does not permit the same field as input and output")]
    AliasedFields { operation: &'static str },
    #[error("fermion boundary sign on axis {direction} must be +1 or -1, found {found}")]
    InvalidBoundary { direction: usize, found: i8 },
    #[error("kappa must be finite, found {found}")]
    NonFiniteKappa { found: f64 },
    #[error("kappa must be positive, found {found}")]
    NonPositiveKappa { found: f64 },
    #[error("Wilson r must be finite, found {found}")]
    NonFiniteWilsonR { found: f64 },
    #[error("only Wilson r=1 is supported, found {found}")]
    UnsupportedWilsonR { found: f64 },
    #[error("the Wilson result contains a non-finite value")]
    NumericalRange,
    #[error("gauge-field operation failed: {0}")]
    Gauge(#[from] gaugefields::GaugeError),
    #[error("tensor operation failed: {0}")]
    Tensor(String),
    #[error("solver operation failed: {0}")]
    Solver(#[from] SolverError),
}
