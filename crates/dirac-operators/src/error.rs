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
}
