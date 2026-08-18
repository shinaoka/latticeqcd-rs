use gaugefields::GaugeError;
use std::fmt;

/// Validation and host-evaluation failures for signed Wilson paths and actions.
#[derive(Debug)]
pub enum WilsonError {
    /// A path must contain at least one signed unit step.
    EmptyPath,
    /// A step is not one of `-4..=-1` or `1..=4`.
    InvalidStep { step: i8 },
    /// A checked signed displacement would leave the `isize` range.
    DisplacementOverflow { axis: usize },
    /// An action coefficient is not finite.
    NonFiniteCoefficient { coefficient: f64 },
    /// A loop term was built from an open path.
    OpenPath { displacement: [isize; 4] },
    /// An action must contain at least one term.
    EmptyAction,
    /// A helper axis is outside the public one-based `1..=4` range.
    InvalidAxis { axis: usize },
    /// A helper was given the same axis twice.
    RepeatedAxis { axis: usize },
    /// An evaluation origin is outside the field volume.
    OriginOutOfBounds { origin: usize, volume: usize },
    /// The underlying gauge-field boundary rejected the operation.
    Gauge(GaugeError),
    /// Private compiled metadata disagreed with its validated source path.
    InvalidCompiledMetadata,
    /// A metadata allocation could not be represented or reserved.
    AllocationOverflow,
}

impl From<GaugeError> for WilsonError {
    fn from(source: GaugeError) -> Self {
        Self::Gauge(source)
    }
}

impl fmt::Display for WilsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPath => write!(f, "Wilson paths must not be empty"),
            Self::InvalidStep { step } => write!(f, "Wilson step {step} is outside ±1..±4"),
            Self::DisplacementOverflow { axis } => {
                write!(f, "Wilson displacement overflows isize on axis {axis}")
            }
            Self::NonFiniteCoefficient { coefficient } => {
                write!(f, "loop coefficient must be finite, found {coefficient}")
            }
            Self::OpenPath { displacement } => {
                write!(
                    f,
                    "loop term path is open with displacement {displacement:?}"
                )
            }
            Self::EmptyAction => write!(f, "loop actions must contain at least one term"),
            Self::InvalidAxis { axis } => write!(f, "helper axis {axis} is outside 1..=4"),
            Self::RepeatedAxis { axis } => write!(f, "helper repeats axis {axis}"),
            Self::OriginOutOfBounds { origin, volume } => {
                write!(f, "origin {origin} exceeds lattice volume {volume}")
            }
            Self::Gauge(source) => source.fmt(f),
            Self::InvalidCompiledMetadata => write!(f, "compiled Wilson metadata is invalid"),
            Self::AllocationOverflow => write!(f, "Wilson metadata allocation overflows"),
        }
    }
}

impl std::error::Error for WilsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gauge(source) => Some(source),
            _ => None,
        }
    }
}
