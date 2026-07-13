use std::path::PathBuf;

/// Validation and fixture-loading failures.
#[derive(Debug, thiserror::Error)]
pub enum GaugeError {
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
    #[error("tenferro tensor construction failed: {0}")]
    Tensor(String),
    #[error("beta must be finite, found {found}")]
    NonFiniteBeta { found: f64 },
    #[error("tenferro graph construction failed: {0}")]
    Graph(#[source] tenferro_runtime::Error),
}
