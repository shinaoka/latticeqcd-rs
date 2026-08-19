use crate::{params::ParamsError, report::RunReport};
use std::{fmt, path::PathBuf};

/// A typed failure raised while validating or executing a run.
///
/// [`RunFailure`] adds the execution context and the partial report without
/// changing the underlying error category.
///
/// # Examples
///
/// ```
/// use latticeqcd::RunError;
/// use std::path::PathBuf;
///
/// let error = RunError::OutputExists {
///     path: PathBuf::from("cfg_00000001.ildg"),
/// };
/// assert!(error.to_string().contains("already exists"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error(transparent)]
    Params(#[from] ParamsError),
    #[error(transparent)]
    Gauge(#[from] gaugefields::GaugeError),
    #[error(transparent)]
    Dirac(#[from] dirac_operators::DiracError),
    #[error(transparent)]
    FermionMeasurement(#[from] measurements::fermions::FermionMeasurementError),
    #[error(transparent)]
    Measurement(#[from] measurements::MeasurementError),
    #[error(transparent)]
    Wilson(#[from] wilsonloop::WilsonError),
    #[error("ILDG lattice does not match the configured lattice")]
    InitialLatticeMismatch {
        expected: [usize; 4],
        found: [usize; 4],
    },
    #[error("output directory creation failed")]
    OutputDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("ILDG output operation failed")]
    OutputIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("ILDG output destination already exists")]
    OutputExists { path: PathBuf },
}

/// A run failure together with the work completed before it.
///
/// The partial report remains available through the public `report` field; the
/// boxed report and source keep this error small enough for `Result` without
/// changing partial-report access (`failure.report.completed_updates` still
/// works).
///
/// # Examples
///
/// ```
/// use latticeqcd::{run_lqcd, Params, UpdateParams};
///
/// let mut params = Params::from_toml(include_str!(concat!(
///     env!("CARGO_MANIFEST_DIR"),
///     "/../../examples/phase4.toml",
/// )))?;
/// params.update = UpdateParams::Hmc {
///     step_size: 0.0,
///     steps: 1,
/// };
/// let failure = run_lqcd(&params).expect_err("invalid update");
/// assert_eq!(failure.report.completed_updates, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct RunFailure {
    /// The typed operation failure.
    pub source: Box<RunError>,
    /// Trajectory being processed, if any.
    pub trajectory_id: Option<usize>,
    /// Flow step being processed, if any.
    pub flow_step: Option<usize>,
    /// Scheduled measurement index being processed, if any.
    pub measurement_index: Option<usize>,
    /// The report accumulated before the failure.
    pub report: Box<RunReport>,
}

impl RunFailure {
    pub(crate) fn new(
        source: RunError,
        report: RunReport,
        trajectory_id: Option<usize>,
        flow_step: Option<usize>,
        measurement_index: Option<usize>,
    ) -> Self {
        Self {
            source: Box::new(source),
            trajectory_id,
            flow_step,
            measurement_index,
            report: Box::new(report),
        }
    }
}

impl fmt::Display for RunFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.source)?;
        if let Some(trajectory_id) = self.trajectory_id {
            write!(formatter, " at trajectory {trajectory_id}")?;
        }
        if let Some(flow_step) = self.flow_step {
            write!(formatter, " at flow step {flow_step}")?;
        }
        if let Some(measurement_index) = self.measurement_index {
            write!(formatter, " at measurement {measurement_index}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RunFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}
