use dirac_operators::SolverReport;
use num_complex::Complex64;
use std::path::PathBuf;

/// The dynamical update family used for one trajectory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateKind {
    QuenchedHmc,
    WilsonHmc,
    StaggeredHmc,
    Heatbath,
}

/// Per-trajectory update diagnostics.
///
/// Each variant carries its exact dispatched [`UpdateKind`]; the frontend does
/// not substitute a quenched or metadata-only fallback for a requested action.
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateOutcome {
    Hmc {
        kind: UpdateKind,
        accepted: bool,
        delta_h: f64,
        acceptance_probability: f64,
    },
    Heatbath {
        kind: UpdateKind,
        updated_links: usize,
        su2_attempts: usize,
    },
}

/// One completed update and its outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct UpdateRecord {
    pub trajectory_id: usize,
    pub outcome: UpdateOutcome,
}

/// The public name of a scheduled scalar or fermion measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasurementKind {
    Plaquette,
    PolyakovLoop,
    CloverTopologicalCharge,
    PionWilson,
    PionStaggered,
    ChiralStaggered,
}

/// Measurement payload stored in the ordered run report.
#[derive(Clone, Debug, PartialEq)]
pub enum MeasurementValue {
    Scalar(f64),
    PolyakovLoop(Complex64),
    Pion {
        values: Vec<f64>,
        solver_reports: Vec<SolverReport>,
    },
    Chiral {
        value: f64,
        source_values: Vec<f64>,
        solver_reports: Vec<SolverReport>,
    },
}

/// One bare or flow measurement record.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementRecord {
    pub trajectory_id: usize,
    pub measurement_index: usize,
    pub kind: MeasurementKind,
    pub value: MeasurementValue,
}

/// Measurements collected from one flow trajectory and step.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowRecord {
    pub trajectory_id: usize,
    pub step: usize,
    pub measurements: Vec<MeasurementRecord>,
}

/// Completed work and diagnostics returned by [`crate::run_lqcd`].
///
/// `lattice` and `initial_rng_state` are the validated inputs used to start the
/// run. A failure before those fields validate carries zero sentinels and no
/// completed records. The report intentionally does not expose a final RNG state
/// or claim that it can resume a run. Records preserve update, measurement, flow,
/// and output order, including completed rejected HMC updates.
///
/// # Examples
///
/// ```
/// use latticeqcd::{run_lqcd, Params};
///
/// let params = Params::from_toml(include_str!(concat!(
///     env!("CARGO_MANIFEST_DIR"),
///     "/../../examples/phase4.toml",
/// )))?;
/// let report = run_lqcd(&params)?;
/// assert_eq!(report.lattice, [2, 2, 2, 2]);
/// assert_eq!(report.initial_rng_state, [1, 2, 3, 4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RunReport {
    /// Number of updates requested by the parameter document.
    pub requested_updates: usize,
    /// Validated four-dimensional lattice extents.
    pub lattice: [usize; 4],
    /// Initial four-word RNG state; no final-state export is provided.
    pub initial_rng_state: [u64; 4],
    /// Number of updates that completed, including rejected HMC proposals.
    pub completed_updates: usize,
    /// Number of accepted HMC updates.
    pub accepted_updates: usize,
    /// Number of rejected HMC updates.
    pub rejected_updates: usize,
    /// Ordered per-trajectory update records.
    pub updates: Vec<UpdateRecord>,
    /// Ordered bare measurement records.
    pub measurements: Vec<MeasurementRecord>,
    /// Ordered gradient-flow records.
    pub flows: Vec<FlowRecord>,
    /// ILDG destinations successfully published in order.
    pub published_paths: Vec<PathBuf>,
}

impl RunReport {
    pub(crate) fn new(
        requested_updates: usize,
        lattice: [usize; 4],
        initial_rng_state: [u64; 4],
    ) -> Self {
        Self {
            requested_updates,
            lattice,
            initial_rng_state,
            completed_updates: 0,
            accepted_updates: 0,
            rejected_updates: 0,
            updates: Vec::new(),
            measurements: Vec::new(),
            flows: Vec::new(),
            published_paths: Vec::new(),
        }
    }

    /// Return completed per-trajectory update records.
    pub fn update_outcomes(&self) -> &[UpdateRecord] {
        &self.updates
    }
}
