use dirac_operators::{DiracError, FermionBoundary, SolverParams};
use gaugefields::LatticeShape4;
use serde::Deserialize;
use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

/// A strictly validated Phase 4 parameter document.
///
/// Every nested table rejects unknown fields. Parsing and validation are pure:
/// they do not create directories, read ILDG input, construct a backend, or
/// consume RNG words. [`crate::run_lqcd`] performs those effects only after this
/// value has been validated.
///
/// # Examples
///
/// ```
/// use latticeqcd::Params;
///
/// let params = Params::from_toml(include_str!(concat!(
///     env!("CARGO_MANIFEST_DIR"),
///     "/../../examples/phase4.toml",
/// )))?;
/// assert_eq!(params.schema_version, 1);
/// # Ok::<(), latticeqcd::ParamsError>(())
/// ```
#[derive(Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub schema_version: u32,
    pub physical: PhysicalParams,
    pub initial: InitialParams,
    pub fermions: FermionParams,
    pub update: UpdateParams,
    pub rng: RngParams,
    pub control: ControlParams,
    #[serde(default)]
    pub measurements: Vec<MeasurementParams>,
    pub gradient_flow: Option<GradientFlowParams>,
    pub output: Option<OutputParams>,
}

impl fmt::Debug for Params {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Params")
            .field("schema_version", &self.schema_version)
            .field("physical", &self.physical)
            .field("initial", &self.initial)
            .field("fermions", &self.fermions)
            .field("update", &self.update)
            .field("rng", &self.rng)
            .field("control", &self.control)
            .field("measurements", &self.measurements)
            .field("gradient_flow", &self.gradient_flow)
            .field("output", &self.output)
            .finish()
    }
}

impl Params {
    /// Parse and validate one strict TOML document.
    ///
    /// The checked repository example is parsed by this same entry point:
    ///
    /// ```
    /// use latticeqcd::Params;
    ///
    /// let params = Params::from_toml(include_str!(concat!(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "/../../examples/phase4.toml",
    /// )))?;
    /// assert_eq!(params.physical.lattice, [2, 2, 2, 2]);
    /// # Ok::<(), latticeqcd::ParamsError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ParamsError::Toml`] for syntax/shape failures, or typed
    /// schema, numerical, compatibility, scheduling, RNG, and output errors
    /// such as [`ParamsError::InvalidHmcStepSize`]. No execution-side effect
    /// is performed on either success or failure.
    pub fn from_toml(source: &str) -> Result<Self, ParamsError> {
        let params: Self = toml::from_str(source).map_err(ParamsError::Toml)?;
        params.validate()?;
        Ok(params)
    }

    /// Read, parse, and validate one strict TOML document from `path`.
    ///
    /// Reading the file is the only I/O performed by this method; validation
    /// still does not initialize a run or consume RNG words.
    ///
    /// # Errors
    ///
    /// Returns [`ParamsError::Io`] for file access, or the same parse and
    /// validation variants as [`Self::from_toml`].
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ParamsError> {
        let path = path.as_ref().to_path_buf();
        let source =
            fs::read_to_string(&path).map_err(|source| ParamsError::Io { path, source })?;
        Self::from_toml(&source)
    }

    /// Validate a value assembled by a caller rather than by [`Self::from_toml`].
    ///
    /// This checks the full schema and all supported parameter combinations,
    /// including update dispatch, schedules, and output safety.
    ///
    /// # Errors
    ///
    /// Returns the typed schema, numerical, compatibility, scheduling, or
    /// output-safety variant describing the first invalid value.
    pub fn validate(&self) -> Result<(), ParamsError> {
        if self.schema_version != 1 {
            return Err(ParamsError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        validate_physical(&self.physical)?;
        validate_control(&self.control)?;
        let _ = self.rng.state()?;
        validate_fermions(&self.fermions)?;
        validate_update(&self.physical, &self.fermions, &self.update)?;
        validate_measurements(&self.measurements)?;
        if let Some(flow) = &self.gradient_flow {
            validate_flow(flow)?;
        }
        if let Some(output) = &self.output {
            validate_output(output)?;
        }
        if let InitialParams::Ildg { path } = &self.initial {
            if path.as_os_str().is_empty() {
                return Err(ParamsError::EmptyIldgPath);
            }
        }
        Ok(())
    }

    /// Return the validated lattice shape.
    ///
    /// # Errors
    ///
    /// Returns the first validation error, or [`ParamsError::Gauge`] if the
    /// validated extents cannot construct a lower-level lattice shape.
    pub fn lattice_shape(&self) -> Result<LatticeShape4, ParamsError> {
        self.validate()?;
        LatticeShape4::new(self.physical.lattice).map_err(ParamsError::Gauge)
    }
}

/// Physical lattice and gauge coupling parameters.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PhysicalParams {
    pub lattice: [usize; 4],
    pub beta: f64,
}

/// The initial gauge configuration source.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InitialParams {
    Cold {},
    Ildg { path: PathBuf },
}

/// Sea-fermion action selected by the run.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FermionParams {
    Quenched {},
    WilsonNf2 {
        kappa: f64,
        boundary: [i8; 4],
        solver: SolverConfig,
    },
    StaggeredNf2 {
        mass: f64,
        boundary: [i8; 4],
        lambda_low: f64,
        lambda_high: f64,
        solver: SolverConfig,
    },
}

/// Gauge update selected by the run.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum UpdateParams {
    Hmc { step_size: f64, steps: usize },
    Heatbath { max_attempts: usize },
}

/// Solver stopping parameters in the TOML schema.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SolverConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
}

/// The one caller-owned reproducible RNG state.
#[derive(Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RngParams {
    pub state_hex: [String; 4],
}

impl fmt::Debug for RngParams {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RngParams")
            .field("state_hex", &"<redacted>")
            .finish()
    }
}

impl RngParams {
    /// Decode the four exact-width hexadecimal words.
    ///
    /// # Errors
    ///
    /// Returns [`ParamsError::InvalidRngWord`] for malformed words and
    /// [`ParamsError::ZeroRngState`] for an all-zero state.
    pub fn state(&self) -> Result<[u64; 4], ParamsError> {
        let mut state = [0_u64; 4];
        for (index, word) in self.state_hex.iter().enumerate() {
            if word.len() != 16
                || !word.is_ascii()
                || !word.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(ParamsError::InvalidRngWord { index });
            }
            state[index] =
                u64::from_str_radix(word, 16).map_err(|_| ParamsError::InvalidRngWord { index })?;
        }
        if state == [0; 4] {
            return Err(ParamsError::ZeroRngState);
        }
        Ok(state)
    }
}

/// Trajectory numbering and measurement-control parameters.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlParams {
    pub first_trajectory: usize,
    pub trajectories: usize,
    pub thermalization: usize,
    pub measure_initial: bool,
}

/// One scheduled measurement.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeasurementParams {
    Plaquette {
        every: usize,
    },
    PolyakovLoop {
        every: usize,
    },
    CloverTopologicalCharge {
        every: usize,
    },
    PionWilson {
        every: usize,
        kappa: f64,
        boundary: [i8; 4],
        solver: SolverConfig,
    },
    PionStaggered {
        every: usize,
        mass: f64,
        boundary: [i8; 4],
        solver: SolverConfig,
    },
    ChiralStaggered {
        every: usize,
        mass: f64,
        boundary: [i8; 4],
        solver: SolverConfig,
        sources: usize,
        flavors: usize,
    },
}

/// Optional Wilson gradient-flow schedule.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GradientFlowParams {
    pub every_trajectories: usize,
    pub step_size: f64,
    pub steps: usize,
    pub measure_every_steps: usize,
    pub measurements: Vec<FlowMeasurement>,
}

/// Gluonic quantity recorded along gradient flow.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FlowMeasurement {
    Plaquette,
    PolyakovLoop,
    CloverTopologicalCharge,
}

/// Optional no-clobber ILDG output schedule.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutputParams {
    pub directory: PathBuf,
    pub prefix: String,
    pub every: usize,
}

/// Typed parse and validation failures raised before a run can start.
///
/// These errors are intentionally separate from execution failures so callers
/// can reject a document without creating a gauge field, backend, output
/// directory, or RNG.
///
/// # Examples
///
/// ```
/// use latticeqcd::{Params, ParamsError};
///
/// let error = Params::from_toml("schema_version = 1")
///     .expect_err("required tables are absent");
/// assert!(matches!(error, ParamsError::Toml(_)));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum ParamsError {
    #[error("parameter file could not be read")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("parameter TOML is invalid")]
    Toml(#[source] toml::de::Error),
    #[error("schema_version must be 1")]
    UnsupportedSchemaVersion { found: u32 },
    #[error("lattice extents must be positive")]
    InvalidLattice,
    #[error("lattice volume overflows usize")]
    LatticeVolumeOverflow,
    #[error("beta must be finite and positive")]
    InvalidBeta,
    #[error("first_trajectory must be at least one")]
    InvalidFirstTrajectory,
    #[error("trajectories must be positive")]
    InvalidTrajectoryCount,
    #[error("thermalization cannot exceed trajectories")]
    ThermalizationExceedsTrajectories,
    #[error("trajectory identifiers overflow usize")]
    TrajectoryIdOverflow,
    #[error("initial measurement requires zero thermalization")]
    InvalidInitialMeasurement,
    #[error("RNG state word is not exactly sixteen ASCII hexadecimal digits")]
    InvalidRngWord { index: usize },
    #[error("RNG state must not be all zero")]
    ZeroRngState,
    #[error("invalid fermion parameters")]
    Fermion(#[source] DiracError),
    #[error("HMC step_size must be finite and positive")]
    InvalidHmcStepSize,
    #[error("HMC steps must be positive")]
    InvalidHmcSteps,
    #[error("unsupported fermion/update combination")]
    UnsupportedCombination,
    #[error("heatbath requires even lattice extents")]
    OddHeatbathExtent { axis: usize, extent: usize },
    #[error("measurement interval must be positive")]
    InvalidMeasurementInterval,
    #[error("chiral measurement sources must be positive")]
    InvalidChiralSources,
    #[error("chiral measurement flavors must be positive and exactly representable as f64")]
    InvalidChiralFlavors,
    #[error("duplicate measurement with identical valence parameters")]
    DuplicateMeasurement,
    #[error("flow interval must be positive")]
    InvalidFlowInterval,
    #[error("flow step size must be finite and positive")]
    InvalidFlowStepSize,
    #[error("flow steps must be positive")]
    InvalidFlowSteps,
    #[error("flow measurement interval must be positive")]
    InvalidFlowMeasurementInterval,
    #[error("flow steps must be divisible by measure_every_steps")]
    MisalignedFlowMeasurements,
    #[error("flow measurements must be non-empty")]
    EmptyFlowMeasurements,
    #[error("duplicate flow measurement")]
    DuplicateFlowMeasurement,
    #[error("output directory must not be empty")]
    EmptyOutputDirectory,
    #[error("output interval must be positive")]
    InvalidOutputInterval,
    #[error("output prefix is not a safe file prefix")]
    UnsafeOutputPrefix,
    #[error("ILDG input path must not be empty")]
    EmptyIldgPath,
    #[error(transparent)]
    Gauge(#[from] gaugefields::GaugeError),
}

fn validate_physical(physical: &PhysicalParams) -> Result<(), ParamsError> {
    if physical.lattice.contains(&0) {
        return Err(ParamsError::InvalidLattice);
    }
    physical
        .lattice
        .iter()
        .try_fold(1_usize, |volume, &extent| {
            volume
                .checked_mul(extent)
                .ok_or(ParamsError::LatticeVolumeOverflow)
        })?;
    if !physical.beta.is_finite() || physical.beta <= 0.0 {
        return Err(ParamsError::InvalidBeta);
    }
    Ok(())
}

fn validate_control(control: &ControlParams) -> Result<(), ParamsError> {
    if control.first_trajectory == 0 {
        return Err(ParamsError::InvalidFirstTrajectory);
    }
    if control.trajectories == 0 {
        return Err(ParamsError::InvalidTrajectoryCount);
    }
    if control.thermalization > control.trajectories {
        return Err(ParamsError::ThermalizationExceedsTrajectories);
    }
    control
        .first_trajectory
        .checked_add(control.trajectories - 1)
        .ok_or(ParamsError::TrajectoryIdOverflow)?;
    if control.measure_initial && control.thermalization != 0 {
        return Err(ParamsError::InvalidInitialMeasurement);
    }
    Ok(())
}

fn validate_fermions(fermions: &FermionParams) -> Result<(), ParamsError> {
    match fermions {
        FermionParams::Quenched {} => Ok(()),
        FermionParams::WilsonNf2 {
            kappa,
            boundary,
            solver,
        } => {
            let boundary = FermionBoundary::new(*boundary).map_err(ParamsError::Fermion)?;
            let solver = solver.to_lower()?;
            dirac_operators::WilsonFermiAction::new(*kappa, boundary, solver)
                .map(|_| ())
                .map_err(ParamsError::Fermion)
        }
        FermionParams::StaggeredNf2 {
            mass,
            boundary,
            lambda_low,
            lambda_high,
            solver,
        } => {
            let boundary = FermionBoundary::new(*boundary).map_err(ParamsError::Fermion)?;
            let solver = solver.to_lower()?;
            dirac_operators::StaggeredFermiAction::new(
                *mass,
                boundary,
                *lambda_low,
                *lambda_high,
                solver,
            )
            .map(|_| ())
            .map_err(ParamsError::Fermion)
        }
    }
}

fn validate_update(
    physical: &PhysicalParams,
    fermions: &FermionParams,
    update: &UpdateParams,
) -> Result<(), ParamsError> {
    match update {
        UpdateParams::Hmc { step_size, steps } => {
            if !step_size.is_finite() || *step_size <= 0.0 {
                return Err(ParamsError::InvalidHmcStepSize);
            }
            if *steps == 0 {
                return Err(ParamsError::InvalidHmcSteps);
            }
            match fermions {
                FermionParams::Quenched {} => {
                    gaugefields::HmcParams::new(physical.beta, *step_size, *steps)
                        .map(|_| ())
                        .map_err(ParamsError::Gauge)
                }
                FermionParams::WilsonNf2 {
                    kappa,
                    boundary,
                    solver,
                } => {
                    let boundary = FermionBoundary::new(*boundary).map_err(ParamsError::Fermion)?;
                    let solver = solver.to_lower()?;
                    dirac_operators::WilsonHmcParams::new(
                        physical.beta,
                        *kappa,
                        *step_size,
                        *steps,
                        boundary,
                        solver,
                    )
                    .map(|_| ())
                    .map_err(ParamsError::Fermion)
                }
                FermionParams::StaggeredNf2 {
                    mass,
                    boundary,
                    lambda_low,
                    lambda_high,
                    solver,
                } => {
                    let boundary = FermionBoundary::new(*boundary).map_err(ParamsError::Fermion)?;
                    let solver = solver.to_lower()?;
                    dirac_operators::StaggeredHmcParams::new(
                        physical.beta,
                        *mass,
                        *step_size,
                        *steps,
                        boundary,
                        *lambda_low,
                        *lambda_high,
                        solver,
                    )
                    .map(|_| ())
                    .map_err(ParamsError::Fermion)
                }
            }
        }
        UpdateParams::Heatbath { max_attempts } => {
            if !matches!(fermions, FermionParams::Quenched {}) {
                return Err(ParamsError::UnsupportedCombination);
            }
            if let Some((axis, &extent)) = physical
                .lattice
                .iter()
                .enumerate()
                .find(|(_, extent)| **extent % 2 != 0)
            {
                return Err(ParamsError::OddHeatbathExtent { axis, extent });
            }
            gaugefields::HeatbathParams::new(physical.beta, *max_attempts)
                .map(|_| ())
                .map_err(ParamsError::Gauge)
        }
    }
}

fn validate_measurements(measurements: &[MeasurementParams]) -> Result<(), ParamsError> {
    for (index, measurement) in measurements.iter().enumerate() {
        if measurement_every(measurement) == 0 {
            return Err(ParamsError::InvalidMeasurementInterval);
        }
        validate_measurement_values(measurement)?;
        if measurements[..index]
            .iter()
            .any(|previous| same_measurement(previous, measurement))
        {
            return Err(ParamsError::DuplicateMeasurement);
        }
    }
    Ok(())
}

fn validate_measurement_values(measurement: &MeasurementParams) -> Result<(), ParamsError> {
    match measurement {
        MeasurementParams::Plaquette { .. }
        | MeasurementParams::PolyakovLoop { .. }
        | MeasurementParams::CloverTopologicalCharge { .. } => Ok(()),
        MeasurementParams::PionWilson {
            kappa,
            boundary,
            solver,
            ..
        } => {
            let boundary = FermionBoundary::new(*boundary).map_err(ParamsError::Fermion)?;
            let solver = solver.to_lower()?;
            dirac_operators::WilsonFermiAction::new(*kappa, boundary, solver)
                .map(|_| ())
                .map_err(ParamsError::Fermion)
        }
        MeasurementParams::PionStaggered {
            mass,
            boundary,
            solver,
            ..
        } => {
            let boundary = FermionBoundary::new(*boundary).map_err(ParamsError::Fermion)?;
            let solver = solver.to_lower()?;
            dirac_operators::StaggeredFermiAction::new(*mass, boundary, 0.0004, 64.0, solver)
                .map(|_| ())
                .map_err(ParamsError::Fermion)
        }
        MeasurementParams::ChiralStaggered {
            mass,
            boundary,
            solver,
            sources,
            flavors,
            ..
        } => {
            if *sources == 0 {
                return Err(ParamsError::InvalidChiralSources);
            }
            validate_chiral_flavors(*flavors)?;
            let boundary = FermionBoundary::new(*boundary).map_err(ParamsError::Fermion)?;
            let solver = solver.to_lower()?;
            dirac_operators::StaggeredFermiAction::new(*mass, boundary, 0.0004, 64.0, solver)
                .map(|_| ())
                .map_err(ParamsError::Fermion)
        }
    }
}

fn validate_chiral_flavors(flavors: usize) -> Result<(), ParamsError> {
    if flavors == 0 || (flavors as u64) > (1_u64 << 53) {
        return Err(ParamsError::InvalidChiralFlavors);
    }
    Ok(())
}

fn measurement_every(measurement: &MeasurementParams) -> usize {
    match measurement {
        MeasurementParams::Plaquette { every }
        | MeasurementParams::PolyakovLoop { every }
        | MeasurementParams::CloverTopologicalCharge { every }
        | MeasurementParams::PionWilson { every, .. }
        | MeasurementParams::PionStaggered { every, .. }
        | MeasurementParams::ChiralStaggered { every, .. } => *every,
    }
}

fn same_measurement(left: &MeasurementParams, right: &MeasurementParams) -> bool {
    match (left, right) {
        (MeasurementParams::Plaquette { .. }, MeasurementParams::Plaquette { .. })
        | (MeasurementParams::PolyakovLoop { .. }, MeasurementParams::PolyakovLoop { .. })
        | (
            MeasurementParams::CloverTopologicalCharge { .. },
            MeasurementParams::CloverTopologicalCharge { .. },
        ) => true,
        (
            MeasurementParams::PionWilson {
                kappa: left_kappa,
                boundary: left_boundary,
                solver: left_solver,
                ..
            },
            MeasurementParams::PionWilson {
                kappa: right_kappa,
                boundary: right_boundary,
                solver: right_solver,
                ..
            },
        ) => {
            left_kappa == right_kappa
                && left_boundary == right_boundary
                && left_solver == right_solver
        }
        (
            MeasurementParams::PionStaggered {
                mass: left_mass,
                boundary: left_boundary,
                solver: left_solver,
                ..
            },
            MeasurementParams::PionStaggered {
                mass: right_mass,
                boundary: right_boundary,
                solver: right_solver,
                ..
            },
        ) => {
            left_mass == right_mass
                && left_boundary == right_boundary
                && left_solver == right_solver
        }
        (
            MeasurementParams::ChiralStaggered {
                mass: left_mass,
                boundary: left_boundary,
                solver: left_solver,
                sources: left_sources,
                flavors: left_flavors,
                ..
            },
            MeasurementParams::ChiralStaggered {
                mass: right_mass,
                boundary: right_boundary,
                solver: right_solver,
                sources: right_sources,
                flavors: right_flavors,
                ..
            },
        ) => {
            left_mass == right_mass
                && left_boundary == right_boundary
                && left_solver == right_solver
                && left_sources == right_sources
                && left_flavors == right_flavors
        }
        _ => false,
    }
}

fn validate_flow(flow: &GradientFlowParams) -> Result<(), ParamsError> {
    if flow.every_trajectories == 0 {
        return Err(ParamsError::InvalidFlowInterval);
    }
    if !flow.step_size.is_finite() || flow.step_size <= 0.0 {
        return Err(ParamsError::InvalidFlowStepSize);
    }
    if flow.steps == 0 {
        return Err(ParamsError::InvalidFlowSteps);
    }
    if flow.measure_every_steps == 0 {
        return Err(ParamsError::InvalidFlowMeasurementInterval);
    }
    if !flow.steps.is_multiple_of(flow.measure_every_steps) {
        return Err(ParamsError::MisalignedFlowMeasurements);
    }
    if flow.measurements.is_empty() {
        return Err(ParamsError::EmptyFlowMeasurements);
    }
    for (index, measurement) in flow.measurements.iter().enumerate() {
        if flow.measurements[..index].contains(measurement) {
            return Err(ParamsError::DuplicateFlowMeasurement);
        }
    }
    Ok(())
}

fn validate_output(output: &OutputParams) -> Result<(), ParamsError> {
    if output.directory.as_os_str().is_empty() {
        return Err(ParamsError::EmptyOutputDirectory);
    }
    if output.every == 0 {
        return Err(ParamsError::InvalidOutputInterval);
    }
    if output.prefix.is_empty()
        || output.prefix == "."
        || output.prefix == ".."
        || !output
            .prefix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(ParamsError::UnsafeOutputPrefix);
    }
    Ok(())
}

impl SolverConfig {
    fn to_lower(&self) -> Result<SolverParams, ParamsError> {
        SolverParams::new(self.tolerance, self.max_iterations).map_err(ParamsError::Fermion)
    }
}

impl MeasurementParams {
    pub(crate) fn every(&self) -> usize {
        measurement_every(self)
    }
}

impl GradientFlowParams {
    pub(crate) fn every_trajectories(&self) -> usize {
        self.every_trajectories
    }
}

impl OutputParams {
    pub(crate) fn destination(&self, trajectory_id: usize) -> PathBuf {
        self.directory
            .join(format!("{}_{trajectory_id:08}.ildg", self.prefix))
    }
}
