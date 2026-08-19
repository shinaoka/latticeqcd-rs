//! Strict TOML-driven execution for the host lattice-QCD primitives.

mod error;
mod params;
mod report;
mod run;

pub use error::{RunError, RunFailure};
pub use params::{
    ControlParams, FermionParams, FlowMeasurement, GradientFlowParams, InitialParams,
    MeasurementParams, OutputParams, Params, ParamsError, PhysicalParams, RngParams, SolverConfig,
    UpdateParams,
};
pub use report::{
    FlowRecord, MeasurementKind, MeasurementRecord, MeasurementValue, RunReport, UpdateKind,
    UpdateOutcome, UpdateRecord,
};
pub use run::run_lqcd;
