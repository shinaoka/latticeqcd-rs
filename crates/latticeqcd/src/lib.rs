//! Strict TOML-driven execution for the host lattice-QCD primitives.
//!
//! Parameter/control naming and workflow structure follow the MIT-licensed
//! LatticeQCD.jl v1.3.7 at revision
//! `c09de20aae10f28f6a9c7e84e7711fce94d50915`; Rust uses strict tagged
//! validation and the explicit scheduler documented by this crate rather than
//! copying the pinned frontend's permissive parser behavior.
//!
//! The checked frontend path is:
//!
//! ```text
//! cargo run -p latticeqcd -- examples/phase4.toml
//! ```
//!
//! It validates one strict TOML document, owns one RNG and CPU evolution
//! context, dispatches the selected quenched/Wilson/staggered update, and
//! records ordered update and measurement reports. The Phase 4 ensemble
//! evidence is intentionally separate: its fixed Julia schedule and the
//! independent Rust comparison live in
//! `fixtures/fermion_measurements_phase4_ensemble` and
//! `crates/measurements/tests/phase4_ensemble.rs`. That evidence uses the
//! corrected pion contraction and canonical Z4 sources, records pinned
//! Gaugefields.jl 0.7.2, LatticeDiracOperators.jl 0.6.4, Wilsonloop.jl 0.1.5,
//! QCDMeasurements.jl 0.2.13, and Julia 1.12.5 provenance, and avoids the
//! pinned upstream Issue #27, #29, and #30 paths. It does not promise bitwise
//! Julia/Rust configuration parity; the statistical gate is six combined
//! standard errors of four block means.

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
