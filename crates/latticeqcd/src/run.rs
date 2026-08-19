use crate::{
    error::{RunError, RunFailure},
    params::{
        FermionParams, FlowMeasurement, GradientFlowParams, InitialParams, MeasurementParams,
        OutputParams, Params, SolverConfig, UpdateParams,
    },
    report::{
        FlowRecord, MeasurementKind, MeasurementRecord, MeasurementValue, RunReport, UpdateKind,
        UpdateOutcome, UpdateRecord,
    },
};
use dirac_operators::{FermionBoundary, SolverParams, StaggeredDirac, WilsonDirac};
use gaugefields::{
    cold_su3, heatbath_sweep, hmc_update, normalized_plaquette, read_ildg, write_ildg,
    CpuEvolutionContext, GaugeLinks, HeatbathParams, HmcParams, LatticeShape4, ReproducibleRng,
};
use measurements::fermions::{pion_correlator, stochastic_chiral_condensate};
use measurements::{
    clover_topological_charge, gradient_flow, polyakov_loop, GradientFlowParams as LowerFlowParams,
};
use std::{
    fs,
    fs::OpenOptions,
    path::{Path, PathBuf},
};
use tenferro_cpu::CpuBackend;
use wilsonloop::{LoopAction, LoopTerm};

/// Execute one validated strict parameter document.
///
/// Validation completes before the initial gauge field, backend context, or
/// RNG is created. Updates are dispatched exactly from the selected sea action;
/// measurements, flow, and output then run in schedule order. If execution
/// stops after validation, the returned [`RunFailure`] contains the completed
/// prefix of the report. HMC rejections count as completed updates, while RNG
/// words already consumed by a failed operation are not rolled back.
///
/// # Errors
///
/// Returns a typed [`RunFailure`] with trajectory/measurement context and a
/// partial report. [`RunError::Params`] covers pre-run validation;
/// [`RunError::Gauge`], [`RunError::Dirac`], measurement/flow errors, and
/// [`RunError::OutputIo`] cover execution; [`RunError::OutputExists`] preserves
/// the no-clobber guarantee. Invalid parameters fail before execution-side
/// effects.
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
/// assert_eq!(report.completed_updates, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn run_lqcd(params: &Params) -> Result<RunReport, RunFailure> {
    if let Err(source) = params.validate() {
        return Err(RunFailure::new(
            source.into(),
            RunReport::new(params.control.trajectories, [0; 4], [0; 4]),
            None,
            None,
            None,
        ));
    }

    let lattice = match LatticeShape4::new(params.physical.lattice) {
        Ok(lattice) => lattice,
        Err(source) => {
            return Err(RunFailure::new(
                source.into(),
                RunReport::new(params.control.trajectories, [0; 4], [0; 4]),
                None,
                None,
                None,
            ))
        }
    };
    let state = match params.rng.state() {
        Ok(state) => state,
        Err(source) => {
            return Err(RunFailure::new(
                source.into(),
                RunReport::new(params.control.trajectories, lattice.extents(), [0; 4]),
                None,
                None,
                None,
            ))
        }
    };
    let mut report = RunReport::new(params.control.trajectories, lattice.extents(), state);
    let mut links = match &params.initial {
        InitialParams::Cold {} => match cold_su3(lattice) {
            Ok(links) => links,
            Err(source) => return Err(RunFailure::new(source.into(), report, None, None, None)),
        },
        InitialParams::Ildg { path } => {
            let links = match read_ildg(path) {
                Ok(links) => links,
                Err(source) => {
                    return Err(RunFailure::new(source.into(), report, None, None, None))
                }
            };
            if links.lattice() != lattice {
                return Err(RunFailure::new(
                    RunError::InitialLatticeMismatch {
                        expected: lattice.extents(),
                        found: links.lattice().extents(),
                    },
                    report,
                    None,
                    None,
                    None,
                ));
            }
            links
        }
    };

    let mut rng = match ReproducibleRng::from_state(state) {
        Ok(rng) => rng,
        Err(source) => return Err(RunFailure::new(source.into(), report, None, None, None)),
    };
    let flow_action = if params.gradient_flow.is_some() {
        match wilson_flow_action() {
            Ok(action) => Some(action),
            Err(source) => return Err(RunFailure::new(source, report, None, None, None)),
        }
    } else {
        None
    };
    let mut context = CpuEvolutionContext::new(CpuBackend::new());

    if params.control.measure_initial {
        let trajectory_id = params.control.first_trajectory - 1;
        append_bare_measurements(
            &params.measurements,
            trajectory_id,
            &links,
            &mut rng,
            &mut report,
        )?;
    }

    for update_index in 0..params.control.trajectories {
        let trajectory_id = params.control.first_trajectory + update_index;
        let outcome = match execute_update(params, &mut links, &mut context, &mut rng) {
            Ok(outcome) => outcome,
            Err(source) => {
                return Err(RunFailure::new(
                    source,
                    report,
                    Some(trajectory_id),
                    None,
                    None,
                ))
            }
        };
        let accepted = match &outcome {
            UpdateOutcome::Hmc { accepted, .. } => Some(*accepted),
            UpdateOutcome::Heatbath { .. } => None,
        };
        report.completed_updates += 1;
        if let Some(accepted) = accepted {
            if accepted {
                report.accepted_updates += 1;
            } else {
                report.rejected_updates += 1;
            }
        }
        report.updates.push(UpdateRecord {
            trajectory_id,
            outcome,
        });

        if report.completed_updates <= params.control.thermalization {
            continue;
        }

        append_bare_measurements(
            &params.measurements,
            trajectory_id,
            &links,
            &mut rng,
            &mut report,
        )?;

        if let Some(flow) = &params.gradient_flow {
            if trajectory_id.is_multiple_of(flow.every_trajectories()) {
                let Some(action) = flow_action.as_ref() else {
                    return Err(RunFailure::new(
                        RunError::Params(crate::params::ParamsError::UnsupportedCombination),
                        report,
                        Some(trajectory_id),
                        None,
                        None,
                    ));
                };
                append_flow_records(
                    flow,
                    action,
                    trajectory_id,
                    &links,
                    &mut context,
                    &mut report,
                )?;
            }
        }

        if let Some(output) = &params.output {
            if trajectory_id.is_multiple_of(output.every) {
                match publish_output(output, trajectory_id, &links) {
                    Ok(path) => report.published_paths.push(path),
                    Err(source) => {
                        return Err(RunFailure::new(
                            source,
                            report,
                            Some(trajectory_id),
                            None,
                            None,
                        ))
                    }
                }
            }
        }
    }

    Ok(report)
}

fn execute_update(
    params: &Params,
    links: &mut GaugeLinks,
    context: &mut CpuEvolutionContext,
    rng: &mut ReproducibleRng,
) -> Result<UpdateOutcome, RunError> {
    match (&params.fermions, &params.update) {
        (FermionParams::Quenched {}, UpdateParams::Hmc { step_size, steps }) => {
            let outcome = hmc_update(
                context,
                links,
                HmcParams::new(params.physical.beta, *step_size, *steps)?,
                rng,
            )?;
            Ok(UpdateOutcome::Hmc {
                kind: UpdateKind::QuenchedHmc,
                accepted: outcome.accepted,
                delta_h: outcome.delta_h,
                acceptance_probability: outcome.acceptance_probability,
            })
        }
        (
            FermionParams::WilsonNf2 {
                kappa,
                boundary,
                solver,
            },
            UpdateParams::Hmc { step_size, steps },
        ) => {
            let outcome = dirac_operators::wilson_hmc_update(
                context,
                links,
                dirac_operators::WilsonHmcParams::new(
                    params.physical.beta,
                    *kappa,
                    *step_size,
                    *steps,
                    FermionBoundary::new(*boundary)?,
                    lower_solver(solver)?,
                )?,
                rng,
            )?;
            Ok(UpdateOutcome::Hmc {
                kind: UpdateKind::WilsonHmc,
                accepted: outcome.accepted,
                delta_h: outcome.delta_h,
                acceptance_probability: outcome.acceptance_probability,
            })
        }
        (
            FermionParams::StaggeredNf2 {
                mass,
                boundary,
                lambda_low,
                lambda_high,
                solver,
            },
            UpdateParams::Hmc { step_size, steps },
        ) => {
            let outcome = dirac_operators::staggered_hmc_update(
                context,
                links,
                dirac_operators::StaggeredHmcParams::new(
                    params.physical.beta,
                    *mass,
                    *step_size,
                    *steps,
                    FermionBoundary::new(*boundary)?,
                    *lambda_low,
                    *lambda_high,
                    lower_solver(solver)?,
                )?,
                rng,
            )?;
            Ok(UpdateOutcome::Hmc {
                kind: UpdateKind::StaggeredHmc,
                accepted: outcome.accepted,
                delta_h: outcome.delta_h,
                acceptance_probability: outcome.acceptance_probability,
            })
        }
        (FermionParams::Quenched {}, UpdateParams::Heatbath { max_attempts }) => {
            let outcome = heatbath_sweep(
                links,
                HeatbathParams::new(params.physical.beta, *max_attempts)?,
                rng,
            )?;
            Ok(UpdateOutcome::Heatbath {
                kind: UpdateKind::Heatbath,
                updated_links: outcome.updated_links,
                su2_attempts: outcome.su2_attempts,
            })
        }
        _ => Err(RunError::Params(
            crate::params::ParamsError::UnsupportedCombination,
        )),
    }
}

fn append_bare_measurements(
    measurements: &[MeasurementParams],
    trajectory_id: usize,
    links: &GaugeLinks,
    rng: &mut ReproducibleRng,
    report: &mut RunReport,
) -> Result<(), RunFailure> {
    for (measurement_index, measurement) in measurements.iter().enumerate() {
        if !trajectory_id.is_multiple_of(measurement.every()) {
            continue;
        }
        let record = match measure_one(measurement, trajectory_id, measurement_index, links, rng) {
            Ok(record) => record,
            Err(source) => {
                return Err(RunFailure::new(
                    source,
                    report.clone(),
                    Some(trajectory_id),
                    None,
                    Some(measurement_index),
                ))
            }
        };
        report.measurements.push(record);
    }
    Ok(())
}

fn measure_one(
    measurement: &MeasurementParams,
    trajectory_id: usize,
    measurement_index: usize,
    links: &GaugeLinks,
    rng: &mut ReproducibleRng,
) -> Result<MeasurementRecord, RunError> {
    let (kind, value) = match measurement {
        MeasurementParams::Plaquette { .. } => (
            MeasurementKind::Plaquette,
            MeasurementValue::Scalar(normalized_plaquette(links)?),
        ),
        MeasurementParams::PolyakovLoop { .. } => (
            MeasurementKind::PolyakovLoop,
            MeasurementValue::PolyakovLoop(polyakov_loop(links)?),
        ),
        MeasurementParams::CloverTopologicalCharge { .. } => (
            MeasurementKind::CloverTopologicalCharge,
            MeasurementValue::Scalar(clover_topological_charge(links)?),
        ),
        MeasurementParams::PionWilson {
            kappa,
            boundary,
            solver,
            ..
        } => {
            let operator =
                WilsonDirac::with_boundary(links, *kappa, FermionBoundary::new(*boundary)?)?;
            let result = pion_correlator(&operator, lower_solver(solver)?)?;
            (
                MeasurementKind::PionWilson,
                MeasurementValue::Pion {
                    values: result.values,
                    solver_reports: result.solver_reports,
                },
            )
        }
        MeasurementParams::PionStaggered {
            mass,
            boundary,
            solver,
            ..
        } => {
            let operator =
                StaggeredDirac::with_boundary(links, *mass, FermionBoundary::new(*boundary)?)?;
            let result = pion_correlator(&operator, lower_solver(solver)?)?;
            (
                MeasurementKind::PionStaggered,
                MeasurementValue::Pion {
                    values: result.values,
                    solver_reports: result.solver_reports,
                },
            )
        }
        MeasurementParams::ChiralStaggered {
            mass,
            boundary,
            solver,
            sources,
            flavors,
            ..
        } => {
            let operator =
                StaggeredDirac::with_boundary(links, *mass, FermionBoundary::new(*boundary)?)?;
            let result = stochastic_chiral_condensate(
                &operator,
                *flavors as f64 / 4.0,
                *sources,
                lower_solver(solver)?,
                rng,
            )?;
            (
                MeasurementKind::ChiralStaggered,
                MeasurementValue::Chiral {
                    value: result.value,
                    source_values: result.source_values,
                    solver_reports: result.solver_reports,
                },
            )
        }
    };
    Ok(MeasurementRecord {
        trajectory_id,
        measurement_index,
        kind,
        value,
    })
}

fn append_flow_records(
    flow: &GradientFlowParams,
    action: &LoopAction,
    trajectory_id: usize,
    links: &GaugeLinks,
    context: &mut CpuEvolutionContext,
    report: &mut RunReport,
) -> Result<(), RunFailure> {
    let one_step = match LowerFlowParams::new(flow.step_size, 1) {
        Ok(params) => params,
        Err(source) => {
            return Err(RunFailure::new(
                source.into(),
                report.clone(),
                Some(trajectory_id),
                None,
                None,
            ))
        }
    };
    let mut flowed = match links.try_clone() {
        Ok(links) => links,
        Err(source) => {
            return Err(RunFailure::new(
                source.into(),
                report.clone(),
                Some(trajectory_id),
                Some(0),
                None,
            ))
        }
    };
    for step in 1..=flow.steps {
        flowed = match gradient_flow(context, &flowed, action, one_step) {
            Ok(links) => links,
            Err(source) => {
                return Err(RunFailure::new(
                    source.into(),
                    report.clone(),
                    Some(trajectory_id),
                    Some(step),
                    None,
                ))
            }
        };
        if step % flow.measure_every_steps != 0 {
            continue;
        }
        let mut measurements = Vec::with_capacity(flow.measurements.len());
        for (measurement_index, measurement) in flow.measurements.iter().enumerate() {
            let (kind, value) = match measure_flow(measurement, &flowed) {
                Ok(value) => value,
                Err(source) => {
                    return Err(RunFailure::new(
                        source,
                        report.clone(),
                        Some(trajectory_id),
                        Some(step),
                        Some(measurement_index),
                    ))
                }
            };
            measurements.push(MeasurementRecord {
                trajectory_id,
                measurement_index,
                kind,
                value,
            });
        }
        report.flows.push(FlowRecord {
            trajectory_id,
            step,
            measurements,
        });
    }
    Ok(())
}

fn measure_flow(
    measurement: &FlowMeasurement,
    links: &GaugeLinks,
) -> Result<(MeasurementKind, MeasurementValue), RunError> {
    Ok(match measurement {
        FlowMeasurement::Plaquette => (
            MeasurementKind::Plaquette,
            MeasurementValue::Scalar(normalized_plaquette(links)?),
        ),
        FlowMeasurement::PolyakovLoop => (
            MeasurementKind::PolyakovLoop,
            MeasurementValue::PolyakovLoop(polyakov_loop(links)?),
        ),
        FlowMeasurement::CloverTopologicalCharge => (
            MeasurementKind::CloverTopologicalCharge,
            MeasurementValue::Scalar(clover_topological_charge(links)?),
        ),
    })
}

fn lower_solver(config: &SolverConfig) -> Result<SolverParams, RunError> {
    Ok(SolverParams::new(config.tolerance, config.max_iterations)?)
}

fn wilson_flow_action() -> Result<LoopAction, RunError> {
    let mut terms = Vec::with_capacity(6);
    for mu in 1..=3 {
        for nu in (mu + 1)..=4 {
            terms.push(LoopTerm::plaquette(1.0, mu, nu)?);
        }
    }
    Ok(LoopAction::new(terms)?)
}

fn publish_output(
    output: &OutputParams,
    trajectory_id: usize,
    links: &GaugeLinks,
) -> Result<PathBuf, RunError> {
    fs::create_dir_all(&output.directory).map_err(|source| RunError::OutputDirectory {
        path: output.directory.clone(),
        source,
    })?;
    let destination = output.destination(trajectory_id);
    if destination.exists() {
        return Err(RunError::OutputExists { path: destination });
    }

    let temporary = create_temporary(&output.directory, &output.prefix, trajectory_id)?;
    if let Err(source) = write_ildg(&temporary, links) {
        let _ = fs::remove_file(&temporary);
        return Err(source.into());
    }
    if let Err(source) = fs::hard_link(&temporary, &destination) {
        let _ = fs::remove_file(&temporary);
        if source.kind() == std::io::ErrorKind::AlreadyExists {
            return Err(RunError::OutputExists { path: destination });
        }
        return Err(RunError::OutputIo {
            path: destination,
            source,
        });
    }
    let _ = fs::remove_file(&temporary);
    Ok(destination)
}

fn create_temporary(
    directory: &Path,
    prefix: &str,
    trajectory_id: usize,
) -> Result<PathBuf, RunError> {
    for attempt in 0..1024_usize {
        let suffix = if attempt == 0 {
            String::new()
        } else {
            format!("_{attempt}")
        };
        let path = directory.join(format!(".{prefix}_{trajectory_id:08}{suffix}.tmp"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                drop(file);
                return Ok(path);
            }
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(RunError::OutputIo { path, source }),
        }
    }
    Err(RunError::OutputIo {
        path: directory.join(format!(".{prefix}_{trajectory_id:08}.tmp")),
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique temporary ILDG path",
        ),
    })
}
