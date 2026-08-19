use latticeqcd::{
    run_lqcd, Params, ParamsError, RunError, RunReport, UpdateKind, UpdateOutcome, UpdateParams,
};
use std::{
    fs,
    path::PathBuf,
    process,
    time::{SystemTime, UNIX_EPOCH},
};

const BASE: &str = r#"
schema_version = 1

[physical]
lattice = [2, 2, 2, 2]
beta = 5.7

[initial]
kind = "cold"

[fermions]
kind = "quenched"

[update]
kind = "hmc"
step_size = 0.0001
steps = 1

[rng]
state_hex = ["0000000000000001", "0000000000000002", "0000000000000003", "0000000000000004"]

[control]
first_trajectory = 1
trajectories = 1
thermalization = 0
measure_initial = false
"#;

fn with_measurements(measurements: &str) -> String {
    format!("{BASE}\n{measurements}")
}

#[test]
fn all_approved_tagged_variants_validate() {
    let wilson = BASE
        .replace("kind = \"quenched\"", "kind = \"wilson_nf2\"\nkappa = 0.08\nboundary = [1, 1, 1, -1]")
        .replace("[update]\nkind = \"hmc\"", "[fermions.solver]\ntolerance = 1e-20\nmax_iterations = 1000\n\n[update]\nkind = \"hmc\"");
    Params::from_toml(&wilson).expect("wilson HMC schema");

    let staggered = BASE
        .replace("kind = \"quenched\"", "kind = \"staggered_nf2\"\nmass = 0.17\nboundary = [1, 1, 1, -1]\nlambda_low = 0.0004\nlambda_high = 64.0")
        .replace("[update]\nkind = \"hmc\"", "[fermions.solver]\ntolerance = 1e-20\nmax_iterations = 1000\n\n[update]\nkind = \"hmc\"");
    Params::from_toml(&staggered).expect("staggered HMC schema");

    let heatbath = BASE.replace(
        "kind = \"hmc\"\nstep_size = 0.0001\nsteps = 1",
        "kind = \"heatbath\"\nmax_attempts = 10",
    );
    Params::from_toml(&heatbath).expect("quenched heatbath schema");
}

#[test]
fn measurements_and_flow_are_strictly_scheduled() {
    let source = with_measurements(
        r#"
[[measurements]]
every = 1
kind = "plaquette"

[[measurements]]
every = 1
kind = "polyakov_loop"

[[measurements]]
every = 1
kind = "clover_topological_charge"

[gradient_flow]
every_trajectories = 1
step_size = 0.001
steps = 2
measure_every_steps = 1
measurements = ["plaquette", "polyakov_loop", "clover_topological_charge"]
"#,
    );
    let params = Params::from_toml(&source).expect("flow schema");
    let report = run_lqcd(&params).expect("flow run");
    assert_eq!(report.completed_updates, 1);
    assert_eq!(report.measurements.len(), 3);
    assert_eq!(report.flows.len(), 2);
    assert_eq!(report.flows[0].step, 1);
    assert_eq!(report.flows[1].step, 2);
    assert!(matches!(
        &report.updates[0].outcome,
        UpdateOutcome::Hmc { .. }
    ));
}

#[test]
fn fermion_measurements_use_the_approved_paths() {
    let source = with_measurements(
        r#"
[[measurements]]
every = 1
kind = "pion_wilson"
kappa = 0.08
boundary = [1, 1, 1, -1]
[measurements.solver]
tolerance = 1e-12
max_iterations = 1000

[[measurements]]
every = 1
kind = "pion_staggered"
mass = 0.17
boundary = [1, 1, 1, -1]
[measurements.solver]
tolerance = 1e-12
max_iterations = 1000

[[measurements]]
every = 1
kind = "chiral_staggered"
mass = 0.17
boundary = [1, 1, 1, -1]
sources = 1
flavors = 2
[measurements.solver]
tolerance = 1e-12
max_iterations = 1000
"#,
    );
    let params = Params::from_toml(&source).expect("fermion measurement schema");
    let report = run_lqcd(&params).expect("fermion measurements run");
    assert_eq!(report.measurements.len(), 3);
    assert!(report
        .measurements
        .iter()
        .all(|record| record.trajectory_id == 1));
}

#[test]
fn invalid_combinations_and_schedules_are_rejected_before_run() {
    let heatbath_with_wilson = BASE
        .replace("kind = \"quenched\"", "kind = \"wilson_nf2\"\nkappa = 0.08\nboundary = [1, 1, 1, -1]")
        .replace("[update]\nkind = \"hmc\"", "[fermions.solver]\ntolerance = 1e-20\nmax_iterations = 1000\n\n[update]\nkind = \"heatbath\"\nmax_attempts = 10")
        .replace("step_size = 0.0001\nsteps = 1\n", "");
    assert!(Params::from_toml(&heatbath_with_wilson).is_err());

    let duplicate = with_measurements(
        r#"
[[measurements]]
every = 1
kind = "plaquette"
[[measurements]]
every = 2
kind = "plaquette"
"#,
    );
    assert!(Params::from_toml(&duplicate).is_err());

    let misaligned_flow = with_measurements(
        r#"
[gradient_flow]
every_trajectories = 1
step_size = 0.01
steps = 3
measure_every_steps = 2
measurements = ["plaquette"]
"#,
    );
    assert!(Params::from_toml(&misaligned_flow).is_err());
}

#[test]
fn initial_measurement_and_thermalization_boundaries_are_reported() {
    let source = BASE
        .replace("measure_initial = false", "measure_initial = true")
        .replace(
            "[control]",
            "[[measurements]]\nevery = 2\nkind = \"plaquette\"\n\n[control]",
        );
    let params = Params::from_toml(&source).expect("initial measurement schema");
    let report = run_lqcd(&params).expect("initial measurement run");
    assert_eq!(report.measurements.len(), 1);
    assert_eq!(report.measurements[0].trajectory_id, 0);

    let invalid = source.replace("thermalization = 0", "thermalization = 1");
    assert!(Params::from_toml(&invalid).is_err());
}

#[test]
fn ildg_start_and_no_clobber_output_are_checked() {
    let directory = unique_directory("latticeqcd-task-b");
    fs::create_dir_all(&directory).expect("temporary directory");
    let input = directory.join("input.ildg");
    let output = directory.join("out");
    let cold = gaugefields::cold_su3(gaugefields::LatticeShape4::new([2, 2, 2, 2]).unwrap())
        .expect("cold links");
    gaugefields::write_ildg(&input, &cold).expect("ILDG input");

    let source = BASE.replace(
        "[initial]\nkind = \"cold\"",
        &format!("[initial]\nkind = \"ildg\"\npath = \"{}\"", input.display()),
    ) + &format!(
        "\n[output]\ndirectory = \"{}\"\nprefix = \"cfg\"\nevery = 1\n",
        output.display()
    );
    let params = Params::from_toml(&source).expect("ILDG start schema");
    let first = run_lqcd(&params).expect("ILDG start run");
    assert_eq!(first.published_paths.len(), 1);
    let destination = output.join("cfg_00000001.ildg");
    let bytes = fs::read(&destination).expect("published ILDG");
    let entries = fs::read_dir(&output)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(entries, vec![destination.file_name().unwrap().to_owned()]);

    let failure = run_lqcd(&params).expect_err("existing output must fail");
    assert!(matches!(
        failure.source.as_ref(),
        RunError::OutputExists { .. }
    ));
    assert_eq!(failure.report.completed_updates, 1);
    assert_eq!(failure.report.updates.len(), 1);
    assert!(failure.report.published_paths.is_empty());
    assert_eq!(fs::read(&destination).expect("existing bytes"), bytes);
    let entries = fs::read_dir(&output)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(entries, vec![destination.file_name().unwrap().to_owned()]);

    let _ = fs::remove_dir_all(directory);
}

#[test]
fn executes_quenched_heatbath_with_exact_dispatch_and_counts() {
    let source = BASE.replace(
        "kind = \"hmc\"\nstep_size = 0.0001\nsteps = 1",
        "kind = \"heatbath\"\nmax_attempts = 100000",
    );
    let params = Params::from_toml(&source).expect("heatbath schema");
    let report = run_lqcd(&params).expect("heatbath run");

    assert_report_header(&report);
    assert_eq!(report.accepted_updates, 0);
    assert_eq!(report.rejected_updates, 0);
    assert!(matches!(
        report.updates[0].outcome,
        UpdateOutcome::Heatbath {
            kind: UpdateKind::Heatbath,
            updated_links,
            su2_attempts,
        } if updated_links > 0 && su2_attempts >= updated_links * 3
    ));
}

#[test]
fn executes_wilson_nf2_hmc_with_exact_dispatch_and_finite_diagnostics() {
    let source = BASE
        .replace(
            "kind = \"quenched\"",
            "kind = \"wilson_nf2\"\nkappa = 0.13\nboundary = [1, 1, 1, -1]",
        )
        .replace(
            "[update]\nkind = \"hmc\"",
            "[fermions.solver]\ntolerance = 1e-20\nmax_iterations = 2000\n\n[update]\nkind = \"hmc\"",
        );
    let params = Params::from_toml(&source).expect("Wilson HMC schema");
    let report = run_lqcd(&params).expect("Wilson HMC run");

    assert_hmc_report(&report, UpdateKind::WilsonHmc);
}

#[test]
fn executes_staggered_nf2_hmc_with_exact_dispatch_and_finite_diagnostics() {
    let source = BASE
        .replace(
            "kind = \"quenched\"",
            "kind = \"staggered_nf2\"\nmass = 0.17\nboundary = [1, 1, 1, -1]\nlambda_low = 0.0004\nlambda_high = 64.0",
        )
        .replace(
            "[update]\nkind = \"hmc\"",
            "[fermions.solver]\ntolerance = 1e-24\nmax_iterations = 2000\n\n[update]\nkind = \"hmc\"",
        )
        .replace(
            "step_size = 0.0001\nsteps = 1",
            "step_size = 0.001\nsteps = 2",
        );
    let params = Params::from_toml(&source).expect("staggered HMC schema");
    let report = run_lqcd(&params).expect("staggered HMC run");

    assert_hmc_report(&report, UpdateKind::StaggeredHmc);
}

#[test]
fn invalid_hmc_and_chiral_values_have_distinct_typed_errors() {
    let invalid_step_size = BASE.replace("step_size = 0.0001", "step_size = 0.0");
    assert!(matches!(
        Params::from_toml(&invalid_step_size),
        Err(ParamsError::InvalidHmcStepSize)
    ));

    let invalid_steps = BASE.replace("steps = 1", "steps = 0");
    assert!(matches!(
        Params::from_toml(&invalid_steps),
        Err(ParamsError::InvalidHmcSteps)
    ));

    let zero_sources = chiral_source(1, 0, 2);
    assert!(matches!(
        Params::from_toml(&with_measurements(&zero_sources)),
        Err(ParamsError::InvalidChiralSources)
    ));

    let zero_flavors = chiral_source(1, 1, 0);
    assert!(matches!(
        Params::from_toml(&with_measurements(&zero_flavors)),
        Err(ParamsError::InvalidChiralFlavors)
    ));

    let inexact_flavors = chiral_source(1, 1, 9_007_199_254_740_993);
    assert!(matches!(
        Params::from_toml(&with_measurements(&inexact_flavors)),
        Err(ParamsError::InvalidChiralFlavors)
    ));
}

#[test]
fn invalid_input_has_no_side_effects_or_completed_updates() {
    let directory = unique_directory("latticeqcd-invalid");
    let source = BASE.to_owned()
        + &format!(
            "\n[output]\ndirectory = \"{}\"\nprefix = \"cfg\"\nevery = 1\n",
            directory.display()
        );
    let mut params = Params::from_toml(&source).expect("valid base before mutation");
    params.update = UpdateParams::Hmc {
        step_size: 0.0,
        steps: 1,
    };

    let failure = run_lqcd(&params).expect_err("invalid input must fail before execution");
    assert!(matches!(
        failure.source.as_ref(),
        RunError::Params(ParamsError::InvalidHmcStepSize)
    ));
    assert_eq!(failure.report.completed_updates, 0);
    assert!(!directory.exists());
}

#[test]
fn initial_interval_and_thermalization_only_scheduling_are_exact() {
    let initial_source = BASE
        .replace("first_trajectory = 1", "first_trajectory = 3")
        .replace("trajectories = 1", "trajectories = 2")
        .replace("measure_initial = false", "measure_initial = true")
        + r#"
[[measurements]]
every = 2
kind = "plaquette"
"#;
    let initial_report = run_lqcd(&Params::from_toml(&initial_source).unwrap()).unwrap();
    let ids = initial_report
        .measurements
        .iter()
        .map(|record| record.trajectory_id)
        .collect::<Vec<_>>();
    assert_eq!(ids, [2, 4]);

    let directory = unique_directory("latticeqcd-thermal-only");
    let thermal_source = BASE.replace("thermalization = 0", "thermalization = 1")
        + r#"
[[measurements]]
every = 1
kind = "plaquette"

[gradient_flow]
every_trajectories = 1
step_size = 0.001
steps = 1
measure_every_steps = 1
measurements = ["plaquette"]
"# + &format!(
        "\n[output]\ndirectory = \"{}\"\nprefix = \"cfg\"\nevery = 1\n",
        directory.display()
    );
    let thermal_report = run_lqcd(&Params::from_toml(&thermal_source).unwrap()).unwrap();
    assert_eq!(thermal_report.completed_updates, 1);
    assert!(thermal_report.measurements.is_empty());
    assert!(thermal_report.flows.is_empty());
    assert!(thermal_report.published_paths.is_empty());
    assert!(!directory.exists());
}

#[test]
fn rng_consumption_follows_measurement_order_and_invalid_input_is_pure() {
    let base = BASE.replace("trajectories = 1", "trajectories = 2");
    let no_chiral = run_lqcd(&Params::from_toml(&base).unwrap()).unwrap();
    let every_one =
        run_lqcd(&Params::from_toml(&format!("{base}\n{}", chiral_source(1, 1, 2))).unwrap())
            .unwrap();
    let every_two =
        run_lqcd(&Params::from_toml(&format!("{base}\n{}", chiral_source(2, 1, 2))).unwrap())
            .unwrap();

    // The first update is the same raw stream prefix. A measurement after
    // trajectory 1 shifts the subsequent HMC outcome; delaying it to
    // trajectory 2 leaves that outcome unchanged.
    assert_eq!(no_chiral.updates[0], every_one.updates[0]);
    assert_eq!(no_chiral.updates[1], every_two.updates[1]);
    assert_ne!(no_chiral.updates[1], every_one.updates[1]);
    assert_eq!(every_one.measurements.len(), 2);
    assert_eq!(every_two.measurements.len(), 1);
    assert_eq!(every_two.measurements[0].trajectory_id, 2);

    // Invalid input is rejected before run_lqcd owns or advances its local
    // RNG; the following valid run reproduces the exact baseline outcome.
    let invalid = format!("{base}\n{}", chiral_source(1, 0, 2));
    assert!(matches!(
        Params::from_toml(&invalid),
        Err(ParamsError::InvalidChiralSources)
    ));
    let after_invalid = run_lqcd(&Params::from_toml(&base).unwrap()).unwrap();
    assert_eq!(no_chiral.updates, after_invalid.updates);
}

#[test]
fn wrong_ildg_lattice_fails_before_any_update() {
    let directory = unique_directory("latticeqcd-wrong-ildg");
    fs::create_dir_all(&directory).unwrap();
    let input = directory.join("wrong.ildg");
    let wrong =
        gaugefields::cold_su3(gaugefields::LatticeShape4::new([2, 2, 2, 4]).unwrap()).unwrap();
    gaugefields::write_ildg(&input, &wrong).unwrap();
    let source = BASE.replace(
        "[initial]\nkind = \"cold\"",
        &format!("[initial]\nkind = \"ildg\"\npath = \"{}\"", input.display()),
    );

    let failure = run_lqcd(&Params::from_toml(&source).unwrap()).expect_err("wrong lattice");
    assert!(matches!(
        failure.source.as_ref(),
        RunError::InitialLatticeMismatch { .. }
    ));
    assert_eq!(failure.report.completed_updates, 0);
    assert_eq!(failure.report.lattice, [2, 2, 2, 2]);
    assert_eq!(failure.report.initial_rng_state, [1, 2, 3, 4]);
    assert!(failure.report.updates.is_empty());
    let _ = fs::remove_dir_all(directory);
}

fn assert_report_header(report: &RunReport) {
    assert_eq!(report.lattice, [2, 2, 2, 2]);
    assert_eq!(report.initial_rng_state, [1, 2, 3, 4]);
    assert_eq!(report.requested_updates, 1);
    assert_eq!(report.completed_updates, 1);
    assert_eq!(report.updates.len(), 1);
}

fn assert_hmc_report(report: &RunReport, expected_kind: UpdateKind) {
    assert_report_header(report);
    assert_eq!(report.accepted_updates + report.rejected_updates, 1);
    match &report.updates[0].outcome {
        UpdateOutcome::Hmc {
            kind,
            delta_h,
            acceptance_probability,
            ..
        } => {
            assert_eq!(*kind, expected_kind);
            assert!(delta_h.is_finite());
            assert!(acceptance_probability.is_finite());
        }
        UpdateOutcome::Heatbath { .. } => panic!("unexpected heatbath fallback"),
    }
}

fn chiral_source(every: usize, sources: usize, flavors: usize) -> String {
    format!(
        r#"[[measurements]]
every = {every}
kind = "chiral_staggered"
mass = 0.17
boundary = [1, 1, 1, -1]
sources = {sources}
flavors = {flavors}
[measurements.solver]
tolerance = 1e-12
max_iterations = 1000
"#
    )
}

#[test]
fn deterministic_hmc_rejection_is_completed_and_recorded() {
    let source = BASE.replace("step_size = 0.0001", "step_size = 1.0");
    let report = run_lqcd(&Params::from_toml(&source).expect("rejection schema"))
        .expect("a rejected proposal is a successful update");
    assert_eq!(report.completed_updates, 1);
    assert_eq!(report.accepted_updates, 0);
    assert_eq!(report.rejected_updates, 1);
    assert!(matches!(
        report.updates.as_slice(),
        [latticeqcd::UpdateRecord {
            outcome: UpdateOutcome::Hmc {
                kind: latticeqcd::UpdateKind::QuenchedHmc,
                accepted: false,
                ..
            },
            ..
        }]
    ));
}

fn unique_directory(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", process::id()))
}
