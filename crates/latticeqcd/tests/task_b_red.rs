use latticeqcd::{run_lqcd, Params};

const CONFIG: &str = r#"
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

[[measurements]]
every = 1
kind = "plaquette"
"#;

#[test]
fn task_b_frontend_runs_a_validated_cold_configuration() {
    let params = Params::from_toml(CONFIG).expect("valid Task B configuration");
    let report = run_lqcd(&params).expect("cold run succeeds");
    assert_eq!(report.requested_updates, 1);
    assert_eq!(report.completed_updates, 1);
    assert_eq!(report.measurements.len(), 1);
}

#[test]
fn task_b_rejects_unknown_fields_before_execution() {
    let physical = CONFIG.replace("beta = 5.7", "beta = 5.7\nunknown = 1");
    assert!(Params::from_toml(&physical).is_err());

    let initial = CONFIG.replace(
        "[initial]\nkind = \"cold\"",
        "[initial]\nkind = \"cold\"\nunknown = 1",
    );
    assert!(Params::from_toml(&initial).is_err());

    let fermions = CONFIG.replace(
        "[fermions]\nkind = \"quenched\"",
        "[fermions]\nkind = \"quenched\"\nunknown = 1",
    );
    assert!(Params::from_toml(&fermions).is_err());
}
