#![cfg(feature = "fermions")]

use dirac_operators::{FermionBoundary, SolverMethod, SolverParams, StaggeredDirac};
use gaugefields::{cold_su3, CpuEvolutionContext, LatticeShape4, ReproducibleRng};
use measurements::fermions::{pion_correlator, stochastic_chiral_condensate};
use serde_json::{json, Value};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use tenferro_cpu::CpuBackend;

const NC: usize = 3;
const SOURCES: usize = 2;
const NV: usize = 16;
const BLOCKS: usize = 4;
const MEASUREMENTS: usize = 16;
const THERMALIZATION: usize = 4;
const TOTAL_TRAJECTORIES: usize = THERMALIZATION + MEASUREMENTS;
const STEP_SIZE: f64 = 0.01;
const MD_STEPS: usize = 2;
const BETA: f64 = 5.7;
const MASS: f64 = 0.5;
const NF_OVER_FOUR: f64 = 0.5;
const TOLERANCE: f64 = 1.0e-24;
const MAX_ITERATIONS: usize = 2_000;
const SIX_SIGMA: f64 = 6.0;

// This is deliberately a different, continuous Rust stream from the Julia
// per-trajectory seeds recorded by the fixture.
const RUST_UPDATE_STATE: [u64; 4] = [
    0x5048345f55504431,
    0x5048345f55504432,
    0x5048345f55504433,
    0x5048345f55504434,
];
const RUST_SOURCE_STATE: [u64; 4] = [
    0x5048345f53524331,
    0x5048345f53524332,
    0x5048345f53524333,
    0x5048345f53524334,
];

type TestResult<T> = Result<T, Box<dyn Error>>;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/fermion_measurements_phase4_ensemble")
}

fn metadata() -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(
        fixture_dir().join("metadata.json"),
    )?)?)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_error(block_means: &[f64]) -> f64 {
    let average = mean(block_means);
    let variance = block_means
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (block_means.len() - 1) as f64;
    (variance / block_means.len() as f64).sqrt()
}

fn assert_close(label: &str, actual: f64, expected: f64) {
    let residual = (actual - expected).abs();
    assert!(
        residual <= 2.0e-15,
        "{label}: actual={actual:.17e}, expected={expected:.17e}, residual={residual:.17e}"
    );
}

fn normalized_difference(difference: f64, combined_standard_error: f64) -> f64 {
    if combined_standard_error == 0.0 {
        if difference == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        difference / combined_standard_error
    }
}

fn as_usize(value: &Value, field: &str) -> usize {
    value[field]
        .as_u64()
        .unwrap_or_else(|| panic!("metadata field {field} is not an unsigned integer")) as usize
}

fn as_finite(value: &Value, field: &str) -> f64 {
    let value = value[field]
        .as_f64()
        .unwrap_or_else(|| panic!("metadata field {field} is not a number"));
    assert!(value.is_finite(), "metadata field {field} is not finite");
    value
}

fn as_string<'a>(value: &'a Value, field: &str) -> &'a str {
    value[field]
        .as_str()
        .unwrap_or_else(|| panic!("metadata field {field} is not a string"))
}

fn consume_metadata(meta: &Value) {
    assert_eq!(meta["schema"], "fermion_measurements_phase4_ensemble.v1");
    assert_eq!(meta["lattice"], json!([2, 2, 2, 2]));
    assert_eq!(meta["nc"], NC);
    assert_eq!(meta["beta"], BETA);
    assert_eq!(meta["mass"], MASS);
    assert_eq!(meta["nf"], 2);
    assert_eq!(meta["boundaries"], json!([1, 1, 1, -1]));

    let bounds = &meta["spectral_bounds"];
    assert_eq!(bounds["claimed_lower"], 0.0004);
    assert_eq!(bounds["claimed_upper"], 64.0);
    assert_eq!(bounds["coefficient_interval"], json!([0.0004, 64.0]));

    let tables = &meta["rhmc_tables"];
    let expected_tables = [
        (
            "refresh",
            "coeffs_18",
            0.125,
            15,
            "x^(+1/8) degree-15 refresh",
        ),
        (
            "action",
            "coeffs_m18",
            -0.125,
            15,
            "x^(-1/8) degree-15 action",
        ),
        (
            "md_force",
            "coeffs_m14_n10",
            -0.25,
            10,
            "x^(-1/4) degree-10 MD force",
        ),
    ];
    for (key, name, power, degree, role) in expected_tables {
        let table = &tables[key];
        assert_eq!(as_string(table, "name"), name);
        assert_eq!(as_string(table, "role"), role);
        assert_eq!(table["power"].as_f64(), Some(power));
        assert_eq!(as_usize(table, "degree"), degree);
        let alpha0 = as_finite(table, "alpha0");
        assert!(alpha0.is_finite());
        let alpha = table["alpha"].as_array().expect("alpha array");
        let beta = table["beta"].as_array().expect("beta array");
        let bits = table["bits"].as_array().expect("bits array");
        assert_eq!(alpha.len(), degree);
        assert_eq!(beta.len(), degree);
        assert_eq!(bits.len(), 1 + 2 * degree);
        for value in alpha.iter().chain(beta) {
            assert!(value.as_f64().expect("coefficient number").is_finite());
        }
        let coefficients = std::iter::once(alpha0)
            .chain(alpha.iter().map(|value| value.as_f64().unwrap()))
            .chain(beta.iter().map(|value| value.as_f64().unwrap()));
        for (coefficient, bit) in coefficients.zip(bits) {
            let bit = bit.as_str().expect("coefficient bit string");
            assert!(bit.starts_with("0x") && bit.len() == 18);
            let expected_bits = u64::from_str_radix(&bit[2..], 16).expect("valid coefficient bits");
            assert!(coefficient.to_bits().abs_diff(expected_bits) <= 1);
        }
    }
    let grid = &meta["scalar_log_grid"];
    assert_eq!(as_usize(grid, "points"), 4097);
    assert_eq!(
        as_string(grid, "spacing"),
        "lambda_low*exp(log(lambda_high/lambda_low)*i/(points-1)); endpoints exact"
    );
    for (key, expected_error) in [
        ("refresh", 2.505791796281187e-9),
        ("action", 3.9620045022559225e-9),
        ("md_force", 1.5595609319518644e-5),
    ] {
        assert_close(
            "RHMC scalar-grid error",
            grid["max_abs_error"][key].as_f64().unwrap(),
            expected_error,
        );
        assert_eq!(grid["powers"][key].as_f64(), tables[key]["power"].as_f64());
    }

    let solver = &meta["solver_parameters"];
    assert_eq!(solver["absolute_squared_tolerance"], TOLERANCE);
    assert_eq!(as_usize(solver, "max_iterations"), MAX_ITERATIONS);
    assert_eq!(as_string(solver, "trajectory_method"), "multi_shift_cg");
    assert_eq!(as_string(solver, "measurement_method"), "bicgstab");
    assert_eq!(
        solver["julia_keys"],
        json!([
            "Dirac_operator",
            "mass",
            "verbose_level",
            "boundarycondition",
            "eps",
            "MaxCGstep",
            "method_CG"
        ])
    );
    assert_eq!(
        as_string(solver, "rust_solver"),
        "checked true-residual solvers"
    );
    assert_eq!(
        as_string(solver, "true_residual_gate"),
        "fresh sum(abs2, b-D*x) <= 1e-24"
    );

    let schedule = &meta["schedule"];
    assert_eq!(as_string(schedule, "initial_condition"), "cold");
    assert_eq!(
        as_usize(schedule, "thermalization_trajectories"),
        THERMALIZATION
    );
    assert_eq!(as_usize(schedule, "measurements"), MEASUREMENTS);
    assert_eq!(as_usize(schedule, "measurement_interval"), 1);
    assert_eq!(as_usize(schedule, "blocks"), BLOCKS);
    assert_eq!(as_usize(schedule, "measurements_per_block"), 4);
    assert_eq!(as_usize(schedule, "total_trajectories"), TOTAL_TRAJECTORIES);
    assert_eq!(schedule["step_size"], STEP_SIZE);
    assert_eq!(as_usize(schedule, "md_steps"), MD_STEPS);
    assert_eq!(as_string(schedule, "measurement"), "after each trajectory after thermalization; rejected links are restored before measurement");
    assert_eq!(
        as_string(schedule, "integrator"),
        "U <- exp((dt/2)P)U; P <- P - dt*(gauge_force/NC + fermion_force); U <- exp((dt/2)P)U"
    );
    assert_eq!(as_string(schedule, "acceptance"), "unconditional rand() draw; accept iff rand() <= min(1, exp(-delta_h)); rejected links roll back");

    let streams = &meta["streams"];
    assert_eq!(as_string(streams, "independence"), "Julia and Rust start from independent cold configurations and never exchange configurations or measurement payloads");
    let julia = &streams["julia"];
    let update_seeds = julia["update_seeds"]
        .as_array()
        .expect("Julia update seeds");
    assert_eq!(update_seeds.len(), TOTAL_TRAJECTORIES);
    for seed in update_seeds {
        assert!(seed.as_u64().is_some());
    }
    let source_seeds = julia["source_seeds"]
        .as_array()
        .expect("Julia source seeds");
    assert_eq!(source_seeds.len(), MEASUREMENTS);
    for seeds in source_seeds {
        assert_eq!(seeds.as_array().unwrap().len(), SOURCES);
        assert!(seeds
            .as_array()
            .unwrap()
            .iter()
            .all(|seed| seed.as_u64().is_some()));
    }
    assert_eq!(as_string(julia, "update_policy"), "Random.seed!(trajectory_seed) before each pinned momentum/pseudofermion/acceptance sequence");
    assert_eq!(as_string(julia, "source_policy"), "Random.seed!(source_seed) immediately before each explicit source; codes are recorded per source");
    assert_eq!(
        as_string(julia, "source_order"),
        "Julia eachindex order [color, x, y, z, t, component] with color fastest; one component"
    );
    assert_eq!(
        as_string(julia, "z4_mapping"),
        "code 0 -> 1, code 1 -> i, code 2 -> -1, code 3 -> -i"
    );
    let rust = &streams["rust"];
    assert_eq!(rust["update_state"], json!(RUST_UPDATE_STATE));
    assert_eq!(rust["source_state"], json!(RUST_SOURCE_STATE));
    assert_eq!(
        as_string(rust, "update_policy"),
        "one continuous ReproducibleRng xoshiro256++ stream owned by the Rust HMC test"
    );
    assert_eq!(
        as_string(rust, "source_policy"),
        "one separate continuous ReproducibleRng xoshiro256++ stream; two sources per measurement"
    );
    assert_eq!(
        as_string(rust, "configuration_policy"),
        "cold Rust configuration created independently; no Julia configuration is read"
    );
    assert_eq!(
        as_string(rust, "z4_mapping"),
        "raw word & 3: 0 -> 1, 1 -> i, 2 -> -1, 3 -> -i"
    );
    assert_ne!(RUST_UPDATE_STATE, RUST_SOURCE_STATE);

    let normalization = &meta["normalization"];
    assert_eq!(
        as_string(normalization, "pion"),
        "sum over all three point sources and sink color of abs2(G_sink,source); no normalization"
    );
    assert_eq!(
        as_string(normalization, "chiral"),
        "(Nf/4) * mean(Re(dot(r, D^-1*r))) / NV"
    );
    assert_eq!(normalization["nv"], NV);
    assert_eq!(normalization["nf_over_four"], NF_OVER_FOUR);
    assert_eq!(
        as_string(normalization, "standard_error"),
        "sample standard deviation of four consecutive block means divided by sqrt(4)"
    );

    let contraction = &meta["contraction"];
    assert_eq!(as_string(contraction, "pion"), "corrected full Frobenius contraction over all sink colors for each of the three color point sources");
    assert_eq!(contraction["high_level_pion_reconstruction_called"], false);
    assert_eq!(contraction["staggered_sign"], "none");
    assert_eq!(
        as_string(contraction, "legacy_issue"),
        "Issue #29 source-diagonal duplication is not used"
    );
    assert_eq!(contraction["normalization"], normalization["pion"]);

    let trajectories = meta["trajectories"].as_array().expect("trajectory records");
    assert_eq!(trajectories.len(), TOTAL_TRAJECTORIES);
    let julia_update_seeds = meta["streams"]["julia"]["update_seeds"].as_array().unwrap();
    for (index, trajectory) in trajectories.iter().enumerate() {
        assert_eq!(as_usize(trajectory, "trajectory"), index + 1);
        assert_eq!(trajectory["seed"], julia_update_seeds[index]);
        let expected_phase = if index < THERMALIZATION {
            "thermalization"
        } else {
            "measurement"
        };
        assert_eq!(as_string(trajectory, "phase"), expected_phase);
        assert!(trajectory["seed"].as_u64().is_some());
        for field in ["delta_h", "acceptance_probability", "acceptance_uniform"] {
            assert!(as_finite(trajectory, field).is_finite());
        }
        assert!((0.0..=1.0).contains(&trajectory["acceptance_probability"].as_f64().unwrap()));
        assert!((0.0..=1.0).contains(&trajectory["acceptance_uniform"].as_f64().unwrap()));
        assert!(trajectory["accepted"].as_bool().is_some());
    }

    let measurements = meta["measurements"].as_array().expect("measurements");
    assert_eq!(measurements.len(), MEASUREMENTS);
    for (index, measurement) in measurements.iter().enumerate() {
        assert_eq!(as_usize(measurement, "measurement"), index + 1);
        assert_eq!(
            as_usize(measurement, "trajectory"),
            THERMALIZATION + index + 1
        );
        assert_eq!(as_usize(measurement, "block"), index / 4 + 1);
        assert_eq!(as_usize(measurement, "trajectory_in_block"), index % 4 + 1);
        let pion = measurement["pion"].as_array().expect("pion values");
        assert_eq!(pion.len(), 2);
        assert!(pion.iter().all(|value| value.as_f64().unwrap().is_finite()));
        let chiral_sources = measurement["chiral_source_values"]
            .as_array()
            .expect("chiral source values");
        assert_eq!(chiral_sources.len(), SOURCES);
        assert!(chiral_sources
            .iter()
            .all(|value| value.as_f64().unwrap().is_finite()));
        assert_close(
            "Julia per-configuration chiral mean",
            as_finite(measurement, "chiral"),
            mean(
                &chiral_sources
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .collect::<Vec<_>>(),
            ),
        );
        assert_eq!(
            measurement["source_seeds"],
            meta["streams"]["julia"]["source_seeds"][index]
        );
        let codes = measurement["source_codes"]
            .as_array()
            .expect("source codes");
        assert_eq!(codes.len(), SOURCES);
        for source in codes {
            let source = source.as_array().expect("source code array");
            assert_eq!(source.len(), NC * NV);
            assert!(source.iter().all(|code| code.as_u64().unwrap() < 4));
        }
    }

    let blocks = meta["blocks"].as_array().expect("blocks");
    assert_eq!(blocks.len(), BLOCKS);
    let mut computed_pion_blocks = Vec::with_capacity(BLOCKS);
    let mut computed_chiral_blocks = Vec::with_capacity(BLOCKS);
    let mut computed_delta_h_blocks = Vec::with_capacity(BLOCKS);
    for (index, block) in blocks.iter().enumerate() {
        assert_eq!(as_usize(block, "block"), index + 1);
        assert_eq!(
            block["measurements"],
            json!([index * 4 + 1, index * 4 + 2, index * 4 + 3, index * 4 + 4])
        );
        let records = &measurements[index * 4..index * 4 + 4];
        let pion_mean = (0..2)
            .map(|timeslice| {
                mean(
                    &records
                        .iter()
                        .map(|record| record["pion"][timeslice].as_f64().unwrap())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let chiral_mean = mean(
            &records
                .iter()
                .map(|record| record["chiral"].as_f64().unwrap())
                .collect::<Vec<_>>(),
        );
        let delta_h_mean = mean(
            &trajectories[THERMALIZATION + index * 4..THERMALIZATION + index * 4 + 4]
                .iter()
                .map(|record| record["delta_h"].as_f64().unwrap())
                .collect::<Vec<_>>(),
        );
        computed_pion_blocks.push(pion_mean.clone());
        computed_chiral_blocks.push(chiral_mean);
        computed_delta_h_blocks.push(delta_h_mean);
        let pion = block["pion_mean"].as_array().expect("block pion mean");
        assert_eq!(pion.len(), 2);
        for timeslice in 0..2 {
            assert_close(
                "Julia block pion mean",
                pion[timeslice].as_f64().unwrap(),
                pion_mean[timeslice],
            );
        }
        assert_close(
            "Julia block chiral mean",
            as_finite(block, "chiral_mean"),
            chiral_mean,
        );
        assert_close(
            "Julia block delta-H mean",
            as_finite(block, "delta_h_mean"),
            delta_h_mean,
        );
        let accepted_in_block = trajectories
            [THERMALIZATION + index * 4..THERMALIZATION + index * 4 + 4]
            .iter()
            .filter(|record| record["accepted"].as_bool() == Some(true))
            .count();
        assert_eq!(as_usize(block, "accepted"), accepted_in_block);
        assert_eq!(as_usize(block, "total"), 4);
    }

    let statistics = &meta["statistics"];
    let pion_summary = &statistics["pion"];
    let pion_block_means = pion_summary["block_means"].as_array().unwrap();
    assert_eq!(pion_block_means.len(), BLOCKS);
    for block in 0..BLOCKS {
        for timeslice in 0..2 {
            assert_close(
                "Julia statistical pion block mean",
                pion_block_means[block][timeslice].as_f64().unwrap(),
                computed_pion_blocks[block][timeslice],
            );
        }
    }
    for timeslice in 0..2 {
        let values = computed_pion_blocks
            .iter()
            .map(|block| block[timeslice])
            .collect::<Vec<_>>();
        assert_close(
            "Julia pion mean",
            pion_summary["mean"][timeslice].as_f64().unwrap(),
            mean(&values),
        );
        assert_close(
            "Julia pion standard error",
            pion_summary["standard_error"][timeslice].as_f64().unwrap(),
            standard_error(&values),
        );
    }
    let chiral_summary = &statistics["chiral"];
    let chiral_block_means = chiral_summary["block_means"].as_array().unwrap();
    assert_eq!(chiral_block_means.len(), BLOCKS);
    for block in 0..BLOCKS {
        assert_close(
            "Julia statistical chiral block mean",
            chiral_block_means[block].as_f64().unwrap(),
            computed_chiral_blocks[block],
        );
    }
    assert_close(
        "Julia chiral mean",
        chiral_summary["mean"].as_f64().unwrap(),
        mean(&computed_chiral_blocks),
    );
    assert_close(
        "Julia chiral standard error",
        chiral_summary["standard_error"].as_f64().unwrap(),
        standard_error(&computed_chiral_blocks),
    );
    let delta_summary = &statistics["delta_h"];
    let delta_block_means = delta_summary["block_means"].as_array().unwrap();
    assert_eq!(delta_block_means.len(), BLOCKS);
    for block in 0..BLOCKS {
        assert_close(
            "Julia statistical delta-H block mean",
            delta_block_means[block].as_f64().unwrap(),
            computed_delta_h_blocks[block],
        );
    }
    assert_close(
        "Julia delta-H mean",
        delta_summary["mean"].as_f64().unwrap(),
        mean(&computed_delta_h_blocks),
    );
    assert_close(
        "Julia delta-H standard error",
        delta_summary["standard_error"].as_f64().unwrap(),
        standard_error(&computed_delta_h_blocks),
    );
    assert_close(
        "Julia duplicate delta-H mean",
        statistics["mean_delta_h"].as_f64().unwrap(),
        mean(&computed_delta_h_blocks),
    );
    let acceptance = &statistics["acceptance"];
    let measured_accepted = trajectories[THERMALIZATION..]
        .iter()
        .filter(|record| record["accepted"].as_bool() == Some(true))
        .count();
    assert_eq!(as_usize(acceptance, "accepted"), measured_accepted);
    assert_eq!(
        as_usize(acceptance, "rejected"),
        MEASUREMENTS - measured_accepted
    );
    assert_eq!(
        as_usize(acceptance, "accepted") + as_usize(acceptance, "rejected"),
        MEASUREMENTS
    );
    assert_eq!(as_usize(acceptance, "total"), MEASUREMENTS);
    assert_close(
        "Julia acceptance rate",
        acceptance["rate"].as_f64().unwrap(),
        measured_accepted as f64 / MEASUREMENTS as f64,
    );

    let burn_in = &meta["burn_in_summary"];
    assert_eq!(burn_in["delta_h"].as_array().unwrap().len(), THERMALIZATION);
    assert_eq!(
        burn_in["accepted"].as_array().unwrap().len(),
        THERMALIZATION
    );
    assert!(burn_in["delta_h"]
        .as_array()
        .unwrap()
        .iter()
        .all(|value| value.as_f64().unwrap().is_finite()));
    assert!(burn_in["accepted"]
        .as_array()
        .unwrap()
        .iter()
        .all(|value| value.as_bool().is_some()));
    for index in 0..THERMALIZATION {
        assert_close(
            "Julia burn-in delta-H",
            burn_in["delta_h"][index].as_f64().unwrap(),
            trajectories[index]["delta_h"].as_f64().unwrap(),
        );
        assert_eq!(burn_in["accepted"][index], trajectories[index]["accepted"]);
    }

    let provenance = &meta["provenance"];
    assert_eq!(
        provenance["julia"],
        json!({"version": "1.12.5", "source_commit": "5fe89b8ddc166260bfcd4a195b305aff0ccad686"})
    );
    for (key, package, version, commit) in [
        (
            "gaugefields_jl",
            "Gaugefields.jl",
            "0.7.2",
            "9e5719970770f4497405a856315c90bef7f74449",
        ),
        (
            "latticediracoperators_jl",
            "LatticeDiracOperators.jl",
            "0.6.4",
            "bdef628184597815ba3e0cddf2536df767e78a02",
        ),
        (
            "wilsonloop_jl",
            "Wilsonloop.jl",
            "0.1.5",
            "e1a617fdedb19b785f89bdeb13c30e53b20743a7",
        ),
        (
            "qcdmeasurements_jl",
            "QCDMeasurements.jl",
            "0.2.13",
            "9e04c37bbd68712cf7a749ae5aff10eb6aae4566",
        ),
    ] {
        assert_eq!(provenance[key]["package"], package);
        assert_eq!(provenance[key]["version"], version);
        assert_eq!(provenance[key]["commit"], commit);
        assert_eq!(provenance[key]["clean"], true);
    }

    assert_eq!(
        meta["source_functions"],
        json!([
            "StaggeredFermiAction",
            "FermiAction",
            "gauss_sampling_in_action!",
            "sample_pseudofermions!",
            "evaluate_FermiAction",
            "calc_UdSfdU!",
            "shiftedcg",
            "gauss_distribution!",
            "gauss_distribution_fermion!",
            "Staggered_Dirac_operator",
            "solve_DinvX!",
            "GaugeAction",
            "make_loops_fromname",
            "Traceless_antihermitian_add!",
            "exptU!",
            "Random.seed!",
            "rand",
            "QCDMeasurements pion/chiral conventions only"
        ])
    );
    let source_urls = meta["source_urls"].as_array().expect("source URLs");
    assert_eq!(source_urls.len(), 11);
    for url in source_urls {
        assert!(url.as_str().unwrap().starts_with("https://github.com/"));
        assert!(url.as_str().unwrap().contains("/blob/"));
    }

    let layout = &meta["layout"];
    assert_eq!(as_string(layout, "julia_shape"), "[3,NX,NY,NZ,NT,1]");
    assert_eq!(as_string(layout, "rust_shape"), "[3,1,NX,NY,NZ,NT]");
    assert_eq!(layout["permutation"], json!([1, 6, 2, 3, 4, 5]));
    assert_eq!(as_string(layout, "site_order"), "x fastest");
    assert_eq!(
        as_string(layout, "source_site"),
        "(1,1,1,1) Julia / (0,0,0,0) Rust"
    );
    assert_eq!(
        as_string(layout, "source_codes"),
        "NC*NV canonical codes, color fastest"
    );
    assert_eq!(
        as_string(layout, "configuration"),
        "Julia and Rust each construct cold_su3 independently"
    );

    let issues = meta["upstream_issues"].as_array().expect("upstream issues");
    assert_eq!(issues.len(), 3);
    for issue in issues {
        assert!(issue["number"].as_u64().is_some());
        assert!(issue["package"].as_str().is_some());
        assert!(issue["revision"].as_str().is_some());
        assert!(issue["function"].as_str().is_some());
        assert!(issue["detail"].as_str().is_some());
        assert!(issue["rust_decision"].as_str().is_some());
    }
    assert_eq!(issues[0]["number"], 27);
    assert_eq!(issues[1]["number"], 29);
    assert_eq!(issues[2]["number"], 30);

    let criterion = &meta["criterion"];
    assert_eq!(criterion["sigma_multiplier"], SIX_SIGMA);
    assert_eq!(
        as_string(criterion, "formula"),
        "abs(mean_rust - mean_julia) <= 6 * sqrt(se_rust^2 + se_julia^2)"
    );
    assert_eq!(
        as_string(criterion, "standard_error"),
        "sample standard deviation of four consecutive block means divided by sqrt(4)"
    );
    assert_eq!(as_string(criterion, "zero_combined_standard_error"), "combined SE == 0 requires exact zero difference; normalized reporting does not divide by zero");
    assert_eq!(
        criterion["metrics"],
        json!(["pion each temporal timeslice", "stochastic chiral scalar"])
    );
    assert_eq!(criterion["rust_results_recorded_in_julia_metadata"], false);

    let comparison = &meta["comparison"];
    assert_eq!(comparison["gate"], SIX_SIGMA);
    assert_eq!(comparison["rust_results_recorded"], false);
    assert_eq!(as_string(comparison, "scope"), "only Julia metadata is generated; Rust independently computes all Rust means, errors, and normalized differences in the integration test");

    let generator = &meta["generator"];
    assert_eq!(as_string(generator, "script"), "fixtures/generate.jl");
    assert_eq!(
        as_string(generator, "mode"),
        "fermion_measurements_phase4_ensemble"
    );
    assert_eq!(as_string(generator, "julia_version"), "1.12.5");
    assert_eq!(
        as_string(generator, "randomness"),
        "Julia-only deterministic draws; Rust stream states are predeclared constants, not generated results or configurations"
    );
    assert_eq!(generator["files"], json!(["metadata.json"]));
    let mut actual_files = fs::read_dir(fixture_dir())
        .expect("fixture directory")
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    actual_files.sort();
    assert_eq!(actual_files, ["metadata.json"]);
    assert_eq!(
        generator["complete_tree_hash_scope"],
        "metadata.json plus every declared file"
    );

    assert!(!meta.as_object().unwrap().contains_key("rust_results"));
}

fn consume_solver_reports(result: &measurements::fermions::PionCorrelator) {
    assert_eq!(result.solver_reports.len(), NC);
    for report in &result.solver_reports {
        assert_eq!(report.method, SolverMethod::BiCgStab);
        assert!(report.iterations <= report.maximum_iterations);
        assert!(report.recursive_residual_squared.is_finite());
        assert!(report.initial_residual_squared.is_finite());
        assert!(report.true_residual_squared.is_finite());
        assert!(report.tolerance <= TOLERANCE);
    }
}

fn consume_chiral_solver_reports(result: &measurements::fermions::ChiralCondensate) {
    assert_eq!(result.source_values.len(), SOURCES);
    assert_eq!(result.solver_reports.len(), SOURCES);
    assert!(result.value.is_finite());
    for report in &result.solver_reports {
        assert_eq!(report.method, SolverMethod::BiCgStab);
        assert!(report.iterations <= report.maximum_iterations);
        assert!(report.true_residual_squared.is_finite());
        assert!(report.tolerance <= TOLERANCE);
    }
}

#[test]
fn zero_combined_standard_error_is_reported_without_division() {
    assert_eq!(normalized_difference(0.0, 0.0), 0.0);
    assert!(normalized_difference(1.0, 0.0).is_infinite());
}

#[test]
fn phase4_ensemble_matches_independent_rust_stream() -> TestResult<()> {
    let meta = metadata()?;
    consume_metadata(&meta);

    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let boundary = FermionBoundary::new([1, 1, 1, -1])?;
    let solver = SolverParams::new(TOLERANCE, MAX_ITERATIONS)?;
    let params = dirac_operators::StaggeredHmcParams::new(
        BETA, MASS, STEP_SIZE, MD_STEPS, boundary, 0.0004, 64.0, solver,
    )?;
    let mut links = cold_su3(lattice)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let mut update_rng = ReproducibleRng::from_state(
        meta["streams"]["rust"]["update_state"]
            .as_array()
            .unwrap()
            .iter()
            .map(|word| word.as_u64().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    )?;
    let mut source_rng = ReproducibleRng::from_state(
        meta["streams"]["rust"]["source_state"]
            .as_array()
            .unwrap()
            .iter()
            .map(|word| word.as_u64().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    )?;

    let mut pion_values = Vec::with_capacity(MEASUREMENTS);
    let mut chiral_values = Vec::with_capacity(MEASUREMENTS);
    let mut delta_h_values = Vec::with_capacity(MEASUREMENTS);
    let mut accepted = 0usize;
    for trajectory in 1..=TOTAL_TRAJECTORIES {
        let outcome = dirac_operators::staggered_hmc_update(
            &mut context,
            &mut links,
            params,
            &mut update_rng,
        )?;
        assert!(outcome.delta_h.is_finite());
        assert!(outcome.acceptance_probability.is_finite());
        if outcome.accepted {
            accepted += 1;
        }
        println!(
            "Rust trajectory {trajectory:02}: accepted={} delta_h={:.17e} acceptance_probability={:.17e}",
            outcome.accepted, outcome.delta_h, outcome.acceptance_probability
        );
        if trajectory <= THERMALIZATION {
            continue;
        }
        let operator = StaggeredDirac::with_boundary(&links, MASS, boundary)?;
        let pion = pion_correlator(&operator, solver)?;
        consume_solver_reports(&pion);
        let chiral = stochastic_chiral_condensate(
            &operator,
            NF_OVER_FOUR,
            SOURCES,
            solver,
            &mut source_rng,
        )?;
        consume_chiral_solver_reports(&chiral);
        println!(
            "Rust measurement {:02}: pion={:?} chiral={:.17e}",
            trajectory - THERMALIZATION,
            pion.values,
            chiral.value
        );
        pion_values.push(pion.values);
        chiral_values.push(chiral.value);
        delta_h_values.push(outcome.delta_h);
    }
    assert_eq!(
        accepted,
        meta["statistics"]["acceptance"]["accepted"]
            .as_u64()
            .unwrap() as usize
            + meta["burn_in_summary"]["accepted"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|value| value.as_bool() == Some(true))
                .count()
    );
    assert_eq!(pion_values.len(), MEASUREMENTS);
    assert_eq!(chiral_values.len(), MEASUREMENTS);
    assert_eq!(delta_h_values.len(), MEASUREMENTS);

    let rust_pion_blocks = (0..BLOCKS)
        .map(|block| {
            (0..2)
                .map(|timeslice| {
                    mean(
                        &(0..4)
                            .map(|index| pion_values[block * 4 + index][timeslice])
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rust_chiral_blocks = (0..BLOCKS)
        .map(|block| mean(&chiral_values[block * 4..block * 4 + 4]))
        .collect::<Vec<_>>();
    let rust_delta_h_blocks = (0..BLOCKS)
        .map(|block| mean(&delta_h_values[block * 4..block * 4 + 4]))
        .collect::<Vec<_>>();
    let rust_pion_mean = (0..2)
        .map(|timeslice| {
            mean(
                &rust_pion_blocks
                    .iter()
                    .map(|block| block[timeslice])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let rust_pion_se = (0..2)
        .map(|timeslice| {
            standard_error(
                &rust_pion_blocks
                    .iter()
                    .map(|block| block[timeslice])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let rust_chiral_mean = mean(&rust_chiral_blocks);
    let rust_chiral_se = standard_error(&rust_chiral_blocks);

    let julia_pion_mean = meta["statistics"]["pion"]["mean"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let julia_pion_se = meta["statistics"]["pion"]["standard_error"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let julia_chiral_mean = meta["statistics"]["chiral"]["mean"].as_f64().unwrap();
    let julia_chiral_se = meta["statistics"]["chiral"]["standard_error"]
        .as_f64()
        .unwrap();

    println!(
        "Julia pion blocks={:?}",
        meta["statistics"]["pion"]["block_means"]
    );
    println!("Rust pion blocks={rust_pion_blocks:?}");
    println!(
        "Julia chiral blocks={:?}",
        meta["statistics"]["chiral"]["block_means"]
    );
    println!("Rust chiral blocks={rust_chiral_blocks:?}");
    println!(
        "Julia delta-H blocks={:?}",
        meta["statistics"]["delta_h"]["block_means"]
    );
    println!("Rust delta-H blocks={rust_delta_h_blocks:?}");
    println!(
        "acceptance: Rust measured={}/{} ({:.6}), Julia measured={}/{} ({:.6})",
        accepted
            - meta["burn_in_summary"]["accepted"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|value| value.as_bool() == Some(true))
                .count(),
        MEASUREMENTS,
        (accepted
            - meta["burn_in_summary"]["accepted"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|value| value.as_bool() == Some(true))
                .count()) as f64
            / MEASUREMENTS as f64,
        meta["statistics"]["acceptance"]["accepted"],
        meta["statistics"]["acceptance"]["total"],
        meta["statistics"]["acceptance"]["rate"],
    );

    for timeslice in 0..2 {
        let difference = (rust_pion_mean[timeslice] - julia_pion_mean[timeslice]).abs();
        let combined_se =
            (rust_pion_se[timeslice].powi(2) + julia_pion_se[timeslice].powi(2)).sqrt();
        let normalized = normalized_difference(difference, combined_se);
        println!(
            "normalized pion t={timeslice}: difference={difference:.17e} combined_se={combined_se:.17e} ratio={normalized:.6e}"
        );
        assert!(difference <= SIX_SIGMA * combined_se);
    }
    let chiral_difference = (rust_chiral_mean - julia_chiral_mean).abs();
    let chiral_combined_se = (rust_chiral_se.powi(2) + julia_chiral_se.powi(2)).sqrt();
    let chiral_normalized = normalized_difference(chiral_difference, chiral_combined_se);
    println!(
        "normalized chiral: difference={chiral_difference:.17e} combined_se={chiral_combined_se:.17e} ratio={chiral_normalized:.6e}"
    );
    assert!(chiral_difference <= SIX_SIGMA * chiral_combined_se);

    for (name, values) in [
        ("Rust pion mean", rust_pion_mean.as_slice()),
        ("Rust pion SE", rust_pion_se.as_slice()),
    ] {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name} not finite"
        );
    }
    assert!(rust_chiral_mean.is_finite() && rust_chiral_se.is_finite());
    Ok(())
}
