use dirac_operators::{
    bicgstab, FermionBoundary, FermionField, SolverParams, StaggeredDirac, StaggeredHmcParams,
};
use gaugefields::{
    cold_su3, normalized_plaquette, CpuEvolutionContext, LatticeShape4, ReproducibleRng,
};
use num_complex::Complex64;
use serde_json::{json, Value};
use std::error::Error;
use std::fs;
use std::path::Path;
use tenferro_cpu::CpuBackend;

const HMC_STATE: [u64; 4] = [2026081901, 2026081902, 2026081903, 2026081904];
const SOURCE_STATE: [u64; 4] = [2026181902, 2026181903, 2026181904, 2026181905];
const NV: f64 = 16.0;
const NF_OVER_FOUR: f64 = 0.5;

type TestResult<T> = Result<T, Box<dyn Error>>;

fn metadata() -> TestResult<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/fermions_task_e_ensemble/metadata.json");
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn assert_metadata_contract(meta: &Value) {
    assert_eq!(meta["schema"], "fermions_task_e_ensemble.v1");
    assert_eq!(meta["lattice"], json!([2, 2, 2, 2]));
    assert_eq!(meta["nc"], 3);
    assert_eq!(meta["beta"], 5.7);
    assert_eq!(meta["mass"], 0.17);
    assert_eq!(meta["nf"], 2);
    assert_eq!(meta["boundaries"], json!([1, 1, 1, -1]));
    assert_eq!(
        meta["spectral_bounds"],
        json!({
            "claimed_lower": 0.0004,
            "claimed_upper": 64.0,
            "coefficient_interval": [0.0004, 64.0]
        })
    );
    assert_eq!(
        meta["degrees"],
        json!({"refresh": 15, "action": 15, "md_force": 10})
    );
    assert_eq!(
        meta["solver_parameters"],
        json!({
            "absolute_squared_tolerance": 1.0e-24,
            "max_iterations": 2000,
            "trajectory_method": "cg",
            "chiral_method": "bicg",
            "julia_keys": [
                "Dirac_operator", "mass", "verbose_level", "boundarycondition",
                "eps", "MaxCGstep", "method_CG"
            ]
        })
    );
    assert_eq!(
        meta["schedule"],
        json!({
            "initial_condition": "cold",
            "burn_in_trajectories": 4,
            "blocks": 3,
            "trajectories_per_block": 4,
            "measured_trajectories": 12,
            "dt": 0.001,
            "steps": 2,
            "measurement": "after each trajectory; rejected links are restored before measurement",
            "integrator": "U <- exp((dt/2)P)U; P <- P - dt*(gauge_force/NC + fermion_force); U <- exp((dt/2)P)U",
            "acceptance": "unconditional rand() draw; accept iff rand() <= min(1, exp(-delta_h)); rejected links roll back"
        })
    );
    assert_eq!(
        meta["normalization"],
        json!({
            "plaquette": "real(calculate_Plaquette(U,temp1,temp2)) / (6 * NV * NC)",
            "chiral_condensate": "(Nf/4) / NV * Re(dot(r, D^-1*r))",
            "nv": 16,
            "nf_over_four": 0.5,
            "standard_error": "sample_stddev(block_means) / sqrt(number_of_blocks)"
        })
    );
    assert_eq!(
        meta["source_generation"],
        json!({
            "sources_per_configuration": 2,
            "distribution": "canonical Z4 implemented in the fixture",
            "seed_call": "Random.seed!(source_seed) immediately before each source",
            "source_formula": "theta = rand(0:3)*pi/2; r = cos(theta) + im*sin(theta)"
        })
    );
    assert_eq!(
        meta["provenance"]["julia"],
        json!({"version": "1.12.5", "source_commit": "5fe89b8ddc166260bfcd4a195b305aff0ccad686"})
    );
    assert_eq!(
        meta["provenance"]["gaugefields_jl"],
        json!({
            "package": "Gaugefields.jl",
            "version": "0.7.2",
            "commit": "9e5719970770f4497405a856315c90bef7f74449",
            "clean": true
        })
    );
    assert_eq!(
        meta["provenance"]["latticediracoperators_jl"],
        json!({
            "package": "LatticeDiracOperators.jl",
            "version": "0.6.4",
            "commit": "bdef628184597815ba3e0cddf2536df767e78a02",
            "clean": true
        })
    );
    assert_eq!(
        meta["provenance"]["wilsonloop_jl"],
        json!({
            "package": "Wilsonloop.jl",
            "version": "0.1.5",
            "commit": "e1a617fdedb19b785f89bdeb13c30e53b20743a7",
            "clean": true
        })
    );
    assert_eq!(
        meta["provenance"]["qcdmeasurements_jl"],
        json!({
            "package": "QCDMeasurements.jl",
            "version": "0.2.13",
            "commit": "9e04c37bbd68712cf7a749ae5aff10eb6aae4566"
        })
    );
    assert_eq!(
        meta["upstream_issues"],
        json!([{
            "package": "LatticeDiracOperators.jl",
            "revision": "bdef628184597815ba3e0cddf2536df767e78a02",
            "function": "Z4_distribution_fermi!",
            "detail": "The pinned implementation uses theta=rand(0:3)*pi/4, not the canonical 2*pi/4 phase grid; this fixture avoids that biased source and implements canonical Z4 explicitly."
        }])
    );
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
            "solve_DinvX!",
            "GaugeAction",
            "make_loops_fromname",
            "calculate_Plaquette",
            "Traceless_antihermitian_add!",
            "exptU!",
            "Random.seed!",
            "rand"
        ])
    );
    assert_eq!(
        meta["source_urls"],
        json!([
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/action/StaggeredFermiAction.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/rhmc/rhmc.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/Diracoperators.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/src/AbstractFermions_4D.jl",
            "https://github.com/shinaoka/LatticeDiracOperators.jl/blob/bdef628184597815ba3e0cddf2536df767e78a02/test/wilsonhmc.jl",
            "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/action/GaugeActions.jl",
            "https://github.com/shinaoka/Gaugefields.jl/blob/9e5719970770f4497405a856315c90bef7f74449/src/4D/TA_gaugefields_4D_serial.jl",
            "https://github.com/akio-tomiya/Wilsonloop.jl/blob/e1a617fdedb19b785f89bdeb13c30e53b20743a7/src/Wilsonloop.jl",
            "https://github.com/akio-tomiya/QCDMeasurements.jl/blob/9e04c37bbd68712cf7a749ae5aff10eb6aae4566/src/measurements/measure_chiral_condensate.jl"
        ])
    );
    assert_eq!(
        meta["generator"],
        json!({
            "script": "fixtures/generate.jl",
            "mode": "fermions_task_e_ensemble",
            "files": ["metadata.json"],
            "randomness": "explicit Julia task-local seeds; no Rust or binary payloads"
        })
    );

    let seeds = &meta["seeds"];
    assert_eq!(seeds["master"], 2026081901_u64);
    assert_eq!(
        seeds["trajectory_seeds"]["burn_in"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        seeds["trajectory_seeds"]["measured"]
            .as_array()
            .unwrap()
            .len(),
        12
    );
    assert_eq!(seeds["source_seeds"].as_array().unwrap().len(), 12);
    assert_eq!(
        seeds["stream_policy"],
        "Random.seed!(trajectory_seed) before each pinned momentum/pseudofermion/acceptance sequence; Random.seed!(source_seed) before each pinned Z4 source"
    );
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
    variance.sqrt() / (block_means.len() as f64).sqrt()
}

fn metadata_statistics(meta: &Value, name: &str) -> (f64, f64) {
    let block_means = meta["statistics"][name]["block_means"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let expected_mean = meta["statistics"][name]["mean"].as_f64().unwrap();
    let expected_se = meta["statistics"][name]["standard_error"].as_f64().unwrap();
    assert!((mean(&block_means) - expected_mean).abs() < 1.0e-15);
    assert!((standard_error(&block_means) - expected_se).abs() < 1.0e-15);
    (expected_mean, expected_se)
}

fn canonical_z4_source(
    lattice: LatticeShape4,
    rng: &mut ReproducibleRng,
) -> TestResult<FermionField> {
    let mut values = Vec::with_capacity(3 * lattice.nv());
    for _ in 0..3 * lattice.nv() {
        // Deliberately use the canonical 2*pi/4 grid; metadata records the
        // pinned Julia pi/4 source defect but this independent estimator does not copy it.
        let phase = (4.0 * rng.open_unit_f64()) as usize;
        values.push(match phase {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!("open-unit mapping must produce four phases"),
        });
    }
    Ok(FermionField::from_vec_col_major(lattice, 1, values)?)
}

fn chiral_condensate(
    links: &gaugefields::GaugeLinks,
    params: StaggeredHmcParams,
    source_rng: &mut ReproducibleRng,
) -> TestResult<f64> {
    let lattice = links.lattice();
    let dirac = StaggeredDirac::with_boundary(links, params.mass(), params.action().boundary())?;
    let mut value = 0.0;
    for _ in 0..2 {
        let source = canonical_z4_source(lattice, source_rng)?;
        let mut solution = FermionField::zeros(lattice, 1)?;
        bicgstab(
            &mut solution,
            &dirac,
            &source,
            params.action().solver_params(),
        )?;
        value += NF_OVER_FOUR / NV * source.inner_product(&solution)?.re;
    }
    Ok(value / 2.0)
}

#[test]
fn task_e_ensemble_matches_the_independent_canonical_rust_estimator() -> TestResult<()> {
    let meta = metadata()?;
    assert_metadata_contract(&meta);

    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let boundary = FermionBoundary::new([1, 1, 1, -1])?;
    let solver = SolverParams::new(1.0e-24, 2_000)?;
    let params = StaggeredHmcParams::new(5.7, 0.17, 0.001, 2, boundary, 0.0004, 64.0, solver)?;
    let mut links = cold_su3(lattice)?;
    let mut evolution = CpuEvolutionContext::new(CpuBackend::new());
    let mut hmc_rng = ReproducibleRng::from_state(HMC_STATE)?;
    let mut source_rng = ReproducibleRng::from_state(SOURCE_STATE)?;

    let mut plaquettes = Vec::with_capacity(12);
    let mut chirals = Vec::with_capacity(12);
    let mut delta_h = Vec::with_capacity(12);
    let mut measured_accepted = 0usize;
    for trajectory in 0..16 {
        let outcome = dirac_operators::staggered_hmc_update(
            &mut evolution,
            &mut links,
            params,
            &mut hmc_rng,
        )?;
        assert!(outcome.delta_h.is_finite());
        if trajectory >= 4 {
            if outcome.accepted {
                measured_accepted += 1;
            }
            plaquettes.push(normalized_plaquette(&links)?);
            chirals.push(chiral_condensate(&links, params, &mut source_rng)?);
            delta_h.push(outcome.delta_h);
        }
    }

    let rust_plaquette_blocks = plaquettes.chunks_exact(4).map(mean).collect::<Vec<_>>();
    let rust_chiral_blocks = chirals.chunks_exact(4).map(mean).collect::<Vec<_>>();
    let rust_plaquette_mean = mean(&rust_plaquette_blocks);
    let rust_chiral_mean = mean(&rust_chiral_blocks);
    let rust_plaquette_se = standard_error(&rust_plaquette_blocks);
    let rust_chiral_se = standard_error(&rust_chiral_blocks);
    let (julia_plaquette_mean, julia_plaquette_se) = metadata_statistics(&meta, "plaquette");
    let (julia_chiral_mean, julia_chiral_se) = metadata_statistics(&meta, "chiral_condensate");
    let plaquette_difference = (rust_plaquette_mean - julia_plaquette_mean).abs();
    let chiral_difference = (rust_chiral_mean - julia_chiral_mean).abs();
    let plaquette_scale = (rust_plaquette_se.powi(2) + julia_plaquette_se.powi(2)).sqrt();
    let chiral_scale = (rust_chiral_se.powi(2) + julia_chiral_se.powi(2)).sqrt();

    println!(
        "Rust plaquette blocks={rust_plaquette_blocks:?} mean={rust_plaquette_mean:.17e} SE={rust_plaquette_se:.17e}"
    );
    println!(
        "Rust chiral blocks={rust_chiral_blocks:?} mean={rust_chiral_mean:.17e} SE={rust_chiral_se:.17e}"
    );
    println!(
        "normalized plaquette difference={plaquette_difference:.17e} ({:.6} SE_combined)",
        plaquette_difference / plaquette_scale
    );
    println!(
        "normalized chiral difference={chiral_difference:.17e} ({:.6} SE_combined)",
        chiral_difference / chiral_scale
    );
    println!(
        "mean measured dH={:.17e}, acceptance={measured_accepted}/12 ({:.6})",
        mean(&delta_h),
        measured_accepted as f64 / 12.0
    );

    assert!(plaquette_difference <= 6.0 * plaquette_scale);
    assert!(chiral_difference <= 6.0 * chiral_scale);
    Ok(())
}
