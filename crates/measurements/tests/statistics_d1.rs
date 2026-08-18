use gaugefields::{
    cold_su3, heatbath_sweep, normalized_plaquette, HeatbathParams, LatticeShape4, ReproducibleRng,
};
use measurements::{clover_topological_charge, polyakov_loop};
use serde_json::Value;
use std::{fs, path::Path};

const BETA: f64 = 5.7;
const JULIA_SEED: u64 = 2026081802;
const BURN_IN: usize = 512;
const BLOCKS: usize = 32;
const SWEEPS_PER_BLOCK: usize = 32;
const MAX_ATTEMPTS: usize = 100_000;
const RUST_STATE: [u64; 4] = [
    0x2468_ace1_1357_9bdf,
    0x1111_2222_3333_4445,
    0x5555_6666_7777_8889,
    0x9999_aaaa_bbbb_cccd,
];
const OBSERVABLES: [&str; 6] = [
    "plaquette",
    "polyakov_real",
    "polyakov_imag",
    "polyakov_magnitude",
    "q",
    "q_squared",
];

#[derive(Debug)]
struct Statistics {
    block_means: Vec<f64>,
    mean: f64,
    variance: f64,
    standard_error: f64,
}

fn fixture_metadata() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/measurements_task_d1/metadata.json");
    serde_json::from_slice(&fs::read(path).expect("D1 metadata")).expect("valid D1 metadata")
}

fn summarize(block_means: Vec<f64>) -> Statistics {
    let mean = block_means.iter().sum::<f64>() / block_means.len() as f64;
    let variance = block_means
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (block_means.len() - 1) as f64;
    Statistics {
        standard_error: (variance / block_means.len() as f64).sqrt(),
        block_means,
        mean,
        variance,
    }
}

fn links_changed_from_cold(
    links: &gaugefields::GaugeLinks,
    cold: &gaugefields::GaugeLinks,
) -> bool {
    links
        .links()
        .iter()
        .zip(cold.links())
        .any(|(actual, expected)| {
            actual
                .typed()
                .host_data()
                .unwrap()
                .iter()
                .zip(expected.typed().host_data().unwrap())
                .any(|(actual, expected)| {
                    actual.re.to_bits() != expected.re.to_bits()
                        || actual.im.to_bits() != expected.im.to_bits()
                })
        })
}

fn run_rust_chain() -> Result<([Statistics; 6], bool), Box<dyn std::error::Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let cold = cold_su3(lattice)?;
    let mut links = cold_su3(lattice)?;
    let params = HeatbathParams::new(BETA, MAX_ATTEMPTS)?;
    let mut rng = ReproducibleRng::from_state(RUST_STATE)?;

    for _ in 0..BURN_IN {
        heatbath_sweep(&mut links, params, &mut rng)?;
    }

    let mut blocks: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(BLOCKS));
    for _ in 0..BLOCKS {
        let mut sums = [0.0; 6];
        for _ in 0..SWEEPS_PER_BLOCK {
            heatbath_sweep(&mut links, params, &mut rng)?;
            let polyakov = polyakov_loop(&links)?;
            let q = clover_topological_charge(&links)?;
            let values = [
                normalized_plaquette(&links)?,
                polyakov.re,
                polyakov.im,
                polyakov.norm(),
                q,
                q * q,
            ];
            for (sum, value) in sums.iter_mut().zip(values) {
                *sum += value;
            }
        }
        for (series, sum) in blocks.iter_mut().zip(sums) {
            series.push(sum / SWEEPS_PER_BLOCK as f64);
        }
    }

    let statistics = blocks.map(summarize);
    Ok((statistics, links_changed_from_cold(&links, &cold)))
}

fn julia_chain(metadata: &Value) -> &Value {
    metadata["expected_observables"]["ensemble"]["chains"]
        .as_array()
        .expect("D1 chains")
        .iter()
        .find(|chain| chain["beta"].as_f64() == Some(BETA))
        .expect("beta=5.7 D1 chain")
}

#[test]
fn beta57_statistics_match_independent_julia_chain() -> Result<(), Box<dyn std::error::Error>> {
    let metadata = fixture_metadata();
    assert_eq!(
        metadata["expected_observables"]["schema"],
        "measurements_task_d1.v1"
    );
    assert_eq!(
        metadata["expected_observables"]["ensemble"]["schema"],
        "measurements_task_d1_ensemble.v1"
    );
    assert_eq!(
        metadata["gaugefields_jl_commit"],
        "9e5719970770f4497405a856315c90bef7f74449"
    );
    assert_eq!(
        metadata["expected_observables"]["provenance"]["wilsonloop_jl"]["commit"],
        "e1a617fdedb19b785f89bdeb13c30e53b20743a7"
    );
    assert_eq!(
        metadata["expected_observables"]["provenance"]["qcdmeasurements_jl"]["commit"],
        "9e04c37bbd68712cf7a749ae5aff10eb6aae4566"
    );

    let schedule = &metadata["expected_observables"]["ensemble"]["schedule"];
    assert_eq!(schedule["initial_condition"], "cold");
    assert_eq!(schedule["burn_in_sweeps"], BURN_IN);
    assert_eq!(schedule["blocks"], BLOCKS);
    assert_eq!(schedule["sweeps_per_block"], SWEEPS_PER_BLOCK);
    assert_eq!(schedule["measurements"], BLOCKS * SWEEPS_PER_BLOCK);
    assert_eq!(schedule["max_attempts"], MAX_ATTEMPTS);

    let chain = julia_chain(&metadata);
    assert_eq!(chain["julia_seed"], JULIA_SEED);
    assert_eq!(chain["measurements"], BLOCKS * SWEEPS_PER_BLOCK);
    let (rust, changed) = run_rust_chain()?;
    assert!(changed, "measured field stayed cold");

    for (index, name) in OBSERVABLES.iter().enumerate() {
        let reference = &chain["observables"][*name];
        let julia_blocks: Vec<f64> = reference["block_means"]
            .as_array()
            .expect("Julia block means")
            .iter()
            .map(|value| value.as_f64().expect("finite Julia number"))
            .collect();
        assert_eq!(julia_blocks.len(), BLOCKS, "{name}");
        let julia_mean = reference["mean"].as_f64().expect("Julia mean");
        let julia_variance = reference["variance"].as_f64().expect("Julia variance");
        let julia_se = reference["standard_error"].as_f64().expect("Julia SE");
        let actual = &rust[index];
        assert!(julia_blocks.iter().all(|value| value.is_finite()), "{name}");
        assert!(julia_mean.is_finite() && julia_variance.is_finite() && julia_se.is_finite());
        assert!(
            actual.block_means.iter().all(|value| value.is_finite()),
            "{name}"
        );
        assert!(
            actual.mean.is_finite()
                && actual.variance.is_finite()
                && actual.standard_error.is_finite()
        );
        assert!(
            julia_variance > 0.0 && julia_se > 0.0,
            "{name} Julia variance"
        );
        assert!(
            actual.variance > 0.0 && actual.standard_error > 0.0,
            "{name} Rust variance"
        );

        let combined_se = (actual
            .standard_error
            .mul_add(actual.standard_error, julia_se * julia_se))
        .sqrt();
        let difference = (actual.mean - julia_mean).abs();
        let z = difference / combined_se;
        eprintln!(
            "{name}: julia={julia_mean:.16e} rust={:.16e} difference={difference:.16e} combined_se={combined_se:.16e} z={z:.16e}",
            actual.mean
        );
        assert!(z <= 6.0, "{name} z={z:.6} exceeds six combined SE");

        if *name == "q_squared" {
            let scale = julia_mean.abs().max(actual.mean.abs());
            let relative = difference / scale;
            eprintln!("{name}: relative={relative:.16e} ceiling=2.5e-1");
            assert!(
                scale > 0.0 && relative <= 0.25,
                "{name} relative={relative:.6e}"
            );
        }
    }

    assert!(rust[0].variance > 0.0, "plaquette block variance");
    assert!(rust[3].variance > 0.0, "Polyakov magnitude block variance");
    assert!(rust[5].variance > 0.0, "Q² block variance");
    Ok(())
}
