use gaugefields::{
    cold_su3, heatbath_sweep, normalized_plaquette, HeatbathParams, LatticeShape4, ReproducibleRng,
};
use serde_json::Value;
use std::{fs, path::Path};

const JULIA_COMMIT: &str = "9e5719970770f4497405a856315c90bef7f74449";
const JULIA_VERSION: &str = "0.7.2";
const BURN_IN: usize = 512;
const BLOCKS: usize = 32;
const SWEEPS_PER_BLOCK: usize = 32;
const MAX_ATTEMPTS: usize = 100_000;
const BETAS: [f64; 3] = [5.5, 5.7, 6.0];
const RUST_STATES: [[u64; 4]; 3] = [
    [
        0x1357_9bdf_2468_ace1,
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3211,
        0x0bad_f00d_cafe_beef,
    ],
    [
        0x2468_ace1_1357_9bdf,
        0x1111_2222_3333_4445,
        0x5555_6666_7777_8889,
        0x9999_aaaa_bbbb_cccd,
    ],
    [
        0xdead_beef_cafe_babe,
        0x1020_3040_5060_7081,
        0x8090_a0b0_c0d0_e0f1,
        0x3141_5926_5358_9794,
    ],
];

fn fixture_metadata() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/heatbath_statistics/metadata.json");
    serde_json::from_slice(&fs::read(path).expect("heatbath statistics fixture"))
        .expect("valid metadata")
}

fn assert_schedule(metadata: &Value) {
    assert_eq!(metadata["schema"], "heatbath_statistics.v1");
    assert_eq!(metadata["gaugefields_jl"]["version"], JULIA_VERSION);
    assert_eq!(metadata["gaugefields_jl"]["commit"], JULIA_COMMIT);
    assert_eq!(metadata["nc"], 3);
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["schedule"]["burn_in_sweeps"], BURN_IN);
    assert_eq!(metadata["schedule"]["blocks"], BLOCKS);
    assert_eq!(metadata["schedule"]["sweeps_per_block"], SWEEPS_PER_BLOCK);
    assert_eq!(
        metadata["schedule"]["measurements"],
        BLOCKS * SWEEPS_PER_BLOCK
    );
    assert_eq!(metadata["schedule"]["max_attempts"], MAX_ATTEMPTS);
    assert_eq!(
        metadata["schedule"]["measurement"],
        "after each measured heatbath! sweep"
    );
    assert_eq!(metadata["comparison"]["sigma_multiplier"], 6.0);
    assert_eq!(metadata["comparison"]["rust_max_attempts"], MAX_ATTEMPTS);
    assert_eq!(
        metadata["gaugefields_jl"]["operations"],
        serde_json::json!(["Heatbath", "heatbath!", "calculate_Plaquette"])
    );
    assert_eq!(
        metadata["deliberate_corrections"].as_array().unwrap().len(),
        5
    );
}

fn chain_metadata(metadata: &Value, beta: f64) -> &Value {
    metadata["chains"]
        .as_array()
        .unwrap()
        .iter()
        .find(|chain| chain["beta"].as_f64() == Some(beta))
        .unwrap_or_else(|| panic!("missing beta {beta}"))
}

fn links_changed_from_cold(
    actual: &gaugefields::GaugeLinks,
    cold: &gaugefields::GaugeLinks,
) -> bool {
    actual
        .links()
        .iter()
        .zip(cold.links())
        .any(|(actual, cold)| {
            actual
                .typed()
                .host_data()
                .unwrap()
                .iter()
                .zip(cold.typed().host_data().unwrap())
                .any(|(actual, cold)| {
                    actual.re.to_bits() != cold.re.to_bits()
                        || actual.im.to_bits() != cold.im.to_bits()
                })
        })
}

fn run_rust_chain(
    beta: f64,
    state: [u64; 4],
) -> Result<(Vec<f64>, f64, f64, bool), gaugefields::GaugeError> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let cold = cold_su3(lattice)?;
    let mut links = cold_su3(lattice)?;
    let mut rng = ReproducibleRng::from_state(state)?;
    let params = HeatbathParams::new(beta, MAX_ATTEMPTS)?;

    for _ in 0..BURN_IN {
        heatbath_sweep(&mut links, params, &mut rng)?;
    }

    let mut block_means = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let mut sum = 0.0;
        for _ in 0..SWEEPS_PER_BLOCK {
            heatbath_sweep(&mut links, params, &mut rng)?;
            sum += normalized_plaquette(&links)?;
        }
        block_means.push(sum / SWEEPS_PER_BLOCK as f64);
    }
    let mean = block_means.iter().sum::<f64>() / BLOCKS as f64;
    let variance = block_means
        .iter()
        .map(|value| {
            let difference = value - mean;
            difference * difference
        })
        .sum::<f64>()
        / (BLOCKS - 1) as f64;
    let standard_error = (variance / BLOCKS as f64).sqrt();
    Ok((
        block_means,
        mean,
        standard_error,
        links_changed_from_cold(&links, &cold),
    ))
}

#[test]
fn heatbath_statistics_match_julia_within_combined_standard_error(
) -> Result<(), gaugefields::GaugeError> {
    let metadata = fixture_metadata();
    assert_schedule(&metadata);

    for (index, &beta) in BETAS.iter().enumerate() {
        let reference = chain_metadata(&metadata, beta);
        let block_means = reference["block_means"].as_array().unwrap();
        assert_eq!(block_means.len(), BLOCKS, "beta={beta}");
        assert_eq!(reference["measurements"], BLOCKS * SWEEPS_PER_BLOCK);
        let julia_mean = reference["mean"].as_f64().unwrap();
        let julia_se = reference["standard_error"].as_f64().unwrap();
        assert!(julia_mean.is_finite() && (0.0..=1.0).contains(&julia_mean));
        assert!(julia_se.is_finite() && julia_se > 0.0 && julia_se < 0.03);
        assert!(block_means.iter().all(|value| {
            value
                .as_f64()
                .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        }));

        let (rust_blocks, rust_mean, rust_se, changed) = run_rust_chain(beta, RUST_STATES[index])?;
        assert_eq!(rust_blocks.len(), BLOCKS, "beta={beta}");
        let combined_se = (rust_se.mul_add(rust_se, julia_se * julia_se)).sqrt();
        let z = (rust_mean - julia_mean).abs() / combined_se;
        eprintln!(
            "beta={beta} julia_mean={julia_mean:.16} julia_se={julia_se:.16} rust_mean={rust_mean:.16} rust_se={rust_se:.16} z={z:.6}"
        );
        assert!(
            rust_mean.is_finite() && (0.0..=1.0).contains(&rust_mean),
            "beta={beta}"
        );
        assert!(
            rust_se.is_finite() && rust_se > 0.0 && rust_se < 0.03,
            "beta={beta}"
        );
        assert!(changed, "beta={beta} chain did not change from cold");
        assert!(z <= 6.0, "beta={beta} z={z:.6} exceeds 6");
    }
    Ok(())
}
