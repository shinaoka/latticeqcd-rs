use gaugefields::{GaugeError, ReproducibleRng};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::Deserialize;
use std::{fs, path::Path};

const STATE: [u64; 4] = [1, 2, 3, 4];
const RAW_OUTPUTS: [u64; 10] = [
    41943041,
    58720359,
    3588806011781223,
    3591011842654386,
    9228616714210784205,
    9973669472204895162,
    14011001112246962877,
    12406186145184390807,
    15849039046786891736,
    10450023813501588000,
];
const NORMAL_TOLERANCE: f64 = 1e-14;

#[derive(Debug, Deserialize)]
struct RngFixture {
    julia_version: String,
    julia_commit: String,
    julia_source: JuliaSource,
    algorithm: String,
    rand_xoshiro_version: String,
    state: [u64; 4],
    state_word_order: String,
    raw_generation: String,
    raw_outputs: Vec<String>,
    uniform_formula: String,
    box_muller: BoxMuller,
    normal_values: Vec<f64>,
    normal_bits: Vec<String>,
    normal_comparison_tolerance: f64,
}

#[derive(Debug, Deserialize)]
struct JuliaSource {
    url: String,
    revision: String,
}

#[derive(Debug, Deserialize)]
struct BoxMuller {
    u_order: String,
    pair_order: String,
    odd_fill_policy: String,
}

fn fixture() -> RngFixture {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reproducible_rng/metadata.json");
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn seed(state: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (index, word) in state.into_iter().enumerate() {
        bytes[index * 8..(index + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn direct_rng(state: [u64; 4]) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::from_seed(seed(state))
}

fn open_unit(raw: u64) -> f64 {
    ((raw >> 12) as f64 + 0.5) * 2f64.powi(-52)
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    let residual = (actual - expected).abs();
    assert!(
        actual.is_finite() && residual <= NORMAL_TOLERANCE,
        "{label}: actual={actual:.17e} expected={expected:.17e} residual={residual:.3e}"
    );
}

fn raw_words(fixture: &RngFixture) -> Vec<u64> {
    fixture
        .raw_outputs
        .iter()
        .map(|word| {
            assert!(word.starts_with("0x") && word.len() == 18);
            u64::from_str_radix(&word[2..], 16).unwrap()
        })
        .collect()
}

#[test]
fn fixture_provenance_and_raw_stream_are_exact() {
    let fixture = fixture();
    assert_eq!(fixture.julia_version, "1.12.5");
    assert_eq!(
        fixture.julia_commit,
        "5fe89b8ddc166260bfcd4a195b305aff0ccad686"
    );
    assert!(fixture.julia_source.url.contains("JuliaLang/julia"));
    assert_eq!(fixture.julia_source.revision, fixture.julia_commit);
    assert_eq!(fixture.algorithm, "xoshiro256++");
    assert_eq!(fixture.rand_xoshiro_version, "0.6.0");
    assert_eq!(fixture.state, STATE);
    assert!(fixture.state_word_order.contains("s0"));
    assert!(fixture.state_word_order.contains("s3"));
    assert!(fixture.raw_generation.contains("scalar"));
    assert!(fixture.raw_generation.contains("no array"));
    assert!(fixture.raw_generation.contains("bulk generation"));
    assert!(fixture.uniform_formula.contains("next_u64"));
    assert!(fixture.uniform_formula.contains("2^-52"));
    assert_eq!(fixture.box_muller.u_order, "u1 then u2");
    assert!(fixture.box_muller.pair_order.contains("cos"));
    assert!(fixture.box_muller.pair_order.contains("sin"));
    assert!(fixture.box_muller.odd_fill_policy.contains("discard"));
    assert_eq!(fixture.normal_comparison_tolerance, NORMAL_TOLERANCE);

    let raw = raw_words(&fixture);
    assert_eq!(raw, RAW_OUTPUTS);
    let mut rng = ReproducibleRng::from_state(STATE).unwrap();
    for expected in raw {
        assert_eq!(rng.next_u64(), expected);
    }
}

#[test]
fn open_unit_handles_concrete_zero_and_max_words() {
    let mut zero = ReproducibleRng::from_state([1, 0, 0, u64::MAX - 1]).unwrap();
    let zero_unit = zero.open_unit_f64();
    assert_eq!(zero_unit, 2f64.powi(-53));
    assert!(zero_unit.is_finite() && zero_unit > 0.0 && zero_unit < 1.0);
    let mut zero_expected = direct_rng([1, 0, 0, u64::MAX - 1]);
    zero_expected.next_u64();
    assert_eq!(zero.next_u64(), zero_expected.next_u64());

    let mut max = ReproducibleRng::from_state([0, 1, 0, u64::MAX]).unwrap();
    let max_unit = max.open_unit_f64();
    assert_eq!(max_unit, 1.0 - 2f64.powi(-53));
    assert!(max_unit.is_finite() && max_unit > 0.0 && max_unit < 1.0);
    let mut max_expected = direct_rng([0, 1, 0, u64::MAX]);
    max_expected.next_u64();
    assert_eq!(max.next_u64(), max_expected.next_u64());
}

#[test]
fn normal_pairs_match_julia_values_with_finite_cross_libm_residuals() {
    let fixture = fixture();
    assert_eq!(fixture.normal_values.len(), 10);
    assert_eq!(fixture.normal_bits.len(), 10);
    for (value, bits) in fixture.normal_values.iter().zip(&fixture.normal_bits) {
        assert!(bits.starts_with("0x") && bits.len() == 18);
        let bit_value = f64::from_bits(u64::from_str_radix(&bits[2..], 16).unwrap());
        assert_close(*value, bit_value, "fixture decimal/bit normal value");
    }

    let mut rng = ReproducibleRng::from_state(STATE).unwrap();
    for pair_index in 0..5 {
        let pair = rng.standard_normal_pair();
        for (component, actual) in pair.into_iter().enumerate() {
            assert_close(
                actual,
                fixture.normal_values[2 * pair_index + component],
                &format!("pair {pair_index} component {component}"),
            );
        }
    }
}

#[test]
fn normal_fill_preserves_pair_order_and_documented_draw_counts() {
    let fixture = fixture();
    for len in 0..=3 {
        let mut rng = ReproducibleRng::from_state(STATE).unwrap();
        let mut output = vec![0.0; len];
        rng.fill_standard_normals(&mut output);
        for (index, &actual) in output.iter().enumerate() {
            assert_close(
                actual,
                fixture.normal_values[index],
                &format!("len {len} index {index}"),
            );
        }

        let mut expected = direct_rng(STATE);
        for _ in 0..(2 * len.div_ceil(2)) {
            expected.next_u64();
        }
        assert_eq!(
            rng.next_u64(),
            expected.next_u64(),
            "length {len} draw count"
        );
    }
}

#[test]
fn reset_restarts_stream_and_zero_state_failure_is_transactional() {
    let mut rng = ReproducibleRng::from_state(STATE).unwrap();
    let _ = rng.standard_normal_pair();
    rng.set_state(STATE).unwrap();
    let mut fresh = ReproducibleRng::from_state(STATE).unwrap();
    assert_eq!(rng.next_u64(), fresh.next_u64());
    assert_eq!(
        rng.open_unit_f64().to_bits(),
        fresh.open_unit_f64().to_bits()
    );
    assert_eq!(rng.standard_normal_pair(), fresh.standard_normal_pair());

    assert!(matches!(
        ReproducibleRng::from_state([0; 4]),
        Err(GaugeError::InvalidRngState)
    ));
    let mut expected = rng.clone();
    assert!(matches!(
        rng.set_state([0; 4]),
        Err(GaugeError::InvalidRngState)
    ));
    assert_eq!(rng.next_u64(), expected.next_u64());
}

#[test]
fn clone_copies_position_and_debug_hides_state_words() {
    let mut original = ReproducibleRng::from_state(STATE).unwrap();
    let _ = original.next_u64();
    let mut clone = original.clone();
    assert_eq!(format!("{original:?}"), "ReproducibleRng");
    assert_eq!(original.next_u64(), clone.next_u64());
    assert_eq!(
        original.open_unit_f64().to_bits(),
        clone.open_unit_f64().to_bits()
    );
    assert_eq!(
        original.standard_normal_pair(),
        clone.standard_normal_pair()
    );
    let mut left = [0.0; 1];
    let mut right = [0.0; 1];
    original.fill_standard_normals(&mut left);
    clone.fill_standard_normals(&mut right);
    assert_eq!(left, right);
}

#[test]
fn rng_core_delegates_to_direct_xoshiro() {
    let state = [7, 11, 13, 17];
    let mut actual = ReproducibleRng::from_state(state).unwrap();
    let mut expected = direct_rng(state);
    assert_eq!(actual.next_u32(), expected.next_u32());
    assert_eq!(actual.next_u64(), expected.next_u64());

    let mut actual_bytes = [0u8; 37];
    let mut expected_bytes = [0u8; 37];
    actual.fill_bytes(&mut actual_bytes);
    expected.fill_bytes(&mut expected_bytes);
    assert_eq!(actual_bytes, expected_bytes);

    let mut actual_try = [0u8; 19];
    let mut expected_try = [0u8; 19];
    actual.try_fill_bytes(&mut actual_try).unwrap();
    expected.try_fill_bytes(&mut expected_try).unwrap();
    assert_eq!(actual_try, expected_try);
}

#[test]
fn interleaved_calls_share_one_raw_stream_without_a_cached_spare() {
    let fixture = fixture();
    let raw = raw_words(&fixture);
    let mut rng = ReproducibleRng::from_state(STATE).unwrap();

    assert_eq!(rng.next_u64(), raw[0]);
    assert_eq!(rng.open_unit_f64().to_bits(), open_unit(raw[1]).to_bits());
    let pair = rng.standard_normal_pair();
    assert_close(pair[0], fixture.normal_values[2], "interleaved pair cosine");
    assert_close(pair[1], fixture.normal_values[3], "interleaved pair sine");

    let mut output = [0.0; 1];
    rng.fill_standard_normals(&mut output);
    assert_close(output[0], fixture.normal_values[4], "interleaved odd fill");
    assert_eq!(rng.next_u64(), raw[6]);
}
