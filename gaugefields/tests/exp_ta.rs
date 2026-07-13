use npyz::Order;
use num_complex::Complex64;
use serde_json::Value;
use std::{fs, path::Path};

const JULIA_COMMIT: &str = "9e5719970770f4497405a856315c90bef7f74449";

#[test]
fn julia_exp_ta_fixture_has_branch_provenance() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/exp_ta");
    let metadata: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json")).unwrap())
        .unwrap();
    assert_eq!(metadata["gaugefields_jl_commit"], JULIA_COMMIT);
    assert_eq!(metadata["source_function"], "exptU!");
    let cases = metadata["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    assert!(cases.iter().any(|case| case["name"] == "zero"));
    assert!(cases.iter().any(|case| case["branch"] == "analytic"));
    assert!(cases.iter().any(|case| case["branch"] == "fallback"));
    assert!(cases.iter().any(|case| case["name"] == "near_below"));
    assert!(cases.iter().any(|case| case["name"] == "near_above"));
    for case in cases {
        assert_eq!(case["coefficients"].as_array().unwrap().len(), 8);
        assert!(case["t"].as_f64().unwrap().is_finite());
    }

    let bytes = fs::read(dir.join("expected.npy")).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(npy.order(), Order::Fortran);
    assert_eq!(npy.shape(), &[3, 3, cases.len() as u64]);
    assert_eq!(npy.into_vec::<Complex64>().unwrap().len(), 9 * cases.len());
}
