use gaugefields::{
    load_fixture, measurement_staple, normalized_plaquette, plaquette_sum, wilson_action,
};
use num_complex::Complex64;
use std::{fs, path::Path};
#[test]
fn julia_random_observables_and_staples_match() {
    for name in ["random_2x2x2x2", "random_4x4x4x4"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures")
            .join(name);
        let f = load_fixture(&dir).unwrap();
        let e = &f.metadata().expected_observables;
        let checks = [
            (
                plaquette_sum(f.links()).unwrap(),
                e["plaquette_sum"].as_f64().unwrap(),
            ),
            (
                normalized_plaquette(f.links()).unwrap(),
                e["normalized_plaquette"].as_f64().unwrap(),
            ),
            (
                wilson_action(f.links(), f.metadata().beta).unwrap(),
                e["wilson_action"].as_f64().unwrap(),
            ),
        ];
        for (a, b) in checks {
            let rel = (a - b).abs() / b.abs().max(1.0);
            assert!(rel < 1e-12, "{name}: actual={a} expected={b} rel={rel}");
        }
        let v = measurement_staple(f.links()).unwrap();
        let mut max = 0.0_f64;
        for mu in 0..4 {
            let bytes = fs::read(dir.join(format!("measurement_staple{mu}.npy"))).unwrap();
            let expected = npyz::NpyFile::new(&bytes[..])
                .unwrap()
                .into_vec::<Complex64>()
                .unwrap();
            for (a, b) in v.links()[mu]
                .tensor()
                .as_slice::<Complex64>()
                .unwrap()
                .iter()
                .zip(expected)
            {
                max = max.max((*a - b).norm());
            }
        }
        assert!(max < 1e-13, "{name}: max staple residual={max}");
    }
}
