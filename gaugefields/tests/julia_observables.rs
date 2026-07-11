use gaugefields::{
    load_fixture, measurement_staple, normalized_plaquette, plaquette_sum, wilson_action,
};
use num_complex::Complex64;
use std::{fs, path::Path};
fn residual(a: f64, b: f64) -> f64 {
    assert!(
        a.is_finite() && b.is_finite(),
        "nonfinite observable: actual={a} expected={b}"
    );
    let r = (a - b).abs();
    assert!(r.is_finite());
    r
}
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
            let rel = residual(a, b) / b.abs().max(1.0);
            assert!(rel.is_finite());
            assert!(rel < 1e-12, "{name}: actual={a} expected={b} rel={rel}");
        }
        let mut max = 0.0_f64;
        for mu in 0..4 {
            let v = measurement_staple(f.links(), mu).unwrap();
            let bytes = fs::read(dir.join(format!("measurement_staple{mu}.npy"))).unwrap();
            let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
            assert_eq!(npy.order(), npyz::Order::Fortran);
            let mut shape = vec![3, 3];
            shape.extend(f.links().lattice().extents().map(|x| x as u64));
            assert_eq!(npy.shape(), shape);
            let expected = npy.into_vec::<Complex64>().unwrap();
            let actual = v.tensor().as_slice::<Complex64>().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!(
                    a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite(),
                    "{name}: nonfinite staple mu={mu}"
                );
                let r = (*a - b).norm();
                assert!(r.is_finite());
                max = max.max(r);
            }
        }
        assert!(max < 1e-13, "{name}: max staple residual={max}");
    }
}
