use gaugefields::{dsdu, gauge_force, load_fixture};
use num_complex::Complex64;
use std::{fs, path::Path};
fn residual(a: f64, b: f64) -> f64 {
    assert!(
        a.is_finite() && b.is_finite(),
        "nonfinite parity input: actual={a} expected={b}"
    );
    let r = (a - b).abs();
    assert!(r.is_finite(), "nonfinite residual");
    r
}
#[test]
#[should_panic(expected = "nonfinite parity input")]
fn parity_rejects_nan() {
    let _ = residual(f64::NAN, 0.0);
}
#[test]
fn julia_dsdu_and_ta_force_match_all_components() {
    for name in ["random_2x2x2x2", "random_4x4x4x4"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures")
            .join(name);
        let f = load_fixture(&dir).unwrap();
        let d = dsdu(f.links(), f.metadata().beta).unwrap();
        let force = gauge_force(f.links(), f.metadata().beta).unwrap();
        let mut md = 0.0_f64;
        let mut mf = 0.0_f64;
        for mu in 0..4 {
            let bytes = fs::read(dir.join(format!("dsdu{mu}.npy"))).unwrap();
            let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
            assert_eq!(npy.order(), npyz::Order::Fortran);
            let mut shape = vec![3, 3];
            shape.extend(f.links().lattice().extents().map(|x| x as u64));
            assert_eq!(npy.shape(), shape);
            let e = npy.into_vec::<Complex64>().unwrap();
            let actual = d.links()[mu].tensor().as_slice::<Complex64>().unwrap();
            assert_eq!(actual.len(), e.len());
            for (a, b) in actual.iter().zip(&e) {
                assert!(
                    a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite(),
                    "{name}: nonfinite dsdu mu={mu}"
                );
                let r = (*a - *b).norm();
                assert!(r.is_finite());
                md = md.max(r);
            }
            let bytes = fs::read(dir.join(format!("force_coeff{mu}.npy"))).unwrap();
            let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
            assert_eq!(npy.order(), npyz::Order::Fortran);
            let mut shape = vec![8];
            shape.extend(f.links().lattice().extents().map(|x| x as u64));
            assert_eq!(npy.shape(), shape);
            let e = npy.into_vec::<f64>().unwrap();
            let actual = force.tensors()[mu].as_slice::<f64>().unwrap();
            assert_eq!(actual.len(), e.len());
            for (a, b) in actual.iter().zip(e) {
                mf = mf.max(residual(*a, b));
            }
        }
        assert!(md < 1e-13, "{name}: max dsdu residual={md}");
        assert!(mf < 1e-13, "{name}: max force residual={mf}");
    }
}
