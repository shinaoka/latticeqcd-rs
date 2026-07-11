use gaugefields::{dsdu, gauge_force, load_fixture};
use num_complex::Complex64;
use std::{fs, path::Path};
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
            let e = npyz::NpyFile::new(&bytes[..])
                .unwrap()
                .into_vec::<Complex64>()
                .unwrap();
            for (a, b) in d.links()[mu]
                .tensor()
                .as_slice::<Complex64>()
                .unwrap()
                .iter()
                .zip(&e)
            {
                md = md.max((*a - *b).norm());
            }
            let bytes = fs::read(dir.join(format!("force_coeff{mu}.npy"))).unwrap();
            let e = npyz::NpyFile::new(&bytes[..])
                .unwrap()
                .into_vec::<f64>()
                .unwrap();
            for (a, b) in force.tensors()[mu].as_slice::<f64>().unwrap().iter().zip(e) {
                mf = mf.max((*a - b).abs());
            }
        }
        assert!(md < 1e-13, "{name}: max dsdu residual={md}");
        assert!(mf < 1e-13, "{name}: max force residual={mf}");
    }
}
