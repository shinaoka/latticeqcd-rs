use gaugefields::{load_fixture, GaugeError};
use npyz::{Order, WriterBuilder};
use num_complex::Complex64;
use std::{fs, path::Path};

fn write_npy(path: &Path, shape: &[u64], order: Order, values: &[Complex64]) {
    let file = fs::File::create(path).unwrap();
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .order(order)
        .writer(file)
        .begin_nd()
        .unwrap();
    writer.extend(values.iter().copied()).unwrap();
    writer.finish().unwrap();
}

fn fixture_dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("gaugefields-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_valid(dir: &Path) {
    let values = (0..9)
        .map(|x| Complex64::new(x as f64, -1.0))
        .collect::<Vec<_>>();
    for mu in 0..4 {
        write_npy(
            &dir.join(format!("u{mu}.npy")),
            &[3, 3, 1, 1, 1, 1],
            Order::Fortran,
            &values,
        );
    }
    fs::write(dir.join("metadata.json"), r#"{"nc":3,"lattice":[1,1,1,1],"beta":6.0,"expected_observables":{"plaquette":1.0},"gaugefields_jl_version":"0.5.0","gaugefields_jl_commit":"abc123"}"#).unwrap();
}

#[test]
fn valid_fixture_roundtrips_exact_values() {
    let dir = fixture_dir("valid");
    write_valid(&dir);
    let fixture = load_fixture(&dir).unwrap();
    assert_eq!(
        fixture.links().links()[2]
            .tensor()
            .as_slice::<Complex64>()
            .unwrap()[7],
        Complex64::new(7.0, -1.0)
    );
    assert_eq!(fixture.metadata().nc, 3);
}

#[test]
fn reader_rejects_order_dtype_rank_shape_mu_and_metadata() {
    let dir = fixture_dir("invalid");
    write_valid(&dir);
    let values = vec![Complex64::default(); 9];
    write_npy(&dir.join("u0.npy"), &[3, 3, 1, 1, 1, 1], Order::C, &values);
    assert!(matches!(
        load_fixture(&dir),
        Err(GaugeError::NpyOrder { mu: 0 })
    ));
    write_valid(&dir);
    let file = fs::File::create(dir.join("u0.npy")).unwrap();
    let mut w = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[3, 3, 1, 1, 1, 1])
        .order(Order::Fortran)
        .writer(file)
        .begin_nd()
        .unwrap();
    w.extend(vec![0.0_f64; 9]).unwrap();
    w.finish().unwrap();
    assert!(matches!(
        load_fixture(&dir),
        Err(GaugeError::NpyDType { mu: 0, .. })
    ));
    write_valid(&dir);
    write_npy(&dir.join("u0.npy"), &[3, 3, 1], Order::Fortran, &values);
    assert!(matches!(
        load_fixture(&dir),
        Err(GaugeError::NpyRank { mu: 0, .. })
    ));
    write_valid(&dir);
    write_npy(
        &dir.join("u0.npy"),
        &[3, 3, 1, 1, 1, 1],
        Order::Fortran,
        &values,
    );
    write_npy(
        &dir.join("u1.npy"),
        &[3, 3, 1, 1, 1, 2],
        Order::Fortran,
        &[Complex64::default(); 18],
    );
    assert!(matches!(
        load_fixture(&dir),
        Err(GaugeError::MetadataMismatch { .. }) | Err(GaugeError::InconsistentMu { .. })
    ));
    write_valid(&dir);
    fs::write(dir.join("metadata.json"), "{}").unwrap();
    assert!(matches!(load_fixture(&dir), Err(GaugeError::Metadata(_))));
}

#[test]
fn checked_julia_cold_fixture_loads_with_exact_provenance() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/cold_1x1x1x1");
    let fixture = load_fixture(path).unwrap();
    assert_eq!(fixture.metadata().gaugefields_jl_version, "0.7.2");
    assert_eq!(fixture.metadata().gaugefields_jl_commit.len(), 40);
    let values = fixture.links().links()[0]
        .tensor()
        .as_slice::<Complex64>()
        .unwrap();
    assert_eq!(values[0], Complex64::new(1.0, 0.0));
    assert_eq!(values[4], Complex64::new(1.0, 0.0));
    assert_eq!(values[8], Complex64::new(1.0, 0.0));
}
