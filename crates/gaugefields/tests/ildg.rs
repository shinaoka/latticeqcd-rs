use gaugefields::{
    cold_su3, load_link, normalized_plaquette, read_ildg, store_link, write_ildg, GaugeError,
    GaugeLinkTensor, GaugeLinks, LatticeShape4, Mat3,
};
use npyz::NpyFile;
use num_complex::Complex64;
use serde_json::Value;
use std::{fs, path::PathBuf};
use tenferro_tensor::TypedTensor;

fn sentinel_links(lattice: LatticeShape4) -> GaugeLinks {
    let [nx, ny, nz, nt] = lattice.extents();
    let values = |mu| {
        let mut values = Vec::with_capacity(9 * lattice.nv());
        for site in 0..lattice.nv() {
            for column in 0..3 {
                for row in 0..3 {
                    let value = mu * 1_000_000 + site * 10_000 + column * 100 + row;
                    values.push(Complex64::new(value as f64, -(value as f64) - 0.25));
                }
            }
        }
        TypedTensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], values).unwrap()
    };
    GaugeLinks::new(std::array::from_fn(|mu| {
        GaugeLinkTensor::from_typed(values(mu), lattice).unwrap()
    }))
    .unwrap()
}

fn nc2_links(lattice: LatticeShape4) -> GaugeLinks {
    let [nx, ny, nz, nt] = lattice.extents();
    let values = vec![Complex64::new(1.0, 0.0); 4 * lattice.nv()];
    GaugeLinks::new(std::array::from_fn(|_| {
        GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(vec![2, 2, nx, ny, nz, nt], values.clone()).unwrap(),
            lattice,
        )
        .unwrap()
    }))
    .unwrap()
}

fn temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("gaugefields-{label}-{}.ildg", std::process::id()))
}

#[test]
fn host_view_and_ildg_roundtrip_preserve_matrix_color_direction_and_site_order() {
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let links = sentinel_links(lattice);
    let view = links.host_view().unwrap();
    assert_eq!(view.lattice(), lattice);
    assert_eq!(view.shifted_site(0, 0, -1).unwrap(), 1);
    assert_eq!(view.shifted_site(1, 0, 1).unwrap(), 0);
    assert_eq!(
        view.link(2, 3).unwrap()[(1, 2)],
        Complex64::new(2_000_000.0 + 30_000.0 + 200.0 + 1.0, -2_030_201.25)
    );

    let path = temp_path("roundtrip");
    let _ = fs::remove_file(&path);
    write_ildg(&path, &links).unwrap();
    let loaded = read_ildg(&path).unwrap();
    let loaded_view = loaded.host_view().unwrap();
    for mu in 0..4 {
        for site in 0..lattice.nv() {
            assert_eq!(
                loaded_view.link(mu, site).unwrap(),
                view.link(mu, site).unwrap()
            );
        }
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn host_view_clone_and_kernel_parity_remain_fallible_and_read_only() {
    let lattice = LatticeShape4::new([2, 3, 1, 2]).unwrap();
    let links = cold_su3(lattice).unwrap();
    let view = links.host_view().unwrap();
    assert_eq!(view.link(1, 0).unwrap(), load_link(&links, 1, 0).unwrap());
    assert_eq!(
        view.shifted_site(0, 1, 8).unwrap(),
        view.shifted_site(0, 1, 2).unwrap()
    );
    assert!(matches!(
        view.link(4, 0),
        Err(GaugeError::InvalidDirection { .. })
    ));
    assert!(matches!(
        view.shifted_site(lattice.nv(), 0, 1),
        Err(GaugeError::SiteOutOfBounds { .. })
    ));
    let copy = links.try_clone().unwrap();
    assert_eq!(
        normalized_plaquette(&copy).unwrap(),
        normalized_plaquette(&links).unwrap()
    );
}

#[test]
fn pinned_julia_fixture_is_component_exact_and_rust_writer_is_canonical() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/ildg_task_a");
    let metadata: Value =
        serde_json::from_slice(&fs::read(fixture.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["schema"], "ildg_task_a.v1");
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["nc"], 3);
    assert_eq!(
        metadata["gaugefields_jl_commit"],
        "9e5719970770f4497405a856315c90bef7f74449"
    );
    assert_eq!(metadata["source_urls"].as_array().unwrap().len(), 2);
    assert_eq!(
        metadata["writer"],
        "independent manual LIME writer; no c-lime and no incomplete save_binarydata! implementation"
    );
    assert_eq!(metadata["lime"]["header_bytes"], 144);
    assert_eq!(metadata["xml"]["precision"], 64);
    assert_eq!(
        metadata["files"],
        serde_json::json!(["gauge.ildg", "u0.npy", "u1.npy", "u2.npy", "u3.npy"])
    );
    assert_eq!(metadata["comparison"]["field_max_abs_tolerance"], 0.0);

    let path = fixture.join("gauge.ildg");
    let links = read_ildg(&path).unwrap();
    let view = links.host_view().unwrap();
    for mu in 0..4 {
        let bytes = fs::read(fixture.join(format!("u{mu}.npy"))).unwrap();
        let values = NpyFile::new(&bytes[..])
            .unwrap()
            .into_vec::<Complex64>()
            .unwrap();
        for site in 0..links.lattice().nv() {
            assert_eq!(
                view.link(mu, site).unwrap().as_array().as_slice(),
                &values[site * 9..site * 9 + 9]
            );
        }
    }

    let output = temp_path("fixture-canonical");
    let _ = fs::remove_file(&output);
    write_ildg(&output, &links).unwrap();
    assert_eq!(fs::read(&output).unwrap(), fs::read(&path).unwrap());
    fs::remove_file(output).unwrap();
}

#[test]
fn invalid_nc2_write_fails_before_destination_creation() {
    let path = temp_path("invalid-nc2");
    let _ = fs::remove_file(&path);
    let links = nc2_links(LatticeShape4::new([1, 1, 1, 1]).unwrap());
    assert!(matches!(
        write_ildg(&path, &links),
        Err(GaugeError::UnsupportedNc { found: 2 })
    ));
    assert!(!path.exists());
}

#[test]
fn nonfinite_write_fails_before_destination_truncation() {
    let path = temp_path("invalid-nonfinite");
    let _ = fs::remove_file(&path);
    fs::write(&path, b"sentinel").unwrap();
    let lattice = LatticeShape4::new([1, 1, 1, 1]).unwrap();
    let mut links = cold_su3(lattice).unwrap();
    let mut matrix = Mat3::identity();
    matrix[(0, 0)] = Complex64::new(f64::NAN, 0.0);
    store_link(&mut links, 0, 0, matrix).unwrap();
    assert!(matches!(
        write_ildg(&path, &links),
        Err(GaugeError::IldgNonFinite { .. })
    ));
    assert_eq!(fs::read(&path).unwrap(), b"sentinel");
    fs::remove_file(path).unwrap();
}
