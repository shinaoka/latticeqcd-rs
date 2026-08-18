use gaugefields::{cold_su3, load_fixture, site_index, store_link, LatticeShape4, Mat3};
use measurements::{clover_topological_charge, polyakov_loop};
use npyz::Order;
use num_complex::Complex64;
use serde_json::Value;
use std::{f64::consts::PI, fs, path::Path};
use wilsonloop::{evaluate_path, WilsonPath};

#[test]
fn cold_polyakov_is_three() -> Result<(), Box<dyn std::error::Error>> {
    let links = cold_su3(LatticeShape4::new([2, 3, 2, 4])?)?;
    assert_eq!(polyakov_loop(&links)?, Complex64::new(3.0, 0.0));
    assert_eq!(clover_topological_charge(&links)?, 0.0);
    Ok(())
}

fn explicit_polyakov_product(links: &gaugefields::GaugeLinks) -> Complex64 {
    let lattice = links.lattice();
    let [nx, ny, nz, nt] = lattice.extents();
    let values = links.links()[3].typed().host_data().unwrap();
    let mut sum = Complex64::default();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut product = Mat3::identity();
                for t in 0..nt {
                    let site = site_index([x, y, z, t], lattice).unwrap();
                    product =
                        product.mul(Mat3::load(values, site.checked_mul(9).unwrap()).unwrap());
                }
                sum += product.trace();
            }
        }
    }
    sum / (nx * ny * nz) as f64
}

#[test]
fn polyakov_matches_independent_storage_product() -> Result<(), Box<dyn std::error::Error>> {
    let directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/measurements_task_d1");
    let fixture = load_fixture(directory)?;
    let actual = polyakov_loop(fixture.links())?;
    let independent = explicit_polyakov_product(fixture.links());
    assert!((actual - independent).norm() <= 1e-15);
    Ok(())
}

#[test]
fn pinned_scalar_oracle_matches() -> Result<(), Box<dyn std::error::Error>> {
    let directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/measurements_task_d1");
    let fixture = load_fixture(directory)?;
    let scalar = &fixture.metadata().expected_observables["scalar"];
    let expected_polyakov = Complex64::new(
        scalar["polyakov_loop"]["real"]
            .as_f64()
            .ok_or("missing real")?,
        scalar["polyakov_loop"]["imag"]
            .as_f64()
            .ok_or("missing imag")?,
    );
    let expected_charge = scalar["clover_topological_charge"]
        .as_f64()
        .ok_or("missing clover charge")?;

    let actual_polyakov = polyakov_loop(fixture.links())?;
    let actual_charge = clover_topological_charge(fixture.links())?;
    let polyakov_residual = (actual_polyakov - expected_polyakov).norm();
    let charge_residual = (actual_charge - expected_charge).abs();
    eprintln!(
        "scalar oracle: polyakov residual={polyakov_residual:.16e}, clover residual={charge_residual:.16e}"
    );
    assert!(polyakov_residual <= 2e-12);
    assert!(charge_residual <= 2e-12);
    Ok(())
}

#[test]
fn noncubic_polyakov_uses_axis_three_and_spatial_normalization(
) -> Result<(), Box<dyn std::error::Error>> {
    let lattice = LatticeShape4::new([2, 3, 2, 4])?;
    let mut links = cold_su3(lattice)?;
    let mut expected = 0.0;
    for z in 0..2 {
        for y in 0..3 {
            for x in 0..2 {
                let theta = 0.05 * (1 + x + 2 * y + 3 * z) as f64;
                let mut link = Mat3::identity();
                link[(0, 0)] = Complex64::from_polar(1.0, theta);
                link[(1, 1)] = Complex64::from_polar(1.0, -theta);
                store_link(&mut links, 3, site_index([x, y, z, 0], lattice)?, link)?;
                expected += 1.0 + 2.0 * theta.cos();
            }
        }
    }
    expected /= 12.0;
    let actual = polyakov_loop(&links)?;
    assert!((actual - Complex64::new(expected, 0.0)).norm() <= 2e-15);
    Ok(())
}

fn epsilon(indices: [usize; 4]) -> f64 {
    if (0..4).any(|left| indices[left + 1..].contains(&indices[left])) {
        return 0.0;
    }
    let inversions = (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|&(left, right)| indices[left] > indices[right])
        .count();
    if inversions % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn independent_clover_charge(
    links: &gaugefields::GaugeLinks,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut total = 0.0;
    for site in 0..links.lattice().nv() {
        let mut clovers = [Mat3::zero(); 16];
        for mu in 0..4 {
            for nu in 0..4 {
                if mu == nu {
                    continue;
                }
                let m = (mu + 1) as i8;
                let n = (nu + 1) as i8;
                let paths = [
                    [m, n, -m, -n],
                    [n, -m, -n, m],
                    [-n, m, n, -m],
                    [-m, -n, m, n],
                ];
                let mut sum = Mat3::zero();
                for steps in paths {
                    sum.add_scaled_real(
                        1.0,
                        evaluate_path(links, site, &WilsonPath::new(steps.to_vec())?)?,
                    );
                }
                clovers[4 * mu + nu] = sum.ta();
            }
        }
        for mu in 0..4 {
            for nu in 0..4 {
                for rho in 0..4 {
                    for sigma in 0..4 {
                        total += epsilon([mu, nu, rho, sigma])
                            * clovers[4 * mu + nu].real_trace_mul(clovers[4 * rho + sigma])
                            / 16.0;
                    }
                }
            }
        }
    }
    Ok(-total / (32.0 * PI * PI))
}

#[test]
fn clover_sign_paths_and_normalization_match_independent_wilson_paths(
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = load_fixture(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/measurements_task_d1"),
    )?;
    let actual = clover_topological_charge(fixture.links())?;
    let independent = independent_clover_charge(fixture.links())?;
    assert!(actual.abs() > 1e-6, "vacuous clover fixture");
    assert!((actual - independent).abs() <= 2e-15);
    Ok(())
}

#[test]
fn pinned_representative_paths_match_evaluate_path() -> Result<(), Box<dyn std::error::Error>> {
    let directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/measurements_task_d1");
    let fixture = load_fixture(&directory)?;
    let metadata: Value = serde_json::from_slice(&fs::read(directory.join("metadata.json"))?)?;
    let representative = &metadata["expected_observables"]["representative_paths"];
    assert_eq!(
        representative["origin"]["rust_site_index"].as_u64(),
        Some(0)
    );
    assert_eq!(
        representative["origin"]["rust_coordinates"],
        serde_json::json!([0, 0, 0, 0])
    );
    assert_eq!(
        representative["origin"]["julia_coordinates"],
        serde_json::json!([1, 1, 1, 1])
    );
    let tolerance = representative["tolerance"].as_f64().unwrap();
    assert_eq!(tolerance, 2e-12);

    let required = [
        ("forward", vec![1_i8]),
        ("backward", vec![-1_i8]),
        ("open", vec![1_i8, 2, -3]),
        ("plaquette", vec![1_i8, 2, -1, -2]),
        ("rectangle", vec![1_i8, 2, 2, -1, -2, -2]),
        ("clover_right_bottom", vec![-2_i8, 1, 2, -1]),
    ];
    let paths = representative["paths"].as_array().unwrap();
    assert_eq!(paths.len(), required.len());
    for (name, steps) in required {
        let entry = paths
            .iter()
            .find(|entry| entry["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("missing representative path {name}"));
        let actual_steps = entry["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| step.as_i64().unwrap() as i8)
            .collect::<Vec<_>>();
        assert_eq!(actual_steps, steps);

        let file = directory.join(entry["file"].as_str().unwrap());
        let bytes = fs::read(&file)?;
        let npy = npyz::NpyFile::new(&bytes[..])?;
        assert_eq!(npy.order(), Order::Fortran);
        assert_eq!(npy.shape(), &[3, 3]);
        let expected = npy.into_vec::<Complex64>()?;
        let path = WilsonPath::new(steps)?;
        if name == "open" {
            assert!(!path.is_closed(), "open path unexpectedly closes");
        }
        let actual = evaluate_path(fixture.links(), 0, &path)?;
        let residual = actual
            .as_array()
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max);
        assert!(residual <= tolerance, "{name} residual={residual:.16e}");
        let nontrivial = actual
            .as_array()
            .iter()
            .zip(Mat3::identity().as_array())
            .map(|(actual, identity)| (*actual - *identity).norm())
            .fold(0.0_f64, f64::max);
        assert!(nontrivial > 1e-12, "{name} path oracle is vacuous");
        eprintln!("path oracle {name}: residual={residual:.16e}");
    }
    Ok(())
}
