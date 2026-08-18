use gaugefields::{
    cold_su3, load_fixture, stout_step, CpuEvolutionContext, GaugeError, GaugeLinkTensor,
    GaugeLinks, LatticeShape4, Mat3,
};
use npyz::{NpyFile, Order};
use num_complex::Complex64;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::TypedTensor;

const LINK_SHAPE: [u64; 6] = [3, 3, 2, 2, 2, 2];
const FIELD_TOLERANCE: f64 = 5e-12;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/stout_task_c")
}

fn read_complex_npy(path: &Path) -> Vec<Complex64> {
    let bytes = fs::read(path).unwrap();
    let npy = NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(npy.order(), Order::Fortran);
    assert_eq!(npy.shape(), &LINK_SHAPE);
    npy.into_vec::<Complex64>().unwrap()
}

fn capture_links(links: &GaugeLinks) -> [Vec<Complex64>; 4] {
    std::array::from_fn(|mu| links.links()[mu].typed().host_data().unwrap().to_vec())
}

fn assert_links_bitwise_unchanged(links: &GaugeLinks, before: &[Vec<Complex64>; 4]) {
    for (mu, expected) in before.iter().enumerate() {
        let actual = links.links()[mu].typed().host_data().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.re.to_bits(),
                expected.re.to_bits(),
                "mu={mu}, index={index}"
            );
            assert_eq!(
                actual.im.to_bits(),
                expected.im.to_bits(),
                "mu={mu}, index={index}"
            );
        }
    }
}

fn max_matrix_residual(actual: &Mat3, expected: &Mat3) -> f64 {
    actual
        .as_array()
        .iter()
        .zip(expected.as_array())
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max)
}

fn su3_residual(links: &GaugeLinks) -> (f64, f64) {
    let mut unitary = 0.0_f64;
    let mut determinant = 0.0_f64;
    for link in links.links() {
        for block in link.typed().host_data().unwrap().chunks_exact(9) {
            let matrix = Mat3::load(block, 0).unwrap();
            let product = matrix.mul_adj_right(matrix);
            for row in 0..3 {
                for column in 0..3 {
                    let expected = if row == column {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::default()
                    };
                    unitary = unitary.max((product[(row, column)] - expected).norm());
                }
            }
            let a = matrix.as_array();
            let det = a[0] * (a[4] * a[8] - a[7] * a[5]) - a[3] * (a[1] * a[8] - a[7] * a[2])
                + a[6] * (a[1] * a[5] - a[4] * a[2]);
            determinant = determinant.max((det - Complex64::new(1.0, 0.0)).norm());
        }
    }
    (unitary, determinant)
}

#[test]
fn cold_field_is_exact_for_positive_zero_and_negative_rho() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([2, 2, 1, 1])?;
    let links = cold_su3(lattice)?;
    let before = capture_links(&links);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());

    for rho in [0.12, 0.0, -0.07] {
        let output = stout_step(&mut context, &links, rho)?;
        for link in output.links() {
            for block in link.typed().host_data().unwrap().chunks_exact(9) {
                assert_eq!(block[0], Complex64::new(1.0, 0.0));
                assert_eq!(block[4], Complex64::new(1.0, 0.0));
                assert_eq!(block[8], Complex64::new(1.0, 0.0));
                for &index in &[1, 2, 3, 5, 6, 7] {
                    assert_eq!(block[index], Complex64::default());
                }
            }
        }
        assert_links_bitwise_unchanged(&links, &before);
    }
    Ok(())
}

#[test]
fn nonfinite_rho_is_rejected_before_context_or_input_changes() -> Result<(), GaugeError> {
    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
    let before = capture_links(&links);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let cache_before = context.cache_stats();

    for rho in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            stout_step(&mut context, &links, rho),
            Err(GaugeError::NonFiniteRho { found }) if found.to_bits() == rho.to_bits()
        ));
        assert_eq!(context.cache_stats(), cache_before);
        assert_links_bitwise_unchanged(&links, &before);
    }
    Ok(())
}

#[test]
fn nonfinite_input_is_rejected_before_allocation_or_backend_work() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut values = vec![Complex64::default(); 9];
    values[0] = Complex64::new(f64::NAN, 0.0);
    values[4] = Complex64::new(1.0, 0.0);
    values[8] = Complex64::new(1.0, 0.0);
    let links = GaugeLinks::new(std::array::from_fn(|_| {
        GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(vec![3, 3, 1, 1, 1, 1], values.clone()).unwrap(),
            lattice,
        )
        .unwrap()
    }))?;
    let before = capture_links(&links);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let cache_before = context.cache_stats();
    assert!(matches!(
        stout_step(&mut context, &links, 0.12),
        Err(GaugeError::NonFiniteSu3Input {
            operation: "stout_step",
            ..
        })
    ));
    assert_eq!(context.cache_stats(), cache_before);
    assert_links_bitwise_unchanged(&links, &before);
    Ok(())
}

#[test]
fn positive_staple_has_the_pinned_six_term_orientation_and_sign() -> Result<(), GaugeError> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2");
    let fixture = load_fixture(path)?;
    let view = fixture.links().host_view()?;
    let site = 5;
    let mu = 0;
    let mut expected = Mat3::zero();
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let plus_nu = view.shifted_site(site, nu, 1)?;
        let plus_mu = view.shifted_site(site, mu, 1)?;
        let upper = view
            .link(nu, site)?
            .mul(view.link(mu, plus_nu)?)
            .mul_adj_right(view.link(nu, plus_mu)?);
        let back = view.shifted_site(site, nu, -1)?;
        let lower = view
            .link(nu, back)?
            .adjoint()
            .mul(view.link(mu, back)?)
            .mul(view.link(nu, view.shifted_site(back, mu, 1)?)?);
        expected.add_scaled_real(1.0, upper);
        expected.add_scaled_real(1.0, lower);
    }

    let actual = view.force_staple(site, mu)?;
    let residual = max_matrix_residual(&actual, &expected);
    assert!(residual <= 1e-14, "positive staple residual={residual:e}");
    println!("Task C positive staple residual={residual:.17e}");
    assert!(
        actual
            .as_array()
            .iter()
            .map(|value| value.norm())
            .sum::<f64>()
            > 1e-6
    );
    Ok(())
}

#[test]
fn pinned_julia_stout_is_synchronous_and_compares_every_component() -> Result<(), GaugeError> {
    let directory = fixture_dir();
    let metadata: Value =
        serde_json::from_slice(&fs::read(directory.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["schema"].as_str(), Some("stout_task_c.v1"));
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["nc"].as_u64(), Some(3));
    assert_eq!(
        metadata["gaugefields_jl"]["commit"].as_str(),
        Some("9e5719970770f4497405a856315c90bef7f74449")
    );
    assert_eq!(
        metadata["comparison"]["field_max_abs_tolerance"].as_f64(),
        Some(FIELD_TOLERANCE)
    );
    assert_eq!(
        metadata["update"].as_str(),
        Some("all links use one unchanged input snapshot")
    );

    let input =
        load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2"))?;
    let mut input_values = Vec::with_capacity(4);
    for mu in 0..4 {
        let expected = read_complex_npy(&directory.join(format!("u{mu}.npy")));
        let actual = input.links().links()[mu].typed().host_data().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.re.to_bits(),
                expected.re.to_bits(),
                "input mu={mu}, index={index}"
            );
            assert_eq!(
                actual.im.to_bits(),
                expected.im.to_bits(),
                "input mu={mu}, index={index}"
            );
        }
        input_values.push(expected);
    }
    let before = capture_links(input.links());
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let mut outputs = Vec::new();
    let mut reusable_cache = None;

    for (rho, label) in [(0.12, "plus"), (-0.07, "minus")] {
        let output = stout_step(&mut context, input.links(), rho)?;
        let mut max_residual = 0.0_f64;
        let mut output_difference = 0.0_f64;
        for (mu, input_direction) in input_values.iter().enumerate() {
            let expected = read_complex_npy(&directory.join(format!("stout_{label}{mu}.npy")));
            let actual = output.links()[mu].typed().host_data().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let residual = (*actual - *expected).norm();
                assert!(residual.is_finite());
                max_residual = max_residual.max(residual);
                output_difference =
                    output_difference.max((*actual - input_direction[index]).norm());
            }
        }
        assert!(
            output_difference > 1e-6,
            "rho={rho} produced a trivial output"
        );
        assert!(
            max_residual <= FIELD_TOLERANCE,
            "rho={rho} residual={max_residual:e}"
        );
        let drift = su3_residual(&output);
        assert!(drift.0 <= 2e-12, "rho={rho} unitarity residual={}", drift.0);
        assert!(
            drift.1 <= 2e-12,
            "rho={rho} determinant residual={}",
            drift.1
        );
        println!(
            "Task C rho={rho}: Julia max={max_residual:.17e}, unitarity={:.17e}, determinant={:.17e}",
            drift.0,
            drift.1
        );
        let cache_stats = context.cache_stats();
        if let Some(previous) = reusable_cache.as_ref() {
            assert_eq!(&cache_stats, previous);
        } else {
            reusable_cache = Some(cache_stats);
        }
        outputs.push(output);
        assert_links_bitwise_unchanged(input.links(), &before);
    }

    let repeated = stout_step(&mut context, input.links(), 0.12)?;
    for mu in 0..4 {
        assert_eq!(
            repeated.links()[mu].typed().host_data().unwrap(),
            outputs[0].links()[mu].typed().host_data().unwrap()
        );
    }
    assert_links_bitwise_unchanged(input.links(), &before);
    Ok(())
}

#[test]
fn numerical_failure_is_typed_and_does_not_mutate_input() -> Result<(), GaugeError> {
    let fixture =
        load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2"))?;
    let links = fixture.links().try_clone()?;
    let before = capture_links(&links);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let result = stout_step(&mut context, &links, f64::MAX);
    assert!(matches!(
        result,
        Err(GaugeError::Su3NumericalRange {
            operation: "stout_step",
            stage: "TA coefficients",
        })
    ));
    assert_links_bitwise_unchanged(&links, &before);
    Ok(())
}

#[test]
fn stout_outputs_have_the_expected_su3_residual() -> Result<(), GaugeError> {
    let fixture =
        load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2"))?;
    let links = fixture.links();
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    for rho in [0.12, -0.07] {
        let output = stout_step(&mut context, links, rho)?;
        let (unitary, determinant) = su3_residual(&output);
        assert!(unitary <= 2e-12, "rho={rho} unitarity residual={unitary:e}");
        assert!(
            determinant <= 2e-12,
            "rho={rho} determinant residual={determinant:e}"
        );
    }
    Ok(())
}
