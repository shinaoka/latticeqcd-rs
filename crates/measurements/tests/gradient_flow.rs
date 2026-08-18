use gaugefields::{
    cold_su3, load_fixture, normalized_plaquette, store_link, CpuEvolutionContext, GaugeError,
    GaugeLinks, LatticeShape4, Mat3,
};
use measurements::{gradient_flow, GradientFlowParams, MeasurementError};
use npyz::{NpyFile, Order};
use num_complex::Complex64;
use serde_json::Value;
use std::{fs, path::Path};
use tenferro_cpu::CpuBackend;
use wilsonloop::{LoopAction, LoopTerm};

const LINK_SHAPE: [u64; 6] = [3, 3, 2, 2, 2, 2];
const FIELD_TOLERANCE: f64 = 5e-12;

fn random_fixture() -> Result<gaugefields::Fixture, GaugeError> {
    load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2"))
}

fn flow_fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gradientflow_task_d2")
}

fn read_complex_npy(path: &Path) -> Vec<Complex64> {
    let bytes = fs::read(path).expect("flow fixture file");
    let npy = NpyFile::new(&bytes[..]).expect("valid flow NPY");
    assert_eq!(npy.order(), Order::Fortran);
    assert_eq!(npy.shape(), &LINK_SHAPE);
    npy.into_vec::<Complex64>().expect("Complex64 flow NPY")
}

fn capture_links(links: &GaugeLinks) -> [Vec<Complex64>; 4] {
    std::array::from_fn(|direction| {
        links.links()[direction]
            .typed()
            .host_data()
            .expect("host links")
            .to_vec()
    })
}

fn assert_links_bitwise_equal(links: &GaugeLinks, expected: &[Vec<Complex64>; 4]) {
    for (direction, values) in expected.iter().enumerate() {
        let actual = links.links()[direction]
            .typed()
            .host_data()
            .expect("host links");
        assert_eq!(actual.len(), values.len());
        for (index, (actual, expected)) in actual.iter().zip(values).enumerate() {
            assert_eq!(
                actual.re.to_bits(),
                expected.re.to_bits(),
                "mu={direction} i={index}"
            );
            assert_eq!(
                actual.im.to_bits(),
                expected.im.to_bits(),
                "mu={direction} i={index}"
            );
        }
    }
}

fn wilson_action() -> LoopAction {
    let mut terms = Vec::with_capacity(6);
    for mu in 1..=3 {
        for nu in (mu + 1)..=4 {
            // Julia f=0.5 inserts f*W and f*W†; Rust c=2*f=1.0.
            terms.push(LoopTerm::plaquette(1.0, mu, nu).expect("plaquette term"));
        }
    }
    LoopAction::new(terms).expect("Wilson action")
}

fn mixed_action() -> LoopAction {
    let mut terms = Vec::with_capacity(18);
    for mu in 1..=3 {
        for nu in (mu + 1)..=4 {
            // Pinned Julia f values are mapped to Rust c=2*f.
            terms.push(LoopTerm::plaquette(0.73, mu, nu).expect("plaquette term"));
            terms.extend(LoopTerm::rectangle_1x2(-0.31, mu, nu).expect("rectangle terms"));
        }
    }
    LoopAction::new(terms).expect("mixed action")
}

fn su3_residual(links: &GaugeLinks) -> (f64, f64) {
    let mut unitary = 0.0_f64;
    let mut determinant = 0.0_f64;
    for link in links.links() {
        for block in link
            .typed()
            .host_data()
            .expect("host links")
            .chunks_exact(9)
        {
            let matrix = Mat3::load(block, 0).expect("matrix block");
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

fn compare_output(links: &GaugeLinks, prefix: &str) -> f64 {
    let directory = flow_fixture_dir();
    let mut residual = 0.0_f64;
    for direction in 0..4 {
        let expected = read_complex_npy(&directory.join(format!("{prefix}{direction}.npy")));
        let actual = links.links()[direction]
            .typed()
            .host_data()
            .expect("host output");
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(actual.re.is_finite() && actual.im.is_finite());
            residual = residual.max((actual.re - expected.re).abs());
            residual = residual.max((actual.im - expected.im).abs());
            assert!(
                (actual.re - expected.re).abs() <= FIELD_TOLERANCE
                    && (actual.im - expected.im).abs() <= FIELD_TOLERANCE,
                "prefix={prefix} mu={direction} i={index} residual={residual:e}"
            );
        }
    }
    residual
}

fn max_link_difference(actual: &GaugeLinks, input: &GaugeLinks) -> [f64; 4] {
    std::array::from_fn(|direction| {
        actual.links()[direction]
            .typed()
            .host_data()
            .expect("host output")
            .iter()
            .zip(
                input.links()[direction]
                    .typed()
                    .host_data()
                    .expect("host input"),
            )
            .map(|(actual, input)| (*actual - *input).norm())
            .fold(0.0, f64::max)
    })
}

fn assert_nonzero_force(
    action: &LoopAction,
    links: &GaugeLinks,
) -> Result<(), Box<dyn std::error::Error>> {
    let force = wilsonloop::loop_action_force(links, action)?;
    for direction in 0..4 {
        let mut magnitude = 0.0_f64;
        for site in 0..links.lattice().nv() {
            for value in force.site_coefficients(direction, site)? {
                magnitude = magnitude.max(value.abs());
            }
        }
        assert!(magnitude > 1e-8, "vacuous force direction {direction}");
    }
    Ok(())
}

#[test]
fn gradient_flow_params_validate_every_branch_and_expose_values() {
    for step_size in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            GradientFlowParams::new(step_size, 1),
            Err(MeasurementError::NonFiniteStepSize { .. })
        ));
    }
    for step_size in [-1.0, -0.0, 0.0] {
        assert!(matches!(
            GradientFlowParams::new(step_size, 1),
            Err(MeasurementError::NonPositiveStepSize { .. })
        ));
    }
    assert!(matches!(
        GradientFlowParams::new(0.01, 0),
        Err(MeasurementError::ZeroFlowSteps)
    ));

    let params = GradientFlowParams::new(0.01, 4).expect("valid parameters");
    assert_eq!(params.step_size(), 0.01);
    assert_eq!(params.steps(), 4);
    assert_eq!(params, params);
    assert!(!format!("{params:?}").is_empty());
}

#[test]
fn cold_field_is_exact_after_one_and_four_wilson_flow_steps(
) -> Result<(), Box<dyn std::error::Error>> {
    let links = cold_su3(LatticeShape4::new([2, 2, 2, 2])?)?;
    let before = capture_links(&links);
    let action = wilson_action();
    let mut context = CpuEvolutionContext::new(CpuBackend::new());

    for steps in [1, 4] {
        let output = gradient_flow(
            &mut context,
            &links,
            &action,
            GradientFlowParams::new(0.01, steps)?,
        )?;
        assert_links_bitwise_equal(&output, &before);
        assert_links_bitwise_equal(&links, &before);
    }
    Ok(())
}

#[test]
fn pinned_julia_rk3_outputs_consume_every_component_and_metadata(
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = flow_fixture_dir();
    let metadata: Value = serde_json::from_slice(&fs::read(directory.join("metadata.json"))?)?;
    assert_eq!(metadata["schema"], "gradientflow_task_d2.v1");
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["nc"], 3);
    assert_eq!(metadata["step_size"], 0.01);
    assert_eq!(metadata["field_tolerance"], FIELD_TOLERANCE);
    assert_eq!(
        metadata["gaugefields_jl"]["commit"],
        "9e5719970770f4497405a856315c90bef7f74449"
    );
    assert_eq!(metadata["gaugefields_jl"]["version"], "0.7.2");
    assert_eq!(
        metadata["wilsonloop_jl"]["commit"],
        "e1a617fdedb19b785f89bdeb13c30e53b20743a7"
    );
    assert_eq!(metadata["wilsonloop_jl"]["version"], "0.1.5");
    assert_eq!(metadata["actions"]["wilson"]["julia_f"], 0.5);
    assert_eq!(metadata["actions"]["wilson"]["rust_c"], 1.0);
    assert_eq!(metadata["actions"]["mixed"]["plaquette_julia_f"], 0.365);
    assert_eq!(metadata["actions"]["mixed"]["plaquette_rust_c"], 0.73);
    assert_eq!(metadata["actions"]["mixed"]["rectangle_julia_f"], -0.155);
    assert_eq!(metadata["actions"]["mixed"]["rectangle_rust_c"], -0.31);
    assert_eq!(
        metadata["coefficient_mapping"],
        "Rust c=2*f because Julia inserts f*W and f*W†; Rust evaluates c*sum_x Re tr(W)"
    );
    assert_eq!(metadata["force_mapping"], "Julia calc_dSdU is holomorphic: Rust loop_action_force uses c/2=f per occurrence; dS/dt=-sum(force_a*v_a), and RK3 supplies the negative flow coefficients");
    assert_eq!(metadata["routine_order"], "F0; W1=exp(-eps/4 F0)U; F1; W2=exp(eps*(-8/9 F1+17/36 F0))W1; F2; U'=exp(eps*(-3/4 F2+8/9 F1-17/36 F0))W2");
    assert_eq!(metadata["files"].as_array().expect("file list").len(), 16);
    assert!(
        metadata["source_functions"]
            .as_array()
            .expect("source functions")
            .len()
            >= 8
    );

    let input = random_fixture()?;
    let action = wilson_action();
    let mixed = mixed_action();
    assert_nonzero_force(&action, input.links())?;
    assert_nonzero_force(&mixed, input.links())?;

    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let one = gradient_flow(
        &mut context,
        input.links(),
        &action,
        GradientFlowParams::new(0.01, 1)?,
    )?;
    let four = gradient_flow(
        &mut context,
        input.links(),
        &action,
        GradientFlowParams::new(0.01, 4)?,
    )?;
    let mixed_one = gradient_flow(
        &mut context,
        input.links(),
        &mixed,
        GradientFlowParams::new(0.01, 1)?,
    )?;

    for (output, prefix) in [
        (&one, "flow_one"),
        (&four, "flow_four"),
        (&mixed_one, "flow_mixed"),
    ] {
        let residual = compare_output(output, prefix);
        let (unitary, determinant) = su3_residual(output);
        assert!(unitary <= 2e-12, "{prefix} unitarity residual={unitary:e}");
        assert!(
            determinant <= 2e-12,
            "{prefix} determinant residual={determinant:e}"
        );
        let differences = max_link_difference(output, input.links());
        assert!(
            differences.iter().all(|value| *value > 1e-8),
            "{prefix} trivial direction: {differences:?}"
        );
        println!(
            "Task D2 {prefix}: max={residual:.17e}, unitarity={unitary:.17e}, determinant={determinant:.17e}, differences={differences:?}"
        );
    }
    Ok(())
}

#[test]
fn wilson_flow_normalized_plaquette_increases_at_each_step(
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = random_fixture()?;
    let action = wilson_action();
    let input = fixture.links();
    let mut current = input.try_clone()?;
    let initial = normalized_plaquette(&current)?;
    let mut previous = initial;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());

    for step in 1..=4 {
        current = gradient_flow(
            &mut context,
            &current,
            &action,
            GradientFlowParams::new(0.01, 1)?,
        )?;
        let value = normalized_plaquette(&current)?;
        assert!(value.is_finite());
        assert!(
            value > previous,
            "step={step} previous={previous:.17e} value={value:.17e}"
        );
        previous = value;
    }
    assert!(previous - initial > 1e-6, "vacuous Wilson flow increase");
    println!("Task D2 normalized plaquette: initial={initial:.17e} final={previous:.17e}");
    Ok(())
}

#[test]
fn failures_are_transactional_and_context_is_reusable() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = random_fixture()?;
    let action = wilson_action();
    let links = fixture.links();
    let before = capture_links(links);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let initial_cache = context.cache_stats();

    let mut invalid = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
    let mut nan_link = Mat3::identity();
    nan_link[(0, 0)] = Complex64::new(f64::NAN, 0.0);
    store_link(&mut invalid, 0, 0, nan_link)?;
    let invalid_before = capture_links(&invalid);
    assert!(matches!(
        gradient_flow(
            &mut context,
            &invalid,
            &action,
            GradientFlowParams::new(0.01, 1)?,
        ),
        Err(MeasurementError::NonFiniteInput {
            direction: 0,
            site: 0,
            component: 0,
        })
    ));
    assert_eq!(context.cache_stats(), initial_cache);
    assert_links_bitwise_equal(&invalid, &invalid_before);

    let numerical = gradient_flow(
        &mut context,
        links,
        &action,
        GradientFlowParams::new(f64::MAX, 1)?,
    );
    assert!(matches!(
        numerical,
        Err(MeasurementError::NumericalRange { .. })
            | Err(MeasurementError::Gauge(
                GaugeError::Su3NumericalRange { .. }
            ))
    ));
    assert_links_bitwise_equal(links, &before);

    let first = gradient_flow(
        &mut context,
        links,
        &action,
        GradientFlowParams::new(0.01, 1)?,
    )?;
    let cache_after_first = context.cache_stats();
    let second = gradient_flow(
        &mut context,
        links,
        &action,
        GradientFlowParams::new(0.01, 1)?,
    )?;
    assert_eq!(context.cache_stats(), cache_after_first);
    assert_links_bitwise_equal(&first, &capture_links(&second));
    assert_links_bitwise_equal(links, &before);
    assert!(max_link_difference(&first, links)
        .iter()
        .any(|value| *value > 1e-8));
    Ok(())
}
