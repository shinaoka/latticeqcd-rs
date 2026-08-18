use gaugefields::{
    cold_su3, exp_ta, load_fixture, load_link, site_index, store_link, GaugeLinks, LatticeShape4,
    Mat3,
};
use npyz::Order;
use num_complex::Complex64;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};
use wilsonloop::{
    evaluate_path, loop_action_force, loop_action_value, loop_trace_sum, LoopAction, LoopTerm,
    WilsonError, WilsonPath,
};

fn scalar_matrix(value: f64) -> Mat3 {
    let mut matrix = Mat3::zero();
    for i in 0..3 {
        matrix[(i, i)] = Complex64::new(value, 0.0);
    }
    matrix
}

fn scalar_links(lattice: LatticeShape4) -> GaugeLinks {
    let mut links = cold_su3(lattice).unwrap();
    for mu in 0..4 {
        for site in 0..lattice.nv() {
            store_link(
                &mut links,
                mu,
                site,
                scalar_matrix(1.0 + 10.0 * mu as f64 + site as f64),
            )
            .unwrap();
        }
    }
    links
}

fn varied_link(
    base: &GaugeLinks,
    direction: usize,
    site: usize,
    component: usize,
    sign: f64,
    h: f64,
) -> GaugeLinks {
    let mut links = base.try_clone().unwrap();
    let u = load_link(&links, direction, site).unwrap();
    let mut coefficients = [0.0; 8];
    coefficients[component] = 1.0;
    let left = exp_ta(sign * h, &coefficients).unwrap();
    store_link(&mut links, direction, site, left.mul(u)).unwrap();
    links
}

fn mixed_action() -> LoopAction {
    let mut terms = Vec::with_capacity(18);
    for mu in 1..=3 {
        for nu in (mu + 1)..=4 {
            terms.push(LoopTerm::plaquette(0.73, mu, nu).unwrap());
            terms.extend(LoopTerm::rectangle_1x2(-0.31, mu, nu).unwrap());
        }
    }
    LoopAction::new(terms).unwrap()
}

fn random_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2")
}

fn task_b_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/wilsonloop_task_b")
}

fn read_complex_npy(path: &Path, shape: &[u64]) -> Vec<Complex64> {
    let bytes = fs::read(path).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(npy.order(), Order::Fortran);
    assert_eq!(npy.shape(), shape);
    npy.into_vec::<Complex64>().unwrap()
}

fn read_real_npy(path: &Path, shape: &[u64]) -> Vec<f64> {
    let bytes = fs::read(path).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(npy.order(), Order::Fortran);
    assert_eq!(npy.shape(), shape);
    npy.into_vec::<f64>().unwrap()
}

#[test]
fn loop_action_clone_preserves_terms_and_evaluation() {
    let fixture = gaugefields::load_fixture(random_fixture_dir()).unwrap();
    let links = fixture.links();
    let action = mixed_action();
    let cloned = action.clone();

    assert!(!format!("{cloned:?}").is_empty());
    assert_eq!(action.terms().len(), cloned.terms().len());
    for (term, clone) in action.terms().iter().zip(cloned.terms()) {
        assert_eq!(term.coefficient(), clone.coefficient());
        assert_eq!(term.path().steps(), clone.path().steps());
    }
    assert_eq!(
        loop_action_value(links, &action).unwrap(),
        loop_action_value(links, &cloned).unwrap()
    );
    let force = loop_action_force(links, &action).unwrap();
    let cloned_force = loop_action_force(links, &cloned).unwrap();
    for direction in 0..4 {
        for site in 0..links.lattice().nv() {
            assert_eq!(
                force.site_coefficients(direction, site).unwrap(),
                cloned_force.site_coefficients(direction, site).unwrap()
            );
        }
    }
}

#[test]
fn pinned_julia_task_b_force_parity_uses_every_fixture_array() {
    let directory = task_b_fixture_dir();
    let metadata: Value =
        serde_json::from_slice(&fs::read(directory.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["schema"].as_str(), Some("wilsonloop_task_b.v1"));
    assert_eq!(metadata["lattice"], serde_json::json!([2, 2, 2, 2]));
    assert_eq!(metadata["nc"].as_u64(), Some(3));
    assert_eq!(
        metadata["gaugefields_jl"]["version"].as_str(),
        Some("0.7.2")
    );
    assert_eq!(
        metadata["gaugefields_jl"]["commit"].as_str(),
        Some("9e5719970770f4497405a856315c90bef7f74449")
    );
    assert_eq!(metadata["wilsonloop_jl"]["version"].as_str(), Some("0.1.5"));
    assert_eq!(
        metadata["wilsonloop_jl"]["commit"].as_str(),
        Some("e1a617fdedb19b785f89bdeb13c30e53b20743a7")
    );
    assert_eq!(
        metadata["coefficient_mapping"].as_str(),
        Some("Rust c=2*f because Julia inserts f*W and f*W†; Rust evaluates c*sum_x Re tr(W)")
    );
    assert_eq!(
        metadata["planes"],
        serde_json::json!([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
    );
    assert_eq!(metadata["expanded_rust_terms"].as_u64(), Some(18));
    assert_eq!(
        metadata["per_plane_terms"][0]["name"].as_str(),
        Some("plaquette")
    );
    assert_eq!(
        metadata["per_plane_terms"][0]["julia_coefficient_f"].as_f64(),
        Some(0.365)
    );
    assert_eq!(
        metadata["per_plane_terms"][0]["rust_coefficient_c"].as_f64(),
        Some(0.73)
    );
    assert_eq!(
        metadata["per_plane_terms"][1]["name"].as_str(),
        Some("rectangle_nu_long")
    );
    assert_eq!(
        metadata["per_plane_terms"][1]["julia_coefficient_f"].as_f64(),
        Some(-0.155)
    );
    assert_eq!(
        metadata["per_plane_terms"][1]["rust_coefficient_c"].as_f64(),
        Some(-0.31)
    );
    assert_eq!(
        metadata["per_plane_terms"][2]["name"].as_str(),
        Some("rectangle_mu_long")
    );
    assert_eq!(
        metadata["per_plane_terms"][2]["julia_coefficient_f"].as_f64(),
        Some(-0.155)
    );
    assert_eq!(
        metadata["per_plane_terms"][2]["rust_coefficient_c"].as_f64(),
        Some(-0.31)
    );
    assert_eq!(
        metadata["force_mapping"].as_str(),
        Some("Julia calc_dSdU is holomorphic: each Rust occurrence uses c/2=f; for U -> exp((i/2)sum(v_a lambda_a)t)U, dS/dt=-sum(force_a v_a)")
    );

    let link_shape = [3, 3, 2, 2, 2, 2];
    let derivative_shape = link_shape;
    let force_shape = [8, 2, 2, 2, 2];
    let mut links_from_julia = Vec::with_capacity(4);
    let mut dsdu = Vec::with_capacity(4);
    let mut force_coefficients = Vec::with_capacity(4);
    for direction in 0..4 {
        let links = read_complex_npy(&directory.join(format!("u{direction}.npy")), &link_shape);
        let established = read_complex_npy(
            &random_fixture_dir().join(format!("u{direction}.npy")),
            &link_shape,
        );
        assert_eq!(
            links, established,
            "Task B link fixture differs at direction {direction}"
        );
        links_from_julia.push(links);
        dsdu.push(read_complex_npy(
            &directory.join(format!("dsdu{direction}.npy")),
            &derivative_shape,
        ));
        force_coefficients.push(read_real_npy(
            &directory.join(format!("force_coeff{direction}.npy")),
            &force_shape,
        ));
    }

    let fixture = gaugefields::load_fixture(random_fixture_dir()).unwrap();
    let view = fixture.links().host_view().unwrap();
    for (direction, expected_values) in links_from_julia.iter().enumerate() {
        for site in 0..fixture.links().lattice().nv() {
            let expected = Mat3::load(expected_values, site * 9).unwrap();
            assert_eq!(view.link(direction, site).unwrap(), expected);
        }
    }

    let mut max_ta_residual = 0.0_f64;
    for direction in 0..4 {
        for site in 0..fixture.links().lattice().nv() {
            let product = view
                .link(direction, site)
                .unwrap()
                .mul(Mat3::load(&dsdu[direction], site * 9).unwrap());
            let expected = product.gell_mann_coefficients();
            for component in 0..8 {
                let stored = force_coefficients[direction][site * 8 + component];
                let residual = (expected[component] - stored).abs();
                assert!(residual.is_finite());
                max_ta_residual = max_ta_residual.max(residual);
            }
        }
    }
    assert!(
        max_ta_residual <= 2.0e-12,
        "Julia TA(U*dsdu) residual={max_ta_residual:.17e}"
    );

    let action = mixed_action();
    let rust_force = loop_action_force(fixture.links(), &action).unwrap();
    let mut max_rust_residual = 0.0_f64;
    for (direction, expected_force) in force_coefficients.iter().enumerate() {
        let mut direction_magnitude = 0.0_f64;
        for site in 0..fixture.links().lattice().nv() {
            let actual = rust_force.site_coefficients(direction, site).unwrap();
            for component in 0..8 {
                let expected = expected_force[site * 8 + component];
                direction_magnitude = direction_magnitude.max(expected.abs());
                let residual = (actual[component] - expected).abs();
                assert!(residual.is_finite());
                max_rust_residual = max_rust_residual.max(residual);
            }
        }
        assert!(
            direction_magnitude > 1.0e-6,
            "Julia force direction {direction} is vacuous"
        );
    }
    println!(
        "Task B Julia parity: max TA(U*dsdu)-stored={max_ta_residual:.17e}, max Rust-stored={max_rust_residual:.17e}"
    );
    assert!(
        max_rust_residual <= 2.0e-12,
        "Rust loop_action_force residual={max_rust_residual:.17e}"
    );
}

#[test]
fn public_validation_covers_all_constructor_branches() {
    assert!(matches!(
        WilsonPath::new(Vec::<i8>::new()),
        Err(WilsonError::EmptyPath)
    ));
    for step in [0, 5, -5, i8::MIN] {
        assert!(matches!(
            WilsonPath::new(vec![step]),
            Err(WilsonError::InvalidStep { .. })
        ));
    }
    let original = vec![0i8];
    assert!(WilsonPath::new(original.clone()).is_err());
    assert_eq!(original, vec![0]);

    let open = WilsonPath::new(vec![1]).unwrap();
    assert_eq!(open.displacement(), [1, 0, 0, 0]);
    assert!(!open.is_closed());
    assert!(matches!(
        LoopTerm::new(f64::NAN, open.clone()),
        Err(WilsonError::NonFiniteCoefficient { .. })
    ));
    assert!(matches!(
        LoopTerm::new(f64::INFINITY, open.clone()),
        Err(WilsonError::NonFiniteCoefficient { .. })
    ));
    assert!(matches!(
        LoopTerm::new(1.0, open),
        Err(WilsonError::OpenPath { .. })
    ));
    assert!(matches!(
        LoopAction::new(Vec::<LoopTerm>::new()),
        Err(WilsonError::EmptyAction)
    ));
    assert!(matches!(
        WilsonPath::plaquette(0, 2),
        Err(WilsonError::InvalidAxis { .. })
    ));
    assert!(matches!(
        WilsonPath::plaquette(1, 5),
        Err(WilsonError::InvalidAxis { .. })
    ));
    assert!(matches!(
        WilsonPath::plaquette(2, 2),
        Err(WilsonError::RepeatedAxis { .. })
    ));
}

#[test]
fn adjoint_is_an_involution_and_reverses_displacement() {
    let path = WilsonPath::new(vec![1, 1, 2, -3, -1]).unwrap();
    assert_eq!(path.adjoint().adjoint(), path);
    assert_eq!(
        path.adjoint().displacement(),
        path.displacement().map(|x| -x)
    );
}

#[test]
fn forward_backward_and_periodic_wrap_use_exact_link_sites() {
    let lattice = LatticeShape4::new([2, 3, 2, 4]).unwrap();
    let links = scalar_links(lattice);
    let path = WilsonPath::plaquette(1, 2).unwrap();
    let origin = site_index([0, 0, 0, 0], lattice).unwrap();
    let x_plus_y_plus = site_index([1, 1, 0, 0], lattice).unwrap();
    let expected = scalar_matrix(1.0)
        .mul(scalar_matrix(12.0))
        .mul(scalar_matrix(3.0))
        .mul(scalar_matrix(11.0));
    assert_eq!(evaluate_path(&links, origin, &path).unwrap(), expected);

    let wrap_forward = evaluate_path(
        &links,
        site_index([1, 0, 0, 0], lattice).unwrap(),
        &WilsonPath::new(vec![1]).unwrap(),
    )
    .unwrap();
    assert_eq!(wrap_forward, scalar_matrix(2.0));
    let wrap_backward = evaluate_path(&links, origin, &WilsonPath::new(vec![-1]).unwrap()).unwrap();
    assert_eq!(wrap_backward, scalar_matrix(2.0));
    assert_eq!(x_plus_y_plus, 3);
}

#[test]
fn cold_open_closed_trace_and_action_normalization_are_exact() {
    let lattice = LatticeShape4::new([2, 3, 2, 4]).unwrap();
    let links = cold_su3(lattice).unwrap();
    let open = WilsonPath::new(vec![1, 4, -2]).unwrap();
    assert_eq!(evaluate_path(&links, 0, &open).unwrap(), Mat3::identity());
    assert_eq!(
        loop_trace_sum(&links, &open).unwrap(),
        Complex64::new(3.0 * lattice.nv() as f64, 0.0)
    );
    let action = LoopAction::new(vec![LoopTerm::plaquette(0.25, 1, 2).unwrap()]).unwrap();
    assert_eq!(
        loop_action_value(&links, &action).unwrap(),
        0.25 * 3.0 * lattice.nv() as f64
    );
    let force = loop_action_force(&links, &action).unwrap();
    for direction in 0..4 {
        for site in 0..lattice.nv() {
            assert_eq!(force.site_coefficients(direction, site).unwrap(), [0.0; 8]);
        }
    }
}

#[test]
fn invalid_origin_is_typed() {
    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let error = evaluate_path(&links, 1, &WilsonPath::new(vec![1]).unwrap()).unwrap_err();
    assert!(matches!(
        error,
        WilsonError::OriginOutOfBounds {
            origin: 1,
            volume: 1
        }
    ));
}

#[test]
fn independent_centered_left_variations_match_plaquette_and_mixed_force() {
    let base = load_fixture(random_fixture_dir())
        .unwrap()
        .links()
        .try_clone()
        .unwrap();
    let plaquette = LoopAction::new(vec![LoopTerm::plaquette(0.73, 3, 4).unwrap()]).unwrap();
    let mixed = mixed_action();
    for action in [&plaquette, &mixed] {
        for term in action.terms() {
            assert!(term.path().steps().iter().any(|step| *step > 0));
            assert!(term.path().steps().iter().any(|step| *step < 0));
        }
    }
    let plaquette_force = loop_action_force(&base, &plaquette).unwrap();
    let mixed_force = loop_action_force(&base, &mixed).unwrap();
    let cases = [
        ("plaquette", &plaquette, &plaquette_force, 2, 0, 0),
        ("plaquette", &plaquette, &plaquette_force, 3, 3, 7),
        ("mixed", &mixed, &mixed_force, 0, 5, 3),
        ("mixed", &mixed, &mixed_force, 1, 7, 6),
        ("mixed", &mixed, &mixed_force, 0, 10, 1),
        ("plaquette", &plaquette, &plaquette_force, 2, 12, 5),
    ];
    let h = 2.0e-6;
    let mut max_residual = 0.0_f64;
    for &(name, action, force, direction, site, component) in &cases {
        // Centered left variation: U -> exp(±h (i/2) λ_component) U.
        let plus = loop_action_value(
            &varied_link(&base, direction, site, component, 1.0, h),
            action,
        )
        .unwrap();
        let minus = loop_action_value(
            &varied_link(&base, direction, site, component, -1.0, h),
            action,
        )
        .unwrap();
        let finite_difference = (plus - minus) / (2.0 * h);
        let force_component = force.site_coefficients(direction, site).unwrap()[component];
        // Exact Julia convention for c=2*f: dS/dt = -sum_a force_a * v_a.
        let residual = (finite_difference + force_component).abs();
        assert!(
            finite_difference.abs() > 1.0e-4,
            "{name} case is vacuous: fd={finite_difference}"
        );
        assert!(
            force_component.abs() > 1.0e-4,
            "{name} case is vacuous: force={force_component}"
        );
        assert!(
            residual < 2.0e-8,
            "{name} direction={direction} site={site} component={component} fd={finite_difference} force={force_component} residual={residual}"
        );
        max_residual = max_residual.max(residual);
    }
    println!("Task B finite differences: max residual={max_residual:.17e}");
}

#[test]
fn source_boundary_has_no_second_site_indexing_path() {
    let source = include_str!("../src/evaluate.rs");
    assert!(!source.contains("site_index("));
    assert!(!source.contains(".links()"));
    assert!(!source.contains("TypedTensor"));
    let value_source = source
        .split("pub fn loop_action_value")
        .nth(1)
        .unwrap()
        .split("pub fn loop_action_force")
        .next()
        .unwrap();
    assert!(!value_source.contains(".zip("));
    assert_eq!(source.matches("host_view()?").count(), 4);
}
