#![cfg(feature = "autodiff")]

use gaugefields::{
    action_gradient, ad_rules, load_fixture, runtime_modules, wilson_action, wilson_action_traced,
    GaugeLinkTensor, GaugeLinks,
};
use num_complex::Complex64;
use std::path::Path;
use tenferro_ad::AdContext;
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_runtime::program::SemanticOpRef;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::TypedTensor;

fn link(value: f64) -> TracedTensor {
    TracedTensor::from_vec_col_major(
        vec![3, 3, 1, 1, 1, 1],
        (0..9)
            .map(|i| Complex64::new(value + i as f64 / 17.0, i as f64 / 23.0))
            .collect(),
    )
    .unwrap()
}

#[test]
fn active_direction_payload_omits_inactive_tangents() {
    for active_dirs in [vec![1], vec![0, 2], vec![0, 1, 2, 3]] {
        let source = link(1.0);
        let fixed: [TracedTensor; 4] = std::array::from_fn(|mu| link(2.0 + mu as f64));
        let tangent = link(0.25);
        // `AdContext::jvp` accepts one `wrt`; aliasing that one traced source
        // into selected action slots deliberately exercises multi-slot activity.
        let action_inputs: [&TracedTensor; 4] = std::array::from_fn(|mu| {
            if active_dirs.contains(&mu) {
                &source
            } else {
                &fixed[mu]
            }
        });
        let action = wilson_action_traced(action_inputs, 5.7).unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(ad_rules().unwrap())
            .unwrap()
            .build()
            .unwrap();
        let jvp = ad.jvp(&action, &source, &tangent).unwrap();

        let program = GraphCompiler::new().compile(&jvp).unwrap();
        let matching = program
            .program()
            .operations()
            .filter_map(|operation| match operation.op() {
                SemanticOpRef::Extension(op)
                    if op.family_id() == "gaugefields.wilson_action_jvp.v1" =>
                {
                    Some((
                        format!("{op:?}"),
                        op.input_count(),
                        operation.inputs().len(),
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "active_dirs={active_dirs:?}");
        let expected_arity = 4 + active_dirs.len();
        assert_eq!(matching[0].1, expected_arity);
        assert_eq!(matching[0].2, expected_arity);
        assert!(
            matching[0]
                .0
                .contains(&format!("active_dirs: {active_dirs:?}")),
            "payload={} expected active_dirs={active_dirs:?}",
            matching[0].0
        );
    }
}

fn tangent(mu: usize, len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            Complex64::new(
                (1 + i + 3 * mu) as f64 / 211.0,
                -((2 + 2 * i + mu) as f64) / 307.0,
            )
        })
        .collect()
}

fn perturbed(base: &GaugeLinks, mu: usize, tangent: &[Complex64], scale: f64) -> GaugeLinks {
    let links = std::array::from_fn(|direction| {
        let mut data = base.links()[direction]
            .typed()
            .host_data()
            .unwrap()
            .to_vec();
        if direction == mu {
            for (value, delta) in data.iter_mut().zip(tangent) {
                *value += scale * delta;
            }
        }
        GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(base.links()[direction].typed().shape().to_vec(), data)
                .unwrap(),
            base.lattice(),
        )
        .unwrap()
    });
    GaugeLinks::new(links).unwrap()
}

#[test]
fn jvp_matches_finite_difference_sweep_and_gradient_inner_product() {
    let fixture =
        load_fixture(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/random_2x2x2x2"))
            .unwrap();
    let beta = fixture.metadata().beta;
    let base = fixture.links();
    let gradient = action_gradient(base, beta).unwrap();
    for mu in 0..4 {
        let values = tangent(mu, 9 * base.lattice().nv());
        let traced_links: [TracedTensor; 4] = std::array::from_fn(|direction| {
            TracedTensor::from_vec_col_major(
                base.links()[direction].typed().shape().to_vec(),
                base.links()[direction]
                    .typed()
                    .host_data()
                    .unwrap()
                    .to_vec(),
            )
            .unwrap()
        });
        let traced_tangent =
            TracedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values.clone()).unwrap();
        let action = wilson_action_traced(
            [
                &traced_links[0],
                &traced_links[1],
                &traced_links[2],
                &traced_links[3],
            ],
            beta,
        )
        .unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(ad_rules().unwrap())
            .unwrap()
            .build()
            .unwrap();
        let traced = ad.jvp(&action, &traced_links[mu], &traced_tangent).unwrap();
        let program = GraphCompiler::new().compile(&traced).unwrap();
        let backend = CpuBackend::new();
        let mut builder = Runtime::builder();
        builder
            .register_engine(runtime_engine_registration(&backend).unwrap())
            .unwrap();
        for module in runtime_modules::<CpuBackend>(runtime_engine_id().unwrap()).unwrap() {
            builder.install_extension_module(module).unwrap();
        }
        let actual = builder
            .build()
            .unwrap()
            .run_compiled(&program, &[])
            .unwrap()[0]
            .as_slice::<f64>()
            .unwrap()[0];
        let direct = gradient[mu]
            .typed()
            .host_data()
            .unwrap()
            .iter()
            .zip(&values)
            .map(|(g, delta)| (g.conj() * delta).re)
            .sum::<f64>();
        let residuals = [1e-2, 5e-3, 2.5e-3, 1.25e-3].map(|h| {
            let fd = (wilson_action(&perturbed(base, mu, &values, h), beta).unwrap()
                - wilson_action(&perturbed(base, mu, &values, -h), beta).unwrap())
                / (2.0 * h);
            (h, fd, (actual - fd).abs())
        });
        let best = residuals.iter().min_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
        let direct_residual = (actual - direct).abs();
        assert!(actual.is_finite() && direct.is_finite() && best.1.is_finite());
        assert!(
            direct_residual < 1e-10,
            "mu={mu} actual={actual} direct={direct} residual={direct_residual}"
        );
        assert!(
            best.2 < 1e-7,
            "mu={mu} actual={actual} best_h={} fd={} residual={} sweep={residuals:?}",
            best.0,
            best.1,
            best.2
        );
    }

    let links: [TracedTensor; 4] = std::array::from_fn(|_| link(1.0));
    assert!(wilson_action_traced([&links[0], &links[1], &links[2], &links[3]], f64::NAN).is_err());
}
