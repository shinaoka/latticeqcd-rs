use super::*;
use tenferro_cpu::CpuBackend;

#[test]
fn injected_evolution_failure_is_transactional_for_links_and_momentum() {
    let lattice = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    let mut links = crate::cold_su3(lattice).unwrap();
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4]).unwrap();
    let mut momentum = sample_momentum(lattice, &mut rng).unwrap();
    let link_before: [Vec<_>; 4] =
        std::array::from_fn(|mu| links.links()[mu].typed().host_data().unwrap().to_vec());
    let momentum_before: [Vec<_>; 4] =
        std::array::from_fn(|mu| momentum.tensors()[mu].host_data().unwrap().to_vec());
    let params = HmcParams::new(5.7, 0.01, 2).unwrap();
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let result = leapfrog_trajectory_with(
        &mut context,
        &mut links,
        &mut momentum,
        params,
        &mut |_context, _links, _t, _momentum| {
            Err(GaugeError::Evolution {
                operation: "HMC test injection",
                source: tenferro_tensor::Error::backend_failure(
                    "HMC test injection",
                    "direction failure",
                ),
            })
        },
    );
    assert!(matches!(result, Err(GaugeError::Evolution { .. })));
    for (mu, expected) in link_before.iter().enumerate() {
        for (actual, expected) in links.links()[mu]
            .typed()
            .host_data()
            .unwrap()
            .iter()
            .zip(expected)
        {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }
    for (mu, expected) in momentum_before.iter().enumerate() {
        for (actual, expected) in momentum.tensors()[mu]
            .host_data()
            .unwrap()
            .iter()
            .zip(expected)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}
