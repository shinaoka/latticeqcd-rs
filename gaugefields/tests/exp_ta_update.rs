use gaugefields::{
    cold_su3, exp_ta, exp_ta_update, CpuEvolutionContext, GaugeError, LatticeShape4, Mat3,
    TaGaugeField,
};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::CacheStats;
use tenferro_tensor::TypedTensor;

use std::time::Instant;

fn momentum(lattice: LatticeShape4) -> TaGaugeField {
    let [nx, ny, nz, nt] = lattice.extents();
    let tensors = std::array::from_fn(|mu| {
        let mut values = vec![0.0; 8 * lattice.nv()];
        for site in 0..lattice.nv() {
            for a in 0..8 {
                values[8 * site + a] = (1 + mu + site + a) as f64 / 37.0;
            }
        }
        TypedTensor::from_vec_col_major(vec![8, nx, ny, nz, nt], values).unwrap()
    });
    TaGaugeField::new(tensors, lattice).unwrap()
}

#[test]
fn nonzero_update_matches_explicit_mat3_and_reuses_cache() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([2, 1, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let momentum = momentum(lattice);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let t = 0.125;
    exp_ta_update(&mut context, &mut links, t, &momentum)?;
    for mu in 0..4 {
        let coefficients: [f64; 8] = momentum.tensors()[mu].host_data().unwrap()[..8]
            .try_into()
            .unwrap();
        let expected = exp_ta(t, &coefficients)?;
        let actual = &links.links()[mu].typed().host_data().unwrap()[..9];
        let residual = actual
            .iter()
            .zip(expected.as_array())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        assert!(residual < 2e-12, "mu={mu} residual={residual}");
    }
    let first = context.cache_stats();
    exp_ta_update(&mut context, &mut links, t, &momentum)?;
    assert_eq!(context.cache_stats(), first);
    context.clear_cache();
    assert_eq!(context.cache_stats().entries, 0);
    Ok(())
}

#[test]
fn context_contract_cache_and_zero_update() -> Result<(), GaugeError> {
    let _: fn(CpuBackend) -> CpuEvolutionContext = CpuEvolutionContext::new;
    let _: fn(&CpuEvolutionContext) -> &CpuBackend = CpuEvolutionContext::backend;
    let _: fn(&mut CpuEvolutionContext) = CpuEvolutionContext::clear_cache;
    let _: fn(&CpuEvolutionContext) -> CacheStats = CpuEvolutionContext::cache_stats;
    let _: fn(
        &mut CpuEvolutionContext,
        &mut gaugefields::GaugeLinks,
        f64,
        &TaGaugeField,
    ) -> Result<(), GaugeError> = exp_ta_update;

    let lattice = LatticeShape4::new([2, 2, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let before: [Vec<_>; 4] =
        std::array::from_fn(|mu| links.links()[mu].typed().host_data().unwrap().to_vec());
    let momentum = momentum(lattice);
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    exp_ta_update(&mut context, &mut links, 0.0, &momentum)?;
    for (mu, expected) in before.iter().enumerate() {
        assert_eq!(links.links()[mu].typed().host_data().unwrap(), expected);
    }
    assert!(!format!("{context:?}").contains("cache:"));
    Ok(())
}

#[test]
fn cancelling_momentum_propagates_through_field_update() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let tensors = std::array::from_fn(|_| {
        TypedTensor::from_vec_col_major(
            vec![8, 1, 1, 1, 1],
            vec![1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap()
    });
    let momentum = TaGaugeField::new(tensors, lattice)?;
    let mut links = cold_su3(lattice)?;
    exp_ta_update(
        &mut CpuEvolutionContext::new(CpuBackend::new()),
        &mut links,
        1.0,
        &momentum,
    )?;
    let expected = exp_ta(1.0, &[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])?;
    for link in links.links() {
        let residual = link
            .typed()
            .host_data()
            .unwrap()
            .iter()
            .zip(expected.as_array())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        assert!(residual < 2e-12);
        assert_ne!(
            link.typed().host_data().unwrap(),
            Mat3::identity().as_array()
        );
    }
    Ok(())
}

#[test]
#[ignore = "release-only scaling diagnostic"]
fn release_update_scaling() -> Result<(), GaugeError> {
    for extent in [2, 4, 8] {
        let lattice = LatticeShape4::new([extent; 4])?;
        let mut links = cold_su3(lattice)?;
        let momentum = momentum(lattice);
        let mut context = CpuEvolutionContext::new(CpuBackend::new());
        exp_ta_update(&mut context, &mut links, 0.01, &momentum)?;
        let started = Instant::now();
        for _ in 0..3 {
            exp_ta_update(&mut context, &mut links, 0.01, &momentum)?;
        }
        let elapsed = started.elapsed();
        eprintln!(
            "evolution scaling extent={extent} sites={} elapsed_ms={:.3} ns_per_site_step={:.1}",
            lattice.nv(),
            elapsed.as_secs_f64() * 1e3,
            elapsed.as_nanos() as f64 / (3 * lattice.nv()) as f64
        );
    }
    Ok(())
}
