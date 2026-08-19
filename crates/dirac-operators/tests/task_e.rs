use dirac_operators::{FermionBoundary, SolverParams, StaggeredFermiAction, StaggeredHmcParams};
use gaugefields::{LatticeShape4, ReproducibleRng};
use rand::RngCore;

#[test]
fn task_e_rejects_claimed_bounds_outside_pinned_table() {
    let boundary = FermionBoundary::new([1, 1, 1, -1]).unwrap();
    let solver = SolverParams::new(1.0e-24, 2_000).unwrap();
    assert!(StaggeredFermiAction::new(0.17, boundary, 0.0003, 64.0, solver).is_err());
    assert!(StaggeredFermiAction::new(0.17, boundary, 0.0004, 65.0, solver).is_err());
}

#[test]
fn task_e_action_contract_has_one_component() {
    let lattice = LatticeShape4::new([1, 1, 1, 1]).unwrap();
    let boundary = FermionBoundary::new([1, 1, 1, -1]).unwrap();
    let solver = SolverParams::new(1.0e-24, 2_000).unwrap();
    let action = StaggeredFermiAction::new(0.17, boundary, 0.0004, 64.0, solver).unwrap();
    assert_eq!(action.mass(), 0.17);
    assert_eq!(lattice.nv(), 1);
}

#[test]
fn task_e_rejected_hmc_rolls_back_links_but_not_rng() {
    let lattice = LatticeShape4::new([1, 1, 1, 1]).unwrap();
    let boundary = FermionBoundary::new([1, 1, 1, -1]).unwrap();
    let params = StaggeredHmcParams::new(
        5.7,
        0.17,
        1.0,
        2,
        boundary,
        0.0004,
        64.0,
        SolverParams::new(1.0e-20, 256).unwrap(),
    )
    .unwrap();
    let mut found_rejection = false;
    for seed in 1..=64_u64 {
        let mut links = gaugefields::cold_su3(lattice).unwrap();
        let before = links.try_clone().unwrap();
        let state = [seed, seed + 1, seed + 2, seed + 3];
        let mut rng = ReproducibleRng::from_state(state).unwrap();
        let mut replay = ReproducibleRng::from_state(state).unwrap();
        let _ = gaugefields::sample_momentum(lattice, &mut replay).unwrap();
        let _ = params.action().sample_xi(lattice, &mut replay).unwrap();
        let mut evolution = gaugefields::CpuEvolutionContext::new(tenferro_cpu::CpuBackend::new());
        let outcome =
            dirac_operators::staggered_hmc_update(&mut evolution, &mut links, params, &mut rng)
                .unwrap();
        if !outcome.accepted {
            found_rejection = true;
            for direction in 0..4 {
                assert_eq!(
                    links.links()[direction].typed().host_data().unwrap(),
                    before.links()[direction].typed().host_data().unwrap()
                );
            }
            let _ = replay.open_unit_f64();
            assert_eq!(rng.next_u64(), replay.next_u64());
            break;
        }
    }
    assert!(
        found_rejection,
        "the deterministic seed sweep found no rejection"
    );
}
