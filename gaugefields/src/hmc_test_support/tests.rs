use super::*;
use crate::{cold_su3, exp_ta_update, Mat3};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const BETA: f64 = 5.7;

fn hot_field(rng: &mut ChaCha8Rng) -> GaugeLinks {
    let lattice = LatticeShape4::new([4, 4, 4, 4]).unwrap();
    let mut links = cold_su3(lattice).unwrap();
    let momentum = random_momentum(lattice, rng, 1.5).unwrap();
    exp_ta_update(
        &mut CpuEvolutionContext::new(CpuBackend::new()),
        &mut links,
        1.0,
        &momentum,
    )
    .unwrap();
    links
}

fn link_residual(lhs: &GaugeLinks, rhs: &GaugeLinks) -> f64 {
    (0..4)
        .flat_map(|mu| {
            lhs.links()[mu]
                .typed()
                .host_data()
                .unwrap()
                .iter()
                .zip(rhs.links()[mu].typed().host_data().unwrap())
                .map(|(a, b)| (*a - *b).norm())
        })
        .fold(0.0, f64::max)
}

fn su3_drift(links: &GaugeLinks) -> (f64, f64) {
    let mut unitary: f64 = 0.0;
    let mut determinant: f64 = 0.0;
    for mu in 0..4 {
        for block in links.links()[mu]
            .typed()
            .host_data()
            .unwrap()
            .chunks_exact(9)
        {
            let matrix = Mat3::load(block, 0).unwrap();
            let product = matrix.adjoint().mul(matrix);
            for column in 0..3 {
                for row in 0..3 {
                    let expected = if row == column { 1.0 } else { 0.0 };
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
fn leapfrog_is_finite_su3_and_reversible() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x484d_435f_5355_3308);
    let initial = hot_field(&mut rng);
    let mut links = clone_links(&initial).unwrap();
    let mut momentum = random_momentum(initial.lattice(), &mut rng, 0.35).unwrap();
    let original_momentum = negate(&negate(&momentum).unwrap()).unwrap();
    trajectory(&mut links, &mut momentum, BETA, 0.01, 4).unwrap();
    momentum = negate(&momentum).unwrap();
    trajectory(&mut links, &mut momentum, BETA, 0.01, 4).unwrap();
    momentum = negate(&momentum).unwrap();
    let link_error = link_residual(&links, &initial);
    let momentum_error = momentum
        .tensors()
        .iter()
        .zip(original_momentum.tensors())
        .flat_map(|(a, b)| {
            a.host_data()
                .unwrap()
                .iter()
                .zip(b.host_data().unwrap())
                .map(|(a, b)| (a - b).abs())
        })
        .fold(0.0, f64::max);
    let (unitary, determinant) = su3_drift(&links);
    eprintln!("reversibility links={link_error:e} momentum={momentum_error:e} unitarity={unitary:e} determinant={determinant:e}");
    assert!(link_error < 2e-10 && momentum_error < 2e-10);
    assert!(unitary < 2e-10 && determinant < 2e-10);
    assert!(hamiltonian(&links, &momentum, BETA).unwrap().is_finite());
    assert!(links.links().iter().all(|link| link
        .typed()
        .host_data()
        .unwrap()
        .iter()
        .all(|z| z.re.is_finite() && z.im.is_finite())));
}

#[test]
fn energy_error_scales_second_order_and_short_run_accepts() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x484d_435f_454e_4552);
    let initial = hot_field(&mut rng);
    let base_p = random_momentum(initial.lattice(), &mut rng, 0.2).unwrap();
    let mut errors = Vec::new();
    for (dt, steps) in [(0.02, 4), (0.01, 8), (0.005, 16)] {
        let mut links = clone_links(&initial).unwrap();
        let mut p = negate(&negate(&base_p).unwrap()).unwrap();
        let s0 = wilson_action(&links, BETA).unwrap();
        let k0 = kinetic(&p);
        let before = hamiltonian(&links, &p, BETA).unwrap();
        trajectory(&mut links, &mut p, BETA, dt, steps).unwrap();
        let s1 = wilson_action(&links, BETA).unwrap();
        let k1 = kinetic(&p);
        let error = (hamiltonian(&links, &p, BETA).unwrap() - before).abs();
        eprintln!(
            "energy dt={dt:e} steps={steps} ds={:e} dk={:e} abs_dh={error:e}",
            s1 - s0,
            k1 - k0
        );
        errors.push(error);
    }
    let ratios = [errors[0] / errors[1], errors[1] / errors[2]];
    eprintln!("energy ratios={ratios:?}");
    assert!(ratios.iter().all(|ratio| (2.5..6.5).contains(ratio)));

    let mut accepted = 0;
    let trials = 6;
    for trial in 0..trials {
        let mut links = clone_links(&initial).unwrap();
        let mut p = random_momentum(initial.lattice(), &mut rng, 0.2).unwrap();
        let before = hamiltonian(&links, &p, BETA).unwrap();
        trajectory(&mut links, &mut p, BETA, 0.01, 6).unwrap();
        let dh = hamiltonian(&links, &p, BETA).unwrap() - before;
        let draw = rng.gen::<f64>();
        let accept = draw < (-dh).exp().min(1.0);
        accepted += usize::from(accept);
        eprintln!("acceptance trial={trial} dh={dh:e} draw={draw:e} accepted={accept}");
    }
    eprintln!("acceptance {accepted}/{trials}");
    assert!(accepted * 2 > trials);
}
