use gaugefields::{cold_su3, normalized_plaquette, CpuEvolutionContext, LatticeShape4};
use measurements::{clover_topological_charge, gradient_flow, polyakov_loop, GradientFlowParams};
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use wilsonloop::{LoopAction, LoopTerm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let links = cold_su3(LatticeShape4::new([2, 2, 2, 2])?)?;
    assert_eq!(normalized_plaquette(&links)?, 1.0);
    assert_eq!(polyakov_loop(&links)?, Complex64::new(3.0, 0.0));
    assert_eq!(clover_topological_charge(&links)?, 0.0);

    let mut terms = Vec::with_capacity(6);
    for mu in 1..=3 {
        for nu in (mu + 1)..=4 {
            terms.push(LoopTerm::plaquette(1.0, mu, nu)?);
        }
    }
    let action = LoopAction::new(terms)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let flowed = gradient_flow(
        &mut context,
        &links,
        &action,
        GradientFlowParams::new(0.01, 1)?,
    )?;
    assert_eq!(normalized_plaquette(&flowed)?, 1.0);

    println!("plaquette=1 Polyakov=3+0im clover_Q=0 after one cold Wilson-flow step");
    Ok(())
}
