use gaugefields::{
    cold_su3, hmc_update, normalized_plaquette, CpuEvolutionContext, HmcParams, LatticeShape4,
    ReproducibleRng,
};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), gaugefields::GaugeError> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let mut links = cold_su3(lattice)?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let params = HmcParams::new(5.7, 0.01, 4)?;
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    let mut accepted = 0;
    for _ in 0..3 {
        accepted += usize::from(hmc_update(&mut context, &mut links, params, &mut rng)?.accepted);
    }
    println!(
        "accepted={accepted}/3 normalized_plaquette={}",
        normalized_plaquette(&links)?
    );
    Ok(())
}
