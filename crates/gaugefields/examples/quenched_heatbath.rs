use gaugefields::{
    cold_su3, heatbath_sweep, normalized_plaquette, HeatbathParams, LatticeShape4, ReproducibleRng,
};

fn main() -> Result<(), gaugefields::GaugeError> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let mut links = cold_su3(lattice)?;
    let mut rng = ReproducibleRng::from_state([0x1357_9bdf_2468_ace1, 1, 2, 3])?;
    let params = HeatbathParams::new(5.7, 100_000)?;

    for sweep in 1..=3 {
        let stats = heatbath_sweep(&mut links, params, &mut rng)?;
        println!(
            "sweep={sweep} updated_links={} su2_attempts={} normalized_plaquette={}",
            stats.updated_links,
            stats.su2_attempts,
            normalized_plaquette(&links)?,
        );
    }
    Ok(())
}
