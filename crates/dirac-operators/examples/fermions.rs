use dirac_operators::{
    staggered_hmc_update, FermionBoundary, FermionField, FermionOperator, SolverParams,
    StaggeredDirac, StaggeredHmcParams, WilsonDirac,
};
use gaugefields::{cold_su3, CpuEvolutionContext, LatticeShape4, ReproducibleRng};
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let mut links = cold_su3(lattice)?;
    let boundary = FermionBoundary::new([1, 1, 1, -1])?;

    let wilson_input = FermionField::from_vec_col_major(
        lattice,
        4,
        (1..=12)
            .map(|value| Complex64::new(value as f64 / 20.0, 0.0))
            .collect(),
    )?;
    let mut wilson_output = FermionField::zeros(lattice, 4)?;
    WilsonDirac::with_boundary(&links, 0.13, boundary)?
        .apply_into(&mut wilson_output, &wilson_input)?;

    let staggered_input =
        FermionField::from_vec_col_major(lattice, 1, vec![Complex64::new(1.0, 0.0); 3])?;
    let mut staggered_output = FermionField::zeros(lattice, 1)?;
    StaggeredDirac::with_boundary(&links, 0.17, boundary)?
        .apply_into(&mut staggered_output, &staggered_input)?;

    let params = StaggeredHmcParams::new(
        5.7,
        0.17,
        1.0e-4,
        1,
        boundary,
        0.0004,
        64.0,
        SolverParams::new(1.0e-20, 512)?,
    )?;
    let mut context = CpuEvolutionContext::new(CpuBackend::new());
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    let outcome = staggered_hmc_update(&mut context, &mut links, params, &mut rng)?;

    println!(
        "wilson_norm={} staggered_norm={} rhmc_accepted={} delta_h={}",
        wilson_output.norm_squared()?,
        staggered_output.norm_squared()?,
        outcome.accepted,
        outcome.delta_h,
    );
    Ok(())
}
