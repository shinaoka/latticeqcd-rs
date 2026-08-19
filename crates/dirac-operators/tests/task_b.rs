use dirac_operators::{
    bicgstab, conjugate_gradient, DiracError, FermionField, FermionOperator, SolverError,
    SolverParams, WilsonDirac,
};
use gaugefields::{cold_su3, LatticeShape4};
use num_complex::Complex64;
use std::error::Error;

fn values(field: &FermionField) -> Result<Vec<Complex64>, DiracError> {
    let mut values = Vec::with_capacity(field.len());
    for site in 0..field.lattice().nv() {
        for component in 0..field.components() {
            for color in 0..3 {
                values.push(field.component(color, component, site)?);
            }
        }
    }
    Ok(values)
}

fn residual_squared<O: FermionOperator>(
    operator: &O,
    solution: &FermionField,
    rhs: &FermionField,
) -> Result<f64, DiracError> {
    let mut applied = FermionField::zeros(solution.lattice(), solution.components())?;
    operator.apply_into(&mut applied, solution)?;
    let mut residual = values(rhs)?;
    for (value, applied_value) in residual.iter_mut().zip(values(&applied)?) {
        *value -= applied_value;
    }
    Ok(residual.iter().map(Complex64::norm_sqr).sum())
}

fn rhs(lattice: LatticeShape4) -> Result<FermionField, DiracError> {
    FermionField::from_vec_col_major(
        lattice,
        4,
        (0..(3 * 4 * lattice.nv()))
            .map(|index| {
                Complex64::new(0.017 * (index + 1) as f64, -0.011 * (2 * index + 3) as f64)
            })
            .collect(),
    )
}

#[test]
fn cg_and_bicgstab_solve_from_zero_and_nonzero_guesses() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = cold_su3(lattice)?;
    let dirac = WilsonDirac::new(&links, 0.1)?;
    let normal = dirac.normal();
    let source = rhs(lattice)?;
    let params = SolverParams::new(1e-20, 2_000)?;

    let mut cg_zero = FermionField::zeros(lattice, 4)?;
    let cg_report = conjugate_gradient(&mut cg_zero, &normal, &source, params)?;
    assert_eq!(cg_report.method.to_string(), "cg");
    assert!(residual_squared(&normal, &cg_zero, &source)? / source.norm_squared()? <= 1e-11);

    let mut cg_nonzero = FermionField::from_vec_col_major(
        lattice,
        4,
        (0..source.len())
            .map(|index| Complex64::new(0.001 * index as f64, -0.0007 * index as f64))
            .collect(),
    )?;
    let before = values(&cg_nonzero)?;
    let cg_nonzero_report = conjugate_gradient(&mut cg_nonzero, &normal, &source, params)?;
    assert!(cg_nonzero_report.iterations > 0);
    assert_ne!(values(&cg_nonzero)?, before);
    assert!(residual_squared(&normal, &cg_nonzero, &source)? / source.norm_squared()? <= 1e-11);

    let mut bicg_zero = FermionField::zeros(lattice, 4)?;
    let bicg_report = bicgstab(&mut bicg_zero, &dirac, &source, params)?;
    assert_eq!(bicg_report.method.to_string(), "bicgstab");
    assert!(residual_squared(&dirac, &bicg_zero, &source)? / source.norm_squared()? <= 1e-11);

    let mut bicg_nonzero = FermionField::from_vec_col_major(
        lattice,
        4,
        (0..source.len())
            .map(|index| Complex64::new(-0.0009 * index as f64, 0.0013 * index as f64))
            .collect(),
    )?;
    let before = values(&bicg_nonzero)?;
    let bicg_nonzero_report = bicgstab(&mut bicg_nonzero, &dirac, &source, params)?;
    assert!(bicg_nonzero_report.iterations > 0);
    assert_ne!(values(&bicg_nonzero)?, before);
    assert!(residual_squared(&dirac, &bicg_nonzero, &source)? / source.norm_squared()? <= 1e-11);
    Ok(())
}

#[test]
fn initial_convergence_and_parameter_validation_are_explicit() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let links = cold_su3(lattice)?;
    let dirac = WilsonDirac::new(&links, 0.1)?;
    let normal = dirac.normal();
    let zero = FermionField::zeros(lattice, 4)?;
    let params = SolverParams::new(1e-20, 4)?;
    let mut cg_solution = zero.try_clone()?;
    let cg_report = conjugate_gradient(&mut cg_solution, &normal, &zero, params)?;
    assert_eq!(cg_report.iterations, 0);
    assert_eq!(cg_report.restart_count, 0);
    assert_eq!(cg_report.convergence_branch.to_string(), "initial_residual");

    let mut bicg_solution = zero.try_clone()?;
    let bicg_report = bicgstab(&mut bicg_solution, &dirac, &zero, params)?;
    assert_eq!(bicg_report.iterations, 0);
    assert_eq!(
        bicg_report.convergence_branch.to_string(),
        "initial_residual"
    );

    assert!(matches!(
        SolverParams::new(0.0, 1),
        Err(DiracError::Solver(SolverError::InvalidTolerance))
    ));
    assert!(matches!(
        SolverParams::new(f64::NAN, 1),
        Err(DiracError::Solver(SolverError::InvalidTolerance))
    ));
    assert!(matches!(
        SolverParams::new(1e-12, 0),
        Err(DiracError::Solver(SolverError::InvalidMaximumIterations))
    ));
    Ok(())
}
