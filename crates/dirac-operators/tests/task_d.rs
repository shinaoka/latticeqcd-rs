use dirac_operators::{
    multi_shift_cg, DiracError, FermionBoundary, FermionField, FermionOperator, SolverError,
    SolverParams, StaggeredDirac,
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

fn max_abs_difference(left: &FermionField, right: &FermionField) -> Result<f64, DiracError> {
    Ok(values(left)?
        .into_iter()
        .zip(values(right)?)
        .map(|(left, right)| (left - right).norm())
        .fold(0.0, f64::max))
}

#[test]
fn staggered_operator_and_shifted_solver_contract_is_present() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = cold_su3(lattice).map_err(|error| {
        eprintln!("links error: {error:?}");
        error
    })?;
    let operator = StaggeredDirac::with_boundary(&links, 0.17, FermionBoundary::default())
        .map_err(|error| {
            eprintln!("operator error: {error:?}");
            error
        })?;
    let input = FermionField::from_vec_col_major(
        lattice,
        1,
        (0..3 * lattice.nv())
            .map(|index| num_complex::Complex64::new(index as f64 + 1.0, 0.0))
            .collect(),
    )
    .map_err(|error| {
        eprintln!("input error: {error:?}");
        error
    })?;
    let mut direct = FermionField::zeros(lattice, 1).map_err(|error| {
        eprintln!("direct field error: {error:?}");
        error
    })?;
    operator.apply_into(&mut direct, &input).map_err(|error| {
        eprintln!("direct error: {error:?}");
        error
    })?;
    let mut normal = FermionField::zeros(lattice, 1).map_err(|error| {
        eprintln!("normal field error: {error:?}");
        error
    })?;
    operator
        .normal()
        .apply_into(&mut normal, &input)
        .map_err(|error| {
            eprintln!("normal error: {error:?}");
            error
        })?;
    let mut closed = FermionField::zeros(lattice, 1).map_err(|error| {
        eprintln!("closed field error: {error:?}");
        error
    })?;
    operator
        .normal_closed_form()
        .apply_into(&mut closed, &input)
        .map_err(|error| {
            eprintln!("closed error: {error:?}");
            error
        })?;
    assert_eq!(normal.len(), closed.len());
    let normal_residual = max_abs_difference(&normal, &closed)?;
    eprintln!("normal closed-form residual={normal_residual:.17e}");
    assert!(normal_residual <= 2.0e-12);

    let rhs = FermionField::from_vec_col_major(
        lattice,
        1,
        vec![num_complex::Complex64::new(1.0, 0.25); 3 * lattice.nv()],
    )
    .map_err(|error| {
        eprintln!("rhs error: {error:?}");
        error
    })?;
    let shifts = [0.0, 0.07, 0.31];
    let mut solutions = (0..shifts.len())
        .map(|_| FermionField::zeros(lattice, 1))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            eprintln!("solutions error: {error:?}");
            error
        })?;
    let normal_operator = operator.normal();
    let reports = multi_shift_cg(
        &mut solutions,
        &normal_operator,
        &rhs,
        &shifts,
        SolverParams::new(1.0e-20, 512)?,
    )
    .map_err(|error| {
        eprintln!("solver error: {error:?}");
        error
    })?;
    assert_eq!(reports.len(), shifts.len());
    for report in &reports {
        assert!(report.true_residual_squared < 1.0e-20);
        assert!(
            (report.true_residual_squared / rhs.norm_squared()?).sqrt() <= 1.0e-11,
            "shift {} relative residual={:e}",
            report.shift,
            (report.true_residual_squared / rhs.norm_squared()?).sqrt()
        );
    }

    let mut nonzero_solutions = (0..shifts.len())
        .map(|_| {
            FermionField::from_vec_col_major(
                lattice,
                1,
                (0..3 * lattice.nv())
                    .map(|index| Complex64::new(0.001 * index as f64, -0.0007 * index as f64))
                    .collect(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let nonzero_reports = multi_shift_cg(
        &mut nonzero_solutions,
        &normal_operator,
        &rhs,
        &shifts,
        SolverParams::new(1.0e-20, 512)?,
    )?;
    assert_eq!(nonzero_reports.len(), shifts.len());
    assert!(nonzero_reports
        .iter()
        .all(|report| report.true_residual_squared < 1.0e-20));
    Ok(())
}

#[test]
fn staggered_adjoint_antihermiticity_and_boundary_impulses_are_checked(
) -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = cold_su3(lattice)?;
    let input = FermionField::from_vec_col_major(
        lattice,
        1,
        (0..3 * lattice.nv())
            .map(|index| Complex64::new(0.013 * (index + 1) as f64, -0.007 * index as f64))
            .collect(),
    )?;
    let other = FermionField::from_vec_col_major(
        lattice,
        1,
        (0..3 * lattice.nv())
            .map(|index| Complex64::new(-0.011 * (index + 2) as f64, 0.009 * index as f64))
            .collect(),
    )?;
    let operator = StaggeredDirac::new(&links, 0.17)?;
    let mut d_input = FermionField::zeros(lattice, 1)?;
    let mut ddag_input = FermionField::zeros(lattice, 1)?;
    let mut d_other = FermionField::zeros(lattice, 1)?;
    let mut ddag_other = FermionField::zeros(lattice, 1)?;
    operator.apply_into(&mut d_input, &input)?;
    operator.adjoint().apply_into(&mut ddag_input, &input)?;
    operator.apply_into(&mut d_other, &other)?;
    operator.adjoint().apply_into(&mut ddag_other, &other)?;
    let adjoint_residual =
        (other.inner_product(&d_input)? - ddag_other.inner_product(&input)?).norm();
    let k_input = FermionField::from_vec_col_major(
        lattice,
        1,
        values(&d_input)?
            .into_iter()
            .zip(values(&ddag_input)?)
            .map(|(d, ddag)| 0.5 * (d - ddag))
            .collect(),
    )?;
    let k_other = FermionField::from_vec_col_major(
        lattice,
        1,
        values(&d_other)?
            .into_iter()
            .zip(values(&ddag_other)?)
            .map(|(d, ddag)| 0.5 * (d - ddag))
            .collect(),
    )?;
    let antihermiticity_residual =
        (k_input.inner_product(&other)? + input.inner_product(&k_other)?).norm();
    eprintln!(
        "adjoint residual={adjoint_residual:.17e}, K antihermiticity residual={antihermiticity_residual:.17e}"
    );
    assert!(adjoint_residual <= 2.0e-12);
    assert!(antihermiticity_residual <= 2.0e-12);

    let impulse_lattice = LatticeShape4::new([2, 1, 1, 1])?;
    let impulse_links = cold_su3(impulse_lattice)?;
    let impulse = FermionField::from_vec_col_major(
        impulse_lattice,
        1,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
        ],
    )?;
    let periodic =
        StaggeredDirac::with_boundary(&impulse_links, 0.17, FermionBoundary::new([1, 1, 1, 1])?)?;
    let antiperiodic_x =
        StaggeredDirac::with_boundary(&impulse_links, 0.17, FermionBoundary::new([-1, 1, 1, 1])?)?;
    let mut periodic_output = FermionField::zeros(impulse_lattice, 1)?;
    let mut antiperiodic_output = FermionField::zeros(impulse_lattice, 1)?;
    periodic.apply_into(&mut periodic_output, &impulse)?;
    antiperiodic_x.apply_into(&mut antiperiodic_output, &impulse)?;
    assert_eq!(periodic_output.component(0, 0, 1)?, Complex64::default());
    assert_eq!(
        antiperiodic_output.component(0, 0, 1)?,
        Complex64::new(-1.0, 0.0)
    );
    Ok(())
}

#[test]
fn multi_shift_validation_and_failure_are_transactional() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = cold_su3(lattice)?;
    let dirac = StaggeredDirac::new(&links, 0.17)?;
    let operator = dirac.normal();
    let rhs = FermionField::from_vec_col_major(
        lattice,
        1,
        vec![Complex64::new(1.0, 0.25); 3 * lattice.nv()],
    )?;
    let sentinel = FermionField::from_vec_col_major(
        lattice,
        1,
        vec![Complex64::new(0.3, -0.2); 3 * lattice.nv()],
    )?;
    let before = values(&sentinel)?;

    let mut count_mismatch = vec![sentinel.try_clone()?];
    let error = multi_shift_cg(
        &mut count_mismatch,
        &operator,
        &rhs,
        &[0.0, 0.1],
        SolverParams::new(1.0e-20, 64)?,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracError::Solver(SolverError::ShiftCountMismatch { .. })
    ));
    assert_eq!(values(&count_mismatch[0])?, before);

    let mut invalid_shift = vec![sentinel.try_clone()?];
    let error = multi_shift_cg(
        &mut invalid_shift,
        &operator,
        &rhs,
        &[f64::NAN],
        SolverParams::new(1.0e-20, 64)?,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracError::Solver(SolverError::InvalidShift { index: 0 })
    ));
    assert_eq!(values(&invalid_shift[0])?, before);

    let mut negative_shift = vec![sentinel.try_clone()?];
    let error = multi_shift_cg(
        &mut negative_shift,
        &operator,
        &rhs,
        &[-0.1],
        SolverParams::new(1.0e-20, 64)?,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracError::Solver(SolverError::InvalidShift { index: 0 })
    ));
    assert_eq!(values(&negative_shift[0])?, before);

    let mut wrong_rhs_output = vec![sentinel.try_clone()?];
    let wrong_rhs = FermionField::zeros(LatticeShape4::new([1, 1, 1, 1])?, 1)?;
    let error = multi_shift_cg(
        &mut wrong_rhs_output,
        &operator,
        &wrong_rhs,
        &[0.0],
        SolverParams::new(1.0e-20, 64)?,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracError::LatticeMismatch { operand: "rhs", .. }
    ));
    assert_eq!(values(&wrong_rhs_output[0])?, before);

    let mut wrong_components = vec![FermionField::zeros(lattice, 4)?];
    let error = multi_shift_cg(
        &mut wrong_components,
        &operator,
        &rhs,
        &[0.0],
        SolverParams::new(1.0e-20, 64)?,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracError::ComponentsMismatch {
            operand: "output",
            ..
        }
    ));

    let mut exhausted = vec![sentinel.try_clone()?];
    let error = multi_shift_cg(
        &mut exhausted,
        &operator,
        &rhs,
        &[0.0],
        SolverParams::new(1.0e-30, 1)?,
    )
    .unwrap_err();
    assert!(matches!(error, DiracError::Solver(SolverError::Exhaustion)));
    assert_eq!(values(&exhausted[0])?, before);
    Ok(())
}
