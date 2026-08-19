#![cfg(feature = "fermions")]

use dirac_operators::{
    DiracError, FermionBoundary, FermionField, FermionOperator, SolverError, SolverParams,
    StaggeredDirac, WilsonDirac,
};
use gaugefields::{cold_su3, LatticeShape4, ReproducibleRng};
use measurements::fermions::{
    pion_correlator, stochastic_chiral_condensate, FermionMeasurementError,
};
use rand::RngCore;

struct IdentityOperator {
    lattice: LatticeShape4,
}

impl FermionOperator for IdentityOperator {
    fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        *output = input.try_clone()?;
        Ok(())
    }
}

struct FailingOperator {
    lattice: LatticeShape4,
}

impl FermionOperator for FailingOperator {
    fn lattice(&self) -> LatticeShape4 {
        self.lattice
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        _output: &mut FermionField,
        _input: &FermionField,
    ) -> Result<(), DiracError> {
        Err(SolverError::Exhaustion.into())
    }
}

#[test]
fn synthetic_identity_pion_contraction_has_known_timeslices(
) -> Result<(), Box<dyn std::error::Error>> {
    let operator = IdentityOperator {
        lattice: LatticeShape4::new([2, 1, 1, 2])?,
    };
    let pion = pion_correlator(&operator, SolverParams::new(1.0e-24, 4)?)?;
    assert_eq!(pion.values, vec![3.0, 0.0]);
    assert_eq!(pion.solver_reports.len(), 3);
    Ok(())
}

#[test]
fn fermion_measurements_return_values_and_reports() -> Result<(), Box<dyn std::error::Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let links = cold_su3(lattice)?;
    let boundary = FermionBoundary::new([1, 1, 1, -1])?;
    let solver = SolverParams::new(1.0e-24, 64)?;
    let wilson = WilsonDirac::with_boundary(&links, 0.08, boundary)?;
    let pion = pion_correlator(&wilson, solver)?;
    assert_eq!(pion.values.len(), 1);
    assert_eq!(pion.solver_reports.len(), 12);
    assert!(pion.values[0].is_finite());

    let staggered = StaggeredDirac::with_boundary(&links, 0.17, boundary)?;
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    let chiral = stochastic_chiral_condensate(&staggered, 0.5, 1, solver, &mut rng)?;
    assert_eq!(chiral.source_values.len(), 1);
    assert_eq!(chiral.solver_reports.len(), 1);
    assert!(chiral.value.is_finite());
    Ok(())
}

#[test]
fn chiral_solver_failure_consumes_exactly_the_source_words(
) -> Result<(), Box<dyn std::error::Error>> {
    let lattice = LatticeShape4::new([2, 1, 1, 1])?;
    let operator = FailingOperator { lattice };
    let state = [0x1234, 0x5678, 0x9abc, 0xdef0];
    let mut rng = ReproducibleRng::from_state(state)?;
    let mut expected = ReproducibleRng::from_state(state)?;
    let error =
        stochastic_chiral_condensate(&operator, 0.5, 1, SolverParams::new(1.0e-24, 16)?, &mut rng)
            .expect_err("the test operator must fail");
    assert!(matches!(
        error,
        FermionMeasurementError::Dirac(DiracError::Solver(SolverError::Exhaustion))
    ));
    for _ in 0..(3 * lattice.nv()) {
        expected.next_u64();
    }
    assert_eq!(rng.next_u64(), expected.next_u64());
    Ok(())
}

#[test]
fn invalid_chiral_parameters_do_not_advance_rng() -> Result<(), Box<dyn std::error::Error>> {
    let lattice = LatticeShape4::new([1, 1, 1, 1])?;
    let operator = FailingOperator { lattice };
    let state = [1, 2, 3, 4];
    let mut rng = ReproducibleRng::from_state(state)?;
    let mut expected = ReproducibleRng::from_state(state)?;
    let error =
        stochastic_chiral_condensate(&operator, 0.0, 1, SolverParams::new(1.0e-24, 16)?, &mut rng)
            .expect_err("zero flavor factor must be rejected");
    assert!(matches!(
        error,
        FermionMeasurementError::InvalidFlavorFactor { found: 0.0 }
    ));
    assert_eq!(rng.next_u64(), expected.next_u64());
    Ok(())
}
