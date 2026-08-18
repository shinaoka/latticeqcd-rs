use super::*;
use crate::error::SolverError;
use std::cell::{Cell, RefCell};

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);

fn lattice() -> gaugefields::LatticeShape4 {
    gaugefields::LatticeShape4::new([1, 1, 1, 1]).unwrap()
}

fn field(values: [Complex64; 3]) -> FermionField {
    FermionField::from_vec_col_major(lattice(), 1, values.to_vec()).unwrap()
}

fn output_values(field: &FermionField) -> [Complex64; 3] {
    field.host_data().unwrap().try_into().unwrap()
}

fn copy_scaled(
    output: &mut FermionField,
    input: &FermionField,
    scales: [f64; 3],
) -> Result<(), DiracError> {
    output.ensure_compatible(input, "test operator")?;
    let input_data = input.host_data()?;
    let destination = output.host_data_mut()?;
    for ((destination_value, input_value), scale) in
        destination.iter_mut().zip(input_data).zip(scales)
    {
        *destination_value = *input_value * scale;
    }
    Ok(())
}

struct DiagonalOperator {
    scales: [f64; 3],
}

impl FermionOperator for DiagonalOperator {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        copy_scaled(output, input, self.scales)
    }
}

impl HermitianPositiveOperator for DiagonalOperator {}

struct ZeroOperator;

impl FermionOperator for ZeroOperator {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        output.ensure_compatible(input, "zero operator")?;
        output.host_data_mut()?.fill(C0);
        Ok(())
    }
}

impl HermitianPositiveOperator for ZeroOperator {}

struct NonFiniteOperator;

impl FermionOperator for NonFiniteOperator {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        output.ensure_compatible(input, "nonfinite operator")?;
        output.host_data_mut()?.fill(Complex64::new(f64::NAN, 0.0));
        Ok(())
    }
}

struct RecordingDiagonal {
    pointers: RefCell<Vec<usize>>,
}

impl FermionOperator for RecordingDiagonal {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        output.ensure_compatible(input, "recording operator")?;
        self.pointers
            .borrow_mut()
            .push(output.host_data_mut()?.as_mut_ptr() as usize);
        copy_scaled(output, input, [1.0, 2.0, 3.0])
    }
}

impl HermitianPositiveOperator for RecordingDiagonal {}

struct RecordingOperatorScratch {
    pointers: RefCell<Vec<usize>>,
}

impl FermionOperator for RecordingOperatorScratch {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        copy_scaled(output, input, [1.0, 2.0, 3.0])
    }

    fn apply_into_with_scratch(
        &self,
        output: &mut FermionField,
        input: &FermionField,
        scratch: &mut [FermionField],
    ) -> Result<(), DiracError> {
        assert_eq!(scratch.len(), 2);
        self.pointers
            .borrow_mut()
            .push(scratch[0].host_data_mut()?.as_mut_ptr() as usize);
        copy_scaled(output, input, [1.0, 2.0, 3.0])
    }
}

impl HermitianPositiveOperator for RecordingOperatorScratch {}

struct RestartOperator;

impl FermionOperator for RestartOperator {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        output.ensure_compatible(input, "restart operator")?;
        let input = input.host_data()?;
        let output = output.host_data_mut()?;
        output[0] = input[0] + 1.0e-16 * input[1];
        output[1] = input[0] + 2.0 * input[1];
        output[2] = input[0] + 3.0 * input[2];
        Ok(())
    }
}

struct DriftingIdentity {
    calls: Cell<usize>,
}

impl FermionOperator for DriftingIdentity {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        copy_scaled(output, input, [if call >= 2 { 2.0 } else { 1.0 }; 3])
    }
}

struct RestartBreakdownOperator;

impl FermionOperator for RestartBreakdownOperator {
    fn lattice(&self) -> gaugefields::LatticeShape4 {
        lattice()
    }

    fn components(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        output: &mut FermionField,
        input: &FermionField,
    ) -> Result<(), DiracError> {
        output.ensure_compatible(input, "restart breakdown operator")?;
        let input = input.host_data()?;
        let output = output.host_data_mut()?;
        output[0] = input[0];
        output[1] = input[0] + input[2];
        output[2] = input[0] + 2.0 * input[2];
        Ok(())
    }
}

#[test]
fn cg_breakdown_and_exhaustion_are_transactional() {
    let rhs = field([C1, C1, C1]);
    let sentinel = field([
        Complex64::new(3.0, -2.0),
        Complex64::new(4.0, -1.0),
        Complex64::new(5.0, -3.0),
    ]);
    let before = output_values(&sentinel);

    let mut breakdown = sentinel.try_clone().unwrap();
    let error = conjugate_gradient(
        &mut breakdown,
        &ZeroOperator,
        &rhs,
        SolverParams::new(1e-20, 4).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(error, DiracError::Solver(SolverError::Breakdown)));
    assert_eq!(output_values(&breakdown), before);

    let mut exhausted = sentinel.try_clone().unwrap();
    let error = conjugate_gradient(
        &mut exhausted,
        &DiagonalOperator {
            scales: [1.0, 2.0, 3.0],
        },
        &rhs,
        SolverParams::new(1e-30, 1).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(error, DiracError::Solver(SolverError::Exhaustion)));
    assert_eq!(output_values(&exhausted), before);
}

#[test]
fn bicgstab_nonfinite_breakdown_exhaustion_and_restart_are_typed() {
    let rhs = field([C1, C1, C1]);
    let sentinel = field([
        Complex64::new(-3.0, 2.0),
        Complex64::new(-4.0, 1.0),
        Complex64::new(-5.0, 3.0),
    ]);
    let before = output_values(&sentinel);

    let mut nonfinite = sentinel.try_clone().unwrap();
    let error = bicgstab(
        &mut nonfinite,
        &NonFiniteOperator,
        &rhs,
        SolverParams::new(1e-20, 4).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracError::Solver(SolverError::NonFiniteIntermediate)
    ));
    assert_eq!(output_values(&nonfinite), before);

    let mut breakdown = sentinel.try_clone().unwrap();
    let error = bicgstab(
        &mut breakdown,
        &ZeroOperator,
        &rhs,
        SolverParams::new(1e-20, 4).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(error, DiracError::Solver(SolverError::Breakdown)));
    assert_eq!(output_values(&breakdown), before);

    let mut exhausted = sentinel.try_clone().unwrap();
    let error = bicgstab(
        &mut exhausted,
        &DiagonalOperator {
            scales: [1.0, 2.0, 3.0],
        },
        &rhs,
        SolverParams::new(1e-30, 1).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(error, DiracError::Solver(SolverError::Exhaustion)));
    assert_eq!(output_values(&exhausted), before);

    let mut singular_restart = field([C0, C0, C0]);
    let singular_before = output_values(&singular_restart);
    let error = bicgstab(
        &mut singular_restart,
        &RestartBreakdownOperator,
        &field([C1, C0, C0]),
        SolverParams::new(1e-30, 4).unwrap(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            DiracError::Solver(SolverError::SingularShadowRestart)
        ),
        "error={error:?}"
    );
    assert_eq!(output_values(&singular_restart), singular_before);
}

#[test]
fn bicgstab_shadow_restart_is_used_for_a_fixed_linear_operator() {
    assert!(shadow_near_zero(Complex64::new(0.5 * f64::EPSILON, 0.0), 1.0).unwrap());
    assert!(!shadow_near_zero(Complex64::new(2.0 * f64::EPSILON, 0.0), 1.0).unwrap());
    let rhs = field([C1, C0, C0]);
    let mut solution = field([C0, C0, C0]);
    let report = bicgstab(
        &mut solution,
        &RestartOperator,
        &rhs,
        SolverParams::new(1e-30, 16).unwrap(),
    )
    .unwrap();
    assert_eq!(report.restart_count, 1);
    assert!(report.true_residual_squared < 1e-30);
}

#[test]
fn bicgstab_intermediate_residual_branch_updates_solution() {
    let rhs = field([C1, Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)]);
    let mut solution = field([C0, C0, C0]);
    let report = bicgstab(
        &mut solution,
        &DiagonalOperator {
            scales: [1.0, 1.0, 1.0],
        },
        &rhs,
        SolverParams::new(1e-20, 4).unwrap(),
    )
    .unwrap();
    assert_eq!(
        report.convergence_branch,
        ConvergenceBranch::IntermediateResidual
    );
    assert_eq!(report.iterations, 1);
    assert_eq!(report.recursive_residual_squared, 0.0);
    assert_eq!(report.true_residual_squared, 0.0);
    assert_eq!(output_values(&solution), output_values(&rhs));
}

#[test]
fn true_residual_mismatch_leaves_output_unchanged() {
    let operator = DriftingIdentity {
        calls: Cell::new(0),
    };
    let rhs = field([C1, C1, C1]);
    let mut solution = field([C0, C0, C0]);
    let before = output_values(&solution);
    let error = bicgstab(
        &mut solution,
        &operator,
        &rhs,
        SolverParams::new(1e-20, 4).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracError::Solver(SolverError::TrueResidualMismatch)
    ));
    assert_eq!(output_values(&solution), before);
}

#[test]
fn cg_stagnation_and_scratch_destinations_are_checked() {
    let mut solution = field([
        Complex64::new(7.0, 2.0),
        Complex64::new(8.0, 1.0),
        Complex64::new(9.0, 3.0),
    ]);
    let before = output_values(&solution);
    let stagnation_rhs = field([
        Complex64::new(8.0, 2.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(18.0, 6.0),
    ]);
    let error = conjugate_gradient(
        &mut solution,
        &DiagonalOperator {
            scales: [1.0, 0.0, 2.0],
        },
        &stagnation_rhs,
        SolverParams::new(1e-30, 8).unwrap(),
    )
    .unwrap_err();
    assert!(
        matches!(error, DiracError::Solver(SolverError::Stagnation)),
        "error={error:?}"
    );
    assert_eq!(output_values(&solution), before);

    let operator = RecordingDiagonal {
        pointers: RefCell::new(Vec::new()),
    };
    let mut solution = field([C0, C0, C0]);
    let report = conjugate_gradient(
        &mut solution,
        &operator,
        &field([C1, C1, C1]),
        SolverParams::new(1e-30, 8).unwrap(),
    )
    .unwrap();
    assert_eq!(
        report.convergence_branch,
        ConvergenceBranch::UpdatedResidual
    );
    let pointers = operator.pointers.borrow();
    assert!(pointers.len() >= 3);
    assert_eq!(pointers[0], *pointers.last().unwrap());
    assert!(pointers[1..pointers.len() - 1]
        .iter()
        .all(|pointer| *pointer == pointers[1]));
}

#[test]
fn solver_reuses_operator_scratch_across_iterations() {
    let operator = RecordingOperatorScratch {
        pointers: RefCell::new(Vec::new()),
    };
    let rhs = field([C1, C1, C1]);
    let mut solution = field([C0, C0, C0]);
    let report = conjugate_gradient(
        &mut solution,
        &operator,
        &rhs,
        SolverParams::new(1e-30, 8).unwrap(),
    )
    .unwrap();
    assert!(report.iterations > 1);
    let pointers = operator.pointers.borrow();
    assert!(pointers.len() >= 2);
    assert!(pointers.windows(2).all(|window| window[0] == window[1]));
}

#[test]
fn wrong_lattice_and_components_leave_output_bitwise_unchanged() {
    let links = gaugefields::cold_su3(lattice()).unwrap();
    let operator = crate::wilson::WilsonDirac::new(&links, 0.1).unwrap();
    let rhs = FermionField::zeros(lattice(), 4).unwrap();
    let mut wrong_lattice = FermionField::from_vec_col_major(
        gaugefields::LatticeShape4::new([2, 1, 1, 1]).unwrap(),
        4,
        vec![Complex64::new(2.0, -1.0); 24],
    )
    .unwrap();
    let before = wrong_lattice.host_data().unwrap().to_vec();
    let error = conjugate_gradient(
        &mut wrong_lattice,
        &operator.normal(),
        &rhs,
        SolverParams::new(1e-20, 4).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(error, DiracError::LatticeMismatch { .. }));
    assert_eq!(wrong_lattice.host_data().unwrap(), before.as_slice());

    let mut wrong_components =
        FermionField::from_vec_col_major(lattice(), 1, vec![Complex64::new(2.0, -1.0); 3]).unwrap();
    let error = bicgstab(
        &mut wrong_components,
        &operator,
        &FermionField::zeros(lattice(), 4).unwrap(),
        SolverParams::new(1e-20, 4).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(error, DiracError::ComponentsMismatch { .. }));
    assert_eq!(
        output_values(&wrong_components),
        [Complex64::new(2.0, -1.0); 3]
    );
}
