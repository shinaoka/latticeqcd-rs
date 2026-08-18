use super::*;
use crate::path::checked_step_displacement;

#[test]
fn checked_displacement_overflow_does_not_mutate_input() {
    let before = [isize::MAX, 0, 0, 0];
    assert!(matches!(
        checked_step_displacement(before, 1),
        Err(WilsonError::DisplacementOverflow { axis: 0 })
    ));
    assert_eq!(before, [isize::MAX, 0, 0, 0]);

    let before = [isize::MIN + 1, 0, 0, 0];
    assert!(matches!(
        checked_step_displacement(before, -1),
        Err(WilsonError::DisplacementOverflow { axis: 0 })
    ));
    assert_eq!(before, [isize::MIN + 1, 0, 0, 0]);
}

#[test]
fn path_validation_and_adjoint_are_transactional() {
    assert!(matches!(
        WilsonPath::new(Vec::<i8>::new()),
        Err(WilsonError::EmptyPath)
    ));
    assert!(matches!(
        WilsonPath::new(vec![0]),
        Err(WilsonError::InvalidStep { .. })
    ));
    assert!(matches!(
        WilsonPath::new(vec![5]),
        Err(WilsonError::InvalidStep { .. })
    ));
    assert!(matches!(
        LoopTerm::new(1.0, WilsonPath::new(vec![1]).unwrap()),
        Err(WilsonError::OpenPath { .. })
    ));
    let path = WilsonPath::new(vec![1, 2, -1, -2]).unwrap();
    assert_eq!(path.adjoint().adjoint(), path);
}

#[test]
fn helpers_compile_closed_positive_orientations() {
    let plaquette = WilsonPath::plaquette(1, 2).unwrap();
    assert_eq!(plaquette.steps(), &[1, 2, -1, -2]);
    let [nu_long, mu_long] = WilsonPath::rectangle_1x2(1, 2).unwrap();
    assert!(nu_long.is_closed());
    assert!(mu_long.is_closed());
    assert_eq!(nu_long.steps(), &[1, 2, 2, -1, -2, -2]);
    assert_eq!(mu_long.steps(), &[1, 1, 2, -1, -1, -2]);
}

#[test]
fn action_compiles_one_occurrence_per_step() {
    let term = LoopTerm::plaquette(1.0, 1, 2).unwrap();
    let action = LoopAction::new(vec![term]).unwrap();
    let occurrences = &action.compiled[0].occurrences;
    assert_eq!(occurrences.len(), 4);
    assert_eq!(occurrences[0].step_index, 0);
    assert_eq!(occurrences[0].direction, 0);
    assert!(occurrences[0].forward);
    assert_eq!(occurrences[0].link_offset, [0; 4]);
    assert_eq!(occurrences[1].link_offset, [1, 0, 0, 0]);
    assert!(!occurrences[2].forward);
    assert_eq!(occurrences[2].direction, 0);
    assert_eq!(occurrences[2].link_offset, [0, 1, 0, 0]);
    assert!(!occurrences[3].forward);
    assert_eq!(occurrences[3].direction, 1);
    assert_eq!(occurrences[3].link_offset, [0, 0, 0, 0]);
}
