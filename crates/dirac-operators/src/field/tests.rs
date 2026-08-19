use super::*;

#[test]
fn physical_offset_keeps_color_fastest() {
    let lattice = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    let field = FermionField::from_vec_col_major(
        lattice,
        4,
        (0..24)
            .map(|value| Complex64::new(value as f64, 0.0))
            .collect(),
    )
    .unwrap();
    assert_eq!(field.component(0, 0, 0).unwrap().re, 0.0);
    assert_eq!(field.component(1, 0, 0).unwrap().re, 1.0);
    assert_eq!(field.component(0, 1, 0).unwrap().re, 3.0);
    assert_eq!(field.component(0, 0, 1).unwrap().re, 12.0);
}

#[test]
fn gamma5_is_involutive_and_preserves_layout() {
    let lattice = LatticeShape4::new([1, 1, 1, 1]).unwrap();
    let field = FermionField::from_vec_col_major(
        lattice,
        4,
        (0..12)
            .map(|value| Complex64::new(value as f64 + 1.0, 0.0))
            .collect(),
    )
    .unwrap();
    let transformed = field.gamma5().unwrap();
    assert_eq!(transformed.component(0, 0, 0).unwrap().re, -1.0);
    assert_eq!(transformed.component(0, 1, 0).unwrap().re, -4.0);
    assert_eq!(transformed.component(0, 2, 0).unwrap().re, 7.0);
    assert_eq!(
        transformed.gamma5().unwrap().component(2, 3, 0).unwrap().re,
        12.0
    );
}
