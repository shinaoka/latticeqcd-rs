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
fn point_source_sets_exactly_one_unit_in_compact_storage_order() {
    let lattice = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    let field = FermionField::point_source(lattice, 4, 2, 3, 1).unwrap();
    for site in 0..lattice.nv() {
        for component in 0..4 {
            for color in 0..3 {
                let expected = usize::from(site == 1 && component == 3 && color == 2);
                assert_eq!(
                    field.component(color, component, site).unwrap().re as usize,
                    expected
                );
                assert_eq!(field.component(color, component, site).unwrap().im, 0.0);
            }
        }
    }
}

#[test]
fn point_source_rejects_invalid_indices_with_typed_errors() {
    let lattice = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    assert!(matches!(
        FermionField::point_source(lattice, 1, 3, 0, 0),
        Err(DiracError::ColorOutOfBounds { color: 3 })
    ));
    assert!(matches!(
        FermionField::point_source(lattice, 1, 0, 1, 0),
        Err(DiracError::ComponentOutOfBounds {
            component: 1,
            components: 1
        })
    ));
    assert!(matches!(
        FermionField::point_source(lattice, 1, 0, 0, 2),
        Err(DiracError::SiteOutOfBounds { site: 2, volume: 2 })
    ));
    assert!(matches!(
        FermionField::point_source(lattice, 2, 0, 0, 0),
        Err(DiracError::InvalidComponents { found: 2 })
    ));
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
