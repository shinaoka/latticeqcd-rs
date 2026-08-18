use dirac_operators::{
    DiracError, FermionBoundary, FermionField, FermionOperator, NormalOperator, WilsonDirac,
};
use gaugefields::{
    cold_su3, coords_from_site_index, neighbor_site, GaugeLinkTensor, GaugeLinks, LatticeShape4,
};
use num_complex::Complex64;
use std::error::Error;
use tenferro_tensor::{BackendStorageHandle, Placement, StorageBuffer, Tensor, TypedTensor};

type C = Complex64;
type Spinor = [[C; 4]; 3];

const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);
const I: C = C::new(0.0, 1.0);
const NEG_I: C = C::new(0.0, -1.0);
const NEG_ONE: C = C::new(-1.0, 0.0);

fn gamma(direction: usize) -> [[C; 4]; 4] {
    match direction {
        0 => [
            [ZERO, ZERO, ZERO, NEG_I],
            [ZERO, ZERO, NEG_I, ZERO],
            [ZERO, I, ZERO, ZERO],
            [I, ZERO, ZERO, ZERO],
        ],
        1 => [
            [ZERO, ZERO, ZERO, NEG_ONE],
            [ZERO, ZERO, ONE, ZERO],
            [ZERO, ONE, ZERO, ZERO],
            [NEG_ONE, ZERO, ZERO, ZERO],
        ],
        2 => [
            [ZERO, ZERO, NEG_I, ZERO],
            [ZERO, ZERO, ZERO, I],
            [I, ZERO, ZERO, ZERO],
            [ZERO, NEG_I, ZERO, ZERO],
        ],
        3 => [
            [ZERO, ZERO, NEG_ONE, ZERO],
            [ZERO, ZERO, ZERO, NEG_ONE],
            [NEG_ONE, ZERO, ZERO, ZERO],
            [ZERO, NEG_ONE, ZERO, ZERO],
        ],
        _ => panic!("test gamma direction"),
    }
}

fn project(direction: usize, gamma_sign: i8, input: [C; 4]) -> [C; 4] {
    let matrix = gamma(direction);
    let mut result = [ZERO; 4];
    for row in 0..4 {
        for column in 0..4 {
            let mut coefficient = if row == column { ONE } else { ZERO };
            if gamma_sign > 0 {
                coefficient += matrix[row][column];
            } else {
                coefficient -= matrix[row][column];
            }
            result[row] += coefficient * input[column];
        }
    }
    result
}

fn spinor_at(values: &[C], site: usize) -> Spinor {
    let mut result = [[ZERO; 4]; 3];
    let offset = 12 * site;
    for color in 0..3 {
        for component in 0..4 {
            result[color][component] = values[offset + color + 3 * component];
        }
    }
    result
}

fn store_spinor(values: &mut [C], site: usize, spinor: Spinor) {
    let offset = 12 * site;
    for color in 0..3 {
        for component in 0..4 {
            values[offset + color + 3 * component] = spinor[color][component];
        }
    }
}

fn free_expected(
    lattice: LatticeShape4,
    kappa: f64,
    boundary: FermionBoundary,
    input: &[C],
    adjoint: bool,
) -> Result<Vec<C>, Box<dyn Error>> {
    let mut output = vec![ZERO; input.len()];
    for site in 0..lattice.nv() {
        let coordinates = coords_from_site_index(site, lattice)?;
        let mut result = spinor_at(input, site);
        for (direction, (&coordinate, &extent)) in
            coordinates.iter().zip(lattice.extents().iter()).enumerate()
        {
            let plus_wrap = coordinate == extent - 1;
            let minus_wrap = coordinate == 0;
            let plus_site = neighbor_site(site, direction, 1, lattice)?;
            let minus_site = neighbor_site(site, direction, -1, lattice)?;
            let plus_sign = if plus_wrap {
                boundary.sign(direction)?
            } else {
                1
            };
            let minus_sign = if minus_wrap {
                boundary.sign(direction)?
            } else {
                1
            };
            let plus = spinor_at(input, plus_site);
            let minus = spinor_at(input, minus_site);
            let plus_projector = if adjoint { 1 } else { -1 };
            let minus_projector = if adjoint { -1 } else { 1 };
            for color in 0..3 {
                let plus_value = project(direction, plus_projector, plus[color]);
                let minus_value = project(direction, minus_projector, minus[color]);
                for component in 0..4 {
                    result[color][component] -=
                        kappa * f64::from(plus_sign) * plus_value[component];
                    result[color][component] -=
                        kappa * f64::from(minus_sign) * minus_value[component];
                }
            }
        }
        store_spinor(&mut output, site, result);
    }
    Ok(output)
}

fn field_values(field: &FermionField) -> Result<Vec<C>, DiracError> {
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

fn max_abs_difference(left: &[C], right: &[C]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

fn assert_close(left: &[C], right: &[C], tolerance: f64) {
    let residual = max_abs_difference(left, right);
    eprintln!("max_abs_residual={residual:.17e}");
    assert!(
        residual <= tolerance,
        "maximum residual {residual:.3e} exceeds {tolerance:.3e}"
    );
}

fn input_field(lattice: LatticeShape4) -> Result<FermionField, DiracError> {
    let mut values = vec![ZERO; 12 * lattice.nv()];
    for site in 0..lattice.nv() {
        for component in 0..4 {
            for color in 0..3 {
                let real = 0.013 * (1 + color + 3 * component + 12 * site) as f64;
                let imag = -0.009 * (1 + 2 * color + component + 5 * site) as f64;
                values[12 * site + color + 3 * component] = C::new(real, imag);
            }
        }
    }
    FermionField::from_vec_col_major(lattice, 4, values)
}

fn diagonal_links(lattice: LatticeShape4) -> Result<GaugeLinks, Box<dyn Error>> {
    let mut links = Vec::with_capacity(4);
    for direction in 0..4 {
        let mut values = vec![ZERO; 9 * lattice.nv()];
        for site in 0..lattice.nv() {
            let [x, y, z, t] = coords_from_site_index(site, lattice)?;
            let a = 0.017 * (direction + 1) as f64 + 0.031 * x as f64 - 0.013 * y as f64
                + 0.007 * z as f64
                + 0.011 * t as f64;
            let b = -0.023 * (direction + 1) as f64 + 0.019 * x as f64 + 0.005 * y as f64
                - 0.009 * z as f64
                + 0.003 * t as f64;
            let diagonal = [
                C::from_polar(1.0, a),
                C::from_polar(1.0, b),
                C::from_polar(1.0, -a - b),
            ];
            for color in 0..3 {
                values[color + 3 * color + 9 * site] = diagonal[color];
            }
        }
        let tensor = TypedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values)?;
        links.push(GaugeLinkTensor::from_typed(tensor, lattice)?);
    }
    Ok(GaugeLinks::new(
        links.try_into().map_err(|_| "four links")?,
    )?)
}

#[test]
fn layout_and_logical_component_mapping_are_x_fast_and_checked() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 3, 1, 2])?;
    let mut values = vec![ZERO; 3 * 4 * lattice.nv()];
    for (offset, value) in values.iter_mut().enumerate() {
        *value = C::new(offset as f64, 0.0);
    }
    let field = FermionField::from_vec_col_major(lattice, 4, values)?;
    let site = gaugefields::site_index([1, 2, 0, 1], lattice)?;
    assert_eq!(site, 11);
    assert_eq!(
        field.component(2, 3, site)?,
        C::new((2 + 3 * 3 + 3 * 4 * site) as f64, 0.0)
    );
    assert!(matches!(
        field.component(3, 0, site),
        Err(DiracError::ColorOutOfBounds { .. })
    ));
    assert!(matches!(
        field.component(0, 4, site),
        Err(DiracError::ComponentOutOfBounds { .. })
    ));
    assert!(matches!(
        field.component(0, 0, lattice.nv()),
        Err(DiracError::SiteOutOfBounds { .. })
    ));
    Ok(())
}

#[test]
fn invalid_fields_reject_rank_shape_components_dtype_nonfinite_and_backend(
) -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let wrong_rank = TypedTensor::from_vec_col_major(vec![3, 4, 2, 2, 2], vec![ZERO; 96])?;
    assert!(matches!(
        FermionField::from_typed(wrong_rank, lattice, 4),
        Err(DiracError::Rank { .. })
    ));
    let wrong_shape = TypedTensor::from_vec_col_major(vec![3, 4, 2, 2, 2, 3], vec![ZERO; 288])?;
    assert!(matches!(
        FermionField::from_typed(wrong_shape, lattice, 4),
        Err(DiracError::Shape { .. })
    ));
    assert!(matches!(
        FermionField::from_vec_col_major(lattice, 2, vec![ZERO; 6]),
        Err(DiracError::InvalidComponents { found: 2 })
    ));
    assert!(matches!(
        FermionField::from_vec_col_major(
            lattice,
            4,
            vec![C::new(f64::NAN, 0.0); 3 * 4 * lattice.nv()]
        ),
        Err(DiracError::NonFinite { .. })
    ));
    let wrong_dtype =
        Tensor::from_vec_col_major(vec![3, 4, 2, 2, 2, 2], vec![0.0_f64; 3 * 4 * lattice.nv()])?;
    assert!(matches!(
        FermionField::try_from_tensor(wrong_dtype, lattice, 4),
        Err(DiracError::DType { .. })
    ));
    let backend = TypedTensor::from_buffer_col_major(
        vec![3, 4, 2, 2, 2, 2],
        StorageBuffer::Backend(Box::new(BackendStorageHandle::<C>::new_with_len(
            7,
            3 * 4 * lattice.nv(),
        ))),
        Placement {
            memory_kind: tenferro_tensor::MemoryKind::Device,
            device: Some(tenferro_tensor::DeviceId {
                kind: tenferro_tensor::DeviceKind::Gpu(tenferro_tensor::GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )?;
    assert!(matches!(
        FermionField::from_typed(backend, lattice, 4),
        Err(DiracError::Placement { .. })
    ));
    Ok(())
}

#[test]
fn all_axis_boundary_signs_are_applied_once_per_wrap() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = cold_su3(lattice)?;
    let mut values = vec![ZERO; 12 * lattice.nv()];
    values[0] = ONE;
    let input = FermionField::from_vec_col_major(lattice, 4, values.clone())?;
    let periodic = FermionBoundary::new([1, 1, 1, 1])?;
    let periodic_operator = WilsonDirac::with_boundary(&links, 0.125, periodic)?;
    let mut periodic_output = FermionField::zeros(lattice, 4)?;
    periodic_operator.apply_into(&mut periodic_output, &input)?;
    let periodic_values = field_values(&periodic_output)?;
    for direction in 0..4 {
        let mut signs = [1; 4];
        signs[direction] = -1;
        let boundary = FermionBoundary::new(signs)?;
        let operator = WilsonDirac::with_boundary(&links, 0.125, boundary)?;
        let mut output = FermionField::zeros(lattice, 4)?;
        operator.apply_into(&mut output, &input)?;
        let expected = free_expected(lattice, 0.125, boundary, &values, false)?;
        assert_close(&field_values(&output)?, &expected, 2e-14);
        assert_close(
            &periodic_values,
            &free_expected(lattice, 0.125, periodic, &values, false)?,
            2e-14,
        );
        let differing = max_abs_difference(&field_values(&output)?, &periodic_values);
        assert!(
            differing > 0.0,
            "axis {direction} did not change the wrapped hop"
        );
    }
    Ok(())
}

#[test]
fn cold_impulse_matches_independent_projector_stencil() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = cold_su3(lattice)?;
    let boundary = FermionBoundary::new([1, 1, 1, 1])?;
    let input = input_field(lattice)?;
    let input_values = field_values(&input)?;
    let operator = WilsonDirac::with_boundary(&links, 0.11, boundary)?;
    let mut output = FermionField::zeros(lattice, 4)?;
    operator.apply_into(&mut output, &input)?;
    let expected = free_expected(lattice, 0.11, boundary, &input_values, false)?;
    assert_close(&field_values(&output)?, &expected, 2e-14);
    Ok(())
}

#[test]
fn nontrivial_links_cover_d_dagger_normal_adjoint_and_gamma5_identities(
) -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = diagonal_links(lattice)?;
    let input = input_field(lattice)?;
    let other = FermionField::from_vec_col_major(
        lattice,
        4,
        (0..input.len())
            .map(|index| C::new(0.021 * index as f64, -0.014 * (index + 3) as f64))
            .collect(),
    )?;
    let input_before = field_values(&input)?;
    let operator = WilsonDirac::new(&links, 0.13)?;
    let mut d_input = FermionField::zeros(lattice, 4)?;
    let mut ddag_input = FermionField::zeros(lattice, 4)?;
    operator.apply_into(&mut d_input, &input)?;
    operator.adjoint().apply_into(&mut ddag_input, &input)?;
    let mut normal_input = FermionField::zeros(lattice, 4)?;
    NormalOperator::new(&operator).apply_into(&mut normal_input, &input)?;
    let mut composed = FermionField::zeros(lattice, 4)?;
    operator.adjoint().apply_into(&mut composed, &d_input)?;
    assert_close(
        &field_values(&normal_input)?,
        &field_values(&composed)?,
        2e-14,
    );
    assert_close(&field_values(&input)?, &input_before, 0.0);

    let left = other.inner_product(&d_input)?;
    let mut ddag_other = FermionField::zeros(lattice, 4)?;
    operator.adjoint().apply_into(&mut ddag_other, &other)?;
    let rhs = ddag_other.inner_product(&input)?;
    let adjoint_residual = (left - rhs).norm();
    eprintln!("adjoint_inner_product_residual={adjoint_residual:.17e}");
    assert!(
        adjoint_residual <= 2e-12,
        "adjoint residual {adjoint_residual:.3e}"
    );

    let gamma5_input = input.gamma5()?;
    let mut d_gamma5 = FermionField::zeros(lattice, 4)?;
    operator.apply_into(&mut d_gamma5, &gamma5_input)?;
    let gamma5_d_gamma5 = d_gamma5.gamma5()?;
    assert_close(
        &field_values(&gamma5_d_gamma5)?,
        &field_values(&ddag_input)?,
        2e-14,
    );
    Ok(())
}

#[test]
fn free_plane_wave_matches_closed_form() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([4, 4, 4, 4])?;
    let links = cold_su3(lattice)?;
    let boundary = FermionBoundary::new([1, 1, 1, 1])?;
    let momentum = [
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        0.0,
        3.0 * std::f64::consts::FRAC_PI_2,
    ];
    let base = [
        [
            C::new(0.2, -0.1),
            C::new(-0.3, 0.4),
            C::new(0.5, 0.7),
            C::new(-0.2, -0.6),
        ],
        [
            C::new(-0.1, 0.8),
            C::new(0.6, -0.2),
            C::new(-0.4, 0.3),
            C::new(0.7, 0.1),
        ],
        [
            C::new(0.9, 0.2),
            C::new(-0.5, -0.7),
            C::new(0.1, 0.6),
            C::new(0.4, -0.4),
        ],
    ];
    let mut input_values = vec![ZERO; 12 * lattice.nv()];
    for site in 0..lattice.nv() {
        let [x, y, z, t] = coords_from_site_index(site, lattice)?;
        let phase = C::from_polar(
            1.0,
            momentum[0] * x as f64
                + momentum[1] * y as f64
                + momentum[2] * z as f64
                + momentum[3] * t as f64,
        );
        let mut spinor = [[ZERO; 4]; 3];
        for color in 0..3 {
            for component in 0..4 {
                spinor[color][component] = base[color][component] * phase;
            }
        }
        store_spinor(&mut input_values, site, spinor);
    }
    let input = FermionField::from_vec_col_major(lattice, 4, input_values.clone())?;
    let operator = WilsonDirac::with_boundary(&links, 0.09, boundary)?;
    let mut output = FermionField::zeros(lattice, 4)?;
    operator.apply_into(&mut output, &input)?;
    let mut expected = vec![ZERO; input_values.len()];
    let scalar = 1.0 - 2.0 * 0.09 * momentum.iter().map(|p| p.cos()).sum::<f64>();
    for site in 0..lattice.nv() {
        let [x, y, z, t] = coords_from_site_index(site, lattice)?;
        let phase = C::from_polar(
            1.0,
            momentum[0] * x as f64
                + momentum[1] * y as f64
                + momentum[2] * z as f64
                + momentum[3] * t as f64,
        );
        let mut spinor = [[ZERO; 4]; 3];
        for color in 0..3 {
            for (row, &base_value) in base[color].iter().enumerate() {
                let mut value = scalar * base_value;
                for (direction, &momentum_value) in momentum.iter().enumerate() {
                    let coefficient = C::new(0.0, 2.0 * 0.09 * momentum_value.sin());
                    let gamma_row = gamma(direction)[row];
                    for (column, &gamma_value) in gamma_row.iter().enumerate() {
                        value += coefficient * gamma_value * base[color][column];
                    }
                }
                spinor[color][row] = value * phase;
            }
        }
        store_spinor(&mut expected, site, spinor);
    }
    assert_close(&field_values(&output)?, &expected, 2e-14);
    Ok(())
}

#[test]
fn normal_operator_is_transactional_when_the_second_stencil_overflows() -> Result<(), Box<dyn Error>>
{
    let lattice = LatticeShape4::new([3, 1, 1, 1])?;
    let mut link_tensors = Vec::with_capacity(4);
    for direction in 0..4 {
        let mut values = vec![ZERO; 9 * lattice.nv()];
        for site in 0..lattice.nv() {
            let scale = if direction == 0 && site == 1 {
                1e-100
            } else {
                1e-308
            };
            for color in 0..3 {
                values[color + 3 * color + 9 * site] = C::new(scale, 0.0);
            }
        }
        let tensor = TypedTensor::from_vec_col_major(vec![3, 3, 3, 1, 1, 1], values)?;
        link_tensors.push(GaugeLinkTensor::from_typed(tensor, lattice)?);
    }
    let links = GaugeLinks::new(link_tensors.try_into().map_err(|_| "four links")?)?;
    let input =
        FermionField::from_vec_col_major(lattice, 4, vec![C::new(1.0, 0.0); 12 * lattice.nv()])?;
    let operator = WilsonDirac::new(&links, 1e308)?;
    let mut output =
        FermionField::from_vec_col_major(lattice, 4, vec![C::new(3.0, -2.0); 12 * lattice.nv()])?;
    let before = field_values(&output)?;

    assert!(matches!(
        operator.normal().apply_into(&mut output, &input),
        Err(DiracError::NumericalRange)
    ));
    assert_close(&field_values(&output)?, &before, 0.0);
    Ok(())
}

#[test]
fn invalid_operator_and_output_errors_are_transactional() -> Result<(), Box<dyn Error>> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let links = cold_su3(lattice)?;
    let input = input_field(lattice)?;
    let mut output =
        FermionField::from_vec_col_major(lattice, 4, vec![C::new(3.0, -2.0); 12 * lattice.nv()])?;
    let before = field_values(&output)?;
    let wrong_lattice = FermionField::zeros(LatticeShape4::new([1, 2, 2, 2])?, 4)?;
    let operator = WilsonDirac::new(&links, 0.1)?;
    assert!(matches!(
        operator.apply_into(&mut output, &wrong_lattice),
        Err(DiracError::LatticeMismatch { .. })
    ));
    assert_close(&field_values(&output)?, &before, 0.0);
    assert!(matches!(
        WilsonDirac::with_r(&links, 0.1, 0.9, FermionBoundary::default()),
        Err(DiracError::UnsupportedWilsonR { .. })
    ));
    assert!(matches!(
        WilsonDirac::new(&links, 0.0),
        Err(DiracError::NonPositiveKappa { .. })
    ));
    assert!(matches!(
        WilsonDirac::new(&links, f64::NAN),
        Err(DiracError::NonFiniteKappa { .. })
    ));
    let mut wrong_components = FermionField::zeros(lattice, 1)?;
    assert!(matches!(
        operator.apply_into(&mut wrong_components, &input),
        Err(DiracError::ComponentsMismatch { .. })
    ));
    assert_close(
        &field_values(&wrong_components)?,
        &vec![ZERO; 3 * lattice.nv()],
        0.0,
    );
    Ok(())
}
