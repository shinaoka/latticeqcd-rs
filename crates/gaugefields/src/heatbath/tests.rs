use super::*;
use crate::field::duplicate_links;
use crate::{GaugeLinkTensor, GaugeLinks, ReproducibleRng};
use num_complex::Complex64 as C;
use rand::RngCore;
use tenferro_tensor::TypedTensor;

fn scripted_first_try_draws() -> [f64; 6] {
    [0.5, 0.5, 0.25, 0.5, 0.25, 0.5]
}

fn zero_links(lattice: LatticeShape4) -> GaugeLinks {
    let [nx, ny, nz, nt] = lattice.extents();
    let values = vec![C::default(); 9 * lattice.nv()];
    let make = || {
        GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], values.clone()).unwrap(),
            lattice,
        )
        .unwrap()
    };
    GaugeLinks::new([make(), make(), make(), make()]).unwrap()
}

fn nonfinite_links(lattice: LatticeShape4) -> GaugeLinks {
    let [nx, ny, nz, nt] = lattice.extents();
    let mut values = vec![C::new(1.0, 0.0); 9 * lattice.nv()];
    values[0] = C::new(f64::NAN, 0.0);
    let make = || {
        GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(vec![3, 3, nx, ny, nz, nt], values.clone()).unwrap(),
            lattice,
        )
        .unwrap()
    };
    GaugeLinks::new([make(), make(), make(), make()]).unwrap()
}

fn bitwise_links_equal(lhs: &GaugeLinks, rhs: &GaugeLinks) {
    for mu in 0..4 {
        for (left, right) in lhs.links()[mu]
            .typed()
            .host_data()
            .unwrap()
            .iter()
            .zip(rhs.links()[mu].typed().host_data().unwrap())
        {
            assert_eq!(left.re.to_bits(), right.re.to_bits());
            assert_eq!(left.im.to_bits(), right.im.to_bits());
        }
    }
}

#[test]
fn injected_sampler_records_direction_parity_site_subgroup_order() {
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let mut links = crate::cold_su3(lattice).unwrap();
    let params = HeatbathParams::new(5.7, 1).unwrap();
    let draws = scripted_first_try_draws();
    let mut draw_index = 0;
    let mut trace = Vec::new();
    let stats = heatbath_sweep_core(
        &mut links,
        params,
        &mut || {
            let value = draws[draw_index % draws.len()];
            draw_index += 1;
            value
        },
        &mut |direction, parity, site, subgroup, attempt| {
            trace.push((direction, parity, site, subgroup, attempt));
        },
    )
    .unwrap();

    assert_eq!(stats.updated_links, 4 * lattice.nv());
    assert_eq!(stats.su2_attempts, 3 * stats.updated_links);
    assert_eq!(draw_index, 6 * stats.su2_attempts);
    let mut expected = Vec::new();
    for direction in 0..4 {
        for parity in [true, false] {
            for site in 0..lattice.nv() {
                let [x, y, z, t] = crate::coords_from_site_index(site, lattice).unwrap();
                if ((x + y + z + t) % 2 == 0) != parity {
                    continue;
                }
                for subgroup in 0..3 {
                    expected.push((direction, parity, site, subgroup, 1));
                }
            }
        }
    }
    assert_eq!(trace, expected);
}

#[test]
fn projected_subgroup_uses_square_root_norm() {
    let input = [
        C::new(2.0, 0.5),
        C::new(-1.0, 0.25),
        C::new(0.75, -1.25),
        C::new(0.5, -0.25),
    ];
    let alpha = (input[0] + input[3].conj()) * 0.5;
    let beta = (input[2] - input[1].conj()) * 0.5;
    let norm = (alpha.norm_sqr() + beta.norm_sqr()).sqrt();
    assert!(norm.is_finite() && norm > 0.0 && (norm - 1.0).abs() > 1e-6);
    let actual = project_and_normalize_su2(input).unwrap();
    let expected = [
        alpha / norm,
        -beta.conj() / norm,
        beta / norm,
        alpha.conj() / norm,
    ];
    assert_eq!(actual, expected);
    assert!((actual[0].norm_sqr() + actual[2].norm_sqr() - 1.0).abs() < 1e-15);
}

#[test]
fn singular_and_nonfinite_kernel_fail_without_draws() {
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let mut links = zero_links(lattice);
    let before = duplicate_links(&links).unwrap();
    let mut draws = 0;
    let result = heatbath_sweep_core(
        &mut links,
        HeatbathParams::new(5.7, 1).unwrap(),
        &mut || {
            draws += 1;
            0.5
        },
        &mut |_, _, _, _, _| {},
    );
    assert!(matches!(
        result,
        Err(GaugeError::SingularHeatbathStaple {
            direction: 0,
            site: 0,
            subgroup: 0
        })
    ));
    assert_eq!(draws, 0);
    bitwise_links_equal(&links, &before);

    let mut links = nonfinite_links(lattice);
    let before = duplicate_links(&links).unwrap();
    let mut draws = 0;
    let result = heatbath_sweep_core(
        &mut links,
        HeatbathParams::new(5.7, 1).unwrap(),
        &mut || {
            draws += 1;
            0.5
        },
        &mut |_, _, _, _, _| {},
    );
    assert!(matches!(
        result,
        Err(GaugeError::HeatbathNumericalRange { stage: "staple" })
    ));
    assert_eq!(draws, 0);
    bitwise_links_equal(&links, &before);
}

#[test]
fn rejection_limit_counts_every_attempt_and_consumes_only_attempt_draws() {
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let mut links = crate::cold_su3(lattice).unwrap();
    let before = duplicate_links(&links).unwrap();
    let scripted = [0.5, 0.5, 0.25, 0.999];
    let mut draw_index = 0;
    let result = heatbath_sweep_core(
        &mut links,
        HeatbathParams::new(5.7, 2).unwrap(),
        &mut || {
            let value = scripted[draw_index % scripted.len()];
            draw_index += 1;
            value
        },
        &mut |_, _, _, _, _| panic!("rejection must not notify an update"),
    );
    assert!(matches!(
        result,
        Err(GaugeError::HeatbathRejectionLimit { max_attempts: 2 })
    ));
    assert_eq!(draw_index, 8);
    bitwise_links_equal(&links, &before);
}

#[test]
fn heatbath_allocation_metadata_rejects_supported_volume_out_of_range() {
    let extent = usize::MAX / 8 - 1;
    let lattice = LatticeShape4::new([extent, 2, 2, 2]).unwrap();
    assert!(matches!(
        checked_heatbath_sizes(lattice),
        Err(GaugeError::AllocationOverflow)
    ));
}

#[test]
fn public_validation_is_before_rng_and_rejection_is_transactional() {
    let odd = LatticeShape4::new([3, 2, 2, 2]).unwrap();
    let mut odd_links = crate::cold_su3(odd).unwrap();
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4]).unwrap();
    let mut replay = rng.clone();
    assert!(matches!(
        crate::heatbath_sweep(
            &mut odd_links,
            HeatbathParams::new(5.7, 1).unwrap(),
            &mut rng,
        ),
        Err(GaugeError::OddHeatbathExtent { axis: 0, extent: 3 })
    ));
    assert_eq!(rng.next_u64(), replay.next_u64());
}
