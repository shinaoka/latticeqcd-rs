use super::*;
use crate::{cold_su3, LatticeShape4};

fn momentum(lattice: LatticeShape4) -> TaGaugeField {
    let [nx, ny, nz, nt] = lattice.extents();
    let tensors = std::array::from_fn(|mu| {
        TypedTensor::from_vec_col_major(
            vec![8, nx, ny, nz, nt],
            vec![(mu + 1) as f64 / 17.0; 8 * lattice.nv()],
        )
        .unwrap()
    });
    TaGaugeField::new(tensors, lattice).unwrap()
}

#[test]
fn injected_direction_failures_leave_every_link_bitwise_unchanged() {
    let lattice = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    for direction in 0..4 {
        let mut links = cold_su3(lattice).unwrap();
        let before: [Vec<C>; 4] =
            std::array::from_fn(|mu| links.links()[mu].typed().host_data().unwrap().to_vec());
        let result = exp_ta_update_with(
            &mut CpuEvolutionContext::new(CpuBackend::new()),
            &mut links,
            0.1,
            &momentum(lattice),
            |session, mu, lhs, rhs, config| {
                if mu == direction {
                    Err(tenferro_tensor::Error::backend_failure(
                        "exp_ta_update test injection",
                        format!("direction {mu}"),
                    ))
                } else {
                    SessionCachedDot::dot_general_cached(session, Some(mu), lhs, rhs, config)
                }
            },
        );
        assert!(matches!(result, Err(GaugeError::Evolution { .. })));
        for (mu, expected_direction) in before.iter().enumerate() {
            for (actual, expected) in links.links()[mu]
                .typed()
                .host_data()
                .unwrap()
                .iter()
                .zip(expected_direction)
            {
                assert_eq!(actual.re.to_bits(), expected.re.to_bits());
                assert_eq!(actual.im.to_bits(), expected.im.to_bits());
            }
        }
    }
}
