use gaugefields::{
    cold_su3, heatbath_sweep, normalized_plaquette, GaugeError, GaugeLinkTensor, GaugeLinks,
    HeatbathParams, LatticeShape4, Mat3, ReproducibleRng,
};
use num_complex::Complex64;
use rand::RngCore;
use tenferro_tensor::{
    BackendStorageHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
    StorageBuffer, TypedTensor,
};

fn arbitrary_links(lattice: LatticeShape4, nc: usize, value: Complex64) -> GaugeLinks {
    let [nx, ny, nz, nt] = lattice.extents();
    let values = vec![value; nc * nc * lattice.nv()];
    let make = || {
        GaugeLinkTensor::from_typed(
            TypedTensor::from_vec_col_major(vec![nc, nc, nx, ny, nz, nt], values.clone()).unwrap(),
            lattice,
        )
        .unwrap()
    };
    GaugeLinks::new([make(), make(), make(), make()]).unwrap()
}

fn snapshot(links: &GaugeLinks) -> [Vec<Complex64>; 4] {
    std::array::from_fn(|mu| links.links()[mu].typed().host_data().unwrap().to_vec())
}

fn assert_snapshot(links: &GaugeLinks, before: &[Vec<Complex64>; 4]) {
    for (mu, expected) in before.iter().enumerate() {
        for (actual, expected) in links.links()[mu]
            .typed()
            .host_data()
            .unwrap()
            .iter()
            .zip(expected)
        {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits(), "mu={mu} real");
            assert_eq!(actual.im.to_bits(), expected.im.to_bits(), "mu={mu} imag");
        }
    }
}

fn su3_residual(links: &GaugeLinks) -> (f64, f64) {
    let mut unitary = 0.0_f64;
    let mut determinant = 0.0_f64;
    for link in links.links() {
        for block in link.typed().host_data().unwrap().chunks_exact(9) {
            let matrix = Mat3::load(block, 0).unwrap();
            let product = matrix.adjoint().mul(matrix);
            for row in 0..3 {
                for column in 0..3 {
                    let expected = if row == column { 1.0 } else { 0.0 };
                    unitary = unitary.max((product[(row, column)] - expected).norm());
                }
            }
            let a = matrix.as_array();
            let det = a[0] * (a[4] * a[8] - a[7] * a[5]) - a[3] * (a[1] * a[8] - a[7] * a[2])
                + a[6] * (a[1] * a[5] - a[4] * a[2]);
            determinant = determinant.max((det - Complex64::new(1.0, 0.0)).norm());
        }
    }
    (unitary, determinant)
}

#[test]
fn heatbath_public_contract_is_present() -> Result<(), GaugeError> {
    assert!(matches!(
        HeatbathParams::new(f64::NAN, 1),
        Err(GaugeError::NonFiniteBeta { .. })
    ));
    assert!(matches!(
        HeatbathParams::new(0.0, 1),
        Err(GaugeError::NonPositiveHeatbathBeta { .. })
    ));
    assert!(matches!(
        HeatbathParams::new(5.7, 0),
        Err(GaugeError::ZeroHeatbathAttempts)
    ));

    let params = HeatbathParams::new(5.7, 100_000)?;
    assert_eq!(params.beta(), 5.7);
    assert_eq!(params.max_attempts(), 100_000);
    assert!(!format!("{params:?}").contains("GaugeLinks"));

    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let mut links = cold_su3(lattice)?;
    let before = normalized_plaquette(&links)?;
    let mut replay_links = cold_su3(lattice)?;
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    let mut replay_rng = rng.clone();
    let stats = heatbath_sweep(&mut links, params, &mut rng)?;
    let replay_stats = heatbath_sweep(&mut replay_links, params, &mut replay_rng)?;
    assert_eq!(stats, replay_stats);
    assert_snapshot(&replay_links, &snapshot(&links));
    assert_eq!(rng.next_u64(), replay_rng.next_u64());
    assert_eq!(stats.updated_links, 4 * lattice.nv());
    assert_eq!(stats.su2_attempts, 195);
    let (unitary, determinant) = su3_residual(&links);
    assert!(unitary < 2e-14, "unitarity residual={unitary:e}");
    assert!(determinant < 2e-14, "determinant residual={determinant:e}");
    assert!(normalized_plaquette(&links)?.is_finite());
    assert_ne!(normalized_plaquette(&links)?, before);
    Ok(())
}

#[test]
fn host_boundary_rejects_device_link_storage() {
    let lattice = LatticeShape4::new([2, 2, 2, 2]).unwrap();
    let placement = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
        cpu_affinity: None,
    };
    let link = TypedTensor::from_buffer_col_major(
        vec![3, 3, 2, 2, 2, 2],
        StorageBuffer::Backend(Box::new(BackendStorageHandle::<Complex64>::new_with_len(
            1, 144,
        ))),
        placement,
    )
    .unwrap();
    assert!(matches!(
        GaugeLinkTensor::from_typed(link, lattice),
        Err(GaugeError::Placement { .. })
    ));
}

#[test]
fn allocation_overflow_is_rejected_before_field_creation() {
    let extent = usize::MAX / 8 - 1;
    let lattice = LatticeShape4::new([extent, 2, 2, 2]).unwrap();
    assert!(matches!(
        cold_su3(lattice),
        Err(GaugeError::AllocationOverflow)
    ));
}

#[test]
fn validation_errors_are_typed_and_do_not_advance_rng() -> Result<(), GaugeError> {
    let params = HeatbathParams::new(5.7, 1)?;

    let odd_lattice = LatticeShape4::new([3, 2, 2, 2])?;
    let mut odd_links = cold_su3(odd_lattice)?;
    let mut odd_rng = ReproducibleRng::from_state([7, 11, 13, 17])?;
    let mut odd_replay = odd_rng.clone();
    assert!(matches!(
        heatbath_sweep(&mut odd_links, params, &mut odd_rng),
        Err(GaugeError::OddHeatbathExtent { axis: 0, extent: 3 })
    ));
    assert_eq!(odd_rng.next_u64(), odd_replay.next_u64());

    let nc_lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let mut nc_links = arbitrary_links(nc_lattice, 2, Complex64::new(1.0, 0.0));
    let mut nc_rng = ReproducibleRng::from_state([19, 23, 29, 31])?;
    let mut nc_replay = nc_rng.clone();
    assert!(matches!(
        heatbath_sweep(&mut nc_links, params, &mut nc_rng),
        Err(GaugeError::UnsupportedNc { found: 2 })
    ));
    assert_eq!(nc_rng.next_u64(), nc_replay.next_u64());
    Ok(())
}

#[test]
fn singular_and_nonfinite_fields_are_transactional() -> Result<(), GaugeError> {
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let params = HeatbathParams::new(5.7, 1)?;

    let mut singular = arbitrary_links(lattice, 3, Complex64::new(0.0, 0.0));
    let before = snapshot(&singular);
    let mut rng = ReproducibleRng::from_state([37, 41, 43, 47])?;
    let mut replay = rng.clone();
    assert!(matches!(
        heatbath_sweep(&mut singular, params, &mut rng),
        Err(GaugeError::SingularHeatbathStaple {
            direction: 0,
            site: 0,
            subgroup: 0
        })
    ));
    assert_snapshot(&singular, &before);
    assert_eq!(rng.next_u64(), replay.next_u64());

    let mut nonfinite = arbitrary_links(lattice, 3, Complex64::new(1.0, 0.0));
    let mut values = nonfinite.links()[0].typed().host_data().unwrap().to_vec();
    values[0] = Complex64::new(f64::NAN, 0.0);
    let bad = GaugeLinkTensor::from_typed(
        TypedTensor::from_vec_col_major(vec![3, 3, 2, 2, 2, 2], values).unwrap(),
        lattice,
    )
    .unwrap();
    let [_, link1, link2, link3] = nonfinite.into_links();
    nonfinite = GaugeLinks::new([bad, link1, link2, link3])?;
    let before = snapshot(&nonfinite);
    let mut rng = ReproducibleRng::from_state([53, 59, 61, 67])?;
    let mut replay = rng.clone();
    assert!(matches!(
        heatbath_sweep(&mut nonfinite, params, &mut rng),
        Err(GaugeError::HeatbathNumericalRange { .. })
    ));
    assert_snapshot(&nonfinite, &before);
    assert_eq!(rng.next_u64(), replay.next_u64());
    Ok(())
}

#[test]
fn rejection_failure_is_transactional_but_rng_consumes_completed_attempts() -> Result<(), GaugeError>
{
    let lattice = LatticeShape4::new([2, 2, 2, 2])?;
    let mut links = cold_su3(lattice)?;
    let before = snapshot(&links);
    let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    let mut replay = rng.clone();
    assert!(matches!(
        heatbath_sweep(&mut links, HeatbathParams::new(5.7, 1)?, &mut rng),
        Err(GaugeError::HeatbathRejectionLimit { max_attempts: 1 })
    ));
    assert_snapshot(&links, &before);
    for _ in 0..4 {
        replay.open_unit_f64();
    }
    assert_eq!(rng.next_u64(), replay.next_u64());
    Ok(())
}
