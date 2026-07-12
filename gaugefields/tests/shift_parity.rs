use gaugefields::{load_fixture, neighbor_site};
use num_complex::Complex64;
use std::{fs, path::Path};

#[test]
fn direct_neighbor_reads_match_all_julia_materialized_shifts() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/shifts_3x2x4x5");
    let fixture = load_fixture(&dir).unwrap();
    let links = fixture.links();
    for link_mu in 0..4 {
        let source = links.links()[link_mu]
            .tensor()
            .as_slice::<Complex64>()
            .unwrap();
        for axis in 0..4 {
            for (sign, label) in [(-1, "minus"), (1, "plus")] {
                let bytes =
                    fs::read(dir.join(format!("u{link_mu}_shift{axis}_{label}.npy"))).unwrap();
                let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
                assert_eq!(npy.order(), npyz::Order::Fortran);
                assert_eq!(npy.shape(), &[3, 3, 3, 2, 4, 5]);
                let shifted = npy.into_vec::<Complex64>().unwrap();
                for site in 0..links.lattice().nv() {
                    let neighbor = neighbor_site(site, axis, sign, links.lattice()).unwrap();
                    assert_eq!(
                        &shifted[9 * site..9 * site + 9],
                        &source[9 * neighbor..9 * neighbor + 9],
                        "link_mu={link_mu} axis={axis} sign={sign} site={site}"
                    );
                }
            }
        }
    }
}
