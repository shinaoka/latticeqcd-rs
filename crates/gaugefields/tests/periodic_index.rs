use gaugefields::{
    cold_su3, coords_from_site_index, load_link, neighbor_site, site_index, store_link, GaugeError,
    LatticeShape4, Mat3,
};
use num_complex::Complex64 as C;

#[test]
fn site_coordinate_roundtrip_and_errors() {
    let l = LatticeShape4::new([3, 2, 4, 5]).unwrap();
    for t in 0..5 {
        for z in 0..4 {
            for y in 0..2 {
                for x in 0..3 {
                    let c = [x, y, z, t];
                    let s = site_index(c, l).unwrap();
                    assert_eq!(coords_from_site_index(s, l).unwrap(), c);
                }
            }
        }
    }
    assert!(matches!(
        site_index([3, 0, 0, 0], l),
        Err(GaugeError::CoordinateOutOfBounds { axis: 0, .. })
    ));
    assert!(matches!(
        coords_from_site_index(l.nv(), l),
        Err(GaugeError::SiteOutOfBounds { .. })
    ));
}

#[test]
fn neighbors_use_euclidean_periodicity_for_arbitrary_offsets() {
    let l = LatticeShape4::new([3, 2, 4, 5]).unwrap();
    for site in 0..l.nv() {
        let c = coords_from_site_index(site, l).unwrap();
        for mu in 0..4 {
            let n = l.extents()[mu] as i64;
            for d in (-2 * n - 1)..=(2 * n + 1) {
                let q = neighbor_site(site, mu, d, l).unwrap();
                let mut expected = c;
                expected[mu] = (c[mu] as i64 + d).rem_euclid(n) as usize;
                assert_eq!(coords_from_site_index(q, l).unwrap(), expected);
            }
        }
    }
    assert!(matches!(
        neighbor_site(0, 4, 1, l),
        Err(GaugeError::InvalidDirection { direction: 4 })
    ));
}

#[test]
fn direct_link_load_store_uses_contiguous_nine_element_blocks() {
    let l = LatticeShape4::new([2, 1, 1, 1]).unwrap();
    let mut links = cold_su3(l).unwrap();
    let value = Mat3::from_array(std::array::from_fn(|i| C::new(i as f64, -(i as f64))));
    store_link(&mut links, 2, 1, value).unwrap();
    assert_eq!(load_link(&links, 2, 1).unwrap(), value);
    assert_eq!(load_link(&links, 2, 0).unwrap(), Mat3::identity());
    assert!(matches!(
        load_link(&links, 4, 0),
        Err(GaugeError::InvalidDirection { .. })
    ));
}
