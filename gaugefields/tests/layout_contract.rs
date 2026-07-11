use gaugefields::flat_offset;

#[test]
fn flat_offset_is_column_major_with_x_fastest_sites() {
    let [nc, nx, ny, nz] = [3, 2, 3, 4];
    assert_eq!(flat_offset(1, 0, 0, 0, 0, 0, nc, nx, ny, nz), 1);
    assert_eq!(flat_offset(0, 1, 0, 0, 0, 0, nc, nx, ny, nz), 3);
    assert_eq!(flat_offset(0, 0, 1, 0, 0, 0, nc, nx, ny, nz), 9);
    assert_eq!(flat_offset(0, 0, 0, 1, 0, 0, nc, nx, ny, nz), 18);
    assert_eq!(flat_offset(0, 0, 0, 0, 1, 0, nc, nx, ny, nz), 54);
    assert_eq!(flat_offset(0, 0, 0, 0, 0, 1, nc, nx, ny, nz), 216);
}
