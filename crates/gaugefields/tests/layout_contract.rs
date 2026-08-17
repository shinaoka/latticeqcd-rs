#[test]
fn flat_offset_is_column_major_with_x_fastest_sites() {
    let flat_offset =
        |a, b, x, y, z, t, nc, nx, ny, nz| a + nc * (b + nc * (x + nx * (y + ny * (z + nz * t))));
    let [nc, nx, ny, nz] = [3, 2, 3, 4];
    assert_eq!(flat_offset(1, 0, 0, 0, 0, 0, nc, nx, ny, nz), 1);
    assert_eq!(flat_offset(0, 1, 0, 0, 0, 0, nc, nx, ny, nz), 3);
    assert_eq!(flat_offset(0, 0, 1, 0, 0, 0, nc, nx, ny, nz), 9);
    assert_eq!(flat_offset(0, 0, 0, 1, 0, 0, nc, nx, ny, nz), 18);
    assert_eq!(flat_offset(0, 0, 0, 0, 1, 0, nc, nx, ny, nz), 54);
    assert_eq!(flat_offset(0, 0, 0, 0, 0, 1, nc, nx, ny, nz), 216);
}

#[test]
fn distinct_c64_values_detect_every_axis_and_direction() {
    use num_complex::Complex64;
    use tenferro_tensor::Tensor;
    let [nc, nx, ny, nz, nt] = [3, 2, 2, 2, 2];
    for mu in 0..4 {
        let mut values = Vec::new();
        for t in 0..nt {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        for b in 0..nc {
                            for a in 0..nc {
                                values.push(Complex64::new(
                                    (a + 10 * b
                                        + 100 * x
                                        + 1_000 * y
                                        + 10_000 * z
                                        + 100_000 * t
                                        + 1_000_000 * mu)
                                        as f64,
                                    -(mu as f64),
                                ));
                            }
                        }
                    }
                }
            }
        }
        let tensor = Tensor::from_vec_col_major(vec![nc, nc, nx, ny, nz, nt], values).unwrap();
        let slice = tensor.as_slice::<Complex64>().unwrap();
        for &(a, b, x, y, z, t) in &[
            (1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 1),
        ] {
            let offset = a + nc * (b + nc * (x + nx * (y + ny * (z + nz * t))));
            assert_eq!(
                slice[offset],
                Complex64::new(
                    (a + 10 * b + 100 * x + 1_000 * y + 10_000 * z + 100_000 * t + 1_000_000 * mu)
                        as f64,
                    -(mu as f64)
                )
            );
        }
    }
}
