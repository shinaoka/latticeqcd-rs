use super::*;

#[test]
fn pinned_gamma_matrices_are_hermitian_and_clifford() {
    for (mu, gamma_mu) in GAMMA.iter().enumerate() {
        for (row, row_values) in gamma_mu.iter().enumerate() {
            for (column, &value) in row_values.iter().enumerate() {
                assert_eq!(value, gamma_mu[column][row].conj());
            }
        }
        for (row, row_values) in gamma_mu.iter().enumerate() {
            for (column, _) in row_values.iter().enumerate() {
                let mut square = C0;
                for (middle, _) in row_values.iter().enumerate() {
                    square += gamma_mu[row][middle] * gamma_mu[middle][column];
                }
                assert_eq!(square, if row == column { C1 } else { C0 });
            }
        }
        let projected = project_spin(mu, -1, [C1, C0, C0, C0]);
        let reconstructed = project_spin(mu, 1, projected);
        let norm = reconstructed
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>();
        assert!(norm.is_finite());
    }
    assert_eq!(GAMMA[0][0][3], CNI);
    assert_eq!(GAMMA[3][0][2], CN1);
}
