use super::*;

fn assert_bits(actual: f64, expected: u64) {
    assert_eq!(actual.to_bits(), expected, "actual={actual:?}");
}

fn scalar_error(alpha0: f64, alpha: &[f64], beta: &[f64], power: f64) -> f64 {
    let mut maximum = 0.0_f64;
    for index in 0..=4_096 {
        let x = if index == 0 {
            TABLE_LAMBDA_LOW
        } else if index == 4_096 {
            TABLE_LAMBDA_HIGH
        } else {
            TABLE_LAMBDA_LOW
                * ((TABLE_LAMBDA_HIGH / TABLE_LAMBDA_LOW).ln() * index as f64 / 4_096.0).exp()
        };
        let value = alpha0
            + alpha
                .iter()
                .zip(beta)
                .map(|(alpha, beta)| alpha / (x + beta))
                .sum::<f64>();
        maximum = maximum.max((value - x.powf(power)).abs());
    }
    maximum
}

fn assert_within_ulps(actual: f64, expected: f64, maximum_ulps: u64) {
    assert!(actual.is_finite() && actual >= 0.0);
    assert!(expected.is_finite() && expected >= 0.0);
    let ulps = actual.to_bits().abs_diff(expected.to_bits());
    assert!(
        ulps <= maximum_ulps,
        "actual={actual:.17e} expected={expected:.17e} ulps={ulps}"
    );
}

#[test]
fn pinned_coefficient_bits_and_roles_are_exact() {
    assert_bits(REFRESH_COEFFICIENTS.alpha0, 0x4004_fb33_4399_8740);
    assert_bits(ACTION_INVERSE_COEFFICIENTS.alpha0, 0x3fd8_6719_edfe_5877);
    assert_bits(MD_FORCE_INVERSE_COEFFICIENTS.alpha0, 0x3fc6_641f_7427_6577);

    let refresh_alpha = [
        0xbed9_0054_503f_038c,
        0xbef8_cefc_7b54_43d7,
        0xbf13_2ba7_87f7_57f0,
        0xbf2c_5f9d_55f0_13e4,
        0xbf44_c891_ffc6_ec5c,
        0xbf5e_5c0e_7c7f_0279,
        0xbf76_28ef_e922_3944,
        0xbf90_2f8d_2e93_4be3,
        0xbfa7_b4aa_cd3e_2693,
        0xbfc1_7b7f_76aa_71e2,
        0xbfda_43d0_92a5_edc6,
        0xbff4_b3fc_384b_e067,
        0xc012_8be3_6568_a154,
        0xc037_c821_72f5_b4c5,
        0xc07a_ac60_8b39_8a07,
    ];
    let action_alpha = [
        0x3f0d_e885_afab_aaff,
        0x3f23_f082_95f8_a13b,
        0x3f36_e9b8_40da_de4f,
        0x3f4a_251b_4b0c_fadb,
        0x3f5d_efb3_ec8e_32ce,
        0x3f71_2dd3_e4b3_9bb4,
        0x3f83_be08_01e4_2c2b,
        0x3f96_b74e_0ab8_40a5,
        0x3faa_32ac_679f_92e1,
        0x3fbe_623f_1e06_520a,
        0x3fd1_e027_709a_7169,
        0x3fe5_d6f8_d33f_0a0f,
        0x3ffd_7854_ba16_b72e,
        0x401a_2519_179c_baee,
        0x404c_740c_224d_7811,
    ];
    let force_alpha = [
        0x3f45_9f19_959f_2361,
        0x3f5e_1471_4e09_8a43,
        0x3f74_e670_4a79_22c3,
        0x3f8d_aeae_fcf4_ff28,
        0x3fa5_3a4a_f954_da27,
        0x3fbe_7a4a_e736_58f4,
        0x3fd6_134f_6a0b_6b2c,
        0x3ff0_8e86_d7ef_479b,
        0x400c_ae87_739b_6fbc,
        0x4037_be3a_0510_06d9,
    ];
    let refresh_beta = [
        0x3f11_11bb_2223_8fa8,
        0x3f39_15e0_3fcf_abee,
        0x3f54_c52e_a329_50ed,
        0x3f6d_5a48_f0b5_e094,
        0x3f83_a288_eaf9_7fd0,
        0x3f99_bdf2_e857_f34c,
        0x3fb0_bf5d_1bf5_e8cb,
        0x3fc5_bcde_cbf9_7d67,
        0x3fdc_3814_c858_717c,
        0x3ff2_5ea3_2e04_3bde,
        0x4008_1f17_a82d_03bc,
        0x4020_343c_3b07_6cac,
        0x4037_2b39_fd11_146a,
        0x4053_e305_5263_d598,
        0x4081_3255_449c_50e7,
    ];
    let action_beta = [
        0x3f08_63ea_130d_709e,
        0x3f35_1744_dcd6_86d6,
        0x3f52_1a6b_57c6_2d77,
        0x3f69_e261_bd08_9721,
        0x3f81_6365_c035_fed8,
        0x3f96_d52f_a194_4224,
        0x3fad_ba0b_779a_1dba,
        0x3fc3_4b85_3cc2_05ac,
        0x3fd9_0b59_aec8_70f6,
        0x3ff0_4b2b_46a3_b15f,
        0x4005_5c89_6b56_afaa,
        0x401c_9424_8d41_5da9,
        0x4034_319d_ba05_7100,
        0x4050_b856_2aa3_47c5,
        0x4078_927f_e090_29f9,
    ];
    let force_beta = [
        0x3f16_9534_cf32_fe7c,
        0x3f4a_33b2_394e_3705,
        0x3f6f_51dc_e007_c4cd,
        0x3f90_8be2_349d_f265,
        0x3fb0_fed1_9a78_e3fe,
        0x3fd1_5b53_7136_4d8d,
        0x3ff1_c839_f784_ffaa,
        0x4012_941c_323f_f108,
        0x4035_20be_7d02_873e,
        0x4062_326c_230c_3e42,
    ];
    for (actual, expected) in REFRESH_COEFFICIENTS.alpha.iter().zip(refresh_alpha) {
        assert_bits(*actual, expected);
    }
    for (actual, expected) in ACTION_INVERSE_COEFFICIENTS.alpha.iter().zip(action_alpha) {
        assert_bits(*actual, expected);
    }
    for (actual, expected) in MD_FORCE_INVERSE_COEFFICIENTS.alpha.iter().zip(force_alpha) {
        assert_bits(*actual, expected);
    }
    for (actual, expected) in REFRESH_COEFFICIENTS.beta.iter().zip(refresh_beta) {
        assert_bits(*actual, expected);
    }
    for (actual, expected) in ACTION_INVERSE_COEFFICIENTS.beta.iter().zip(action_beta) {
        assert_bits(*actual, expected);
    }
    for (actual, expected) in MD_FORCE_INVERSE_COEFFICIENTS.beta.iter().zip(force_beta) {
        assert_bits(*actual, expected);
    }
    assert_ne!(
        REFRESH_COEFFICIENTS.beta[0].to_bits(),
        ACTION_INVERSE_COEFFICIENTS.beta[0].to_bits()
    );
    assert_ne!(
        ACTION_INVERSE_COEFFICIENTS.alpha[0].to_bits(),
        MD_FORCE_INVERSE_COEFFICIENTS.alpha[0].to_bits()
    );
}

#[test]
fn pinned_scalar_log_grid_errors_are_bounded() {
    let refresh_error = scalar_error(
        REFRESH_COEFFICIENTS.alpha0,
        &REFRESH_COEFFICIENTS.alpha,
        &REFRESH_COEFFICIENTS.beta,
        1.0 / 8.0,
    );
    let action_error = scalar_error(
        ACTION_INVERSE_COEFFICIENTS.alpha0,
        &ACTION_INVERSE_COEFFICIENTS.alpha,
        &ACTION_INVERSE_COEFFICIENTS.beta,
        -1.0 / 8.0,
    );
    let force_error = scalar_error(
        MD_FORCE_INVERSE_COEFFICIENTS.alpha0,
        &MD_FORCE_INVERSE_COEFFICIENTS.alpha,
        &MD_FORCE_INVERSE_COEFFICIENTS.beta,
        -1.0 / 4.0,
    );
    eprintln!(
        "scalar log-grid max errors: refresh={refresh_error:.17e}, action={action_error:.17e}, force={force_error:.17e}"
    );
    assert_within_ulps(refresh_error, 2.505791796281187e-9, 4);
    assert_within_ulps(action_error, 3.9620045022559225e-9, 4);
    assert_within_ulps(force_error, 1.5595609319518644e-5, 4);
}
