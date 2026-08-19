//! Pinned two-flavor staggered RHMC coefficients and U-P-U trajectory.
//!
//! The coefficient tables and rational ordering follow the private hard-coded
//! tables in `src/rhmc/rhmc.jl` of LatticeDiracOperators.jl v0.6.4 at
//! `bdef628184597815ba3e0cddf2536df767e78a02`.  The positive degree-15 table
//! is the refresh `x^(+1/8)` table, the negative degree-15 table is the action
//! `x^(-1/8)` table, and the degree-10 negative table is the MD-force
//! `x^(-1/4)` table.  Rust keeps those roles in distinct private types; in
//! particular, it does not reproduce the pinned `RHMC(-1/8)` inverse-table
//! selection defect tracked as upstream issue #26.
//!
//! `RHMC` here is intentionally not a coefficient generator.  Only the
//! accepted two-flavor tables on `[0.0004, 64]` exist.  The public action
//! constructor validates a caller-provided finite positive spectral assertion
//! against that interval; it does not estimate or clamp the spectrum.

use crate::{
    multi_shift_cg, DiracError, FermionBoundary, FermionField, HermitianPositiveOperator,
    MultiShiftSolverReport, SolverParams, StaggeredDirac, StaggeredFermiAction,
};
use gaugefields::{
    exp_ta_update, gauge_force, kinetic_energy, require_su3, sample_momentum, wilson_action,
    CpuEvolutionContext, GaugeLinks, HmcParams as GaugeHmcParams, ReproducibleRng, TaGaugeField,
};
use num_complex::Complex64;
pub(crate) const TABLE_LAMBDA_LOW: f64 = 0.0004;
pub(crate) const TABLE_LAMBDA_HIGH: f64 = 64.0;
const RANK_ONE: usize = 15;
const RANK_TWO: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClaimedSpectralBounds {
    lower: f64,
    upper: f64,
}

impl ClaimedSpectralBounds {
    pub(crate) fn new(lower: f64, upper: f64) -> Result<Self, DiracError> {
        if !lower.is_finite() {
            return Err(DiracError::NonFiniteSpectralBound {
                bound: "lower",
                found: lower,
            });
        }
        if !upper.is_finite() {
            return Err(DiracError::NonFiniteSpectralBound {
                bound: "upper",
                found: upper,
            });
        }
        if lower <= 0.0 {
            return Err(DiracError::NonPositiveSpectralBound {
                bound: "lower",
                found: lower,
            });
        }
        if upper <= 0.0 {
            return Err(DiracError::NonPositiveSpectralBound {
                bound: "upper",
                found: upper,
            });
        }
        if lower > upper {
            return Err(DiracError::InvertedSpectralBounds { lower, upper });
        }
        if lower < TABLE_LAMBDA_LOW || upper > TABLE_LAMBDA_HIGH {
            return Err(DiracError::SpectralBoundsOutsideTable {
                lower,
                upper,
                table_lower: TABLE_LAMBDA_LOW,
                table_upper: TABLE_LAMBDA_HIGH,
            });
        }
        Ok(Self { lower, upper })
    }

    pub(crate) const fn lower(self) -> f64 {
        self.lower
    }

    pub(crate) const fn upper(self) -> f64 {
        self.upper
    }
}

#[derive(Clone, Copy)]
struct RefreshCoefficients {
    alpha0: f64,
    alpha: [f64; RANK_ONE],
    beta: [f64; RANK_ONE],
}

#[derive(Clone, Copy)]
struct ActionInverseCoefficients {
    alpha0: f64,
    alpha: [f64; RANK_ONE],
    beta: [f64; RANK_ONE],
}

#[derive(Clone, Copy)]
struct MdForceInverseCoefficients {
    alpha0: f64,
    alpha: [f64; RANK_TWO],
    beta: [f64; RANK_TWO],
}

// These are from `coeffs_18` in the pinned Julia source.  from_bits makes the
// recorded Julia Float64 payloads an explicit contract instead of depending on
// a source-language decimal parser.
const REFRESH_COEFFICIENTS: RefreshCoefficients = RefreshCoefficients {
    alpha0: f64::from_bits(0x4004_fb33_4399_8740),
    alpha: [
        f64::from_bits(0xbed9_0054_503f_038c),
        f64::from_bits(0xbef8_cefc_7b54_43d7),
        f64::from_bits(0xbf13_2ba7_87f7_57f0),
        f64::from_bits(0xbf2c_5f9d_55f0_13e4),
        f64::from_bits(0xbf44_c891_ffc6_ec5c),
        f64::from_bits(0xbf5e_5c0e_7c7f_0279),
        f64::from_bits(0xbf76_28ef_e922_3944),
        f64::from_bits(0xbf90_2f8d_2e93_4be3),
        f64::from_bits(0xbfa7_b4aa_cd3e_2693),
        f64::from_bits(0xbfc1_7b7f_76aa_71e2),
        f64::from_bits(0xbfda_43d0_92a5_edc6),
        f64::from_bits(0xbff4_b3fc_384b_e067),
        f64::from_bits(0xc012_8be3_6568_a154),
        f64::from_bits(0xc037_c821_72f5_b4c5),
        f64::from_bits(0xc07a_ac60_8b39_8a07),
    ],
    beta: [
        f64::from_bits(0x3f11_11bb_2223_8fa8),
        f64::from_bits(0x3f39_15e0_3fcf_abee),
        f64::from_bits(0x3f54_c52e_a329_50ed),
        f64::from_bits(0x3f6d_5a48_f0b5_e094),
        f64::from_bits(0x3f83_a288_eaf9_7fd0),
        f64::from_bits(0x3f99_bdf2_e857_f34c),
        f64::from_bits(0x3fb0_bf5d_1bf5_e8cb),
        f64::from_bits(0x3fc5_bcde_cbf9_7d67),
        f64::from_bits(0x3fdc_3814_c858_717c),
        f64::from_bits(0x3ff2_5ea3_2e04_3bde),
        f64::from_bits(0x4008_1f17_a82d_03bc),
        f64::from_bits(0x4020_343c_3b07_6cac),
        f64::from_bits(0x4037_2b39_fd11_146a),
        f64::from_bits(0x4053_e305_5263_d598),
        f64::from_bits(0x4081_3255_449c_50e7),
    ],
};

// These are from `coeffs_m18`, the private Julia x^(-1/8) table.  It is
// deliberately a different role from REFRESH_COEFFICIENTS.
const ACTION_INVERSE_COEFFICIENTS: ActionInverseCoefficients = ActionInverseCoefficients {
    alpha0: f64::from_bits(0x3fd8_6719_edfe_5877),
    alpha: [
        f64::from_bits(0x3f0d_e885_afab_aaff),
        f64::from_bits(0x3f23_f082_95f8_a13b),
        f64::from_bits(0x3f36_e9b8_40da_de4f),
        f64::from_bits(0x3f4a_251b_4b0c_fadb),
        f64::from_bits(0x3f5d_efb3_ec8e_32ce),
        f64::from_bits(0x3f71_2dd3_e4b3_9bb4),
        f64::from_bits(0x3f83_be08_01e4_2c2b),
        f64::from_bits(0x3f96_b74e_0ab8_40a5),
        f64::from_bits(0x3faa_32ac_679f_92e1),
        f64::from_bits(0x3fbe_623f_1e06_520a),
        f64::from_bits(0x3fd1_e027_709a_7169),
        f64::from_bits(0x3fe5_d6f8_d33f_0a0f),
        f64::from_bits(0x3ffd_7854_ba16_b72e),
        f64::from_bits(0x401a_2519_179c_baee),
        f64::from_bits(0x404c_740c_224d_7811),
    ],
    beta: [
        f64::from_bits(0x3f08_63ea_130d_709e),
        f64::from_bits(0x3f35_1744_dcd6_86d6),
        f64::from_bits(0x3f52_1a6b_57c6_2d77),
        f64::from_bits(0x3f69_e261_bd08_9721),
        f64::from_bits(0x3f81_6365_c035_fed8),
        f64::from_bits(0x3f96_d52f_a194_4224),
        f64::from_bits(0x3fad_ba0b_779a_1dba),
        f64::from_bits(0x3fc3_4b85_3cc2_05ac),
        f64::from_bits(0x3fd9_0b59_aec8_70f6),
        f64::from_bits(0x3ff0_4b2b_46a3_b15f),
        f64::from_bits(0x4005_5c89_6b56_afaa),
        f64::from_bits(0x401c_9424_8d41_5da9),
        f64::from_bits(0x4034_319d_ba05_7100),
        f64::from_bits(0x4050_b856_2aa3_47c5),
        f64::from_bits(0x4078_927f_e090_29f9),
    ],
};

// These are from `coeffs_m14_n10`, the private Julia x^(-1/4) MD-force table.
const MD_FORCE_INVERSE_COEFFICIENTS: MdForceInverseCoefficients = MdForceInverseCoefficients {
    alpha0: f64::from_bits(0x3fc6_641f_7427_6577),
    alpha: [
        f64::from_bits(0x3f45_9f19_959f_2361),
        f64::from_bits(0x3f5e_1471_4e09_8a43),
        f64::from_bits(0x3f74_e670_4a79_22c3),
        f64::from_bits(0x3f8d_aeae_fcf4_ff28),
        f64::from_bits(0x3fa5_3a4a_f954_da27),
        f64::from_bits(0x3fbe_7a4a_e736_58f4),
        f64::from_bits(0x3fd6_134f_6a0b_6b2c),
        f64::from_bits(0x3ff0_8e86_d7ef_479b),
        f64::from_bits(0x400c_ae87_739b_6fbc),
        f64::from_bits(0x4037_be3a_0510_06d9),
    ],
    beta: [
        f64::from_bits(0x3f16_9534_cf32_fe7c),
        f64::from_bits(0x3f4a_33b2_394e_3705),
        f64::from_bits(0x3f6f_51dc_e007_c4cd),
        f64::from_bits(0x3f90_8be2_349d_f265),
        f64::from_bits(0x3fb0_fed1_9a78_e3fe),
        f64::from_bits(0x3fd1_5b53_7136_4d8d),
        f64::from_bits(0x3ff1_c839_f784_ffaa),
        f64::from_bits(0x4012_941c_323f_f108),
        f64::from_bits(0x4035_20be_7d02_873e),
        f64::from_bits(0x4062_326c_230c_3e42),
    ],
};

pub(crate) struct RationalApplyResult {
    pub(crate) field: FermionField,
    pub(crate) reports: Vec<MultiShiftSolverReport>,
}

pub(crate) fn apply_refresh<O: HermitianPositiveOperator>(
    operator: &O,
    rhs: &FermionField,
    params: SolverParams,
) -> Result<RationalApplyResult, DiracError> {
    apply_rational(
        operator,
        rhs,
        params,
        REFRESH_COEFFICIENTS.alpha0,
        &REFRESH_COEFFICIENTS.alpha,
        &REFRESH_COEFFICIENTS.beta,
    )
}

pub(crate) fn apply_action_inverse<O: HermitianPositiveOperator>(
    operator: &O,
    rhs: &FermionField,
    params: SolverParams,
) -> Result<RationalApplyResult, DiracError> {
    apply_rational(
        operator,
        rhs,
        params,
        ACTION_INVERSE_COEFFICIENTS.alpha0,
        &ACTION_INVERSE_COEFFICIENTS.alpha,
        &ACTION_INVERSE_COEFFICIENTS.beta,
    )
}

pub(crate) fn solve_md_force_shifts<O: HermitianPositiveOperator>(
    operator: &O,
    rhs: &FermionField,
    params: SolverParams,
) -> Result<(Vec<FermionField>, Vec<MultiShiftSolverReport>), DiracError> {
    if !MD_FORCE_INVERSE_COEFFICIENTS.alpha0.is_finite() {
        return Err(DiracError::NumericalRange);
    }
    solve_shifts(operator, rhs, params, &MD_FORCE_INVERSE_COEFFICIENTS.beta)
}

pub(crate) fn md_force_coefficients() -> &'static [f64; RANK_TWO] {
    &MD_FORCE_INVERSE_COEFFICIENTS.alpha
}

fn apply_rational<O: HermitianPositiveOperator>(
    operator: &O,
    rhs: &FermionField,
    params: SolverParams,
    alpha0: f64,
    alpha: &[f64],
    beta: &[f64],
) -> Result<RationalApplyResult, DiracError> {
    if alpha.len() != beta.len() || !alpha0.is_finite() {
        return Err(DiracError::NumericalRange);
    }
    let (shifted, reports) = solve_shifts(operator, rhs, params, beta)?;
    let mut field = FermionField::zeros(rhs.lattice(), rhs.components())?;
    field.add_scaled(Complex64::new(alpha0, 0.0), rhs)?;
    for (coefficient, solution) in alpha.iter().zip(&shifted) {
        if !coefficient.is_finite() {
            return Err(DiracError::NumericalRange);
        }
        field.add_scaled(Complex64::new(*coefficient, 0.0), solution)?;
    }
    Ok(RationalApplyResult { field, reports })
}

fn solve_shifts<O: HermitianPositiveOperator>(
    operator: &O,
    rhs: &FermionField,
    params: SolverParams,
    beta: &[f64],
) -> Result<(Vec<FermionField>, Vec<MultiShiftSolverReport>), DiracError> {
    let mut shifted = Vec::with_capacity(beta.len());
    for _ in beta {
        shifted.push(FermionField::zeros(rhs.lattice(), rhs.components())?);
    }
    let reports = multi_shift_cg(&mut shifted, operator, rhs, beta, params)?;
    Ok((shifted, reports))
}

/// Validated parameters for a two-flavor staggered RHMC update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaggeredHmcParams {
    gauge: GaugeHmcParams,
    action: StaggeredFermiAction,
}

impl StaggeredHmcParams {
    /// Construct beta, mass, spectral assertion, boundary, and U-P-U values.
    ///
    /// `lambda_low` and `lambda_high` are caller assertions for the spectrum of
    /// `M=D†D`.  They must be finite, positive, ordered, and contained in the
    /// pinned `[0.0004, 64]` coefficient interval.  No spectral estimate is
    /// performed by this constructor.
    ///
    /// # Errors
    ///
    /// Returns typed gauge HMC, mass, spectral-bound, boundary, or solver
    /// validation errors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        beta: f64,
        mass: f64,
        step_size: f64,
        steps: usize,
        boundary: FermionBoundary,
        lambda_low: f64,
        lambda_high: f64,
        solver_params: SolverParams,
    ) -> Result<Self, DiracError> {
        Ok(Self {
            gauge: GaugeHmcParams::new(beta, step_size, steps)?,
            action: StaggeredFermiAction::new(
                mass,
                boundary,
                lambda_low,
                lambda_high,
                solver_params,
            )?,
        })
    }

    /// Return the gauge beta.
    pub const fn beta(self) -> f64 {
        self.gauge.beta()
    }

    /// Return the staggered mass.
    pub const fn mass(self) -> f64 {
        self.action.mass()
    }

    /// Return the positive U-P-U step size.
    pub const fn step_size(self) -> f64 {
        self.gauge.step_size()
    }

    /// Return the number of U-P-U steps.
    pub const fn steps(self) -> usize {
        self.gauge.steps()
    }

    /// Return the validated two-flavor action parameters.
    pub const fn action(self) -> StaggeredFermiAction {
        self.action
    }
}

/// Scalar diagnostics for one two-flavor staggered RHMC proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaggeredHmcOutcome {
    /// Whether the private link proposal replaced the caller's links.
    pub accepted: bool,
    /// Initial gauge, kinetic, and pseudofermion Hamiltonian.
    pub initial_hamiltonian: f64,
    /// Proposed Hamiltonian after the private trajectory.
    pub proposed_hamiltonian: f64,
    /// `proposed_hamiltonian - initial_hamiltonian`.
    pub delta_h: f64,
    /// Branch-stable `min(1, exp(-delta_h))` probability.
    pub acceptance_probability: f64,
}

/// Evolve a supplied staggered pseudofermion with the Julia-parallel U-P-U path.
///
/// The order is `U_update!`, gauge `P_update!`, fermion `P_update_fermion!`,
/// then `U_update!`. Gauge momentum receives `-dt/NC` times the existing gauge
/// force; the staggered force receives `-dt` and has no additional `1/NC`.
/// Links and momentum are copied and committed only after every step succeeds.
///
/// # Errors
///
/// Returns typed lattice, field, gauge, evolution, solver, or numerical-range
/// errors. Both caller-owned fields remain unchanged on every error.
pub fn staggered_leapfrog_trajectory(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    phi: &FermionField,
    params: StaggeredHmcParams,
) -> Result<(), DiracError> {
    validate_state(links, momentum, phi, params.action)?;
    let mut proposed_links = links.try_clone()?;
    let mut proposed_momentum = crate::wilson_hmc::clone_momentum(momentum)?;
    leapfrog_steps(
        context,
        &mut proposed_links,
        &mut proposed_momentum,
        phi,
        params,
    )?;
    *links = proposed_links;
    *momentum = proposed_momentum;
    Ok(())
}

/// Sample momentum and staggered pseudofermions, run one private RHMC
/// trajectory, consume one unconditional Metropolis draw, and commit links on
/// acceptance.
///
/// The two-flavor pseudofermion sampler consumes explicit
/// `ReproducibleRng` Box--Muller pairs with `1/sqrt(2)` applied independently to
/// the real and imaginary parts. RNG advancement is never rolled back. Link
/// storage is unchanged on rejection and on every error.
///
/// # Errors
///
/// Returns typed gauge, field, spectral-bound, allocation, solver, evolution,
/// or numerical-range errors. Caller-owned links remain unchanged on rejection
/// and on every error.
pub fn staggered_hmc_update(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    params: StaggeredHmcParams,
    rng: &mut ReproducibleRng,
) -> Result<StaggeredHmcOutcome, DiracError> {
    require_su3(links)?;
    let _ = StaggeredDirac::with_boundary(links, params.mass(), params.action().boundary())?;

    let momentum = sample_momentum(links.lattice(), rng)?;
    let phi = params.action().sample_pseudofermion(links, rng)?;
    validate_state(links, &momentum, &phi, params.action)?;

    let initial_action = params.action().evaluate(links, &phi)?.action;
    let initial_hamiltonian = hamiltonian(links, &momentum, params.beta(), initial_action)?;

    let mut proposed_links = links.try_clone()?;
    let mut proposed_momentum = crate::wilson_hmc::clone_momentum(&momentum)?;
    leapfrog_steps(
        context,
        &mut proposed_links,
        &mut proposed_momentum,
        &phi,
        params,
    )?;
    let proposed_action = params.action().evaluate(&proposed_links, &phi)?.action;
    let proposed_hamiltonian = hamiltonian(
        &proposed_links,
        &proposed_momentum,
        params.beta(),
        proposed_action,
    )?;
    let delta_h = proposed_hamiltonian - initial_hamiltonian;
    if !delta_h.is_finite() {
        return Err(gaugefields::GaugeError::NonFiniteHamiltonianDelta.into());
    }
    let acceptance_probability = if delta_h <= 0.0 {
        1.0
    } else {
        (-delta_h).exp()
    };
    let draw = rng.open_unit_f64();
    let accepted = draw <= acceptance_probability;
    if accepted {
        *links = proposed_links;
    }
    Ok(StaggeredHmcOutcome {
        accepted,
        initial_hamiltonian,
        proposed_hamiltonian,
        delta_h,
        acceptance_probability,
    })
}

fn hamiltonian(
    links: &GaugeLinks,
    momentum: &TaGaugeField,
    beta: f64,
    pseudofermion_action: f64,
) -> Result<f64, DiracError> {
    let value = wilson_action(links, beta)? + kinetic_energy(momentum)? + pseudofermion_action;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(gaugefields::GaugeError::NonFiniteHamiltonian.into())
    }
}

fn validate_state(
    links: &GaugeLinks,
    momentum: &TaGaugeField,
    phi: &FermionField,
    action: StaggeredFermiAction,
) -> Result<(), DiracError> {
    require_su3(links)?;
    if links.lattice() != momentum.lattice() {
        return Err(gaugefields::GaugeError::Shape {
            expected: links.lattice().extents().to_vec(),
            found: momentum.lattice().extents().to_vec(),
        }
        .into());
    }
    if phi.lattice() != links.lattice() {
        return Err(DiracError::LatticeMismatch {
            operand: "phi",
            expected: links.lattice(),
            found: phi.lattice(),
        });
    }
    if phi.components() != 1 {
        return Err(DiracError::ComponentsMismatch {
            operand: "phi",
            expected: 1,
            found: phi.components(),
        });
    }
    phi.ensure_finite()?;
    let _ = kinetic_energy(momentum)?;
    let _ = StaggeredDirac::with_boundary(links, action.mass(), action.boundary())?;
    Ok(())
}

fn leapfrog_steps(
    context: &mut CpuEvolutionContext,
    links: &mut GaugeLinks,
    momentum: &mut TaGaugeField,
    phi: &FermionField,
    params: StaggeredHmcParams,
) -> Result<(), DiracError> {
    let dt = params.step_size();
    let half_step = 0.5 * dt;
    let gauge_factor = -dt / links.nc() as f64;
    let fermion_factor = -dt;
    for _ in 0..params.steps() {
        exp_ta_update(context, links, half_step, momentum)?;
        let gauge = gauge_force(links, params.beta())?;
        *momentum = crate::wilson_hmc::add_scaled(momentum, &gauge, gauge_factor)?;
        let fermion = params.action().force(links, phi)?;
        *momentum = crate::wilson_hmc::add_scaled(momentum, &fermion.force, fermion_factor)?;
        exp_ta_update(context, links, half_step, momentum)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
