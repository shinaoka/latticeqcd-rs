//! Host-only SU(3) Wilson heatbath updates.
//!
//! The sweep order and Kennedy--Pendleton convention follow Gaugefields.jl
//! v0.7.2 at commit `9e5719970770f4497405a856315c90bef7f74449`,
//! `src/heatbath/heatbathmodule.jl`. The fixed SU(3) subgroup order and
//! direction/even-odd traversal are retained, while the rejection draw order,
//! open uniforms, and square-root SU(2) normalization follow the reviewed
//! compatibility corrections in `docs/design/su3-heatbath.md`.

use crate::field::duplicate_links;
use crate::{
    normalize_su3, require_su3, store_link, GaugeError, GaugeLinks, LatticeShape4, Mat3,
    ReproducibleRng,
};
use num_complex::Complex64 as C;

const SUBGROUPS: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];
const TAU: f64 = std::f64::consts::TAU;

/// Validated parameters for one Wilson-action SU(3) heatbath sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatbathParams {
    beta: f64,
    max_attempts: usize,
}

impl HeatbathParams {
    /// Constructs heatbath parameters.
    ///
    /// # Errors
    ///
    /// Returns [`GaugeError::NonFiniteBeta`] for a non-finite `beta`,
    /// [`GaugeError::NonPositiveHeatbathBeta`] for a non-positive finite
    /// `beta`, or [`GaugeError::ZeroHeatbathAttempts`] for zero
    /// `max_attempts`.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaugefields::{GaugeError, HeatbathParams};
    ///
    /// let params = HeatbathParams::new(5.7, 100_000)?;
    /// assert_eq!(params.beta(), 5.7);
    /// assert_eq!(params.max_attempts(), 100_000);
    /// # Ok::<(), GaugeError>(())
    /// ```
    pub fn new(beta: f64, max_attempts: usize) -> Result<Self, GaugeError> {
        if !beta.is_finite() {
            return Err(GaugeError::NonFiniteBeta { found: beta });
        }
        if beta <= 0.0 {
            return Err(GaugeError::NonPositiveHeatbathBeta { found: beta });
        }
        if max_attempts == 0 {
            return Err(GaugeError::ZeroHeatbathAttempts);
        }
        Ok(Self { beta, max_attempts })
    }

    /// Returns the positive Wilson coupling.
    pub const fn beta(self) -> f64 {
        self.beta
    }

    /// Returns the maximum Kennedy--Pendleton rejection iterations per subgroup.
    pub const fn max_attempts(self) -> usize {
        self.max_attempts
    }
}

/// Counts the link updates and Kennedy--Pendleton iterations in one sweep.
///
/// The attempt count includes the accepting iteration. The type deliberately
/// contains only compact counters and never exposes link or RNG storage.
///
/// # Examples
///
/// ```
/// use gaugefields::HeatbathSweepStats;
///
/// let stats = HeatbathSweepStats {
///     updated_links: 64,
///     su2_attempts: 192,
/// };
/// assert_eq!(stats.su2_attempts / 3, stats.updated_links);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeatbathSweepStats {
    /// Number of links successfully updated.
    pub updated_links: usize,
    /// Total rejection-loop iterations across all three subgroups per link.
    pub su2_attempts: usize,
}

/// Applies one transactional, host-resident SU(3) Wilson heatbath sweep.
///
/// Directions run in order `0..4`; within each direction, even sites precede
/// odd sites, and each parity is visited in ascending compact site order. Each
/// parity is computed from one immutable field snapshot before its updates are
/// stored. The caller's links are replaced only after the complete sweep
/// succeeds. RNG draws already consumed before an error are not rolled back.
///
/// # Errors
///
/// Returns a typed error for invalid parameters, non-SU(3) links, odd extents,
/// non-host storage, allocation range overflow, singular staples, numerical
/// range failures, or rejection-limit exhaustion.
///
/// # Examples
///
/// ```
/// use gaugefields::{
///     cold_su3, heatbath_sweep, HeatbathParams, LatticeShape4, ReproducibleRng,
/// };
///
/// let lattice = LatticeShape4::new([2, 2, 2, 2])?;
/// let mut links = cold_su3(lattice)?;
/// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
/// let stats = heatbath_sweep(
///     &mut links,
///     HeatbathParams::new(5.7, 100_000)?,
///     &mut rng,
/// )?;
/// assert_eq!(stats.updated_links, 4 * lattice.nv());
/// # Ok::<(), gaugefields::GaugeError>(())
/// ```
pub fn heatbath_sweep(
    links: &mut GaugeLinks,
    params: HeatbathParams,
    rng: &mut ReproducibleRng,
) -> Result<HeatbathSweepStats, GaugeError> {
    validate_heatbath_inputs(links, params)?;
    let mut proposed = duplicate_links(links)?;
    let mut draw = || rng.open_unit_f64();
    let mut observe = |_, _, _, _, _| {};
    let stats = heatbath_sweep_core(&mut proposed, params, &mut draw, &mut observe)?;
    *links = proposed;
    Ok(stats)
}

fn heatbath_sweep_core<F, O>(
    links: &mut GaugeLinks,
    params: HeatbathParams,
    draw: &mut F,
    observe: &mut O,
) -> Result<HeatbathSweepStats, GaugeError>
where
    F: FnMut() -> f64,
    O: FnMut(usize, bool, usize, usize, usize),
{
    validate_heatbath_inputs(links, params)?;
    let parity_count = checked_heatbath_sizes(links.lattice())?;
    let mut stats = HeatbathSweepStats {
        updated_links: 0,
        su2_attempts: 0,
    };

    for direction in 0..4 {
        for parity in [true, false] {
            let updates = {
                let prepared = links.host_view()?;
                let mut updates = Vec::with_capacity(parity_count);
                for site in 0..prepared.nv() {
                    if site_is_even(site, prepared.lattice()) != parity {
                        continue;
                    }
                    let current = prepared.link(direction, site)?;
                    let staple = prepared.force_staple(site, direction)?.adjoint();
                    let (updated, attempts) = update_link(
                        current,
                        staple,
                        params,
                        (direction, parity, site),
                        draw,
                        observe,
                    )?;
                    stats.su2_attempts = stats
                        .su2_attempts
                        .checked_add(attempts)
                        .ok_or(GaugeError::AllocationOverflow)?;
                    updates.push((site, updated));
                    stats.updated_links = stats
                        .updated_links
                        .checked_add(1)
                        .ok_or(GaugeError::AllocationOverflow)?;
                }
                updates
            };
            for (site, value) in updates {
                store_link(links, direction, site, value)?;
            }
        }
    }

    Ok(stats)
}

fn validate_heatbath_inputs(links: &GaugeLinks, params: HeatbathParams) -> Result<(), GaugeError> {
    if !params.beta.is_finite() {
        return Err(GaugeError::NonFiniteBeta { found: params.beta });
    }
    if params.beta <= 0.0 {
        return Err(GaugeError::NonPositiveHeatbathBeta { found: params.beta });
    }
    if params.max_attempts == 0 {
        return Err(GaugeError::ZeroHeatbathAttempts);
    }
    require_su3(links)?;
    for (axis, &extent) in links.lattice().extents().iter().enumerate() {
        if extent % 2 != 0 {
            return Err(GaugeError::OddHeatbathExtent { axis, extent });
        }
    }
    checked_heatbath_sizes(links.lattice())?;
    links.host_view()?;
    Ok(())
}

fn checked_heatbath_sizes(lattice: LatticeShape4) -> Result<usize, GaugeError> {
    let values = 9usize
        .checked_mul(lattice.nv())
        .ok_or(GaugeError::AllocationOverflow)?;
    checked_bytes(values, std::mem::size_of::<C>())?;
    let updated_links = 4usize
        .checked_mul(lattice.nv())
        .ok_or(GaugeError::AllocationOverflow)?;
    updated_links
        .checked_mul(SUBGROUPS.len())
        .ok_or(GaugeError::AllocationOverflow)?;
    let parity_count = lattice
        .nv()
        .checked_div(2)
        .ok_or(GaugeError::AllocationOverflow)?;
    checked_bytes(parity_count, std::mem::size_of::<(usize, Mat3)>())?;
    Ok(parity_count)
}

fn checked_bytes(count: usize, element_size: usize) -> Result<(), GaugeError> {
    let bytes = count
        .checked_mul(element_size)
        .ok_or(GaugeError::AllocationOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(GaugeError::AllocationOverflow);
    }
    Ok(())
}

fn site_is_even(site: usize, lattice: LatticeShape4) -> bool {
    let [nx, ny, nz, _] = lattice.extents();
    let x = site % nx;
    let q = site / nx;
    let y = q % ny;
    let q = q / ny;
    let z = q % nz;
    let t = q / nz;
    (x + y + z + t).is_multiple_of(2)
}

fn update_link<F, O>(
    mut current: Mat3,
    staple_v: Mat3,
    params: HeatbathParams,
    (direction, parity, site): (usize, bool, usize),
    draw: &mut F,
    observe: &mut O,
) -> Result<(Mat3, usize), GaugeError>
where
    F: FnMut() -> f64,
    O: FnMut(usize, bool, usize, usize, usize),
{
    if !finite_matrix(&staple_v) {
        return Err(GaugeError::HeatbathNumericalRange { stage: "staple" });
    }
    if !finite_matrix(&current) {
        return Err(GaugeError::HeatbathNumericalRange { stage: "link" });
    }

    let mut attempts_total = 0usize;
    for (subgroup, &(row, column)) in SUBGROUPS.iter().enumerate() {
        let uv = current.mul(staple_v);
        if !finite_matrix(&uv) {
            return Err(GaugeError::HeatbathNumericalRange { stage: "U*V" });
        }
        let block = [
            uv[(row, row)],
            uv[(row, column)],
            uv[(column, row)],
            uv[(column, column)],
        ];
        let projected = project_su2(block);
        let alpha = projected[0];
        let beta = projected[2];
        let norm_squared = alpha.norm_sqr() + beta.norm_sqr();
        if !norm_squared.is_finite() {
            return Err(GaugeError::HeatbathNumericalRange {
                stage: "staple norm",
            });
        }
        let rho = norm_squared.sqrt();
        if !rho.is_finite() {
            return Err(GaugeError::HeatbathNumericalRange {
                stage: "staple norm",
            });
        }
        if rho == 0.0 {
            return Err(GaugeError::SingularHeatbathStaple {
                direction,
                site,
                subgroup,
            });
        }
        let v0 = [
            alpha.conj() / rho,
            beta.conj() / rho,
            -beta / rho,
            alpha / rho,
        ];
        if !finite_su2(&v0) {
            return Err(GaugeError::HeatbathNumericalRange { stage: "V0" });
        }
        let k = 2.0 * (params.beta / 3.0) * rho;
        if !k.is_finite() || k <= 0.0 {
            return Err(GaugeError::HeatbathNumericalRange {
                stage: "Kennedy-Pendleton k",
            });
        }
        let (sampled, attempts) = sample_kp(k, params.max_attempts, v0, draw)?;
        attempts_total = attempts_total
            .checked_add(attempts)
            .ok_or(GaugeError::AllocationOverflow)?;
        let mut embedded = Mat3::identity();
        embedded[(row, row)] = sampled[0];
        embedded[(row, column)] = sampled[1];
        embedded[(column, row)] = sampled[2];
        embedded[(column, column)] = sampled[3];
        current = embedded.mul(current);
        if !finite_matrix(&current) {
            return Err(GaugeError::HeatbathNumericalRange {
                stage: "subgroup update",
            });
        }
        observe(direction, parity, site, subgroup, attempts);
    }

    normalize_su3(&mut current).map_err(|error| match error {
        GaugeError::NonFiniteSu3Input { .. }
        | GaugeError::Su3NumericalRange { .. }
        | GaugeError::SingularSu3Normalization { .. } => GaugeError::HeatbathNumericalRange {
            stage: "SU(3) normalization",
        },
        other => other,
    })?;
    if !finite_matrix(&current) {
        return Err(GaugeError::HeatbathNumericalRange {
            stage: "SU(3) normalization",
        });
    }
    Ok((current, attempts_total))
}

fn sample_kp<F>(
    k: f64,
    max_attempts: usize,
    v0: [C; 4],
    draw: &mut F,
) -> Result<([C; 4], usize), GaugeError>
where
    F: FnMut() -> f64,
{
    let mut attempt = 0usize;
    loop {
        if attempt == max_attempts {
            return Err(GaugeError::HeatbathRejectionLimit { max_attempts });
        }
        attempt = attempt
            .checked_add(1)
            .ok_or(GaugeError::AllocationOverflow)?;
        let u1 = checked_uniform(draw)?;
        let u2 = checked_uniform(draw)?;
        let u3 = checked_uniform(draw)?;
        let u4 = checked_uniform(draw)?;
        let x = -u1.ln() / k;
        let x_prime = -u2.ln() / k;
        let cosine = (TAU * u3).cos();
        let delta = x_prime + x * cosine * cosine;
        let threshold = 1.0 - delta / 2.0;
        let lhs = u4 * u4;
        if !x.is_finite()
            || !x_prime.is_finite()
            || !cosine.is_finite()
            || !delta.is_finite()
            || !threshold.is_finite()
            || !lhs.is_finite()
        {
            return Err(GaugeError::HeatbathNumericalRange {
                stage: "Kennedy-Pendleton rejection",
            });
        }
        if lhs <= threshold {
            let a0 = 1.0 - delta;
            let radial_squared = 1.0 - a0 * a0;
            if !a0.is_finite() || !radial_squared.is_finite() || radial_squared < 0.0 {
                return Err(GaugeError::HeatbathNumericalRange {
                    stage: "Kennedy-Pendleton quaternion",
                });
            }
            let radial = radial_squared.sqrt();
            let u5 = checked_uniform(draw)?;
            let u6 = checked_uniform(draw)?;
            let phi = TAU * u5;
            let cos_theta = 2.0 * u6 - 1.0;
            let sin_squared = 1.0 - cos_theta * cos_theta;
            if !radial.is_finite()
                || !phi.is_finite()
                || !cos_theta.is_finite()
                || !sin_squared.is_finite()
                || sin_squared < 0.0
            {
                return Err(GaugeError::HeatbathNumericalRange {
                    stage: "Kennedy-Pendleton quaternion",
                });
            }
            let sin_theta = sin_squared.sqrt();
            let a1 = radial * phi.cos() * sin_theta;
            let a2 = radial * phi.sin() * sin_theta;
            let a3 = radial * cos_theta;
            let k_raw = [
                C::new(a0, a3),
                C::new(a2, a1),
                C::new(-a2, a1),
                C::new(a0, -a3),
            ];
            if !finite_su2(&k_raw) {
                return Err(GaugeError::HeatbathNumericalRange {
                    stage: "Kennedy-Pendleton quaternion",
                });
            }
            let kv0 = mul_su2(k_raw, v0);
            let normalized = project_and_normalize_su2(kv0)?;
            return Ok((normalized, attempt));
        }
    }
}

fn checked_uniform<F>(draw: &mut F) -> Result<f64, GaugeError>
where
    F: FnMut() -> f64,
{
    let value = draw();
    if value.is_finite() && value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err(GaugeError::HeatbathNumericalRange {
            stage: "uniform draw",
        })
    }
}

fn project_su2(block: [C; 4]) -> [C; 4] {
    let alpha = (block[0] + block[3].conj()) * 0.5;
    let beta = (block[2] - block[1].conj()) * 0.5;
    [alpha, -beta.conj(), beta, alpha.conj()]
}

fn project_and_normalize_su2(block: [C; 4]) -> Result<[C; 4], GaugeError> {
    let projected = project_su2(block);
    let alpha = projected[0];
    let beta = projected[2];
    let norm_squared = alpha.norm_sqr() + beta.norm_sqr();
    if !norm_squared.is_finite() {
        return Err(GaugeError::HeatbathNumericalRange {
            stage: "SU(2) projection",
        });
    }
    let norm = norm_squared.sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(GaugeError::HeatbathNumericalRange {
            stage: "SU(2) projection",
        });
    }
    let normalized = [
        alpha / norm,
        -beta.conj() / norm,
        beta / norm,
        alpha.conj() / norm,
    ];
    if finite_su2(&normalized) {
        Ok(normalized)
    } else {
        Err(GaugeError::HeatbathNumericalRange {
            stage: "SU(2) projection",
        })
    }
}

fn mul_su2(a: [C; 4], b: [C; 4]) -> [C; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

fn finite_su2(matrix: &[C; 4]) -> bool {
    matrix
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

fn finite_matrix(matrix: &Mat3) -> bool {
    matrix
        .as_array()
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

#[cfg(test)]
mod tests;
