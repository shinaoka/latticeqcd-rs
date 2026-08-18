use crate::{
    action::{CompiledTerm, LoopAction},
    path::decode_step,
    WilsonError, WilsonPath,
};
use gaugefields::{GaugeLinks, HostGaugeLinks, Mat3, TaGaugeField};
use num_complex::Complex64;

/// Evaluates one path product from a checked lattice origin.
///
/// Forward `+(mu + 1)` loads `U_mu(x)` and then moves to `x + mu`.
/// Backward `-(mu + 1)` first moves to `x - mu` and loads the adjoint link at
/// that site. All moves use the field's periodic host view.
///
/// # Errors
///
/// Returns a typed origin, placement, gauge-field, or matrix-boundary error.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, LatticeShape4};
/// use wilsonloop::{evaluate_path, WilsonPath};
///
/// let links = cold_su3(LatticeShape4::new([2, 2, 2, 2])?)?;
/// let value = evaluate_path(&links, 0, &WilsonPath::plaquette(1, 2)?)?;
/// assert_eq!(value, gaugefields::Mat3::identity());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn evaluate_path(
    links: &GaugeLinks,
    origin: usize,
    path: &WilsonPath,
) -> Result<Mat3, WilsonError> {
    let view = links.host_view()?;
    check_origin(&view, origin)?;
    evaluate_path_view(&view, origin, path)
}

/// Sums the complex trace of a path over every lattice origin.
///
/// Open paths are valid here; the result is not normalized and no real-part
/// projection is applied.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, LatticeShape4};
/// use wilsonloop::{loop_trace_sum, WilsonPath};
///
/// let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
/// let sum = loop_trace_sum(&links, &WilsonPath::new(vec![1])?)?;
/// assert_eq!(sum, num_complex::Complex64::new(3.0, 0.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn loop_trace_sum(links: &GaugeLinks, path: &WilsonPath) -> Result<Complex64, WilsonError> {
    let view = links.host_view()?;
    let mut sum = Complex64::default();
    for origin in 0..view.lattice().nv() {
        sum += evaluate_path_view(&view, origin, path)?.trace();
    }
    Ok(sum)
}

/// Evaluates the documented unnormalized real loop action.
///
/// The result is exactly
///
/// ```text
/// sum_terms coefficient * sum_x Re tr(path at x)
/// ```
///
/// There is no hidden `1 / NC`, volume, or plaquette-count normalization.
/// For a real Julia coefficient `f` that inserts `W` and `W†`, use one Rust
/// term with coefficient `2*f`.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, LatticeShape4};
/// use wilsonloop::{loop_action_value, LoopAction, LoopTerm};
///
/// let links = cold_su3(LatticeShape4::new([2, 2, 2, 2])?)?;
/// let action = LoopAction::new(vec![LoopTerm::plaquette(0.5, 1, 2)?])?;
/// assert_eq!(loop_action_value(&links, &action)?, 0.5 * 3.0 * 16.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn loop_action_value(links: &GaugeLinks, action: &LoopAction) -> Result<f64, WilsonError> {
    let view = links.host_view()?;
    let mut value = 0.0;
    for term in action.terms() {
        for origin in 0..view.lattice().nv() {
            value +=
                term.coefficient() * evaluate_path_view(&view, origin, term.path())?.trace().re;
        }
    }
    Ok(value)
}

/// Computes the positive loop-action force in `TaGaugeField` coefficients.
///
/// For a forward occurrence the contribution is
/// `coefficient / 2 * TA(U * after * before)`. For a backward occurrence it is
/// `-coefficient / 2 * TA(after * before * U†)`. The factor `1/2` is the
/// established mapping from one Rust `coefficient * Re tr(W)` term to the
/// Julia pair `f * W + f * W†`, where `coefficient = 2*f`; the returned field
/// is the positive `U * calc_dSdU` convention. Site loops allocate no heap;
/// path offsets and occurrence tables are compiled by `LoopAction::new`.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, LatticeShape4};
/// use wilsonloop::{loop_action_force, LoopAction, LoopTerm};
///
/// let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
/// let action = LoopAction::new(vec![LoopTerm::plaquette(1.0, 1, 2)?])?;
/// let force = loop_action_force(&links, &action)?;
/// assert_eq!(force.lattice(), links.lattice());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn loop_action_force(
    links: &GaugeLinks,
    action: &LoopAction,
) -> Result<TaGaugeField, WilsonError> {
    let view = links.host_view()?;
    let mut force = TaGaugeField::zeros(view.lattice())?;
    for (term, compiled) in action.terms().iter().zip(action.compiled.iter()) {
        accumulate_term_force(&view, term, compiled, &mut force)?;
    }
    Ok(force)
}

fn accumulate_term_force(
    view: &HostGaugeLinks<'_>,
    term: &crate::LoopTerm,
    compiled: &CompiledTerm,
    force: &mut TaGaugeField,
) -> Result<(), WilsonError> {
    let step_count = term.path().steps().len();
    let product_count = step_count
        .checked_add(1)
        .ok_or(WilsonError::AllocationOverflow)?;
    let mut factors = Vec::new();
    factors
        .try_reserve_exact(step_count)
        .map_err(|_| WilsonError::AllocationOverflow)?;
    factors.resize(step_count, Mat3::identity());
    let mut prefixes = Vec::new();
    prefixes
        .try_reserve_exact(product_count)
        .map_err(|_| WilsonError::AllocationOverflow)?;
    prefixes.resize(product_count, Mat3::identity());
    let mut suffixes = Vec::new();
    suffixes
        .try_reserve_exact(product_count)
        .map_err(|_| WilsonError::AllocationOverflow)?;
    suffixes.resize(product_count, Mat3::identity());

    for origin in 0..view.lattice().nv() {
        let mut site = origin;
        for (&step, factor) in term.path().steps().iter().zip(factors.iter_mut()) {
            *factor = load_step(view, &mut site, step)?;
        }
        let mut running = Mat3::identity();
        for (prefix, &factor) in prefixes.iter_mut().skip(1).zip(&factors) {
            running = running.mul(factor);
            *prefix = running;
        }
        running = Mat3::identity();
        for (suffix, &factor) in suffixes[..step_count]
            .iter_mut()
            .rev()
            .zip(factors.iter().rev())
        {
            running = factor.mul(running);
            *suffix = running;
        }

        for occurrence in &compiled.occurrences {
            let link_site = shift_by_offset(view, origin, occurrence.link_offset)?;
            let u = view.link(occurrence.direction, link_site)?;
            let before = *prefixes
                .get(occurrence.step_index)
                .ok_or(WilsonError::InvalidCompiledMetadata)?;
            let after = *suffixes
                .get(
                    occurrence
                        .step_index
                        .checked_add(1)
                        .ok_or(WilsonError::InvalidCompiledMetadata)?,
                )
                .ok_or(WilsonError::InvalidCompiledMetadata)?;
            let after_before = after.mul(before);
            let contribution = if occurrence.forward {
                u.mul(after_before)
            } else {
                after_before.mul(u.adjoint())
            };
            let factor = 0.5
                * if occurrence.forward {
                    term.coefficient()
                } else {
                    -term.coefficient()
                };
            let mut coefficients = [0.0; 8];
            Mat3::add_ta_coefficients(&mut coefficients, factor, contribution);
            force.add_site_coefficients(occurrence.direction, link_site, coefficients)?;
        }
    }
    Ok(())
}

fn evaluate_path_view(
    view: &HostGaugeLinks<'_>,
    origin: usize,
    path: &WilsonPath,
) -> Result<Mat3, WilsonError> {
    let mut site = origin;
    let mut product = Mat3::identity();
    for &step in path.steps() {
        product = product.mul(load_step(view, &mut site, step)?);
    }
    Ok(product)
}

fn load_step(view: &HostGaugeLinks<'_>, site: &mut usize, step: i8) -> Result<Mat3, WilsonError> {
    let (direction, forward) = decode_step(step).ok_or(WilsonError::InvalidStep { step })?;
    if forward {
        let link = view.link(direction, *site)?;
        *site = view.shifted_site(*site, direction, 1)?;
        Ok(link)
    } else {
        let base = view.shifted_site(*site, direction, -1)?;
        let link = view.link(direction, base)?;
        *site = base;
        Ok(link.adjoint())
    }
}

fn shift_by_offset(
    view: &HostGaugeLinks<'_>,
    origin: usize,
    offset: [isize; 4],
) -> Result<usize, WilsonError> {
    let mut site = origin;
    for (direction, &displacement) in offset.iter().enumerate() {
        if displacement != 0 {
            site = view.shifted_site(site, direction, displacement)?;
        }
    }
    Ok(site)
}

fn check_origin(view: &HostGaugeLinks<'_>, origin: usize) -> Result<(), WilsonError> {
    let volume = view.lattice().nv();
    if origin >= volume {
        return Err(WilsonError::OriginOutOfBounds { origin, volume });
    }
    Ok(())
}
