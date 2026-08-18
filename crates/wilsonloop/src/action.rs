use crate::{
    path::{checked_step_displacement, decode_step},
    WilsonError, WilsonPath,
};
use std::fmt;

/// One finite real coefficient multiplying one closed Wilson path.
///
/// The term means exactly
///
/// ```text
/// coefficient * sum_x Re tr(path at x)
/// ```
///
/// A Julia `Gradientflow_general` real coefficient `f` inserts both `f * W`
/// and `f * W†`; represent that pair here as one term with `coefficient = 2*f`.
#[derive(Clone, Debug)]
pub struct LoopTerm {
    coefficient: f64,
    path: WilsonPath,
}

impl LoopTerm {
    /// Creates a finite-coefficient term from a closed path.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite coefficient or an open path.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::{LoopTerm, WilsonPath};
    ///
    /// let term = LoopTerm::new(0.5, WilsonPath::plaquette(1, 2)?)?;
    /// assert_eq!(term.coefficient(), 0.5);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn new(coefficient: f64, path: WilsonPath) -> Result<Self, WilsonError> {
        if !coefficient.is_finite() {
            return Err(WilsonError::NonFiniteCoefficient { coefficient });
        }
        if !path.is_closed() {
            return Err(WilsonError::OpenPath {
                displacement: path.displacement(),
            });
        }
        Ok(Self { coefficient, path })
    }

    /// Creates a positive oriented plaquette term.
    ///
    /// The axes are one-based (`1..=4`) and distinct. The coefficient has the
    /// exact `coefficient * sum_x Re tr(W)` meaning documented on `LoopTerm`.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::LoopTerm;
    ///
    /// let term = LoopTerm::plaquette(1.0, 1, 2)?;
    /// assert_eq!(term.path().steps(), &[1, 2, -1, -2]);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn plaquette(coefficient: f64, mu: usize, nu: usize) -> Result<Self, WilsonError> {
        Self::new(coefficient, WilsonPath::plaquette(mu, nu)?)
    }

    /// Creates both positive 1x2 rectangle terms in Gaugefields.jl order.
    ///
    /// The first returned path has its `nu` side doubled and the second has
    /// its `mu` side doubled. Both terms use the supplied coefficient.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::LoopTerm;
    ///
    /// let [nu_long, mu_long] = LoopTerm::rectangle_1x2(0.25, 1, 2)?;
    /// assert_eq!(nu_long.path().steps(), &[1, 2, 2, -1, -2, -2]);
    /// assert_eq!(mu_long.path().steps(), &[1, 1, 2, -1, -1, -2]);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn rectangle_1x2(coefficient: f64, mu: usize, nu: usize) -> Result<[Self; 2], WilsonError> {
        let [nu_long, mu_long] = WilsonPath::rectangle_1x2(mu, nu)?;
        Ok([
            Self::new(coefficient, nu_long)?,
            Self::new(coefficient, mu_long)?,
        ])
    }

    /// Returns the real coefficient.
    pub const fn coefficient(&self) -> f64 {
        self.coefficient
    }

    /// Returns the validated closed path.
    pub fn path(&self) -> &WilsonPath {
        &self.path
    }
}

/// A nonempty collection of closed loop terms with precompiled force metadata.
#[derive(Clone)]
pub struct LoopAction {
    terms: Box<[LoopTerm]>,
    pub(crate) compiled: Box<[CompiledTerm]>,
}

impl fmt::Debug for LoopAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoopAction")
            .field("terms", &self.terms)
            .finish()
    }
}

impl LoopAction {
    /// Validates and compiles a nonempty loop action.
    ///
    /// The path offsets and link-occurrence table are built here, not in a
    /// lattice-site loop. Construction accepts the standard `Vec<LoopTerm>`
    /// produced by callers assembling an action.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty action, invalid metadata, or a metadata
    /// allocation overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::{LoopAction, LoopTerm};
    ///
    /// let action = LoopAction::new(vec![LoopTerm::plaquette(1.0, 1, 2)?])?;
    /// assert_eq!(action.terms().len(), 1);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn new(terms: impl Into<Vec<LoopTerm>>) -> Result<Self, WilsonError> {
        let terms = terms.into();
        if terms.is_empty() {
            return Err(WilsonError::EmptyAction);
        }
        let mut compiled = Vec::new();
        compiled
            .try_reserve_exact(terms.len())
            .map_err(|_| WilsonError::AllocationOverflow)?;
        for term in &terms {
            compiled.push(compile_term(term)?);
        }
        Ok(Self {
            terms: terms.into_boxed_slice(),
            compiled: compiled.into_boxed_slice(),
        })
    }

    /// Returns the validated terms in construction order.
    pub fn terms(&self) -> &[LoopTerm] {
        &self.terms
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Occurrence {
    pub(crate) step_index: usize,
    pub(crate) direction: usize,
    pub(crate) forward: bool,
    pub(crate) link_offset: [isize; 4],
}

#[derive(Clone)]
pub(crate) struct CompiledTerm {
    pub(crate) occurrences: Box<[Occurrence]>,
}

fn compile_term(term: &LoopTerm) -> Result<CompiledTerm, WilsonError> {
    let steps = term.path.steps();
    let mut occurrences = Vec::new();
    occurrences
        .try_reserve_exact(steps.len())
        .map_err(|_| WilsonError::AllocationOverflow)?;
    let mut cursor = [0isize; 4];
    for (step_index, &step) in steps.iter().enumerate() {
        let (direction, forward) = decode_step(step).ok_or(WilsonError::InvalidStep { step })?;
        let after_offset = checked_step_displacement(cursor, step)?;
        let link_offset = if forward { cursor } else { after_offset };
        occurrences.push(Occurrence {
            step_index,
            direction,
            forward,
            link_offset,
        });
        cursor = after_offset;
    }
    Ok(CompiledTerm {
        occurrences: occurrences.into_boxed_slice(),
    })
}
