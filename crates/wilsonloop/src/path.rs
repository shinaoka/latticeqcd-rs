use crate::WilsonError;
use std::fmt;

/// A nonempty sequence of signed unit lattice directions.
///
/// Positive step `+(mu + 1)` multiplies `U_mu` before moving forward; negative
/// step `-(mu + 1)` first moves backward and multiplies `U_mu†` there. The
/// directions are one-based to match Wilsonloop.jl and are always periodic.
#[derive(Clone, Eq, PartialEq)]
pub struct WilsonPath {
    steps: Vec<i8>,
    displacement: [isize; 4],
}

impl fmt::Debug for WilsonPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WilsonPath")
            .field("steps", &self.steps)
            .field("displacement", &self.displacement)
            .finish()
    }
}

impl WilsonPath {
    /// Validates and stores signed unit steps.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty path, an invalid direction, or a checked
    /// displacement overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::WilsonPath;
    ///
    /// let path = WilsonPath::new(vec![1, 2, -1, -2])?;
    /// assert!(path.is_closed());
    /// assert_eq!(path.displacement(), [0, 0, 0, 0]);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn new(steps: impl Into<Vec<i8>>) -> Result<Self, WilsonError> {
        let steps = steps.into();
        if steps.is_empty() {
            return Err(WilsonError::EmptyPath);
        }
        let mut displacement = [0isize; 4];
        for &step in &steps {
            displacement = checked_step_displacement(displacement, step)?;
        }
        Ok(Self {
            steps,
            displacement,
        })
    }

    /// Returns the validated signed unit steps.
    pub fn steps(&self) -> &[i8] {
        &self.steps
    }

    /// Returns the exact checked displacement in `[x, y, z, t]` order.
    pub const fn displacement(&self) -> [isize; 4] {
        self.displacement
    }

    /// Returns whether the path returns to its starting site.
    pub fn is_closed(&self) -> bool {
        self.displacement == [0; 4]
    }

    /// Returns the reversed path with every orientation flipped.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::WilsonPath;
    ///
    /// let path = WilsonPath::new(vec![1, 2, -1, -2])?;
    /// assert_eq!(path.adjoint().adjoint(), path);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn adjoint(&self) -> Self {
        let steps = self.steps.iter().rev().map(|&step| -step).collect();
        Self {
            steps,
            displacement: self.displacement.map(|value| -value),
        }
    }

    /// Constructs the positive oriented plaquette `+mu,+nu,-mu,-nu`.
    ///
    /// Axes are one-based (`1..=4`) and must be distinct.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::WilsonPath;
    ///
    /// let path = WilsonPath::plaquette(1, 2)?;
    /// assert_eq!(path.steps(), &[1, 2, -1, -2]);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn plaquette(mu: usize, nu: usize) -> Result<Self, WilsonError> {
        let (mu, nu) = distinct_axes(mu, nu)?;
        Self::new(vec![mu, nu, -mu, -nu])
    }

    /// Constructs the two positive 1x2 rectangle orientations.
    ///
    /// The first path is `mu,nu,nu,-mu,-nu,-nu`; the second is
    /// `mu,mu,nu,-mu,-mu,-nu`, matching Gaugefields.jl's rectangle ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// use wilsonloop::WilsonPath;
    ///
    /// let [nu_long, mu_long] = WilsonPath::rectangle_1x2(1, 2)?;
    /// assert_eq!(nu_long.steps(), &[1, 2, 2, -1, -2, -2]);
    /// assert_eq!(mu_long.steps(), &[1, 1, 2, -1, -1, -2]);
    /// # Ok::<(), wilsonloop::WilsonError>(())
    /// ```
    pub fn rectangle_1x2(mu: usize, nu: usize) -> Result<[Self; 2], WilsonError> {
        let (mu, nu) = distinct_axes(mu, nu)?;
        Ok([
            Self::new(vec![mu, nu, nu, -mu, -nu, -nu])?,
            Self::new(vec![mu, mu, nu, -mu, -mu, -nu])?,
        ])
    }
}

pub(crate) fn checked_step_displacement(
    mut displacement: [isize; 4],
    step: i8,
) -> Result<[isize; 4], WilsonError> {
    let (axis, forward) = decode_step(step).ok_or(WilsonError::InvalidStep { step })?;
    let delta = if forward { 1 } else { -1 };
    let next = displacement[axis]
        .checked_add(delta)
        .ok_or(WilsonError::DisplacementOverflow { axis })?;
    if next == isize::MIN {
        return Err(WilsonError::DisplacementOverflow { axis });
    }
    displacement[axis] = next;
    Ok(displacement)
}

pub(crate) fn decode_step(step: i8) -> Option<(usize, bool)> {
    match step {
        1..=4 => Some(((step - 1) as usize, true)),
        -4..=-1 => Some((((-step) - 1) as usize, false)),
        _ => None,
    }
}

fn distinct_axes(mu: usize, nu: usize) -> Result<(i8, i8), WilsonError> {
    if !(1..=4).contains(&mu) {
        return Err(WilsonError::InvalidAxis { axis: mu });
    }
    if !(1..=4).contains(&nu) {
        return Err(WilsonError::InvalidAxis { axis: nu });
    }
    if mu == nu {
        return Err(WilsonError::RepeatedAxis { axis: mu });
    }
    Ok((mu as i8, nu as i8))
}
