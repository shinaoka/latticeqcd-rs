use crate::DiracError;
use std::fmt;

/// Per-axis fermion boundary signs.
///
/// Each sign is applied once, and only once, when a one-hop displacement wraps
/// its corresponding lattice axis. This follows `boundarycondition_default` and
/// `shifted_fermion!` in
/// `LatticeDiracOperators.jl/src/WilsonFermion/WilsonFermion_4D_nowing.jl` at
/// revision `bdef628184597815ba3e0cddf2536df767e78a02`. The default is periodic
/// in space and antiperiodic in time: `[+1, +1, +1, -1]`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct FermionBoundary([i8; 4]);

impl fmt::Debug for FermionBoundary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FermionBoundary").field(&self.0).finish()
    }
}

impl Default for FermionBoundary {
    fn default() -> Self {
        Self([1, 1, 1, -1])
    }
}

impl FermionBoundary {
    /// Validate and construct four per-axis signs.
    ///
    /// # Errors
    ///
    /// Returns an error if any entry is not exactly `+1` or `-1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dirac_operators::FermionBoundary;
    ///
    /// let boundary = FermionBoundary::new([1, -1, 1, 1])?;
    /// assert_eq!(boundary.signs(), [1, -1, 1, 1]);
    /// # Ok::<(), dirac_operators::DiracError>(())
    /// ```
    pub fn new(signs: [i8; 4]) -> Result<Self, DiracError> {
        for (direction, &found) in signs.iter().enumerate() {
            if !matches!(found, -1 | 1) {
                return Err(DiracError::InvalidBoundary { direction, found });
            }
        }
        Ok(Self(signs))
    }

    /// Return the four validated signs in `[x, y, z, t]` order.
    pub const fn signs(self) -> [i8; 4] {
        self.0
    }

    /// Return one validated sign by direction.
    ///
    /// # Errors
    ///
    /// Returns a gaugefield direction error for a direction outside `0..4`.
    pub fn sign(self, direction: usize) -> Result<i8, DiracError> {
        self.0
            .get(direction)
            .copied()
            .ok_or(gaugefields::GaugeError::InvalidDirection { direction }.into())
    }
}
