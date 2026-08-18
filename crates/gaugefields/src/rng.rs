//! Cross-language reproducible xoshiro256++ random streams.
//!
//! Julia 1.12.5's `Random.Xoshiro` state and scalar `rand(rng, UInt64)` stream
//! (commit `5fe89b8ddc166260bfcd4a195b305aff0ccad686`) are the compatibility
//! oracle for this module. The xoshiro transition is delegated to
//! `rand_xoshiro` 0.6.0 rather than copied from Julia's source;
//! the state-word ordering and little-endian seed conversion follow Julia's
//! `stdlib/Random/src/Xoshiro.jl` and `rand_xoshiro::Xoshiro256PlusPlus`.
//! The algorithm is described by Blackman and Vigna, "Scrambled Linear
//! Pseudorandom Number Generators," ACM TOMS 47(4), 2021. This is not a
//! cryptographic random-number generator.

use crate::GaugeError;
use rand::{Error, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::fmt;

/// A Julia-compatible, explicitly state-imported xoshiro256++ stream.
///
/// The four words are ordered as Julia's `(s0, s1, s2, s3)` and are converted
/// to little-endian bytes before entering `rand_xoshiro`. The wrapped generator
/// is the complete hidden state: normal generation does not cache a spare
/// Box--Muller value.
#[derive(Clone)]
pub struct ReproducibleRng {
    inner: Xoshiro256PlusPlus,
}

impl fmt::Debug for ReproducibleRng {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ReproducibleRng")
    }
}

impl ReproducibleRng {
    /// Imports a Julia `(s0, s1, s2, s3)` xoshiro256++ state.
    ///
    /// # Errors
    ///
    /// Returns [`GaugeError::InvalidRngState`] when all four state words are
    /// zero. No other nonzero state is remapped or rejected.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaugefields::ReproducibleRng;
    /// use rand::RngCore;
    ///
    /// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    /// assert_eq!(rng.next_u64(), 41_943_041);
    /// # Ok::<(), gaugefields::GaugeError>(())
    /// ```
    pub fn from_state(state: [u64; 4]) -> Result<Self, GaugeError> {
        validate_state(state)?;
        Ok(Self {
            inner: Xoshiro256PlusPlus::from_seed(seed_bytes(state)),
        })
    }

    /// Replaces the stream with a Julia `(s0, s1, s2, s3)` state.
    ///
    /// The replacement is transactional: an all-zero state returns an error
    /// and leaves the current stream position unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GaugeError::InvalidRngState`] when all four state words are
    /// zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaugefields::ReproducibleRng;
    /// use rand::RngCore;
    ///
    /// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4])?;
    /// let _ = rng.next_u64();
    /// rng.set_state([1, 2, 3, 4])?;
    /// assert_eq!(rng.next_u64(), 41_943_041);
    /// # Ok::<(), gaugefields::GaugeError>(())
    /// ```
    pub fn set_state(&mut self, state: [u64; 4]) -> Result<(), GaugeError> {
        let replacement = Self::from_state(state)?;
        self.inner = replacement.inner;
        Ok(())
    }

    /// Returns one open-unit-interval uniform from one raw `u64` word.
    ///
    /// The exact mapping is
    /// `(Float64(next_u64() >> 12) + 0.5) * 2^-52`, so this method consumes
    /// exactly one raw word and always returns a finite value strictly between
    /// zero and one.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaugefields::ReproducibleRng;
    ///
    /// let mut rng = ReproducibleRng::from_state([1, 0, 0, u64::MAX - 1]).unwrap();
    /// let u = rng.open_unit_f64();
    /// assert_eq!(u, 2f64.powi(-53));
    /// assert!(u > 0.0 && u < 1.0);
    /// ```
    pub fn open_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 12) as f64 + 0.5) * 2f64.powi(-52)
    }

    /// Returns an uncached Box--Muller standard-normal pair.
    ///
    /// Two raw words are consumed in order as `u1`, then `u2`; the result is
    /// `[r * cos(TAU * u2), r * sin(TAU * u2)]` with
    /// `r = sqrt(-2 * log(u1))`. No spare value is retained for a later call.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaugefields::ReproducibleRng;
    /// use rand::RngCore;
    ///
    /// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4]).unwrap();
    /// let pair = rng.standard_normal_pair();
    /// assert!(pair.into_iter().all(f64::is_finite));
    /// let mut after_pair = ReproducibleRng::from_state([1, 2, 3, 4]).unwrap();
    /// after_pair.next_u64();
    /// after_pair.next_u64();
    /// assert_eq!(rng.next_u64(), after_pair.next_u64());
    /// ```
    pub fn standard_normal_pair(&mut self) -> [f64; 2] {
        let u1 = self.open_unit_f64();
        let u2 = self.open_unit_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        [radius * theta.cos(), radius * theta.sin()]
    }

    /// Fills standard normals in Box--Muller pair order without caching.
    ///
    /// An odd-length output consumes one complete pair for its final element
    /// and discards the sine result. Thus a length `len` consumes
    /// `2 * ceil(len / 2)` raw words; an empty output consumes none.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaugefields::ReproducibleRng;
    /// use rand::RngCore;
    ///
    /// let mut rng = ReproducibleRng::from_state([1, 2, 3, 4]).unwrap();
    /// let mut output = [0.0; 3];
    /// rng.fill_standard_normals(&mut output);
    /// assert!(output.into_iter().all(f64::is_finite));
    /// let mut after_fill = ReproducibleRng::from_state([1, 2, 3, 4]).unwrap();
    /// for _ in 0..4 {
    ///     after_fill.next_u64();
    /// }
    /// assert_eq!(rng.next_u64(), after_fill.next_u64());
    /// ```
    pub fn fill_standard_normals(&mut self, output: &mut [f64]) {
        let mut pairs = output.chunks_exact_mut(2);
        for chunk in &mut pairs {
            chunk.copy_from_slice(&self.standard_normal_pair());
        }
        if let Some(last) = pairs.into_remainder().first_mut() {
            *last = self.standard_normal_pair()[0];
        }
    }
}

impl RngCore for ReproducibleRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn fill_bytes(&mut self, destination: &mut [u8]) {
        self.inner.fill_bytes(destination)
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Error> {
        self.inner.try_fill_bytes(destination)
    }
}

fn validate_state(state: [u64; 4]) -> Result<(), GaugeError> {
    if state == [0; 4] {
        Err(GaugeError::InvalidRngState)
    } else {
        Ok(())
    }
}

fn seed_bytes(state: [u64; 4]) -> [u8; 32] {
    let mut seed = [0u8; 32];
    for (index, word) in state.into_iter().enumerate() {
        seed[index * 8..(index + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    seed
}
