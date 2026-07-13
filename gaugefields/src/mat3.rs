use crate::GaugeError;
use num_complex::Complex64 as C;
use std::ops::{Index, IndexMut};

/// Fixed-size column-major SU(3) matrix kernel payload.
///
/// The TA convention follows Gaugefields.jl
/// `src/4D/TA_gaugefields_4D_serial.jl:181-269,356-433`; generator storage is
/// defined at `:1-29`. Gell-Mann multiplication follows
/// `src/4D/wing/gaugefields_4D_wing.jl:2788-2944`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3(pub [C; 9]);

impl Mat3 {
    /// Wraps column-major components; Julia storage: `gaugefields_4D_nowing.jl:18-49`.
    pub const fn from_array(values: [C; 9]) -> Self {
        Self(values)
    }
    /// Borrows column-major components; Julia indexing: `gaugefields_4D_nowing.jl:73-79`.
    pub const fn as_array(&self) -> &[C; 9] {
        &self.0
    }
    /// Zero matrix; Julia allocation: `gaugefields_4D_nowing.jl:41-42`.
    pub const fn zero() -> Self {
        Self([C::new(0.0, 0.0); 9])
    }
    /// Identity matrix; Julia cold initialization convention: `TA_gaugefields_4D_serial.jl:800-810`.
    pub fn identity() -> Self {
        let z = C::new(0.0, 0.0);
        let o = C::new(1.0, 0.0);
        Self([o, z, z, z, o, z, z, z, o])
    }
    /// Loads one contiguous site block; Julia component order: `gaugefields_4D_nowing.jl:73-79`.
    pub fn load(values: &[C], offset: usize) -> Result<Self, GaugeError> {
        let end = offset
            .checked_add(9)
            .ok_or(GaugeError::MatrixBlockOutOfBounds {
                offset,
                len: values.len(),
            })?;
        let s = values
            .get(offset..end)
            .ok_or(GaugeError::MatrixBlockOutOfBounds {
                offset,
                len: values.len(),
            })?;
        Ok(Self(s.try_into().expect("length checked")))
    }
    /// Stores one contiguous site block; Julia component order: `gaugefields_4D_nowing.jl:73-79`.
    pub fn store(self, values: &mut [C], offset: usize) -> Result<(), GaugeError> {
        let len = values.len();
        let end = offset
            .checked_add(9)
            .ok_or(GaugeError::MatrixBlockOutOfBounds { offset, len })?;
        values
            .get_mut(offset..end)
            .ok_or(GaugeError::MatrixBlockOutOfBounds { offset, len })?
            .copy_from_slice(&self.0);
        Ok(())
    }
    /// Conjugate transpose; Julia component convention: `gaugefields_4D_nowing.jl:77-79`.
    pub fn adjoint(self) -> Self {
        let a = self.0;
        Self([
            a[0].conj(),
            a[3].conj(),
            a[6].conj(),
            a[1].conj(),
            a[4].conj(),
            a[7].conj(),
            a[2].conj(),
            a[5].conj(),
            a[8].conj(),
        ])
    }
    /// Matrix trace; Julia diagonal access: `TA_gaugefields_4D_serial.jl:197-201`.
    pub fn trace(self) -> C {
        self[(0, 0)] + self[(1, 1)] + self[(2, 2)]
    }
    #[allow(clippy::should_implement_trait)]
    /// Explicit 3×3 product; Julia fixed-size kernel model: `gaugefields_4D_wing.jl:2788-2944`.
    pub fn mul(self, b: Self) -> Self {
        let a = self;
        Self([
            a[(0, 0)] * b[(0, 0)] + a[(0, 1)] * b[(1, 0)] + a[(0, 2)] * b[(2, 0)],
            a[(1, 0)] * b[(0, 0)] + a[(1, 1)] * b[(1, 0)] + a[(1, 2)] * b[(2, 0)],
            a[(2, 0)] * b[(0, 0)] + a[(2, 1)] * b[(1, 0)] + a[(2, 2)] * b[(2, 0)],
            a[(0, 0)] * b[(0, 1)] + a[(0, 1)] * b[(1, 1)] + a[(0, 2)] * b[(2, 1)],
            a[(1, 0)] * b[(0, 1)] + a[(1, 1)] * b[(1, 1)] + a[(1, 2)] * b[(2, 1)],
            a[(2, 0)] * b[(0, 1)] + a[(2, 1)] * b[(1, 1)] + a[(2, 2)] * b[(2, 1)],
            a[(0, 0)] * b[(0, 2)] + a[(0, 1)] * b[(1, 2)] + a[(0, 2)] * b[(2, 2)],
            a[(1, 0)] * b[(0, 2)] + a[(1, 1)] * b[(1, 2)] + a[(1, 2)] * b[(2, 2)],
            a[(2, 0)] * b[(0, 2)] + a[(2, 1)] * b[(1, 2)] + a[(2, 2)] * b[(2, 2)],
        ])
    }
    /// Explicit `self† rhs`; Julia adjoint multiplication model: `gaugefields_4D_wing.jl:2788-2944`.
    pub fn mul_adj_left(self, b: Self) -> Self {
        let a = self;
        Self([
            a[(0, 0)].conj() * b[(0, 0)]
                + a[(1, 0)].conj() * b[(1, 0)]
                + a[(2, 0)].conj() * b[(2, 0)],
            a[(0, 1)].conj() * b[(0, 0)]
                + a[(1, 1)].conj() * b[(1, 0)]
                + a[(2, 1)].conj() * b[(2, 0)],
            a[(0, 2)].conj() * b[(0, 0)]
                + a[(1, 2)].conj() * b[(1, 0)]
                + a[(2, 2)].conj() * b[(2, 0)],
            a[(0, 0)].conj() * b[(0, 1)]
                + a[(1, 0)].conj() * b[(1, 1)]
                + a[(2, 0)].conj() * b[(2, 1)],
            a[(0, 1)].conj() * b[(0, 1)]
                + a[(1, 1)].conj() * b[(1, 1)]
                + a[(2, 1)].conj() * b[(2, 1)],
            a[(0, 2)].conj() * b[(0, 1)]
                + a[(1, 2)].conj() * b[(1, 1)]
                + a[(2, 2)].conj() * b[(2, 1)],
            a[(0, 0)].conj() * b[(0, 2)]
                + a[(1, 0)].conj() * b[(1, 2)]
                + a[(2, 0)].conj() * b[(2, 2)],
            a[(0, 1)].conj() * b[(0, 2)]
                + a[(1, 1)].conj() * b[(1, 2)]
                + a[(2, 1)].conj() * b[(2, 2)],
            a[(0, 2)].conj() * b[(0, 2)]
                + a[(1, 2)].conj() * b[(1, 2)]
                + a[(2, 2)].conj() * b[(2, 2)],
        ])
    }
    /// Explicit `self rhs†`; Julia adjoint multiplication model: `gaugefields_4D_wing.jl:2788-2944`.
    pub fn mul_adj_right(self, b: Self) -> Self {
        let a = self;
        Self([
            a[(0, 0)] * b[(0, 0)].conj()
                + a[(0, 1)] * b[(0, 1)].conj()
                + a[(0, 2)] * b[(0, 2)].conj(),
            a[(1, 0)] * b[(0, 0)].conj()
                + a[(1, 1)] * b[(0, 1)].conj()
                + a[(1, 2)] * b[(0, 2)].conj(),
            a[(2, 0)] * b[(0, 0)].conj()
                + a[(2, 1)] * b[(0, 1)].conj()
                + a[(2, 2)] * b[(0, 2)].conj(),
            a[(0, 0)] * b[(1, 0)].conj()
                + a[(0, 1)] * b[(1, 1)].conj()
                + a[(0, 2)] * b[(1, 2)].conj(),
            a[(1, 0)] * b[(1, 0)].conj()
                + a[(1, 1)] * b[(1, 1)].conj()
                + a[(1, 2)] * b[(1, 2)].conj(),
            a[(2, 0)] * b[(1, 0)].conj()
                + a[(2, 1)] * b[(1, 1)].conj()
                + a[(2, 2)] * b[(1, 2)].conj(),
            a[(0, 0)] * b[(2, 0)].conj()
                + a[(0, 1)] * b[(2, 1)].conj()
                + a[(0, 2)] * b[(2, 2)].conj(),
            a[(1, 0)] * b[(2, 0)].conj()
                + a[(1, 1)] * b[(2, 1)].conj()
                + a[(1, 2)] * b[(2, 2)].conj(),
            a[(2, 0)] * b[(2, 0)].conj()
                + a[(2, 1)] * b[(2, 1)].conj()
                + a[(2, 2)] * b[(2, 2)].conj(),
        ])
    }
    /// `Re tr(self rhs)` without a product; Julia trace-product model: `TA_gaugefields_4D_serial.jl:106-129`.
    pub fn real_trace_mul(self, b: Self) -> f64 {
        (self[(0, 0)] * b[(0, 0)]
            + self[(0, 1)] * b[(1, 0)]
            + self[(0, 2)] * b[(2, 0)]
            + self[(1, 0)] * b[(0, 1)]
            + self[(1, 1)] * b[(1, 1)]
            + self[(1, 2)] * b[(2, 1)]
            + self[(2, 0)] * b[(0, 2)]
            + self[(2, 1)] * b[(1, 2)]
            + self[(2, 2)] * b[(2, 2)])
            .re
    }
    /// Complex scaling; Julia `TA_gaugefields_4D_serial.jl:150-169`.
    pub fn scaled(self, a: C) -> Self {
        let x = self.0;
        Self([
            a * x[0],
            a * x[1],
            a * x[2],
            a * x[3],
            a * x[4],
            a * x[5],
            a * x[6],
            a * x[7],
            a * x[8],
        ])
    }
    /// Real scaled addition; Julia `TA_gaugefields_4D_serial.jl:150-169`.
    pub fn add_scaled_real(&mut self, a: f64, rhs: Self) {
        self.0[0] += a * rhs.0[0];
        self.0[1] += a * rhs.0[1];
        self.0[2] += a * rhs.0[2];
        self.0[3] += a * rhs.0[3];
        self.0[4] += a * rhs.0[4];
        self.0[5] += a * rhs.0[5];
        self.0[6] += a * rhs.0[6];
        self.0[7] += a * rhs.0[7];
        self.0[8] += a * rhs.0[8];
    }
    /// Complex scaled addition; Julia `TA_gaugefields_4D_serial.jl:150-169`.
    pub fn add_scaled_complex(&mut self, a: C, rhs: Self) {
        self.0[0] += a * rhs.0[0];
        self.0[1] += a * rhs.0[1];
        self.0[2] += a * rhs.0[2];
        self.0[3] += a * rhs.0[3];
        self.0[4] += a * rhs.0[4];
        self.0[5] += a * rhs.0[5];
        self.0[6] += a * rhs.0[6];
        self.0[7] += a * rhs.0[7];
        self.0[8] += a * rhs.0[8];
    }
    /// Traceless anti-Hermitian projection; Gaugefields.jl `TA_gaugefields_4D_serial.jl:197-240,356-433`.
    pub fn ta(self) -> Self {
        let a = self.0;
        let tr = (a[0].im + a[4].im + a[8].im) / 3.0;
        let x01 = (a[3] - a[1].conj()) * 0.5;
        let x02 = (a[6] - a[2].conj()) * 0.5;
        let x12 = (a[7] - a[5].conj()) * 0.5;
        Self([
            C::new(0.0, a[0].im - tr),
            -x01.conj(),
            -x02.conj(),
            x01,
            C::new(0.0, a[4].im - tr),
            -x12.conj(),
            x02,
            x12,
            C::new(0.0, a[8].im - tr),
        ])
    }
    /// Extracts `u` from TA input `A=(i/2)Σu_aλ_a`; Gaugefields.jl `TA_gaugefields_4D_serial.jl:243-260`.
    pub fn gell_mann_coefficients(self) -> [f64; 8] {
        let a = self.ta();
        [
            a.0[3].im + a.0[1].im,
            a.0[3].re - a.0[1].re,
            a.0[0].im - a.0[4].im,
            a.0[6].im + a.0[2].im,
            a.0[6].re - a.0[2].re,
            a.0[7].im + a.0[5].im,
            a.0[7].re - a.0[5].re,
            (a.0[0].im + a.0[4].im - 2.0 * a.0[8].im) / 3f64.sqrt(),
        ]
    }
    /// Accumulates `out += factor * coefficients(TA(vin))`; Gaugefields.jl `TA_gaugefields_4D_serial.jl:181-269`.
    pub fn add_ta_coefficients(out: &mut [f64; 8], factor: f64, vin: Self) {
        let c = vin.gell_mann_coefficients();
        out[0] += factor * c[0];
        out[1] += factor * c[1];
        out[2] += factor * c[2];
        out[3] += factor * c[3];
        out[4] += factor * c[4];
        out[5] += factor * c[5];
        out[6] += factor * c[6];
        out[7] += factor * c[7];
    }
    /// Reconstructs `V=Σc_aλ_a` (no half); Gaugefields.jl `TA_gaugefields_4D_serial.jl:816-847`.
    pub fn hermitian_from_gell_mann(c: [f64; 8]) -> Self {
        let [c1, c2, c3, c4, c5, c6, c7, c8] = c;
        let r = 3f64.sqrt();
        Self([
            C::new(c3 + c8 / r, 0.0),
            C::new(c1, c2),
            C::new(c4, c5),
            C::new(c1, -c2),
            C::new(-c3 + c8 / r, 0.0),
            C::new(c6, c7),
            C::new(c4, -c5),
            C::new(c6, -c7),
            C::new(-2.0 * c8 / r, 0.0),
        ])
    }
    /// Constructs `A=(i/2)Σ c_a λ_a` in the established TA convention.
    pub fn from_gell_mann_coefficients(c: [f64; 8]) -> Self {
        Self::hermitian_from_gell_mann(c).scaled(C::new(0.0, 0.5))
    }
}
impl Index<(usize, usize)> for Mat3 {
    type Output = C;
    fn index(&self, (i, j): (usize, usize)) -> &C {
        &self.0[i + 3 * j]
    }
}
impl IndexMut<(usize, usize)> for Mat3 {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut C {
        &mut self.0[i + 3 * j]
    }
}
