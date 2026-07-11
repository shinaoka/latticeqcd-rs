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
pub struct Mat3([C; 9]);

impl Mat3 {
    pub const fn from_array(values: [C; 9]) -> Self {
        Self(values)
    }
    pub const fn as_array(&self) -> &[C; 9] {
        &self.0
    }
    pub const fn zero() -> Self {
        Self([C::new(0.0, 0.0); 9])
    }
    pub fn identity() -> Self {
        let mut x = Self::zero();
        for i in 0..3 {
            x[(i, i)] = C::new(1.0, 0.0);
        }
        x
    }
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
    pub fn adjoint(self) -> Self {
        Self::from_array(std::array::from_fn(|q| {
            let i = q % 3;
            let j = q / 3;
            self[(j, i)].conj()
        }))
    }
    pub fn trace(self) -> C {
        self[(0, 0)] + self[(1, 1)] + self[(2, 2)]
    }
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        let mut o = Self::zero();
        for j in 0..3 {
            for i in 0..3 {
                for k in 0..3 {
                    o[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        o
    }
    pub fn mul_adj_left(self, rhs: Self) -> Self {
        let mut o = Self::zero();
        for j in 0..3 {
            for i in 0..3 {
                for k in 0..3 {
                    o[(i, j)] += self[(k, i)].conj() * rhs[(k, j)];
                }
            }
        }
        o
    }
    pub fn mul_adj_right(self, rhs: Self) -> Self {
        let mut o = Self::zero();
        for j in 0..3 {
            for i in 0..3 {
                for k in 0..3 {
                    o[(i, j)] += self[(i, k)] * rhs[(j, k)].conj();
                }
            }
        }
        o
    }
    pub fn real_trace_mul(self, rhs: Self) -> f64 {
        let mut s = C::default();
        for i in 0..3 {
            for k in 0..3 {
                s += self[(i, k)] * rhs[(k, i)];
            }
        }
        s.re
    }
    pub fn scaled(self, a: C) -> Self {
        Self::from_array(self.0.map(|x| a * x))
    }
    pub fn add_scaled_real(&mut self, a: f64, rhs: Self) {
        for i in 0..9 {
            self.0[i] += a * rhs.0[i];
        }
    }
    pub fn add_scaled_complex(&mut self, a: C, rhs: Self) {
        for i in 0..9 {
            self.0[i] += a * rhs.0[i];
        }
    }
    pub fn ta(self) -> Self {
        let mut a = Self::zero();
        for j in 0..3 {
            for i in 0..3 {
                a[(i, j)] = (self[(i, j)] - self[(j, i)].conj()) * 0.5;
            }
        }
        let tr = a.trace() / 3.0;
        for i in 0..3 {
            a[(i, i)] -= tr;
        }
        a
    }
    fn lambda(k: usize) -> Self {
        let z = C::new(0.0, 0.0);
        let o = C::new(1.0, 0.0);
        let i = C::new(0.0, 1.0);
        let r = 3f64.sqrt();
        match k {
            0 => Self([z, o, z, o, z, z, z, z, z]),
            1 => Self([z, i, z, -i, z, z, z, z, z]),
            2 => Self([o, z, z, z, -o, z, z, z, z]),
            3 => Self([z, z, o, z, z, z, o, z, z]),
            4 => Self([z, z, i, z, z, z, -i, z, z]),
            5 => Self([z, z, z, z, z, o, z, o, z]),
            6 => Self([z, z, z, z, z, i, z, -i, z]),
            7 => Self([o / r, z, z, z, o / r, z, z, z, -2.0 * o / r]),
            _ => unreachable!(),
        }
    }
    pub fn gell_mann_coefficients(self) -> [f64; 8] {
        std::array::from_fn(|k| Self::lambda(k).mul(self).trace().im)
    }
    pub fn hermitian_from_gell_mann(c: [f64; 8]) -> Self {
        let mut o = Self::zero();
        for (k, &v) in c.iter().enumerate() {
            o.add_scaled_real(0.5 * v, Self::lambda(k));
        }
        o
    }
    pub fn add_gell_mann_factor(&mut self, c: [f64; 8], factor: C) {
        for (k, &v) in c.iter().enumerate() {
            self.add_scaled_complex(factor * (0.5 * v), Self::lambda(k));
        }
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
