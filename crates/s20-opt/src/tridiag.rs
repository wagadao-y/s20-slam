//! Symmetric block-tridiagonal normal equations (15x15 blocks) and their direct solution (block LDL^T).

use crate::imu::{M15, V15};
use anyhow::{Result, bail};
use rayon::prelude::*;

pub(crate) const PARALLEL_MIN_KNOTS: usize = 128;

pub struct BlockTridiag {
    /// diagonal blocks H_kk
    pub d: Vec<M15>,
    /// upper blocks H_k,k+1
    pub u: Vec<M15>,
    /// gradient
    pub g: Vec<V15>,
}

impl BlockTridiag {
    pub fn new(k: usize) -> Self {
        BlockTridiag {
            d: vec![M15::zeros(); k],
            u: vec![M15::zeros(); k.saturating_sub(1)],
            g: vec![V15::zeros(); k],
        }
    }

    pub fn len(&self) -> usize {
        self.d.len()
    }
    pub fn is_empty(&self) -> bool {
        self.d.is_empty()
    }

    /// Solve (H + lambda diag(H)) x = -g.
    pub fn solve(&self, lambda: f64) -> Result<Vec<V15>> {
        let f = self.factor(lambda)?;
        let rhs: Vec<V15> = self.g.iter().map(|g| -g).collect();
        Ok(f.apply(&rhs))
    }

    /// y = (H + lambda diag(H)) x
    pub fn mul(&self, x: &[V15], lambda: f64) -> Vec<V15> {
        let mut y = vec![V15::zeros(); self.len()];
        self.mul_into(x, lambda, &mut y);
        y
    }

    pub(crate) fn mul_into(&self, x: &[V15], lambda: f64, out: &mut [V15]) {
        let n = self.d.len();
        assert_eq!(out.len(), n);
        let row = |k: usize| {
            let mut y = self.d[k] * x[k];
            for i in 0..15 {
                // same damping as the factorisation (which adds 1e-12 to the Jacobi-scaled diagonal)
                y[i] += (lambda + 1e-12) * self.d[k][(i, i)].abs() * x[k][i];
            }
            if k + 1 < n {
                y += self.u[k] * x[k + 1];
            }
            if k > 0 {
                y += self.u[k - 1].transpose() * x[k - 1];
            }
            y
        };
        if n >= PARALLEL_MIN_KNOTS {
            out.par_iter_mut()
                .enumerate()
                .with_min_len(64)
                .for_each(|(k, y)| *y = row(k));
        } else {
            for (k, y) in out.iter_mut().enumerate() {
                *y = row(k);
            }
        }
    }

    /// Block LDL^T factorisation of the Jacobi-scaled matrix (the blocks mix very different magnitudes).
    pub fn factor(&self, lambda: f64) -> Result<Factor> {
        let n = self.d.len();
        let sc: Vec<V15> = self
            .d
            .iter()
            .map(|d| V15::from_fn(|i, _| 1.0 / d[(i, i)].abs().max(1e-30).sqrt()))
            .collect();
        let us: Vec<M15> = self
            .u
            .iter()
            .enumerate()
            .map(|(k, u)| M15::from_fn(|i, j| u[(i, j)] * sc[k][i] * sc[k + 1][j]))
            .collect();
        let mut sinv: Vec<M15> = Vec::with_capacity(n);
        let mut l: Vec<M15> = Vec::with_capacity(n);
        for k in 0..n {
            let mut s = M15::from_fn(|i, j| self.d[k][(i, j)] * sc[k][i] * sc[k][j]);
            for i in 0..15 {
                // Levenberg-Marquardt damping relative to the diagonal
                s[(i, i)] += lambda * s[(i, i)].abs() + 1e-12;
            }
            if k > 0 {
                let lk = us[k - 1].transpose() * sinv[k - 1];
                s -= lk * us[k - 1];
                l.push(lk);
            } else {
                l.push(M15::zeros());
            }
            let s = 0.5 * (s + s.transpose());
            let Some(c) = s.cholesky() else {
                let ev = s.symmetric_eigenvalues();
                bail!(
                    "block {k}/{n} is not positive definite (finite: {}, min eig {:.3e}, max eig {:.3e}, diag {:?})",
                    s.iter().all(|v| v.is_finite()),
                    ev.min(),
                    ev.max(),
                    self.d[k].diagonal().as_slice()
                )
            };
            sinv.push(c.inverse());
        }
        Ok(Factor { sc, us, sinv, l })
    }
}

pub struct Factor {
    sc: Vec<V15>,
    us: Vec<M15>,
    sinv: Vec<M15>,
    l: Vec<M15>,
}

/// Single-precision factorisation blocks.
type M15f = nalgebra::SMatrix<f32, 15, 15>;

/// The factorisation with its blocks in single precision, as a preconditioner: the substitution is
/// bound by memory traffic, which this halves. The scaling stays in double precision.
pub struct Factor32 {
    sc: Vec<V15>,
    us: Vec<M15f>,
    sinv: Vec<M15f>,
    l: Vec<M15f>,
}

impl Factor32 {
    /// x ~ (H + lambda diag(H))^-1 r
    pub(crate) fn apply_into(&self, r: &[V15], x: &mut [V15], y: &mut [V15]) {
        let n = self.sinv.len();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);
        let mv = |m: &M15f, v: &V15| -> V15 { (m * v.cast::<f32>()).cast::<f64>() };
        for k in 0..n {
            let mut v = r[k].component_mul(&self.sc[k]);
            if k > 0 {
                v -= mv(&self.l[k], &y[k - 1]);
            }
            y[k] = v;
        }
        for k in (0..n).rev() {
            let mut v = y[k];
            if k + 1 < n {
                v -= mv(&self.us[k], &x[k + 1]);
            }
            x[k] = mv(&self.sinv[k], &v);
        }
        for (x, s) in x.iter_mut().zip(&self.sc) {
            *x = x.component_mul(s);
        }
    }
}

impl Factor {
    /// The blocks in single precision (for preconditioning).
    pub fn single(self) -> Factor32 {
        let f = |v: Vec<M15>| -> Vec<M15f> { v.par_iter().map(|m| m.cast::<f32>()).collect() };
        Factor32 {
            sc: self.sc,
            us: f(self.us),
            sinv: f(self.sinv),
            l: f(self.l),
        }
    }

    /// x = (H + lambda diag(H))^-1 r
    pub fn apply(&self, r: &[V15]) -> Vec<V15> {
        let n = self.sinv.len();
        let mut y = vec![V15::zeros(); n];
        let mut x = vec![V15::zeros(); n];
        self.apply_into(r, &mut x, &mut y);
        x
    }

    pub(crate) fn apply_into(&self, r: &[V15], x: &mut [V15], y: &mut [V15]) {
        let n = self.sinv.len();
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);
        for k in 0..n {
            let mut v = r[k].component_mul(&self.sc[k]);
            if k > 0 {
                v -= self.l[k] * y[k - 1];
            }
            y[k] = v;
        }
        for k in (0..n).rev() {
            let mut v = y[k];
            if k + 1 < n {
                v -= self.us[k] * x[k + 1];
            }
            x[k] = self.sinv[k] * v;
        }
        for (x, s) in x.iter_mut().zip(&self.sc) {
            *x = x.component_mul(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn reused_buffers_match_original_arithmetic() {
        for n in [0, 1, 7, PARALLEL_MIN_KNOTS, 257] {
            let mut h = BlockTridiag::new(n);
            for k in 0..n {
                let a = M15::from_fn(|i, j| ((k * 13 + i * 7 + j * 19) as f64).sin() * 0.1);
                h.d[k] = a * a.transpose() + M15::identity() * 4.0;
                if k + 1 < n {
                    h.u[k] = a * 0.2;
                }
            }
            let mut out = vec![V15::repeat(f64::NAN); n];
            let mut scratch = out.clone();
            for lambda in [0.0, 1e-6, 0.1] {
                let f = h.factor(lambda).unwrap();
                for phase in [0.3, 1.7] {
                    let r: Vec<_> = (0..n)
                        .map(|k| V15::from_fn(|i, _| (k as f64 + i as f64 * phase).cos()))
                        .collect();
                    // The original allocation-based triangular substitution.
                    let mut y: Vec<V15> = Vec::with_capacity(n);
                    for k in 0..n {
                        let mut v = r[k].component_mul(&f.sc[k]);
                        if k > 0 {
                            v -= f.l[k] * y[k - 1];
                        }
                        y.push(v);
                    }
                    let mut expected = vec![V15::zeros(); n];
                    for k in (0..n).rev() {
                        let mut v = y[k];
                        if k + 1 < n {
                            v -= f.us[k] * expected[k + 1];
                        }
                        expected[k] = f.sinv[k] * v;
                    }
                    for (x, s) in expected.iter_mut().zip(&f.sc) {
                        *x = x.component_mul(s);
                    }
                    f.apply_into(&r, &mut out, &mut scratch);
                    assert_bits(&out, &expected);
                    assert_bits(&f.apply(&r), &expected);
                    let expected: Vec<_> = (0..n)
                        .map(|k| {
                            let mut y = h.d[k] * r[k];
                            for i in 0..15 {
                                y[i] += (lambda + 1e-12) * h.d[k][(i, i)].abs() * r[k][i];
                            }
                            if k + 1 < n {
                                y += h.u[k] * r[k + 1];
                            }
                            if k > 0 {
                                y += h.u[k - 1].transpose() * r[k - 1];
                            }
                            y
                        })
                        .collect();
                    h.mul_into(&r, lambda, &mut out);
                    assert_bits(&out, &expected);
                    assert_bits(&h.mul(&r, lambda), &expected);
                }
            }
        }
    }

    fn assert_bits(a: &[V15], b: &[V15]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn matches_dense() {
        let n = 5;
        let mut h = BlockTridiag::new(n);
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for k in 0..n {
            let a = M15::from_fn(|_, _| rnd());
            h.d[k] = a * a.transpose() + M15::identity() * 20.0;
            h.g[k] = V15::from_fn(|_, _| rnd());
            if k + 1 < n {
                h.u[k] = M15::from_fn(|_, _| rnd());
            }
        }
        let x = h.solve(0.0).unwrap();
        // mul is the inverse of apply
        let f = h.factor(0.0).unwrap();
        let y = h.mul(&f.apply(&h.g), 0.0);
        for (yk, gk) in y.iter().zip(&h.g) {
            assert!((yk - gk).norm() < 1e-9 * (1.0 + gk.norm()));
        }
        let mut dm = DMatrix::<f64>::zeros(15 * n, 15 * n);
        for k in 0..n {
            dm.view_mut((15 * k, 15 * k), (15, 15)).copy_from(&h.d[k]);
            if k + 1 < n {
                dm.view_mut((15 * k, 15 * k + 15), (15, 15))
                    .copy_from(&h.u[k]);
                dm.view_mut((15 * k + 15, 15 * k), (15, 15))
                    .copy_from(&h.u[k].transpose());
            }
        }
        let gv =
            nalgebra::DVector::from_iterator(15 * n, h.g.iter().flat_map(|g| g.iter().copied()));
        let xd = dm.lu().solve(&(-gv)).unwrap();
        for k in 0..n {
            for i in 0..15 {
                assert!((x[k][i] - xd[15 * k + i]).abs() < 1e-9);
            }
        }
    }
}
