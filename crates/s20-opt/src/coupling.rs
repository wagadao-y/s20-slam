//! Plane centroid coupling between knots.
//!
//! With the plane centroid c = sum w q / sum w moving with the points, the Gauss-Newton matrix of the plane
//! terms is sum w J^T J - (1/W) (sum w J)^T (sum w J). The block-diagonal part sits in the tridiagonal system;
//! the low-rank part (one rank-1 term per plane, over the knots that see it) is applied here, and the full
//! system is solved by conjugate gradients preconditioned with the tridiagonal factorisation.
//!
//! The point Jacobian on [theta, p, v] is [J_theta; n; tau n] with the plane normal n, so the sum
//! u = sum w J of one (plane, knot) is [a; w n; wtau n]: only a = sum w J_theta, w and wtau = sum w tau
//! are stored, the normal once per plane. The products are bound by memory traffic, which this halves.

use crate::imu::M15;
use crate::imu::V15;
use crate::tridiag::BlockTridiag;
use anyhow::Result;
use nalgebra::SVector;
use rayon::prelude::*;
use s20_geom::so3::V3;

/// Aggregate of the points of one plane seen from one knot: a = sum w J_theta, w = sum w,
/// wtau = sum w tau, wr = sum w r.
#[derive(Clone, Copy)]
pub struct Coupling {
    pub plane: u32,
    pub knot: u32,
    pub a: V3,
    pub w: f64,
    pub wtau: f64,
    pub wr: f64,
}

impl Coupling {
    /// u = sum w J on [theta, p, v], with the plane normal `n`.
    pub fn u(&self, n: &V3) -> SVector<f64, 9> {
        let mut u = SVector::<f64, 9>::zeros();
        u.fixed_rows_mut::<3>(0).copy_from(&self.a);
        u.fixed_rows_mut::<3>(3).copy_from(&(n * self.w));
        u.fixed_rows_mut::<3>(6).copy_from(&(n * self.wtau));
        u
    }
}

/// Merge entries of the same plane (entries of one knot).
pub fn merge(mut cp: Vec<Coupling>) -> Vec<Coupling> {
    cp.sort_unstable_by_key(|c| c.plane);
    let mut m: Vec<Coupling> = Vec::new();
    for c in cp {
        match m.last_mut() {
            Some(l) if l.plane == c.plane => {
                l.a += c.a;
                l.w += c.w;
                l.wtau += c.wtau;
                l.wr += c.wr;
            }
            _ => m.push(c),
        }
    }
    m
}

/// One (plane, knot) entry as the products use it.
#[derive(Clone, Copy)]
struct Entry {
    a: [f64; 3],
    w: f64,
    wtau: f64,
    plane: u32,
}

impl Entry {
    #[inline]
    fn dot(&self, n: &V3, x: &V15) -> f64 {
        let a = self.a[0] * x[0] + self.a[1] * x[1] + self.a[2] * x[2];
        let p = self.w * x[3] + self.wtau * x[6];
        let q = self.w * x[4] + self.wtau * x[7];
        let r = self.w * x[5] + self.wtau * x[8];
        a + n.x * p + n.y * q + n.z * r
    }

    /// y -= f u
    #[inline]
    fn sub(&self, n: &V3, f: f64, y: &mut V15) {
        for d in 0..3 {
            y[d] -= f * self.a[d];
            y[3 + d] -= f * self.w * n[d];
            y[6 + d] -= f * self.wtau * n[d];
        }
    }
}

pub struct PlaneCoupling {
    /// entries sorted by knot (in their original order within a knot); knot k uses start[k]..start[k+1]
    e: Vec<Entry>,
    start: Vec<usize>,
    normal: Vec<V3>,
    wsum: Vec<f64>,
    wrsum: Vec<f64>,
    /// knot ranges of about equal entry counts, processed in parallel; each has its own plane sums
    chunks: Vec<(usize, usize)>,
}

impl PlaneCoupling {
    /// `normal`: normal of every plane (the planes of the entries index it).
    pub fn new(e: Vec<Coupling>, normal: Vec<V3>) -> Self {
        let n_planes = normal.len();
        let knots = e.iter().map(|c| c.knot as usize + 1).max().unwrap_or(0);
        let mut start = vec![0usize; knots + 1];
        for c in &e {
            start[c.knot as usize + 1] += 1;
        }
        for k in 0..knots {
            start[k + 1] += start[k];
        }
        let mut wsum = vec![0.0; n_planes];
        let mut wrsum = vec![0.0; n_planes];
        for c in &e {
            wsum[c.plane as usize] += c.w;
            wrsum[c.plane as usize] += c.wr;
        }
        // stable bucketing by knot
        let mut fill = start[..knots].to_vec();
        let mut entries = vec![
            Entry {
                a: [0.0; 3],
                w: 0.0,
                wtau: 0.0,
                plane: 0,
            };
            e.len()
        ];
        for c in &e {
            let pos = &mut fill[c.knot as usize];
            entries[*pos] = Entry {
                a: [c.a.x, c.a.y, c.a.z],
                w: c.w,
                wtau: c.wtau,
                plane: c.plane,
            };
            *pos += 1;
        }
        // small systems in one chunk; else a few chunks per thread for balance
        let n_chunks = if e.len() < 4096 {
            1
        } else {
            (rayon::current_num_threads() * 2).min(knots.max(1))
        };
        let mut chunks = Vec::with_capacity(n_chunks);
        let mut k0 = 0;
        for c in 1..=n_chunks {
            let target = e.len() * c / n_chunks;
            let k1 = if c == n_chunks {
                knots
            } else {
                start.partition_point(|&s| s < target).clamp(k0, knots)
            };
            if k1 > k0 {
                chunks.push((k0, k1));
            }
            k0 = k1;
        }
        PlaneCoupling {
            e: entries,
            start,
            normal,
            wsum,
            wrsum,
            chunks,
        }
    }

    pub fn entries(&self) -> usize {
        self.e.len()
    }

    /// g -= (sum w r / W) u: gradient with the weighted centroid.
    pub fn correct_gradient(&self, g: &mut [V15]) {
        for (k, gk) in g
            .iter_mut()
            .enumerate()
            .take(self.start.len().saturating_sub(1))
        {
            for c in &self.e[self.start[k]..self.start[k + 1]] {
                let p = c.plane as usize;
                if self.wsum[p] > 0.0 {
                    c.sub(&self.normal[p], self.wrsum[p] / self.wsum[p], gk);
                }
            }
        }
    }

    /// y -= sum_p (1/W_p) u_p (u_p . x). `sums`: one plane-sum buffer per chunk; `s`: plane factors.
    fn apply_lowrank(&self, x: &[V15], y: &mut [V15], sums: &mut [Vec<f64>], s: &mut [f64]) {
        if self.e.is_empty() {
            return;
        }
        // s_p = (u_p . x) / W_p, summed per chunk then over the chunks in a fixed order
        self.chunks
            .par_iter()
            .zip(sums.par_iter_mut())
            .for_each(|(&(k0, k1), acc)| {
                acc.fill(0.0);
                for (k, xk) in x.iter().enumerate().take(k1).skip(k0) {
                    for c in &self.e[self.start[k]..self.start[k + 1]] {
                        let p = c.plane as usize;
                        acc[p] += c.dot(&self.normal[p], xk);
                    }
                }
            });
        let sums = &*sums;
        s.par_iter_mut()
            .enumerate()
            .with_min_len(1024)
            .for_each(|(p, v)| {
                *v = if self.wsum[p] > 0.0 {
                    sums.iter().map(|a| a[p]).sum::<f64>() / self.wsum[p]
                } else {
                    0.0
                };
            });
        let s = &*s;
        y[..self.start.len() - 1]
            .par_iter_mut()
            .enumerate()
            .with_min_len(64)
            .for_each(|(k, y)| {
                for c in &self.e[self.start[k]..self.start[k + 1]] {
                    let p = c.plane as usize;
                    if s[p] != 0.0 {
                        c.sub(&self.normal[p], s[p], y);
                    }
                }
            });
    }

    /// Preconditioned conjugate gradients on (H_tri - lowrank) x = -g, preconditioned by H_tri^-1
    /// (factorised in single precision: a preconditioner only steers the iterations).
    /// Returns the solution and the number of iterations.
    /// `far`: off-diagonal blocks H_ab of knots further apart than one (loops), applied in the
    /// product only (the preconditioner is the tridiagonal part).
    pub fn solve(
        &self,
        h: &BlockTridiag,
        lambda: f64,
        tol: f64,
        max_it: usize,
        far: &[(usize, usize, M15)],
    ) -> Result<(Vec<V15>, usize)> {
        let f = h.factor(lambda)?.single();
        let b: Vec<V15> = h.g.iter().map(|g| -g).collect();
        let mut plane_values = vec![0.0; self.wsum.len()];
        let mut plane_sums = vec![vec![0.0; self.wsum.len()]; self.chunks.len()];
        let mut forward = vec![V15::zeros(); h.len()];
        let dot = |a: &[V15], b: &[V15]| a.iter().zip(b).map(|(x, y)| x.dot(y)).sum::<f64>();
        let mut mul = |x: &[V15], y: &mut [V15]| {
            h.mul_into(x, lambda, y);
            self.apply_lowrank(x, y, &mut plane_sums, &mut plane_values);
            for (a, b, m) in far {
                y[*a] += m * x[*b];
                y[*b] += m.transpose() * x[*a];
            }
        };
        let mut x = vec![V15::zeros(); h.len()];
        let mut ap = vec![V15::zeros(); h.len()];
        f.apply_into(&b, &mut x, &mut forward);
        mul(&x, &mut ap);
        let mut r: Vec<V15> = b.iter().zip(&ap).map(|(b, a)| b - a).collect();
        let mut z = vec![V15::zeros(); h.len()];
        f.apply_into(&r, &mut z, &mut forward);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let rz0 = dot(&b, &x).max(1e-300);
        let mut it = 0;
        while it < max_it && rz > tol * tol * rz0 {
            mul(&p, &mut ap);
            let pap = dot(&p, &ap);
            if pap <= 0.0 {
                break;
            }
            let alpha = rz / pap;
            for i in 0..x.len() {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            f.apply_into(&r, &mut z, &mut forward);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            for i in 0..p.len() {
                p[i] = z[i] + beta * p[i];
            }
            it += 1;
        }
        Ok((x, it))
    }
}

#[cfg(test)]
#[path = "solver_regression.rs"]
mod regression;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::M15;
    use nalgebra::{DMatrix, DVector};
    use std::ops::AddAssign;

    #[test]
    fn compact_entries_give_the_full_jacobian_sum() {
        let n = V3::new(0.6, -0.48, 0.64);
        let mut c = Coupling {
            plane: 0,
            knot: 0,
            a: V3::zeros(),
            w: 0.0,
            wtau: 0.0,
            wr: 0.0,
        };
        let mut u = SVector::<f64, 9>::zeros();
        for (i, (wt, tau)) in [(0.7, 0.01), (1.3, 0.05), (0.2, 0.09)]
            .into_iter()
            .enumerate()
        {
            let jt = V3::new(i as f64 - 0.5, 0.3 * i as f64, 1.0);
            let mut j = SVector::<f64, 9>::zeros();
            j.fixed_rows_mut::<3>(0).copy_from(&jt);
            j.fixed_rows_mut::<3>(3).copy_from(&n);
            j.fixed_rows_mut::<3>(6).copy_from(&(n * tau));
            u += wt * j;
            c.a += wt * jt;
            c.w += wt;
            c.wtau += wt * tau;
        }
        assert!((c.u(&n) - u).norm() < 1e-12);
    }

    #[test]
    fn pcg_matches_dense() {
        let n = 6;
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        let mut h = BlockTridiag::new(n);
        for k in 0..n {
            let a = M15::from_fn(|_, _| rnd());
            h.d[k] = a * a.transpose() + M15::identity() * 5.0;
            h.g[k] = V15::from_fn(|_, _| rnd());
            if k + 1 < n {
                h.u[k] = M15::from_fn(|_, _| 0.3 * rnd());
            }
        }
        // two planes seen from several knots; add their block-diagonal part (sum w J^T J) to the tridiagonal system
        let normals = vec![V3::new(0.0, 0.6, 0.8), V3::new(1.0, 0.0, 0.0)];
        let mut e = Vec::new();
        for (plane, knots) in [(0u32, vec![0usize, 2, 5]), (1u32, vec![1usize, 3, 4, 5])] {
            let nrm = normals[plane as usize];
            for &k in &knots {
                let mut hh = nalgebra::SMatrix::<f64, 9, 9>::zeros();
                let mut c = Coupling {
                    plane,
                    knot: k as u32,
                    a: V3::zeros(),
                    w: 0.0,
                    wtau: 0.0,
                    wr: 0.0,
                };
                for _ in 0..4 {
                    let jt = V3::new(rnd(), rnd(), rnd());
                    let tau = 0.1 * rnd().abs();
                    let mut j = SVector::<f64, 9>::zeros();
                    j.fixed_rows_mut::<3>(0).copy_from(&jt);
                    j.fixed_rows_mut::<3>(3).copy_from(&nrm);
                    j.fixed_rows_mut::<3>(6).copy_from(&(nrm * tau));
                    let wt = 1.0 + rnd().abs();
                    hh += wt * j * j.transpose();
                    c.a += wt * jt;
                    c.w += wt;
                    c.wtau += wt * tau;
                }
                h.d[k].fixed_view_mut::<9, 9>(0, 0).add_assign(&hh);
                e.push(c);
            }
        }
        let pc = PlaneCoupling::new(e.clone(), normals.clone());
        let (x, _) = pc.solve(&h, 0.0, 1e-12, 500, &[]).unwrap();
        // dense reference
        let m = 15 * n;
        let mut dm = DMatrix::<f64>::zeros(m, m);
        for k in 0..n {
            dm.view_mut((15 * k, 15 * k), (15, 15)).copy_from(&h.d[k]);
            if k + 1 < n {
                dm.view_mut((15 * k, 15 * k + 15), (15, 15))
                    .copy_from(&h.u[k]);
                dm.view_mut((15 * k + 15, 15 * k), (15, 15))
                    .copy_from(&h.u[k].transpose());
            }
        }
        for p in 0..2u32 {
            let mut uv = DVector::<f64>::zeros(m);
            let mut w = 0.0;
            for c in e.iter().filter(|c| c.plane == p) {
                let u = c.u(&normals[p as usize]);
                for i in 0..9 {
                    uv[15 * c.knot as usize + i] += u[i];
                }
                w += c.w;
            }
            dm -= &uv * uv.transpose() / w;
        }
        let gv = DVector::from_iterator(m, h.g.iter().flat_map(|g| g.iter().copied()));
        let xd = dm.lu().solve(&(-gv)).unwrap();
        for k in 0..n {
            for i in 0..15 {
                assert!(
                    (x[k][i] - xd[15 * k + i]).abs() < 1e-7 * (1.0 + xd[15 * k + i].abs()),
                    "{k} {i}: {} vs {}",
                    x[k][i],
                    xd[15 * k + i]
                );
            }
        }
    }
}
