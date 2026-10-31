//! Voxelized GICP linearisation against a Gaussian voxel map (small_gicp `GICPFactor` + `DistanceRejector`).
//!
//! The pose T = (R, t) maps source to target. The perturbation is on the right: T * exp([omega, v]),
//! rotation first. For each source point p with a voxel whose mean is within `max_corr`:
//! r = mu - T p, C = Sigma_voxel + R Sigma_p R^T, M = C^-1, J = [R [p]x, -R],
//! H += J^T M J, b += J^T M r, e += 0.5 r^T M r.

use crate::cloud::CovCloud;
use crate::so3::{M3, V3, skew};
use crate::voxelmap::GaussianVoxelMap;
use nalgebra::{Matrix3x6, Matrix6, Vector6};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Linearized {
    pub h: Matrix6<f64>,
    pub b: Vector6<f64>,
    pub e: f64,
    pub inliers: usize,
}

impl Linearized {
    fn zero() -> Self {
        Linearized {
            h: Matrix6::zeros(),
            b: Vector6::zeros(),
            e: 0.0,
            inliers: 0,
        }
    }
    fn add(mut self, o: Self) -> Self {
        self.h += o.h;
        self.b += o.b;
        self.e += o.e;
        self.inliers += o.inliers;
        self
    }
}

pub fn linearize(
    map: &GaussianVoxelMap,
    src: &CovCloud,
    r: &M3,
    t: &V3,
    max_corr: f64,
) -> Linearized {
    linearize_opts(map, src, r, t, max_corr, &Opts::default())
}

/// Options of [`linearize_opts`]; the defaults are small_gicp's.
#[derive(Clone, Copy, Debug, Default)]
pub struct Opts {
    /// search the 6 face neighbours of the voxel too (small_gicp `search_offsets = 7`)
    pub search7: bool,
    /// Huber threshold on the Mahalanobis residual sqrt(r^T M r); 0: plain least squares
    pub huber: f64,
}

pub fn linearize_opts(
    map: &GaussianVoxelMap,
    src: &CovCloud,
    r: &M3,
    t: &V3,
    max_corr: f64,
    opts: &Opts,
) -> Linearized {
    let max_sq = max_corr * max_corr;
    let rt = r.transpose();
    // fixed chunks summed in order: the same sums whatever the thread scheduling
    const CHUNK: usize = 512;
    let parts: Vec<Linearized> = src
        .points
        .par_chunks(CHUNK)
        .zip(src.covs.par_chunks(CHUNK))
        .map(|(ps, cs)| {
            ps.iter()
                .zip(cs)
                .fold(Linearized::zero(), |mut acc, (p, c)| {
                    let q = r * p + t;
                    let found = if opts.search7 {
                        map.nearest7(&q)
                    } else {
                        map.nearest(&q)
                    };
                    let Some((v, d2)) = found else {
                        return acc;
                    };
                    if d2 > max_sq {
                        return acc;
                    }
                    let rcr = v.cov + r * c * rt;
                    let Some(m) = rcr.try_inverse() else {
                        return acc;
                    };
                    let res = v.mean - q;
                    let mut j = Matrix3x6::<f64>::zeros();
                    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&(r * skew(p)));
                    j.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-r));
                    let m2 = res.dot(&(m * res));
                    let w = if opts.huber > 0.0 && m2 > opts.huber * opts.huber {
                        opts.huber / m2.sqrt()
                    } else {
                        1.0
                    };
                    let jtm = j.transpose() * m * w;
                    acc.h += jtm * j;
                    acc.b += jtm * res;
                    acc.e += 0.5 * w * m2;
                    acc.inliers += 1;
                    acc
                })
        })
        .collect();
    parts.into_iter().fold(Linearized::zero(), Linearized::add)
}

/// Full Gauss-Newton alignment (small_gicp `align` with the default optimizer), for tests and tools.
pub fn align(
    map: &GaussianVoxelMap,
    src: &CovCloud,
    r0: &M3,
    t0: &V3,
    max_corr: f64,
    max_iter: usize,
) -> (M3, V3, Linearized) {
    let (mut r, mut t) = (*r0, *t0);
    let mut lin = Linearized::zero();
    for _ in 0..max_iter {
        lin = linearize(map, src, &r, &t, max_corr);
        let a = lin.h + Matrix6::identity() * 1e-6;
        let Some(d) = a.cholesky().map(|c| c.solve(&(-lin.b))) else {
            break;
        };
        let (dr, dt) = crate::so3::se3_exp(&[d[0], d[1], d[2], d[3], d[4], d[5]]);
        t += r * dt;
        r *= dr;
        if d.fixed_rows::<3>(0).norm() < 0.1f64.to_radians() && d.fixed_rows::<3>(3).norm() < 1e-3 {
            break;
        }
    }
    (r, t, lin)
}
