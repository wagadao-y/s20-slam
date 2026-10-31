//! Point clouds with per-point covariances and their preprocessing (small_gicp `preprocess_points`):
//! voxel-grid downsampling (per-voxel mean) and covariance estimation from k nearest neighbours.

use crate::so3::{M3, V3};
use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::SymmetricEigen;
use rayon::prelude::*;

#[derive(Default, Clone)]
pub struct CovCloud {
    pub points: Vec<V3>,
    pub covs: Vec<M3>,
}

impl CovCloud {
    pub fn len(&self) -> usize {
        self.points.len()
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

const COORD_BITS: u32 = 21;
const COORD_MASK: i64 = (1 << COORD_BITS) - 1;
const COORD_OFFSET: i64 = 1 << (COORD_BITS - 1);

#[inline]
fn voxel_key(p: &V3, inv: f64) -> Option<u64> {
    let mut key = 0u64;
    for k in 0..3 {
        let c = (p[k] * inv).floor() as i64 + COORD_OFFSET;
        if !(0..=COORD_MASK).contains(&c) {
            return None;
        }
        key |= (c as u64) << (COORD_BITS * k as u32);
    }
    Some(key)
}

/// Voxel-grid downsampling: one point (the mean) per occupied voxel. Output is ordered by voxel key.
pub fn voxelgrid_sampling(points: &[V3], leaf: f64) -> Vec<V3> {
    let inv = 1.0 / leaf;
    let mut kp: Vec<(u64, u32)> = points
        .par_iter()
        .enumerate()
        .filter_map(|(i, p)| voxel_key(p, inv).map(|k| (k, i as u32)))
        .collect();
    kp.par_sort_unstable();
    let mut out = Vec::new();
    let mut i = 0;
    while i < kp.len() {
        let k = kp[i].0;
        let mut s = V3::zeros();
        let mut n = 0;
        // small_gicp (OMP) sums in blocks of 1024 sorted points: a voxel crossing a block border yields two points
        while i < kp.len() && kp[i].0 == k && (n == 0 || i % 1024 != 0) {
            s += points[kp[i].1 as usize];
            n += 1;
            i += 1;
        }
        out.push(s / n as f64);
    }
    out
}

/// Covariances from `k` nearest neighbours, regularised to eigenvalues (1e-3, 1, 1) like small_gicp.
/// Points with fewer than 5 neighbours get the identity.
pub fn estimate_covariances(points: &[V3], k: usize) -> Vec<M3> {
    if points.is_empty() {
        return Vec::new();
    }
    let pts: Vec<[f64; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let tree: KdTree<f64, 3> = KdTree::from(&pts);
    pts.par_iter()
        .map(|q| {
            let nn = tree.nearest_n::<SquaredEuclidean>(q, k);
            if nn.len() < 5 {
                return M3::identity();
            }
            let n = nn.len() as f64;
            let mut s = V3::zeros();
            let mut c = M3::zeros();
            for m in &nn {
                let p = points[m.item as usize];
                s += p;
                c += p * p.transpose();
            }
            let mean = s / n;
            let cov = (c - mean * s.transpose()) / n;
            regularize(&cov)
        })
        .collect()
}

/// Replace the eigenvalues of a covariance by (1e-3, 1, 1) (smallest -> 1e-3).
pub fn regularize(cov: &M3) -> M3 {
    let e = SymmetricEigen::new(*cov);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| e.eigenvalues[a].total_cmp(&e.eigenvalues[b]));
    let v0 = e.eigenvectors.column(idx[0]).into_owned();
    // V diag(1e-3,1,1) V^T = I - (1 - 1e-3) v0 v0^T
    M3::identity() - (1.0 - 1e-3) * v0 * v0.transpose()
}

/// Downsample and estimate covariances (small_gicp `preprocess_points`).
pub fn preprocess(points: &[V3], leaf: f64, k: usize) -> CovCloud {
    let p = voxelgrid_sampling(points, leaf);
    let covs = estimate_covariances(&p, k);
    CovCloud { points: p, covs }
}
