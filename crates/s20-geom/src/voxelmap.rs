//! Incremental Gaussian voxel map (small_gicp `GaussianVoxelMap` = `IncrementalVoxelMap<GaussianVoxel>`).
//!
//! Each voxel keeps the mean and covariance of the points (and of their transformed covariances) inserted
//! into it. Voxels not touched for `lru_horizon` insertions are removed every `lru_clear_cycle` insertions.
//! Nearest-neighbour search only looks at the voxel containing the query (search_offsets = 1).

use crate::cloud::CovCloud;
use crate::so3::{M3, V3};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Small multiplicative hasher for voxel coordinates.
#[derive(Default)]
pub struct CoordHasher(u64);
impl Hasher for CoordHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    fn write_i32(&mut self, i: i32) {
        self.0 = (self.0.rotate_left(5) ^ (i as u32 as u64)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}
pub type CoordMap<V> = HashMap<[i32; 3], V, BuildHasherDefault<CoordHasher>>;

#[derive(Clone)]
pub struct GaussianVoxel {
    pub coord: [i32; 3],
    pub lru: usize,
    pub num_points: usize,
    sum_mean: V3,
    sum_cov: M3,
    pub mean: V3,
    pub cov: M3,
}

#[derive(Clone)]
pub struct GaussianVoxelMap {
    pub inv_leaf: f64,
    pub lru_horizon: usize,
    pub lru_clear_cycle: usize,
    pub lru_counter: usize,
    pub voxels: Vec<GaussianVoxel>,
    index: CoordMap<usize>,
}

#[inline]
pub fn coord_of(p: &V3, inv_leaf: f64) -> [i32; 3] {
    [
        (p.x * inv_leaf).floor() as i32,
        (p.y * inv_leaf).floor() as i32,
        (p.z * inv_leaf).floor() as i32,
    ]
}

impl GaussianVoxelMap {
    pub fn new(leaf: f64) -> Self {
        GaussianVoxelMap {
            inv_leaf: 1.0 / leaf,
            lru_horizon: 100,
            lru_clear_cycle: 10,
            lru_counter: 0,
            voxels: Vec::new(),
            index: CoordMap::default(),
        }
    }

    pub fn set_lru(&mut self, horizon: usize, clear_cycle: usize) {
        self.lru_horizon = horizon;
        self.lru_clear_cycle = clear_cycle.max(1);
    }

    pub fn len(&self) -> usize {
        self.voxels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Insert a cloud transformed by (R, t).
    pub fn insert(&mut self, cloud: &CovCloud, r: &M3, t: &V3) {
        let mut touched = Vec::with_capacity(cloud.len());
        let rt = r.transpose();
        for (p, c) in cloud.points.iter().zip(&cloud.covs) {
            let q = r * p + t;
            let coord = coord_of(&q, self.inv_leaf);
            let n = self.voxels.len();
            let vi = *self.index.entry(coord).or_insert(n);
            if vi == n {
                self.voxels.push(GaussianVoxel {
                    coord,
                    lru: self.lru_counter,
                    num_points: 0,
                    sum_mean: V3::zeros(),
                    sum_cov: M3::zeros(),
                    mean: V3::zeros(),
                    cov: M3::zeros(),
                });
            }
            let v = &mut self.voxels[vi];
            v.lru = self.lru_counter;
            v.num_points += 1;
            v.sum_mean += q;
            v.sum_cov += r * c * rt;
            touched.push(vi);
        }
        for vi in touched {
            let v = &mut self.voxels[vi];
            let n = v.num_points as f64;
            v.mean = v.sum_mean / n;
            v.cov = v.sum_cov / n;
        }
        self.lru_counter += 1;
        if self.lru_counter.is_multiple_of(self.lru_clear_cycle) {
            let (h, c) = (self.lru_horizon, self.lru_counter);
            self.voxels.retain(|v| v.lru + h >= c);
            self.index.clear();
            for (i, v) in self.voxels.iter().enumerate() {
                self.index.insert(v.coord, i);
            }
        }
    }

    /// Voxel containing `q` and the squared distance to its mean.
    #[inline]
    pub fn nearest(&self, q: &V3) -> Option<(&GaussianVoxel, f64)> {
        let vi = *self.index.get(&coord_of(q, self.inv_leaf))?;
        let v = &self.voxels[vi];
        Some((v, (v.mean - q).norm_squared()))
    }

    /// Voxel whose mean is nearest to `q` among the one containing it and its 6 face neighbours
    /// (small_gicp `search_offsets = 7`).
    #[inline]
    pub fn nearest7(&self, q: &V3) -> Option<(&GaussianVoxel, f64)> {
        const OFF: [[i32; 3]; 7] = [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ];
        let c = coord_of(q, self.inv_leaf);
        let mut best: Option<(&GaussianVoxel, f64)> = None;
        for o in OFF {
            let Some(&vi) = self.index.get(&[c[0] + o[0], c[1] + o[1], c[2] + o[2]]) else {
                continue;
            };
            let v = &self.voxels[vi];
            let d2 = (v.mean - q).norm_squared();
            if best.is_none_or(|b| d2 < b.1) {
                best = Some((v, d2));
            }
        }
        best
    }
}
