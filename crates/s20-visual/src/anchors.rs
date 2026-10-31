//! Features anchored on the LiDAR cloud: a sampled cloud with a k-d tree, per-image z-buffers of
//! it, local planes, the ray-plane intersection of a feature and the occlusion test.

use crate::camera::Fisheye;
use anyhow::Result;
use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use std::path::Path;
use std::sync::OnceLock;

pub type V3 = Vector3<f64>;
pub type M3 = Matrix3<f64>;

pub const WIDTH: usize = 3504;
pub const HEIGHT: usize = 4672;
const CELL: usize = 8;

/// Camera pose: camera-to-world rotation and camera centre.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub r: M3,
    pub p: V3,
}

pub struct Cloud {
    pub xyz: Vec<[f64; 3]>,
    pub normal: Vec<[f32; 3]>,
    tree: KdTree<f64, 3>,
    /// each point's local plane, computed once (centre, normal, sigma in f32)
    patches: Vec<OnceLock<Option<[f32; 7]>>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Patch {
    pub centre: V3,
    pub normal: V3,
    pub sigma: f64,
}

impl Cloud {
    /// Every `step`-th point of a LAS/LAZ (by its index in the file) with its normal.
    pub fn load(path: &Path, step: usize) -> Result<Self> {
        let mut r = s20_io::las::LasReader::open(path)?;
        let (mut xyz, mut normal, mut at) = (Vec::new(), Vec::new(), 0usize);
        loop {
            let pts = r.read(1 << 20)?;
            if pts.is_empty() {
                break;
            }
            for q in &pts {
                if at % step == 0 {
                    xyz.push(q.xyz);
                    normal.push(q.normal);
                }
                at += 1;
            }
        }
        Ok(Self::new(xyz, normal))
    }

    pub fn new(xyz: Vec<[f64; 3]>, normal: Vec<[f32; 3]>) -> Self {
        let tree = (&xyz).into();
        let patches = (0..xyz.len()).map(|_| OnceLock::new()).collect();
        Self {
            xyz,
            normal,
            tree,
            patches,
        }
    }

    fn point(&self, i: usize) -> V3 {
        V3::from(self.xyz[i])
    }

    fn within(&self, p: &V3, r: f64) -> Vec<usize> {
        self.tree
            .within_unsorted::<SquaredEuclidean>(&[p.x, p.y, p.z], r * r)
            .into_iter()
            .map(|n| n.item as usize)
            .collect()
    }

    /// The local plane around point `i`: 16 or more neighbours within 12 cm, thinner than 7 mm.
    pub fn patch(&self, i: usize) -> Option<Patch> {
        // the centre is kept relative to the point, so f32 stays exact far from the origin
        let at = self.point(i);
        let p = (*self.patches[i].get_or_init(|| {
            self.fit_patch(i).map(|p| {
                let d = p.centre - at;
                [
                    d.x as f32,
                    d.y as f32,
                    d.z as f32,
                    p.normal.x as f32,
                    p.normal.y as f32,
                    p.normal.z as f32,
                    p.sigma as f32,
                ]
            })
        }))?;
        Some(Patch {
            centre: at + V3::new(p[0] as f64, p[1] as f64, p[2] as f64),
            normal: V3::new(p[3] as f64, p[4] as f64, p[5] as f64).normalize(),
            sigma: p[6] as f64,
        })
    }

    fn fit_patch(&self, i: usize) -> Option<Patch> {
        let n = self.normal[i];
        if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() < 0.5 {
            return None;
        }
        let nb = self.within(&self.point(i), 0.12);
        if nb.len() < 16 {
            return None;
        }
        let c = nb.iter().fold(V3::zeros(), |s, &j| s + self.point(j)) / nb.len() as f64;
        let mut cov = M3::zeros();
        for &j in &nb {
            let d = self.point(j) - c;
            cov += d * d.transpose();
        }
        cov /= (nb.len() - 1) as f64;
        let e = SymmetricEigen::new(cov);
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| e.eigenvalues[a].total_cmp(&e.eigenvalues[b]));
        let (l0, l1) = (e.eigenvalues[order[0]], e.eigenvalues[order[1]]);
        if l0.max(0.).sqrt() > 0.007 || l0 > 0.04 * l1 || l1 < 0.02f64.powi(2) {
            return None;
        }
        Some(Patch {
            centre: c,
            normal: e.eigenvectors.column(order[0]).into_owned(),
            sigma: l0.max(0.).sqrt(),
        })
    }

    /// Points within the cylinder from `centre` to `point` (radius growing with range) that lie off
    /// the anchored plane on the camera side (`visual_surface_occlusion.occluders`).
    pub fn occluders(&self, centre: &V3, point: &V3, normal: &V3) -> usize {
        let (radius, angular, margin, near, step) = (0.02, 0.003, 0.03, 0.3, 0.02);
        let ray = point - centre;
        let length = ray.norm();
        let ray = ray / length;
        let mut hits: Vec<usize> = Vec::new();
        let mut t = near;
        while t < length {
            hits.extend(self.within(&(centre + ray * t), radius + angular * length));
            t += step;
        }
        hits.sort_unstable();
        hits.dedup();
        let side = normal.dot(&(centre - point)).signum();
        hits.iter()
            .filter(|&&i| {
                let q = self.point(i) - centre;
                let along = q.dot(&ray);
                let lateral = (q - ray * along).norm();
                let off = side * (q + centre - point).dot(normal);
                along > near && along < length && lateral < radius + angular * along && off > margin
            })
            .count()
    }
}

/// The nearest cloud point per 8-pixel cell of an image (indices, -1 where none; 1 MB an image, so
/// a recording's grids can be kept).
pub struct ZGrid {
    w: usize,
    h: usize,
    idx: Vec<i32>,
}

impl ZGrid {
    /// The cloud points seen, one per cell.
    pub fn ids(&self) -> Vec<usize> {
        self.idx
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect()
    }
}

pub fn zgrid(cloud: &Cloud, pose: &Pose, cam: &Fisheye) -> ZGrid {
    let (w, h) = (WIDTH.div_ceil(CELL), HEIGHT.div_ceil(CELL));
    let max_theta = 82f64.to_radians();
    let mut hits: Vec<(usize, f64, usize, [f64; 2])> = cloud
        .xyz
        .par_iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let d = V3::from(*x) - pose.p;
            let dist = d.norm();
            if dist > 25. || dist <= 0.3 {
                return None;
            }
            let l = pose.r.transpose() * d;
            if l.xy().norm().atan2(l.z) >= max_theta {
                return None;
            }
            let uv = cam.project([l.x, l.y, l.z]);
            if !(uv[0] >= 0. && uv[0] < WIDTH as f64 && uv[1] >= 0. && uv[1] < HEIGHT as f64) {
                return None;
            }
            let cell = (uv[1] as usize / CELL) * w + uv[0] as usize / CELL;
            Some((cell, dist, i, uv))
        })
        .collect();
    hits.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut idx = vec![-1i32; w * h];
    let mut last = usize::MAX;
    for (cell, _, i, _) in hits {
        if cell != last {
            idx[cell] = i as i32;
            last = cell;
        }
    }
    ZGrid { w, h, idx }
}

/// A feature at `uv` anchored on the cloud: the world point where its ray meets a local plane near
/// it in the z-buffer, that plane, and the depth along the ray.
pub struct Anchor {
    pub point: V3,
    pub centre: V3,
    pub normal: V3,
    pub sigma: f64,
    pub depth: f64,
}

/// The z-buffer's cells within two of the one at `uv` holding a point, nearest projected first
/// (pixel distance, cell).
fn nearest_cells(
    uv: [f64; 2],
    grid: &ZGrid,
    pose: &Pose,
    cam: &Fisheye,
    cloud: &Cloud,
) -> Vec<(f64, usize)> {
    let (cx, cy) = (
        (uv[0] / CELL as f64).floor() as isize,
        (uv[1] / CELL as f64).floor() as isize,
    );
    let mut cells: Vec<(f64, usize)> = Vec::new();
    for dy in -2..=2 {
        for dx in -2..=2 {
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= grid.w as isize || y >= grid.h as isize {
                continue;
            }
            let c = y as usize * grid.w + x as usize;
            if grid.idx[c] >= 0 {
                let l = pose.r.transpose() * (V3::from(cloud.xyz[grid.idx[c] as usize]) - pose.p);
                let p = cam.project([l.x, l.y, l.z]);
                cells.push(((p[0] - uv[0]).hypot(p[1] - uv[1]), c));
            }
        }
    }
    cells.sort_by(|a, b| a.0.total_cmp(&b.0));
    cells
}

/// Where a feature at `uv` meets the cloud without a plane (an edge, clutter): along its ray at
/// the depth of the cloud point projected nearest to it, within two cells. Its normal is zero.
pub fn hint(
    uv: [f64; 2],
    grid: &ZGrid,
    pose: &Pose,
    cam: &Fisheye,
    cloud: &Cloud,
) -> Option<Anchor> {
    let &(d, c) = nearest_cells(uv, grid, pose, cam, cloud).first()?;
    if d > 2. * CELL as f64 {
        return None;
    }
    let q = V3::from(cloud.xyz[grid.idx[c] as usize]);
    let ray = pose.r * V3::from(cam.ray(uv));
    let depth = (q - pose.p).dot(&ray);
    (0.3 < depth && depth < 25.).then(|| Anchor {
        point: pose.p + ray * depth,
        centre: q,
        normal: V3::zeros(),
        sigma: 0.,
        depth,
    })
}

pub fn associate(
    uv: [f64; 2],
    grid: &ZGrid,
    pose: &Pose,
    cam: &Fisheye,
    cloud: &Cloud,
) -> Option<Anchor> {
    let cells = nearest_cells(uv, grid, pose, cam, cloud);
    let ray = pose.r * V3::from(cam.ray(uv));
    for &(_, c) in cells.iter().take(4) {
        let Some(pt) = cloud.patch(grid.idx[c] as usize) else {
            continue;
        };
        let cosine = pt.normal.dot(&ray);
        if cosine.abs() < 0.2 {
            continue;
        }
        let depth = pt.normal.dot(&(pt.centre - pose.p)) / cosine;
        let point = pose.p + ray * depth;
        if !(0.3 < depth && depth < 25.) || (point - pt.centre).norm() > 0.12 {
            continue;
        }
        return Some(Anchor {
            point,
            centre: pt.centre,
            normal: pt.normal,
            sigma: pt.sigma,
            depth,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall() -> Cloud {
        let mut xyz = Vec::new();
        for i in 0..201 {
            for j in 0..201 {
                xyz.push([2., -1. + i as f64 * 0.01, j as f64 * 0.02]);
            }
        }
        let normal = vec![[1f32, 0., 0.]; xyz.len()];
        Cloud::new(xyz, normal)
    }

    #[test]
    fn a_plane_is_a_patch_and_its_own_points_do_not_occlude() {
        let c = wall();
        let p = c.patch(100 * 201 + 100).unwrap();
        assert!(p.sigma < 1e-9 && p.normal.x.abs() > 0.999);
        let (centre, point, n) = (V3::zeros(), V3::new(2., 0., 3.), V3::new(1., 0., 0.));
        assert_eq!(c.occluders(&centre, &point, &n), 0);
        let mut xyz = c.xyz.clone();
        let mut normal = c.normal.clone();
        for k in 0..6 {
            xyz.push([1. + k as f64 * 1e-3, 0., 1.5]);
            normal.push([1., 0., 0.]);
        }
        assert!(Cloud::new(xyz, normal).occluders(&centre, &point, &n) >= 5);
    }

    #[test]
    fn a_feature_ray_meets_the_wall_where_it_looks() {
        let c = wall();
        let cam = Fisheye {
            k: [0.; 6],
            a11: 1000.,
            a12: 0.,
            a22: 1000.,
            u0: 1752.,
            v0: 2336.,
        };
        // camera at the origin looking along +x (camera z = world x)
        let pose = Pose {
            r: M3::new(0., 0., 1., 1., 0., 0., 0., 1., 0.),
            p: V3::new(0., 0., 1.),
        };
        let grid = zgrid(&c, &pose, &cam);
        let target = V3::new(2., 0.3, 1.4);
        let l = pose.r.transpose() * (target - pose.p);
        let a = associate(cam.project([l.x, l.y, l.z]), &grid, &pose, &cam, &c).unwrap();
        assert!((a.point - target).norm() < 1e-6, "{:?}", a.point);
    }
}
