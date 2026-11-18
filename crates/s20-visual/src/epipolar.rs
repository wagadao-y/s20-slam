//! Matches searched along the epipolar line, without the cloud: for the surfaces the laser missed
//! (too close to it, or seen only from where it could not reach), whose features the guided
//! matching cannot pair, as it predicts them where their rays meet the cloud, behind. A feature's
//! ray, from `near` to a little beyond where it meets the cloud, projects into the other image as a
//! curve; the features of that image within `gate` pixels of it are compared by descriptor, the
//! best kept when clearly better than the second (`ratio`; a lone one when close by itself) and
//! when the other image's feature picks it back. A pair can still be wrong when the right partner
//! lies beyond the searched depths; tracks seen by three images or more can be checked by their
//! residuals, pairs cannot. Only for the image points written beside the COLMAP model: the poses are not
//! adjusted with them.

use crate::anchors::{Pose, V3};
use crate::camera::Fisheye;
use crate::pairs::{Cells, Features};
use rayon::prelude::*;

/// One image's side of a search: its pose, camera and features, with the features by cell.
pub struct Side<'a> {
    pub pose: Pose,
    pub cam: &'a Fisheye,
    pub features: &'a Features,
    pub cells: &'a Cells,
}

const CELL: f64 = Cells::CELL;
/// the descriptor distance a lone candidate must be within (that of the guided matching)
const ALONE: f32 = 250.;
/// depths sampled along a ray (evenly in inverse depth)
const SAMPLES: usize = 64;

fn distance(a: &[u8; 128], b: &[u8; 128]) -> f32 {
    let mut s = 0i32;
    for k in 0..128 {
        let d = a[k] as i32 - b[k] as i32;
        s += d * d;
    }
    (s as f32).sqrt()
}

/// The distance from `p` to the segment `a`-`b` (pixels).
fn to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0. {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0., 1.)
    } else {
        0.
    };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}

/// For every feature of `x`, the feature of `y` it matches along its epipolar curve, if any.
/// `far(i)`: how deep to search feature `i`'s ray (m).
fn one_way(
    x: &Side,
    y: &Side,
    far: &(dyn Fn(usize) -> f64 + Sync),
    near: f64,
    gate: f64,
    ratio: f32,
) -> Vec<Option<usize>> {
    let (cols, rows) = y.cells.size();
    let limit = 80f64.to_radians();
    // (the curve and the features near it in buffers of each thread)
    (0..x.features.uv.len())
        .into_par_iter()
        .map_init(
            || (Vec::with_capacity(SAMPLES), Vec::new()),
            |(curve, near_curve): &mut (Vec<[f64; 2]>, Vec<usize>), i| {
                curve.clear();
                near_curve.clear();
                let ray = x.pose.r * V3::from(x.features.ray[i]);
                let (n, f) = (near, far(i).max(near * 1.5));
                // the ray's points projected into y, from near to far
                for k in 0..SAMPLES {
                    let inv = 1. / n + (1. / f - 1. / n) * k as f64 / (SAMPLES - 1) as f64;
                    let l = y.pose.r.transpose() * (x.pose.p + ray / inv - y.pose.p);
                    if l.z <= 0. || l.xy().norm().atan2(l.z) > limit {
                        continue;
                    }
                    curve.push(y.cam.project([l.x, l.y, l.z]));
                }
                if curve.len() < 2 {
                    return None;
                }
                // the features within `gate` of some piece of the curve, each once
                for w in curve.windows(2) {
                    let (lo_x, hi_x) = (w[0][0].min(w[1][0]) - gate, w[0][0].max(w[1][0]) + gate);
                    let (lo_y, hi_y) = (w[0][1].min(w[1][1]) - gate, w[0][1].max(w[1][1]) + gate);
                    let (c0, c1) = (
                        (lo_x / CELL).floor() as isize,
                        (hi_x / CELL).floor() as isize,
                    );
                    let (r0, r1) = (
                        (lo_y / CELL).floor() as isize,
                        (hi_y / CELL).floor() as isize,
                    );
                    for gy in r0.max(0)..=r1.min(rows as isize - 1) {
                        for gx in c0.max(0)..=c1.min(cols as isize - 1) {
                            for &j in y.cells.at(gx, gy) {
                                let j = j as usize;
                                if to_segment(y.features.uv[j], w[0], w[1]) <= gate {
                                    near_curve.push(j);
                                }
                            }
                        }
                    }
                }
                near_curve.sort_unstable();
                near_curve.dedup();
                let (mut best, mut d1, mut d2) = (usize::MAX, f32::INFINITY, f32::INFINITY);
                for &j in near_curve.iter() {
                    let d = distance(&x.features.desc[i], &y.features.desc[j]);
                    if d < d1 {
                        d2 = d1;
                        d1 = d;
                        best = j;
                    } else if d < d2 {
                        d2 = d;
                    }
                }
                // a lone candidate must be close by itself (as in the guided matching)
                let ok = best != usize::MAX
                    && if d2.is_finite() {
                        d1 < ratio * d2
                    } else {
                        d1 < ALONE
                    };
                ok.then_some(best)
            },
        )
        .collect()
}

/// The features of `a` and `b` matched along their epipolar curves both ways (feature indices).
/// `far_a(i)`, `far_b(j)`: how deep to search the features' rays (m), e.g. a little beyond where
/// they meet the cloud.
pub fn match_epipolar(
    a: &Side,
    b: &Side,
    far_a: &(dyn Fn(usize) -> f64 + Sync),
    far_b: &(dyn Fn(usize) -> f64 + Sync),
    near: f64,
    gate: f64,
    ratio: f32,
) -> Vec<(usize, usize)> {
    let ab = one_way(a, b, far_a, near, gate, ratio);
    let ba = one_way(b, a, far_b, near, gate, ratio);
    ab.iter()
        .enumerate()
        .filter_map(|(i, &j)| {
            let j = j?;
            (ba[j] == Some(i)).then_some((i, j))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn camera() -> Fisheye {
        Fisheye {
            k: [0.; 6],
            a11: 1000.,
            a12: 0.,
            a22: 1000.,
            u0: 1752.,
            v0: 2336.,
        }
    }

    /// Points at 0.6 to 2 m seen from two poses 15 cm apart, each with its own descriptor.
    #[test]
    fn features_of_near_points_match_along_the_curve_and_nothing_else() {
        let cam = camera();
        let pa = Pose {
            r: Rotation3::from_euler_angles(0., 0.02, 0.).into_inner(),
            p: V3::new(0., 0., 0.),
        };
        let pb = Pose {
            r: Rotation3::from_euler_angles(0.01, -0.03, 0.).into_inner(),
            p: V3::new(0.15, 0.02, 0.),
        };
        let mut points = Vec::new();
        for k in 0..200 {
            let t = k as f64;
            points.push(V3::new(
                (t * 0.37).sin() * 0.8,
                (t * 0.23).cos() * 0.6,
                0.6 + (t * 0.11).sin().abs() * 1.4,
            ));
        }
        // a pseudo-random descriptor per point, as different from one another as SIFT's
        let desc = |k: usize| -> [u8; 128] {
            let mut d = [0u8; 128];
            let mut x = (k as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
            for v in d.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *v = (x >> 56) as u8;
            }
            d
        };
        let view = |p: &Pose, order: &dyn Fn(usize) -> usize| -> Features {
            let mut f = Features {
                uv: Vec::new(),
                ray: Vec::new(),
                desc: Vec::new(),
            };
            for k in 0..points.len() {
                let k = order(k);
                let l = p.r.transpose() * (points[k] - p.p);
                let uv = cam.project([l.x, l.y, l.z]);
                f.uv.push(uv);
                f.ray.push(cam.ray(uv));
                f.desc.push(desc(k));
            }
            f
        };
        // b's features in another order: the match must be found, not the index
        let n = points.len();
        let fa = view(&pa, &|k| k);
        let fb = view(&pb, &|k| (k * 7 + 3) % n);
        let (ca, cb) = (Cells::new(&fa), Cells::new(&fb));
        let a = Side {
            pose: pa,
            cam: &cam,
            features: &fa,
            cells: &ca,
        };
        let b = Side {
            pose: pb,
            cam: &cam,
            features: &fb,
            cells: &cb,
        };
        // the cloud behind them all, as if the laser saw only a wall at 3 m
        let far = |_: usize| 3.;
        let m = match_epipolar(&a, &b, &far, &far, 0.3, 3., 0.8);
        assert!(m.len() > 190, "{}", m.len());
        for &(i, j) in &m {
            assert_eq!((j * 7 + 3) % n, i, "a's {i} paired with b's {j}");
        }
        // searched no deeper than 0.5 m, most points (0.6 m and beyond) are out of reach; those
        // found are those whose rays run nearly along the baseline, still with their right partner
        let shallow = |_: usize| 0.5;
        let few = match_epipolar(&a, &b, &shallow, &shallow, 0.3, 3., 0.8);
        assert!(few.len() < m.len() / 4, "{}", few.len());
        for &(i, j) in &few {
            assert_eq!((j * 7 + 3) % n, i);
        }
    }
}
