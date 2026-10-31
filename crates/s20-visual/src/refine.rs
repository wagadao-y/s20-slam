//! Per-image pose refinement against the fixed LiDAR cloud:
//! each image's camera pose gets a small world rotation and offset; observations anchored on the
//! cloud tie pairs of images, a prior keeps the corrections small and consecutive images of a
//! camera are tied smoothly. Gauss-Newton with soft-L1 weights; the 6x6-block normal equations
//! are solved by conjugate gradients with the block diagonal as preconditioner. Optionally the two
//! cameras' fisheye angle and principal point, and their mounts, are solved too, shared by every
//! image of a camera.

use crate::anchors::{M3, Pose, V3};
use crate::camera::Fisheye;
use crate::pairs::Obs;
use nalgebra::{Matrix6, Rotation3, SMatrix, Vector6};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// prior on each image's rotation (rad) and offset (m)
    pub rot_sigma: f64,
    pub pos_sigma: f64,
    /// consecutive images of a camera: their corrections differ by about this much
    pub rot_smooth: f64,
    pub pos_smooth: f64,
    /// residual scale (pixel-equivalents) of the soft-L1 loss
    pub pixel: f64,
    pub iterations: usize,
    /// also solve each camera's `Intrinsic`, with these priors (angle scale, cubic term, pixels)
    pub fit_camera: bool,
    pub angle_sigma: f64,
    pub cubic_sigma: f64,
    pub centre_sigma: f64,
    /// also solve each camera's mount, with these priors (rad, m)
    pub fit_mount: bool,
    pub mount_rot_sigma: f64,
    pub mount_pos_sigma: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            rot_sigma: 0.5f64.to_radians(),
            pos_sigma: 0.01,
            rot_smooth: 0.15f64.to_radians(),
            pos_smooth: 0.005,
            pixel: 1.5,
            iterations: 12,
            fit_camera: false,
            angle_sigma: 0.01,
            cubic_sigma: 0.01,
            centre_sigma: 20.,
            fit_mount: false,
            mount_rot_sigma: 2f64.to_radians(),
            mount_pos_sigma: 0.02,
        }
    }
}

/// The per-image corrections (world rotation vector, offset) indexed like the poses.
pub type Correction = Vector6<f64>;

/// A camera's change on top of its calibration, as `visual_stereo_diagnose.change_bearings`: the
/// pixel moves by (`[2]`, `[3]`) before the ray, whose angle from the axis becomes
/// theta (1 + `[0]`) + `[1]` theta^3.
pub type Intrinsic = [f64; 4];

/// The two cameras' changes (left, right): `Intrinsic`, and the mount as a rotation vector and an
/// offset, both in the camera's frame: the pose (R, p) becomes (R exp(w), p + R t).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cameras {
    pub intrinsic: [Intrinsic; 2],
    pub mount: [[f64; 6]; 2],
}

pub fn mounted(pose: &Pose, m: &[f64; 6]) -> Pose {
    if m.iter().all(|v| *v == 0.) {
        return *pose;
    }
    let w = Rotation3::new(V3::new(m[0], m[1], m[2]));
    Pose {
        r: pose.r * w.matrix(),
        p: pose.p + pose.r * V3::new(m[3], m[4], m[5]),
    }
}

pub(crate) fn bend(ray: &V3, dx: &V3, dy: &V3, k: &Intrinsic) -> V3 {
    if k.iter().all(|v| *v == 0.) {
        return *ray;
    }
    let r = (ray + dx * k[2] + dy * k[3]).normalize();
    let rho = r.xy().norm();
    let theta = rho.atan2(r.z);
    let angle = theta + k[0] * theta + k[1] * theta.powi(3);
    let xy = if rho > 1e-12 {
        r.xy() / rho
    } else {
        nalgebra::Vector2::new(1., 0.)
    };
    V3::new(xy.x * angle.sin(), xy.y * angle.sin(), angle.cos())
}

pub fn apply(pose: &Pose, c: &Correction) -> Pose {
    let e = Rotation3::new(c.fixed_rows::<3>(0).into_owned());
    Pose {
        r: e.matrix() * pose.r,
        p: pose.p + c.fixed_rows::<3>(3),
    }
}

/// Precomputed geometry of one observation: the ray in camera a, the measured ray and its tangent
/// basis in camera b, camera b's focal length, both rays' changes per pixel and the two cameras.
pub struct Prepared {
    pub ray_a: V3,
    pub measured: V3,
    pub ta: V3,
    pub tb: V3,
    pub focal: f64,
    pub da: [V3; 2],
    pub db: [V3; 2],
    pub side: [usize; 2],
}

pub(crate) fn ray_and_derivatives(cam: &Fisheye, uv: [f64; 2]) -> (V3, [V3; 2]) {
    let r = |du: f64, dv: f64| V3::from(cam.ray([uv[0] + du, uv[1] + dv]));
    (
        r(0., 0.),
        [
            (r(0.1, 0.) - r(-0.1, 0.)) / 0.2,
            (r(0., 0.1) - r(0., -0.1)) / 0.2,
        ],
    )
}

/// `side_a`, `side_b`: which of `cams` took image a and image b.
pub fn prepare(o: &Obs, cams: &[Fisheye; 2], side_a: usize, side_b: usize) -> Prepared {
    let (ray_a, da) = ray_and_derivatives(&cams[side_a], o.uv_a);
    let (m, db) = ray_and_derivatives(&cams[side_b], o.uv_b);
    let cam_b = &cams[side_b];
    let rho = m.xy().norm();
    let xy = if rho > 1e-12 {
        m.xy() / rho
    } else {
        nalgebra::Vector2::new(1., 0.)
    };
    // d(ray)/d(theta) and the azimuthal direction, as visual_rig_surface_fit.tangent_basis
    let ta = V3::new(xy.x * m.z, xy.y * m.z, -rho);
    let tb = V3::new(-xy.y, xy.x, 0.);
    Prepared {
        ray_a,
        measured: m,
        ta,
        tb,
        focal: cam_b.a11,
        da,
        db,
        side: [side_a, side_b],
    }
}

/// Bearing residual (pixel-equivalents) of an observation with the two images' poses.
pub fn residual(o: &Obs, g: &Prepared, a: &Pose, b: &Pose) -> [f64; 2] {
    residual_with(o, g, a, b, &Cameras::default())
}

/// The same with the cameras changed by `camera`.
pub fn residual_with(o: &Obs, g: &Prepared, a: &Pose, b: &Pose, camera: &Cameras) -> [f64; 2] {
    let (a, b) = (
        &mounted(a, &camera.mount[g.side[0]]),
        &mounted(b, &camera.mount[g.side[1]]),
    );
    let dir = a.r * bend(&g.ray_a, &g.da[0], &g.da[1], &camera.intrinsic[g.side[0]]);
    let depth = o.normal.dot(&(o.world - a.p)) / o.normal.dot(&dir);
    let x = a.p + dir * depth;
    let e = (b.r.transpose() * (x - b.p)).normalize();
    let d = e - bend(
        &g.measured,
        &g.db[0],
        &g.db[1],
        &camera.intrinsic[g.side[1]],
    );
    [d.dot(&g.ta) * g.focal, d.dot(&g.tb) * g.focal]
}

/// Columns: image a's correction, image b's, the left and the right camera's `Intrinsic`, then the
/// left and the right mount.
type J = SMatrix<f64, 2, 32>;

#[allow(clippy::too_many_arguments)]
fn jacobian(
    o: &Obs,
    g: &Prepared,
    a: &Pose,
    b: &Pose,
    ca: &Correction,
    cb: &Correction,
    camera: &Cameras,
    s: &Settings,
) -> ([f64; 2], J) {
    let r0 = residual_with(o, g, &apply(a, ca), &apply(b, cb), camera);
    let mut j = J::zeros();
    for k in 0..32 {
        if (12..20).contains(&k) && !s.fit_camera || k >= 20 && !s.fit_mount {
            continue;
        }
        let (mut pa, mut pb, mut ma, mut mb) = (*ca, *cb, *ca, *cb);
        let (mut pk, mut mk) = (*camera, *camera);
        let h = if k < 12 {
            let h = if k % 6 < 3 { 1e-7 } else { 1e-6 };
            if k < 6 {
                pa[k] += h;
                ma[k] -= h;
            } else {
                pb[k - 6] += h;
                mb[k - 6] -= h;
            }
            h
        } else if k < 20 {
            let (side, i) = ((k - 12) / 4, (k - 12) % 4);
            if !g.side.contains(&side) {
                continue;
            }
            let h = if i < 2 { 1e-7 } else { 1e-4 };
            pk.intrinsic[side][i] += h;
            mk.intrinsic[side][i] -= h;
            h
        } else {
            let (side, i) = ((k - 20) / 6, (k - 20) % 6);
            if !g.side.contains(&side) {
                continue;
            }
            let h = if i < 3 { 1e-7 } else { 1e-6 };
            pk.mount[side][i] += h;
            mk.mount[side][i] -= h;
            h
        };
        let rp = residual_with(o, g, &apply(a, &pa), &apply(b, &pb), &pk);
        let rm = residual_with(o, g, &apply(a, &ma), &apply(b, &mb), &mk);
        j[(0, k)] = (rp[0] - rm[0]) / (2. * h);
        j[(1, k)] = (rp[1] - rm[1]) / (2. * h);
    }
    (r0, j)
}

/// Block-sparse symmetric matrix (upper blocks by image pair).
#[derive(Default)]
pub(crate) struct Blocks {
    pub(crate) diag: HashMap<usize, Matrix6<f64>>,
    off: HashMap<(usize, usize), Matrix6<f64>>,
}

impl Blocks {
    pub(crate) fn add(&mut self, i: usize, j: usize, m: Matrix6<f64>) {
        if i == j {
            *self.diag.entry(i).or_insert_with(Matrix6::zeros) += m;
        } else if i < j {
            *self.off.entry((i, j)).or_insert_with(Matrix6::zeros) += m;
        } else {
            *self.off.entry((j, i)).or_insert_with(Matrix6::zeros) += m.transpose();
        }
    }

    pub(crate) fn merge(mut self, other: Blocks) -> Blocks {
        for (k, m) in other.diag {
            *self.diag.entry(k).or_insert_with(Matrix6::zeros) += m;
        }
        for (k, m) in other.off {
            *self.off.entry(k).or_insert_with(Matrix6::zeros) += m;
        }
        self
    }

    fn mul(&self, x: &[Vector6<f64>]) -> Vec<Vector6<f64>> {
        let mut y = vec![Vector6::zeros(); x.len()];
        for (&i, m) in &self.diag {
            y[i] += m * x[i];
        }
        for (&(i, j), m) in &self.off {
            y[i] += m * x[j];
            y[j] += m.transpose() * x[i];
        }
        y
    }
}

fn dot(a: &[Vector6<f64>], b: &[Vector6<f64>]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.dot(y)).sum()
}

/// Preconditioned conjugate gradients on the block system.
pub(crate) fn pcg(a: &Blocks, b: &[Vector6<f64>], iters: usize, tol: f64) -> Vec<Vector6<f64>> {
    let n = b.len();
    let inv: Vec<Matrix6<f64>> = (0..n)
        .map(|i| {
            a.diag
                .get(&i)
                .and_then(|m| m.try_inverse())
                .unwrap_or_else(Matrix6::identity)
        })
        .collect();
    let mut x = vec![Vector6::zeros(); n];
    let mut r = b.to_vec();
    let mut z: Vec<Vector6<f64>> = r.iter().zip(&inv).map(|(r, m)| m * r).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let b_norm = dot(b, b).sqrt().max(1e-30);
    for _ in 0..iters {
        let ap = a.mul(&p);
        let alpha = rz / dot(&p, &ap).max(1e-300);
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        if dot(&r, &r).sqrt() < tol * b_norm {
            break;
        }
        z = r.iter().zip(&inv).map(|(r, m)| m * r).collect();
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for k in 0..n {
            p[k] = z[k] + beta * p[k];
        }
    }
    x
}

/// Refined corrections for every pose (zero where neither an observation nor the chain reaches an image). `chain` lists
/// consecutive images of a camera to tie; `use_obs` selects the observations used (the others can
/// be held out for checking).
pub fn refine(
    poses: &[Pose],
    obs: &[Obs],
    prepared: &[Prepared],
    use_obs: &[bool],
    chain: &[(usize, usize)],
    s: &Settings,
) -> Vec<Correction> {
    refine_camera(
        poses,
        obs,
        prepared,
        use_obs,
        chain,
        s,
        &mut Cameras::default(),
    )
}

/// `refine` with the cameras changed by `camera`, whose intrinsics (`s.fit_camera`) and mounts
/// (`s.fit_mount`) are solved too: four more blocks after the images, the left and the right
/// intrinsics then the left and the right mount.
pub fn refine_camera(
    poses: &[Pose],
    obs: &[Obs],
    prepared: &[Prepared],
    use_obs: &[bool],
    chain: &[(usize, usize)],
    s: &Settings,
    camera: &mut Cameras,
) -> Vec<Correction> {
    let n = poses.len();
    let mut c = vec![Correction::zeros(); n];
    let prior = Matrix6::from_diagonal(&Vector6::new(
        1. / s.rot_sigma.powi(2),
        1. / s.rot_sigma.powi(2),
        1. / s.rot_sigma.powi(2),
        1. / s.pos_sigma.powi(2),
        1. / s.pos_sigma.powi(2),
        1. / s.pos_sigma.powi(2),
    ));
    let smooth = Matrix6::from_diagonal(&Vector6::new(
        1. / s.rot_smooth.powi(2),
        1. / s.rot_smooth.powi(2),
        1. / s.rot_smooth.powi(2),
        1. / s.pos_smooth.powi(2),
        1. / s.pos_smooth.powi(2),
        1. / s.pos_smooth.powi(2),
    ));
    let camera_prior = Matrix6::from_diagonal(&Vector6::new(
        1. / s.angle_sigma.powi(2),
        1. / s.cubic_sigma.powi(2),
        1. / s.centre_sigma.powi(2),
        1. / s.centre_sigma.powi(2),
        1.,
        1.,
    ));
    let mount_prior = Matrix6::from_diagonal(&Vector6::new(
        1. / s.mount_rot_sigma.powi(2),
        1. / s.mount_rot_sigma.powi(2),
        1. / s.mount_rot_sigma.powi(2),
        1. / s.mount_pos_sigma.powi(2),
        1. / s.mount_pos_sigma.powi(2),
        1. / s.mount_pos_sigma.powi(2),
    ));
    // an image no observation reaches still follows its neighbours along the chain
    let mut active = vec![false; n];
    for (o, u) in obs.iter().zip(use_obs) {
        if *u {
            active[o.a] = true;
            active[o.b] = true;
        }
    }
    for &(i, j) in chain {
        active[i] = true;
        active[j] = true;
    }
    for _ in 0..s.iterations {
        let cam = *camera;
        let (blocks, grad) = obs
            .par_iter()
            .zip(prepared)
            .zip(use_obs)
            .filter(|(_, u)| **u)
            .fold(
                || (Blocks::default(), HashMap::<usize, Vector6<f64>>::new()),
                |(mut bl, mut gr), ((o, g), _)| {
                    let (r, j) =
                        jacobian(o, g, &poses[o.a], &poses[o.b], &c[o.a], &c[o.b], &cam, s);
                    // soft-L1 as reweighted least squares: rho(z) = 2(sqrt(1+z)-1) on z = (r/s)^2
                    let z = (r[0] * r[0] + r[1] * r[1]) / (s.pixel * s.pixel);
                    let w = 1. / (1. + z).sqrt() / (s.pixel * s.pixel);
                    let rv = nalgebra::Vector2::new(r[0], r[1]);
                    let mut parts: Vec<(usize, SMatrix<f64, 2, 6>)> = vec![
                        (o.a, j.fixed_columns::<6>(0).into_owned()),
                        (o.b, j.fixed_columns::<6>(6).into_owned()),
                    ];
                    for side in 0..2 {
                        if !g.side.contains(&side) {
                            continue;
                        }
                        if s.fit_camera {
                            let mut m = SMatrix::<f64, 2, 6>::zeros();
                            m.fixed_columns_mut::<4>(0)
                                .copy_from(&j.fixed_columns::<4>(12 + 4 * side));
                            parts.push((n + side, m));
                        }
                        if s.fit_mount {
                            parts.push((
                                n + 2 + side,
                                j.fixed_columns::<6>(20 + 6 * side).into_owned(),
                            ));
                        }
                    }
                    for (p, (bp, jp)) in parts.iter().enumerate() {
                        for (bq, jq) in &parts[p..] {
                            bl.add(*bp, *bq, w * jp.transpose() * jq);
                        }
                        *gr.entry(*bp).or_insert_with(Vector6::zeros) += w * jp.transpose() * rv;
                    }
                    (bl, gr)
                },
            )
            .reduce(
                || (Blocks::default(), HashMap::new()),
                |(a, mut ga), (b, gb)| {
                    for (k, v) in gb {
                        *ga.entry(k).or_insert_with(Vector6::zeros) += v;
                    }
                    (a.merge(b), ga)
                },
            );
        let mut blocks = blocks;
        let mut g = vec![Vector6::zeros(); n + 4];
        for (k, v) in grad {
            g[k] = v;
        }
        for i in 0..n {
            if active[i] {
                blocks.add(i, i, prior);
                g[i] += prior * c[i];
            } else {
                blocks.add(i, i, Matrix6::identity());
            }
        }
        for side in 0..2 {
            if s.fit_camera {
                let k = &camera.intrinsic[side];
                blocks.add(n + side, n + side, camera_prior);
                g[n + side] += camera_prior * Vector6::new(k[0], k[1], k[2], k[3], 0., 0.);
            } else {
                blocks.add(n + side, n + side, Matrix6::identity());
            }
            if s.fit_mount {
                blocks.add(n + 2 + side, n + 2 + side, mount_prior);
                g[n + 2 + side] += mount_prior * Vector6::from_column_slice(&camera.mount[side]);
            } else {
                blocks.add(n + 2 + side, n + 2 + side, Matrix6::identity());
            }
        }
        for &(i, j) in chain {
            if active[i] && active[j] {
                blocks.add(i, i, smooth);
                blocks.add(j, j, smooth);
                blocks.add(i, j, -smooth);
                let d = smooth * (c[i] - c[j]);
                g[i] += d;
                g[j] -= d;
            }
        }
        let rhs: Vec<Vector6<f64>> = g.iter().map(|v| -v).collect();
        let step = pcg(&blocks, &rhs, 1000, 1e-10);
        let mut largest = 0f64;
        for i in 0..n {
            if active[i] {
                c[i] += step[i];
                largest = largest.max(step[i].fixed_rows::<3>(0).norm());
            }
        }
        for side in 0..2 {
            if s.fit_camera {
                // `k` pairs the parameter with its step
                #[allow(clippy::needless_range_loop)]
                for k in 0..4 {
                    camera.intrinsic[side][k] += step[n + side][k];
                }
                // a pixel moves the ray about 1/focal; compare in the same units as rotations
                largest = largest
                    .max(step[n + side][0].abs())
                    .max(step[n + side][2].abs() * 1e-3);
            }
            if s.fit_mount {
                #[allow(clippy::needless_range_loop)]
                for k in 0..6 {
                    camera.mount[side][k] += step[n + 2 + side][k];
                }
                largest = largest.max(step[n + 2 + side].fixed_rows::<3>(0).norm());
            }
        }
        if largest < 1e-7 {
            break;
        }
    }
    c
}

/// Rotation matrix to the (x, y, z, w) quaternion and the roll-pitch-yaw used in `ImgPose.txt`.
pub fn pose_row(r: &M3) -> ([f64; 3], [f64; 4]) {
    let q = nalgebra::UnitQuaternion::from_matrix(r);
    let rot = Rotation3::from_matrix_unchecked(*r);
    let (roll, pitch, yaw) = rot.euler_angles();
    (
        [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()],
        [q.i, q.j, q.k, q.w],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_rotation_of_one_image_from_anchored_observations() {
        let cam = Fisheye {
            k: [0.; 6],
            a11: 1000.,
            a12: 0.,
            a22: 1000.,
            u0: 1752.,
            v0: 2336.,
        };
        let poses = vec![
            Pose {
                r: M3::identity(),
                p: V3::zeros(),
            },
            Pose {
                r: *Rotation3::from_euler_angles(0., 0.2, 0.).matrix(),
                p: V3::new(0.8, 0., 0.),
            },
        ];
        let truth = Correction::new(0.003, -0.002, 0.001, 0.004, -0.003, 0.002);
        let true_b = apply(&poses[1], &truth);
        let mut obs = Vec::new();
        for i in 0..400 {
            let t = i as f64;
            let world = V3::new(
                (t * 0.37).sin() * 2.,
                (t * 0.53).cos() * 1.5,
                4. + (t * 0.11).sin() * 0.5,
            );
            let normal = V3::new(0., 0., 1.);
            // anchored in image a (true pose), seen in b (true pose)
            let la = poses[0].r.transpose() * (world - poses[0].p);
            let lb = true_b.r.transpose() * (world - true_b.p);
            obs.push(Obs {
                a: 0,
                b: 1,
                fa: 0,
                fb: 0,
                uv_a: cam.project([la.x, la.y, la.z]),
                uv_b: cam.project([lb.x, lb.y, lb.z]),
                world,
                normal,
                depth: la.norm(),
            });
        }
        let prepared: Vec<Prepared> = obs
            .iter()
            .map(|o| prepare(o, &[cam.clone(), cam.clone()], 0, 0))
            .collect();
        let s = Settings {
            rot_sigma: 1.,
            pos_sigma: 1.,
            iterations: 20,
            ..Default::default()
        };
        let c = refine(&poses, &obs, &prepared, &vec![true; obs.len()], &[], &s);
        let rel = apply(&poses[1], &c[1]);
        let rel_a = apply(&poses[0], &c[0]);
        // the relative pose of b with respect to a is recovered
        let want = poses[0].r.transpose() * true_b.r;
        let got = rel_a.r.transpose() * rel.r;
        let ang = Rotation3::from_matrix_unchecked(want.transpose() * got).angle();
        assert!(ang < 1e-4, "{ang}");
        let worst = obs
            .iter()
            .zip(&prepared)
            .map(|(o, g)| {
                let r = residual(o, g, &rel_a, &rel);
                r[0].hypot(r[1])
            })
            .fold(0f64, f64::max);
        assert!(worst < 1e-3, "{worst}");
    }

    #[test]
    fn recovers_a_change_of_the_camera_from_several_images() {
        let cam = Fisheye {
            k: [0.; 6],
            a11: 1000.,
            a12: 0.,
            a22: 1000.,
            u0: 1752.,
            v0: 2336.,
        };
        let truth: Intrinsic = [0.003, -0.002, 4., -3.];
        // the pixel at which the changed camera sees a bearing
        let pixel = |e: V3| {
            let target = e.xy().norm().atan2(e.z);
            let mut t = target;
            for _ in 0..30 {
                t -= (t * (1. + truth[0]) + truth[1] * t.powi(3) - target)
                    / (1. + truth[0] + 3. * truth[1] * t * t);
            }
            let xy = e.xy() / e.xy().norm();
            let uv = cam.project([xy.x * t.sin(), xy.y * t.sin(), t.cos()]);
            [uv[0] - truth[2], uv[1] - truth[3]]
        };
        let poses: Vec<Pose> = (0..8)
            .map(|i| Pose {
                r: *Rotation3::from_euler_angles(0., 0.25 * (i as f64 - 3.5), 0.).matrix(),
                p: V3::new(0.3 * i as f64, 0., 0.),
            })
            .collect();
        let mut obs = Vec::new();
        for a in 0..8 {
            for b in 0..8 {
                if a == b {
                    continue;
                }
                for k in 0..150 {
                    let t = (a * 1000 + b * 150 + k) as f64;
                    let world = V3::new(
                        1. + (t * 0.37).sin() * 4.,
                        (t * 0.53).cos() * 3.,
                        3. + (t * 0.11).sin() * 0.3,
                    );
                    let la = poses[a].r.transpose() * (world - poses[a].p);
                    let lb = poses[b].r.transpose() * (world - poses[b].p);
                    if la.xy().norm().atan2(la.z) > 1.3 || lb.xy().norm().atan2(lb.z) > 1.3 {
                        continue;
                    }
                    obs.push(Obs {
                        a,
                        b,
                        fa: 0,
                        fb: 0,
                        uv_a: pixel(la.normalize()),
                        uv_b: pixel(lb.normalize()),
                        world,
                        normal: V3::new(0., 0., 1.),
                        depth: la.norm(),
                    });
                }
            }
        }
        let prepared: Vec<Prepared> = obs
            .iter()
            .map(|o| prepare(o, &[cam.clone(), cam.clone()], 0, 0))
            .collect();
        let s = Settings {
            fit_camera: true,
            iterations: 30,
            ..Default::default()
        };
        let mut cams = Cameras::default();
        refine_camera(
            &poses,
            &obs,
            &prepared,
            &vec![true; obs.len()],
            &[],
            &s,
            &mut cams,
        );
        let k = cams.intrinsic;
        assert!(
            (k[0][0] - truth[0]).abs() < 2e-4 && (k[0][1] - truth[1]).abs() < 2e-4,
            "{k:?}"
        );
        assert!(
            (k[0][2] - truth[2]).abs() < 0.3 && (k[0][3] - truth[3]).abs() < 0.3,
            "{k:?}"
        );
        assert_eq!(k[1], [0.; 4]);
    }

    #[test]
    fn recovers_a_change_of_a_camera_mount() {
        let cam = Fisheye {
            k: [0.; 6],
            a11: 1000.,
            a12: 0.,
            a22: 1000.,
            u0: 1752.,
            v0: 2336.,
        };
        let truth = [0.004, -0.006, 0.002, 0.003, -0.002, 0.004];
        // left and right images of a device turning and moving; the right camera's mount is off by `truth`
        let mut poses = Vec::new();
        let mut actual = Vec::new();
        for i in 0..10 {
            let t = i as f64;
            let device = Pose {
                r: *Rotation3::from_euler_angles(
                    0.1 * (t * 0.7).sin(),
                    0.3 * (t - 4.5) / 4.5,
                    0.2 * (t * 0.4).cos(),
                )
                .matrix(),
                p: V3::new(0.3 * t, 0.1 * (t * 0.5).sin(), 0.),
            };
            let right = Pose {
                r: device.r,
                p: device.p + device.r * V3::new(0.1, 0., 0.),
            };
            poses.push(device);
            actual.push(device);
            poses.push(right);
            actual.push(mounted(&right, &truth));
        }
        let side = |i: usize| i % 2;
        let mut obs = Vec::new();
        for a in 0..poses.len() {
            for b in 0..poses.len() {
                if a == b {
                    continue;
                }
                for k in 0..120 {
                    let t = (a * 1000 + b * 120 + k) as f64;
                    let world = V3::new(
                        1.4 + (t * 0.37).sin() * 4.,
                        (t * 0.53).cos() * 3.,
                        3. + (t * 0.11).sin() * 0.3,
                    );
                    let la = actual[a].r.transpose() * (world - actual[a].p);
                    let lb = actual[b].r.transpose() * (world - actual[b].p);
                    if la.xy().norm().atan2(la.z) > 1.3 || lb.xy().norm().atan2(lb.z) > 1.3 {
                        continue;
                    }
                    obs.push(Obs {
                        a,
                        b,
                        fa: 0,
                        fb: 0,
                        uv_a: cam.project([la.x, la.y, la.z]),
                        uv_b: cam.project([lb.x, lb.y, lb.z]),
                        world,
                        normal: V3::new(0., 0., 1.),
                        depth: la.norm(),
                    });
                }
            }
        }
        let prepared: Vec<Prepared> = obs
            .iter()
            .map(|o| prepare(o, &[cam.clone(), cam.clone()], side(o.a), side(o.b)))
            .collect();
        let s = Settings {
            fit_mount: true,
            iterations: 30,
            ..Default::default()
        };
        let mut cams = Cameras::default();
        refine_camera(
            &poses,
            &obs,
            &prepared,
            &vec![true; obs.len()],
            &[],
            &s,
            &mut cams,
        );
        // the left mount is the reference: only the difference between the cameras is seen
        let left = Rotation3::new(V3::new(
            cams.mount[0][0],
            cams.mount[0][1],
            cams.mount[0][2],
        ));
        let right = Rotation3::new(V3::new(
            cams.mount[1][0],
            cams.mount[1][1],
            cams.mount[1][2],
        ));
        let want = Rotation3::new(V3::new(truth[0], truth[1], truth[2]));
        let err = (left.inverse() * right * want.inverse()).angle();
        assert!(err < 5e-4, "{cams:?}");
    }
}
