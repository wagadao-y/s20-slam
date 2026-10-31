//! Tracks across images and their bundle adjustment, after `refine`: the guided matches of every
//! pair joined by feature into tracks, each a free 3-D point seen by its images and held near the
//! local LiDAR plane its features were anchored on. Every frame (the images taken together) gets a
//! small correction shared by its images, with a weak prior and smoothness along the frames, so
//! the images with few tracks follow the plane-held points and their neighbours rather than drift.
//! Levenberg-Marquardt with the points eliminated (Schur
//! complement); the frames' block system by `refine`'s conjugate gradients.

use crate::anchors::{M3, Pose, V3};
use crate::camera::Fisheye;
use crate::pairs::Obs;
use crate::refine::{Blocks, Intrinsic, bend, pcg, ray_and_derivatives};
use nalgebra::{Matrix3, Matrix6, Rotation3, SMatrix, SymmetricEigen, Vector2, Vector3, Vector6};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// soft-L1 scale of the bearing residual (pixel-equivalents)
    pub pixel: f64,
    /// a point's distance to its plane (m), and the soft-L1 scale of that residual in these units
    pub plane_sigma: f64,
    pub plane_scale: f64,
    /// prior on each frame's correction (rad, m)
    pub rot_sigma: f64,
    pub pos_sigma: f64,
    /// consecutive frames' corrections differ by about this much (rad, m)
    pub rot_smooth: f64,
    pub pos_smooth: f64,
    pub iterations: usize,
    /// sightings missing by more than this (pixel-equivalents) are dropped after the third and the
    /// sixth step
    pub outlier: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            pixel: 1.5,
            plane_sigma: 0.005,
            plane_scale: 2.,
            rot_sigma: 1f64.to_radians(),
            pos_sigma: 0.05,
            rot_smooth: 0.5f64.to_radians(),
            pos_smooth: 0.02,
            iterations: 15,
            outlier: 4.,
        }
    }
}

/// One image's sighting of a track: the image, the measured bearing in its camera and the
/// bearing's tangent basis (along the angle from the axis and around it), with the focal length
/// that turns them into pixel-equivalents.
#[derive(Clone, Debug)]
pub struct Sighting {
    pub image: usize,
    pub bearing: V3,
    ta: V3,
    tb: V3,
    focal: f64,
}

impl Sighting {
    /// `k`: the camera's change solved by `refine` on top of `cam`.
    pub fn new(image: usize, uv: [f64; 2], cam: &Fisheye, k: &Intrinsic) -> Self {
        let (ray, d) = ray_and_derivatives(cam, uv);
        let m = bend(&ray, &d[0], &d[1], k);
        let rho = m.xy().norm();
        let xy = if rho > 1e-12 {
            m.xy() / rho
        } else {
            Vector2::new(1., 0.)
        };
        Self {
            image,
            bearing: m,
            ta: V3::new(xy.x * m.z, xy.y * m.z, -rho),
            tb: V3::new(-xy.y, xy.x, 0.),
            focal: cam.a11,
        }
    }

    /// Bearing residual (pixel-equivalents) of the point `x` seen from `pose`, and the camera-frame
    /// vector to it.
    fn residual(&self, pose: &Pose, x: &V3) -> (Vector2<f64>, V3) {
        let v = pose.r.transpose() * (x - pose.p);
        let d = v.normalize() - self.bearing;
        (
            Vector2::new(d.dot(&self.ta) * self.focal, d.dot(&self.tb) * self.focal),
            v,
        )
    }
}

#[derive(Clone, Debug)]
pub struct Track {
    pub sightings: Vec<Sighting>,
    /// where its features met the cloud (the median of their anchors) and that plane's normal (zero
    /// when none of them met a plane: the point is then free)
    pub anchor: V3,
    pub normal: V3,
    /// held out of the adjustment, for the check
    pub held: bool,
}

impl Track {
    pub fn planar(&self) -> bool {
        self.normal.norm_squared() > 0.5
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Tracks of the observations' features: two features matched in a pair are the same point.
/// Tracks holding two features of one image are dropped (a wrong match joined them). `sides`: the
/// camera of each image; `intrinsic`: the cameras' changes solved by `refine`; a deterministic
/// share `holdout` of the tracks is held out.
pub fn tracks(
    obs: &[Obs],
    cams: &[Fisheye; 2],
    intrinsic: &[Intrinsic; 2],
    sides: &[usize],
    holdout: f64,
) -> Vec<Track> {
    let mut id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, [f64; 2])> = Vec::new();
    let mut anchor: Vec<Option<(V3, V3)>> = Vec::new();
    let mut parent: Vec<usize> = Vec::new();
    let mut node = |image: usize, feature: usize, uv: [f64; 2]| -> usize {
        *id.entry((image, feature)).or_insert_with(|| {
            nodes.push((image, uv));
            anchor.push(None);
            parent.push(parent.len());
            parent.len() - 1
        })
    };
    let mut edges = Vec::with_capacity(obs.len());
    for o in obs {
        let (a, b) = (node(o.a, o.fa, o.uv_a), node(o.b, o.fb, o.uv_b));
        edges.push((a, b, o.world, o.normal));
    }
    for &(a, b, world, normal) in &edges {
        // the anchor is that of image a's feature (its ray met the plane)
        if anchor[a].is_none() {
            anchor[a] = Some((world, normal));
        }
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    // groups in the order of their first node, so the tracks do not depend on hashing
    let mut group: HashMap<usize, usize> = HashMap::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for k in 0..nodes.len() {
        let r = find(&mut parent, k);
        let g = *group.entry(r).or_insert_with(|| {
            members.push(Vec::new());
            members.len() - 1
        });
        members[g].push(k);
    }
    members
        .iter()
        .enumerate()
        .filter_map(|(g, m)| {
            let mut images: Vec<usize> = m.iter().map(|&k| nodes[k].0).collect();
            images.sort_unstable();
            if images.len() < 2 || images.windows(2).any(|w| w[0] == w[1]) {
                return None;
            }
            // the anchors on planes, else those placed without one
            let all: Vec<(V3, V3)> = m.iter().filter_map(|&k| anchor[k]).collect();
            let planar: Vec<(V3, V3)> = all
                .iter()
                .copied()
                .filter(|a| a.1.norm_squared() > 0.5)
                .collect();
            let anchors = if planar.is_empty() { all } else { planar };
            if anchors.is_empty() {
                return None;
            }
            let mid = |axis: usize| {
                let mut v: Vec<f64> = anchors.iter().map(|a| a.0[axis]).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let centre = V3::new(mid(0), mid(1), mid(2));
            let nearest = anchors
                .iter()
                .min_by(|a, b| (a.0 - centre).norm().total_cmp(&(b.0 - centre).norm()))
                .unwrap();
            let held = (((g as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64
                / (1u64 << 24) as f64)
                < holdout;
            Some(Track {
                sightings: m
                    .iter()
                    .map(|&k| {
                        let (image, uv) = nodes[k];
                        Sighting::new(image, uv, &cams[sides[image]], &intrinsic[sides[image]])
                    })
                    .collect(),
                anchor: nearest.0,
                normal: nearest.1,
                held,
            })
        })
        .collect()
}

/// The point of the sightings' rays from `poses` nearest to them all (none when the rays are
/// nearly parallel).
pub fn triangulate(t: &Track, poses: &[Pose]) -> Option<V3> {
    let (mut a, mut b) = (M3::zeros(), V3::zeros());
    for s in &t.sightings {
        let p = &poses[s.image];
        let d = p.r * s.bearing;
        let m = M3::identity() - d * d.transpose();
        a += m;
        b += m * p.p;
    }
    let n = t.sightings.len() as f64;
    let e = SymmetricEigen::new(a);
    if e.eigenvalues.min() <= 1e-3 * n {
        return None;
    }
    a.try_inverse().map(|i| i * b)
}

/// The held-out tracks (three sightings or more) triangulated with `poses`: every sighting's
/// bearing residual (pixel-equivalents) and every point's distance to its plane (m; of the tracks
/// on a plane).
pub fn check(tracks: &[Track], poses: &[Pose]) -> (Vec<f64>, Vec<f64>) {
    let per: Vec<(Vec<f64>, Option<f64>)> = tracks
        .par_iter()
        .filter(|t| t.held && t.sightings.len() >= 3)
        .filter_map(|t| {
            let x = triangulate(t, poses)?;
            let r = t
                .sightings
                .iter()
                .map(|s| s.residual(&poses[s.image], &x).0.norm())
                .collect();
            Some((r, t.planar().then(|| t.normal.dot(&(x - t.anchor)).abs())))
        })
        .collect();
    let mut bearing = Vec::new();
    let mut plane = Vec::new();
    for (r, d) in per {
        bearing.extend(r);
        plane.extend(d);
    }
    (bearing, plane)
}

/// Each frame's correction: the world rotation applied to its images' rotations, and the shift of
/// their centres.
#[derive(Clone, Debug)]
pub struct Solution {
    pub rot: Vec<M3>,
    pub shift: Vec<V3>,
    pub iterations: usize,
    pub cost: [f64; 2],
}

impl Solution {
    pub fn pose(&self, frame: usize, p: &Pose) -> Pose {
        Pose {
            r: self.rot[frame] * p.r,
            p: p.p + self.shift[frame],
        }
    }
}

fn skew(v: &V3) -> M3 {
    M3::new(0., -v.z, v.y, v.z, 0., -v.x, -v.y, v.x, 0.)
}

fn rotvec(m: &M3) -> V3 {
    Rotation3::from_matrix_unchecked(*m).scaled_axis()
}

/// soft-L1 on z = (r / scale)^2: the cost 2 (sqrt(1 + z) - 1) and the reweighting 1 / sqrt(1 + z)
fn soft_l1(z: f64) -> (f64, f64) {
    let s = (1. + z).sqrt();
    (2. * (s - 1.), 1. / s)
}

/// A frame's block and gradient from one track.
type FrameBlocks = Vec<(usize, Matrix6<f64>, Vector6<f64>)>;

/// A track linearised: the inverse of its point block (damped), the point's gradient and, per frame
/// that sees it, the frame-point block.
struct Linear {
    hinv: Matrix3<f64>,
    gp: Vector3<f64>,
    w: Vec<(usize, SMatrix<f64, 6, 3>)>,
}

struct Problem<'a> {
    poses: &'a [Pose],
    frame: &'a [usize],
    frames: usize,
    chain: &'a [(usize, usize)],
    tracks: Vec<Track>,
    s: &'a Settings,
}

/// A track to adjust: on a plane and seen twice, or free and seen three times or more.
fn usable(t: &Track) -> bool {
    t.sightings.len() >= if t.planar() { 2 } else { 3 }
}

impl Problem<'_> {
    fn pose(&self, rot: &[M3], shift: &[V3], image: usize) -> Pose {
        let f = self.frame[image];
        Pose {
            r: rot[f] * self.poses[image].r,
            p: self.poses[image].p + shift[f],
        }
    }

    fn track_cost(&self, t: &Track, x: &V3, rot: &[M3], shift: &[V3]) -> f64 {
        let s = self.s;
        let mut c = 0.;
        for sg in &t.sightings {
            let (r, _) = sg.residual(&self.pose(rot, shift, sg.image), x);
            c += soft_l1(r.norm_squared() / (s.pixel * s.pixel)).0;
        }
        if t.planar() {
            let rp = t.normal.dot(&(x - t.anchor)) / s.plane_sigma;
            c += soft_l1(rp * rp / (s.plane_scale * s.plane_scale)).0;
        }
        c
    }

    /// Drops the sightings missing by more than `outlier`, then the tracks no longer usable;
    /// returns the sightings dropped.
    fn prune(&mut self, x: &mut Vec<V3>, rot: &[M3], shift: &[V3]) -> usize {
        let tracks = std::mem::take(&mut self.tracks);
        let points = std::mem::take(x);
        let mut dropped = 0;
        for (mut t, p) in tracks.into_iter().zip(points) {
            let n = t.sightings.len();
            t.sightings.retain(|sg| {
                sg.residual(&self.pose(rot, shift, sg.image), &p).0.norm() <= self.s.outlier
            });
            dropped += n - t.sightings.len();
            if usable(&t) {
                self.tracks.push(t);
                x.push(p);
            }
        }
        dropped
    }

    fn prior_cost(&self, rot: &[M3], shift: &[V3]) -> f64 {
        let s = self.s;
        let w: Vec<V3> = rot.iter().map(rotvec).collect();
        let mut c: f64 = (0..self.frames)
            .map(|f| {
                w[f].norm_squared() / s.rot_sigma.powi(2)
                    + shift[f].norm_squared() / s.pos_sigma.powi(2)
            })
            .sum();
        for &(f, g) in self.chain {
            c += (w[f] - w[g]).norm_squared() / s.rot_smooth.powi(2)
                + (shift[f] - shift[g]).norm_squared() / s.pos_smooth.powi(2);
        }
        c
    }

    fn cost(&self, x: &[V3], rot: &[M3], shift: &[V3]) -> f64 {
        let tracks: f64 = self
            .tracks
            .par_iter()
            .zip(x)
            .map(|(t, x)| self.track_cost(t, x, rot, shift))
            .sum();
        tracks + self.prior_cost(rot, shift)
    }

    /// The point block, the frames' blocks and gradients, and the frame-point blocks of a track.
    fn linearise(
        &self,
        t: &Track,
        x: &V3,
        rot: &[M3],
        shift: &[V3],
        lambda: f64,
    ) -> Option<(FrameBlocks, Linear)> {
        let s = self.s;
        let mut hpp = Matrix3::zeros();
        let mut gp = Vector3::zeros();
        // per frame: its block, gradient and frame-point block
        #[allow(clippy::type_complexity)]
        let mut per: Vec<(usize, Matrix6<f64>, Vector6<f64>, SMatrix<f64, 6, 3>)> = Vec::new();
        for sg in &t.sightings {
            let pose = self.pose(rot, shift, sg.image);
            let (r, v) = sg.residual(&pose, x);
            let n = v.norm();
            if n < 1e-9 {
                continue;
            }
            let e = v / n;
            let w = soft_l1(r.norm_squared() / (s.pixel * s.pixel)).1 / (s.pixel * s.pixel);
            let proj = (M3::identity() - e * e.transpose()) / n;
            let rows = SMatrix::<f64, 2, 3>::from_rows(&[
                (sg.ta * sg.focal).transpose(),
                (sg.tb * sg.focal).transpose(),
            ]);
            // v = R^T exp(-dw) (x - p - dt): d/dx = R^T, d/dt = -R^T, d/dw = R^T [x - p]x
            let jx = rows * proj * pose.r.transpose();
            let jw = jx * skew(&(x - pose.p));
            let mut jc = SMatrix::<f64, 2, 6>::zeros();
            jc.fixed_columns_mut::<3>(0).copy_from(&jw);
            jc.fixed_columns_mut::<3>(3).copy_from(&(-jx));
            hpp += w * jx.transpose() * jx;
            gp += w * jx.transpose() * r;
            let f = self.frame[sg.image];
            let (h, g, wf) = (
                w * jc.transpose() * jc,
                w * jc.transpose() * r,
                w * jc.transpose() * jx,
            );
            match per.iter_mut().find(|q| q.0 == f) {
                Some(q) => {
                    q.1 += h;
                    q.2 += g;
                    q.3 += wf;
                }
                None => per.push((f, h, g, wf)),
            }
        }
        if t.planar() {
            let rp = t.normal.dot(&(x - t.anchor)) / s.plane_sigma;
            let wp = soft_l1(rp * rp / (s.plane_scale * s.plane_scale)).1
                / (s.plane_scale * s.plane_scale);
            let jp = t.normal / s.plane_sigma;
            hpp += wp * jp * jp.transpose();
            gp += wp * jp * rp;
        }
        let damped =
            hpp + Matrix3::from_diagonal(&hpp.diagonal()) * lambda + Matrix3::identity() * 1e-12;
        let hinv = damped.try_inverse()?;
        let blocks = per.iter().map(|q| (q.0, q.1, q.2)).collect();
        let w = per.into_iter().map(|q| (q.0, q.3)).collect();
        Some((blocks, Linear { hinv, gp, w }))
    }
}

/// Adjusts the frames' corrections and the (not held-out) tracks' points, starting from `poses`
/// (each image's pose, the cameras' mounts applied). `frame`: each image's frame among `frames`;
/// `chain`: consecutive frames, tied smoothly.
pub fn solve(
    poses: &[Pose],
    frame: &[usize],
    frames: usize,
    chain: &[(usize, usize)],
    tracks: &[Track],
    s: &Settings,
    log: &dyn Fn(&str),
) -> Solution {
    let mut pb = Problem {
        poses,
        frame,
        frames,
        chain,
        tracks: tracks
            .iter()
            .filter(|t| !t.held && usable(t))
            .cloned()
            .collect(),
        s,
    };
    log(&format!(
        "tracks: {} to adjust ({} on planes) of {}",
        pb.tracks.len(),
        pb.tracks.iter().filter(|t| t.planar()).count(),
        tracks.len()
    ));
    let mut x: Vec<V3> = pb.tracks.iter().map(|t| t.anchor).collect();
    let mut rot = vec![M3::identity(); frames];
    let mut shift = vec![V3::zeros(); frames];
    let rot_prior = 1. / s.rot_sigma.powi(2);
    let pos_prior = 1. / s.pos_sigma.powi(2);
    let rot_smooth = 1. / s.rot_smooth.powi(2);
    let pos_smooth = 1. / s.pos_smooth.powi(2);
    let diag6 = |r: f64, p: f64| Matrix6::from_diagonal(&Vector6::new(r, r, r, p, p, p));
    let mut cost = pb.cost(&x, &rot, &shift);
    let start = cost;
    let mut lambda = 1e-4;
    let mut iterations = 0;
    let mut accepted = 0;
    while iterations < s.iterations {
        iterations += 1;
        let lin: Vec<Option<(FrameBlocks, Linear)>> = pb
            .tracks
            .par_iter()
            .zip(&x)
            .map(|(t, x)| pb.linearise(t, x, &rot, &shift, lambda))
            .collect();
        // the points eliminated: H_ff - W H_pp^-1 W^T and g_f - W H_pp^-1 g_p
        let (mut blocks, mut g) = lin
            .par_iter()
            .flatten()
            .fold(
                || (Blocks::default(), vec![Vector6::zeros(); frames]),
                |(mut b, mut g), (hf, l)| {
                    for (f, h, gf) in hf {
                        b.add(*f, *f, *h);
                        g[*f] += gf;
                    }
                    for (k, (fa, wa)) in l.w.iter().enumerate() {
                        let wh = wa * l.hinv;
                        g[*fa] -= wh * l.gp;
                        for (fb, wb) in &l.w[k..] {
                            b.add(*fa, *fb, -(wh * wb.transpose()));
                        }
                    }
                    (b, g)
                },
            )
            .reduce(
                || (Blocks::default(), vec![Vector6::zeros(); frames]),
                |(a, mut ga), (b, gb)| {
                    for (x, y) in ga.iter_mut().zip(gb) {
                        *x += y;
                    }
                    (a.merge(b), ga)
                },
            );
        let w: Vec<V3> = rot.iter().map(rotvec).collect();
        for f in 0..frames {
            blocks.add(f, f, diag6(rot_prior, pos_prior));
            g[f] += Vector6::new(
                w[f].x * rot_prior,
                w[f].y * rot_prior,
                w[f].z * rot_prior,
                shift[f].x * pos_prior,
                shift[f].y * pos_prior,
                shift[f].z * pos_prior,
            );
        }
        let smooth = diag6(rot_smooth, pos_smooth);
        for &(f, h) in chain {
            blocks.add(f, f, smooth);
            blocks.add(h, h, smooth);
            blocks.add(f, h, -smooth);
            let (dw, dt) = (w[f] - w[h], shift[f] - shift[h]);
            let d = Vector6::new(
                dw.x * rot_smooth,
                dw.y * rot_smooth,
                dw.z * rot_smooth,
                dt.x * pos_smooth,
                dt.y * pos_smooth,
                dt.z * pos_smooth,
            );
            g[f] += d;
            g[h] -= d;
        }
        for f in 0..frames {
            let d = blocks.diag[&f].diagonal() * lambda;
            blocks.add(f, f, Matrix6::from_diagonal(&d));
        }
        let rhs: Vec<Vector6<f64>> = g.iter().map(|v| -v).collect();
        let step = pcg(&blocks, &rhs, 2000, 1e-10);
        let mut tx = x.clone();
        for ((tx, l), x) in tx.iter_mut().zip(&lin).zip(&x) {
            if let Some((_, l)) = l {
                let mut r = l.gp;
                for (f, wf) in &l.w {
                    r += wf.transpose() * step[*f];
                }
                *tx = x - l.hinv * r;
            }
        }
        let trot: Vec<M3> = rot
            .iter()
            .zip(&step)
            .map(|(r, d)| Rotation3::new(d.fixed_rows::<3>(0).into_owned()).matrix() * r)
            .collect();
        let tshift: Vec<V3> = shift
            .iter()
            .zip(&step)
            .map(|(t, d)| t + d.fixed_rows::<3>(3))
            .collect();
        let tcost = pb.cost(&tx, &trot, &tshift);
        let largest = step
            .iter()
            .map(|d| d.fixed_rows::<3>(0).norm())
            .fold(0f64, f64::max);
        if tcost < cost {
            let gain = (cost - tcost) / cost;
            (x, rot, shift, cost) = (tx, trot, tshift, tcost);
            lambda = (lambda / 3.).max(1e-9);
            accepted += 1;
            log(&format!(
                "tracks: iteration {iterations}: cost {cost:.6e}, largest turn {:.4} deg",
                largest.to_degrees()
            ));
            if accepted == 3 || accepted == 6 {
                let dropped = pb.prune(&mut x, &rot, &shift);
                cost = pb.cost(&x, &rot, &shift);
                log(&format!(
                    "tracks: {dropped} sightings over {} px dropped, {} tracks left",
                    s.outlier,
                    pb.tracks.len()
                ));
                continue;
            }
            if gain < 1e-6 || largest < 1e-7 {
                break;
            }
        } else {
            lambda *= 10.;
            if lambda > 1e6 {
                break;
            }
        }
    }
    Solution {
        rot,
        shift,
        iterations,
        cost: [start, cost],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A device walking past a wall (y = 4) above a floor (z = 0); the left camera looks at the
    /// wall and a little down, the right one turned aside.
    fn scene() -> (Vec<Pose>, Vec<V3>, Vec<V3>) {
        // camera x = world x, camera y = world -z, camera z = world y
        let base = M3::new(1., 0., 0., 0., 0., 1., 0., -1., 0.);
        let mut poses = Vec::new();
        for i in 0..12 {
            let t = i as f64;
            let r = Rotation3::from_euler_angles(0., 0., 0.02 * t).matrix()
                * base
                * Rotation3::from_euler_angles(-0.4 + 0.05 * (t * 0.7).sin(), 0., 0.).matrix();
            let p = V3::new(0.25 * t, 0., 1.2);
            poses.push(Pose { r, p });
            poses.push(Pose {
                r: r * Rotation3::from_euler_angles(0., 0.4, 0.).matrix(),
                p: p + r * V3::new(0.1, 0., 0.),
            });
        }
        let mut points = Vec::new();
        let mut normals = Vec::new();
        for k in 0..600 {
            let t = k as f64;
            if k % 2 == 0 {
                points.push(V3::new(
                    -1. + (t * 0.37).sin().abs() * 5.,
                    4.,
                    0.3 + (t * 0.11).cos().abs() * 2.,
                ));
                normals.push(V3::new(0., 1., 0.));
            } else {
                points.push(V3::new(
                    -1. + (t * 0.53).sin().abs() * 5.,
                    0.5 + (t * 0.29).cos().abs() * 3.,
                    0.,
                ));
                normals.push(V3::new(0., 0., 1.));
            }
        }
        (poses, points, normals)
    }

    fn look(p: &Pose, x: &V3) -> Option<V3> {
        let l = p.r.transpose() * (x - p.p);
        (l.xy().norm().atan2(l.z) < 1.2).then_some(l)
    }

    #[test]
    fn a_frame_moved_off_is_brought_back_and_the_points_stay_on_their_planes() {
        let cam = camera();
        let (truth, points, normals) = scene();
        let frame: Vec<usize> = (0..truth.len()).map(|i| i / 2).collect();
        let frames = truth.len() / 2;
        let chain: Vec<(usize, usize)> = (1..frames).map(|f| (f - 1, f)).collect();
        // the observations' tracks, made directly from the true poses
        let tracks: Vec<Track> = points
            .iter()
            .zip(&normals)
            .enumerate()
            .filter_map(|(k, (x, n))| {
                let sightings: Vec<Sighting> = truth
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| {
                        let l = look(p, x)?;
                        Some(Sighting::new(
                            i,
                            cam.project([l.x, l.y, l.z]),
                            &cam,
                            &[0.; 4],
                        ))
                    })
                    .collect();
                (sightings.len() >= 3).then(|| Track {
                    sightings,
                    // anchored a little off, as the cloud's planes are
                    anchor: x + n.cross(&V3::new(0.3, 0.2, 0.1)) * 0.01,
                    normal: *n,
                    held: k % 5 == 0,
                })
            })
            .collect();
        assert!(tracks.len() > 300, "{}", tracks.len());
        // frame 5 turned by 0.3 degrees and moved by 2 cm
        let mut start = truth.clone();
        let off = Rotation3::new(V3::new(0.002, -0.004, 0.003));
        for i in [10, 11] {
            start[i] = Pose {
                r: off.matrix() * truth[i].r,
                p: truth[i].p + V3::new(0.015, -0.01, 0.008),
            };
        }
        let (before, _) = check(&tracks, &start);
        let s = Settings::default();
        let sol = solve(&start, &frame, frames, &chain, &tracks, &s, &|_| {});
        let after_poses: Vec<Pose> = start
            .iter()
            .enumerate()
            .map(|(i, p)| sol.pose(frame[i], p))
            .collect();
        let (after, plane) = check(&tracks, &after_poses);
        // two images of 24 are off: the largest residuals show them
        let max = |v: &[f64]| v.iter().cloned().fold(0f64, f64::max);
        assert!(max(&before) > 5., "{}", max(&before));
        assert!(max(&after) < 0.1, "{}", max(&after));
        assert!(max(&plane) < 2e-3, "{}", max(&plane));
        let back = after_poses[10].r * truth[10].r.transpose();
        assert!(Rotation3::from_matrix_unchecked(back).angle() < 2e-4);
        assert!((after_poses[10].p - truth[10].p).norm() < 2e-3);
    }

    #[test]
    fn matches_join_into_tracks_and_a_track_seen_twice_in_one_image_is_dropped() {
        let cam = camera();
        let o = |a: usize, fa: usize, b: usize, fb: usize| Obs {
            a,
            b,
            fa,
            fb,
            uv_a: [1700., 2300.],
            uv_b: [1710., 2310.],
            world: V3::new(0., 0., 2.),
            normal: V3::new(0., 0., 1.),
            depth: 2.,
        };
        // 0:3 - 1:7 - 2:4 is a track; 3:1 - 4:2 - 3:5 sees image 3 twice
        let obs = vec![o(0, 3, 1, 7), o(1, 7, 2, 4), o(3, 1, 4, 2), o(4, 2, 3, 5)];
        let t = tracks(&obs, &[cam.clone(), cam], &[[0.; 4]; 2], &[0; 5], 0.);
        assert_eq!(t.len(), 1);
        let mut images: Vec<usize> = t[0].sightings.iter().map(|s| s.image).collect();
        images.sort_unstable();
        assert_eq!(images, vec![0, 1, 2]);
    }
}
