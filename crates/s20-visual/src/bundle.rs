//! Tracks across images and their bundle adjustment, after `refine`: the guided matches of every
//! pair joined by feature into tracks, each a free 3-D point seen by its images and held near the
//! local LiDAR plane its features were anchored on. Every frame (the images taken together) gets a
//! small correction shared by its images, with a weak prior and smoothness along the frames, so
//! the images with few tracks follow the plane-held points and their neighbors rather than drift.
//! Levenberg-Marquardt with the points eliminated (Schur
//! complement); the frames' block system by `refine`'s conjugate gradients.

use crate::anchors::{M3, Pose, V3};
use crate::camera::Fisheye;
use crate::pairs::Obs;
use crate::refine::{BlockSystem, Intrinsic, bend, pcg_counted, ray_and_derivatives};
use nalgebra::{Matrix3, Matrix6, Rotation3, SMatrix, SymmetricEigen, Vector2, Vector3, Vector6};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// soft-L1 scale of the bearing residual (pixel-equivalents)
    pub pixel: f64,
    /// a point's distance to its plane (m), and the soft-L1 scale of that residual in these units
    pub plane_sigma: f64,
    pub plane_scale: f64,
    /// a plane-held point's distance along its plane from where the starting poses' rays meet it
    /// (m; soft-L1 with `plane_scale`): the plane alone lets the frames that see one surface slide
    /// along it with their points (a shrine's door: 5 cm); zero: not held along
    pub along_sigma: f64,
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
            along_sigma: 0.03,
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
    // each observation's two features as nodes, then joined at once (the nodes numbered and
    // joined in the observations' order, without a list of the edges between)
    for o in obs {
        let mut node = |image: usize, feature: usize, uv: [f64; 2]| -> usize {
            *id.entry((image, feature)).or_insert_with(|| {
                nodes.push((image, uv));
                anchor.push(None);
                parent.push(parent.len());
                parent.len() - 1
            })
        };
        let (a, b) = (node(o.a, o.fa, o.uv_a), node(o.b, o.fb, o.uv_b));
        // the anchor is that of image a's feature (its ray met the plane)
        if anchor[a].is_none() {
            anchor[a] = Some((o.world, o.normal));
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
    // (in parallel, in the groups' order: each sighting's bearing takes the camera's inverse)
    members
        .par_iter()
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
            let center = V3::new(mid(0), mid(1), mid(2));
            let nearest = anchors
                .iter()
                .min_by(|a, b| (a.0 - center).norm().total_cmp(&(b.0 - center).norm()))
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

/// A plane-held point's offset along its plane from the track's anchor.
fn along(t: &Track, x: &V3) -> V3 {
    let d = x - t.anchor;
    d - t.normal * t.normal.dot(&d)
}

/// Where the sightings' rays from `poses` meet the track's plane: the median of the meeting points
/// (each axis), back onto the plane; none when no ray meets it in front at more than 0.1 of a right
/// angle.
fn meet_plane(t: &Track, poses: &[Pose]) -> Option<V3> {
    let (a, n) = (t.anchor, t.normal);
    let hits: Vec<V3> = t
        .sightings
        .iter()
        .filter_map(|s| {
            let p = &poses[s.image];
            let d = p.r * s.bearing;
            let along = n.dot(&d);
            if along.abs() < 0.1 {
                return None;
            }
            let k = n.dot(&(a - p.p)) / along;
            (k > 0.).then(|| p.p + d * k)
        })
        .collect();
    if hits.is_empty() {
        return None;
    }
    let median = |axis: usize| {
        let mut v: Vec<f64> = hits.iter().map(|h| h[axis]).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let m = V3::new(median(0), median(1), median(2));
    Some(m - n * n.dot(&(m - a)))
}

/// A track triangulated with `poses`, with what tells how well it is determined (for the image
/// points written beside the COLMAP model: surfaces the laser missed, such as those too close to it).
pub struct TrackPoint {
    pub x: V3,
    pub sightings: usize,
    /// the bearing residuals' median and largest (pixel-equivalents)
    pub residual_median: f64,
    pub residual_max: f64,
    /// the widest angle between two sightings' rays (degrees): small for short baselines, whose
    /// points spread along the rays
    pub angle_deg: f64,
    /// the nearest sighting camera's distance to the point (m)
    pub nearest: f64,
    /// the distance to where its features met the cloud (m), and whether that was on a plane: far
    /// when the rays went through a hole in the cloud to a surface behind
    pub to_anchor: f64,
    pub planar: bool,
    /// the sightings' images (indices into `poses`)
    pub images: Vec<usize>,
}

/// `t` triangulated with `poses` (none when its rays are nearly parallel or the point is behind a
/// sighting camera).
pub fn track_point(t: &Track, poses: &[Pose]) -> Option<TrackPoint> {
    let x = triangulate(t, poses)?;
    let mut res = Vec::with_capacity(t.sightings.len());
    let mut rays = Vec::with_capacity(t.sightings.len());
    let mut nearest = f64::INFINITY;
    for s in &t.sightings {
        let p = &poses[s.image];
        let (r, v) = s.residual(p, &x);
        if v.dot(&s.bearing) <= 0. {
            return None;
        }
        res.push(r.norm());
        rays.push((p.r * s.bearing).normalize());
        nearest = nearest.min(v.norm());
    }
    let mut angle: f64 = 0.;
    for i in 0..rays.len() {
        for j in i + 1..rays.len() {
            angle = angle.max(rays[i].dot(&rays[j]).clamp(-1., 1.).acos());
        }
    }
    let mut sorted = res;
    sorted.sort_by(f64::total_cmp);
    Some(TrackPoint {
        x,
        sightings: t.sightings.len(),
        residual_median: sorted[sorted.len() / 2],
        residual_max: sorted[sorted.len() - 1],
        angle_deg: angle.to_degrees(),
        nearest,
        to_anchor: (x - t.anchor).norm(),
        planar: t.planar(),
        images: t.sightings.iter().map(|s| s.image).collect(),
    })
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
            // (as track_point: none behind a sighting camera, where the residual is small)
            let r = t
                .sightings
                .iter()
                .map(|s| {
                    let (r, v) = s.residual(&poses[s.image], &x);
                    (v.dot(&s.bearing) > 0.).then(|| r.norm())
                })
                .collect::<Option<Vec<f64>>>()?;
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
/// their centers.
#[derive(Clone, Debug)]
pub struct Solution {
    pub rot: Vec<M3>,
    pub shift: Vec<V3>,
    pub iterations: usize,
    pub cost: [f64; 2],
    /// the tracks adjusted, with the sightings dropped as outliers left out (on the same planes;
    /// the anchors where the starting poses' rays met them)
    pub kept: Vec<Track>,
}

impl Solution {
    pub fn pose(&self, frame: usize, p: &Pose) -> Pose {
        Pose {
            r: self.rot[frame] * p.r,
            p: p.p + self.shift[frame],
        }
    }
}

/// How far planes tell a scale about `mean` apart from a turn and a shift: the scale's hold left
/// once those are solved for, as a share of its own, from the planes' anchors and normals (the
/// points' places along them aside). Zero when they all pass through one line or point.
fn scale_hold(on: &[(V3, V3, V3)], mean: &V3) -> f64 {
    type M7 = SMatrix<f64, 7, 7>;
    type V7 = SMatrix<f64, 7, 1>;
    let mut h = M7::zeros();
    for (_, a, n) in on {
        let v = a - mean;
        let jw = v.cross(n);
        let j = V7::from_column_slice(&[jw.x, jw.y, jw.z, n.x, n.y, n.z, n.dot(&v)]);
        h += j * j.transpose();
    }
    if h[(6, 6)] <= 0. {
        return 0.;
    }
    // a faint ridge only to invert what the planes leave free (along a corridor)
    let rest =
        h.fixed_view::<6, 6>(0, 0).into_owned() + Matrix6::identity() * (1e-9 * h.trace() / 7.);
    let cross = h.fixed_view::<6, 1>(0, 6).into_owned();
    rest.try_inverse().map_or(0., |inv| {
        ((h[(6, 6)] - (cross.transpose() * inv * cross)[0]) / h[(6, 6)]).max(0.)
    })
}

/// The images moved together onto the cloud by [`align_to_planes`].
pub struct Aligned {
    pub poses: Vec<Pose>,
    /// the turn, the shift (of the points' mean) and the scale
    pub turn: M3,
    pub shift: V3,
    pub scale: f64,
    /// how far the planes tell the scale apart from the turn and the place (its hold left once
    /// they are solved for, as a share of its own): below 1 % it is not solved for
    pub scale_hold: f64,
}

/// `poses` turned, moved and scaled together (one similarity about the points' mean) so that the
/// (not held-out) plane-held tracks, triangulated with them, lie best on their LiDAR planes:
/// Gauss-Newton on the points' distances along the normals, Huber at 1 cm. The adjustment holds the
/// images to one another by bearings, which leave the whole turn, place and scale free, and to the
/// planes only weakly, so in a small room the images came out a few tenths of a degree to a degree
/// turned and up to 1.5 % wider than the cloud; this takes that back without changing how they
/// agree. The scale is left when the planes cannot tell it (all through one line or point, as along
/// a corridor of a wall and the floor).
pub fn align_to_planes(tracks: &[Track], poses: &[Pose]) -> Aligned {
    let on: Vec<(V3, V3, V3)> = tracks
        .par_iter()
        .filter(|t| !t.held && t.planar() && t.sightings.len() >= 2)
        .filter_map(|t| triangulate(t, poses).map(|x| (x, t.anchor, t.normal)))
        .collect();
    let mut out = Aligned {
        poses: poses.to_vec(),
        turn: M3::identity(),
        shift: V3::zeros(),
        scale: 1.,
        scale_hold: 0.,
    };
    if on.len() < 50 {
        return out;
    }
    let mean = on.iter().map(|o| o.0).sum::<V3>() / on.len() as f64;
    let (mut turn, mut shift, mut scale) = (M3::identity(), V3::zeros(), 1f64);
    type M7 = SMatrix<f64, 7, 7>;
    type V7 = SMatrix<f64, 7, 1>;
    let mut scaled = None;
    for _ in 0..30 {
        let mut h = M7::zeros();
        let mut g = V7::zeros();
        for (x, a, n) in &on {
            let v = scale * (turn * (x - mean));
            let y = v + mean + shift;
            let r = n.dot(&(y - a));
            let w = if r.abs() <= 0.01 { 1. } else { 0.01 / r.abs() };
            // d r / d (small turn, shift, log scale) = (v x n, n, n . v)
            let jw = v.cross(n);
            let j = V7::from_column_slice(&[jw.x, jw.y, jw.z, n.x, n.y, n.z, n.dot(&v)]);
            h += w * j * j.transpose();
            g += w * j * r;
        }
        // a faint pull to no change (a millionth of the planes' mean hold): what they cannot tell
        // of the turn and the place (along a corridor, the place along it) stays
        let prior = 1e-6 * h.trace() / 7.;
        let w0 = rotvec(&turn);
        let now = V7::from_column_slice(&[w0.x, w0.y, w0.z, shift.x, shift.y, shift.z, scale.ln()]);
        h += M7::identity() * prior;
        g += now * prior;
        // whether the planes tell the scale apart, once, from where they lie (the anchors) and
        // not from the points: the points' few millimeters off their planes would tell it on one
        // plane too, and shrinking everything onto it would lessen them
        let solve_scale = *scaled.get_or_insert_with(|| {
            out.scale_hold = scale_hold(&on, &mean);
            out.scale_hold >= 0.01
        });
        if !solve_scale {
            h.row_mut(6).fill(0.);
            h.column_mut(6).fill(0.);
            h[(6, 6)] = 1.;
            g[6] = 0.;
        }
        let Some(d) = h.lu().solve(&(-g)) else {
            break;
        };
        turn = *Rotation3::new(d.fixed_rows::<3>(0).into_owned()).matrix() * turn;
        shift += d.fixed_rows::<3>(3);
        scale *= d[6].exp();
        if d.norm() < 1e-10 {
            break;
        }
    }
    out.poses = poses
        .iter()
        .map(|p| Pose {
            r: turn * p.r,
            p: scale * (turn * (p.p - mean)) + mean + shift,
        })
        .collect();
    (out.turn, out.shift, out.scale) = (turn, shift, scale);
    out
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

/// A frame-point block: how a frame's correction and a track's point meet in the normal equations.
type W = SMatrix<f64, 6, 3>;

/// Each track's frames, in the order its sightings first see them, one after the other (track `i`'s
/// from `start[i]` to `start[i + 1]`).
struct TrackFrames {
    start: Vec<usize>,
    frame: Vec<u32>,
}

impl TrackFrames {
    fn new(pb: &Problem) -> Self {
        let mut start = Vec::with_capacity(pb.tracks.len() + 1);
        let mut frame = Vec::new();
        start.push(0);
        for t in &pb.tracks {
            let first = frame.len();
            for sg in &t.sightings {
                let f = pb.frame[sg.image] as u32;
                if !frame[first..].contains(&f) {
                    frame.push(f);
                }
            }
            start.push(frame.len());
        }
        TrackFrames { start, frame }
    }

    fn of(&self, i: usize) -> &[u32] {
        &self.frame[self.start[i]..self.start[i + 1]]
    }
}

/// The pairs of frames that some track or the chain joins, each once with the lower frame first
/// (sorted); per frame, its pairs as the lower frame (a range of them) and as the higher one; and
/// per track, the pair of each two of its frames (`k < l` in its order, `k` major), the top bit
/// set where the track's two frames come the other way round.
struct Pattern {
    pairs: Vec<(u32, u32)>,
    above_start: Vec<usize>,
    below_start: Vec<usize>,
    below: Vec<(u32, u32)>,
    track_start: Vec<usize>,
    track_pair: Vec<u32>,
}

const FLIPPED: u32 = 1 << 31;

impl Pattern {
    fn new(tf: &TrackFrames, tracks: usize, frames: usize, chain: &[(usize, usize)]) -> Self {
        let mut keys: Vec<(u32, u32)> = (0..tracks)
            .into_par_iter()
            .flat_map_iter(|i| {
                let fr = tf.of(i);
                (0..fr.len()).flat_map(move |k| {
                    ((k + 1)..fr.len()).map(move |l| (fr[k].min(fr[l]), fr[k].max(fr[l])))
                })
            })
            .collect();
        keys.extend(
            chain
                .iter()
                .map(|&(f, h)| (f.min(h) as u32, f.max(h) as u32)),
        );
        keys.par_sort_unstable();
        keys.dedup();
        keys.shrink_to_fit();
        let pairs = keys;
        let mut above_start = vec![0usize; frames + 1];
        let mut below_start = vec![0usize; frames + 1];
        for &(a, b) in &pairs {
            above_start[a as usize + 1] += 1;
            below_start[b as usize + 1] += 1;
        }
        for f in 0..frames {
            above_start[f + 1] += above_start[f];
            below_start[f + 1] += below_start[f];
        }
        let mut below = vec![(0u32, 0u32); pairs.len()];
        let mut at = below_start.clone();
        for (p, &(a, b)) in pairs.iter().enumerate() {
            below[at[b as usize]] = (a, p as u32);
            at[b as usize] += 1;
        }
        let mut track_start = Vec::with_capacity(tracks + 1);
        track_start.push(0);
        for i in 0..tracks {
            let m = tf.of(i).len();
            track_start.push(track_start[i] + m * m.saturating_sub(1) / 2);
        }
        let mut pattern = Pattern {
            pairs,
            above_start,
            below_start,
            below,
            track_start,
            track_pair: Vec::new(),
        };
        let mut track_pair = vec![0u32; *pattern.track_start.last().unwrap()];
        split_by(&mut track_pair, &pattern.track_start)
            .into_par_iter()
            .enumerate()
            .for_each(|(i, out)| {
                let fr = tf.of(i);
                let mut n = 0;
                for k in 0..fr.len() {
                    for l in (k + 1)..fr.len() {
                        let flipped = if fr[k] > fr[l] { FLIPPED } else { 0 };
                        out[n] = pattern.pair(fr[k], fr[l]) | flipped;
                        n += 1;
                    }
                }
            });
        pattern.track_pair = track_pair;
        pattern
    }

    /// The pair of two different frames.
    fn pair(&self, f: u32, h: u32) -> u32 {
        let (a, b) = (f.min(h), f.max(h));
        let lo = self.above_start[a as usize];
        let row = &self.pairs[lo..self.above_start[a as usize + 1]];
        (lo + row.binary_search_by(|q| q.1.cmp(&b)).unwrap()) as u32
    }
}

/// `v` split at `start` (offsets, the first 0 and the last `v.len()`), each part writable apart.
fn split_by<'a, T>(v: &'a mut [T], start: &[usize]) -> Vec<&'a mut [T]> {
    let mut parts = Vec::with_capacity(start.len().saturating_sub(1));
    let mut rest = v;
    for w in start.windows(2) {
        let (part, tail) = rest.split_at_mut(w[1] - w[0]);
        parts.push(part);
        rest = tail;
    }
    parts
}

/// The frames' system once the points are eliminated: the blocks on the diagonal and those of the
/// pairs of `pattern`.
struct FrameSystem<'a> {
    diag: &'a [Matrix6<f64>],
    off: &'a [Matrix6<f64>],
    pattern: &'a Pattern,
}

impl BlockSystem for FrameSystem<'_> {
    fn diag(&self, i: usize) -> Option<Matrix6<f64>> {
        self.diag.get(i).copied()
    }

    /// By frame, in parallel: the pairs where it is the lower frame, then the higher.
    fn mul(&self, x: &[Vector6<f64>]) -> Vec<Vector6<f64>> {
        let pt = self.pattern;
        (0..x.len())
            .into_par_iter()
            .map(|i| {
                let mut y = self.diag[i] * x[i];
                for p in pt.above_start[i]..pt.above_start[i + 1] {
                    y += self.off[p] * x[pt.pairs[p].1 as usize];
                }
                for &(a, p) in &pt.below[pt.below_start[i]..pt.below_start[i + 1]] {
                    y += self.off[p as usize].transpose() * x[a as usize];
                }
                y
            })
            .collect()
    }
}

struct Problem<'a> {
    poses: &'a [Pose],
    frame: &'a [usize],
    frames: usize,
    chain: &'a [(usize, usize)],
    tracks: Vec<Track>,
    /// per track: held along its plane too (where the starting poses' rays met it)
    pinned: Vec<bool>,
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

    fn track_cost(&self, t: &Track, pinned: bool, x: &V3, rot: &[M3], shift: &[V3]) -> f64 {
        let s = self.s;
        let mut c = 0.;
        for sg in &t.sightings {
            let (r, _) = sg.residual(&self.pose(rot, shift, sg.image), x);
            c += soft_l1(r.norm_squared() / (s.pixel * s.pixel)).0;
        }
        if t.planar() {
            let rp = t.normal.dot(&(x - t.anchor)) / s.plane_sigma;
            c += soft_l1(rp * rp / (s.plane_scale * s.plane_scale)).0;
            if pinned {
                let ra = along(t, x) / s.along_sigma;
                c += soft_l1(ra.norm_squared() / (s.plane_scale * s.plane_scale)).0;
            }
        }
        c
    }

    /// Drops the sightings missing by more than `outlier`, then the tracks no longer usable;
    /// returns the sightings dropped.
    fn prune(&mut self, x: &mut Vec<V3>, rot: &[M3], shift: &[V3]) -> usize {
        let tracks = std::mem::take(&mut self.tracks);
        let pinned = std::mem::take(&mut self.pinned);
        let points = std::mem::take(x);
        let mut dropped = 0;
        for ((mut t, pin), p) in tracks.into_iter().zip(pinned).zip(points) {
            let n = t.sightings.len();
            // (a point behind the camera misses too: its bearing residual is small, even zero
            // straight behind)
            t.sightings.retain(|sg| {
                let (r, v) = sg.residual(&self.pose(rot, shift, sg.image), &p);
                v.dot(&sg.bearing) > 0. && r.norm() <= self.s.outlier
            });
            dropped += n - t.sightings.len();
            if usable(&t) {
                self.tracks.push(t);
                self.pinned.push(pin);
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
            .zip(&self.pinned)
            .zip(x)
            .map(|((t, &pin), x)| self.track_cost(t, pin, x, rot, shift))
            .sum();
        tracks + self.prior_cost(rot, shift)
    }

    /// A track linearized: per frame of `fr` (its frames), the frame's block and gradient into `h`
    /// and the frame-point block into `w`; returned, the inverse of the point's block (damped) and
    /// the point's gradient, or none where the block cannot be inverted.
    #[allow(clippy::too_many_arguments)]
    fn linearize(
        &self,
        t: &Track,
        pinned: bool,
        x: &V3,
        rot: &[M3],
        shift: &[V3],
        lambda: f64,
        fr: &[u32],
        h: &mut [(Matrix6<f64>, Vector6<f64>)],
        w: &mut [W],
    ) -> Option<(Matrix3<f64>, Vector3<f64>)> {
        let s = self.s;
        let mut hpp = Matrix3::zeros();
        let mut gp = Vector3::zeros();
        h.fill((Matrix6::zeros(), Vector6::zeros()));
        w.fill(W::zeros());
        for sg in &t.sightings {
            let pose = self.pose(rot, shift, sg.image);
            let (r, v) = sg.residual(&pose, x);
            let n = v.norm();
            if n < 1e-9 {
                continue;
            }
            let e = v / n;
            let wt = soft_l1(r.norm_squared() / (s.pixel * s.pixel)).1 / (s.pixel * s.pixel);
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
            hpp += wt * jx.transpose() * jx;
            gp += wt * jx.transpose() * r;
            let f = self.frame[sg.image] as u32;
            let k = fr.iter().position(|&q| q == f).unwrap();
            h[k].0 += wt * jc.transpose() * jc;
            h[k].1 += wt * jc.transpose() * r;
            w[k] += wt * jc.transpose() * jx;
        }
        if t.planar() {
            let rp = t.normal.dot(&(x - t.anchor)) / s.plane_sigma;
            let wp = soft_l1(rp * rp / (s.plane_scale * s.plane_scale)).1
                / (s.plane_scale * s.plane_scale);
            let jp = t.normal / s.plane_sigma;
            hpp += wp * jp * jp.transpose();
            gp += wp * jp * rp;
            if pinned {
                // r = P (x - a) / sigma with P the projection onto the plane: J = P / sigma
                let ra = along(t, x) / s.along_sigma;
                let wa = soft_l1(ra.norm_squared() / (s.plane_scale * s.plane_scale)).1
                    / (s.plane_scale * s.plane_scale);
                let p = (M3::identity() - t.normal * t.normal.transpose()) / s.along_sigma;
                hpp += wa * p * p;
                gp += wa * p * ra;
            }
        }
        let damped =
            hpp + Matrix3::from_diagonal(&hpp.diagonal()) * lambda + Matrix3::identity() * 1e-12;
        Some((damped.try_inverse()?, gp))
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
    // held along the plane to where the starting poses put it (the anchor came from the poses
    // before the refinement); not at all when their rays do not meet the plane
    let (tracks_in, pinned): (Vec<Track>, Vec<bool>) = tracks
        .iter()
        .filter(|t| !t.held && usable(t))
        .cloned()
        .map(|mut t| {
            let met = (t.planar() && s.along_sigma > 0.)
                .then(|| meet_plane(&t, poses))
                .flatten();
            if let Some(a) = met {
                t.anchor = a;
            }
            (t, met.is_some())
        })
        .unzip();
    let mut pb = Problem {
        poses,
        frame,
        frames,
        chain,
        tracks: tracks_in,
        pinned,
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
    // the tracks' frames and the pairs of frames, again when tracks are dropped; the blocks of
    // the pairs and the frames, summed into from all threads, then read by the conjugate gradients
    let mut tf = TrackFrames::new(&pb);
    let mut pattern = Pattern::new(&tf, pb.tracks.len(), frames, chain);
    let mut wbuf: Vec<W> = Vec::new();
    while iterations < s.iterations {
        iterations += 1;
        let started = std::time::Instant::now();
        let n = pb.tracks.len();
        // each track linearized and its point eliminated at once, into the frames' system
        // (H_ff - W H_pp^-1 W^T and g_f - W H_pp^-1 g_p); kept per track only what the
        // back-substitution needs, in buffers of the whole iteration
        let off_acc: Vec<Mutex<Matrix6<f64>>> = (0..pattern.pairs.len())
            .map(|_| Mutex::new(Matrix6::zeros()))
            .collect();
        let diag_acc: Vec<Mutex<(Matrix6<f64>, Vector6<f64>)>> = (0..frames)
            .map(|_| Mutex::new((Matrix6::zeros(), Vector6::zeros())))
            .collect();
        wbuf.clear();
        wbuf.resize(tf.frame.len(), W::zeros());
        let mut point: Vec<Option<(Matrix3<f64>, Vector3<f64>)>> = vec![None; n];
        split_by(&mut wbuf, &tf.start)
            .into_par_iter()
            .zip(point.par_iter_mut())
            .enumerate()
            .for_each_init(Vec::new, |h, (i, (w, point))| {
                let fr = tf.of(i);
                h.resize(fr.len(), (Matrix6::zeros(), Vector6::zeros()));
                let (t, pin) = (&pb.tracks[i], pb.pinned[i]);
                let Some((hinv, gp)) = pb.linearize(t, pin, &x[i], &rot, &shift, lambda, fr, h, w)
                else {
                    w.fill(W::zeros());
                    return;
                };
                *point = Some((hinv, gp));
                let pairs = &pattern.track_pair[pattern.track_start[i]..];
                let mut q = 0;
                for k in 0..fr.len() {
                    let wh = w[k] * hinv;
                    {
                        let mut d = diag_acc[fr[k] as usize].lock().unwrap();
                        d.0 += h[k].0 - wh * w[k].transpose();
                        d.1 += h[k].1 - wh * gp;
                    }
                    for wl in &w[(k + 1)..] {
                        let e = pairs[q];
                        q += 1;
                        let b = -(wh * wl.transpose());
                        let mut o = off_acc[(e & !FLIPPED) as usize].lock().unwrap();
                        if e & FLIPPED == 0 {
                            *o += b;
                        } else {
                            *o += b.transpose();
                        }
                    }
                }
            });
        let mut diag: Vec<Matrix6<f64>> = Vec::with_capacity(frames);
        let mut g: Vec<Vector6<f64>> = Vec::with_capacity(frames);
        for d in diag_acc {
            let (h, gf) = d.into_inner().unwrap();
            diag.push(h);
            g.push(gf);
        }
        let mut off: Vec<Matrix6<f64>> = off_acc
            .into_par_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();
        let w: Vec<V3> = rot.iter().map(rotvec).collect();
        for f in 0..frames {
            diag[f] += diag6(rot_prior, pos_prior);
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
            diag[f] += smooth;
            diag[h] += smooth;
            // (diagonal: the same either way round)
            off[pattern.pair(f as u32, h as u32) as usize] -= smooth;
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
        for d in diag.iter_mut() {
            *d += Matrix6::from_diagonal(&(d.diagonal() * lambda));
        }
        let rhs: Vec<Vector6<f64>> = g.iter().map(|v| -v).collect();
        let assembled = started.elapsed().as_secs_f64();
        let system = FrameSystem {
            diag: &diag,
            off: &off,
            pattern: &pattern,
        };
        let (step, cg) = pcg_counted(&system, &rhs, 2000, 1e-10);
        let solved = started.elapsed().as_secs_f64();
        let off_blocks = off.len();
        drop(off);
        let tx: Vec<V3> = (0..n)
            .into_par_iter()
            .map(|i| match &point[i] {
                Some((hinv, gp)) => {
                    let mut r = *gp;
                    for (f, wf) in tf.of(i).iter().zip(&wbuf[tf.start[i]..tf.start[i + 1]]) {
                        r += wf.transpose() * step[*f as usize];
                    }
                    x[i] - hinv * r
                }
                None => x[i],
            })
            .collect();
        let back = started.elapsed().as_secs_f64();
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
        let before_cost = started.elapsed().as_secs_f64();
        let tcost = pb.cost(&tx, &trot, &tshift);
        let largest = step
            .iter()
            .map(|d| d.fixed_rows::<3>(0).norm())
            .fold(0f64, f64::max);
        // where the iteration's time went: the tracks linearized into the frames' system, the
        // conjugate gradients (their iterations, the frame pairs' blocks), the points back from
        // the frames' step, the trial cost
        let total = started.elapsed().as_secs_f64();
        let took = format!(
            "{total:.0} s: system {assembled:.0}, cg {:.0} ({} its, {off_blocks} pair blocks), points {:.0}, cost {:.0}{}",
            solved - assembled,
            cg.iterations,
            back - solved,
            total - before_cost,
            crate::memory::note()
        );
        if tcost < cost {
            let gain = (cost - tcost) / cost;
            (x, rot, shift, cost) = (tx, trot, tshift, tcost);
            lambda = (lambda / 3.).max(1e-9);
            accepted += 1;
            log(&format!(
                "tracks: iteration {iterations}: cost {cost:.6e}, largest turn {:.4} deg ({took})",
                largest.to_degrees()
            ));
            if accepted == 3 || accepted == 6 {
                let dropped = pb.prune(&mut x, &rot, &shift);
                tf = TrackFrames::new(&pb);
                pattern = Pattern::new(&tf, pb.tracks.len(), frames, chain);
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
            log(&format!(
                "tracks: iteration {iterations}: not taken (cost {tcost:.6e} against {cost:.6e}, damping {lambda:.0e} -> {:.0e}) ({took})",
                lambda * 10.
            ));
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
        kept: pb.tracks,
    }
}

/// What [`solve`] takes, written to a file and read back: the adjustment tried again apart from
/// the matching before it (`S20_DUMP_TRACKS` in `run`, and the `solve_replay` example).
pub struct SolveInput {
    pub poses: Vec<Pose>,
    pub frame: Vec<usize>,
    pub frames: usize,
    pub chain: Vec<(usize, usize)>,
    pub tracks: Vec<Track>,
}

const SOLVE_INPUT_MAGIC: &[u8; 8] = b"S20SLV1\n";

/// Writes [`solve`]'s inputs to `path` (little-endian; see [`SolveInput::load`]).
pub fn save_solve_input(
    path: &std::path::Path,
    poses: &[Pose],
    frame: &[usize],
    frames: usize,
    chain: &[(usize, usize)],
    tracks: &[Track],
) -> anyhow::Result<()> {
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);
    let u = |w: &mut std::io::BufWriter<std::fs::File>, v: u64| w.write_all(&v.to_le_bytes());
    let f = |w: &mut std::io::BufWriter<std::fs::File>, v: &[f64]| -> std::io::Result<()> {
        for x in v {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    };
    w.write_all(SOLVE_INPUT_MAGIC)?;
    u(&mut w, poses.len() as u64)?;
    for p in poses {
        f(&mut w, p.r.as_slice())?;
        f(&mut w, p.p.as_slice())?;
    }
    u(&mut w, frame.len() as u64)?;
    for &k in frame {
        u(&mut w, k as u64)?;
    }
    u(&mut w, frames as u64)?;
    u(&mut w, chain.len() as u64)?;
    for &(a, b) in chain {
        u(&mut w, a as u64)?;
        u(&mut w, b as u64)?;
    }
    u(&mut w, tracks.len() as u64)?;
    for t in tracks {
        f(&mut w, t.anchor.as_slice())?;
        f(&mut w, t.normal.as_slice())?;
        u(&mut w, t.held as u64)?;
        u(&mut w, t.sightings.len() as u64)?;
        for sg in &t.sightings {
            u(&mut w, sg.image as u64)?;
            f(&mut w, sg.bearing.as_slice())?;
            f(&mut w, sg.ta.as_slice())?;
            f(&mut w, sg.tb.as_slice())?;
            f(&mut w, &[sg.focal])?;
        }
    }
    w.flush()?;
    Ok(())
}

impl SolveInput {
    /// Reads what [`save_solve_input`] wrote (an error for a file that does not hold it).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        use std::io::Read;
        let file = std::fs::File::open(path)?;
        let size = file.metadata()?.len();
        // a count cannot ask for more than the file holds (8 bytes an item at the least)
        let fits = |n: u64| -> anyhow::Result<usize> {
            anyhow::ensure!(
                n <= size / 8,
                "not the adjustment's inputs (a count of {n})"
            );
            Ok(n as usize)
        };
        let mut r = std::io::BufReader::new(file);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        anyhow::ensure!(&magic == SOLVE_INPUT_MAGIC, "not the adjustment's inputs");
        let mut b = [0u8; 8];
        let mut u = |r: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<u64> {
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        };
        let f =
            |r: &mut std::io::BufReader<std::fs::File>, n: usize| -> std::io::Result<Vec<f64>> {
                let mut v = vec![0u8; 8 * n];
                r.read_exact(&mut v)?;
                Ok(v.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| f64::from_le_bytes(*c))
                    .collect())
            };
        let v3 = |v: &[f64]| V3::new(v[0], v[1], v[2]);
        let n = fits(u(&mut r)?)?;
        let mut poses = Vec::with_capacity(n);
        for _ in 0..n {
            let v = f(&mut r, 12)?;
            poses.push(Pose {
                r: M3::from_column_slice(&v[..9]),
                p: v3(&v[9..]),
            });
        }
        let n = fits(u(&mut r)?)?;
        let frame = (0..n)
            .map(|_| u(&mut r).map(|k| k as usize))
            .collect::<std::io::Result<_>>()?;
        let frames = fits(u(&mut r)?)?;
        let n = fits(u(&mut r)?)?;
        let mut chain = Vec::with_capacity(n);
        for _ in 0..n {
            chain.push((u(&mut r)? as usize, u(&mut r)? as usize));
        }
        let n = fits(u(&mut r)?)?;
        let mut tracks = Vec::with_capacity(n);
        for _ in 0..n {
            let v = f(&mut r, 6)?;
            let held = u(&mut r)? != 0;
            let m = fits(u(&mut r)?)?;
            let mut sightings = Vec::with_capacity(m);
            for _ in 0..m {
                let image = u(&mut r)? as usize;
                let s = f(&mut r, 10)?;
                sightings.push(Sighting {
                    image,
                    bearing: v3(&s[0..3]),
                    ta: v3(&s[3..6]),
                    tb: v3(&s[6..9]),
                    focal: s[9],
                });
            }
            tracks.push(Track {
                sightings,
                anchor: v3(&v[..3]),
                normal: v3(&v[3..]),
                held,
            });
        }
        let input = SolveInput {
            poses,
            frame,
            frames,
            chain,
            tracks,
        };
        // the indices within what they index
        anyhow::ensure!(
            input.frame.len() == input.poses.len()
                && input.frame.iter().all(|&f| f < input.frames)
                && input
                    .chain
                    .iter()
                    .all(|&(a, b)| a < input.frames && b < input.frames)
                && input
                    .tracks
                    .iter()
                    .flat_map(|t| &t.sightings)
                    .all(|s| s.image < input.poses.len()),
            "not the adjustment's inputs (an index out of range)"
        );
        Ok(input)
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
    fn the_rays_meet_the_plane_where_the_point_is_whatever_the_anchor_on_it() {
        let (truth, points, normals) = scene();
        for t in plane_tracks(&truth, &points, &normals) {
            // the anchor is elsewhere on the plane: the meeting point is the point itself
            let x = triangulate(&t, &truth).unwrap();
            let m = meet_plane(&t, &truth).unwrap();
            assert!((m - x).norm() < 1e-6, "{}", (m - x).norm());
            assert!(along(&t, &m).norm() > 1e-3);
        }
    }

    /// Tracks seen by the images at `poses`, held on the planes of `scene` (every fifth held out).
    fn plane_tracks(poses: &[Pose], points: &[V3], normals: &[V3]) -> Vec<Track> {
        let cam = camera();
        points
            .iter()
            .zip(normals)
            .enumerate()
            .filter_map(|(k, (x, n))| {
                let sightings: Vec<Sighting> = poses
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
                    // anchored elsewhere on the same plane, as the cloud's points are
                    anchor: x + n.cross(&V3::new(0.3, 0.2, 0.1)) * 0.01,
                    normal: *n,
                    held: k % 5 == 0,
                })
            })
            .collect()
    }

    /// The images as the bearings alone leave them: all turned 0.5 deg about a tilted axis,
    /// `scale` wider about a point off the path and moved by `shift`.
    fn widened(truth: &[Pose], scale: f64, shift: V3) -> Vec<Pose> {
        let turn =
            *Rotation3::new(V3::new(0.3, -0.5, 0.8).normalize() * 0.5f64.to_radians()).matrix();
        let pivot = V3::new(3., 2., 1.);
        truth
            .iter()
            .map(|p| Pose {
                r: turn * p.r,
                p: scale * (turn * (p.p - pivot)) + pivot + shift,
            })
            .collect()
    }

    fn angle(m: &M3) -> f64 {
        ((m.trace() - 1.) / 2.).clamp(-1., 1.).acos()
    }

    #[test]
    fn images_turned_widened_and_moved_together_are_put_back_onto_the_planes() {
        let (truth, mut points, mut normals) = scene();
        // a wall slanted across the path (with the wall and the floor, every turn and place) and
        // a table top above the floor (the scale: planes all through one point keep any about it)
        let slant = V3::new(1., 1., 0.).normalize();
        for k in 0..300 {
            let t = k as f64;
            let y = 1. + (t * 0.41).sin().abs() * 3.;
            points.push(V3::new(6. - y, y, 0.3 + (t * 0.17).cos().abs() * 1.7));
            normals.push(slant);
            points.push(V3::new(
                (t * 0.23).sin().abs() * 3.,
                1.5 + (t * 0.31).cos().abs() * 1.5,
                0.75,
            ));
            normals.push(V3::new(0., 0., 1.));
        }
        let tracks = plane_tracks(&truth, &points, &normals);
        let moved = widened(&truth, 1.012, V3::new(0.015, -0.01, 0.02));
        let a = align_to_planes(&tracks, &moved);
        assert!(a.scale_hold > 0.01, "{}", a.scale_hold);
        assert!(
            (angle(&a.turn) - 0.5f64.to_radians()).abs() < 1e-5,
            "{}",
            angle(&a.turn).to_degrees()
        );
        assert!((a.scale * 1.012 - 1.).abs() < 1e-5, "{}", a.scale);
        for (b, p) in a.poses.iter().zip(&truth) {
            let e = angle(&(b.r * p.r.transpose()));
            assert!(e < 1e-5, "{e}");
            assert!((b.p - p.p).norm() < 1e-4, "{}", (b.p - p.p).norm());
        }
    }

    #[test]
    fn on_one_plane_the_points_off_it_do_not_shrink_the_images_onto_it() {
        // the wall only, its points a few millimeters off it either side (as triangulated points
        // are): a scale would lessen those, but the plane cannot tell one
        let (truth, points, normals) = scene();
        let (mut wall, mut up) = (Vec::new(), Vec::new());
        for (k, (x, n)) in points.iter().zip(&normals).enumerate() {
            if n.y > 0.5 {
                let off = if k % 4 == 0 { 0.003 } else { -0.003 };
                wall.push(x + n * off);
                up.push(*n);
            }
        }
        let mut tracks = plane_tracks(&truth, &wall, &up);
        for t in &mut tracks {
            // the anchors on the wall itself
            t.anchor.y = 4.;
        }
        let a = align_to_planes(&tracks, &truth);
        assert!(a.scale_hold < 0.01, "{}", a.scale_hold);
        assert_eq!(a.scale, 1.);
        for (b, p) in a.poses.iter().zip(&truth) {
            assert!((b.p - p.p).norm() < 2e-3, "{}", (b.p - p.p).norm());
        }
    }

    #[test]
    fn along_a_corridor_what_the_planes_cannot_tell_stays_as_it_was() {
        // a wall and the floor only: they hold the turn, but not the place along them, nor the
        // scale about where they meet
        let (truth, points, normals) = scene();
        let tracks = plane_tracks(&truth, &points, &normals);
        let moved = widened(&truth, 1., V3::new(0., -0.01, 0.02));
        let a = align_to_planes(&tracks, &moved);
        assert!(a.scale_hold < 0.01, "{}", a.scale_hold);
        assert_eq!(a.scale, 1.);
        assert!(
            (angle(&a.turn) - 0.5f64.to_radians()).abs() < 1e-4,
            "{}",
            angle(&a.turn).to_degrees()
        );
        // back but for one slide along the corridor, the same for every image
        let slide = a.poses[0].p - truth[0].p;
        for (b, p) in a.poses.iter().zip(&truth) {
            let d = b.p - p.p;
            assert!((d - slide).norm() < 1e-4, "{}", (d - slide).norm());
            assert!(d.y.abs() < 1e-4 && d.z.abs() < 1e-4, "{d:?}");
        }
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
    fn a_track_point_is_its_rays_meeting_point_with_its_spread_and_distance_to_the_cloud() {
        let cam = camera();
        let (poses, _, _) = scene();
        // a point in front of the wall, which the cloud missed: its features met the wall behind
        let x = V3::new(1.2, 2.5, 1.0);
        let wall = V3::new(1.2, 4., 0.9);
        let sightings: Vec<Sighting> = poses
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let l = look(p, &x)?;
                Some(Sighting::new(
                    i,
                    cam.project([l.x, l.y, l.z]),
                    &cam,
                    &[0.; 4],
                ))
            })
            .collect();
        let n = sightings.len();
        assert!(n >= 6, "{n}");
        let t = Track {
            sightings,
            anchor: wall,
            normal: V3::new(0., 1., 0.),
            held: false,
        };
        let p = track_point(&t, &poses).unwrap();
        assert!((p.x - x).norm() < 1e-6, "{:?}", p.x);
        assert_eq!(p.sightings, n);
        assert!(p.residual_max < 1e-3, "{}", p.residual_max);
        assert!((p.to_anchor - (x - wall).norm()).abs() < 1e-6);
        assert!(p.planar);
        // the widest angle is that of the farthest-apart cameras seeing it
        let rays: Vec<V3> = t
            .sightings
            .iter()
            .map(|s| (x - poses[s.image].p).normalize())
            .collect();
        let widest = rays
            .iter()
            .flat_map(|a| rays.iter().map(move |b| a.dot(b).clamp(-1., 1.).acos()))
            .fold(0., f64::max)
            .to_degrees();
        assert!(
            (p.angle_deg - widest).abs() < 1e-3,
            "{} {widest}",
            p.angle_deg
        );
        let nearest = t
            .sightings
            .iter()
            .map(|s| (x - poses[s.image].p).norm())
            .fold(f64::INFINITY, f64::min);
        assert!((p.nearest - nearest).abs() < 1e-6);
        // seen by one camera only from behind: none
        let mut behind = poses.clone();
        let s0 = t.sightings[0].image;
        behind[s0].p = x + (x - poses[s0].p);
        assert!(track_point(&t, &behind).is_none());
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
    #[test]
    fn the_adjustments_inputs_read_back_as_written() {
        let cam = camera();
        let k: Intrinsic = Default::default();
        let poses = vec![
            Pose {
                r: Rotation3::from_euler_angles(0.1, 0.2, 0.3).into_inner(),
                p: V3::new(1., 2., 3.),
            },
            Pose {
                r: M3::identity(),
                p: V3::new(-1., 0.5, 0.25),
            },
        ];
        let tracks = vec![Track {
            sightings: vec![
                Sighting::new(0, [1700., 2300.], &cam, &k),
                Sighting::new(1, [1800., 2400.], &cam, &k),
            ],
            anchor: V3::new(0.5, 4., 1.),
            normal: V3::new(0., 1., 0.),
            held: true,
        }];
        let path = std::env::temp_dir().join(format!("s20_solve_input_{}", std::process::id()));
        save_solve_input(&path, &poses, &[0, 1], 2, &[(0, 1)], &tracks).unwrap();
        let back = SolveInput::load(&path).unwrap();
        // damaged: a count far beyond the file, then a track's image out of range
        let good = std::fs::read(&path).unwrap();
        let mut bad = good.clone();
        bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &bad).unwrap();
        assert!(SolveInput::load(&path).is_err());
        let mut bad = good.clone();
        let at = good.len() - 11 * 8;
        bad[at..at + 8].copy_from_slice(&7u64.to_le_bytes());
        std::fs::write(&path, &bad).unwrap();
        assert!(SolveInput::load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        assert_eq!(back.poses.len(), 2);
        assert_eq!(back.poses[0].r, poses[0].r);
        assert_eq!(back.poses[1].p, poses[1].p);
        assert_eq!(
            (back.frame, back.frames, back.chain),
            (vec![0, 1], 2, vec![(0, 1)])
        );
        let (a, b) = (&back.tracks[0], &tracks[0]);
        assert_eq!((a.anchor, a.normal, a.held), (b.anchor, b.normal, b.held));
        for (x, y) in a.sightings.iter().zip(&b.sightings) {
            assert_eq!(
                (x.image, x.bearing, x.ta, x.tb, x.focal),
                (y.image, y.bearing, y.ta, y.tb, y.focal)
            );
        }
    }
}
