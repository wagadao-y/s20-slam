//! A whole recording: choose image pairs, extract features, anchor observations on the cloud,
//! refine every image's pose, then (with `tracks`) adjust the matches joined into tracks with
//! their points held near the cloud, and write the corrected `ImgPose.txt` with a report.

use crate::anchors::{Cloud, Pose, V3, ZGrid, zgrid};
use crate::bundle;
use crate::camera::Fisheye;
use crate::pairs::{Features, ImagePoses, Obs, View, features, observe_guided, read_image_poses};
use crate::refine::{
    Cameras, Correction, Prepared, Settings, apply, mounted, pose_row, prepare, refine_camera,
    residual_with,
};
use anyhow::{Context, Result};
use nalgebra::Rotation3;
use rayon::prelude::*;
use s20_io::job::Job;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub struct Input {
    /// the recording's data folder holding `images.txt` (images read from its bag)
    pub data: PathBuf,
    /// camera poses with the device correction (and delays) applied
    pub imgpose: PathBuf,
    /// the calibration with the device correction
    pub calib: PathBuf,
    /// the uncoloured cloud and its sampling step
    pub cloud: PathBuf,
    pub step: usize,
    pub out: PathBuf,
    /// share of observations held out for the check
    pub holdout: f64,
    pub max_size: usize,
    pub features: usize,
    /// guided matching: accepted distance from the prediction, search radius (pixels), ratio test
    pub gate: f64,
    pub search: f64,
    pub ratio: f32,
    /// also solve the cameras' mounts, fisheye angles and principal points, written as a
    /// calibration to this file
    pub calib_out: Option<PathBuf>,
    /// how far the cameras may move from the calibration given: mount rotation (degrees) and
    /// offset (mm), principal point (pixels), fisheye angle scale and cubic term
    pub camera_prior: CameraPrior,
    /// also match more pairs (two images apart, the other camera's neighbouring frames, revisits),
    /// join the matches into tracks, adjust them after the refinement (`bundle`) and write those
    /// poses to this file: the images agree with one another (for Gaussian Splatting), while `out`,
    /// each image fitted to the cloud, is for the colouring
    pub tracks: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraPrior {
    pub mount_deg: f64,
    pub mount_mm: f64,
    pub centre_px: f64,
    pub angle: f64,
}

impl CameraPrior {
    /// from a maker's calibration: the cameras may move freely
    pub const MAKER: CameraPrior = CameraPrior {
        mount_deg: 2.,
        mount_mm: 20.,
        centre_px: 20.,
        angle: 0.01,
    };
    /// from a calibration solved on the device's recordings: about their spread between recordings
    pub const DEVICE: CameraPrior = CameraPrior {
        mount_deg: 0.05,
        mount_mm: 2.,
        centre_px: 0.5,
        angle: 0.0002,
    };
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Temporal,
    Simultaneous,
    Cross,
    /// the more pairs matched for the tracks (not in the refinement nor its checks)
    Track,
}

pub fn side(name: &str) -> usize {
    usize::from(!name.starts_with("left"))
}

/// Pairs of image indices: consecutive images of a camera, left and right taken together, and for
/// every second left image the right image 3-60 s away that sees most of the same surfaces from
/// nearly the same side.
fn pairs(p: &ImagePoses, cloud: &Cloud, grids: &[ZGrid]) -> Vec<(usize, usize, Kind)> {
    let mut by_side: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (i, n) in p.names.iter().enumerate() {
        by_side[side(n)].push(i);
    }
    for v in &mut by_side {
        v.sort_by(|&a, &b| p.times[a].total_cmp(&p.times[b]));
    }
    let mut out = Vec::new();
    for v in &by_side {
        for w in v.windows(2) {
            if p.times[w[1]] - p.times[w[0]] < 4.5 {
                out.push((w[0], w[1], Kind::Temporal));
            }
        }
    }
    let rights = &by_side[1];
    for &l in &by_side[0] {
        if let Some(&r) = rights.iter().min_by(|&&a, &&b| {
            (p.times[a] - p.times[l])
                .abs()
                .total_cmp(&(p.times[b] - p.times[l]).abs())
        }) && (p.times[r] - p.times[l]).abs() < 0.01
        {
            out.push((l, r, Kind::Simultaneous));
        }
    }
    let cross: Vec<(usize, usize, Kind)> = by_side[0]
        .par_iter()
        .step_by(2)
        .filter_map(|&l| {
            let seen = grids[l].ids();
            let seen: Vec<V3> = seen
                .iter()
                .step_by((seen.len() / 20000).max(1))
                .map(|&i| V3::from(cloud.xyz[i]))
                .collect();
            let a = p.poses[l].p;
            let best = rights
                .iter()
                .filter(|&&r| (3. ..=60.).contains(&(p.times[r] - p.times[l]).abs()))
                .map(|&r| {
                    let b = &p.poses[r];
                    let score = seen
                        .iter()
                        .filter(|x| {
                            let lb = b.r.transpose() * (*x - b.p);
                            let d = lb.norm();
                            let view = ((*x - a).normalize().dot(&((*x - b.p) / d)))
                                .clamp(-1., 1.)
                                .acos();
                            d > 0.5
                                && d < 12.
                                && lb.xy().norm().atan2(lb.z) < 70f64.to_radians()
                                && view < 35f64.to_radians()
                        })
                        .count();
                    (score, r)
                })
                .max();
            best.filter(|&(s, _)| s >= 500)
                .map(|(_, r)| (l, r, Kind::Cross))
        })
        .collect();
    out.extend(cross);
    out
}

/// More pairs for the tracks, besides `have`: a camera's images two apart, a left image with the
/// right images of the frames before and after it, and for every image the three others more
/// than 9 s away that see most of the same 30 cm cells of the cloud, looking within 60 degrees
/// of the same way (revisits).
fn track_pairs(
    p: &ImagePoses,
    cloud: &Cloud,
    grids: &[ZGrid],
    have: &[(usize, usize, Kind)],
) -> Vec<(usize, usize, Kind)> {
    let mut by_side: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (i, n) in p.names.iter().enumerate() {
        by_side[side(n)].push(i);
    }
    for v in &mut by_side {
        v.sort_by(|&a, &b| p.times[a].total_cmp(&p.times[b]));
    }
    let mut seen: HashSet<(usize, usize)> =
        have.iter().map(|&(a, b, _)| (a.min(b), a.max(b))).collect();
    let mut out = Vec::new();
    let mut add = |a: usize, b: usize, out: &mut Vec<(usize, usize, Kind)>| {
        if a != b && seen.insert((a.min(b), a.max(b))) {
            out.push((a, b, Kind::Track));
        }
    };
    for v in &by_side {
        for w in v.windows(3) {
            if p.times[w[2]] - p.times[w[0]] < 9. {
                add(w[0], w[2], &mut out);
            }
        }
    }
    let rights = &by_side[1];
    for &l in &by_side[0] {
        let k = rights.partition_point(|&r| p.times[r] < p.times[l] - 0.01);
        for r in [k.checked_sub(1), Some(k), Some(k + 1)]
            .into_iter()
            .flatten()
        {
            if let Some(&r) = rights.get(r) {
                let dt = (p.times[r] - p.times[l]).abs();
                if dt > 0.01 && dt < 4.5 {
                    add(l, r, &mut out);
                }
            }
        }
    }
    // revisits: the cloud's 30 cm cells each image sees, and the images seeing each cell
    let key = |x: &[f64; 3]| {
        (
            (x[0] / 0.3).floor() as i64,
            (x[1] / 0.3).floor() as i64,
            (x[2] / 0.3).floor() as i64,
        )
    };
    let cells: Vec<HashSet<(i64, i64, i64)>> = grids
        .par_iter()
        .map(|g| {
            let ids = g.ids();
            ids.iter()
                .step_by((ids.len() / 4000).max(1))
                .map(|&i| key(&cloud.xyz[i]))
                .collect()
        })
        .collect();
    let mut seers: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, c) in cells.iter().enumerate() {
        for k in c {
            seers.entry(*k).or_default().push(i);
        }
    }
    let revisits: Vec<(usize, usize)> = (0..p.names.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut count: HashMap<usize, usize> = HashMap::new();
            for k in &cells[i] {
                for &j in &seers[k] {
                    if j != i {
                        *count.entry(j).or_default() += 1;
                    }
                }
            }
            let axis = p.poses[i].r.column(2).into_owned();
            let mut best: Vec<(usize, usize)> = count
                .into_iter()
                .filter(|&(j, n)| {
                    (p.times[j] - p.times[i]).abs() > 9.
                        && n >= 50
                        && n as f64 >= 0.3 * cells[i].len().min(cells[j].len()) as f64
                        && axis.dot(&p.poses[j].r.column(2)) > 60f64.to_radians().cos()
                })
                .map(|(j, n)| (n, j))
                .collect();
            best.sort_unstable_by(|a, b| b.cmp(a));
            best.into_iter().take(3).map(move |(_, j)| (i, j))
        })
        .collect();
    for (i, j) in revisits {
        add(i, j, &mut out);
    }
    out
}

/// Each image's frame (a left image and the right one taken with it; an image alone otherwise),
/// the number of frames, and consecutive frames less than 4.5 s apart.
fn frames(p: &ImagePoses) -> (Vec<usize>, usize, Vec<(usize, usize)>) {
    let mut order: Vec<usize> = (0..p.names.len()).collect();
    order.sort_by(|&a, &b| p.times[a].total_cmp(&p.times[b]));
    let mut frame = vec![usize::MAX; p.names.len()];
    let mut start: Vec<f64> = Vec::new();
    for (k, &i) in order.iter().enumerate() {
        if frame[i] != usize::MAX {
            continue;
        }
        frame[i] = start.len();
        // the other camera's image taken with it
        if let Some(&j) = order[k + 1..]
            .iter()
            .take_while(|&&j| p.times[j] - p.times[i] < 0.01)
            .find(|&&j| side(&p.names[j]) != side(&p.names[i]) && frame[j] == usize::MAX)
        {
            frame[j] = start.len();
        }
        start.push(p.times[i]);
    }
    let chain = (1..start.len())
        .filter(|&f| start[f] - start[f - 1] < 4.5)
        .map(|f| (f - 1, f))
        .collect();
    (frame, start.len(), chain)
}

/// `ImgPose.txt` of `poses`: the rows read (`ImagePoses::rows`) with the poses replaced.
fn write_poses(path: &Path, rows: &[Vec<String>], poses: &[Pose]) -> Result<()> {
    let mut text = String::from("index x y z roll pitch yaw qx qy qz qw timestamp\n");
    for (row, p) in rows.iter().zip(poses) {
        let (e, q) = pose_row(&p.r);
        writeln!(
            text,
            "{} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {}",
            row[0], p.p.x, p.p.y, p.p.z, e[0], e[1], e[2], q[0], q[1], q[2], q[3], row[11]
        )?;
    }
    std::fs::write(path, text).with_context(|| path.display().to_string())
}

/// An observation of the refinement (and of its checks): of its pairs, on a plane of the cloud.
fn for_refinement(o: &Obs, k: Kind) -> bool {
    k != Kind::Track && o.planar()
}

pub fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A recording's anchored observations, ready to refine.
pub struct Collected {
    pub poses: ImagePoses,
    pub cams: [Fisheye; 2],
    pub obs: Vec<Obs>,
    pub kinds: Vec<Kind>,
    pub prepared: Vec<Prepared>,
    /// observations held out for the check
    pub held: Vec<bool>,
    /// consecutive images of a camera
    pub chain: Vec<(usize, usize)>,
    /// images in some pair
    pub used: Vec<usize>,
    pub pairs: usize,
}

pub fn collect(input: &Input, job: &Job) -> Result<Collected> {
    let poses = read_image_poses(&input.imgpose)?;
    let cal = s20_io::calib::load(&input.calib)?;
    let cams = [Fisheye::from(&cal.left), Fisheye::from(&cal.right)];
    let focal = (cal.left.a11 + cal.right.a11) / 2.;
    job.log("reading the cloud");
    let cloud = Cloud::load(&input.cloud, input.step)?;
    job.log(&format!(
        "cloud: {} points (every {})",
        cloud.xyz.len(),
        input.step
    ));
    let grids: Vec<ZGrid> = (0..poses.names.len())
        .into_par_iter()
        .map(|i| zgrid(&cloud, &poses.poses[i], &cams[side(&poses.names[i])]))
        .collect();
    job.log(&format!("z-buffers of {} images", grids.len()));
    let mut pairs = pairs(&poses, &cloud, &grids);
    if input.tracks.is_some() {
        let more = track_pairs(&poses, &cloud, &grids, &pairs);
        job.log(&format!("{} more pairs for the tracks", more.len()));
        pairs.extend(more);
    }
    let mut used: Vec<usize> = pairs.iter().flat_map(|&(a, b, _)| [a, b]).collect();
    used.sort_unstable();
    used.dedup();
    job.log(&format!("{} pairs over {} images", pairs.len(), used.len()));
    let done = std::sync::atomic::AtomicU64::new(0);
    // one bar: the features about 60% of the time, the observations 35%
    let (features_part, observations_part) = (job.part(0.0, 0.6), job.part(0.6, 0.95));
    let feats: Vec<Option<Features>> = (0..poses.names.len())
        .into_par_iter()
        .map(|i| -> Result<Option<Features>> {
            if used.binary_search(&i).is_err() {
                return Ok(None);
            }
            job.check()?;
            let name = &poses.names[i];
            let jpeg = s20_io::dataset::read_image(&input.data.join("images").join(name))
                .with_context(|| name.clone())?;
            let f = features(&jpeg, &cams[side(name)], input.max_size, input.features)?;
            features_part.report(
                "features",
                done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
                used.len() as u64,
            );
            Ok(Some(f))
        })
        .collect::<Result<_>>()?;
    job.log(&format!("features of {} images", used.len()));
    job.check()?;
    let done = std::sync::atomic::AtomicU64::new(0);
    let observed: Vec<(Obs, Kind)> = pairs
        .par_iter()
        .flat_map_iter(|&(a, b, kind)| {
            let va = View {
                index: a,
                pose: poses.poses[a],
                cam: &cams[side(&poses.names[a])],
                grid: &grids[a],
                features: feats[a].as_ref().unwrap(),
            };
            let vb = View {
                index: b,
                pose: poses.poses[b],
                cam: &cams[side(&poses.names[b])],
                grid: &grids[b],
                features: feats[b].as_ref().unwrap(),
            };
            let o = observe_guided(
                &va,
                &vb,
                &cloud,
                focal,
                input.gate,
                input.search,
                input.ratio,
                input.tracks.is_some(),
            );
            observations_part.report(
                "observations",
                done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
                pairs.len() as u64,
            );
            o.into_iter().map(move |o| (o, kind))
        })
        .collect();
    drop(feats);
    job.check()?;
    // the refinement's observations first, in their order without the tracks, so that it and its
    // hold-out are as they were
    let (mine, rest): (Vec<_>, Vec<_>) = observed
        .into_iter()
        .partition(|(o, k)| for_refinement(o, *k));
    let (obs, kinds): (Vec<Obs>, Vec<Kind>) = mine.into_iter().chain(rest).unzip();
    job.log(&format!("{} observations", obs.len()));
    let prepared: Vec<Prepared> = obs
        .par_iter()
        .map(|o| prepare(o, &cams, side(&poses.names[o.a]), side(&poses.names[o.b])))
        .collect();
    // deterministic hold-out
    let held: Vec<bool> = (0..obs.len())
        .map(|k| {
            (((k as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64 << 24) as f64)
                < input.holdout
        })
        .collect();
    let mut chain = Vec::new();
    for s in 0..2 {
        let mut v: Vec<usize> = used
            .iter()
            .copied()
            .filter(|&i| side(&poses.names[i]) == s)
            .collect();
        v.sort_by(|&a, &b| poses.times[a].total_cmp(&poses.times[b]));
        chain.extend(
            v.windows(2)
                .filter(|w| poses.times[w[1]] - poses.times[w[0]] < 4.5)
                .map(|w| (w[0], w[1])),
        );
    }
    Ok(Collected {
        poses,
        cams,
        obs,
        kinds,
        prepared,
        held,
        chain,
        used,
        pairs: pairs.len(),
    })
}

pub fn run(input: &Input, job: &Job) -> Result<serde_json::Value> {
    let Collected {
        poses,
        cams,
        obs,
        kinds,
        prepared,
        held,
        chain,
        used,
        pairs,
    } = collect(input, job)?;
    // the refinement keeps to its pairs and to the features on planes; the rest is for the tracks
    let train: Vec<bool> = held
        .iter()
        .zip(obs.iter().zip(&kinds))
        .map(|(h, (o, k))| !h && for_refinement(o, *k))
        .collect();
    let fit = input.calib_out.is_some();
    job.log(if fit {
        "refining the image poses and the cameras"
    } else {
        "refining the image poses"
    });
    let p = &input.camera_prior;
    let settings = Settings {
        fit_camera: fit,
        fit_mount: fit,
        iterations: if fit { 25 } else { 12 },
        mount_rot_sigma: p.mount_deg.to_radians(),
        mount_pos_sigma: p.mount_mm / 1000.,
        centre_sigma: p.centre_px,
        angle_sigma: p.angle,
        cubic_sigma: p.angle,
        ..Default::default()
    };
    let mut camera = Cameras::default();
    let c = refine_camera(
        &poses.poses,
        &obs,
        &prepared,
        &train,
        &chain,
        &settings,
        &mut camera,
    );
    // the refined poses with the solved mounts (for the colouring), and the tracks' adjustment
    let refined: Vec<Pose> = (0..poses.poses.len())
        .map(|i| {
            mounted(
                &apply(&poses.poses[i], &c[i]),
                &camera.mount[side(&poses.names[i])],
            )
        })
        .collect();
    let mut tracked: Option<Vec<Pose>> = None;
    let mut tracks_report = serde_json::Value::Null;
    if let Some(tracks_out) = &input.tracks {
        job.log("joining the matches into tracks");
        let sides: Vec<usize> = poses.names.iter().map(|n| side(n)).collect();
        let tracks = bundle::tracks(&obs, &cams, &camera.intrinsic, &sides, input.holdout);
        let sightings: usize = tracks.iter().map(|t| t.sightings.len()).sum();
        job.log(&format!(
            "{} tracks, {} sightings; adjusting them",
            tracks.len(),
            sightings
        ));
        let (frame, frames, frame_chain) = frames(&poses);
        // the images of a frame start together: their refinements averaged, as the frame's
        // correction moves them together
        let mut sum = vec![(Correction::zeros(), 0usize); frames];
        for (i, &f) in frame.iter().enumerate() {
            sum[f].0 += c[i];
            sum[f].1 += 1;
        }
        let start: Vec<Pose> = (0..poses.poses.len())
            .map(|i| {
                let (s, n) = &sum[frame[i]];
                mounted(
                    &apply(&poses.poses[i], &(s / *n as f64)),
                    &camera.mount[side(&poses.names[i])],
                )
            })
            .collect();
        let s = bundle::Settings::default();
        let started = std::time::Instant::now();
        let sol = bundle::solve(
            &start,
            &frame,
            frames,
            &frame_chain,
            &tracks,
            &s,
            &|m: &str| job.log(m),
        );
        let solved: Vec<Pose> = (0..start.len())
            .map(|i| sol.pose(frame[i], &start[i]))
            .collect();
        write_poses(tracks_out, &poses.rows, &solved)?;
        let (b0, p0) = bundle::check(&tracks, &refined);
        let (b1, p1) = bundle::check(&tracks, &solved);
        let share = |v: &[f64], over: f64| {
            v.iter().filter(|&&x| x > over).count() as f64 / v.len().max(1) as f64
        };
        job.log(&format!(
            "held-out tracks: median {:.2} -> {:.2} px-eq, over 5 px {:.1}% -> {:.1}%, to the plane {:.1} -> {:.1} mm",
            median(b0.clone()),
            median(b1.clone()),
            100. * share(&b0, 5.),
            100. * share(&b1, 5.),
            1000. * median(p0.clone()),
            1000. * median(p1.clone())
        ));
        let turn: Vec<f64> = sol
            .rot
            .iter()
            .map(|r| Rotation3::from_matrix_unchecked(*r).angle().to_degrees())
            .collect();
        let moved: Vec<f64> = sol.shift.iter().map(|t| t.norm() * 1000.).collect();
        tracked = Some(solved);
        tracks_report = serde_json::json!({
            "file": tracks_out,
            "tracks": tracks.len(), "sightings": sightings,
            "held_out": tracks.iter().filter(|t| t.held).count(),
            "frames": frames, "iterations": sol.iterations, "cost": sol.cost,
            "seconds": started.elapsed().as_secs_f64(),
            "held_out_median_px_eq": {"before": median(b0.clone()), "after": median(b1.clone())},
            "held_out_over_5px": {"before": share(&b0, 5.), "after": share(&b1, 5.)},
            "held_out_plane_mm": {"before": 1000. * median(p0), "after": 1000. * median(p1)},
            "frame_turn_deg": {"median": median(turn.clone()), "max": turn.iter().cloned().fold(0., f64::max)},
            "frame_shift_mm": {"median": median(moved.clone()), "max": moved.iter().cloned().fold(0., f64::max)},
            "settings": {
                "pixel": s.pixel, "plane_sigma_m": s.plane_sigma, "plane_scale": s.plane_scale,
                "rot_sigma_deg": s.rot_sigma.to_degrees(), "pos_sigma_m": s.pos_sigma,
                "rot_smooth_deg": s.rot_smooth.to_degrees(), "pos_smooth_m": s.pos_smooth,
            },
        });
    }
    // the final poses hold the mounts: only the cameras' intrinsic change remains to apply
    let none = Cameras::default();
    let lens = Cameras {
        intrinsic: camera.intrinsic,
        ..Cameras::default()
    };
    let check = |at: &[Pose], camera: &Cameras, kind: Kind| -> f64 {
        median(
            obs.iter()
                .zip(&prepared)
                .zip(&kinds)
                .zip(&held)
                .filter(|(((o, _), k), h)| **k == kind && **h && o.planar())
                .map(|(((o, g), _), _)| {
                    let r = residual_with(o, g, &at[o.a], &at[o.b], camera);
                    r[0].hypot(r[1])
                })
                .collect(),
        )
    };
    let mut checks = serde_json::Map::new();
    for (kind, label) in [
        (Kind::Temporal, "temporal"),
        (Kind::Simultaneous, "simultaneous"),
        (Kind::Cross, "cross"),
    ] {
        let (b, a) = (
            check(&poses.poses, &none, kind),
            check(&refined, &lens, kind),
        );
        let mut entry = serde_json::json!({"before": b, "after": a});
        if let Some(at) = &tracked {
            let r = check(at, &lens, kind);
            entry["tracks"] = r.into();
            job.log(&format!(
                "held-out {label}: median {b:.2} -> {a:.2} px-eq ({r:.2} with the tracks' poses)"
            ));
        } else {
            job.log(&format!("held-out {label}: median {b:.2} -> {a:.2} px-eq"));
        }
        checks.insert(label.into(), entry);
    }
    write_poses(&input.out, &poses.rows, &refined)?;
    let mut solved = serde_json::Value::Null;
    if let Some(path) = &input.calib_out {
        let source = std::fs::read_to_string(&input.calib)
            .with_context(|| input.calib.display().to_string())?;
        let (written, error) = crate::calibrate::write(&source, &cams, &camera)?;
        std::fs::write(path, written).with_context(|| path.display().to_string())?;
        solved = serde_json::json!({"camera": crate::joint::camera_json(&camera), "polynomial_refit_px": error, "calibration": path});
        job.log(&format!("wrote {}", path.display()));
    }
    // each image's change, from the pose given to the refined one
    let rot: Vec<f64> = used
        .iter()
        .map(|&i| {
            Rotation3::from_matrix_unchecked(refined[i].r * poses.poses[i].r.transpose())
                .angle()
                .to_degrees()
        })
        .collect();
    let pos: Vec<f64> = used
        .iter()
        .map(|&i| (refined[i].p - poses.poses[i].p).norm() * 1000.)
        .collect();
    // per image: observations it takes part in, its correction, and the held-out residuals touching it
    let mut count = vec![0usize; poses.poses.len()];
    let mut before = vec![Vec::new(); poses.poses.len()];
    let mut after = vec![Vec::new(); poses.poses.len()];
    for (((o, g), h), k) in obs.iter().zip(&prepared).zip(&held).zip(&kinds) {
        if !h {
            count[o.a] += 1;
            count[o.b] += 1;
            continue;
        }
        if !for_refinement(o, *k) {
            continue;
        }
        let r0 = residual_with(o, g, &poses.poses[o.a], &poses.poses[o.b], &none);
        let r1 = residual_with(o, g, &refined[o.a], &refined[o.b], &lens);
        for i in [o.a, o.b] {
            before[i].push(r0[0].hypot(r0[1]));
            after[i].push(r1[0].hypot(r1[1]));
        }
    }
    let per_image: Vec<serde_json::Value> = used
        .iter()
        .zip(rot.iter().zip(&pos))
        .map(|(&i, (r, p))| {
            serde_json::json!({
                "name": poses.names[i], "observations": count[i], "rotation_deg": r, "offset_mm": p,
                "held_out": before[i].len(), "held_out_before": median(before[i].clone()), "held_out_after": median(after[i].clone()),
            })
        })
        .collect();
    let report = serde_json::json!({
        "images": used.len(), "pairs": pairs, "observations": obs.len(), "held_out_share": input.holdout,
        "held_out_median_px_eq": checks,
        "rotation_deg_median": median(rot), "offset_mm_median": median(pos),
        "cameras_solved": solved,
        "tracks": tracks_report,
        "per_image": per_image,
    });
    std::fs::write(
        Path::new(&format!("{}.json", input.out.display())),
        serde_json::to_vec_pretty(&report)?,
    )?;
    job.log(&format!("wrote {}", input.out.display()));
    Ok(report)
}
